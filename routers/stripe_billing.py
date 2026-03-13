import logging
from typing import Optional

import stripe
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

from config import settings
from database import get_db
from utils.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["billing"])

stripe.api_key = settings.stripe_secret_key

# ── Pydantic models ──

class CheckoutRequest(BaseModel):
    promo_code: Optional[str] = None


class UsageResponse(BaseModel):
    subscription_status: str
    dossier_count: int
    dossier_limit: int
    stripe_customer_id: Optional[str] = None


# ── Helpers ──

def _get_or_create_price_id() -> str:
    """Return the Stripe Price ID, creating the product + price if needed."""
    if settings.stripe_price_id:
        return settings.stripe_price_id

    # Check if product already exists
    products = stripe.Product.list(limit=1, active=True)
    for p in products.auto_paging_iter():
        if p.metadata.get("app") == "dossier":
            prices = stripe.Price.list(product=p.id, active=True, limit=1)
            for price in prices.auto_paging_iter():
                settings.stripe_price_id = price.id
                return price.id

    # Create product + price
    product = stripe.Product.create(
        name="Dossier Pro",
        description="50 dossiers per month for $19/month",
        metadata={"app": "dossier"},
    )
    price = stripe.Price.create(
        product=product.id,
        unit_amount=1900,  # $19.00
        currency="usd",
        recurring={"interval": "month"},
    )
    settings.stripe_price_id = price.id
    logger.info("Created Stripe product %s with price %s", product.id, price.id)
    return price.id


def _get_or_create_customer(user_email: str, user_name: str, google_id: str) -> str:
    """Return existing Stripe customer ID or create one."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT stripe_customer_id FROM users WHERE google_id = ?",
            (google_id,),
        ).fetchone()
        if row and row["stripe_customer_id"]:
            return row["stripe_customer_id"]

    # Create in Stripe
    customer = stripe.Customer.create(
        email=user_email,
        name=user_name,
        metadata={"google_id": google_id},
    )

    with get_db() as conn:
        conn.execute(
            "UPDATE users SET stripe_customer_id = ? WHERE google_id = ?",
            (customer.id, google_id),
        )
        conn.commit()

    return customer.id


def get_user_usage(google_id: str) -> dict:
    """Return subscription status and usage for a user."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT subscription_status, dossier_count, stripe_customer_id FROM users WHERE google_id = ?",
            (google_id,),
        ).fetchone()

    if not row:
        return {
            "subscription_status": "free",
            "dossier_count": 0,
            "dossier_limit": settings.free_tier_dossier_limit,
            "stripe_customer_id": None,
        }

    status = row["subscription_status"] or "free"
    limit = (
        settings.paid_tier_dossier_limit
        if status == "active"
        else settings.free_tier_dossier_limit
    )
    return {
        "subscription_status": status,
        "dossier_count": row["dossier_count"] or 0,
        "dossier_limit": limit,
        "stripe_customer_id": row["stripe_customer_id"],
    }


def increment_dossier_count(google_id: str):
    """Increment the dossier count for a user. Called after successful research."""
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET dossier_count = COALESCE(dossier_count, 0) + 1 WHERE google_id = ?",
            (google_id,),
        )
        conn.commit()


def can_generate_dossier(google_id: str) -> tuple:
    """Check if user can generate a dossier. Returns (allowed, usage_info)."""
    usage = get_user_usage(google_id)
    if usage["dossier_count"] >= usage["dossier_limit"]:
        return False, usage
    return True, usage


# ── Endpoints ──

@router.get("/billing/usage", response_model=UsageResponse)
async def get_usage(user: dict = Depends(get_current_user)):
    """Return current usage and subscription status."""
    usage = get_user_usage(user["sub"])
    return usage


@router.post("/billing/checkout")
async def create_checkout_session(
    body: CheckoutRequest,
    user: dict = Depends(get_current_user),
):
    """Create a Stripe Checkout session for the $19/month plan."""
    price_id = _get_or_create_price_id()
    customer_id = _get_or_create_customer(
        user["email"], user.get("name", ""), user["sub"]
    )

    params = {
        "customer": customer_id,
        "payment_method_types": ["card"],
        "line_items": [{"price": price_id, "quantity": 1}],
        "mode": "subscription",
        "success_url": "https://api.builddossier.com/billing/success?session_id={CHECKOUT_SESSION_ID}",
        "cancel_url": "https://api.builddossier.com/billing/cancel",
        "allow_promotion_codes": True,
    }

    if body.promo_code:
        # Try to find the promo code in Stripe
        try:
            promo_codes = stripe.PromotionCode.list(code=body.promo_code, active=True, limit=1)
            if promo_codes.data:
                params["discounts"] = [{"promotion_code": promo_codes.data[0].id}]
                # Remove allow_promotion_codes when using discounts param
                params.pop("allow_promotion_codes", None)
        except stripe.error.StripeError:
            pass  # Ignore invalid promo codes, let checkout handle it

    session = stripe.checkout.Session.create(**params)
    return {"checkout_url": session.url, "session_id": session.id}


@router.post("/billing/portal")
async def create_portal_session(user: dict = Depends(get_current_user)):
    """Create a Stripe Customer Portal session for subscription management."""
    google_id = user["sub"]

    with get_db() as conn:
        row = conn.execute(
            "SELECT stripe_customer_id FROM users WHERE google_id = ?",
            (google_id,),
        ).fetchone()

    if not row or not row["stripe_customer_id"]:
        raise HTTPException(status_code=400, detail="No billing account found")

    session = stripe.billing_portal.Session.create(
        customer=row["stripe_customer_id"],
        return_url="https://api.builddossier.com/billing/portal-return",
    )
    return {"portal_url": session.url}


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events to keep subscription status in sync."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    data = event["data"]["object"]

    logger.info("Stripe webhook: %s", event_type)

    if event_type == "checkout.session.completed":
        _handle_checkout_completed(data)
    elif event_type == "customer.subscription.updated":
        _handle_subscription_updated(data)
    elif event_type == "customer.subscription.deleted":
        _handle_subscription_deleted(data)
    elif event_type == "invoice.payment_failed":
        _handle_payment_failed(data)

    return {"status": "ok"}


def _handle_checkout_completed(session):
    """Activate subscription after successful checkout."""
    customer_id = session.get("customer")
    subscription_id = session.get("subscription")
    if not customer_id:
        return

    with get_db() as conn:
        conn.execute(
            """UPDATE users SET
                stripe_subscription_id = ?,
                subscription_status = 'active',
                dossier_count = 0,
                billing_cycle_start = datetime('now')
            WHERE stripe_customer_id = ?""",
            (subscription_id, customer_id),
        )
        conn.commit()

    logger.info("Subscription activated for customer %s", customer_id)


def _handle_subscription_updated(subscription):
    """Update subscription status (handles renewals, status changes)."""
    customer_id = subscription.get("customer")
    status = subscription.get("status")  # active, past_due, canceled, etc.
    sub_id = subscription.get("id")

    if not customer_id:
        return

    # Map Stripe statuses to our simplified model
    if status == "active":
        # Check if this is a renewal (current_period_start changed)
        with get_db() as conn:
            conn.execute(
                """UPDATE users SET
                    stripe_subscription_id = ?,
                    subscription_status = 'active',
                    dossier_count = 0,
                    billing_cycle_start = datetime('now')
                WHERE stripe_customer_id = ?""",
                (sub_id, customer_id),
            )
            conn.commit()
        logger.info("Subscription renewed/updated for customer %s", customer_id)
    elif status in ("past_due", "unpaid"):
        with get_db() as conn:
            conn.execute(
                "UPDATE users SET subscription_status = ? WHERE stripe_customer_id = ?",
                (status, customer_id),
            )
            conn.commit()
        logger.info("Subscription %s for customer %s", status, customer_id)


def _handle_subscription_deleted(subscription):
    """Downgrade to free tier when subscription is cancelled."""
    customer_id = subscription.get("customer")
    if not customer_id:
        return

    with get_db() as conn:
        conn.execute(
            """UPDATE users SET
                subscription_status = 'free',
                stripe_subscription_id = NULL,
                billing_cycle_start = NULL
            WHERE stripe_customer_id = ?""",
            (customer_id,),
        )
        conn.commit()

    logger.info("Subscription deleted, downgraded customer %s to free", customer_id)


def _handle_payment_failed(invoice):
    """Handle failed payment — after retries exhausted, downgrade to free."""
    customer_id = invoice.get("customer")
    if not customer_id:
        return

    # Check attempt count — Stripe retries up to 3 times by default
    attempt = invoice.get("attempt_count", 1)
    next_attempt = invoice.get("next_payment_attempt")

    if next_attempt is None:
        # No more retries — downgrade
        with get_db() as conn:
            conn.execute(
                """UPDATE users SET
                    subscription_status = 'free',
                    stripe_subscription_id = NULL,
                    billing_cycle_start = NULL
                WHERE stripe_customer_id = ?""",
                (customer_id,),
            )
            conn.commit()
        logger.info(
            "Payment failed (attempt %d, no retry), downgraded customer %s",
            attempt, customer_id,
        )
    else:
        logger.info(
            "Payment failed (attempt %d, will retry) for customer %s",
            attempt, customer_id,
        )
