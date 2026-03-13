from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from routers.research import router as research_router
from routers.auth import router as auth_router
from routers.stripe_billing import router as billing_router
from database import init_db
from config import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

app = FastAPI(
    title="Deep Research Engine",
    description="Sales prep research engine powered by YouTube + Gemini",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_origin_regex=r"^chrome-extension://.*$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

init_db()
app.include_router(research_router)
app.include_router(auth_router)
app.include_router(billing_router)


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/billing/success", include_in_schema=False)
async def billing_success():
    from fastapi.responses import HTMLResponse
    return HTMLResponse("""
    <html><head><title>Dossier - Payment Successful</title>
    <style>body{font-family:system-ui;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#f9fafb}
    .card{text-align:center;padding:40px;background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
    h1{color:#059669;font-size:24px;margin:0 0 8px}p{color:#6b7280;margin:0}</style></head>
    <body><div class="card"><h1>Payment Successful!</h1><p>You can close this tab and return to the Dossier extension.</p></div></body></html>
    """)


@app.get("/billing/cancel", include_in_schema=False)
async def billing_cancel():
    from fastapi.responses import HTMLResponse
    return HTMLResponse("""
    <html><head><title>Dossier - Payment Cancelled</title>
    <style>body{font-family:system-ui;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#f9fafb}
    .card{text-align:center;padding:40px;background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
    h1{color:#6b7280;font-size:24px;margin:0 0 8px}p{color:#6b7280;margin:0}</style></head>
    <body><div class="card"><h1>Payment Cancelled</h1><p>No charges were made. You can close this tab.</p></div></body></html>
    """)


@app.get("/billing/portal-return", include_in_schema=False)
async def billing_portal_return():
    from fastapi.responses import HTMLResponse
    return HTMLResponse("""
    <html><head><title>Dossier - Subscription Updated</title>
    <style>body{font-family:system-ui;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#f9fafb}
    .card{text-align:center;padding:40px;background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
    h1{color:#6a57d5;font-size:24px;margin:0 0 8px}p{color:#6b7280;margin:0}</style></head>
    <body><div class="card"><h1>Subscription Updated</h1><p>You can close this tab and return to the Dossier extension.</p></div></body></html>
    """)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
