"""Alerts are manual on app execution. Scheduled checks require an external scheduler."""
from __future__ import annotations
import os
import requests

def dispatch_telegram(text:str)->bool:
    token=os.getenv("TELEGRAM_BOT_TOKEN","")
    chat_id=os.getenv("TELEGRAM_CHAT_ID","")
    if not token or not chat_id:
        return False
    res=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id":chat_id,"text":text},timeout=15)
    res.raise_for_status()
    return True
