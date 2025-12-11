# Full_backend.py
# NeuroSentinel backend (patched)
# - /predict            -> run model
# - Gmail OAuth: /auth-gmail, /oauth2callback, /connect-gmail
# - Outlook OAuth: /auth-outlook, /oauth2callback-outlook, /connect-outlook
#
# Required files & env:
# - Put Google OAuth client JSON as "credentials.json" in this folder.
#   Set its authorized redirect URI to: http://127.0.0.1:8000/oauth2callback
# - Create an Azure app registration for Outlook (Microsoft Graph):
#   - Note CLIENT_ID and CLIENT_SECRET and set REDIRECT_URI to:
#     http://127.0.0.1:8000/oauth2callback-outlook
# - Set environment variables (or edit below):
#   - MS_CLIENT_ID, MS_CLIENT_SECRET, MS_TENANT (optional - default "common")
#
# Install required libs:
# pip install fastapi uvicorn google-auth-oauthlib google-api-python-client google-auth requests transformers torch lightgbm beautifulsoup4 lxml tldextract pandas

# import os
# os.environ["ACCELERATE_DISABLE_TORCH_COMPILE"] = "1"

# from fastapi import FastAPI, Request
# from fastapi.responses import RedirectResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles 
# from pydantic import BaseModel
# import json
# import time
# import base64
# import urllib.parse
# import requests

# # ML imports
# import torch
# import pandas as pd
# import tldextract
# import re
# import html
# from bs4 import BeautifulSoup
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import lightgbm as lgb

# # Google / Gmail
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build

# # ------------------------------
# # CONFIG (edit these as needed)
# # ------------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MAX_LENGTH = 256

# TRANSFORMER_DIR = "saved_models/phish_transformer"   # ensure this exists
# LGB_MODEL_PATH = "saved_models/url_sender_model.txt" # ensure this exists

# # Google credentials file
# GOOGLE_CREDENTIALS_FILE = "credentials.json"  # output from Google Cloud console

# # Gmail redirect (must match Google Console)
# GOOGLE_REDIRECT_URI = "http://127.0.0.1:8000/oauth2callback"
# GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# # Microsoft (Outlook / Graph) settings - set these environment variables or edit below
# MS_CLIENT_ID = os.environ.get("MS_CLIENT_ID", "<YOUR_MS_CLIENT_ID>")
# MS_CLIENT_SECRET = os.environ.get("MS_CLIENT_SECRET", "<YOUR_MS_CLIENT_SECRET>")
# MS_TENANT = os.environ.get("MS_TENANT", "common")  # or your tenant id
# MS_REDIRECT_URI = os.environ.get("MS_REDIRECT_URI", "http://127.0.0.1:8000/oauth2callback-outlook")
# MS_SCOPES = "offline_access%20Mail.Read"  # URL-encoded scopes for the auth URL

# # token storage folder
# TOKEN_FOLDER = "tokens"
# os.makedirs(TOKEN_FOLDER, exist_ok=True)

# # ------------------------------
# # Load models (transformer + lightgbm)
# # ------------------------------
# # If the model folders don't exist, these calls will raise errors — ensure models saved locally.
# try:
#     tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
#     transformer = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR).to(DEVICE)
#     transformer.eval()
# except Exception as e:
#     print("WARNING: Could not load transformer from", TRANSFORMER_DIR, "->", e)
#     tokenizer = None
#     transformer = None

# try:
#     lgb_model = lgb.Booster(model_file=LGB_MODEL_PATH)
# except Exception as e:
#     print("WARNING: Could not load LightGBM from", LGB_MODEL_PATH, "->", e)
#     lgb_model = None

# softmax = torch.nn.Softmax(dim=-1)

# # ------------------------------
# # Utilities: text / URL cleaners
# # ------------------------------
# def clean_html(text):
#     if not isinstance(text, str):
#         return ""
#     text = html.unescape(text)
#     soup = BeautifulSoup(text, "lxml")
#     for tag in soup(["script", "style"]):
#         tag.decompose()
#     text = soup.get_text(" ")
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = re.sub(r"http\S+|www\.\S+", " ", text)
#     text = re.sub(r"[^A-Za-z0-9\s.,!?@-]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip().lower()
#     return text

# def normalize_url(url):
#     if not isinstance(url, str): return ""
#     try:
#         parsed = urllib.parse.urlparse(url.strip())
#         return urllib.parse.urlunparse(parsed._replace(query="")).lower()
#     except:
#         return ""

# def preprocess_email(subject, body, sender, url):
#     cleaned_html = clean_html(body)
#     cleaned_body = clean_text(cleaned_html)
#     cleaned_subject = clean_text(subject)
#     cleaned_url = normalize_url(url)
#     return f"{cleaned_subject} . {cleaned_body}", cleaned_url

# def extract_url_features(url_list):
#     rows = []
#     for u in url_list:
#         u = str(u)
#         te = tldextract.extract(u)
#         rows.append({
#             "url_len": len(u),
#             "num_dots": u.count("."),
#             "has_ip": int(te.domain.replace(".", "").isdigit()),
#             "sus_kw": int(any(k in u for k in ["login", "secure", "verify", "update"]))
#         })
#     return pd.DataFrame(rows)

# def extract_sender_features(sender_list):
#     rows = []
#     for s in sender_list:
#         s = str(s)
#         domain = s.split("@")[-1] if "@" in s else ""
#         rows.append({
#             "sender_len": len(s),
#             "has_digits": int(any(ch.isdigit() for ch in s)),
#             "domain_len": len(domain),
#             "at_count": s.count("@")
#         })
#     return pd.DataFrame(rows)

# # Robust body extractor for Gmail payload (recursive)
# def _get_message_body(payload):
#     if not payload:
#         return ""
#     # direct body
#     try:
#         if payload.get("body", {}).get("data"):
#             return base64.urlsafe_b64decode(payload["body"]["data"]).decode(errors="ignore")
#     except Exception:
#         pass
#     # parts
#     for part in payload.get("parts", []) or []:
#         b = _get_message_body(part)
#         if b:
#             return b
#     return ""

# # ------------------------------
# # Prediction logic
# # ------------------------------
# def predict_email(subject, body, sender, url):
#     # handle missing models gracefully
#     if tokenizer is None or transformer is None or lgb_model is None:
#         return {"error": "Model(s) not loaded on server."}

#     text, clean_url = preprocess_email(subject or "", body or "", sender or "", url or "")

#     enc = tokenizer(text, padding="max_length", truncation=True,
#                     max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         logits = transformer(**enc).logits
#         probs = softmax(logits)[0].cpu().numpy()
#     t_score = float(probs[1])

#     url_f = extract_url_features([clean_url])
#     sender_f = extract_sender_features([sender or ""])
#     meta = pd.concat([url_f, sender_f], axis=1)

#     try:
#         m_score = float(lgb_model.predict(meta.values)[0])
#     except Exception:
#         m_score = 0.0

#     final = 0.6 * t_score + 0.4 * m_score

#     return {
#         "phishing_probability": final,
#         "transformer_score": t_score,
#         "metadata_score": m_score,
#         "verdict": "phishing" if final >= 0.5 else "ham",
#         "explanation": ("High phishing signals detected" if final >= 0.5 else "Looks safe")
#     }

# # ------------------------------
# # FastAPI app
# # ------------------------------
# app = FastAPI(title="NeuroSentinel Backend (Gmail + Outlook)")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # for dev only — lock this in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class EmailRequest(BaseModel):
#     subject: str = ""
#     body: str = ""
#     sender: str = ""
#     url: str = ""

# class GmailRequest(BaseModel):
#     gmail: str

# class OutlookRequest(BaseModel):
#     outlook: str

# @app.get("/health")
# def health():
#     return {"status": "running"}

# @app.post("/predict")
# def api_predict(req: EmailRequest):
#     try:
#         return predict_email(req.subject, req.body, req.sender, req.url)
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)
    
# #app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

# # ------------------------------
# # Gmail: auth -> callback -> connect
# # ------------------------------
# @app.post("/auth-gmail")
# def auth_gmail(req: GmailRequest):
#     gmail = req.gmail
#     # save current user email for callback mapping
#     with open("current_user.txt", "w") as f:
#         f.write(gmail)

#     if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
#         return {"error": f"Missing {GOOGLE_CREDENTIALS_FILE}"}

#     # build flow with explicit redirect URI
#     flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_FILE, GMAIL_SCOPES, redirect_uri=GOOGLE_REDIRECT_URI)
#     auth_url, state = flow.authorization_url(access_type="offline", prompt="consent")
#     return {"auth_url": auth_url}

# @app.get("/oauth2callback")
# def oauth2callback(code: str = None):
#     # Google's redirect will supply `code`
#     if not code:
#         return JSONResponse({"error": "Missing code in callback"}, status_code=400)
#     if not os.path.exists("current_user.txt"):
#         return JSONResponse({"error": "No current user found. Start /auth-gmail first."}, status_code=400)
#     gmail = open("current_user.txt").read().strip()
#     token_file = os.path.join(TOKEN_FOLDER, f"{gmail.replace('@','_')}_gmail.json")
#     try:
#         flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_FILE, GMAIL_SCOPES, redirect_uri=GOOGLE_REDIRECT_URI)
#         flow.fetch_token(code=code)
#         creds = flow.credentials
#         os.makedirs(TOKEN_FOLDER, exist_ok=True)
#         with open(token_file, "w") as f:
#             f.write(creds.to_json())
#         return RedirectResponse("http://127.0.0.1:5500/frontend/index.html")
#     except Exception as e:
#         return JSONResponse({"error": f"Failed to fetch token: {e}"}, status_code=500)

# @app.post("/connect-gmail")
# def connect_gmail(req: GmailRequest):
#     gmail = req.gmail
#     token_file = os.path.join(TOKEN_FOLDER, f"{gmail.replace('@','_')}_gmail.json")
#     if not os.path.exists(token_file):
#         return {"error": "User not authenticated. Please complete /auth-gmail first."}
#     try:
#         creds = Credentials.from_authorized_user_file(token_file, GMAIL_SCOPES)
#         service = build("gmail", "v1", credentials=creds)
#         result = service.users().messages().list(userId="me", maxResults=25).execute()
#         messages = result.get("messages", [])
#         inbox = []
#         for m in messages:
#             detail = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
#             headers = detail.get("payload", {}).get("headers", [])
#             subject = ""
#             sender = ""
#             for h in headers:
#                 if h.get("name") == "Subject": subject = h.get("value", "")
#                 if h.get("name") == "From": sender = h.get("value", "")
#             body = _get_message_body(detail.get("payload", {})) or detail.get("snippet", "")
#             inbox.append({"subject": subject, "sender": sender, "body": clean_html(body), "url": ""})
#         return {"emails": inbox}
#     except Exception as e:
#         return {"error": str(e)}

# # ------------------------------
# # Outlook (Microsoft Graph) OAuth & fetch
# # ------------------------------
# @app.post("/auth-outlook")
# def auth_outlook(req: OutlookRequest):
#     outlook = req.outlook
#     # store mapping for callback
#     with open("current_user_outlook.txt", "w") as f:
#         f.write(outlook)

#     # Build Microsoft OAuth2 authorization URL
#     if "<YOUR_MS_CLIENT_ID>" in MS_CLIENT_ID or "<YOUR_MS_CLIENT_SECRET>" in MS_CLIENT_SECRET:
#         return {"error": "Please set MS_CLIENT_ID and MS_CLIENT_SECRET environment variables (or edit file)."}

#     auth_url = (
#         f"https://login.microsoftonline.com/{MS_TENANT}/oauth2/v2.0/authorize?"
#         f"client_id={urllib.parse.quote(MS_CLIENT_ID)}"
#         f"&response_type=code"
#         f"&redirect_uri={urllib.parse.quote(MS_REDIRECT_URI)}"
#         f"&response_mode=query"
#         f"&scope={MS_SCOPES}"
#         f"&state=12345"
#     )
#     return {"auth_url": auth_url}

# @app.get("/oauth2callback-outlook")
# def oauth2callback_outlook(code: str = None, state: str = None):
#     if not code:
#         return JSONResponse({"error": "Missing code in callback"}, status_code=400)
#     if not os.path.exists("current_user_outlook.txt"):
#         return JSONResponse({"error": "No current user stored. Start /auth-outlook first."}, status_code=400)
#     outlook = open("current_user_outlook.txt").read().strip()
#     token_file = os.path.join(TOKEN_FOLDER, f"{outlook.replace('@','_')}_outlook.json")

#     # Exchange code for tokens
#     token_url = f"https://login.microsoftonline.com/{MS_TENANT}/oauth2/v2.0/token"
#     data = {
#         "client_id": MS_CLIENT_ID,
#         "scope": "offline_access Mail.Read",
#         "code": code,
#         "redirect_uri": MS_REDIRECT_URI,
#         "grant_type": "authorization_code",
#         "client_secret": MS_CLIENT_SECRET,
#     }
#     try:
#         r = requests.post(token_url, data=data)
#         r.raise_for_status()
#         tok = r.json()
#         os.makedirs(TOKEN_FOLDER, exist_ok=True)
#         with open(token_file, "w") as f:
#             json.dump(tok, f)
#         return RedirectResponse("http://127.0.0.1:5500/index.html")
#     except Exception as e:
#         return JSONResponse({"error": f"Token exchange failed: {e}, response: {r.text if 'r' in locals() else ''}"}, status_code=500)

# def _refresh_outlook_token(token_file):
#     with open(token_file, "r") as f:
#         tok = json.load(f)
#     if "refresh_token" not in tok:
#         return tok
#     token_url = f"https://login.microsoftonline.com/{MS_TENANT}/oauth2/v2.0/token"
#     data = {
#         "client_id": MS_CLIENT_ID,
#         "scope": "offline_access Mail.Read",
#         "grant_type": "refresh_token",
#         "refresh_token": tok.get("refresh_token"),
#         "client_secret": MS_CLIENT_SECRET,
#     }
#     r = requests.post(token_url, data=data)
#     r.raise_for_status()
#     new_tok = r.json()
#     # persist new token
#     with open(token_file, "w") as f:
#         json.dump(new_tok, f)
#     return new_tok

# @app.post("/connect-outlook")
# def connect_outlook(req: OutlookRequest):
#     outlook = req.outlook
#     token_file = os.path.join(TOKEN_FOLDER, f"{outlook.replace('@','_')}_outlook.json")
#     if not os.path.exists(token_file):
#         return {"error": "User not authenticated for Outlook. Please complete /auth-outlook first."}
#     try:
#         # refresh if needed (simple implementation)
#         with open(token_file, "r") as f:
#             tok = json.load(f)
#         expires_at = tok.get("expires_at") or (int(time.time()) + int(tok.get("expires_in", 0)))
#         if int(time.time()) > int(expires_at) - 60:
#             tok = _refresh_outlook_token(token_file)
#         access_token = tok.get("access_token")
#         headers = {"Authorization": f"Bearer {access_token}"}
#         # fetch top 25 messages
#         url = "https://graph.microsoft.com/v1.0/me/messages?$top=25"
#         r = requests.get(url, headers=headers)
#         r.raise_for_status()
#         data = r.json()
#         emails = []
#         for msg in data.get("value", []):
#             subj = msg.get("subject", "")
#             sender = ""
#             if msg.get("from") and msg["from"].get("emailAddress"):
#                 sender = msg["from"]["emailAddress"].get("address", "")
#             body = ""
#             if msg.get("body") and msg["body"].get("content"):
#                 body = msg["body"]["content"]
#             emails.append({"subject": subj, "sender": sender, "body": clean_html(body), "url": ""})
#         return {"emails": emails}
#     except Exception as e:
#         return {"error": str(e)}

# # ------------------------------
# # End of backend
# # ------------------------------
import os
os.environ["ACCELERATE_DISABLE_TORCH_COMPILE"] = "1"

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import lightgbm as lgb
from bs4 import BeautifulSoup
import re
import html
import urllib.parse
import tldextract

# --------------------
#  MODEL CONFIG
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256

TRANSFORMER_DIR = "saved_models/phish_transformer"
LGB_MODEL_PATH = "saved_models/url_sender_model.txt"

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
transformer = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR).to(DEVICE)
transformer.eval()

lgb_model = lgb.Booster(model_file=LGB_MODEL_PATH)
softmax = torch.nn.Softmax(dim=-1)


# --------------------
#  PREPROCESSING
# --------------------
def clean_html(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text):
    if not isinstance(text, str):
        return ""
    emoji_pattern = re.compile(r"[\U0001F600-\U0001F64F"
                               r"\U0001F300-\U0001F5FF"
                               r"\U0001F680-\U0001F6FF]+", flags=re.UNICODE)
    text = emoji_pattern.sub("", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?@-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_url(url):
    if not isinstance(url, str):
        return ""
    try:
        parsed = urllib.parse.urlparse(url.strip())
        parsed = parsed._replace(query="")
        return urllib.parse.urlunparse(parsed).lower()
    except:
        return ""


def preprocess_email(subject, body, sender, url):
    cleaned_html = clean_html(body)
    cleaned_body = clean_text(cleaned_html)
    cleaned_subject = clean_text(subject)
    cleaned_url = normalize_url(url)
    return f"{cleaned_subject} . {cleaned_body}", cleaned_url


# --------------------------
#  FEATURE EXTRACTION
# --------------------------
def extract_url_features(url_list):
    rows = []
    for u in url_list:
        u = str(u)
        te = tldextract.extract(u)
        rows.append({
            "url_len": len(u),
            "num_dots": u.count("."),
            "has_ip": int(te.domain.replace(".", "").isdigit()),
            "sus_kw": int(any(k in u for k in ["login", "secure", "verify", "update"]))
        })
    return pd.DataFrame(rows)


def extract_sender_features(sender_list):
    rows = []
    for s in sender_list:
        s = str(s)
        domain = s.split("@")[-1] if "@" in s else ""
        rows.append({
            "sender_len": len(s),
            "has_digits": int(any(ch.isdigit() for ch in s)),
            "domain_len": len(domain),
            "at_count": s.count("@")
        })
    return pd.DataFrame(rows)


# --------------------------
#  PREDICTION
# --------------------------
def predict_email(subject, body, sender, url):
    text, clean_url = preprocess_email(subject, body, sender, url)

    # Transformer prediction
    enc = tokenizer(text, padding="max_length", truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = transformer(**enc).logits
        probs = softmax(logits)[0].cpu().numpy()

    t_score = float(probs[1])

    # Metadata prediction
    url_f = extract_url_features([clean_url])
    sender_f = extract_sender_features([sender])
    meta = pd.concat([url_f, sender_f], axis=1)
    m_score = float(lgb_model.predict(meta.values)[0])

    final = 0.6 * t_score + 0.4 * m_score

    return {
        "phishing_probability": final,
        "transformer_score": t_score,
        "metadata_score": m_score,
        "verdict": "phishing" if final >= 0.5 else "ham",
        "explanation": ("High phishing signals detected" if final >= 0.5 else "Looks safe")
    }


# --------------------------
#  REQUEST MODEL
# --------------------------
class EmailRequest(BaseModel):
    subject: str
    body: str
    sender: str
    url: str = ""


# --------------------------
#  FASTAPI APP
# --------------------------
app = FastAPI(title="NeuroSentinel Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only — lock this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/predict")
def predict(req: EmailRequest):
    return predict_email(req.subject, req.body, req.sender, req.url)


# -------------------------------------------------------------
#  ---- GMAIL LOADER (OAuth Required) ----
# -------------------------------------------------------------
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64


GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_CREDENTIALS = "credentials.json"   # put your Gmail credentials here


def fetch_gmail():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", GMAIL_SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS, GMAIL_SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)

    results = service.users().messages().list(userId="me", maxResults=20).execute()
    messages = results.get("messages", [])

    emails = []

    for msg in messages:
        full = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
        headers = full["payload"]["headers"]

        subject = sender = ""
        for h in headers:
            if h["name"] == "Subject":
                subject = h["value"]
            if h["name"] == "From":
                sender = h["value"]

        body = ""
        parts = full["payload"].get("parts", [])
        for p in parts:
            if p["mimeType"] == "text/html":
                body = base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8")

        emails.append({
            "subject": subject,
            "sender": sender,
            "body": clean_html(body),
            "url": ""
        })

    return emails


@app.get("/fetch-gmail")
def get_gmail():
    return {"emails": fetch_gmail()}


# -------------------------------------------------------------
# ----- OUTLOOK LOADER (Microsoft Graph API) -----
# -------------------------------------------------------------

# import msal
# import requests

# CLIENT_ID = "YOUR_CLIENT_ID"
# TENANT_ID = "YOUR_TENANT_ID"
# CLIENT_SECRET = "YOUR_CLIENT_SECRET"

# AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
# SCOPE = ["https://graph.microsoft.com/.default"]
# GRAPH_URL = "https://graph.microsoft.com/v1.0/me/messages?$top=20"


# def fetch_outlook():
#     app = msal.ConfidentialClientApplication(
#         CLIENT_ID, authority=AUTHORITY, client_credential=CLIENT_SECRET
#     )

#     token = app.acquire_token_for_client(SCOPE)

#     if "access_token" not in token:
#         return []

#     headers = {"Authorization": "Bearer " + token["access_token"]}
#     response = requests.get(GRAPH_URL, headers=headers).json()

#     emails = []

#     for msg in response["value"]:
#         emails.append({
#             "subject": msg.get("subject", ""),
#             "sender": msg["from"]["emailAddress"]["address"],
#             "body": clean_html(msg["body"]["content"]),
#             "url": ""
#         })

#     return emails


# @app.get("/fetch-outlook")
# def get_outlook():
#     return {"emails": fetch_outlook()}
