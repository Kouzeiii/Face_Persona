import os
import json
import uuid
import re
import base64
import secrets
import hmac
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import socket
import ipaddress
import http.client
import ssl
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
import time
import shutil
import threading
import contextlib
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, has_request_context, send_file, get_flashed_messages
from werkzeug.utils import secure_filename
import cv2
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import anthropic

# ── Environment & Configuration ──

def load_dotenv(filepath=None):
    """Lightweight .env loader without third-party dependencies.
    Supports utf-8 / utf-8-sig (BOM), trims 'export ' prefix, strips inline comments, and handles quotes."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(filepath):
        return
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('export '):
                    line = line[7:].strip()
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip()
                if v.startswith(('"', "'")) and len(v) >= 2 and v[0] == v[-1]:
                    v = v[1:-1]
                else:
                    v = v.split('#')[0].strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

load_dotenv()

# Claude AI Model Constants (Configurable via environment variables with official candidate fallbacks)
CLAUDE_SONNET_MODEL = os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')
CLAUDE_HAIKU_MODEL = os.environ.get('ANTHROPIC_HAIKU_MODEL', 'claude-haiku-4-5-20251001')

SONNET_MODEL_CANDIDATES = [
    CLAUDE_SONNET_MODEL,
    'claude-sonnet-4-5-20250929',
    'claude-sonnet-4-6',
    'claude-sonnet-5',
    'claude-sonnet-5-5',
]

HAIKU_MODEL_CANDIDATES = [
    CLAUDE_HAIKU_MODEL,
    'claude-haiku-4-5-20251001',
    'claude-haiku-5-5',
]

INSECURE_SECRET_KEYS = {
    'your_flask_secret_key_here',
    'your_admin_secret_key_here',
    'your_anthropic_api_key_here',
    'facepersona_secret_key_capstone_2026',
    'facepersona_admin_2026',
    'capstone-secret-key',
    'secret',
    'admin',
    'changeme',
    'password',
    '123456',
    'test',
    'placeholder',
}

app = Flask(__name__)

# Security & Session Settings
raw_secret_key = os.environ.get('FLASK_SECRET_KEY')
if (not raw_secret_key
        or raw_secret_key.strip().lower() in INSECURE_SECRET_KEYS
        or len(raw_secret_key.strip()) < 16):
    raw_secret_key = secrets.token_hex(32)
    print("[SECURITY NOTICE] FLASK_SECRET_KEY was unset or insecure. Generated cryptographically secure ephemeral key.")

app.config['SECRET_KEY'] = raw_secret_key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'false').lower() in ('true', '1')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, 'data')
DEFAULT_UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, 'models')
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, 'results')

DATA_DIR = os.environ.get('FACEPERSONA_DATA_DIR') or DEFAULT_DATA_DIR
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.environ.get('FACEPERSONA_MODELS_DIR') or DEFAULT_MODELS_DIR
RESULTS_DIR = os.environ.get('FACEPERSONA_RESULTS_DIR') or DEFAULT_RESULTS_DIR
UPLOAD_DIR = os.environ.get('FACEPERSONA_UPLOAD_DIR') or DEFAULT_UPLOAD_DIR
PROFILES_DIR = os.path.join(DATA_DIR, 'profiles')

FACE_EMBEDDINGS_FILE = os.path.join(PROCESSED_DIR, 'face_embeddings.csv')
PERSONALITY_SCORES_FILE = os.path.join(PROCESSED_DIR, 'personality_scores.csv')
DATASET_FILE = os.path.join(PROCESSED_DIR, 'dataset.csv')
PUBLIC_ASSESSMENTS_FILE = os.path.join(PROCESSED_DIR, 'public_assessments.csv')
TRAINED_MODEL_FILE = os.path.join(MODELS_DIR, 'trained_model.pkl')
MODEL_METRICS_FILE = os.path.join(RESULTS_DIR, 'model_metrics.json')
IMAGE_ASSOCIATIONS_FILE = os.path.join(DATA_DIR, 'image_associations.json')
UNRESOLVED_IMAGES_REPORT_FILE = os.path.join(DATA_DIR, 'unresolved_images_report.json')

# Global Thread & Inter-Process Locks for Public Assessment Storage Integrity
PUBLIC_ASSESSMENT_LOCK = threading.RLock()
PUBLIC_ASSESSMENT_LOCK_FILE = os.path.join(PROCESSED_DIR, 'public_assessments.lock')
_BASELINE_NAMES_CACHE = None

# Reverse Proxy Trust Configuration for Secure Client IP Resolution
TRUST_PROXY_HEADERS = os.environ.get('TRUST_PROXY_HEADERS', 'false').lower() in ('true', '1', 't')
app.config['TRUST_PROXY_HEADERS'] = TRUST_PROXY_HEADERS
TRUSTED_PROXIES = os.environ.get('TRUSTED_PROXIES', '')
app.config['TRUSTED_PROXIES'] = TRUSTED_PROXIES

# Minimum biography word count configuration (Central source of truth for frontend and backend)
# Recommended practical lower minimum: 80-100 meaningful words for adviser review
DEFAULT_MIN_BIOGRAPHY_WORDS = 80
MIN_BIOGRAPHY_WORDS = int(os.environ.get('MIN_BIOGRAPHY_WORDS', DEFAULT_MIN_BIOGRAPHY_WORDS))
app.config['MIN_BIOGRAPHY_WORDS'] = MIN_BIOGRAPHY_WORDS

TRAIT_NAMES = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
TRAIT_LABELS = {
    'openness': 'Openness',
    'conscientiousness': 'Conscientiousness',
    'extraversion': 'Extraversion',
    'agreeableness': 'Agreeableness',
    'neuroticism': 'Neuroticism'
}
TRAIT_DESCRIPTIONS = {
    'openness': 'Curiosity, creativity, and openness to new experiences',
    'conscientiousness': 'Organization, dependability, and self-discipline',
    'extraversion': 'Sociability, assertiveness, and positive emotions',
    'agreeableness': 'Cooperation, trust, and empathy toward others',
    'neuroticism': 'Emotional instability, anxiety, and moodiness'
}
OCEAN_LETTERS = {
    'openness': 'O',
    'conscientiousness': 'C',
    'extraversion': 'E',
    'agreeableness': 'A',
    'neuroticism': 'N'
}
OCEAN_COLORS = {
    'openness': '#7C6FE0',
    'conscientiousness': '#5BA89E',
    'extraversion': '#E07C6F',
    'agreeableness': '#6FAD6F',
    'neuroticism': '#C49A4A'
}

ALLOWED_IMAGE_EXT = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

# ── Celebrity Recognition (AWS Rekognition) ──
# Optional feature: when AWS credentials are configured, uploaded/fetched photos are checked
# against Amazon Rekognition's celebrity database so the subject name and profession can be
# auto-filled. Only well-known public figures are recognized; ordinary people return no match.
AWS_REGION = os.environ.get('AWS_REGION', 'ap-southeast-1').strip() or 'ap-southeast-1'
try:
    CELEBRITY_MIN_CONFIDENCE = float(os.environ.get('CELEBRITY_MIN_CONFIDENCE', '90'))
except ValueError:
    CELEBRITY_MIN_CONFIDENCE = 90.0
REKOGNITION_MAX_BYTES = 5 * 1024 * 1024  # Rekognition limit for raw image bytes

# ── AI-Generated Image Detection ──
# Two layers: (1) metadata/provenance markers left by AI generators (C2PA, IPTC
# "trainedAlgorithmicMedia", Stable Diffusion / ComfyUI PNG chunks, generator names), which are
# definitive when present; (2) Claude Vision visual assessment, blocked only at high confidence.
# No detector is 100% accurate; this is a screening step, documented as a system limitation.
AI_DETECTION_ENABLED = os.environ.get('AI_DETECTION_ENABLED', 'true').strip().lower() not in ('false', '0', 'no')
try:
    AI_IMAGE_BLOCK_CONFIDENCE = float(os.environ.get('AI_IMAGE_BLOCK_CONFIDENCE', '80'))
except ValueError:
    AI_IMAGE_BLOCK_CONFIDENCE = 80.0
_AI_CHECK_CACHE = {}
_AI_CHECK_CACHE_LOCK = threading.Lock()
_AI_CHECK_CACHE_MAX = 256

# ── Private Participant Mode (non-public subjects) ──
# Ordinary participants answer the BFI-10 (Rammstedt & John, 2007) instead of providing a biography.
# Data is stored locally in a separate file, never listed publicly, and identified only by a Participant ID.
PRIVATE_AGE_RANGES = ['18-24', '25-34', '35-44', '45-54', '55+']
PRIVATE_BACKGROUND_MAX_CHARS = 300
BFI10_ITEMS = [
    # (item text, trait, reverse_scored)  — "I see myself as someone who..."
    ("is reserved", 'extraversion', True),
    ("is generally trusting", 'agreeableness', False),
    ("tends to be lazy", 'conscientiousness', True),
    ("is relaxed, handles stress well", 'neuroticism', True),
    ("has few artistic interests", 'openness', True),
    ("is outgoing, sociable", 'extraversion', False),
    ("tends to find fault with others", 'agreeableness', True),
    ("does a thorough job", 'conscientiousness', False),
    ("gets nervous easily", 'neuroticism', False),
    ("has an active imagination", 'openness', False),
]
BFI10_SCALE = ["Disagree strongly", "Disagree a little", "Neutral", "Agree a little", "Agree strongly"]


def get_private_participants_file():
    """Resolved at call time so data-directory overrides (and test sandboxes) are respected."""
    return os.path.join(PROCESSED_DIR, 'private_participants.csv')

PROFESSION_CATEGORIES = sorted([
    {"value": "athlete",        "label": "Athlete / Sports Figure",       "icon": "medal"},
    {"value": "musician",       "label": "Musician / Artist",              "icon": "music"},
    {"value": "actor",          "label": "Actor / Entertainer",            "icon": "film"},
    {"value": "model",          "label": "Model / Fashion Figure",         "icon": "tshirt"},
    {"value": "politician",     "label": "Politician / World Leader",      "icon": "landmark"},
    {"value": "entrepreneur",   "label": "Entrepreneur / Business Leader", "icon": "briefcase"},
    {"value": "scientist",      "label": "Scientist / Researcher",         "icon": "flask"},
    {"value": "influencer",     "label": "Content Creator / Influencer",   "icon": "phone"},
    {"value": "activist",       "label": "Activist / Humanitarian",        "icon": "fist"},
    {"value": "religious",      "label": "Religious / Spiritual Figure",   "icon": "pray"},
    {"value": "historical",     "label": "Historical Figure",              "icon": "scroll"},
    {"value": "other",          "label": "Other / General Public Figure",  "icon": "user"},
], key=lambda x: x['label'])

def init_persistent_storage(data_dir=None, upload_dir=None, models_dir=None):
    """Safely seed external persistent volumes with baseline research data, all 103 profile
    JSON files, models, and images if mounting an external volume.
    Never overwrites or deletes existing files; safe to rerun and idempotent."""
    target_data_dir = data_dir or DATA_DIR
    target_upload_dir = upload_dir or UPLOAD_DIR
    target_models_dir = models_dir or MODELS_DIR
    target_proc_dir = os.path.join(target_data_dir, 'processed')
    target_profiles_dir = os.path.join(target_data_dir, 'profiles')

    for d in [target_data_dir, target_proc_dir, target_models_dir, target_upload_dir, target_profiles_dir, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

    if target_data_dir != DEFAULT_DATA_DIR and os.path.exists(DEFAULT_DATA_DIR):
        # 1. Baseline processed CSV datasets
        default_proc = os.path.join(DEFAULT_DATA_DIR, 'processed')
        if os.path.exists(default_proc):
            for fname in ['dataset.csv', 'face_embeddings.csv', 'personality_scores.csv']:
                src = os.path.join(default_proc, fname)
                dst = os.path.join(target_proc_dir, fname)
                if os.path.exists(src) and not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass

        # 2. Image associations & audit reports
        for fname in ['image_associations.json', 'unresolved_images_report.json']:
            src = os.path.join(DEFAULT_DATA_DIR, fname)
            dst = os.path.join(target_data_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

        # 3. All 103 baseline research profile JSON files
        default_profiles = os.path.join(DEFAULT_DATA_DIR, 'profiles')
        if os.path.exists(default_profiles):
            for fname in os.listdir(default_profiles):
                if fname.lower().endswith('.json'):
                    src = os.path.join(default_profiles, fname)
                    dst = os.path.join(target_profiles_dir, fname)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        try:
                            shutil.copy2(src, dst)
                        except Exception:
                            pass

    # 4. Upload images
    if target_upload_dir != DEFAULT_UPLOAD_DIR and os.path.exists(DEFAULT_UPLOAD_DIR):
        for fname in os.listdir(DEFAULT_UPLOAD_DIR):
            src = os.path.join(DEFAULT_UPLOAD_DIR, fname)
            dst = os.path.join(target_upload_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

    # 5. Trained ML model
    if target_models_dir != DEFAULT_MODELS_DIR and os.path.exists(DEFAULT_MODELS_DIR):
        src = os.path.join(DEFAULT_MODELS_DIR, 'trained_model.pkl')
        dst = os.path.join(target_models_dir, 'trained_model.pkl')
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

init_persistent_storage()


# ── CSRF Protection & Session Helpers ──

def get_csrf_token():
    """Return or generate CSRF token for current session."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']

app.jinja_env.globals['csrf_token'] = get_csrf_token


@app.before_request
def block_sensitive_and_dotfiles():
    """Strictly block direct or encoded access to .env, dotfiles, and hidden system files across all HTTP methods."""
    path = request.path.lower()
    segments = [s for s in path.split('/') if s]
    if any(s.startswith('.') for s in segments) or '.env' in path or '.git' in path:
        return jsonify({'success': False, 'error': 'Access denied: protected system resource.'}), 403


@app.before_request
def csrf_protect():
    """Verify CSRF token on all state-changing requests (POST, PUT, DELETE, PATCH)."""
    if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
        token = request.headers.get('X-CSRFToken') or request.form.get('csrf_token')
        if not token and request.is_json and isinstance(request.json, dict):
            token = request.json.get('csrf_token')
        expected = session.get('csrf_token')
        if not expected or not token or not safe_compare(expected, token):
            return jsonify({'success': False, 'error': 'CSRF token missing or invalid.'}), 403


@app.before_request
def enforce_secure_cookie_on_https():
    """Dynamically enforce secure cookies when accessed over HTTPS or behind an HTTPS reverse proxy."""
    if request.is_secure or request.headers.get('X-Forwarded-Proto', '').lower() == 'https':
        app.config['SESSION_COOKIE_SECURE'] = True


PLACEHOLDER_API_KEYS = {
    'your_anthropic_api_key_here',
    'your_actual_anthropic_api_key_here',
    'placeholder',
    'changeme',
}

def get_claude_client():
    """Initialize Anthropic client using ANTHROPIC_API_KEY environment variable.
    Validates presence, rejects placeholder strings, and configures timeout."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key or not api_key.strip():
        raise ValueError(
            "Anthropic API key is not configured. Please set a valid ANTHROPIC_API_KEY in your .env file."
        )
    cleaned_key = api_key.strip()
    if cleaned_key.lower() in PLACEHOLDER_API_KEYS:
        raise ValueError(
            "Anthropic API key is set to a placeholder value ('your_anthropic_api_key_here'). "
            "Please configure your actual Anthropic API key in .env or at https://console.anthropic.com/."
        )
    return anthropic.Anthropic(api_key=cleaned_key, timeout=30.0)


def call_claude_messages_create(client, candidate_models, **kwargs):
    """Execute client.messages.create with candidate models in fallback order."""
    seen = set()
    models = [m for m in candidate_models if m and not (m in seen or seen.add(m))]
    last_exc = None
    for m in models:
        try:
            return client.messages.create(model=m, **kwargs)
        except (anthropic.NotFoundError, anthropic.BadRequestError) as e:
            err_str = str(e).lower()
            if 'model' in err_str or 'not_found' in err_str:
                last_exc = e
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("No candidate models available for Claude request.")


def format_claude_error(exc):
    """Translate Anthropic API exceptions into clear, user-friendly error messages."""
    if isinstance(exc, anthropic.AuthenticationError):
        return (
            "Anthropic API Authentication Error (401): The provided ANTHROPIC_API_KEY is invalid, revoked, "
            "or unauthorized. Please verify your API key in .env or at https://console.anthropic.com/."
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return (
            "Anthropic API Permission Denied (403): Your account does not have permission to access the requested model."
        )
    if isinstance(exc, anthropic.NotFoundError):
        return f"Anthropic API Resource Not Found (404): {str(exc)}"
    if isinstance(exc, anthropic.RateLimitError):
        return (
            "Anthropic API Rate Limit (429): Your Anthropic account has exceeded its request or token limits. "
            "Please wait a moment and try again."
        )
    if isinstance(exc, anthropic.APITimeoutError):
        return "Anthropic API Timeout: The request to Claude timed out. Please try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Anthropic API Connection Error: Could not connect to Anthropic servers. Please check your network connection."
    if isinstance(exc, anthropic.BadRequestError):
        return f"Anthropic API Bad Request (400): {str(exc)}"
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic API Error ({exc.status_code}): {str(exc)}"
    if isinstance(exc, ValueError):
        return str(exc)
    return f"Claude AI processing error: {str(exc)}"


# ── AI Endpoint Rate Limiting (Server-side API Protection) ──

class SimpleRateLimiter:
    """Thread-safe and cross-process sliding-window rate limiter for public AI endpoints.
    Uses a lightweight SQLite database in DATA_DIR to share rate limits across multiple
    Gunicorn worker processes, with an in-memory fallback for environments without disk write access."""
    def __init__(self, db_path=None):
        self._lock = threading.Lock()
        self._requests = {}
        self._db_path = db_path

    def _get_db_path(self):
        if self._db_path:
            return self._db_path
        try:
            return os.path.join(DATA_DIR, 'rate_limit.db')
        except Exception:
            return None

    def _init_db(self, conn):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                endpoint TEXT,
                client_ip TEXT,
                timestamp REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rate_limits
            ON rate_limits(endpoint, client_ip, timestamp)
        """)

    def is_rate_limited(self, endpoint, client_ip, limit, window_seconds=60):
        """Check if client_ip has exceeded limit requests within window_seconds.
        Synchronized across Gunicorn workers via shared SQLite database in DATA_DIR."""
        now = time.time()
        cutoff = now - window_seconds
        db_path = self._get_db_path()

        if db_path:
            try:
                conn = sqlite3.connect(db_path, timeout=5.0)
                try:
                    self._init_db(conn)
                    with conn:
                        conn.execute(
                            "DELETE FROM rate_limits WHERE endpoint = ? AND client_ip = ? AND timestamp <= ?",
                            (endpoint, client_ip, cutoff)
                        )
                        cur = conn.execute(
                            "SELECT timestamp FROM rate_limits WHERE endpoint = ? AND client_ip = ? ORDER BY timestamp ASC",
                            (endpoint, client_ip)
                        )
                        rows = cur.fetchall()
                        if len(rows) >= limit:
                            oldest = rows[0][0]
                            retry_after = max(1, int(oldest + window_seconds - now))
                            return True, retry_after
                        conn.execute(
                            "INSERT INTO rate_limits (endpoint, client_ip, timestamp) VALUES (?, ?, ?)",
                            (endpoint, client_ip, now)
                        )
                        return False, 0
                finally:
                    conn.close()
            except Exception:
                pass

        # In-memory sliding-window fallback
        with self._lock:
            key = (endpoint, client_ip)
            timestamps = self._requests.get(key, [])
            timestamps = [t for t in timestamps if t > cutoff]
            if len(timestamps) >= limit:
                oldest = timestamps[0]
                retry_after = max(1, int(oldest + window_seconds - now))
                self._requests[key] = timestamps
                return True, retry_after
            timestamps.append(now)
            self._requests[key] = timestamps
            if len(self._requests) > 1000:
                self._requests = {k: v for k, v in self._requests.items() if v and v[-1] > cutoff}
            return False, 0

    def reset(self):
        db_path = self._get_db_path()
        if db_path and os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path, timeout=5.0)
                try:
                    with conn:
                        conn.execute("DELETE FROM rate_limits")
                finally:
                    conn.close()
            except Exception:
                pass
        with self._lock:
            self._requests.clear()

ai_rate_limiter = SimpleRateLimiter()

DEFAULT_LOOPBACK_PROXIES = {'127.0.0.1', '::1', 'localhost'}

def get_trusted_proxy_networks():
    """Parse configured trusted proxy networks from environment or app config."""
    trusted_raw = os.environ.get('TRUSTED_PROXIES', '')
    if app.config.get('TRUSTED_PROXIES'):
        trusted_raw = app.config.get('TRUSTED_PROXIES')

    trust_headers = os.environ.get('TRUST_PROXY_HEADERS', 'false').lower() in ('true', '1', 't')
    if app.config.get('TRUST_PROXY_HEADERS') is not None:
        trust_headers = bool(app.config.get('TRUST_PROXY_HEADERS'))

    nets = set()
    if trust_headers:
        for p in DEFAULT_LOOPBACK_PROXIES:
            try:
                nets.add(ipaddress.ip_network(p, strict=False))
            except ValueError:
                pass

    if trusted_raw:
        items = [i.strip() for i in trusted_raw.split(',') if i.strip()]
        for item in items:
            try:
                nets.add(ipaddress.ip_network(item, strict=False))
            except ValueError:
                pass
    return nets


def is_trusted_proxy(ip_str):
    """Check if direct connection IP is an explicitly configured trusted proxy."""
    if not ip_str:
        return False
    clean_ip = ip_str.strip()
    if clean_ip in ('127.0.0.1', '::1', 'localhost'):
        trust_headers = os.environ.get('TRUST_PROXY_HEADERS', 'false').lower() in ('true', '1', 't')
        if app.config.get('TRUST_PROXY_HEADERS') is not None:
            trust_headers = bool(app.config.get('TRUST_PROXY_HEADERS'))
        if trust_headers or os.environ.get('TRUSTED_PROXIES') or app.config.get('TRUSTED_PROXIES'):
            return True
        return False

    try:
        ip_obj = ipaddress.ip_address(clean_ip)
    except ValueError:
        return False

    networks = get_trusted_proxy_networks()
    for net in networks:
        if ip_obj in net:
            return True
    return False


def get_client_ip():
    """Securely determine the real client IP address for rate limiting and logging.
    Strictly ignores X-Forwarded-For, X-Real-IP, and CF-Connecting-IP unless the direct
    connection (request.remote_addr) originates from an explicitly configured trusted proxy.
    Direct clients cannot spoof forwarding headers to evade rate limits."""
    direct_ip = (request.remote_addr or '127.0.0.1').strip()

    # If direct connection is not from an explicitly trusted proxy, return direct connection IP
    if not is_trusted_proxy(direct_ip):
        return direct_ip

    # Direct connection is from a trusted proxy: safely inspect forwarded headers
    x_forwarded_for = request.headers.get('X-Forwarded-For')
    if x_forwarded_for:
        hops = [h.strip() for h in x_forwarded_for.split(',') if h.strip()]
        # Traverse hops in reverse (from closest proxy to origin client)
        for hop in reversed(hops):
            try:
                ip_obj = ipaddress.ip_address(hop)
                if not is_trusted_proxy(hop):
                    return str(ip_obj)
            except ValueError:
                continue
        if hops:
            try:
                return str(ipaddress.ip_address(hops[0]))
            except ValueError:
                pass

    for header_name in ('CF-Connecting-IP', 'X-Real-IP'):
        header_val = request.headers.get(header_name)
        if header_val:
            clean_val = header_val.strip()
            try:
                return str(ipaddress.ip_address(clean_val))
            except ValueError:
                pass

    return direct_ip

def check_ai_rate_limit(endpoint_name, limit_per_minute=10):
    """Enforce rate limits on public AI endpoints unless requester is authorized researcher or in testing mode."""
    if app.config.get('TESTING') and not app.config.get('TESTING_RATELIMIT'):
        return None

    # Bypass rate limits for verified researchers
    is_auth, _ = verify_admin_authorization()
    if is_auth or session.get('research_admin_authorized'):
        return None

    ip = get_client_ip()
    limited, retry_after = ai_rate_limiter.is_rate_limited(endpoint_name, ip, limit_per_minute, window_seconds=60)
    if limited:
        return jsonify({
            'success': False,
            'ok': False,
            'status': 'RATE_LIMITED',
            'error': f'Rate limit exceeded. Maximum {limit_per_minute} requests per minute allowed for {endpoint_name}. Please wait {retry_after} seconds before trying again.',
            'retry_after': retry_after
        }), 429
    return None


# ── Robust AI Response Parsing & Validation Helpers (Issues 5 & 6) ──

def extract_and_parse_json(raw_text):
    """Extract and parse a JSON object from raw LLM text output.
    Handles ```json ... ``` code fences, surrounding text, and enforces dict schema."""
    if not raw_text or not isinstance(raw_text, str):
        raise ValueError("Empty or non-string response from AI.")
    text = raw_text.strip()
    if '```' in text:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object dict, got {type(data).__name__}.")
    return data


def parse_strict_boolean(val, field_name="boolean"):
    """Strictly parse a boolean value from AI response payload.
    Accepts native bool (True, False).
    Accepts string 'true' / 'false' (case-insensitive).
    Accepts integer 1 (True) or 0 (False), but not arbitrary integers.
    Rejects strings like 'yes', 'no', 'none', '', null/None, dicts, lists, and raises ValueError."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and not isinstance(val, bool) and val in (0, 1):
        return bool(val)
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned == 'true':
            return True
        if cleaned == 'false':
            return False
    raise ValueError(f"Field '{field_name}' must be a boolean (true/false), got {type(val).__name__}: {repr(val)}")


def validate_big_five_scores(scores_dict):
    """Strictly validate Big Five scores dictionary according to research methodology.
    Requirements:
    - Must be a dictionary
    - Must contain all 5 OCEAN traits (openness, conscientiousness, extraversion, agreeableness, neuroticism)
    - Values must be finite numbers (int, float, or valid numerical string)
    - Rejects boolean values (which in Python are subclasses of int)
    - Rejects NaN, inf, -inf
    - Enforces documented scale [0.0, 100.0] inclusive (no silent clamping)
    Returns (cleaned_scores: dict, error_message: str|None)."""
    if not isinstance(scores_dict, dict):
        return None, f"Expected JSON dictionary for scores, got {type(scores_dict).__name__}."
    
    cleaned = {}
    for trait in TRAIT_NAMES:
        if trait not in scores_dict:
            return None, f"Missing required Big Five trait: '{trait}'."
        raw_val = scores_dict[trait]
        if raw_val is None or isinstance(raw_val, bool):
            return None, f"Invalid non-numerical value for trait '{trait}': {repr(raw_val)}."
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            return None, f"Non-numerical value for trait '{trait}': {repr(raw_val)}."
        
        if not np.isfinite(val):
            return None, f"Trait '{trait}' contains invalid non-finite value ({val})."
        
        if val < 0.0 or val > 100.0:
            return None, f"Trait '{trait}' score {val} out of valid range [0, 100]."
        
        cleaned[trait] = round(val, 2)
    
    return cleaned, None


def get_biography_validation_token(name, biography):
    """Generate deterministic session receipt key for a validated name-biography pair."""
    h = hashlib.sha256(f"{name.strip().lower()}:{biography.strip()}".encode('utf-8')).hexdigest()
    return f"val_receipt_{h}"


def count_meaningful_words(text):
    """Count meaningful words in text, ignoring punctuation, formatting, and empty tokens.
    Requires tokens to contain at least one alphanumeric character."""
    if not text or not isinstance(text, str):
        return 0
    cleaned = re.sub(r'https?://\S+', '', text)
    tokens = re.findall(r'\b[a-zA-Z0-9_\'-]+\b', cleaned)
    return len(tokens)


def validate_biography_text(text, min_words=None):
    """Validate that biography meets minimum word requirement and contains genuine content.
    Returns (is_valid: bool, word_count: int, error_message: str|None)."""
    if min_words is None:
        min_words = MIN_BIOGRAPHY_WORDS
    if not text or not isinstance(text, str):
        return False, 0, "Biography cannot be empty."
    cleaned = text.strip()
    if not cleaned:
        return False, 0, "Biography cannot be empty or whitespace only."
    words = count_meaningful_words(cleaned)
    if words < min_words:
        return False, words, f"Biography is too short ({words} word{'s' if words != 1 else ''}). Minimum requirement is {min_words} words of biographical content."
    # Meaningless/repetitive content check
    tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\'-]+\b', cleaned)]
    unique_tokens = set(tokens)
    if len(tokens) >= min_words and len(unique_tokens) < min(15, max(3, min_words // 3)):
        return False, words, "Biography contains repetitive or meaningless text. Please provide genuine biographical content."
    return True, words, None


def normalize_secret_input(value):
    """Strip normal and invisible characters (non-breaking spaces, zero-width spaces, BOM,
    smart quotes) that often get copied along with a pasted key."""
    if value is None:
        return ''
    text = str(value)
    for ch in ('\u00a0', '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff'):
        text = text.replace(ch, '')
    return text.strip().strip('"\'\u201c\u201d\u2018\u2019').strip()


def safe_compare(a, b):
    """Constant-time comparison that never crashes on non-ASCII input
    (hmac.compare_digest raises TypeError for non-ASCII str)."""
    if a is None or b is None:
        return False
    return hmac.compare_digest(str(a).encode('utf-8'), str(b).encode('utf-8'))


def get_admin_key_hash(key_str=None):
    """Compute a truncated SHA-256 fingerprint of the administrative key."""
    if key_str is None:
        key_str = os.environ.get('RESEARCH_ADMIN_KEY', '')
    if not key_str or key_str.strip().lower() in INSECURE_SECRET_KEYS or key_str.strip() in PLACEHOLDER_API_KEYS:
        return ""
    return hashlib.sha256(key_str.strip().encode('utf-8')).hexdigest()[:16]


def verify_admin_authorization():
    """Strictly verify researcher administrative credentials.
    Access is DENIED if RESEARCH_ADMIN_KEY is not configured on the server,
    uses an insecure or placeholder default, is shorter than 12 characters,
    or if the incoming request fails to supply the matching key.
    Query parameters are never accepted for administrative authorization."""
    current_key_hash = get_admin_key_hash()

    # Check if session has active researcher authorization and verify that the server admin key hasn't changed
    if session.get('research_admin_authorized'):
        sess_hash = session.get('research_admin_key_hash')
        if sess_hash and current_key_hash and safe_compare(sess_hash, current_key_hash):
            return True, None
        else:
            # Server admin key changed, removed, or invalidated: revoke session immediately
            session.pop('research_admin_authorized', None)
            session.pop('research_admin_key_hash', None)
            session.modified = True

    admin_key = os.environ.get('RESEARCH_ADMIN_KEY')
    if (not admin_key or not admin_key.strip()
            or admin_key.strip().lower() in INSECURE_SECRET_KEYS
            or admin_key.strip() in PLACEHOLDER_API_KEYS
            or len(admin_key.strip()) < 12):
        return False, "Administrative operations are locked. A secure, non-default RESEARCH_ADMIN_KEY (minimum 12 characters) must be configured on the server, or use the local manage_research.py CLI."

    # Administrative keys accepted ONLY via X-Admin-Key header, JSON payload, or POST form body.
    # Query parameters (e.g. ?admin_key=...) are strictly rejected to prevent credential leakage in logs/history.
    req_key = request.headers.get('X-Admin-Key')
    if not req_key and request.is_json and isinstance(request.json, dict):
        req_key = request.json.get('admin_key')
    if not req_key:
        req_key = request.form.get('admin_key')

    if not req_key or not safe_compare(admin_key.strip(), normalize_secret_input(req_key)):
        return False, "Unauthorized: Invalid administrative credentials."

    return True, None


def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXT


# ── File Path Validation & Controlled Upload Identifiers ──

def validate_and_resolve_upload_path(user_ref):
    """Strictly validates and resolves a user-supplied image reference to a safe file within UPLOAD_DIR.
    Rejects directory traversal, absolute paths outside UPLOAD_DIR, symlinks pointing outside,
    and unapproved extensions."""
    if not user_ref or not isinstance(user_ref, str):
        return None, "Invalid image reference."
    clean_filename = secure_filename(os.path.basename(user_ref.strip()))
    if not clean_filename:
        return None, "Invalid filename."
    if not allowed_image(clean_filename):
        return None, "Unauthorized file extension."
    safe_path = os.path.abspath(os.path.join(UPLOAD_DIR, clean_filename))
    upload_dir_abs = os.path.abspath(UPLOAD_DIR)
    if os.path.commonpath([upload_dir_abs, safe_path]) != upload_dir_abs:
        return None, "Path traversal attempt detected."
    real_path = os.path.realpath(safe_path)
    if os.path.commonpath([upload_dir_abs, real_path]) != upload_dir_abs:
        return None, "Symlink target outside uploads directory."
    if not os.path.isfile(safe_path):
        return None, "Referenced image file does not exist."
    return safe_path, None


# ── External Image URL Security (SSRF Protection) ──

def is_public_ip(ip_str):
    """Check if an IP string is a public, globally routable IP address."""
    try:
        ip = ipaddress.ip_address(ip_str)
        if (ip.is_loopback or ip.is_private or ip.is_link_local or
            ip.is_multicast or ip.is_reserved or ip.is_unspecified or
            not ip.is_global):
            return False
        return True
    except ValueError:
        return False


def resolve_and_validate_public_url(url):
    """Resolve and validate that a URL's destination is strictly a public host.
    Returns (is_safe: bool, error_message: str|None, pinned_ip: str|None)"""
    try:
        parsed = urllib.parse.urlsplit(url)
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}", None

    if parsed.scheme.lower() not in ('http', 'https'):
        return False, "Only HTTP and HTTPS image URLs are permitted.", None

    hostname = parsed.hostname
    if not hostname:
        return False, "URL hostname is missing.", None

    if hostname.lower() in ('localhost', '127.0.0.1', '::1', '0.0.0.0'):
        return False, "Requests to localhost are forbidden.", None

    port = parsed.port or (443 if parsed.scheme.lower() == 'https' else 80)
    try:
        addr_infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
        if not addr_infos:
            return False, "Could not resolve URL hostname.", None

        valid_ip = None
        for family, socktype, proto, canonname, sockaddr in addr_infos:
            ip_str = sockaddr[0]
            if not is_public_ip(ip_str):
                return False, f"Access to private, loopback, or non-public network destination ({ip_str}) is forbidden.", None
            if valid_ip is None:
                valid_ip = ip_str

        return True, None, valid_ip
    except socket.gaierror:
        return False, "Failed to resolve URL host.", None
    except Exception as e:
        return False, f"Host validation error: {str(e)}", None


def validate_url_host_is_public(url):
    """Backward-compatible validation returning (is_safe, error_message)."""
    is_safe, err, _ = resolve_and_validate_public_url(url)
    return is_safe, err


class IPPinnedHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that binds directly to a pre-validated public IP address to prevent DNS rebinding."""
    def __init__(self, host, port=None, pinned_ip=None, **kwargs):
        super().__init__(host, port, **kwargs)
        self.pinned_ip = pinned_ip

    def connect(self):
        self.sock = socket.create_connection((self.pinned_ip, self.port), self.timeout, self.source_address)


class IPPinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that binds directly to a pre-validated public IP while preserving TLS SNI and cert check."""
    def __init__(self, host, port=None, pinned_ip=None, **kwargs):
        super().__init__(host, port, **kwargs)
        self.pinned_ip = pinned_ip

    def connect(self):
        raw_sock = socket.create_connection((self.pinned_ip, self.port), self.timeout, self.source_address)
        self.sock = self._context.wrap_socket(raw_sock, server_hostname=self.host)


def fetch_image_from_url(url):
    """Fetch an image from a public URL safely:
    - Strictly validates scheme and verifies all resolved IPs are public
    - Pins connection to verified public IP to prevent DNS rebinding (TOCTOU)
    - Re-validates every redirect hop against public IP checks
    - Enforces 10-second timeout and 10MB streaming download cap
    - Decodes content via OpenCV to verify authentic image data
    - Returns opaque filename (upload ID), base64 preview, and error.
    """
    current_url = url
    max_redirects = 5

    for _ in range(max_redirects):
        is_safe, err, pinned_ip = resolve_and_validate_public_url(current_url)
        if not is_safe:
            return None, None, err

        try:
            parsed = urllib.parse.urlsplit(current_url)
            scheme = parsed.scheme.lower()
            port = parsed.port or (443 if scheme == 'https' else 80)
            host = parsed.hostname
            path = parsed.path or '/'
            if parsed.query:
                path += '?' + parsed.query

            if scheme == 'https':
                conn = IPPinnedHTTPSConnection(host, port, pinned_ip=pinned_ip, timeout=10)
            else:
                conn = IPPinnedHTTPConnection(host, port, pinned_ip=pinned_ip, timeout=10)

            host_header = host if not parsed.port else f"{host}:{parsed.port}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 FacePersona/2.0',
                'Accept': 'image/avif,image/webp,image/apng,image/png,image/jpeg,image/*,*/*;q=0.8',
                'Host': host_header
            }

            conn.request('GET', path, headers=headers)
            response = conn.getresponse()

            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader('Location')
                conn.close()
                if not location:
                    return None, None, "Redirect without Location header."
                current_url = urllib.parse.urljoin(current_url, location)
                continue

            if response.status != 200:
                conn.close()
                return None, None, f"HTTP Error {response.status}: Failed to download image from URL."

            content_type = response.getheader('Content-Type', '').lower()
            max_size = 10 * 1024 * 1024  # 10MB limit
            chunks = []
            total_bytes = 0
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_size:
                    conn.close()
                    return None, None, "Image exceeds maximum allowed size (10MB)."
                chunks.append(chunk)
            conn.close()
            image_data = b''.join(chunks)

            # Validate authentic image content using OpenCV decode
            nparr = np.frombuffer(image_data, np.uint8)
            decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape[0] < 10 or decoded.shape[1] < 10:
                return None, None, "The fetched URL does not contain valid, readable image data."

            ext = 'jpg'
            if 'png' in content_type or image_data.startswith(b'\x89PNG'):
                ext = 'png'
            elif 'webp' in content_type or (b'WEBP' in image_data[:16]):
                ext = 'webp'

            filename = f"{uuid.uuid4().hex}.{ext}"
            path = os.path.join(UPLOAD_DIR, filename)
            with open(path, 'wb') as f:
                f.write(image_data)
            b64 = base64.b64encode(image_data).decode('utf-8')
            return filename, b64, None

        except Exception as e:
            return None, None, f"Failed to fetch image: {str(e)}"

    return None, None, "Too many redirects encountered while fetching image."


def extract_name_from_url(url):
    """Extract a subject name only from Wikipedia article URLs (e.g. /wiki/Bruno_Mars).
    Direct image links (Google thumbnails, CDNs, etc.) do not contain reliable names
    (they produce values like "images" or random IDs), so they return None."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or '').lower()
        if not (host == 'wikipedia.org' or host.endswith('.wikipedia.org')):
            return None
        match = re.search(r'/wiki/([^/#?]+)', parsed.path)
        if not match:
            return None
        raw = urllib.parse.unquote(match.group(1))
        if ':' in raw:  # Special pages such as File:, Category:, Special:
            return None
        raw = re.sub(r'_\(.*?\)', '', raw)
        name = raw.replace('_', ' ').strip()
        return name or None
    except Exception:
        return None


# ── Safe Atomic File Operations ──

def safe_replace(src, dst, max_retries=5, delay=0.05):
    """Safely replace dst with src with retry logic for transient Windows file locks."""
    for attempt in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, OSError):
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)


def atomic_write_text(filepath, content):
    """Atomically write text content to filepath using a temporary file in the same directory."""
    dirname = os.path.dirname(os.path.abspath(filepath))
    temp_path = os.path.join(dirname, f".tmp_{uuid.uuid4().hex}_{os.path.basename(filepath)}")
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    safe_replace(temp_path, filepath)


def atomic_write_json(filepath, data, indent=2):
    """Atomically serialize JSON data to filepath."""
    content = json.dumps(data, indent=indent)
    atomic_write_text(filepath, content)


def atomic_write_csv(filepath, df):
    """Atomically write pandas DataFrame to CSV at filepath."""
    dirname = os.path.dirname(os.path.abspath(filepath))
    temp_path = os.path.join(dirname, f".tmp_{uuid.uuid4().hex}_{os.path.basename(filepath)}")
    df.to_csv(temp_path, index=False)
    safe_replace(temp_path, filepath)


@contextlib.contextmanager
def interprocess_file_lock(lock_filepath, timeout=10.0, poll_interval=0.05):
    """Cross-platform inter-process file lock using atomic OS file creation.
    Provides safe mutual exclusion across multiple worker processes."""
    start_time = time.time()
    lock_fd = None
    while True:
        try:
            lock_fd = os.open(lock_filepath, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except (FileExistsError, OSError):
            try:
                if os.path.exists(lock_filepath):
                    mtime = os.path.getmtime(lock_filepath)
                    if time.time() - mtime > 60.0:  # Stale lock timeout: 60s
                        try:
                            os.remove(lock_filepath)
                            continue
                        except OSError:
                            pass
            except Exception:
                pass
            if time.time() - start_time >= timeout:
                raise TimeoutError(f"Could not acquire lock on {lock_filepath} within {timeout}s.")
            time.sleep(poll_interval)
    try:
        yield
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        try:
            if os.path.exists(lock_filepath):
                os.remove(lock_filepath)
        except OSError:
            pass


@contextlib.contextmanager
def public_assessment_transaction_lock(timeout=10.0):
    """Composite in-process and inter-process lock protecting public assessment storage."""
    with PUBLIC_ASSESSMENT_LOCK:
        with interprocess_file_lock(PUBLIC_ASSESSMENT_LOCK_FILE, timeout=timeout):
            yield


# ── Image Association & Privacy Helpers ──

def load_image_associations():
    if os.path.exists(IMAGE_ASSOCIATIONS_FILE):
        try:
            with open(IMAGE_ASSOCIATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_image_association(name, filename, status='verified', method='upload'):
    associations = load_image_associations()
    associations[name] = {
        'image_filename': filename,
        'status': status,
        'method': method,
        'updated_at': datetime.now(timezone.utc).isoformat()
    }
    try:
        atomic_write_json(IMAGE_ASSOCIATIONS_FILE, associations)
    except Exception:
        pass


def remove_image_association(name):
    associations = load_image_associations()
    record = associations.pop(name, None)
    try:
        atomic_write_json(IMAGE_ASSOCIATIONS_FILE, associations)
    except Exception:
        pass
    return record


def get_subjects_referencing_image(filename):
    """Return a sorted list of all subject names that reference this image filename."""
    if not filename:
        return []
    clean_name = secure_filename(os.path.basename(filename))
    referencing = set()
    if os.path.exists(PROFILES_DIR):
        for fname in os.listdir(PROFILES_DIR):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(PROFILES_DIR, fname), 'r', encoding='utf-8') as f:
                        pdata = json.load(f)
                        if pdata.get('image_filename') == clean_name:
                            sub_name = pdata.get('name')
                            if sub_name:
                                referencing.add(sub_name)
                except Exception:
                    pass
    associations = load_image_associations()
    for sub_name, rec in associations.items():
        if isinstance(rec, dict) and rec.get('image_filename') == clean_name:
            referencing.add(sub_name)
    return sorted(list(referencing))


def is_image_associated_or_saved(filename):
    """Check if an image filename belongs to any saved profile or registered image association."""
    return len(get_subjects_referencing_image(filename)) > 0


def cleanup_ephemeral_file(filepath):
    """Safely cleans up an unshared, unassociated temporary image file.
    Guarantees that files associated with any subject, shared files, or files outside UPLOAD_DIR
    are never deleted."""
    if not filepath or not os.path.exists(filepath):
        return
    filename = os.path.basename(filepath)
    if is_image_associated_or_saved(filename):
        return
    safe_path, err = validate_and_resolve_upload_path(filename)
    if safe_path and os.path.exists(safe_path):
        try:
            os.remove(safe_path)
        except Exception:
            pass


def get_model_outdated_status():
    """Check if the current model metrics are flagged as outdated."""
    if os.path.exists(MODEL_METRICS_FILE):
        try:
            with open(MODEL_METRICS_FILE, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            return metrics.get('is_outdated', False), metrics.get('outdated_reason', '')
        except Exception:
            pass
    return False, ''


# ── Content Moderation & Facial Feature Extraction ──

# ── Image encoding for Claude Vision (downscaled for speed) ──
CLAUDE_IMAGE_MAX_SIDE = 1024
CLAUDE_IMAGE_KEEP_RAW_BYTES = 1_000_000


def encode_image_for_claude(image_path, max_side=CLAUDE_IMAGE_MAX_SIDE):
    """Return (media_type, base64_data) for Claude Vision.
    Large or unsupported images (e.g. BMP) are resized so the longest side is at most `max_side`
    and re-encoded as JPEG. Smaller uploads are much faster to send and analyze, with the same result."""
    ext = image_path.rsplit('.', 1)[-1].lower()
    with open(image_path, 'rb') as f:
        raw = f.read()
    native = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}
    img = cv2.imread(image_path)
    if img is not None:
        h, w = img.shape[:2]
        if ext not in native or max(h, w) > max_side or len(raw) > CLAUDE_IMAGE_KEEP_RAW_BYTES:
            scale = min(1.0, max_side / float(max(h, w)))
            if scale < 1.0:
                img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if ok:
                return 'image/jpeg', base64.b64encode(buf.tobytes()).decode('utf-8')
    return native.get(ext, 'image/jpeg'), base64.b64encode(raw).decode('utf-8')


def image_digest(image_path):
    try:
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


# Images whose explicit-content check already passed in /check_photo (keyed by content hash),
# so /analyze does not repeat the same moderation call for the same photo.
_MODERATION_PASSED = {}
_MODERATION_PASSED_LOCK = threading.Lock()


def remember_moderation_passed(digest):
    if not digest:
        return
    with _MODERATION_PASSED_LOCK:
        if len(_MODERATION_PASSED) >= 512:
            _MODERATION_PASSED.pop(next(iter(_MODERATION_PASSED)))
        _MODERATION_PASSED[digest] = True


def moderation_already_passed(image_path):
    digest = image_digest(image_path)
    with _MODERATION_PASSED_LOCK:
        return bool(digest and _MODERATION_PASSED.get(digest))


def moderate_image(image_path, profession='other'):
    """Check image for inappropriate content using Claude Vision.
    Context-aware: allows swimwear for models/athletes, etc.
    Returns (is_safe, reason)."""
    if moderation_already_passed(image_path):
        return True, "Already passed the photo check.", 'PASSED'
    try:
        media_type, b64_image = encode_image_for_claude(image_path)

        profession_contexts = {
            'model':       'This subject is a professional model, so swimwear, lingerie, and fashion editorial photos are acceptable.',
            'athlete':     'This subject is an athlete, so sports attire, swimwear (e.g. swimmers), and athletic wear are acceptable.',
            'actor':       'This subject is an actor/entertainer, so costume photos and artistic editorial shots are acceptable.',
            'musician':    'This subject is a musician, so stage costumes and artistic/editorial photos are acceptable.',
            'politician':  'This subject is a politician, so only formal or professional attire is expected.',
            'entrepreneur':'This subject is a business figure, so professional and casual attire are acceptable.',
            'scientist':   'This subject is a scientist/researcher, so professional and casual attire are acceptable.',
            'influencer':  'This subject is a content creator, so lifestyle and casual photos are acceptable.',
            'activist':    'This subject is an activist, so protest and event photos are acceptable.',
            'religious':   'This subject is a religious figure, so traditional/ceremonial attire is acceptable.',
            'historical':  'This subject is a historical figure, so period-appropriate attire including classical art depictions are acceptable.',
            'other':       'Standard content moderation applies.',
        }
        context_note = profession_contexts.get(profession, profession_contexts['other'])

        client = get_claude_client()
        message = call_claude_messages_create(
            client,
            HAIKU_MODEL_CANDIDATES,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": f"""You are a content moderator. Review this image for inappropriate content.

Context: {context_note}

BLOCK only if the image contains:
- Explicit nudity or pornographic content
- Graphic sexual acts
- Graphic violence or gore

DO NOT block:
- Swimwear, bikinis, athletic wear (especially for models/athletes)
- Lingerie in a fashion/editorial context
- Shirtless athletes or performers
- Period costumes or classical art
- Normal portraits and professional photos

Respond ONLY with valid JSON (no markdown):
{{"is_safe": true/false, "reason": "<one sentence>"}}"""
                    }
                ]
            }]
        )
        data = extract_and_parse_json(message.content[0].text)
        if 'is_safe' not in data:
            return False, "Invalid response schema: missing 'is_safe' property.", 'FAILED'
        is_safe = parse_strict_boolean(data['is_safe'], 'is_safe')
        reason = str(data.get('reason') or '').strip()
        if is_safe:
            return True, reason or "Image passed content moderation.", 'PASSED'
        else:
            return False, reason or "Inappropriate image detected.", 'FAILED'
    except (json.JSONDecodeError, ValueError) as e:
        return False, f"Invalid moderation response: {str(e)}", 'FAILED'
    except Exception as e:
        return False, f"Content moderation service unavailable: {format_claude_error(e)}", 'UNAVAILABLE'


# ── Celebrity Recognition Helpers (AWS Rekognition) ──

def is_celebrity_recognition_enabled():
    """Feature is enabled only when AWS credentials are present and boto3 is installed."""
    if os.environ.get('CELEBRITY_RECOGNITION_ENABLED', 'true').strip().lower() in ('false', '0', 'no'):
        return False
    if not (os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY')):
        return False
    try:
        import boto3  # noqa: F401
    except ImportError:
        return False
    return True


def get_rekognition_client():
    import boto3
    return boto3.client('rekognition', region_name=AWS_REGION)


def prepare_image_bytes_for_rekognition(image_path):
    """Rekognition accepts only JPEG/PNG up to 5MB. Convert/shrink other images to JPEG."""
    ext = image_path.rsplit('.', 1)[-1].lower()
    with open(image_path, 'rb') as f:
        raw = f.read()
    if ext in ('jpg', 'jpeg', 'png') and len(raw) <= REKOGNITION_MAX_BYTES:
        return raw

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file for celebrity recognition.")
    quality = 92
    while True:
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise ValueError("Could not convert image for celebrity recognition.")
        data = buf.tobytes()
        if len(data) <= REKOGNITION_MAX_BYTES:
            return data
        # Too large: downscale and retry
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, int(w * 0.75)), max(1, int(h * 0.75))))
        quality = max(70, quality - 5)


def recognize_celebrity(image_path):
    """Identify a well-known public figure in the photo using AWS Rekognition.
    Returns (result_or_None, error_or_None, status) where status is
    'RECOGNIZED', 'NOT_RECOGNIZED', 'DISABLED' or 'UNAVAILABLE'."""
    if not is_celebrity_recognition_enabled():
        return None, "Celebrity recognition is not configured.", 'DISABLED'
    try:
        image_bytes = prepare_image_bytes_for_rekognition(image_path)
        client = get_rekognition_client()
        response = client.recognize_celebrities(Image={'Bytes': image_bytes})
    except Exception as e:
        return None, f"Celebrity recognition service unavailable: {str(e)}", 'UNAVAILABLE'

    candidates = []
    for celeb in response.get('CelebrityFaces', []) or []:
        try:
            confidence = float(celeb.get('MatchConfidence') or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        name = str(celeb.get('Name') or '').strip()
        if name and confidence >= CELEBRITY_MIN_CONFIDENCE:
            candidates.append({
                'name': name,
                'confidence': round(confidence, 2),
                'urls': [u for u in (celeb.get('Urls') or []) if isinstance(u, str)][:3],
            })

    if not candidates:
        return None, None, 'NOT_RECOGNIZED'

    best = max(candidates, key=lambda c: c['confidence'])
    best['other_faces'] = len(candidates) - 1
    return best, None, 'RECOGNIZED'


def suggest_profession_for_public_figure(name):
    """Map a recognized public figure's name to one of the PROFESSION_CATEGORIES values
    using a text-only Claude request. Returns a category value or None on failure."""
    valid_values = [p['value'] for p in PROFESSION_CATEGORIES]
    options = "\n".join(f'- {p["value"]}: {p["label"]}' for p in PROFESSION_CATEGORIES)
    try:
        client = get_claude_client()
        message = call_claude_messages_create(
            client,
            HAIKU_MODEL_CANDIDATES,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"""Which category best describes the primary public role of the well-known public figure "{name}"?

Categories:
{options}

Respond ONLY with valid JSON (no markdown): {{"profession": "<one category value from the list>"}}"""
            }]
        )
        data = extract_and_parse_json(message.content[0].text)
        value = str(data.get('profession') or '').strip().lower()
        return value if value in valid_values else None
    except Exception:
        return None


# ── AI-Generated Image Detection Helpers ──

# (marker bytes, human-readable source). Matched case-insensitively against the raw file bytes.
AI_METADATA_MARKERS = [
    (b'compositewithtrainedalgorithmicmedia', 'IPTC metadata: composite with AI-generated content'),
    (b'trainedalgorithmicmedia', 'IPTC metadata: AI-generated (trainedAlgorithmicMedia)'),
    (b'midjourney', 'Midjourney'),
    (b'dall-e', 'DALL-E'),
    (b'dall\xc2\xb7e', 'DALL-E'),
    (b'stable diffusion', 'Stable Diffusion'),
    (b'novelai', 'NovelAI'),
    (b'adobe firefly', 'Adobe Firefly'),
    (b'leonardo.ai', 'Leonardo.Ai'),
    (b'ideogram', 'Ideogram'),
    (b'flux.1', 'FLUX'),
]


def detect_ai_metadata(image_path):
    """Look for provenance markers that AI image generators embed in the file.
    Returns a short description of the source if found, otherwise None.
    Note: markers are removed when an image is screenshotted or re-saved by some apps,
    so absence of markers does NOT prove an image is real."""
    try:
        with open(image_path, 'rb') as f:
            raw = f.read()
    except Exception:
        return None
    lower = raw.lower()

    for marker, source in AI_METADATA_MARKERS:
        if marker in lower:
            return source

    # C2PA content credentials signed by an AI provider (e.g. ChatGPT / OpenAI images)
    if b'c2pa' in lower:
        for provider, label in ((b'openai', 'OpenAI (ChatGPT / DALL-E)'), (b'google', 'Google AI'),
                                (b'firefly', 'Adobe Firefly'), (b'microsoft designer', 'Microsoft Designer')):
            if provider in lower:
                return f"C2PA content credentials: {label}"

    # PNG text chunks written by Stable Diffusion WebUI (A1111/Forge) and ComfyUI
    if raw[:8] == b'\x89PNG\r\n\x1a\n':
        if b'tEXtparameters' in raw and (b'Steps:' in raw or b'Sampler:' in raw):
            return 'Stable Diffusion (generation parameters in PNG)'
        if (b'tEXtprompt' in raw or b'tEXtworkflow' in raw) and b'class_type' in raw:
            return 'ComfyUI (workflow in PNG)'
    return None


def assess_ai_generated_via_claude(image_path):
    """Ask Claude Vision to classify the image as a real photograph vs AI-generated / digital art.
    Returns (is_ai, confidence_0_100, reason, classification) or raises on service errors."""
    media_type, b64_data = encode_image_for_claude(image_path)
    client = get_claude_client()
    message = call_claude_messages_create(
        client,
        SONNET_MODEL_CANDIDATES,
        max_tokens=250,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
                {"type": "text", "text": """You are screening images for an academic research system that only accepts REAL camera photographs of people.

Classify this image into exactly one category:
- "real_photo": an actual camera photograph of a real person (retouching, beauty filters, color grading, stage lighting, low resolution or compression are still real photos).
- "ai_generated": made or heavily altered by an AI image generator or face-swap/deepfake (Midjourney, DALL-E, Stable Diffusion, etc.), even if it looks photorealistic.
- "digital_art": an illustration, painting, drawing, cartoon/anime style, 3D render, or stylized artwork of a person (AI-made or not).

Signs of AI generation or digital art: painterly or airbrushed rendering, glossy plastic skin with no pores, hair drawn as smooth clumps, overly saturated sunset/fantasy backgrounds, warped or melted details (hands, fingers, teeth, ears, jewelry), garbled text/logos, asymmetric eyes, impossible lighting, and an overall "rendered" or "artwork" look.

"confidence" is how sure you are that the image is NOT a real photograph (0 = certainly a real photo, 100 = certainly AI-generated or digital art).

Respond ONLY with valid JSON (no markdown):
{"classification": "real_photo" | "ai_generated" | "digital_art", "confidence": <0-100>, "reason": "<one short sentence>"}"""}
            ]
        }]
    )
    data = extract_and_parse_json(message.content[0].text)
    classification = str(data.get('classification') or '').strip().lower()
    if classification not in ('real_photo', 'ai_generated', 'digital_art'):
        # Backward-compatible with a plain {"ai_generated": bool} answer
        if 'ai_generated' in data:
            classification = 'ai_generated' if parse_strict_boolean(data['ai_generated'], 'ai_generated') else 'real_photo'
        else:
            raise ValueError("Missing 'classification' property in AI detection response.")
    try:
        confidence = max(0.0, min(100.0, float(data.get('confidence', 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(data.get('reason') or '').strip()
    return classification != 'real_photo', confidence, reason, classification


def check_ai_generated_image(image_path):
    """Combined AI-generated image screening (metadata first, then Claude Vision).
    Returns a dict: {'is_ai': bool, 'method': 'metadata'|'vision'|None, 'confidence': float|None,
                     'reason': str, 'status': 'PASSED'|'FLAGGED'|'UNAVAILABLE'|'DISABLED'}.
    Vision-service failures do not block the user (status UNAVAILABLE, is_ai False), because
    this is a heuristic screening step; metadata matches always block."""
    if not AI_DETECTION_ENABLED:
        return {'is_ai': False, 'method': None, 'confidence': None, 'reason': '', 'status': 'DISABLED'}

    try:
        with open(image_path, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        digest = None
    if digest:
        with _AI_CHECK_CACHE_LOCK:
            cached = _AI_CHECK_CACHE.get(digest)
        if cached:
            return dict(cached)

    source = detect_ai_metadata(image_path)
    if source:
        result = {'is_ai': True, 'method': 'metadata', 'confidence': 100.0,
                  'reason': f"The image file contains AI-generator metadata ({source}).", 'status': 'FLAGGED'}
    else:
        try:
            is_ai, confidence, reason, classification = assess_ai_generated_via_claude(image_path)
            flagged = is_ai and confidence >= AI_IMAGE_BLOCK_CONFIDENCE
            label = 'digital artwork / illustration' if classification == 'digital_art' else 'AI-generated'
            result = {'is_ai': flagged, 'method': 'vision', 'confidence': confidence,
                      'classification': classification,
                      'reason': reason or (f'The image appears to be {label}.' if flagged else ''),
                      'status': 'FLAGGED' if flagged else 'PASSED'}
        except Exception as e:
            # Do not cache failures so the next attempt can retry
            err = str(e) if isinstance(e, ValueError) else format_claude_error(e)
            print(f"[AI-CHECK] {os.path.basename(image_path)}: UNAVAILABLE - {err}", flush=True)
            return {'is_ai': False, 'method': 'vision', 'confidence': None,
                    'reason': f"AI-image check unavailable: {err}",
                    'status': 'UNAVAILABLE'}

    print(f"[AI-CHECK] {os.path.basename(image_path)}: {result['status']} via {result['method']}"
          f" (classification={result.get('classification', '-')}, confidence={result['confidence']}) {result['reason']}",
          flush=True)
    if digest:
        with _AI_CHECK_CACHE_LOCK:
            if len(_AI_CHECK_CACHE) >= _AI_CHECK_CACHE_MAX:
                _AI_CHECK_CACHE.pop(next(iter(_AI_CHECK_CACHE)))
            _AI_CHECK_CACHE[digest] = dict(result)
    return result


NO_FACE_MESSAGES = {
    'covered': ("No visible face detected: the subject's face is covered (e.g. mask, helmet, or costume). "
                "FacePersona analyzes visible facial features, so subjects whose face is hidden, such as masked "
                "performers, cannot be analyzed from this photo. Please use a photo where the face is clearly visible."),
    'turned_away': ("No face detected: the subject is facing away or shown in profile. "
                    "Please use a front-facing photo where both eyes are visible."),
    'too_small_or_blurry': ("No face detected: the face is too small, blurry, or dark in this photo. "
                            "Please use a clearer, closer photo of the subject's face."),
    'no_person': ("No face detected: this photo does not appear to show a person. "
                  "Please upload a clear photo of the subject's face."),
}
GENERIC_NO_FACE_MESSAGE = "No face detected. Please upload a clear facial image."


def explain_missing_face(image_path):
    """Called only when OpenCV finds no face. Asks Claude Vision WHY (covered face, turned away,
    too small, no person) so the user gets a specific message. Falls back to the generic message.
    Returns (category, message)."""
    try:
        media_type, b64_data = encode_image_for_claude(image_path)
        client = get_claude_client()
        message = call_claude_messages_create(
            client,
            HAIKU_MODEL_CANDIDATES,
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
                    {"type": "text", "text": """A face detector found no visible human face in this image. Pick the single best explanation:
- "covered": a person is present but their face is covered or replaced (mask, helmet, costume head, face fully hidden)
- "turned_away": a person is present but facing away or in strict profile
- "too_small_or_blurry": a face is present but too small, blurry, dark, or partially cropped
- "no_person": there is no person in the image
- "other": none of the above

Respond ONLY with valid JSON (no markdown): {"category": "<one of the values above>"}"""}
                ]
            }]
        )
        data = extract_and_parse_json(message.content[0].text)
        category = str(data.get('category') or '').strip().lower()
        if category in NO_FACE_MESSAGES:
            return category, NO_FACE_MESSAGES[category]
        return 'other', GENERIC_NO_FACE_MESSAGE
    except Exception:
        return 'unknown', GENERIC_NO_FACE_MESSAGE


# ── Face quality screening (blur, multiple people, obscured faces) ──
FACE_BLUR_HARD_MIN = 25.0  # Laplacian variance of the 128x128 face crop; real photos are almost always far above this
FACE_QUALITY_MESSAGES = {
    'blurred': ("The face in this photo is too blurry or distorted (e.g. motion blur or artistic effect). "
                "Please upload a clear, sharp photo where the face is easy to see."),
    'multiple_people': ("This photo shows more than one person. Please upload a photo with only the subject in it, "
                        "so the system analyzes the right face."),
    'covered': NO_FACE_MESSAGES['covered'],
    'not_real_face': ("The detected face does not look like a real person's face (e.g. a drawing, statue, mask, or object). "
                      "Please upload a real photo of the subject."),
}
_FACE_QUALITY_CACHE = {}
_FACE_QUALITY_CACHE_LOCK = threading.Lock()


def measure_face_sharpness(image_path):
    """Laplacian variance of the largest detected face (resized to 128x128). Returns None if no face."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    roi = cv2.resize(gray[y:y + h, x:x + w], (128, 128))
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def check_face_quality(image_path):
    """Run after a face is detected. Returns (ok, category, message).
    1) Hard blur floor (no API call). 2) Claude Vision check for blur, multiple people, covered or non-real faces.
    Service failures do not block (heuristic screening)."""
    sharpness = measure_face_sharpness(image_path)
    if sharpness is not None and sharpness < FACE_BLUR_HARD_MIN:
        return False, 'blurred', FACE_QUALITY_MESSAGES['blurred']

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        digest = hashlib.sha256(image_data).hexdigest()
        with _FACE_QUALITY_CACHE_LOCK:
            if digest in _FACE_QUALITY_CACHE:
                return _FACE_QUALITY_CACHE[digest]
        media_type, b64_data = encode_image_for_claude(image_path)
        client = get_claude_client()
        message = call_claude_messages_create(
            client,
            HAIKU_MODEL_CANDIDATES,
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
                    {"type": "text", "text": """This photo will be used for facial feature analysis of ONE person. Judge only whether the face is usable:
- "ok": one real person's face is clearly visible and reasonably sharp (normal lighting/compression/slight softness is fine; small background people are fine)
- "blurred": the main face is heavily blurred, smeared, motion-blurred, or distorted so facial features cannot be made out
- "multiple_people": two or more people's faces are similarly prominent, so it is unclear who the subject is
- "covered": the face is mostly covered or hidden (mask, helmet, hands, hair, heavy shadow)
- "not_real_face": the face is not a real human (drawing, statue, doll, cartoon, object, pattern)

Respond ONLY with valid JSON (no markdown): {"verdict": "<one value above>"}"""}
                ]
            }]
        )
        data = extract_and_parse_json(message.content[0].text)
        verdict = str(data.get('verdict') or '').strip().lower()
        if verdict in FACE_QUALITY_MESSAGES:
            result = (False, verdict, FACE_QUALITY_MESSAGES[verdict])
        else:
            result = (True, 'ok', '')
        with _FACE_QUALITY_CACHE_LOCK:
            if len(_FACE_QUALITY_CACHE) >= 256:
                _FACE_QUALITY_CACHE.pop(next(iter(_FACE_QUALITY_CACHE)))
            _FACE_QUALITY_CACHE[digest] = result
        print(f"[FACE-CHECK] {os.path.basename(image_path)}: {verdict or 'ok'} (sharpness={sharpness and round(sharpness, 1)})", flush=True)
        return result
    except Exception as e:
        print(f"[FACE-CHECK] {os.path.basename(image_path)}: UNAVAILABLE - {e}", flush=True)
        return True, 'unavailable', ''


def detect_face_present(image_path):
    """Quick OpenCV Haar check used for early feedback in /check_photo."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0


# ── Private Participant Helpers ──

def compute_bfi10_scores(answers):
    """Score BFI-10 answers (10 integers, 1-5) into Big Five traits on a 0-100 scale.
    Each trait = mean of its two items (one reverse-scored), mapped from 1-5 to 0-100.
    Returns (scores_dict, error)."""
    if not isinstance(answers, (list, tuple)) or len(answers) != len(BFI10_ITEMS):
        return None, f"Please answer all {len(BFI10_ITEMS)} questions."
    values = []
    for i, a in enumerate(answers):
        if isinstance(a, bool):
            return None, f"Question {i + 1} has an invalid answer."
        if isinstance(a, int):
            v = a
        elif isinstance(a, str) and a.strip().isdigit():
            v = int(a.strip())
        else:
            return None, f"Question {i + 1} has an invalid answer."
        if not 1 <= v <= 5:
            return None, f"Question {i + 1} must be answered from 1 to 5."
        values.append(v)
    per_trait = {t: [] for t in TRAIT_NAMES}
    for (text, trait, reverse), v in zip(BFI10_ITEMS, values):
        per_trait[trait].append(6 - v if reverse else v)
    scores = {t: round((sum(vals) / len(vals) - 1) / 4 * 100, 1) for t, vals in per_trait.items()}
    return scores, None


QUESTIONNAIRE_MIN_SECONDS = 15


def get_response_quality_flags(answers, duration_seconds=None):
    """Flag (never block) low-effort questionnaire responses for researcher review."""
    flags = []
    counts = {v: answers.count(v) for v in set(answers)}
    most = max(counts.values()) if counts else 0
    if most == len(answers):
        flags.append('Same answer for all items')
    elif most >= 8:
        flags.append(f'{most} of {len(answers)} answers identical')
    # Contradiction check: each trait has one normal and one reverse-keyed item.
    # Strongly agreeing (or disagreeing) with BOTH items of a pair is inconsistent.
    pairs = {}
    for (text, trait, reverse), v in zip(BFI10_ITEMS, answers):
        pairs.setdefault(trait, []).append(v)
    contradictions = sum(1 for vals in pairs.values() if len(vals) == 2 and
                         ((vals[0] >= 4 and vals[1] >= 4) or (vals[0] <= 2 and vals[1] <= 2)))
    if contradictions >= 3:
        flags.append(f'Inconsistent answers on {contradictions} of 5 traits')
    if duration_seconds is not None and duration_seconds < QUESTIONNAIRE_MIN_SECONDS:
        flags.append(f'Completed in {duration_seconds:.1f}s (under {QUESTIONNAIRE_MIN_SECONDS}s)')
    return flags


def new_participant_id():
    return f"P-{uuid.uuid4().hex[:8].upper()}"


def extract_face_embedding(image_path):
    """Detect face using OpenCV Haar Cascade and extract 512-dimensional HOG features.
    If no face is detected, returns an error without extracting from the whole image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None, "Could not read image file."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Reject image immediately if no face is detected
        if len(faces) == 0:
            return None, None, "No face detected. Please upload a clear facial image."

        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        gray_resized = cv2.resize(face_roi, (128, 128))
        face_box = {
            'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
            'img_w': img.shape[1], 'img_h': img.shape[0]
        }

        hog = cv2.HOGDescriptor(
            _winSize=(128, 128), _blockSize=(16, 16),
            _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9
        )
        features = hog.compute(gray_resized).flatten()
        embedding = features[:512] if len(features) >= 512 else np.pad(features, (0, 512 - len(features)))
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Build HOG cell grid visualization data (16x16 cells, 9 bins each)
        cell_size = 8
        n_cells_x = 128 // cell_size  # 16
        n_cells_y = 128 // cell_size  # 16
        n_bins = 9
        cell_data = []
        for cy in range(min(8, n_cells_y)):
            row = []
            for cx in range(min(8, n_cells_x)):
                cell_idx = (cy * 8 + cx) * n_bins
                if cell_idx + n_bins <= len(embedding):
                    cell_vals = [round(float(embedding[cell_idx + b]), 4) for b in range(n_bins)]
                    dominant_bin = int(np.argmax(cell_vals))
                    magnitude = round(float(sum(cell_vals)), 4)
                else:
                    cell_vals = [0.0] * n_bins
                    dominant_bin = 0
                    magnitude = 0.0
                row.append({'bins': cell_vals, 'dominant': dominant_bin, 'magnitude': magnitude})
            cell_data.append(row)

        embedding_info = {
            'face_detected': True,
            'face_box': face_box,
            'embedding_dim': 512,
            'norm_before': float(norm),
            'top10_values': [round(float(v), 6) for v in embedding[:10]],
            'top50_values': [round(float(v), 4) for v in embedding[:50]],
            'cell_grid': cell_data,
            'stats': {
                'min': round(float(embedding.min()), 6),
                'max': round(float(embedding.max()), 6),
                'mean': round(float(embedding.mean()), 6),
                'std': round(float(embedding.std()), 6),
                'nonzero': int(np.count_nonzero(embedding))
            }
        }
        return embedding.tolist(), embedding_info, None
    except Exception as e:
        return None, None, str(e)


# ── Claude AI Personality Scoring & Validation ──

def score_personality_via_gemini(name, biography_text, profession='other', profession_label=None):
    try:
        client = get_claude_client()
        if not profession_label:
            profession_label = next((p['label'] for p in PROFESSION_CATEGORIES if p['value'] == profession), 'Public Figure')
        prompt = f"""You are a personality psychologist expert. Analyze the following biography of {name}, who is a {profession_label}, and rate their Big Five (OCEAN) personality traits on a scale of 0 to 100.

Consider their profession context when scoring — for example, athletes may score higher in conscientiousness, artists in openness, politicians in extraversion, etc.

Biography:
{biography_text[:4000]}

Return ONLY a valid JSON object with exactly these keys (no other text, no markdown, no backticks):
{{
  "openness": <number 0-100>,
  "conscientiousness": <number 0-100>,
  "extraversion": <number 0-100>,
  "agreeableness": <number 0-100>,
  "neuroticism": <number 0-100>,
  "reasoning": "<brief 2-3 sentence explanation>"
}}"""
        message = call_claude_messages_create(
            client,
            SONNET_MODEL_CANDIDATES,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        data = extract_and_parse_json(message.content[0].text)
        cleaned_scores, err = validate_big_five_scores(data)
        if err:
            return None, err, 'FAILED'
        cleaned_scores['reasoning'] = str(data.get('reasoning') or '').strip()
        return cleaned_scores, None, 'PASSED'
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"Failed to parse Claude response as valid JSON: {str(e)}", 'FAILED'
    except Exception as e:
        return None, format_claude_error(e), 'UNAVAILABLE'


def paraphrase_biography_via_gemini(name, biography_text):
    try:
        client = get_claude_client()
        prompt = f"""You are an academic writer. Rewrite the following biography of {name} completely in your own words.
Keep all important facts but paraphrase completely — do not copy sentences from the original.
Write in third person, academic style, 3-5 paragraphs.

Original biography:
{biography_text[:4000]}

Return ONLY the paraphrased biography text, no headers, no preamble, no backticks."""
        message = call_claude_messages_create(
            client,
            SONNET_MODEL_CANDIDATES,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip(), None, 'PASSED'
    except Exception as e:
        return biography_text, format_claude_error(e), 'UNAVAILABLE'


def validate_match_via_gemini(name, biography):
    """Use Claude to check if biography matches the given name.
    Returns (is_match: bool, reason: str, status: 'PASSED'|'FAILED'|'UNAVAILABLE').
    Never treats service errors or exceptions as successful matches."""
    try:
        client = get_claude_client()
        prompt = f"""You are a lenient fact-checker. Your job is to verify if a biography is about a specific person.

Name: {name}
Biography excerpt:
{biography[:3000]}

Rules:
- Return is_match = true if the biography is clearly about "{name}" OR if the content is consistent with what is known about "{name}" even if their name is not explicitly mentioned.
- Return is_match = false ONLY if the biography is clearly about a COMPLETELY DIFFERENT person (e.g. name of another person is mentioned prominently and the content is about them instead).
- When in doubt, return is_match = true.

Answer ONLY with a valid JSON object (no markdown, no backticks):
{{"is_match": true/false, "reason": "<one sentence explanation>"}}"""
        message = call_claude_messages_create(
            client,
            SONNET_MODEL_CANDIDATES,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        data = extract_and_parse_json(message.content[0].text)
        if 'is_match' not in data:
            return False, "Invalid response schema: missing 'is_match' property.", 'FAILED'
        is_match = parse_strict_boolean(data['is_match'], 'is_match')
        reason = str(data.get('reason') or '').strip()
        if is_match:
            return True, reason or "Biography matches subject.", 'PASSED'
        else:
            return False, reason or "Biography does not match subject.", 'FAILED'
    except (json.JSONDecodeError, ValueError) as e:
        return False, f"Invalid validation response: {str(e)}", 'FAILED'
    except Exception as e:
        return False, f"Validation service unavailable: {format_claude_error(e)}", 'UNAVAILABLE'


# ── Subjects & Profiles Data Management ──

def normalize_subject_name_for_comparison(name):
    """Normalize subject name for robust canonical matching across spaces, underscores, hyphens, and casing."""
    if not name or not isinstance(name, str):
        return ""
    # Lowercase, replace underscores and hyphens with spaces, collapse multiple spaces
    cleaned = re.sub(r'[\s_\-]+', ' ', name.strip().lower())
    return cleaned


def get_research_baseline_names():
    """Return set of lowercased subject names that constitute the protected research baseline dataset.
    Fail-closed: falls back across dataset.csv, face_embeddings.csv, personality_scores.csv, and in-memory cache."""
    global _BASELINE_NAMES_CACHE
    found_names = set()
    # 1. Try dataset.csv
    if os.path.exists(DATASET_FILE):
        try:
            df_base = pd.read_csv(DATASET_FILE)
            if 'name' in df_base.columns:
                found_names.update(str(n).strip().lower() for n in df_base['name'].dropna().tolist())
        except Exception:
            pass
    # 2. Try face_embeddings.csv
    if not found_names and os.path.exists(FACE_EMBEDDINGS_FILE):
        try:
            df_emb = pd.read_csv(FACE_EMBEDDINGS_FILE)
            if 'name' in df_emb.columns:
                found_names.update(str(n).strip().lower() for n in df_emb['name'].dropna().tolist())
        except Exception:
            pass
    # 3. Try personality_scores.csv
    if not found_names and os.path.exists(PERSONALITY_SCORES_FILE):
        try:
            df_ps = pd.read_csv(PERSONALITY_SCORES_FILE)
            if 'name' in df_ps.columns:
                found_names.update(str(n).strip().lower() for n in df_ps['name'].dropna().tolist())
        except Exception:
            pass
    # 4. If loaded, update cache
    if found_names:
        _BASELINE_NAMES_CACHE = set(found_names)
        return set(found_names)
    # 5. Fallback to cached baseline names if file reading failed
    if _BASELINE_NAMES_CACHE:
        return set(_BASELINE_NAMES_CACHE)
    return set()


def is_baseline_subject(name, profile_path=None):
    """Determine whether a subject belongs to the protected research baseline dataset,
    robust against all URL variations such as 'Albert Einstein', 'Albert_Einstein', 'Albert-Einstein', etc.
    Fail-closed: If the record cannot be verified as an authorized public assessment, or if
    datasets are inaccessible, it fails closed to protect research participant privacy."""
    try:
        baseline_names = get_research_baseline_names()
        normalized_baseline = {normalize_subject_name_for_comparison(b) for b in baseline_names}
        compact_baseline = {re.sub(r'[^a-z0-9]', '', b.lower()) for b in baseline_names}

        names_to_check = [name] if name else []
        if profile_path and os.path.exists(profile_path):
            stored_name = get_profile_subject_name(profile_path)
            if stored_name:
                names_to_check.append(stored_name)

        for n in names_to_check:
            if not n or not isinstance(n, str):
                continue
            clean = n.strip().lower()
            if clean in baseline_names:
                return True
            norm = normalize_subject_name_for_comparison(clean)
            if norm and norm in normalized_baseline:
                return True
            compact = re.sub(r'[^a-z0-9]', '', clean)
            if compact and compact in compact_baseline:
                return True

        # Check if confirmed as a public assessment
        if profile_path and os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    pdata = json.load(f)
                if pdata.get('is_public_assessment') is True or pdata.get('access_token'):
                    return False
            except Exception:
                pass

        if os.path.exists(PUBLIC_ASSESSMENTS_FILE):
            try:
                with public_assessment_transaction_lock():
                    df_pub = pd.read_csv(PUBLIC_ASSESSMENTS_FILE)
                clean_target = (name or '').strip().lower()
                pub_names = {str(x).strip().lower() for x in df_pub['name'].dropna().tolist()}
                if clean_target in pub_names:
                    return False
            except Exception:
                pass

        # If baseline_names could not be read (missing/corrupted dataset) and not confirmed public,
        # fail closed: treat as baseline to prevent unauthorized exposure.
        if not baseline_names:
            return True

        return False
    except Exception:
        # On any error, fail closed
        return True


def get_research_baseline_count():
    """Return count of protected baseline research subjects."""
    return len(get_research_baseline_names())


def get_subjects(authorized_researcher=False):
    """Return subjects available in the system.
    If authorized_researcher is True, returns all subjects including protected research baseline.
    If authorized_researcher is False, returns only demonstration assessments authorized for the current session,
    preserving participant privacy for other users' assessments and the 103 baseline research records."""
    subjects_dict = {}

    # 1. Load research baseline subjects (only if authorized_researcher is True)
    if authorized_researcher and os.path.exists(FACE_EMBEDDINGS_FILE):
        try:
            df = pd.read_csv(FACE_EMBEDDINGS_FILE)
            personality_names = set()
            if os.path.exists(PERSONALITY_SCORES_FILE):
                ps = pd.read_csv(PERSONALITY_SCORES_FILE)
                personality_names = {str(n).strip().lower() for n in ps['name'].dropna().tolist()}
            for n in df['name'].dropna().tolist():
                clean_n = str(n).strip()
                subjects_dict[clean_n.lower()] = {
                    'name': clean_n,
                    'has_personality': clean_n.lower() in personality_names,
                    'is_research': True
                }
        except Exception:
            pass

    # Determine allowed assessment names for non-researchers
    user_assessments = set()
    has_session_filter = False
    if not authorized_researcher:
        try:
            if has_request_context():
                user_assessments = {str(s).strip().lower() for s in session.get('authorized_public_assessments', [])}
                has_session_filter = True
        except Exception:
            pass

    # 2. Include public assessments
    if os.path.exists(PUBLIC_ASSESSMENTS_FILE):
        try:
            with public_assessment_transaction_lock():
                df_pub = pd.read_csv(PUBLIC_ASSESSMENTS_FILE)
            for n in df_pub['name'].dropna().tolist():
                clean_n = str(n).strip()
                # If session filter is active, only include if user created/unlocked this assessment
                if has_session_filter:
                    norm_c = normalize_subject_name_for_comparison(clean_n)
                    if clean_n.lower() not in user_assessments and norm_c not in user_assessments:
                        continue
                if clean_n.lower() not in subjects_dict:
                    subjects_dict[clean_n.lower()] = {
                        'name': clean_n,
                        'has_personality': True,
                        'is_research': False
                    }
        except Exception:
            pass

    # 3. Include any other profile JSONs in PROFILES_DIR (if non-baseline, or if authorized)
    if os.path.exists(PROFILES_DIR):
        try:
            for fname in os.listdir(PROFILES_DIR):
                if fname.lower().endswith('.json'):
                    prof_path = os.path.join(PROFILES_DIR, fname)
                    subj_name = get_profile_subject_name(prof_path)
                    subj_is_base = is_baseline_subject(subj_name, profile_path=prof_path)
                    if subj_name and (authorized_researcher or not subj_is_base):
                        if has_session_filter and not authorized_researcher:
                            norm_c = normalize_subject_name_for_comparison(subj_name)
                            if subj_name.lower() not in user_assessments and norm_c not in user_assessments:
                                continue
                        if subj_name.lower() not in subjects_dict:
                            subjects_dict[subj_name.lower()] = {
                                'name': subj_name,
                                'has_personality': True,
                                'is_research': subj_is_base
                            }
        except Exception:
            pass

    return list(subjects_dict.values())


def get_profile_subject_name(filepath):
    """Safely extract subject name from a profile JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return str(data.get('name', '')).strip()
    except Exception:
        return ''


def resolve_profile_path(name, create_if_missing=False, allow_overwrite=True):
    """Strictly resolve profile JSON path confined within PROFILES_DIR.
    Neutralizes directory traversal (including Windows backslashes, Unix slashes, '..' components, and null bytes).
    Guarantees zero profile filename collisions between different subject names (including Ana-Maria vs Ana_Maria).
    When allow_overwrite=False, preserves existing profile files by resolving to a non-colliding disambiguated path."""
    if not name or not isinstance(name, str):
        return None
    clean_name = str(name).strip().replace('\x00', '')
    if not clean_name:
        return None
    profiles_dir_abs = os.path.abspath(PROFILES_DIR)
    name_hash = hashlib.sha256(clean_name.encode('utf-8')).hexdigest()[:8]

    # 1. Check exact match with existing profile files (direct or disambiguated)
    candidates = [
        clean_name,
        clean_name.replace(' ', '_').replace('/', '_').replace('\\', '_'),
        clean_name.replace("'", '_').replace(' ', '_').replace('/', '_').replace('\\', '_'),
    ]
    for cand in candidates:
        safe_base = secure_filename(cand)
        if not safe_base:
            continue
        p = os.path.abspath(os.path.join(profiles_dir_abs, f"{safe_base}.json"))
        if os.path.commonpath([profiles_dir_abs, p]) == profiles_dir_abs and os.path.exists(p):
            stored_name = get_profile_subject_name(p)
            if stored_name and stored_name.lower() == clean_name.lower():
                if not create_if_missing or allow_overwrite:
                    return p
        p_hash = os.path.abspath(os.path.join(profiles_dir_abs, f"{safe_base}_{name_hash}.json"))
        if os.path.commonpath([profiles_dir_abs, p_hash]) == profiles_dir_abs and os.path.exists(p_hash):
            stored_name = get_profile_subject_name(p_hash)
            if stored_name and stored_name.lower() == clean_name.lower():
                if not create_if_missing or allow_overwrite:
                    return p_hash

    # 2. Check all files in PROFILES_DIR for exact subject name match
    if not create_if_missing or allow_overwrite:
        if os.path.exists(profiles_dir_abs):
            try:
                for fname in os.listdir(profiles_dir_abs):
                    if fname.lower().endswith('.json'):
                        p_scan = os.path.abspath(os.path.join(profiles_dir_abs, fname))
                        stored_name = get_profile_subject_name(p_scan)
                        if stored_name and stored_name.lower() == clean_name.lower():
                            return p_scan
            except Exception:
                pass

    # 3. For reading only: fallback for URL slug matching (e.g. /profile/Albert_Einstein for "Albert Einstein")
    if not create_if_missing and os.path.exists(profiles_dir_abs):
        try:
            norm_target = normalize_subject_name_for_comparison(clean_name)
            for fname in os.listdir(profiles_dir_abs):
                if fname.lower().endswith('.json'):
                    p_scan = os.path.abspath(os.path.join(profiles_dir_abs, fname))
                    stored_name = get_profile_subject_name(p_scan)
                    if stored_name:
                        if stored_name.replace(' ', '_').lower() == clean_name.lower() or \
                           normalize_subject_name_for_comparison(stored_name) == norm_target:
                            return p_scan
        except Exception:
            pass

    # 4. If resolving for save/creation:
    if create_if_missing:
        safe_base = secure_filename(clean_name.replace(' ', '_').replace('/', '_').replace('\\', '_'))
        if not safe_base:
            safe_base = f"subject_{name_hash}"

        canonical_path = os.path.abspath(os.path.join(profiles_dir_abs, f"{safe_base}.json"))
        if os.path.commonpath([profiles_dir_abs, canonical_path]) == profiles_dir_abs:
            if not os.path.exists(canonical_path):
                return canonical_path
            stored_name = get_profile_subject_name(canonical_path)
            if (not stored_name or stored_name.lower() == clean_name.lower()) and allow_overwrite:
                return canonical_path

        # If canonical_path exists and overwrite is False or belongs to a different subject:
        # Check collision-free hashed path
        hashed_path = os.path.abspath(os.path.join(profiles_dir_abs, f"{safe_base}_{name_hash}.json"))
        if os.path.commonpath([profiles_dir_abs, hashed_path]) == profiles_dir_abs:
            if not os.path.exists(hashed_path):
                return hashed_path
            stored_h = get_profile_subject_name(hashed_path)
            if (not stored_h or stored_h.lower() == clean_name.lower()) and allow_overwrite:
                return hashed_path

        # Generate numbered disambiguation to strictly avoid overwriting previous records
        counter = 2
        while counter < 1000:
            disambiguated_path = os.path.abspath(os.path.join(profiles_dir_abs, f"{safe_base}_{counter}.json"))
            if not os.path.exists(disambiguated_path):
                return disambiguated_path
            counter += 1

    return None


def save_profile(name, biography_scores, estimated_scores, reasoning, embedding_info, profession='other', image_filename=None, allow_overwrite=True, access_token=None):
    profile_path = resolve_profile_path(name, create_if_missing=True, allow_overwrite=allow_overwrite)
    if not profile_path:
        raise ValueError(f"Invalid profile name for storage: '{name}'")
    profession_label = next((p['label'] for p in PROFESSION_CATEGORIES if p['value'] == profession), 'Other')
    profile_data = {
        'name': name,
        'profession': profession,
        'profession_label': profession_label,
        'biography_scores': biography_scores,
        'estimated_scores': estimated_scores,
        'reasoning': reasoning,
        'embedding_info': embedding_info,
        'image_filename': image_filename,
        'is_public_assessment': bool(access_token or not is_baseline_subject(name, profile_path=profile_path))
    }
    if access_token:
        profile_data['access_token'] = access_token

    atomic_write_json(profile_path, profile_data)

    if image_filename:
        save_image_association(name, image_filename, status='verified', method='profile_save')

    return profile_path


def load_profile(name):
    profile_path = resolve_profile_path(name)
    if not profile_path or not os.path.exists(profile_path):
        return None
    try:
        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


# ── Machine Learning Training & Evaluation Pipeline ──

def build_and_train_model():
    """Build and train the machine learning pipeline.
    Uses StandardScaler, PCA, and RandomForestRegressor.
    
    Leakage-Free Evaluation:
    - Splits usable paired data into train and held-out test sets.
    - Preprocessing (StandardScaler and PCA) is fit strictly on training data during evaluation.
    - Evaluates predictions using held-out test data not used to fit the evaluation model or preprocessing.
    - Computes MAE, R², and Pearson correlation per trait on the held-out test set.
    - If a statistic is mathematically undefined, represents it as None in Python (null in JSON).
    - Preserves negative R² values when supported by evaluation.
    - Retains OOB R² on the training forest as a secondary diagnostic.
    
    Production Model:
    - Refits the final prediction pipeline using all usable paired records for future inference.
    """
    if not os.path.exists(FACE_EMBEDDINGS_FILE):
        return False, "face_embeddings.csv not found.", None
    if not os.path.exists(PERSONALITY_SCORES_FILE):
        return False, "personality_scores.csv not found.", None

    try:
        df_emb = pd.read_csv(FACE_EMBEDDINGS_FILE)
        df_pers = pd.read_csv(PERSONALITY_SCORES_FILE)
    except Exception as e:
        return False, f"Failed to read CSV files: {str(e)}", None

    # Deduplicate and merge by subject name
    df_emb = df_emb.drop_duplicates(subset=['name'], keep='last')
    df_pers = df_pers.drop_duplicates(subset=['name'], keep='last')
    df = pd.merge(df_emb, df_pers, on='name', how='inner')

    # Validate feature and trait columns
    feature_cols = [f'face_{i}' for i in range(1, 513)]
    missing_feats = [c for c in feature_cols if c not in df.columns]
    missing_traits = [t for t in TRAIT_NAMES if t not in df.columns]

    if missing_feats or missing_traits:
        return False, f"Missing required columns in dataset (features: {len(missing_feats)}, traits: {len(missing_traits)}).", None

    # Filter out incomplete rows or non-finite values
    cols_to_check = ['name'] + feature_cols + TRAIT_NAMES
    df = df.dropna(subset=cols_to_check)
    valid_mask = np.isfinite(df[feature_cols].values.astype(float)).all(axis=1) & \
                 np.isfinite(df[TRAIT_NAMES].values.astype(float)).all(axis=1)
    df = df[valid_mask]

    usable_count = len(df)
    if usable_count < 5:
        return False, f"Not enough valid paired subjects ({usable_count} matched). Need at least 5.", None

    # Save clean dataset
    df.to_csv(DATASET_FILE, index=False)

    X = df[feature_cols].values.astype(float)
    y = df[TRAIT_NAMES].values.astype(float)
    names = df['name'].values

    # Determine train/test split (80/20 train/test split, ensuring at least 1 test sample and at least 3 train samples)
    test_count = max(1, int(usable_count * 0.2))
    if usable_count - test_count < 3:
        test_count = max(1, usable_count - 3)

    test_fraction = test_count / usable_count
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, test_size=test_fraction, random_state=42
    )

    # Dynamically determine PCA components from training set size
    pca_comps = max(1, min(len(X_train) - 1, 10))

    # ── 1. EVALUATION PIPELINE (Leakage-Free Held-out Test) ──
    eval_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_comps, random_state=42)),
        ('rf', RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=2,
            max_features=0.3, oob_score=True, random_state=42, n_jobs=-1
        ))
    ])

    eval_pipeline.fit(X_train, y_train)
    y_test_pred = eval_pipeline.predict(X_test)
    eval_rf = eval_pipeline.named_steps['rf']
    oob_r2 = round(float(eval_rf.oob_score_), 4) if hasattr(eval_rf, 'oob_score_') and eval_rf.oob_score_ is not None else None

    # Calculate per-trait metrics on held-out test data
    trait_metrics = {}
    r2_scores = []
    for i, trait in enumerate(TRAIT_NAMES):
        actual = y_test[:, i]
        predicted = y_test_pred[:, i]
        mae = round(float(np.mean(np.abs(actual - predicted))), 2)

        # R² on held-out test data (represented as None when mathematically undefined)
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        if ss_tot > 1e-9:
            r2 = round(float(1.0 - (ss_res / ss_tot)), 4)
            r2_scores.append(r2)
        else:
            r2 = None  # Undefined: zero variance in reference values

        # Pearson correlation between model predictions and reference scores
        if len(actual) >= 2 and np.std(actual) > 1e-9 and np.std(predicted) > 1e-9:
            r_val, p_val = pearsonr(actual, predicted)
            pearson_r = round(float(r_val), 4)
            p_val = round(float(p_val), 4)
        else:
            pearson_r = None  # Undefined
            p_val = None

        trait_metrics[trait] = {
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_r,
            'p_value': p_val
        }

    # Held-out predictions for visualization and diagnostics
    predictions_data = []
    for i, name in enumerate(names_test):
        row = {'name': name}
        for j, trait in enumerate(TRAIT_NAMES):
            row[f'{trait}_actual'] = round(float(y_test[i, j]), 1)
            row[f'{trait}_predicted'] = round(float(y_test_pred[i, j]), 1)
        predictions_data.append(row)

    # ── 2. PRODUCTION MODEL PIPELINE (Fit on All Eligible Data for Future Inference) ──
    prod_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_comps, random_state=42)),
        ('rf', RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=2,
            max_features=0.3, oob_score=True, random_state=42, n_jobs=-1
        ))
    ])
    prod_pipeline.fit(X, y)

    pca_model = prod_pipeline.named_steps['pca']
    explained_variance = [round(float(v) * 100, 2) for v in pca_model.explained_variance_ratio_[:10]]
    rf_model = prod_pipeline.named_steps['rf']
    importances = [round(float(v), 6) for v in rf_model.feature_importances_.tolist()[:pca_comps]]

    test_r2 = round(float(np.mean(r2_scores)), 4) if r2_scores else None

    metrics = {
        'n_subjects': usable_count,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'evaluation_method': 'Held-out test split (80/20 train/test without preprocessing leakage)',
        'pca_components': pca_comps,
        'oob_r2': oob_r2,
        'test_r2': test_r2,
        'trait_metrics': trait_metrics,
        'explained_variance': explained_variance,
        'feature_importances': importances,
        'predictions': predictions_data,
        'training_subjects': names.tolist(),
        'is_outdated': False,
        'outdated_reason': '',
        'reproducibility': {
            'random_state': 42,
            'model_type': 'RandomForestRegressor',
            'n_estimators': 200,
            'max_depth': 5,
            'min_samples_leaf': 2,
            'max_features': 0.3,
            'n_records': usable_count,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(prod_pipeline, TRAINED_MODEL_FILE)
    atomic_write_json(MODEL_METRICS_FILE, metrics)

    return True, f"Model trained on {usable_count} subjects ({len(X_train)} train / {len(X_test)} held-out test).", metrics


def estimate_personality(embedding):
    """Estimate Big Five scores from a 512-dimensional facial embedding using the trained model."""
    if not os.path.exists(TRAINED_MODEL_FILE):
        return None, "No trained model found."
    try:
        pipeline = joblib.load(TRAINED_MODEL_FILE)
        X = np.array(embedding).reshape(1, -1)
        y_pred = pipeline.predict(X)[0]
        return {trait: round(float(y_pred[i]), 1) for i, trait in enumerate(TRAIT_NAMES)}, None
    except Exception as e:
        return None, str(e)


# ── Flask Application Routes ──

# ── Researcher (admin) area helpers ──

def is_researcher():
    """True when this session has unlocked researcher access with the current RESEARCH_ADMIN_KEY."""
    ok, _ = verify_admin_authorization()
    return bool(ok)


def safe_next_path(target, default=None):
    """Only allow same-site relative paths as post-login redirect targets (prevents open redirects)."""
    if isinstance(target, str) and target.startswith('/') and not target.startswith('//') and '\\' not in target:
        return target
    return default


def researcher_login_redirect():
    return redirect(url_for('researcher_login', next=request.path))


@app.context_processor
def inject_researcher_flag():
    try:
        return {'researcher_nav': is_researcher()}
    except Exception:
        return {'researcher_nav': False}


def load_private_participants():
    path = get_private_participants_file()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def delete_private_participant(participant_id):
    """Delete one private participant's record, photo, and image association.
    Returns (deleted: bool, message: str)."""
    pid = str(participant_id or '').strip().upper()
    path = get_private_participants_file()
    if not pid or not os.path.exists(path) or os.path.getsize(path) == 0:
        return False, f"Participant {pid or '(blank)'} not found."
    with public_assessment_transaction_lock(timeout=10.0):
        df = pd.read_csv(path)
        mask = df['participant_id'].astype(str).str.upper() == pid
        if not mask.any():
            return False, f"Participant {pid} not found."
        images = [str(f) for f in df.loc[mask, 'image_filename'].dropna().tolist()]
        atomic_write_csv(path, df[~mask])
    remove_image_association(pid)
    for fname in images:
        safe_path, err = validate_and_resolve_upload_path(fname)
        if safe_path and os.path.exists(safe_path) and not get_subjects_referencing_image(fname):
            try:
                os.remove(safe_path)
            except Exception:
                pass
    return True, f"Participant {pid} and their photo have been deleted."


@app.route('/researcher', methods=['GET'])
def researcher_login():
    """Researcher login page (admin key). Already-unlocked researchers go straight to the dashboard."""
    next_path = safe_next_path(request.args.get('next'), url_for('index'))
    if is_researcher():
        return redirect(next_path)
    return render_template('researcher_login.html', next_path=next_path,
                           messages=get_flashed_messages(with_categories=True))


@app.route('/researcher/participants')
def researcher_participants():
    if not is_researcher():
        return researcher_login_redirect()
    df = load_private_participants()
    rows = []
    if not df.empty:
        df = df.sort_values('consent_timestamp', ascending=False) if 'consent_timestamp' in df.columns else df
        for _, r in df.iterrows():
            def num(col):
                v = r.get(col)
                return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v))
            bg = r.get('background')
            rows.append({
                'participant_id': r.get('participant_id'),
                'date': str(r.get('consent_timestamp') or '')[:10],
                'age_range': r.get('age_range'),
                'background': '' if bg is None or (isinstance(bg, float) and np.isnan(bg)) else str(bg),
                'scores': {t: num(t) for t in TRAIT_NAMES},
                'pred': {t: num(f'pred_{t}') for t in TRAIT_NAMES},
                'flags': [] if r.get('quality_flags') is None or (isinstance(r.get('quality_flags'), float) and np.isnan(r.get('quality_flags')))
                         else [f for f in str(r.get('quality_flags')).split('; ') if f],
            })
    counts = {
        'baseline': get_research_baseline_count(),
        'public': len(pd.read_csv(PUBLIC_ASSESSMENTS_FILE)) if os.path.exists(PUBLIC_ASSESSMENTS_FILE) and os.path.getsize(PUBLIC_ASSESSMENTS_FILE) > 0 else 0,
        'private': len(rows),
    }
    return render_template('participants.html', rows=rows, counts=counts, trait_names=TRAIT_NAMES,
                           ocean_letters=OCEAN_LETTERS, messages=get_flashed_messages(with_categories=True))



# ── Per-subject computation breakdown (researcher-only) ──

TRAIT_BAND_HIGH = 65
TRAIT_BAND_LOW = 35
PERSONALITY_TYPE_NAMES = {
    'openness':          {'high': ('The Creative Explorer', 'curious'),    'low': ('The Practical Realist', 'practical')},
    'conscientiousness': {'high': ('The Reliable Organizer', 'organized'), 'low': ('The Free Spirit', 'spontaneous')},
    'extraversion':      {'high': ('The Social Energizer', 'outgoing'),    'low': ('The Quiet Thinker', 'reserved')},
    'agreeableness':     {'high': ('The Caring Harmonizer', 'caring'),     'low': ('The Independent Challenger', 'independent')},
    'neuroticism':       {'high': ('The Sensitive Feeler', 'sensitive'),   'low': ('The Calm Anchor', 'calm')},
}


def score_band(v):
    if v is None:
        return None
    return 'high' if v >= TRAIT_BAND_HIGH else ('low' if v <= TRAIT_BAND_LOW else 'average')


def explain_personality_type(scores):
    """Same rule as the results page: the strongest trait (furthest from 50, outside the average band)."""
    rows = []
    for t in TRAIT_NAMES:
        v = scores.get(t)
        if v is None:
            continue
        rows.append({'trait': t, 'score': round(float(v), 1), 'distance': round(abs(float(v) - 50), 1),
                     'band': score_band(float(v))})
    eligible = sorted([r for r in rows if r['band'] != 'average'], key=lambda r: -r['distance'])
    if not eligible:
        return {'rows': rows, 'title': 'The Balanced All-Rounder', 'summary': 'balanced across all five traits', 'winner': None}
    pick = lambda r: PERSONALITY_TYPE_NAMES[r['trait']][r['band']]
    return {'rows': rows, 'title': pick(eligible[0])[0], 'winner': eligible[0]['trait'],
            'summary': 'mostly ' + ' and '.join(pick(r)[1] for r in eligible[:2])}


def explain_bfi10(answers):
    """Step-by-step BFI-10 scoring for display."""
    items = []
    for i, ((text, trait, reverse), a) in enumerate(zip(BFI10_ITEMS, answers), start=1):
        items.append({'num': i, 'text': text, 'trait': trait, 'answer': a, 'reverse': reverse,
                      'scored': (6 - a) if reverse else a})
    traits = []
    for t in TRAIT_NAMES:
        its = [it for it in items if it['trait'] == t]
        mean = sum(it['scored'] for it in its) / len(its)
        traits.append({'trait': t, 'items': its, 'mean': round(mean, 2), 'score': round((mean - 1) / 4 * 100, 1)})
    return {'items': items, 'traits': traits}


def explain_face_prediction(embedding):
    """Recompute the face-based prediction step by step with the trained pipeline:
    HOG vector -> StandardScaler -> PCA (10 components) -> 200 Random Forest trees -> average."""
    if not os.path.exists(TRAINED_MODEL_FILE):
        return None, "No trained model found."
    try:
        pipeline = joblib.load(TRAINED_MODEL_FILE)
        scaler = pipeline.named_steps['scaler']
        pca = pipeline.named_steps['pca']
        rf = pipeline.named_steps['rf']
        x = np.array(embedding, dtype=float).reshape(1, -1)
        x_scaled = scaler.transform(x)
        x_pca = pca.transform(x_scaled)
        tree_preds = np.array([tree.predict(x_pca)[0] for tree in rf.estimators_])  # (n_trees, 5)
        final = tree_preds.mean(axis=0)
        sample_idx = list(range(5))
        result = {
            'hog': {
                'length': int(x.shape[1]),
                'norm': round(float(np.linalg.norm(x)), 4),
                'min': round(float(x.min()), 4), 'max': round(float(x.max()), 4),
                'mean': round(float(x.mean()), 4),
                'first_values': [round(float(v), 4) for v in x[0, :10]],
            },
            'scaling': [{'feature': f'face_{i + 1}', 'raw': round(float(x[0, i]), 4),
                         'mean': round(float(scaler.mean_[i]), 4), 'std': round(float(scaler.scale_[i]), 4),
                         'scaled': round(float(x_scaled[0, i]), 3)} for i in sample_idx],
            'pca': [{'component': f'PC{i + 1}', 'value': round(float(v), 3),
                     'variance': round(float(pca.explained_variance_ratio_[i]) * 100, 2)}
                    for i, v in enumerate(x_pca[0])],
            'pca_total_variance': round(float(pca.explained_variance_ratio_.sum()) * 100, 2),
            'n_trees': len(rf.estimators_),
            'max_depth': rf.max_depth,
            'trees': [{'trait': t,
                       'min': round(float(tree_preds[:, j].min()), 1),
                       'median': round(float(np.median(tree_preds[:, j])), 1),
                       'max': round(float(tree_preds[:, j].max()), 1),
                       'std': round(float(tree_preds[:, j].std()), 1),
                       'final': round(float(final[j]), 1),
                       'examples': [round(float(v), 1) for v in tree_preds[:3, j]]}
                      for j, t in enumerate(TRAIT_NAMES)],
            'final': {t: round(float(final[j]), 1) for j, t in enumerate(TRAIT_NAMES)},
        }
        return result, None
    except Exception as e:
        return None, str(e)


def alignment_label(diff):
    if diff <= 10:
        return 'High Agreement'
    if diff <= 20:
        return 'Moderate Agreement'
    if diff <= 35:
        return 'Low Agreement'
    return 'Divergent'


@app.route('/researcher/computation')
def researcher_computation():
    """Researcher-only, per-subject explanation of how the OCEAN results were computed."""
    if not is_researcher():
        return researcher_login_redirect()
    kind = request.args.get('type', 'subject')
    ident = str(request.args.get('id', '')).strip()
    if not ident:
        return redirect(url_for('subjects'))

    ctx = {'kind': kind, 'ident': ident, 'trait_names': TRAIT_NAMES, 'trait_labels': TRAIT_LABELS,
           'ocean_letters': OCEAN_LETTERS, 'band_high': TRAIT_BAND_HIGH, 'band_low': TRAIT_BAND_LOW,
           'error': None, 'bfi': None, 'claude': None, 'face': None, 'face_error': None, 'meta': {}}
    reference, embedding, stored_pred = None, None, None

    if kind == 'private':
        df = load_private_participants()
        match = df[df['participant_id'].astype(str).str.upper() == ident.upper()] if not df.empty else df
        if match.empty:
            ctx['error'] = f"Participant {ident} not found."
            return render_template('computation.html', **ctx)
        row = match.iloc[0]
        try:
            answers = [int(a) for a in str(row.get('bfi10_answers', '')).split()]
        except ValueError:
            answers = []
        if len(answers) == len(BFI10_ITEMS):
            ctx['bfi'] = explain_bfi10(answers)
        reference = {t: float(row[t]) for t in TRAIT_NAMES}
        face_cols = [f'face_{i}' for i in range(1, 513)]
        if all(c in row.index for c in face_cols):
            embedding = [float(row[c]) for c in face_cols]
        stored_pred = {t: (None if pd.isna(row.get(f'pred_{t}')) else float(row.get(f'pred_{t}'))) for t in TRAIT_NAMES}
        ctx['meta'] = {'title': ident, 'subtitle': 'Private participant · scores from BFI-10 questionnaire',
                       'age_range': row.get('age_range'), 'date': str(row.get('consent_timestamp') or '')[:10]}
    else:
        profile = load_profile(ident)
        if not profile:
            ctx['error'] = f"Subject {ident} not found."
            return render_template('computation.html', **ctx)
        reference = {t: float(profile['biography_scores'][t]) for t in TRAIT_NAMES
                     if t in (profile.get('biography_scores') or {})}
        is_base = is_baseline_subject(ident)
        ctx['claude'] = {'reasoning': profile.get('reasoning') or '',
                         'profession': profile.get('profession_label') or profile.get('profession') or ''}
        ctx['meta'] = {'title': profile.get('name') or ident,
                       'subtitle': ('Research baseline subject' if is_base else 'Public assessment') +
                                   ' · scores from biography (Claude AI)'}
        # Embedding: baseline CSV first, otherwise re-extract from the saved photo
        if os.path.exists(FACE_EMBEDDINGS_FILE):
            fe = pd.read_csv(FACE_EMBEDDINGS_FILE)
            m = fe[fe['name'] == ident]
            if not m.empty:
                embedding = m.iloc[0][[f'face_{i}' for i in range(1, 513)]].astype(float).tolist()
        if embedding is None:
            image_name = profile.get('image_filename')
            if not image_name and os.path.exists(PUBLIC_ASSESSMENTS_FILE):
                pa = pd.read_csv(PUBLIC_ASSESSMENTS_FILE)
                m = pa[pa['name'] == ident]
                if not m.empty and not pd.isna(m.iloc[0].get('image_filename')):
                    image_name = str(m.iloc[0]['image_filename'])
            if image_name:
                img_path, err = validate_and_resolve_upload_path(image_name)
                if img_path:
                    embedding, _, _ = extract_face_embedding(img_path)
        est = profile.get('estimated_scores') or {}
        stored_pred = {t: est.get(t) for t in TRAIT_NAMES} if est else None

    if embedding is not None:
        ctx['face'], ctx['face_error'] = explain_face_prediction(embedding)
    else:
        ctx['face_error'] = "No facial features available for this subject (photo not found)."

    ctx['reference'] = reference
    ctx['ptype'] = explain_personality_type(reference) if reference else None
    comparison = []
    if reference and ctx['face']:
        for t in TRAIT_NAMES:
            if t in reference:
                diff = abs(reference[t] - ctx['face']['final'][t])
                comparison.append({'trait': t, 'reference': round(reference[t], 1), 'predicted': ctx['face']['final'][t],
                                   'diff': round(diff, 1), 'label': alignment_label(diff)})
    ctx['comparison'] = comparison
    ctx['stored_pred'] = stored_pred
    metrics = None
    if os.path.exists(MODEL_METRICS_FILE):
        try:
            with open(MODEL_METRICS_FILE, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
    ctx['trait_metrics'] = (metrics or {}).get('trait_metrics') or {}
    return render_template('computation.html', **ctx)

@app.route('/researcher/participants/delete', methods=['POST'])
def researcher_delete_participant():
    if not is_researcher():
        return researcher_login_redirect()
    ok, msg = delete_private_participant(request.form.get('participant_id'))
    flash(msg, 'success' if ok else 'error')
    return redirect(url_for('researcher_participants'))


EXPORTABLE_DATASETS = {
    'baseline': ('dataset.csv', lambda: DATASET_FILE),
    'public': ('public_assessments.csv', lambda: PUBLIC_ASSESSMENTS_FILE),
    'private': ('private_participants.csv', lambda: get_private_participants_file()),
}


@app.route('/researcher/export/<dataset>')
def researcher_export(dataset):
    if not is_researcher():
        return researcher_login_redirect()
    if dataset not in EXPORTABLE_DATASETS:
        return jsonify({'success': False, 'error': 'Unknown dataset.'}), 404
    download_name, path_fn = EXPORTABLE_DATASETS[dataset]
    path = path_fn()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        flash(f'No data yet for {download_name}.', 'error')
        return redirect(url_for('researcher_participants'))
    stamp = datetime.now().strftime('%Y%m%d')
    return send_file(path, mimetype='text/csv', as_attachment=True,
                     download_name=download_name.replace('.csv', f'_{stamp}.csv'))


@app.route('/')
def home():
    """Everyone (public and researchers) lands on the simple Analyze page first."""
    return redirect(url_for('analyze'))


@app.route('/dashboard')
def index():
    # Researcher dashboard; public visitors are sent to the Analyze page
    if not is_researcher():
        return redirect(url_for('analyze'))
    is_auth, _ = verify_admin_authorization()
    is_authorized = is_auth or session.get('research_admin_authorized', False)
    subjects = get_subjects(authorized_researcher=is_authorized)
    model_exists = os.path.exists(TRAINED_MODEL_FILE)
    dataset_size = len(pd.read_csv(DATASET_FILE)) if os.path.exists(DATASET_FILE) else 0
    baseline_count = get_research_baseline_count()
    public_count = 0
    if os.path.exists(PUBLIC_ASSESSMENTS_FILE):
        try:
            df_pub = pd.read_csv(PUBLIC_ASSESSMENTS_FILE)
            public_count = len(df_pub)
        except Exception:
            pass
    model_is_outdated, outdated_reason = get_model_outdated_status()
    private_count = len(load_private_participants())
    return render_template('index.html', subjects=subjects, model_exists=model_exists, private_count=private_count,
                           dataset_size=dataset_size, subject_count=len(subjects),
                           baseline_count=baseline_count, public_count=public_count,
                           is_authorized=is_authorized,
                           model_is_outdated=model_is_outdated, outdated_reason=outdated_reason)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        model_is_outdated, outdated_reason = get_model_outdated_status()
        return render_template('analyze.html',
                               model_exists=os.path.exists(TRAINED_MODEL_FILE),
                               profession_categories=PROFESSION_CATEGORIES,
                               model_is_outdated=model_is_outdated,
                               outdated_reason=outdated_reason,
                               min_biography_words=MIN_BIOGRAPHY_WORDS,
                               bfi10_items=[item[0] for item in BFI10_ITEMS],
                               bfi10_scale=BFI10_SCALE,
                               private_age_ranges=PRIVATE_AGE_RANGES,
                               private_background_max=PRIVATE_BACKGROUND_MAX_CHARS)

    rl_err = check_ai_rate_limit('analyze', limit_per_minute=5)
    if rl_err:
        return rl_err

    name = request.form.get('name', '').strip()
    biography = request.form.get('biography', '').strip()
    profession = request.form.get('profession', 'other').strip()
    custom_profession = request.form.get('custom_profession', '').strip()

    if profession == 'other' and custom_profession:
        effective_profession_label = custom_profession
    else:
        effective_profession_label = next((p['label'] for p in PROFESSION_CATEGORIES if p['value'] == profession), 'Other')

    photo = request.files.get('photo')
    image_url_path = (request.form.get('image_url_path') or request.form.get('upload_id') or '').strip()

    if not name or not biography:
        return jsonify({'success': False, 'error': 'Name and biography are required.'}), 400

    # Minimum biography words validation (Central configuration)
    is_valid_bio, bio_words, bio_err = validate_biography_text(biography)
    if not is_valid_bio:
        return jsonify({
            'success': False,
            'error': bio_err,
            'word_count': bio_words,
            'min_words': MIN_BIOGRAPHY_WORDS
        }), 400

    if len(name) > 100 or '\x00' in name:
        return jsonify({'success': False, 'error': 'Subject name must not exceed 100 characters and cannot contain control characters.'}), 400

    if len(biography) > 50000:
        return jsonify({'success': False, 'error': 'Biography exceeds maximum allowed length (50,000 characters).'}), 400

    if custom_profession and len(custom_profession) > 100:
        return jsonify({'success': False, 'error': 'Custom profession exceeds maximum allowed length (100 characters).'}), 400

    if not resolve_profile_path(name, create_if_missing=True):
        return jsonify({'success': False, 'error': 'Invalid subject name. Please use standard characters.'}), 400

    if not profession or profession not in [p['value'] for p in PROFESSION_CATEGORIES]:
        return jsonify({'success': False, 'error': 'Please select a profession category.'}), 400

    # Protect core research baseline dataset from public overwrite (robust across all name variations)
    if is_baseline_subject(name):
        return jsonify({
            'success': False,
            'error': f'"{name}" is a protected research baseline subject. Please use a unique subject name for new assessments.',
            'duplicate': True,
            'protected_baseline': True
        }), 409

    # Duplicate detection for existing subjects (both research baseline and prior public assessments)
    existing_subject_names = set()
    if os.path.exists(FACE_EMBEDDINGS_FILE):
        try:
            existing_df = pd.read_csv(FACE_EMBEDDINGS_FILE)
            existing_subject_names.update(str(n).strip() for n in existing_df['name'].dropna().tolist())
        except Exception:
            pass
    if os.path.exists(PUBLIC_ASSESSMENTS_FILE):
        try:
            pub_df = pd.read_csv(PUBLIC_ASSESSMENTS_FILE)
            existing_subject_names.update(str(n).strip() for n in pub_df['name'].dropna().tolist())
        except Exception:
            pass
    if os.path.exists(PROFILES_DIR):
        try:
            for fname in os.listdir(PROFILES_DIR):
                if fname.lower().endswith('.json'):
                    p_name = get_profile_subject_name(os.path.join(PROFILES_DIR, fname))
                    if p_name:
                        existing_subject_names.add(p_name)
        except Exception:
            pass

    norm_input = normalize_subject_name_for_comparison(name)
    for existing in existing_subject_names:
        if name.strip().lower() == existing.strip().lower() or (norm_input and norm_input == normalize_subject_name_for_comparison(existing)):
            return jsonify({
                'success': False,
                'error': f'An assessment for "{existing}" already exists in the system. Please use a unique subject name (e.g. "{name} (2)") to prevent overwriting previous records.',
                'duplicate': True
            }), 409

    # Process image upload or URL identifier
    ephemeral_file_path = None
    saved_filename = None

    if photo and photo.filename:
        if not allowed_image(photo.filename):
            return jsonify({'success': False, 'error': 'Invalid image format.'}), 400
        ext = photo.filename.rsplit('.', 1)[1].lower()
        saved_filename = f"{uuid.uuid4().hex}.{ext}"
        photo_path = os.path.join(UPLOAD_DIR, saved_filename)
        photo.save(photo_path)
        ephemeral_file_path = photo_path
    elif image_url_path:
        # Validate through strict path confinement helper (neutralizes path traversal)
        safe_path, err = validate_and_resolve_upload_path(image_url_path)
        if err:
            return jsonify({'success': False, 'error': f'Invalid image reference: {err}'}), 400
        photo_path = safe_path
        saved_filename = os.path.basename(safe_path)
        # Distinguish between newly created ephemeral file vs existing saved/associated image
        if is_image_associated_or_saved(saved_filename):
            ephemeral_file_path = None
        else:
            ephemeral_file_path = photo_path
    else:
        return jsonify({'success': False, 'error': 'Please upload a photo or load one from a URL.'}), 400

    # Content moderation — context-aware based on profession
    mod_res = moderate_image(photo_path, profession)
    if len(mod_res) == 3:
        is_safe, mod_reason, mod_status = mod_res
    else:
        is_safe, mod_reason = mod_res
        mod_status = 'PASSED' if is_safe else 'FAILED'

    if mod_status == 'UNAVAILABLE':
        cleanup_ephemeral_file(ephemeral_file_path)
        status_code = 401 if ('401' in mod_reason or 'Authentication' in mod_reason) else (429 if '429' in mod_reason else 503)
        return jsonify({
            'success': False,
            'status': 'UNAVAILABLE',
            'error': f'Content moderation unavailable: {mod_reason}'
        }), status_code

    if not is_safe:
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({
            'success': False,
            'status': 'FAILED',
            'error': f'Inappropriate image detected: {mod_reason} Please upload a proper photo.',
            'moderation_flag': True
        }), 400

    # AI-generated image screening (server-side enforcement)
    ai_res = check_ai_generated_image(photo_path)
    if ai_res['is_ai']:
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({
            'success': False,
            'status': 'FAILED',
            'error': f"AI-generated image detected: {ai_res['reason']} Please use a real photograph of the subject.",
            'ai_generated': True,
            'detection_method': ai_res['method'],
            'confidence': ai_res['confidence']
        }), 400

    # Facial feature extraction
    embedding, embedding_info, err = extract_face_embedding(photo_path)
    if err:
        if err.startswith("No face detected"):
            category, no_face_msg = explain_missing_face(photo_path)
            cleanup_ephemeral_file(ephemeral_file_path)
            return jsonify({'success': False, 'error': no_face_msg, 'no_face': True,
                            'no_face_category': category}), 400
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({'success': False, 'error': err}), 400

    face_ok, face_category, face_msg = check_face_quality(photo_path)
    if not face_ok:
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({'success': False, 'error': face_msg, 'no_face': True, 'no_face_category': face_category}), 400

    # ── Server-Side Biography Validation Enforcement (Issue 4) ──
    val_token = get_biography_validation_token(name, biography)
    if not session.get(val_token):
        val_res = validate_match_via_gemini(name, biography)
        if len(val_res) == 3:
            is_match, val_reason, val_status = val_res
        else:
            is_match, val_reason = val_res
            val_status = 'PASSED' if is_match else 'FAILED'

        if val_status == 'UNAVAILABLE':
            cleanup_ephemeral_file(ephemeral_file_path)
            status_code = 401 if ('401' in val_reason or 'Authentication' in val_reason) else (429 if '429' in val_reason else 503)
            return jsonify({
                'success': False,
                'status': 'UNAVAILABLE',
                'error': f'Biography validation unavailable: {val_reason}'
            }), status_code

        if not is_match or val_status == 'FAILED':
            cleanup_ephemeral_file(ephemeral_file_path)
            return jsonify({
                'success': False,
                'status': 'FAILED',
                'mismatch': True,
                'error': f'Biography does not match "{name}": {val_reason}'
            }), 400

        session[val_token] = True

    # Personality scoring via Claude AI
    score_res = score_personality_via_gemini(name, biography, profession, effective_profession_label)
    if len(score_res) == 3:
        scores, err, score_status = score_res
    else:
        scores, err = score_res
        score_status = 'PASSED' if scores else 'FAILED'

    if err or not scores or score_status != 'PASSED':
        cleanup_ephemeral_file(ephemeral_file_path)
        status_code = 401 if ('401' in str(err) or 'Authentication' in str(err)) else (429 if '429' in str(err) else (503 if score_status == 'UNAVAILABLE' else 400))
        return jsonify({
            'success': False,
            'status': score_status if score_status in ('FAILED', 'UNAVAILABLE') else 'UNAVAILABLE',
            'error': f'Personality scoring failed: {err}'
        }), status_code

    reasoning = scores.pop('reasoning', '')
    predicted, _ = estimate_personality(embedding)
    model_is_outdated, outdated_reason = get_model_outdated_status()

    # Verify vector and score integrity before committing to storage
    if not isinstance(embedding, list) or len(embedding) != 512 or not all(np.isfinite(x) for x in embedding):
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({'success': False, 'error': 'Invalid facial embedding generated.'}), 500

    for trait in TRAIT_NAMES:
        if trait not in scores or not isinstance(scores[trait], (int, float)) or not (0.0 <= float(scores[trait]) <= 100.0):
            cleanup_ephemeral_file(ephemeral_file_path)
            return jsonify({'success': False, 'error': f'Invalid trait score for {trait}.'}), 500

    # Save public assessment record separately from research baseline dataset
    assessment_record = {
        'name': name,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'profession': profession,
        'profession_label': effective_profession_label,
        'image_filename': saved_filename or '',
        'openness': scores.get('openness'),
        'conscientiousness': scores.get('conscientiousness'),
        'extraversion': scores.get('extraversion'),
        'agreeableness': scores.get('agreeableness'),
        'neuroticism': scores.get('neuroticism'),
        'has_prediction': predicted is not None,
    }
    if predicted:
        for t in TRAIT_NAMES:
            assessment_record[f'pred_{t}'] = predicted.get(t)

    df_new_rec = pd.DataFrame([assessment_record])
    with public_assessment_transaction_lock(timeout=10.0):
        if os.path.exists(PUBLIC_ASSESSMENTS_FILE) and os.path.getsize(PUBLIC_ASSESSMENTS_FILE) > 0:
            try:
                df_pub = pd.read_csv(PUBLIC_ASSESSMENTS_FILE)
                # Retain all prior assessment records to prevent history overwrites
                df_combined = pd.concat([df_pub, df_new_rec], ignore_index=True)
                atomic_write_csv(PUBLIC_ASSESSMENTS_FILE, df_combined)
            except Exception as e:
                # Storage error: NEVER replace or overwrite existing file!
                # Fail safely, preserve original records untouched, and report error clearly.
                cleanup_ephemeral_file(ephemeral_file_path)
                return jsonify({
                    'success': False,
                    'status': 'STORAGE_ERROR',
                    'error': f"Storage integrity error: Unable to safely merge with existing assessment records ({str(e)}). Original records have been preserved untouched. Please contact an administrator."
                }), 500
        else:
            atomic_write_csv(PUBLIC_ASSESSMENTS_FILE, df_new_rec)

    # Generate unique access token for assessment creator (enables login-free ownership)
    access_token = secrets.token_urlsafe(16)

    # Save profile JSON with image association metadata and access token (allow_overwrite=False prevents profile overwrites)
    save_profile(name, scores, predicted, reasoning, embedding_info, profession,
                 image_filename=saved_filename, allow_overwrite=False, access_token=access_token)

    # Store creator ownership in user session (login-free session authorization)
    user_assessments = session.get('authorized_public_assessments', [])
    norm_name = normalize_subject_name_for_comparison(name)
    if norm_name not in user_assessments:
        user_assessments.append(norm_name)
    if name.lower() not in user_assessments:
        user_assessments.append(name.lower())
    session['authorized_public_assessments'] = user_assessments

    token_map = session.get('assessment_tokens', {})
    token_map[name.lower()] = access_token
    token_map[norm_name] = access_token
    session['assessment_tokens'] = token_map
    session.modified = True

    profession_label = next((p['label'] for p in PROFESSION_CATEGORIES if p['value'] == profession), 'Other')
    return jsonify({
        'success': True,
        'name': name,
        'profession': profession,
        'profession_label': profession_label,
        'biography_scores': scores,
        'estimated_scores': predicted,
        'reasoning': reasoning,
        'embedding_info': embedding_info,
        'trait_labels': TRAIT_LABELS,
        'trait_descriptions': TRAIT_DESCRIPTIONS,
        'ocean_letters': OCEAN_LETTERS,
        'has_estimation': predicted is not None,
        'model_is_outdated': model_is_outdated,
        'outdated_reason': outdated_reason,
        'access_token': access_token,
        'profile_url': url_for('profile', name=name, token=access_token)
    })


@app.route('/analyze_private', methods=['POST'])
def analyze_private():
    """Private Participant assessment: consent + age range + BFI-10 + uploaded photo.
    Personality scores come from the participant's own BFI-10 answers (no biography, no name)."""
    rl_err = check_ai_rate_limit('analyze', limit_per_minute=5)
    if rl_err:
        return rl_err

    data = request.get_json(silent=True) or {}
    if data.get('consent') is not True:
        return jsonify({'success': False, 'error': 'Consent is required before taking part.'}), 400

    age_range = str(data.get('age_range') or '').strip()
    if age_range not in PRIVATE_AGE_RANGES:
        return jsonify({'success': False, 'error': 'Please select your age range (18 and above only).'}), 400

    background = str(data.get('background') or '').strip()
    if len(background) > PRIVATE_BACKGROUND_MAX_CHARS:
        return jsonify({'success': False, 'error': f'Background must be at most {PRIVATE_BACKGROUND_MAX_CHARS} characters.'}), 400
    if '\x00' in background:
        return jsonify({'success': False, 'error': 'Background contains invalid characters.'}), 400

    scores, err = compute_bfi10_scores(data.get('answers'))
    if err:
        return jsonify({'success': False, 'error': err}), 400

    image_ref = str(data.get('upload_id') or '').strip()
    if not image_ref:
        return jsonify({'success': False, 'error': 'Please upload your photo.'}), 400
    photo_path, err = validate_and_resolve_upload_path(image_ref)
    if err:
        return jsonify({'success': False, 'error': f'Invalid image reference: {err}'}), 400
    saved_filename = os.path.basename(photo_path)
    if is_image_associated_or_saved(saved_filename):
        return jsonify({'success': False, 'error': 'This photo is already used by another record. Please upload a new photo.'}), 400
    ephemeral_file_path = photo_path

    # Content moderation
    mod_res = moderate_image(photo_path, 'other')
    is_safe, mod_reason, mod_status = mod_res if len(mod_res) == 3 else (mod_res[0], mod_res[1], 'PASSED' if mod_res[0] else 'FAILED')
    if mod_status == 'UNAVAILABLE':
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({'success': False, 'status': 'UNAVAILABLE', 'error': f'Content moderation unavailable: {mod_reason}'}), 503
    if not is_safe:
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({'success': False, 'status': 'FAILED', 'moderation_flag': True,
                        'error': f'Inappropriate image detected: {mod_reason} Please upload a proper photo.'}), 400

    # AI-generated screening
    ai_res = check_ai_generated_image(photo_path)
    if ai_res['is_ai']:
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({'success': False, 'status': 'FAILED', 'ai_generated': True,
                        'error': f"AI-generated image detected: {ai_res['reason']} Please use a real photograph."}), 400

    # Facial features
    embedding, embedding_info, err = extract_face_embedding(photo_path)
    if err:
        if err.startswith("No face detected"):
            category, no_face_msg = explain_missing_face(photo_path)
            cleanup_ephemeral_file(ephemeral_file_path)
            return jsonify({'success': False, 'error': no_face_msg, 'no_face': True, 'no_face_category': category}), 400
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({'success': False, 'error': err}), 400
    if not isinstance(embedding, list) or len(embedding) != 512 or not all(np.isfinite(x) for x in embedding):
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({'success': False, 'error': 'Invalid facial embedding generated.'}), 500

    face_ok, face_category, face_msg = check_face_quality(photo_path)
    if not face_ok:
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({'success': False, 'error': face_msg, 'no_face': True, 'no_face_category': face_category}), 400

    predicted, _ = estimate_personality(embedding)
    model_is_outdated, outdated_reason = get_model_outdated_status()

    answers_int = [int(a) for a in data.get('answers')]
    try:
        duration_seconds = float(data.get('duration_seconds'))
        if not np.isfinite(duration_seconds) or duration_seconds < 0:
            duration_seconds = None
    except (TypeError, ValueError):
        duration_seconds = None
    quality_flags = get_response_quality_flags(answers_int, duration_seconds)

    participant_id = new_participant_id()
    record = {
        'participant_id': participant_id,
        'consent_timestamp': datetime.now(timezone.utc).isoformat(),
        'age_range': age_range,
        'background': background,
        'image_filename': saved_filename,
        'score_source': 'BFI-10',
        'bfi10_answers': ' '.join(str(a) for a in answers_int),
        'duration_seconds': round(duration_seconds, 1) if duration_seconds is not None else None,
        'quality_flags': '; '.join(quality_flags),
    }
    for t in TRAIT_NAMES:
        record[t] = scores[t]
    for t in TRAIT_NAMES:
        record[f'pred_{t}'] = predicted.get(t) if predicted else None
    for i, v in enumerate(embedding, start=1):
        record[f'face_{i}'] = v

    private_file = get_private_participants_file()
    try:
        with public_assessment_transaction_lock(timeout=10.0):
            df_new = pd.DataFrame([record])
            if os.path.exists(private_file) and os.path.getsize(private_file) > 0:
                df_all = pd.concat([pd.read_csv(private_file), df_new], ignore_index=True)
            else:
                df_all = df_new
            atomic_write_csv(private_file, df_all)
    except Exception as e:
        cleanup_ephemeral_file(ephemeral_file_path)
        return jsonify({'success': False, 'status': 'STORAGE_ERROR',
                        'error': f'Could not save your responses safely ({str(e)}). Nothing was stored.'}), 500

    save_image_association(participant_id, saved_filename, status='verified', method='private_participant')

    return jsonify({
        'success': True,
        'subject_type': 'private',
        'name': participant_id,
        'participant_id': participant_id,
        'score_source': 'BFI-10',
        'biography_scores': scores,
        'estimated_scores': predicted,
        'has_estimation': predicted is not None,
        'reasoning': 'Scores come from your own answers to the BFI-10 (10-item Big Five Inventory), scaled 0-100.',
        'embedding_info': embedding_info,
        'trait_labels': TRAIT_LABELS,
        'trait_descriptions': TRAIT_DESCRIPTIONS,
        'ocean_letters': OCEAN_LETTERS,
        'model_is_outdated': model_is_outdated,
        'outdated_reason': outdated_reason,
    })


@app.route('/fetch_image', methods=['POST'])
def fetch_image():
    url = request.json.get('url', '').strip() if request.is_json else request.form.get('url', '').strip()
    if not url:
        return jsonify({'success': False, 'error': 'URL is required.'}), 400
    filename, b64, err = fetch_image_from_url(url)
    if err:
        return jsonify({'success': False, 'error': err}), 400
    extracted_name = extract_name_from_url(url)
    return jsonify({
        'success': True,
        'image_b64': f"data:image/jpeg;base64,{b64}",
        'extracted_name': extracted_name,
        'local_path': filename,
        'upload_id': filename
    })


@app.route('/check_photo', methods=['POST'])
def check_photo():
    """Context-aware photo moderation check using validated upload references."""
    rl_err = check_ai_rate_limit('check_photo', limit_per_minute=15)
    if rl_err:
        return rl_err

    data = request.json if request.is_json else request.form
    image_ref = (data.get('image_path') or data.get('upload_id') or '').strip()
    profession = data.get('profession', 'other').strip()
    profession_label = next((p['label'] for p in PROFESSION_CATEGORIES if p['value'] == profession), 'Other')
    custom_profession = data.get('custom_profession', '').strip()
    display_profession = custom_profession if profession == 'other' and custom_profession else profession_label

    # Reject missing or empty required image inputs (Issue 7)
    if not image_ref:
        return jsonify({
            'success': False,
            'ok': False,
            'status': 'FAILED',
            'error': 'Image reference is required for photo check.'
        }), 400

    safe_path, err = validate_and_resolve_upload_path(image_ref)
    if err:
        return jsonify({'success': False, 'ok': False, 'error': f'Invalid image reference: {err}', 'status': 'FAILED'}), 400

    try:
        started = time.monotonic()
        digest = image_digest(safe_path)

        def fail(issue, reason, **extra):
            payload = {'success': True, 'ok': False, 'status': 'FAILED', 'issue': issue, 'reason': reason}
            payload.update(extra)
            print(f"[PHOTO-CHECK] {os.path.basename(safe_path)}: {issue} in {time.monotonic() - started:.1f}s", flush=True)
            return jsonify(payload)

        def run_moderation():
            media_type, b64_image = encode_image_for_claude(safe_path)
            client = get_claude_client()
            message = call_claude_messages_create(
                client,
                HAIKU_MODEL_CANDIDATES,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64_image}
                        },
                        {
                            "type": "text",
                            "text": f"""You are a content moderator for an academic personality analysis system.

The user selected profession category: "{display_profession}"

Check this image for TWO things:

1. EXPLICIT CONTENT: Does it contain explicit nudity or pornographic content? (Swimwear, athletic wear, and fashion/editorial photos are NOT explicit.)

2. PROFESSION MISMATCH: Does the photo's appearance/context clearly not match the selected profession?
   - Example mismatch: sexy/swimwear photo selected as "Scientist" or "Politician"
   - Example OK: swimwear photo selected as "Model" or "Athlete"
   - When in doubt, allow it.

Respond ONLY with valid JSON (no markdown):
{{"ok": true/false, "issue": "explicit_content" or "profession_mismatch" or null, "reason": "<one clear sentence explaining the issue, or null if ok>"}}"""
                        }
                    ]
                }]
            )
            return extract_and_parse_json(message.content[0].text)

        # 1. Fast local checks first (no API call): is there a face, and is it extremely blurred?
        face_present = detect_face_present(safe_path)

        # 2. Run the AI checks at the same time instead of one after another
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_mod = pool.submit(run_moderation)
            fut_ai = pool.submit(check_ai_generated_image, safe_path)
            fut_face = pool.submit(check_face_quality if face_present else explain_missing_face, safe_path)
            data = fut_mod.result()
            ai_res = fut_ai.result()
            face_res = fut_face.result()

        if 'ok' not in data:
            return jsonify({
                'success': False,
                'ok': False,
                'status': 'FAILED',
                'error': "Missing 'ok' property in photo check response."
            }), 502

        # 3. Report the most serious problem first
        ok_val = parse_strict_boolean(data['ok'], 'ok')
        if not ok_val:
            issue_val = data.get('issue')
            if issue_val == 'explicit_content':
                cleanup_ephemeral_file(safe_path)
            return fail(issue_val, str(data.get('reason') or 'Photo is not appropriate.').strip())
        remember_moderation_passed(digest)

        if ai_res['is_ai']:
            cleanup_ephemeral_file(safe_path)
            return fail('ai_generated', f"{ai_res['reason']} Please use a real photograph of the subject.",
                        detection_method=ai_res['method'], confidence=ai_res['confidence'])

        if not face_present:
            category, no_face_msg = face_res
            return fail('no_face', no_face_msg, no_face_category=category)

        face_ok, face_category, face_msg = face_res
        if not face_ok:
            return fail('no_face', face_msg, no_face_category=face_category)

        print(f"[PHOTO-CHECK] {os.path.basename(safe_path)}: passed in {time.monotonic() - started:.1f}s", flush=True)
        return jsonify({'success': True, 'ok': True, 'status': 'PASSED',
                        'ai_check_status': ai_res['status'],
                        'ai_check_error': ai_res['reason'] if ai_res['status'] == 'UNAVAILABLE' else None})
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({
            'success': False,
            'ok': False,
            'status': 'FAILED',
            'error': f"Invalid photo check response: {str(e)}"
        }), 502
    except Exception as e:
        err_msg = format_claude_error(e)
        status_code = 401 if ('401' in err_msg or 'Authentication' in err_msg) else (429 if '429' in err_msg else 503)
        return jsonify({
            'success': False,
            'ok': False,
            'status': 'UNAVAILABLE',
            'error': f"Photo moderation service unavailable: {err_msg}"
        }), status_code


@app.route('/identify_photo', methods=['POST'])
def identify_photo():
    """Recognize a well-known public figure in a photo (AWS Rekognition) and suggest
    name + profession for auto-fill. Accepts either a multipart 'photo' file (saved as a
    temporary upload and returned as upload_id for reuse in /check_photo and /analyze)
    or an existing 'upload_id' / 'image_path' reference."""
    rl_err = check_ai_rate_limit('identify_photo', limit_per_minute=15)
    if rl_err:
        return rl_err

    # Private participants must give consent before any photo is uploaded
    if str(request.form.get('skip_recognition') or '').lower() in ('1', 'true') and \
            str(request.form.get('consent') or '').lower() not in ('1', 'true'):
        return jsonify({'success': False, 'error': 'Please tick the consent box before uploading your photo.'}), 400

    photo = request.files.get('photo')
    upload_id = None
    if photo and photo.filename:
        if not allowed_image(photo.filename):
            return jsonify({'success': False, 'error': 'Invalid image format.'}), 400
        ext = photo.filename.rsplit('.', 1)[1].lower()
        upload_id = f"{uuid.uuid4().hex}.{ext}"
        temp_path = os.path.join(UPLOAD_DIR, upload_id)
        photo.save(temp_path)
        if cv2.imread(temp_path) is None:
            cleanup_ephemeral_file(temp_path)
            return jsonify({'success': False, 'error': 'Uploaded file is not a valid image.'}), 400
        safe_path = temp_path
    else:
        data = request.get_json(silent=True) or request.form
        image_ref = (data.get('upload_id') or data.get('image_path') or '').strip()
        if not image_ref:
            return jsonify({'success': False, 'error': 'A photo or image reference is required.'}), 400
        safe_path, err = validate_and_resolve_upload_path(image_ref)
        if err:
            return jsonify({'success': False, 'error': f'Invalid image reference: {err}'}), 400
        upload_id = os.path.basename(safe_path)

    skip_recognition = str((request.form.get('skip_recognition') if photo else
                            (request.get_json(silent=True) or {}).get('skip_recognition')) or '').lower() in ('1', 'true')
    if skip_recognition:
        # Private participants: upload only, never sent to celebrity recognition
        return jsonify({'success': True, 'upload_id': upload_id, 'status': 'SKIPPED',
                        'service_enabled': False, 'recognized': False})

    result, err, status = recognize_celebrity(safe_path)
    payload = {
        'success': True,
        'upload_id': upload_id,
        'status': status,
        'service_enabled': status != 'DISABLED',
        'recognized': status == 'RECOGNIZED',
    }
    if status == 'UNAVAILABLE':
        payload['error'] = err
    if status == 'RECOGNIZED':
        profession = suggest_profession_for_public_figure(result['name'])
        payload.update({
            'name': result['name'],
            'confidence': result['confidence'],
            'urls': result['urls'],
            'other_faces': result['other_faces'],
            'profession': profession,
            'profession_label': next((p['label'] for p in PROFESSION_CATEGORIES if p['value'] == profession), None),
        })
    return jsonify(payload)


@app.route('/validate', methods=['POST'])
def validate():
    rl_err = check_ai_rate_limit('validate', limit_per_minute=15)
    if rl_err:
        return rl_err

    data = request.json if request.is_json else request.form
    name = data.get('name', '').strip()
    biography = data.get('biography', '').strip()
    if not name or not biography:
        return jsonify({'success': False, 'error': 'Name and biography required.'}), 400

    # Enforce minimum biography words in validate endpoint
    is_valid_bio, bio_words, bio_err = validate_biography_text(biography)
    if not is_valid_bio:
        return jsonify({
            'success': False,
            'is_match': False,
            'error': bio_err,
            'word_count': bio_words,
            'min_words': MIN_BIOGRAPHY_WORDS
        }), 400

    res = validate_match_via_gemini(name, biography)
    if len(res) == 3:
        is_match, reason, val_status = res
    else:
        is_match, reason = res
        val_status = 'PASSED' if is_match else 'FAILED'

    if val_status == 'UNAVAILABLE':
        status_code = 401 if ('401' in reason or 'Authentication' in reason) else (429 if '429' in reason else 503)
        return jsonify({
            'success': False,
            'is_match': False,
            'status': 'UNAVAILABLE',
            'error': reason
        }), status_code

    # Store server-side validation token in session when matched successfully
    if is_match and val_status == 'PASSED':
        val_token = get_biography_validation_token(name, biography)
        session[val_token] = True

    return jsonify({
        'success': True,
        'is_match': is_match,
        'status': val_status,
        'reason': reason
    })


@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    rl_err = check_ai_rate_limit('paraphrase', limit_per_minute=10)
    if rl_err:
        return rl_err

    data = request.json if request.is_json else request.form
    name = data.get('name', '').strip()
    biography = data.get('biography', '').strip()
    if not name or not biography:
        return jsonify({'success': False, 'error': 'Name and biography required.'}), 400

    # Enforce minimum biography words in paraphrase endpoint
    is_valid_bio, bio_words, bio_err = validate_biography_text(biography)
    if not is_valid_bio:
        return jsonify({
            'success': False,
            'error': bio_err,
            'word_count': bio_words,
            'min_words': MIN_BIOGRAPHY_WORDS
        }), 400
    res = paraphrase_biography_via_gemini(name, biography)
    if len(res) == 3:
        paraphrased, err, status = res
    else:
        paraphrased, err = res
        status = 'PASSED' if not err else 'UNAVAILABLE'
    if err and paraphrased == biography:
        status_code = 401 if ('401' in err or 'Authentication' in err) else (429 if '429' in err else 503)
        return jsonify({'success': False, 'error': err, 'status': status}), status_code
    return jsonify({'success': True, 'paraphrased': paraphrased, 'status': status})


def generate_societal_impact(name, scores, profession_label, reasoning):
    """Generate societal impact interpretation using Claude API."""
    try:
        client = get_claude_client()
        traits_summary = ', '.join([f"{TRAIT_LABELS[t]}: {round(scores[t])}/100" for t in TRAIT_NAMES])
        prompt = f"""You are writing a brief academic note for a personality research system.

Subject: {name}
Profession: {profession_label}
OCEAN Scores: {traits_summary}

Write exactly 2 short paragraphs (3-4 sentences each):
1. What the dominant personality traits suggest about how this person likely approaches their work and relationships.
2. How this type of personality profile may influence or contribute to society in their field.

Keep it concise, grounded, and academic. Do not make absolute claims — use words like "suggests", "may", "tends to". Do not use bullet points or headers."""

        message = call_claude_messages_create(
            client,
            HAIKU_MODEL_CANDIDATES,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip(), None
    except Exception as e:
        return None, format_claude_error(e)


@app.route('/profile/<path:name>')
def profile(name):
    profile_path = resolve_profile_path(name)
    if not profile_path or not os.path.exists(profile_path):
        return redirect(url_for('subjects'))

    # Check if this subject belongs to the protected research baseline dataset
    # Robust against all URL variations: "Albert Einstein", "Albert_Einstein", "Albert-Einstein", etc.
    is_baseline = is_baseline_subject(name, profile_path=profile_path)

    is_auth, _ = verify_admin_authorization()
    is_authorized = is_auth or session.get('research_admin_authorized', False)

    # 1. Baseline research records: strictly restricted to verified researchers under research ethics
    if is_baseline and not is_authorized:
        return render_template('restricted_profile.html',
                               name=name,
                               is_baseline=True,
                               is_public_assessment=False,
                               model_is_outdated=False,
                               outdated_reason=None)

    data = load_profile(name)
    if not data:
        return redirect(url_for('subjects'))

    # 2. Public demonstration assessments: protected from unauthorized public access
    # Accessible by:
    #   - Authorized researchers (management, retrain, delete, audit)
    #   - Assessment creator via browser session (login-free view of own assessment)
    #   - Assessment creator via valid access token (?token=...)
    if not is_baseline and not is_authorized:
        user_assessments = session.get('authorized_public_assessments', [])
        norm_name = normalize_subject_name_for_comparison(name)
        stored_norm = normalize_subject_name_for_comparison(data.get('name', ''))

        is_creator_session = (
            norm_name in user_assessments or
            stored_norm in user_assessments or
            name.lower() in [s.lower() for s in user_assessments]
        )

        req_token = request.args.get('token') or request.form.get('token')
        expected_token = data.get('access_token')
        token_matches = bool(expected_token and req_token and safe_compare(expected_token, req_token))

        # If valid access token is supplied, authorize this browser session
        if token_matches:
            if norm_name not in user_assessments:
                user_assessments.append(norm_name)
                session['authorized_public_assessments'] = user_assessments
                session.modified = True

        if not is_creator_session and not token_matches:
            return render_template('restricted_profile.html',
                                   name=name,
                                   is_baseline=False,
                                   is_public_assessment=True,
                                   model_is_outdated=False,
                                   outdated_reason=None)

    model_is_outdated, outdated_reason = get_model_outdated_status()

    # Generate societal impact if not yet cached in profile
    societal_impact = data.get('societal_impact')
    if not societal_impact and data.get('biography_scores'):
        profession_label = data.get('profession_label', 'Public Figure')
        impact, err = generate_societal_impact(
            name,
            data['biography_scores'],
            profession_label,
            data.get('reasoning', '')
        )
        if impact:
            societal_impact = impact
            data['societal_impact'] = impact
            try:
                atomic_write_json(profile_path, data)
            except Exception:
                pass

    return render_template('profile.html', data=data,
                           trait_labels=TRAIT_LABELS,
                           trait_descriptions=TRAIT_DESCRIPTIONS,
                           ocean_letters=OCEAN_LETTERS,
                           ocean_colors=OCEAN_COLORS,
                           societal_impact=societal_impact,
                           is_baseline=is_baseline,
                           is_authorized=is_authorized,
                           model_is_outdated=model_is_outdated,
                           outdated_reason=outdated_reason)


@app.route('/authorize_researcher', methods=['POST'])
def authorize_researcher():
    """Authorize researcher administrative privileges for this session using RESEARCH_ADMIN_KEY."""
    data = request.json if request.is_json else request.form
    key = normalize_secret_input(data.get('admin_key'))
    admin_key = os.environ.get('RESEARCH_ADMIN_KEY')

    if (not admin_key or not admin_key.strip()
            or admin_key.strip().lower() in INSECURE_SECRET_KEYS
            or admin_key.strip() in PLACEHOLDER_API_KEYS
            or len(admin_key.strip()) < 12):
        err = "Administrative operations are locked. A secure RESEARCH_ADMIN_KEY must be configured on the server."
        if request.is_json:
            return jsonify({'success': False, 'error': err}), 403
        flash(err, 'error')
        return redirect(url_for('researcher_login', next=safe_next_path(request.form.get('next'), url_for('index'))))

    if not key or not safe_compare(admin_key.strip(), key):
        err = "Unauthorized: Invalid Research Admin Key."
        if request.is_json:
            return jsonify({'success': False, 'error': err}), 403
        flash(err, 'error')
        return redirect(url_for('researcher_login', next=safe_next_path(request.form.get('next'), url_for('index'))))

    session['research_admin_authorized'] = True
    session['research_admin_key_hash'] = get_admin_key_hash(admin_key)
    session.modified = True
    if request.is_json:
        return jsonify({'success': True, 'message': 'Researcher privileges unlocked.'})
    return redirect(safe_next_path(request.form.get('next'), url_for('index')))


@app.route('/deauthorize_researcher', methods=['POST'])
def deauthorize_researcher():
    """Lock researcher privileges for this session."""
    session.pop('research_admin_authorized', None)
    session.pop('research_admin_key_hash', None)
    session.modified = True
    if request.is_json:
        return jsonify({'success': True, 'message': 'Researcher privileges locked.'})
    return redirect(url_for('analyze'))


def perform_retrain_model():
    """Execute model training and update ML estimated scores for all subjects.
    Returns (success: bool, message: str, metrics: dict|None, updated_count: int)."""
    success, message, metrics = build_and_train_model()
    if not success:
        return False, message, None, 0

    updated = 0
    if os.path.exists(FACE_EMBEDDINGS_FILE):
        try:
            df_emb = pd.read_csv(FACE_EMBEDDINGS_FILE)
            feat_cols = [c for c in df_emb.columns if c.startswith('face_')]
            for _, row in df_emb.iterrows():
                subj_name = row['name']
                embedding = row[feat_cols].values.tolist()
                predicted, err = estimate_personality(embedding)
                if predicted:
                    prof = load_profile(subj_name)
                    if prof:
                        prof['estimated_scores'] = predicted
                        profile_path = resolve_profile_path(subj_name)
                        if profile_path:
                            atomic_write_json(profile_path, prof)
                        updated += 1
        except Exception:
            pass

    return True, f"{message} ML scores updated for {updated} subjects.", metrics, updated


@app.route('/retrain', methods=['POST'])
def retrain():
    # Strict administrative authorization verification
    is_auth, auth_err = verify_admin_authorization()
    if not is_auth:
        return jsonify({'success': False, 'message': auth_err, 'error': auth_err}), 403

    success, message, metrics, updated = perform_retrain_model()
    if not success:
        return jsonify({'success': False, 'message': message}), 400

    return jsonify({
        'success': True,
        'message': message
    })


@app.route('/insights')
def insights():
    if not is_researcher():
        return researcher_login_redirect()
    metrics = None
    if os.path.exists(MODEL_METRICS_FILE):
        try:
            with open(MODEL_METRICS_FILE, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
    model_is_outdated, outdated_reason = get_model_outdated_status()
    return render_template('insights.html',
                           model_exists=os.path.exists(TRAINED_MODEL_FILE),
                           metrics=metrics,
                           trait_labels=TRAIT_LABELS,
                           ocean_colors=OCEAN_COLORS,
                           model_is_outdated=model_is_outdated,
                           outdated_reason=outdated_reason)


@app.route('/subjects')
def subjects():
    if not is_researcher():
        return researcher_login_redirect()
    is_auth, _ = verify_admin_authorization()
    is_authorized = is_auth or session.get('research_admin_authorized', False)
    subjs = get_subjects(authorized_researcher=is_authorized)
    baseline_count = get_research_baseline_count()
    public_count = 0
    if os.path.exists(PUBLIC_ASSESSMENTS_FILE):
        try:
            df_pub = pd.read_csv(PUBLIC_ASSESSMENTS_FILE)
            public_count = len(df_pub)
        except Exception:
            pass
    model_is_outdated, outdated_reason = get_model_outdated_status()
    return render_template('subjects.html', subjects=subjs,
                           is_authorized=is_authorized,
                           baseline_count=baseline_count,
                           public_count=public_count,
                           model_is_outdated=model_is_outdated,
                           outdated_reason=outdated_reason)


def perform_delete_subject(name):
    """Safely remove a subject from dataset, profiles, and associated image records.
    Returns (result_dict, http_status_code)."""
    name = (name or '').strip()
    if not name:
        return {'success': False, 'error': 'Name is required.'}, 400

    # 1. Verify existence across records
    profile_path = resolve_profile_path(name)
    has_profile = profile_path is not None and os.path.exists(profile_path)

    in_emb = False
    if os.path.exists(FACE_EMBEDDINGS_FILE):
        try:
            df = pd.read_csv(FACE_EMBEDDINGS_FILE)
            in_emb = name in df['name'].values
        except Exception:
            pass

    in_scores = False
    if os.path.exists(PERSONALITY_SCORES_FILE):
        try:
            df = pd.read_csv(PERSONALITY_SCORES_FILE)
            in_scores = name in df['name'].values
        except Exception:
            pass

    associations = load_image_associations()
    in_assoc = name in associations

    in_public = False
    if os.path.exists(PUBLIC_ASSESSMENTS_FILE):
        try:
            df_pub = pd.read_csv(PUBLIC_ASSESSMENTS_FILE)
            in_public = name.strip().lower() in df_pub['name'].dropna().astype(str).str.strip().str.lower().values
        except Exception:
            pass

    if not (has_profile or in_emb or in_scores or in_assoc or in_public):
        return {'success': False, 'error': f'Subject "{name}" not found in dataset.'}, 404

    # 2. Extract image metadata before removing profile
    profile_data = None
    if has_profile:
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
        except Exception:
            pass

    image_filename = None
    if profile_data and profile_data.get('image_filename'):
        image_filename = profile_data.get('image_filename')
    elif in_assoc:
        assoc = associations.get(name)
        if assoc and assoc.get('image_filename'):
            image_filename = assoc.get('image_filename')

    # 3. Handle image deletion safely
    image_deleted = False
    is_shared_image = False
    other_subjects = []
    manual_review_required = False
    partial_failure = False
    warnings = []

    if image_filename:
        # Check if any OTHER subject references this image
        referencing_subjects = get_subjects_referencing_image(image_filename)
        other_subjects = [s for s in referencing_subjects if s != name]

        if other_subjects:
            # Preserved because image is shared by other subjects
            is_shared_image = True
            remove_image_association(name)
        else:
            # Check association status
            assoc = associations.get(name, {})
            if assoc.get('status') == 'manual_review_required':
                manual_review_required = True
                remove_image_association(name)
            else:
                safe_path, err = validate_and_resolve_upload_path(image_filename)
                if safe_path and os.path.exists(safe_path):
                    try:
                        os.remove(safe_path)
                        image_deleted = True
                        remove_image_association(name)
                    except Exception as e:
                        # Retain unresolved association with failure status so it can be retried
                        save_image_association(name, image_filename, status='cleanup_failed', method='deletion_retry')
                        partial_failure = True
                        warnings.append(f"Image cleanup failed ({str(e)}). Association retained for retry.")
                else:
                    remove_image_association(name)
    else:
        remove_image_association(name)

    # 4. Remove CSV records atomically
    with public_assessment_transaction_lock():
        for filepath in [FACE_EMBEDDINGS_FILE, PERSONALITY_SCORES_FILE, DATASET_FILE, PUBLIC_ASSESSMENTS_FILE]:
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if name.strip().lower() in df['name'].dropna().astype(str).str.strip().str.lower().values:
                        df = df[df['name'].dropna().astype(str).str.strip().str.lower() != name.strip().lower()]
                        atomic_write_csv(filepath, df)
                except Exception as e:
                    warnings.append(f"Failed updating {os.path.basename(filepath)}: {str(e)}")
                    partial_failure = True

    # 5. Remove profile JSON
    if has_profile and os.path.exists(profile_path):
        try:
            os.remove(profile_path)
        except Exception as e:
            warnings.append(f"Failed deleting profile JSON: {str(e)}")
            partial_failure = True

    # 6. Flag model as outdated if deleted subject contributed to training
    model_was_outdated = False
    if os.path.exists(MODEL_METRICS_FILE):
        try:
            with open(MODEL_METRICS_FILE, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            training_subjects = metrics.get('training_subjects', [])
            prediction_subjects = [p.get('name') for p in metrics.get('predictions', [])]
            if name in training_subjects or name in prediction_subjects:
                metrics['is_outdated'] = True
                metrics['outdated_reason'] = (
                    f'Subject "{name}" was removed from the dataset. '
                    f'Current model results reflect data from the deleted subject. Please click Retrain Model.'
                )
                atomic_write_json(MODEL_METRICS_FILE, metrics)
                model_was_outdated = True
        except Exception:
            pass

    # 7. Formulate accurate status response
    if partial_failure:
        return {
            'success': False,
            'partial_failure': True,
            'message': f'Subject "{name}" removal encountered issues: {"; ".join(warnings)}',
            'image_deleted': image_deleted,
            'is_shared_image': is_shared_image,
            'manual_review_required': manual_review_required,
            'model_outdated': model_was_outdated,
            'warnings': warnings
        }, 207

    msg = f'"{name}" removed from dataset.'
    if image_deleted:
        msg += ' Associated managed facial image was removed.'
    elif is_shared_image:
        msg += f' Associated image preserved because it is referenced by other subjects: {", ".join(other_subjects)}.'
    elif manual_review_required:
        msg += ' Note: Associated image could not be verified automatically and requires manual review.'

    if model_was_outdated:
        msg += ' Trained model is now outdated; manual retraining is required.'

    return {
        'success': True,
        'partial_failure': False,
        'message': msg,
        'image_deleted': image_deleted,
        'is_shared_image': is_shared_image,
        'manual_review_required': manual_review_required,
        'model_outdated': model_was_outdated
    }, 200


@app.route('/delete_subject', methods=['POST'])
def delete_subject():
    # Strict administrative authorization verification
    is_auth, auth_err = verify_admin_authorization()
    if not is_auth:
        return jsonify({'success': False, 'error': auth_err, 'message': auth_err}), 403

    data = request.json if request.is_json else request.form
    name = data.get('name', '').strip()
    result, status_code = perform_delete_subject(name)
    return jsonify(result), status_code


@app.route('/unresolved_images_report')
def unresolved_images_report():
    """Endpoint returning the audit report of unresolved legacy subjects and orphaned upload files."""
    is_auth, auth_err = verify_admin_authorization()
    if not is_auth:
        return jsonify({'success': False, 'error': 'Unauthorized: Researcher authorization required to view internal audit reports.'}), 403

    if os.path.exists(UNRESOLVED_IMAGES_REPORT_FILE):
        try:
            with open(UNRESOLVED_IMAGES_REPORT_FILE, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify({'error': 'Audit report not found.'}), 404


app.jinja_env.globals['enumerate'] = enumerate

# Initialize baseline research cache for fail-closed protection
try:
    _BASELINE_NAMES_CACHE = set(get_research_baseline_names())
except Exception:
    _BASELINE_NAMES_CACHE = set()

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    app.run(host=host, port=port, debug=debug_mode)