# FacePersona — Web-Based Personality Analyzer

A Flask web application that examines relationships between facial feature representations and Big Five (OCEAN) personality trait scores inferred from biographical text using Claude AI.

---

## Technical Stack & Algorithms

- **Facial Detection**: OpenCV Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)
- **Public Figure Recognition (optional)**: AWS Rekognition `RecognizeCelebrities` — auto-fills subject name and profession category for recognized public figures (match confidence ≥ `CELEBRITY_MIN_CONFIDENCE`, default 90%)
- **Facial Feature Representation**: Histogram of Oriented Gradients (HOG) — 512-dimensional normalized vector
- **Biographical Personality Scoring**: Anthropic Claude Sonnet (`claude-sonnet-4-5-20250929` with automatic candidate fallbacks)
- **AI-Generated Image Screening**: (1) provenance/metadata markers left by AI generators (C2PA content credentials, IPTC `trainedAlgorithmicMedia`, Stable Diffusion/ComfyUI PNG chunks, Midjourney/DALL-E/Firefly tags) — definitive when present; (2) Claude Vision visual assessment, blocked only at confidence ≥ `AI_IMAGE_BLOCK_CONFIDENCE` (default 80). *Limitation:* no detector is fully accurate; metadata is lost when images are screenshotted or re-saved, and visual detection can miss high-quality AI images.
- **Content Moderation & Verification**: Anthropic Claude Haiku (`claude-haiku-4-5-20251001` with candidate fallbacks)
- **Machine Learning Pipeline**: `StandardScaler` + `PCA` (dynamic components) + `RandomForestRegressor`
- **Evaluation Methodology**: Leakage-free held-out train/test evaluation (80/20 split) with MAE, R², and per-trait Pearson correlation ($r$) and $p$-value. Training forest OOB R² retained as secondary diagnostic.
- **Backend & Storage**: Python 3.14 / Flask, CSV tabular storage, JSON profile store, local image vault.

---

## Setup & Configuration

### 1. Install Dependencies

Ensure Python 3.9+ is installed, then install required packages:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the provided template to create your `.env` file:

```bash
copy .env.example .env
```

Open `.env` and fill in your actual credentials:

```dotenv
# Anthropic Claude API Key (Required for AI personality scoring, validation, and vision moderation)
# Obtain your key at https://console.anthropic.com/
ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here

# Anthropic Claude Models (Defaults to latest stable versions with automatic fallback cascade)
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
ANTHROPIC_HAIKU_MODEL=claude-haiku-4-5-20251001

# Flask Secret Key for session signing
# Generate a cryptographically secure key: python -c "import secrets; print(secrets.token_hex(32))"
FLASK_SECRET_KEY=your_generated_64char_hex_secret_here

# Flask Debug Mode (False for production/evaluation; True only for active local debugging)
FLASK_DEBUG=False

# Application Port (Default: 5000)
PORT=5000

# Research Administrative Key (Required for remote /retrain and /delete_subject calls; local CLI manage_research.py works directly)
# Generate a secure key (min. 12 characters): python -c "import secrets; print(secrets.token_hex(32))"
RESEARCH_ADMIN_KEY=your_generated_admin_secret_here

# Minimum Biography Word Count (Configurable single source of truth for frontend and backend)
# Recommended practical lower minimum: 80 words for adviser review (default: 80)
MIN_BIOGRAPHY_WORDS=80

# Cookie Security (Set to False for local http://127.0.0.1:5000; True behind HTTPS)
SESSION_COOKIE_SECURE=False
```

### 3. (Optional) Enable Public Figure Recognition

1. In the AWS Console, open **IAM → Users → Create user**, attach the `AmazonRekognitionReadOnlyAccess` policy, then create an **access key** for that user.
2. Add these to `.env`:
   ```dotenv
   AWS_ACCESS_KEY_ID=your_access_key_id
   AWS_SECRET_ACCESS_KEY=your_secret_access_key
   AWS_REGION=ap-southeast-1
   CELEBRITY_MIN_CONFIDENCE=90
   ```
3. Restart the app. When a photo is loaded (URL or upload), recognized public figures auto-fill the name and profession; users can still edit both before submitting. If the keys are missing, the feature is silently disabled.

> **Security Note:** Never hardcode API keys or secret keys in source files. The `.gitignore` and `.gitattributes` files are configured to strictly prevent `.env`, local secrets, and Python caches from being committed to version control or packaged into shared archives (`export-ignore`).

---

## Security Architecture & Public Access Model

1. **Public vs. Researcher Access & CSRF Protection**:
   - **Public visitors** only see the **Analyze** page (`/analyze`) and their own results/profile (via session or access token). Visiting `/` redirects to Analyze.
   - **Researcher area** (requires `RESEARCH_ADMIN_KEY` via **Researcher Login** at `/researcher`): Dashboard (`/`), Subjects (`/subjects`), **Participants & Data** (`/researcher/participants`, private participants + CSV export + deletion), and Insights (`/insights`). Locking the researcher view returns to the public Analyze page.
   - All state-mutating requests (`POST`, `PUT`, `DELETE`) are strictly guarded against Cross-Site Request Forgery (CSRF) via session-scoped cryptographic tokens (`X-CSRFToken`).
   - Secure session cookies are configured with `HttpOnly`, `SameSite=Lax`, 8-hour lifetimes, and environment-configurable `SESSION_COOKIE_SECURE` for HTTPS production.

2. **Strict Administrative Authorization & Researcher Tools**:
   - Administrative maintenance functions (`/retrain`, `/delete_subject`) strictly enforce authentication. Unauthenticated requests are denied with HTTP 403 Forbidden.
   - In the web interface, clicking Retrain or Delete prompts researchers for the `RESEARCH_ADMIN_KEY` and retains it securely in client `sessionStorage` for the duration of the browser session.
   - On the server, researchers can manage the system directly without exposing remote endpoints via the dedicated CLI:
     ```bash
     # Inspect system status and model evaluation metrics:
     python manage_research.py status

     # Execute model retraining on the current paired dataset:
     python manage_research.py retrain

     # Safely remove a subject:
     python manage_research.py delete "Subject Name"

     # Inspect image associations and unresolved legacy records:
     python manage_research.py audit-images
     ```

3. **Strict Path Confinement & Traversal Prevention**:
   - Upload references are sanitized using `secure_filename()` and restricted strictly within the authorized `uploads/` directory using `os.path.commonpath()` and `os.path.realpath()` checks.

4. **Server-Side Request Forgery (SSRF) & DNS Rebinding Defense**:
   - Image URL fetching validates IP addresses to block localhost (`127.0.0.1`), loopback, private ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local (`169.254.0.0/16`), and non-public targets.
   - Connections bind directly to pre-validated public IP addresses (`IPPinnedHTTPConnection` / `IPPinnedHTTPSConnection`), completely neutralizing Time-of-Check to Time-of-Use (TOCTOU) DNS rebinding attacks while preserving TLS SNI.
   - Redirect chains are validated at every hop, streams are limited to 10MB, and files are validated using OpenCV before storage.

5. **Safe Image Ownership & Deletion Reliability**:
   - Distinguishes between ephemeral temporary uploads, existing saved images, and shared images.
   - Failed processing or moderation cleanups only remove unassociated temporary files; saved or shared images are never deleted.
   - Subject deletion verifies image ownership, preserves images shared by multiple subjects, uses atomic file writes, and retains associations on partial cleanup failures for retry.

6. **Server-Side Validation Enforcement**:
   - The `/analyze` backend independently enforces name-biography validation. It does not trust client flags and executes server-side validation if no verified session token is present.
   - Malformed AI responses, invalid Big Five score ranges, and non-numerical values are rejected with explicit errors, preventing corrupt data from being committed to research storage.

7. **Research Dataset Confidentiality & Public Assessment Separation**:
   - The original 103 research baseline subjects used to train the machine learning pipeline are strictly preserved in `data/processed/dataset.csv`.
   - In compliance with institutional research ethics and participant privacy protocols, individual profiles of baseline subjects are shielded from anonymous public visitors. Attempting to view a baseline profile displays a confidential research notice (`restricted_profile.html`) unless unlocked with the `RESEARCH_ADMIN_KEY`. Aggregate research statistics on `/insights` remain fully public.
   - Public demonstration assessments are isolated in `data/processed/public_assessments.csv`. Individual assessment profiles and results are protected so they are accessible only to the user who created them (via active browser session or secure access token) and verified researchers, preventing unauthorized visitors from snooping on other participants' names, photos, or biographies.
   - The `/unresolved_images_report` audit endpoint is restricted to authenticated researchers to prevent leakage of internal server directory structures.

8. **Photo Locking & Assessment Integrity**:
   - Users can freely change or upload new photos prior to analysis.
   - Upon successful assessment completion, the photo input, preview, and replace controls are permanently locked with a visual "Photo Locked" badge to prevent accidental substitution after scoring.
   - To analyze a new photo or subject, users click "Start New Assessment", which clears stale state, unlocks photo controls, and resets form validation.
   - Facial ML personality estimations strictly extract a fresh 512-dimensional HOG vector from the uploaded photo and pass it through the ML pipeline (`StandardScaler` -> `PCA` -> `RandomForestRegressor`), distinctly separated from Claude AI's biographical scores.

---

## Statistical Evaluation Methodology

- **Leakage-Free Validation**:
  - The model pipeline utilizes an 80/20 train/test split. Preprocessing (`StandardScaler` and dynamic `PCA`) is fitted **strictly** on training records, preventing data leakage into evaluation metrics.
- **Statistically Sound Undefined Value Handling**:
  - For zero-variance or undefined outcomes (e.g. division by zero in $R^2$ or Pearson $r$), metrics are explicitly represented as `None` in Python and serialized as `null` in JSON.
  - The web interface displays `N/A` rather than misleading default values (such as `0.0` or `1.0`).
  - Genuine negative $R^2$ values are preserved as valid indicators that the model performs worse than the mean baseline predictor.
  - The Out-of-Bag (OOB) $R^2$ is presented as an internal training-forest diagnostic and clearly distinguished from held-out test evaluation.

---

## Running the Application

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000** (Direct access — no login or credentials required).

---

## Automated Testing Suite

To run the fully isolated automated test suite (which executes inside a sandboxed temporary directory without modifying research datasets or models):

```bash
python -B -m unittest discover -s tests -p "test_*.py" -v
```

---

## Production Deployment & Hosting Compatibility

### 1. WSGI Server Configuration
In production Linux environments (Render, Railway, Fly.io, Heroku, AWS EC2, VPS), run the application via **Gunicorn** rather than the Flask development server. A production [`Procfile`](file:///c:/Users/domeng/Documents/CAPSTONE/FacePersona_WebApp_Updated/Procfile) is provided:

```text
web: gunicorn app:app --workers 2 --threads 4 --timeout 120
```

- **Workers & Threads**: 2 workers with 4 threads accommodate concurrent user requests while preventing memory exhaustion from OpenCV and scikit-learn models.
- **Timeout**: Set to 120 seconds to allow sufficient duration for external Claude AI API calls and PCA/ML inference.

### 2. Persistent Storage on Cloud Platforms
Demonstration assessment submissions (`public_assessments.csv`), generated profile JSON files (`data/profiles/`), and uploaded facial photos (`uploads/`) are stored on disk.
To ensure records persist across server redeployments and container restarts on cloud platforms:

- **Volume Mounts**: Attach a persistent disk volume to your container (e.g., Render Persistent Disk mounted at `/data` or Railway Volume).
- **Environment Overrides**: Set the following environment variables in your hosting dashboard:
  ```dotenv
  FACEPERSONA_DATA_DIR=/data
  FACEPERSONA_UPLOAD_DIR=/data/uploads
  ```
- **Automatic Baseline Seeding**: When an external volume is mounted, the application automatically verifies and seeds the 103 baseline research records and models into the persistent mount directory on startup if not already present.

### 3. Platform Compatibility & Architectural Assessment

| Hosting Platform | Compatibility | Considerations & Requirements |
|---|---|---|
| **Render** | **Fully Compatible** (Recommended) | Deploy as Python Web Service using `gunicorn app:app`. Requires attaching a **Persistent Disk** (paid tier) mounted at `/data` for long-term file retention across deploys. Set `TRUST_PROXY_HEADERS=True` and `TRUSTED_PROXIES=127.0.0.1,::1`. |
| **Railway** | **Fully Compatible** (Recommended) | Deploy using the standard Dockerfile or Nixpacks builder with a Persistent Volume attached at `/data`. Set `PORT` and environment variables. |
| **Ubuntu / VPS / Docker** | **Fully Compatible** (Optimal) | Full OS control with local persistent disk. Configure Nginx reverse proxy with SSL (Let's Encrypt / Certbot) proxying to Gunicorn systemd service on `127.0.0.1:5000`. Install system package `libgl1` if using standard OpenCV. |
| **Vercel / AWS Lambda** | **Incompatible / Not Recommended** | Serverless platforms enforce **read-only ephemeral filesystems** (all local writes to CSV, JSON, and uploads are erased after function execution), strict payload size limits (<250MB unzipped bundle vs heavy ML stack: OpenCV + NumPy + SciPy + Scikit-Learn), and cold start timeouts (>15s). A migration to external S3 buckets and Postgres would be required to support serverless. |

### 4. Reverse Proxy & HTTPS Security
When deploying behind an edge reverse proxy (Cloudflare, Nginx, Render Ingress, AWS ALB):
- Set `SESSION_COOKIE_SECURE=True` in `.env` to ensure session cookies are only transmitted over TLS/HTTPS.
- Set `TRUST_PROXY_HEADERS=True` and declare your proxy IP(s) in `TRUSTED_PROXIES` so the server can extract legitimate client IPs for rate limiting without allowing direct header spoofing.

---

## Private Participant Mode (non-public subjects)

The Analyze page has two modes: **🌟 Public Figure** (name + biography → Claude AI scoring) and **👤 Private Participant** for ordinary people.

- **Consent first:** participants must confirm they are 18+ and agree to take part.
- **Minimal data:** age range, uploaded photo, optional short background (max 300 chars). **No name is collected**; each participant gets a random **Participant ID** (e.g. `P-1A2B3C4D`).
- **Personality basis:** the **BFI-10** (Rammstedt & John, 2007), 10 self-report items on a 1–5 scale; each trait = mean of two items (one reverse-keyed), rescaled to 0–100. No biography or AI scoring is used.
- **Same photo safeguards:** moderation, AI-generated screening, and face detection. Photos are never sent to celebrity recognition.
- **Storage:** `data/processed/private_participants.csv` (scores, BFI-10 answers, age range, 512-D facial embedding, image filename). Private participants are **not** shown in the Subjects list and are **not** part of the research baseline or model training.
- **Researcher tools:**
  ```bash
  python manage_research.py private-list
  python manage_research.py private-delete P-1A2B3C4D   # honors a deletion request
  ```
- *Limitation:* the BFI-10 trades reliability for brevity (two items per trait).
