# 🎬 AI Video Assistant

> Transcribe, summarize, and query any YouTube video with AI — powered by Sarvam STT, Mistral LLM, and ChromaDB RAG.

![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **YouTube Audio Download** — Extracts audio from any YouTube video via `yt-dlp`
- **Speech-to-Text** — Transcribes audio using Sarvam AI STT API (supports Hindi → English translation)
- **AI Summarization** — Map-reduce summarization with Mistral LLM
- **Insight Extraction** — Extracts actionable items, discussion points, and open questions
- **RAG Q&A** — Ask follow-up questions answered directly from the transcript via ChromaDB vector search
- **PDF Export** — Generates a structured report of the transcript, summary, and insights via ReportLab/fpdf2
- **Multi-format Deployment** — Docker, Render/Railway/Fly.io, and Heroku support out of the box

---

## 🚀 Deployment

### Option 1: Docker (Recommended)

This app requires a JS runtime (Deno) for `yt-dlp` to bypass YouTube's bot-detection challenges, which isn't available on platforms like Streamlit Community Cloud. Docker gives full control over the environment.

```bash
# Build the image
docker build -t ai-video-assistant .

# Run with environment variables
docker run -p 8501:8501 \
  -e MISTRAL_API_KEY="your-key" \
  -e SARVAM_API="your-key" \
  -e SARVAM_STT_MODEL="saaras:v3" \
  -e WHISPER_MODEL="small" \
  -e PORT="8501" \
  ai-video-assistant
```

Open [http://localhost:8501](http://localhost:8501)

### Option 2: Render / Railway / Fly.io

These platforms support deploying directly from the `Dockerfile`:

1. Push this repo to GitHub
2. Create a new Web Service and connect your repo
3. Select **Docker** as the environment (not the auto-detected Python buildpack)
4. Add the required environment variables in the dashboard (see below)
5. Deploy — the platform will inject a `$PORT` value automatically, which the `Dockerfile`'s `CMD` binds to

### Option 3: Heroku

Uses the `Procfile` and `runtime.txt`:

```bash
heroku create my-video-assistant
heroku config:set MISTRAL_API_KEY=your-key SARVAM_API=your-key SARVAM_STT_MODEL=saaras:v3
heroku buildpacks:add --index 1 heroku-community/apt
git push heroku main
```

> **Note:** Add the `heroku-community/apt` buildpack so `packages.txt` is respected. Heroku doesn't support custom JS runtimes as cleanly as Docker-based hosts, so YouTube downloads may hit bot-detection more often there.

---

## ⚠️ Known Deployment Challenges

This project hit several real-world issues getting YouTube downloads working reliably in production. Documenting them here in case you run into the same things:

### 1. `HTTP Error 403: Forbidden` on Streamlit Community Cloud
**Cause:** As of early 2026, YouTube requires solving a JavaScript signature challenge before it will serve audio/video streams. `yt-dlp` needs an external JS runtime (Deno) to do this, and Streamlit Community Cloud's environment doesn't provide one — its `packages.txt` only supports `apt-get` installs, and Deno isn't in Debian's default repos.
**Fix:** Moved deployment to Docker, where the `Dockerfile` installs Deno directly:
```dockerfile
RUN curl -fsSL https://deno.land/install.sh | sh
ENV PATH="/root/.deno/bin:$PATH"
```

### 2. `HTTP Error 429: Too Many Requests`
**Cause:** YouTube rate-limits repeated requests from the same IP, especially during active development/testing against the same video.
**Fix:** Mostly transient — retrying after a short delay usually resolves it. A retry-with-backoff wrapper around the `yt-dlp` download call is recommended for production use so users aren't hit with a hard failure on a temporary rate limit.

### 3. `Sign in to confirm you're not a bot`
**Cause:** YouTube's bot-detection layer, sometimes triggered after repeated requests or from a "less trusted" IP range (e.g. cloud/datacenter IPs used by hosting platforms).
**Fix:** Added optional cookie-based authentication support:
```python
if os.path.exists("cookies.txt"):
    ydl_opts["cookiefile"] = "cookies.txt"
```
Cookies are exported from a logged-in browser session and passed to `yt-dlp` — see the **YouTube Authentication** section below. In testing, this wasn't always necessary (the Deno fix alone resolved most cases), but it's kept as a fallback since YouTube's bot-detection behavior is inconsistent.

### 4. Render/Railway port binding
**Cause:** These platforms inject a dynamic `$PORT` environment variable at runtime rather than using a fixed port; hardcoding `--server.port=8501` in the `Dockerfile` causes health checks to fail.
**Fix:** `CMD` now binds dynamically:
```dockerfile
CMD sh -c "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
```

> **Takeaway:** YouTube's anti-bot measures change frequently and `yt-dlp` patches in response — treat this as an evolving problem, not a one-time fix. Keep `yt-dlp` updated and expect to revisit cookies/JS-runtime handling periodically.

---

## 🔐 YouTube Authentication (Cookies)

If you hit the **"Sign in to confirm you're not a bot"** error:

1. Install a browser extension like **"Get cookies.txt LOCALLY"** and export your YouTube cookies (while logged in) to `cookies.txt`
2. Locally, mount the file into the container:
   ```bash
   docker run -p 8501:8501 --env-file .env -e PORT=8501 \
     -v "$(pwd)/cookies.txt:/app/cookies.txt" \
     ai-video-assistant
   ```
3. For hosted deployments, base64-encode the file and store it as an environment variable:
   ```bash
   base64 -w 0 cookies.txt
   ```
   Set the output as `YT_COOKIES_B64` in your platform's environment variables. The `Dockerfile` decodes this into `cookies.txt` automatically at container startup.

> **Note:** Cookies expire periodically (typically weeks to a couple months) and will need to be re-exported if this error resurfaces.

---

## 🛠️ Local Development

```bash
# Clone the repo
git clone <your-repo-url>
cd AI_RAG_prompteng

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the app
streamlit run app.py
```

### Prerequisites

- Python 3.11+
- `ffmpeg` installed on your system (`sudo apt install ffmpeg` on Ubuntu)
- For reliable YouTube downloads: a JS runtime such as [Deno](https://deno.land) on your PATH (the `Dockerfile` installs this automatically for containerized runs)

---

## 🔑 Environment Variables

| Variable           | Required | Description                                               |
| ------------------ | -------- | ----------------------------------------------------------- |
| `MISTRAL_API_KEY`  | ✅       | Mistral AI API key for LLM calls                             |
| `SARVAM_API`       | ✅       | Sarvam AI API key for speech-to-text                         |
| `SARVAM_STT_MODEL` | ✅       | Sarvam STT model name (e.g. `saaras:v3`)                      |
| `WHISPER_MODEL`    | ❌       | Whisper model size (default: `small`)                        |
| `CHROMA_DIR`       | ❌       | ChromaDB storage path (default: `vector_db`)                  |
| `PORT`             | ❌       | Port to bind Streamlit to (auto-set by most hosts)             |
| `YT_COOKIES_B64`   | ❌       | Base64-encoded YouTube cookies (see Authentication above)       |

---

## 📁 Project Structure

```
AI_RAG_prompteng/
├── app.py                 # Streamlit web UI
├── main.py                # CLI entry point
├── core/
│   ├── transcriber.py     # Sarvam STT integration
│   ├── summarize.py       # Map-reduce summarization
│   ├── extractor.py       # Insight extraction (actions, questions, points)
│   ├── rag_engine.py      # ChromaDB vector store + RAG chain
│   └── vector_core.py     # Vector store utilities
├── utils/
│   └── audio_processor.py # YouTube download, WAV conversion, chunking
├── .streamlit/
│   └── config.toml        # Streamlit theme & server config
├── requirements.txt       # Python dependencies
├── packages.txt           # System dependencies (for Streamlit Cloud)
├── Dockerfile              # Container deployment (installs Deno, ffmpeg)
├── Procfile                # Heroku/Render/Railway
├── runtime.txt              # Python version for PaaS platforms
└── .env.example              # Environment variable template
```

---

## 📄 License

MIT
