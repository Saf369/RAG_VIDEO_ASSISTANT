# 🎬 Personalized Meeting Assistant

> Your intelligent companion for meetings and video content. Transcribe, summarize, and query any meeting recording or YouTube video with AI — powered by Sarvam STT, Mistral LLM, and ChromaDB RAG.

![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

**🔴 Live Demo:** [https://saf369-rag-video-assistant-app-zijmva.streamlit.app/](https://saf369-rag-video-assistant-app-zijmva.streamlit.app/)

## ✨ Features

- **YouTube Audio Download** — Extracts audio from any YouTube video via `yt-dlp`
- **Speech-to-Text** — Transcribes audio using Sarvam AI STT API (supports Hindi → English translation)
- **AI Summarization** — Map-reduce summarization with Mistral LLM
- **Insight Extraction** — Extracts actionable items, discussion points, and open questions
- **RAG Q&A** — Ask follow-up questions answered directly from the transcript via ChromaDB vector search

---

## 🚀 Deployment

### Option 1: Streamlit Community Cloud (Recommended)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the **main file** to `app.py`
5. Add your secrets in **Settings → Secrets**:
   ```toml
   MISTRAL_API_KEY = "your-mistral-api-key"
   WHISPER_MODEL = "small"
   SARVAM_API = "your-sarvam-api-key"
   SARVAM_STT_MODEL = "saaras:v3"
   ```
6. Deploy!

> The `packages.txt` file ensures `ffmpeg` is installed automatically.

### Option 2: Docker

```bash
# Build the image
docker build -t ai-video-assistant .

# Run with environment variables
docker run -p 8501:8501 \
  -e MISTRAL_API_KEY="your-key" \
  -e SARVAM_API="your-key" \
  -e SARVAM_STT_MODEL="saaras:v3" \
  -e WHISPER_MODEL="small" \
  ai-video-assistant
```

Open [http://localhost:8501](http://localhost:8501)

### Option 3: Heroku / Render / Railway

These platforms use the `Procfile` and `runtime.txt`:

```bash
# Heroku
heroku create my-video-assistant
heroku config:set MISTRAL_API_KEY=your-key SARVAM_API=your-key SARVAM_STT_MODEL=saaras:v3
heroku buildpacks:add --index 1 heroku-community/apt
git push heroku main

# Render / Railway
# Just connect your GitHub repo — they auto-detect the Procfile
```

> **Note:** For Heroku, add the `heroku-community/apt` buildpack so `packages.txt` is respected.

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

---

## 🔑 Environment Variables

| Variable          | Required | Description                          |
| ----------------- | -------- | ------------------------------------ |
| `MISTRAL_API_KEY` | ✅       | Mistral AI API key for LLM calls     |
| `SARVAM_API`      | ✅       | Sarvam AI API key for speech-to-text |
| `SARVAM_STT_MODEL`| ✅       | Sarvam STT model name (e.g. `saaras:v3`) |
| `WHISPER_MODEL`   | ❌       | Whisper model size (default: `small`)|
| `CHROMA_DIR`      | ❌       | ChromaDB storage path (default: `vector_db`) |

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
├── Dockerfile             # Container deployment
├── Procfile               # Heroku/Render/Railway
├── runtime.txt            # Python version for PaaS platforms
└── .env.example           # Environment variable template
```

---

## 📄 License

MIT
