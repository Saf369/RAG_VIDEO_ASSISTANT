import os
import requests
from dotenv import load_dotenv

load_dotenv()

SARVAM_API=os.getenv("SARVAM_API")
SARVAM_STT_URL="https://api.sarvam.ai/speech-to-text"
SARVAM_STT_MODEL=os.getenv("SARVAM_STT_MODEL")

def transcribe_audio(audio_path:str)->str:
    """Transcribes an audio file using Sarvam API."""
    return transcribe_chunk(audio_path)

def transcribe_chunk(chunk_path:str, translate:bool=False, language_code:str="hi-IN")->str:
    """Transcribes a chunk of audio using Sarvam API."""
    if not SARVAM_API:
        raise ValueError("SARVAM_API not found in environment variables")
    if not SARVAM_STT_MODEL:
        raise ValueError("SARVAM_STT_MODEL not found in environment variables")
    if not chunk_path:
        raise ValueError("chunk_path not provided")    
    with open(chunk_path,"rb") as f:
        files={"file": (os.path.basename(chunk_path), f, 'audio/wav')}
        headers={"Authorization":f"Bearer {SARVAM_API}"}
        data={"model":SARVAM_STT_MODEL, "language_code": language_code}
        if translate:
            data["mode"] = "translate"
        response=requests.post(SARVAM_STT_URL,files=files,headers=headers,data=data)
        if response.status_code!=200:
            raise ValueError(f"Error from Sarvam API: {response.text}")
        
        resp_json = response.json()
        return resp_json.get("transcript", resp_json.get("text", str(resp_json)))

def transcribe_all(chunks:list[str], translate:bool=False, language_code:str="hi-IN")->str:
    """Transcribes multiple chunks using Sarvam API and concatenates them."""
    full_text=""
    for i,chunk in enumerate(chunks):
        text=transcribe_chunk(chunk, translate, language_code)
        print(f"Segment {i+1}: {len(text)} chars")
        full_text+=text
    print("Full transcription complete")    
    return full_text
