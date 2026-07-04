import whisper
import os

WHISPER_MODEL=os.getenv("WHISPER_MODEL","small")
_model=None

def load_model():
    global _model
    if _model is None:
        _model=whisper.load_model(WHISPER_MODEL)
        print(f"Whisper model {WHISPER_MODEL} loaded successfully")
    return _model

def transcribe_audio(audio_path:str)->str:
    model=load_model()
    result=model.transcribe(audio_path)
    return result["text"]

def transcribe_chunk(chunk_path:str,translate:bool=False)->str:
    model=load_model()
    result=model.transcribe(chunk_path,task="translate") if translate else model.transcribe(chunk_path)
    return result["text"]

def transcribe_all(chunks:list[str],translate:bool=False)->str:
    full_text=""
    for i,chunk in enumerate(chunks):
        text=transcribe_chunk(chunk,translate)
        print(f"Segment {i+1}: {len(text)} chars")
        full_text+=text
    print("Full transcription complete")    
    return full_text
    

    
