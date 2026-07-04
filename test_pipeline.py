import os
from utils.audio_processor import download_audio_from_youtube, convert_to_wav, chunk_audio
from core.transcriber import transcribe_all

def test_pipeline():
    url = "https://youtu.be/kovGit0HIxE?si=Sl9uwji10sRElGDk"
    
    print(f"1. Downloading audio from {url}...")
    audio_path = download_audio_from_youtube(url)
    if not audio_path or not os.path.exists(audio_path):
        print("Failed to download audio.")
        return
    print(f"   Downloaded to: {audio_path}")
    
    print("\n2. Converting to WAV format...")
    wav_path = convert_to_wav(audio_path)
    print(f"   Converted to: {wav_path}")
    
    print("\n3. Chunking audio for processing...")
    chunks = chunk_audio(wav_path)
    print(f"   Created {len(chunks)} chunks.")
    
    print("\n4. Transcribing chunks with Whisper...")
    transcription = transcribe_all(chunks)
    
    print("\n--- Final Transcription ---")
    print(transcription)

if __name__ == "__main__":
    test_pipeline()
