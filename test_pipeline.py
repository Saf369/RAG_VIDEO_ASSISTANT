import os
from utils.audio_processor import download_audio_from_youtube, convert_to_wav, chunk_audio
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_actionable_items, extract_questions, extract_key_discussion_points
from core.rag_engine import build_vector_store, build_rag_chain, get_retriever, ask_question

def print_section(title: str, content):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    if isinstance(content, list):
        for item in content:
            if item.strip():
                print(f"  {item}")
    else:
        print(content)

def test_pipeline():
    url = "https://youtu.be/_Q-e_nczWqM?si=51Jmgz_jRyhd41JY"

    # --- Step 1: Download ---
    print(f"\n1. Downloading audio from {url}...")
    audio_path = download_audio_from_youtube(url)
    if not audio_path or not os.path.exists(audio_path):
        print("   ERROR: Failed to download audio.")
        return
    print(f"   Downloaded to: {audio_path}")

    # --- Step 2: Convert ---
    print("\n2. Converting to WAV format...")
    wav_path = convert_to_wav(audio_path)
    print(f"   Converted to: {wav_path}")

    # --- Step 3: Chunk ---
    print("\n3. Chunking audio for processing...")
    chunks = chunk_audio(wav_path)
    print(f"   Created {len(chunks)} chunks.")

    # --- Step 4: Transcribe ---
    print("\n4. Transcribing chunks with Sarvam (Hinglish output)...")
    transcription = transcribe_all(chunks, translate=True)
    print_section("Transcription", transcription)

    # --- Step 5: Summarize ---
    print("\n5. Generating title and summary...")
    title = generate_title(transcription)
    summary = summarize(transcription)
    print_section(f"Title: {title}", summary)

    # --- Step 6: Extract ---
    print("\n6. Extracting actionable items, discussion points and questions...")
    actionable_items = extract_actionable_items(transcription)
    discussion_points = extract_key_discussion_points(transcription)
    questions = extract_questions(transcription)

    print_section("Actionable Items", actionable_items)
    print_section("Key Discussion Points", discussion_points)
    print_section("Open Questions", questions)

    # --- Step 7: RAG Q&A ---
    print("\n7. Building vector store for Q&A...")
    vector_store = build_vector_store(transcription)
    chain = build_rag_chain(get_retriever(vector_store))
    print("   Done! You can now ask questions about the video.")
    print("   Type 'exit' to quit.\n")

    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ("exit", "quit", ""):
            print("Exiting Q&A. Goodbye!")
            break
        answer = ask_question(query, chain)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    test_pipeline()
