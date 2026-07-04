import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_FOLDER = "downloads"
os.makedirs(DOWNLOAD_FOLDER,exist_ok=True)


#Download YouTube video as MP3
def download_audio_from_youtube(url: str) -> str:
    output_path=os.path.join(DOWNLOAD_FOLDER,"%(title)s.%(ext)s")
    ydl_opts={
        "format":"bestaudio/best",
        "outtmpl":f"{output_path}",
        "postprocessors":[
            {
                "key":"FFmpegExtractAudio",
                "preferredcodec":"mp3",
                "preferredquality":"192",
            }
        ]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict=ydl.extract_info(url,download=True)
        audio_path=ydl.prepare_filename(info_dict).replace(".webm",".mp3").replace(".m4a","mp3")
        if os.path.exists(audio_path):
            return audio_path
def convert_to_wav(audio_path:str)->str:
    audio=AudioSegment.from_file(audio_path)
    audio=audio.set_frame_rate(16000)
    audio=audio.set_channels(1)
    filename=os.path.basename(audio_path)
    wav_path=os.path.join(DOWNLOAD_FOLDER,f"{filename}_converted.wav")
    audio.export(wav_path,format="wav")
    return wav_path

#print(convert_to_wav(data))
def chunk_audio(audio_path:str,chunk_length_ms:int=600000)->list[str]:
    audio=AudioSegment.from_wav(audio_path)
    chunks=[]
    filename=os.path.basename(audio_path)
    for i in range(0,len(audio),chunk_length_ms):
        chunk=audio[i:i+chunk_length_ms]
        chunk_path=os.path.join(DOWNLOAD_FOLDER,f"{filename}_chunk_{i}.wav")
        chunk.export(chunk_path,format="wav")
        chunks.append(chunk_path)
    return chunks

if __name__ == "__main__":
    data = download_audio_from_youtube("https://youtu.be/kovGit0HIxE?si=Sl9uwji10sRElGDk")            
    print(chunk_audio(convert_to_wav(data)))



    

    