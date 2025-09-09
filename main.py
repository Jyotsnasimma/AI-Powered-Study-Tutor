import os
import faiss
import fitz
import numpy as np
import pyttsx3
import tempfile
import base64
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pydantic import BaseModel
import ollama
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import json
import queue
import sys
import time
from faster_whisper import WhisperModel

WHISPER_MODEL_PATH = "model/base.en"
whisper_model = WhisperModel(WHISPER_MODEL_PATH, device="cpu", compute_type="int8")

if not os.path.exists("./vosk-model-en-us-0.42-gigaspeech"):
    print("Downloading larger Vosk model... (This will take several minutes)")
    import urllib.request
    import zipfile
    model_url = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip"
    try:
        urllib.request.urlretrieve(model_url, "model.zip")
        with zipfile.ZipFile("model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("model.zip")
        print("Vosk model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading Vosk model: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pdf_chunks = {}
faiss_index = {}
embedding_models = {}
chunk_lookup = {}

model_name = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(model_name)

class QuestionInput(BaseModel):
    file_name: str
    question: str
    tts: bool = True

def extract_pdf_text(path):
    """Improved PDF text extraction that works with PyPDF2"""
    reader = PdfReader(path)
    text = ""
    for i, page in enumerate(reader.pages):
        if page_text := page.extract_text():
            text += f"\n\n[PAGE {i + 1}]\n{page_text}"
    return text

def split_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Improved chunking that respects page boundaries and natural breaks"""
    page_chunks = []
    current_page = ""
    page_num = 1
    
    for line in text.split('\n'):
        if line.startswith('[PAGE '):
            if current_page:
                page_chunks.append((f"Page {page_num}", current_page))
                page_num = int(line.split(' ')[1].rstrip(']'))
                current_page = ""
        else:
            current_page += line + '\n'
    
    if current_page:
        page_chunks.append((f"Page {page_num}", current_page))
    
    final_chunks = []
    for page_title, page_text in page_chunks:
        words = page_text.split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = " ".join(words[i:i + chunk_size])
            final_chunks.append(f"{page_title} - Chunk {len(final_chunks) + 1}\n{chunk}")
    
    return final_chunks

def create_faiss_index(chunks, model):
    """Create index with improved metadata handling"""
    clean_chunks = ["\n".join(chunk.split('\n')[1:]) for chunk in chunks]
    embeddings = model.encode(clean_chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def find_top_chunks(question, chunks, model, index, top_n=5):
    """Find top chunks with improved scoring"""
    q_embedding = model.encode([question])
    distances, indices = index.search(np.array(q_embedding), k=top_n)
    
    results = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        similarity = float(1 / (1 + distances[0][i]))
        results.append({
            "text": chunk,
            "similarity": similarity,
            "page": chunk.split('\n')[0].split(' - ')[0]
        })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return [r for r in results if r['similarity'] > 0.3]

def ask_local_llm(context, question):
    """Improved prompt with structured response formatting"""
    prompt = f"""You are an AI tutor that answers questions strictly based on the provided textbook content.

CONTEXT FROM TEXTBOOK:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer ONLY using information from the context above
2. If the question isn't answered in the context, say "This information is not covered in the material"
3. Provide a structured response with clear sections and proper spacing
4. Use bullet points or numbered lists where applicable

ANSWER:
"""
    response = ollama.chat(
        model='tinyllama',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.3}
    )
    return response['message']['content']

def speak_text(text):
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        filename = f.name
    engine.save_to_file(text, filename)
    engine.runAndWait()
    with open(filename, "rb") as f:
        b64_audio = base64.b64encode(f.read()).decode("utf-8")
    os.remove(filename)
    return b64_audio

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        os.makedirs("./model", exist_ok=True)
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
        file_path = os.path.join("./model", safe_filename)
        
        if os.path.exists(file_path):
            base, ext = os.path.splitext(safe_filename)
            counter = 1
            while os.path.exists(os.path.join("./model", f"{base}_{counter}{ext}")):
                counter += 1
            file_path = os.path.join("./model", f"{base}_{counter}{ext}")
        
        contents = await file.read()
        
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp_pdf.pdf")
        
        try:
            with open(temp_path, "wb") as temp_file:
                temp_file.write(contents)
            
            try:
                text = extract_pdf_text(temp_path)
                if not text.strip():
                    raise HTTPException(status_code=400, detail="PDF appears to be empty or unreadable")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")
            
            with open(file_path, "wb") as f:
                f.write(contents)
            
            chunks = split_into_chunks(text)
            index, _ = create_faiss_index(chunks, embedding_model)

            pdf_chunks[os.path.basename(file_path)] = chunks
            faiss_index[os.path.basename(file_path)] = index
            
            return {"message": "PDF processed successfully", "file": os.path.basename(file_path)}
            
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Warning: Error cleaning up temp files: {e}")
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask/")
def ask(input: QuestionInput):
    if input.file_name not in pdf_chunks:
        return {"error": "PDF not uploaded or indexed."}

    chunks = pdf_chunks[input.file_name]
    index = faiss_index[input.file_name]
    
    top_chunks = find_top_chunks(input.question, chunks, embedding_model, index)
    
    if not top_chunks:
        answer = "This information is not covered in the material."
        audio_b64 = speak_text(answer) if input.tts else None
        return {
            "answer": answer,
            "audio": audio_b64,
            "similarity": 0.0
        }
    
    context = "\n\n".join([chunk['text'] for chunk in top_chunks])
    answer = ask_local_llm(context, input.question)

    audio_b64 = speak_text(answer) if input.tts else None
    return {
        "answer": answer,
        "audio": audio_b64,
        "similarity": float(top_chunks[0]['similarity']),
        "page": top_chunks[0]['page'] if 'page' in top_chunks[0] else None
    }

@app.get("/mic-to-text/")
def mic_to_text():
    try:
        samplerate = 16000
        silence_threshold = 0.02
        min_silence_duration = 2.0
        max_recording_duration = 30.0
        
        print("🔊 Starting voice recording (speak now)...")
        
        audio_chunks = []
        is_recording = True
        silent_frames = 0
        frames_since_last_sound = 0
        
        def callback(indata, frames, time, status):
            nonlocal silent_frames, frames_since_last_sound, is_recording
            volume_norm = np.linalg.norm(indata) * 10
            if volume_norm < silence_threshold:
                silent_frames += 1
                frames_since_last_sound += 1
                if (silent_frames > min_silence_duration * samplerate / frames or 
                    len(audio_chunks) * frames / samplerate > max_recording_duration):
                    is_recording = False
            else:
                silent_frames = 0
                frames_since_last_sound = 0
            audio_chunks.append(indata.copy())

        with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
            while is_recording:
                sd.sleep(100)
        
        if not audio_chunks:
            return {"error": "No audio recorded", "status": "error"}
            
        audio = np.concatenate(audio_chunks)
        audio = audio.astype(np.float32).flatten()
        audio /= np.max(np.abs(audio))
        
        segments, _ = whisper_model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,
            initial_prompt="This is a lecture about computer science."
        )
        
        text = " ".join(segment.text for segment in segments).strip()
        
        if not text:
            return {"error": "No speech detected", "status": "error"}
            
        print(f"✅ Transcription: {text}")
        return {"text": text, "status": "success"}
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"error": "System error. Try again.", "status": "error"}

@app.get("/")
async def root():
    return {"message": "AI Tutor Backend is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)