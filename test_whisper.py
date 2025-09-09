from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import os

# Verify files exist
model_dir = "model/base.en"
required_files = ["config.json", "model.bin", "tokenizer.json"]
for file in required_files:
    if not os.path.exists(f"{model_dir}/{file}"):
        print(f"❌ Missing file: {model_dir}/{file}")
        exit()

print("✅ All model files present")

try:
    model = WhisperModel(model_dir, device="cpu", compute_type="int8")
    print("🎤 Recording for 5 seconds... Speak now!")
    
    audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
    sd.wait()
    
    audio = audio.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio)
    text = " ".join(segment.text for segment in segments)
    
    print("You said:", text)
except Exception as e:
    print(f"❌ Error: {str(e)}")