import sounddevice as sd
import numpy as np

def test_recording():
    duration = 5  # seconds
    fs = 16000  # sample rate
    
    print("Available audio devices:")
    print(sd.query_devices())
    
    print("\nRecording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete")
    print(f"Recorded {len(recording)} samples")
    return recording

if __name__ == "__main__":
    test_recording()