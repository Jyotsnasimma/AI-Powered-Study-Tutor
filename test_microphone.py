import sounddevice as sd
import numpy as np

def test_microphone():
    duration = 5  # seconds
    fs = 44100  # sample rate
    
    print("Available audio devices:")
    print(sd.query_devices())
    
    print("\nTesting microphone (speak now)...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    
    print("\nRecording complete")
    print(f"Recorded {len(recording)} samples")
    print(f"Max amplitude: {np.max(np.abs(recording))}")
    
    if np.max(np.abs(recording)) < 0.01:
        print("\nWARNING: Very low audio levels detected")
        print("1. Check your microphone is properly connected")
        print("2. Make sure microphone permissions are granted")
        print("3. Try increasing your microphone volume")
    else:
        print("\nMicrophone appears to be working!")

if __name__ == "__main__":
    test_microphone()