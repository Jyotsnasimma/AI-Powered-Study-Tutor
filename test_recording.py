# test_recording.py
import sounddevice as sd
import numpy as np
import time

def list_devices():
    print("Available devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(f"{i}: {dev['name']} (Inputs: {dev['max_input_channels']})")

def test_recording(device_id=None, duration=3):
    samplerate = 16000
    try:
        print(f"\nTesting recording on device {device_id}...")
        recording = sd.rec(int(duration * samplerate), 
                          samplerate=samplerate, 
                          channels=1,
                          device=device_id)
        sd.wait()
        print(f"Recording shape: {recording.shape}")
        print(f"Max amplitude: {np.max(np.abs(recording))}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    list_devices()
    
    # Test default device
    test_recording()
    
    # Test specific devices
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            test_recording(i)