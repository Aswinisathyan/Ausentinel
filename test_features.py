
import numpy as np
from features import extract_lfcc

# Generate 1 second of test audio
sample_rate = 16000
duration = 1.0

time = np.arange(
    int(sample_rate * duration)
) / sample_rate

audio = 0.5 * np.sin(2 * np.pi * 440 * time)

features = extract_lfcc(audio, sample_rate)

print("Feature shape:", features.shape)
print("Data type:", features.dtype)
print("All values finite:", np.isfinite(features).all())
print("First five values:", features[:5])

assert features.shape == (80,)
assert features.dtype == np.float32
assert np.isfinite(features).all()

print("LFCC feature extraction test passed!")
