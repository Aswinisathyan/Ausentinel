
import numpy as np
import librosa
from scipy.fft import dct


SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FILTERS = 20
N_COEFFICIENTS = 20
MAX_DURATION = 4.0


def create_linear_filterbank(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    n_filters=N_FILTERS,
):
    """Create triangular filters spaced linearly in frequency."""

    frequencies = np.linspace(
        0, sample_rate / 2, n_fft // 2 + 1
    )

    boundaries = np.linspace(
        0, sample_rate / 2, n_filters + 2
    )

    filterbank = np.zeros(
        (n_filters, len(frequencies)),
        dtype=np.float32,
    )

    for i in range(n_filters):
        left, center, right = boundaries[i:i + 3]

        rising = (frequencies >= left) & (
            frequencies <= center
        )
        falling = (frequencies > center) & (
            frequencies <= right
        )

        filterbank[i, rising] = (
            frequencies[rising] - left
        ) / max(center - left, 1e-12)

        filterbank[i, falling] = (
            right - frequencies[falling]
        ) / max(right - center, 1e-12)

    return filterbank


FILTERBANK = create_linear_filterbank()


def extract_lfcc(audio, sample_rate=SAMPLE_RATE):
    """
    Extract a fixed-length LFCC summary vector.

    Input:
        audio: mono or stereo waveform as a NumPy array
        sample_rate: input sample rate

    Output:
        80-dimensional float32 feature vector
    """

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Audio is empty.")

    # Convert stereo to mono
    if audio.ndim == 2:
        if audio.shape[0] <= 2:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)

    if audio.ndim != 1:
        raise ValueError("Expected mono or stereo audio.")

    # Remove invalid values
    audio = np.nan_to_num(audio)

    # Resample to 16 kHz
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE,
        )

    # Standardize duration to 4 seconds
    target_length = int(SAMPLE_RATE * MAX_DURATION)

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio)),
        )
    else:
        audio = audio[:target_length]

    # Short-time Fourier transform
    spectrum = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )

    power_spectrum = np.abs(spectrum) ** 2

    # Linear-frequency filterbank energies
    energies = FILTERBANK @ power_spectrum
    log_energies = np.log(energies + 1e-10)

    # Cepstral coefficients
    lfcc = dct(
        log_energies,
        type=2,
        axis=0,
        norm="ortho",
    )[:N_COEFFICIENTS]

    # First- and second-order temporal differences
    delta = librosa.feature.delta(lfcc, order=1)
    delta2 = librosa.feature.delta(lfcc, order=2)

    # Summarize temporal variation
    features = np.concatenate([
        lfcc.mean(axis=1),
        lfcc.std(axis=1),
        delta.mean(axis=1),
        delta2.mean(axis=1),
    ])

    return features.astype(np.float32)


def extract_lfcc_from_file(path):
    """Load an audio file and extract LFCC features."""

    audio, sample_rate = librosa.load(
        path,
        sr=None,
        mono=True,
    )

    return extract_lfcc(audio, sample_rate)
