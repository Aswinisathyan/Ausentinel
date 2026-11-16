
import numpy as np
import pytest

from features import (
    FILTERBANK,
    MAX_DURATION,
    N_FFT,
    N_FILTERS,
    SAMPLE_RATE,
    extract_lfcc,
)


def sine(duration=1.0, sample_rate=SAMPLE_RATE, frequency=440.0):
    """Generate a test tone."""

    time = np.arange(int(sample_rate * duration)) / sample_rate

    return 0.5 * np.sin(2 * np.pi * frequency * time)


# --------------------------------------------------
# 1. OUTPUT FORMAT
# --------------------------------------------------

def test_output_is_80_finite_float32_values():

    features = extract_lfcc(sine(), SAMPLE_RATE)

    assert features.shape == (80,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_same_input_gives_same_output():

    audio = sine()

    np.testing.assert_array_equal(
        extract_lfcc(audio, SAMPLE_RATE),
        extract_lfcc(audio, SAMPLE_RATE),
    )


def test_different_tones_give_different_features():

    low = extract_lfcc(sine(frequency=300), SAMPLE_RATE)
    high = extract_lfcc(sine(frequency=3000), SAMPLE_RATE)

    assert not np.allclose(low, high)


# --------------------------------------------------
# 2. INPUT HANDLING
# --------------------------------------------------

def test_stereo_channels_first_matches_mono():

    mono = sine()
    stereo = np.stack([mono, mono])

    np.testing.assert_allclose(
        extract_lfcc(stereo, SAMPLE_RATE),
        extract_lfcc(mono, SAMPLE_RATE),
        rtol=1e-5,
        atol=1e-5,
    )


def test_stereo_channels_last_matches_mono():

    mono = sine()
    stereo = np.stack([mono, mono], axis=1)

    np.testing.assert_allclose(
        extract_lfcc(stereo, SAMPLE_RATE),
        extract_lfcc(mono, SAMPLE_RATE),
        rtol=1e-5,
        atol=1e-5,
    )


def test_other_sample_rates_are_resampled():

    features = extract_lfcc(sine(sample_rate=44100), 44100)

    assert features.shape == (80,)
    assert np.isfinite(features).all()


def test_audio_longer_than_limit_is_truncated():

    limit = int(SAMPLE_RATE * MAX_DURATION)

    audio = sine(duration=MAX_DURATION + 2)

    np.testing.assert_array_equal(
        extract_lfcc(audio, SAMPLE_RATE),
        extract_lfcc(audio[:limit], SAMPLE_RATE),
    )


def test_short_audio_is_padded():

    features = extract_lfcc(sine(duration=0.1), SAMPLE_RATE)

    assert features.shape == (80,)
    assert np.isfinite(features).all()


def test_silence_gives_finite_features():

    features = extract_lfcc(np.zeros(SAMPLE_RATE), SAMPLE_RATE)

    assert np.isfinite(features).all()


def test_nan_values_are_removed():

    audio = sine()
    audio[::100] = np.nan

    assert np.isfinite(extract_lfcc(audio, SAMPLE_RATE)).all()


def test_empty_audio_is_rejected():

    with pytest.raises(ValueError, match="empty"):
        extract_lfcc(np.array([]), SAMPLE_RATE)


def test_three_dimensional_audio_is_rejected():

    with pytest.raises(ValueError):
        extract_lfcc(np.zeros((2, 2, SAMPLE_RATE)), SAMPLE_RATE)


# --------------------------------------------------
# 3. FILTERBANK
# --------------------------------------------------

def test_filterbank_shape_and_range():

    assert FILTERBANK.shape == (N_FILTERS, N_FFT // 2 + 1)
    assert FILTERBANK.min() >= 0
    assert FILTERBANK.max() <= 1


def test_every_filter_covers_some_frequencies():

    assert (FILTERBANK.sum(axis=1) > 0).all()
