import numpy as np

DATA_PATH = './data'

def bandpass_eeg(data, fs=125.0, low=4.0, high=30.0, taps=101):
    """Offline windowed-sinc FIR, independently along each trial/channel's time axis.

    Assumes the supplied trials retain the original 125 Hz sampling rate.
    Reflection padding and centered convolution preserve length without delay;
    this uses future samples and is not a causal streaming filter.
    """
    if not (0 < low < high < fs / 2) or taps < 3 or taps % 2 == 0:
        raise ValueError("Use 0 < low < high < fs/2 and an odd tap count >= 3.")
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 4 or x.shape[1:3] != (1, 2) or x.shape[-1] <= taps // 2:
        raise ValueError("Expected EEG shape [N, 1, 2, T] with enough time samples.")
    if not np.isfinite(x).all():
        raise ValueError("Fill missing values before filtering.")
    n = np.arange(taps) - taps // 2
    kernel = (2 * high / fs * np.sinc(2 * high / fs * n)
              - 2 * low / fs * np.sinc(2 * low / fs * n)) * np.hamming(taps)
    # Unit gain at the center of the requested passband.
    kernel /= np.abs(np.sum(kernel * np.exp(-2j * np.pi * ((low + high) / 2) / fs * n)))
    padded = np.pad(x, ((0, 0), (0, 0), (0, 0), (taps // 2, taps // 2)), mode='reflect')
    return np.ascontiguousarray(np.apply_along_axis(
        lambda signal: np.convolve(signal, kernel, mode='valid'), -1, padded))

def single_cutoff_eeg(data, cutoff, kind, fs=125.0, taps=101):
    """Offline high/low-pass FIR with the same window and padding as E26.

    Low-pass has unit DC gain; high-pass has unit Nyquist gain.
    Cutoffs are transition centers, not brick-wall boundaries.
    """
    if kind not in ('highpass', 'lowpass') or not (0 < cutoff < fs / 2):
        raise ValueError("Use highpass/lowpass and 0 < cutoff < fs/2.")
    if taps < 3 or taps % 2 == 0:
        raise ValueError("Use an odd tap count >= 3.")
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 4 or x.shape[1:3] != (1, 2) or x.shape[-1] <= taps // 2:
        raise ValueError("Expected EEG shape [N, 1, 2, T] with enough time samples.")
    if not np.isfinite(x).all():
        raise ValueError("Fill missing values before filtering.")
    n = np.arange(taps) - taps // 2
    kernel = 2 * cutoff / fs * np.sinc(2 * cutoff / fs * n)
    if kind == 'highpass':
        kernel = -kernel
        kernel[taps // 2] += 1
    kernel *= np.hamming(taps)
    reference = fs / 2 if kind == 'highpass' else 0.0
    kernel /= np.abs(np.sum(kernel * np.exp(-2j * np.pi * reference / fs * n)))
    padded = np.pad(x, ((0, 0), (0, 0), (0, 0), (taps // 2, taps // 2)), mode='reflect')
    return np.ascontiguousarray(np.apply_along_axis(
        lambda signal: np.convolve(signal, kernel, mode='valid'), -1, padded))


def read_bci_data():
    S4b_train = np.load(DATA_PATH + '/S4b_train.npz')
    X11b_train = np.load(DATA_PATH + '/X11b_train.npz')
    S4b_test = np.load(DATA_PATH + '/S4b_test.npz')
    X11b_test = np.load(DATA_PATH + '/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label
