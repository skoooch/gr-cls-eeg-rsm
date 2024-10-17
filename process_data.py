import numpy as np
from scipy.signal import butter, sosfilt, filtfilt
# Define the band-pass filter function
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    # Create a Butterworth band-pass filter
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Apply the filter using filtfilt (zero-phase filtering)
    y = filtfilt(b, a, data, axis=1)  # Filter along the time dimension
    return y

def get_data(task):
    lowcut = 0.2  # Low cutoff frequency (Hz)
    highcut = 115  # High cutoff frequency (Hz)
    fs = 512  # Sampling frequency (Hz), adapt this to your actual EEG data sampling rate
    order = 4  # Filter order (typically between 3-5)
    categories = ['figurine', 'pen', 'chair', 'lamp', 'plant']
    all_data = {}
    participant = "8"
    for category in categories:
        all_data[category] = []
        for i in range(1, 6):
            data = np.loadtxt('data/%s_%s_%s%s.csv' % (category, participant, task, i), delimiter=',')  
            num_rows = data.shape[0]
            data = apply_bandpass_filter(data, lowcut, highcut, fs, order=order)
            data = data.reshape(num_rows, 307, 64)
            num_trials, num_timepoints, num_channels = data.shape
            
            # Determine the number of new trials after averaging every 4
            num_new_trials = num_trials // 10
            
            # Initialize the array to hold the averaged data
            averaged_trials = np.zeros((num_new_trials, num_timepoints, num_channels))
            
            # Average every 4 trials
            for k in range(num_new_trials):
                start_idx = k * 4
                end_idx = start_idx + 4
                averaged_trials[k, :, :] = np.mean(data[start_idx:end_idx, :, :], axis=0)
            
            # Append data
            all_data[category].append(averaged_trials)

    return all_data