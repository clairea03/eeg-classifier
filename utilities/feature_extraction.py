import numpy as np
import scipy.signal as signal
from tqdm import tqdm


# This function turns cleaned EEG signals into measurements
# Helps the models distinguish left vs. right hand imagery

def extract_features_mne(X, sfreq=160, frequency_bands=None):
    
    if frequency_bands is None:
        frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    
    # Calculate dimensions for feature matrix
    n_samples = len(X) # Number of EEG recordings
    n_channels = X[0].shape[0] # Number of electrodes/channels
    
    # For each channel, extract:
    # 2 time domain features (mean, std)
    # Power in each freq band (delta, theta, alpha, beta, gamma)
    n_features_per_channel = 2 + len(frequency_bands) 
    
    # Create empty array to store all features
    features = np.zeros((n_samples, n_channels * n_features_per_channel))
    
    # Process each EEG recording
    for i, x in enumerate(tqdm(X, desc="Extracting features")):
        
        # Process each channel (electrode)
        for j in range(n_channels):
            
            # Extract basic statistical properties
            mean = np.mean(x[j])
            std = np.std(x[j])
            
            # Convert time signal to frequency components
            freqs, psd = signal.welch(x[j], fs=sfreq)
            
            # Calculate the position in the feature array
            feat_idx = j * n_features_per_channel
            
            # Store time domain features
            features[i, feat_idx] = mean
            features[i, feat_idx + 1] = std
            
            # Calculate power in each freq band 
            # Different mental activities create different patterns in these frequency bands
            for k, (band_name, (low, high)) in enumerate(frequency_bands.items()):
                
                # Find which frequencies in our spectrum fall in this band
                band_idx = np.logical_and(freqs >= low, freqs <= high)
                
                # Calc mean power in this band
                band_power = np.mean(psd[band_idx]) if np.any(band_idx) else 0
                
                # Store band power in feature array
                features[i, feat_idx + 2 + k] = band_power
    
    return features