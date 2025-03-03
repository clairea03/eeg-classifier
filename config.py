# Configuration parameters


BASE_DIR = "/Users/clairealverson/EEG_proj/physionet.org/files/eegmmidb/1.0.0"  # Update to your path!!
SUBJECTS = range(1, 11)  # Subjects to include
RUNS = [4, 8, 12]  # Runs to include (imagery of left/right hand movement)

# Preprocessing 
SFREQ = 160  # Sampling freq is 160 measurements per second
BANDPASS_FILTER = (0.5, 45)  # Brain activity occurs at 0.5 - 45 Hertz
NOTCH_FILTER = 60  # Remove electrical interference from power lines (@60 Hz) 

# Feature extraction
TIME_WINDOW = 2  # Seconds
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Model parameters
TEST_SIZE = 0.3
RANDOM_STATE = 23