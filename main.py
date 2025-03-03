# Imports

import numpy as np
import os
from sklearn.model_selection import train_test_split

import config
from utilities.data_loader import load_physionet_data
from utilities.preprocessing import preprocess_eeg_data
from utilities.feature_extraction import extract_features_mne
from models.classifiers import train_and_evaluate_models
from visualization.plots import visualize_results


def main():
    # ==== 1: LOAD DATA ====
    # Raw EEG brain signals from the PhysioNet dataset
    print("Loading EEG data...")
    X, y = load_physionet_data(
        config.BASE_DIR,
        subjects=config.SUBJECTS, # Participants to include
        runs=config.RUNS # Runs to include
    )
    y = np.array(y)  # Convert labels to numpy array (0=left hand, 1=right hand)
    print(f"Data loaded: {len(X)} samples with {X[0].shape[0]} channels")
    

    # ==== 2: PREPROCESS DATA ====
    # Filter out noise & unwanted frequencies
    # Look at config.py for more detail
    print("Preprocessing data...")
    print("Preprocessing data...")
    X_preprocessed = preprocess_eeg_data(
        X,
        sfreq=config.SFREQ, 
        bandpass=config.BANDPASS_FILTER,
        notch=config.NOTCH_FILTER 
    )
    
    #  ==== 3: EXTRACT FEATURES ====
    # Transform the cleaned signals into meaningful measurements
    print("Extracting features...")
    features = extract_features_mne(
        X_preprocessed,
        sfreq=config.SFREQ,
        frequency_bands=config.FREQUENCY_BANDS
    )
    print(f"Feature matrix shape: {features.shape}")
    
    # ==== 4: SPLIT DATA FOR TRAINING ====
    # Divide data into training set and test set 
    X_train, X_test, y_train, y_test = train_test_split(
        features, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # ==== 5: TRAIN & EVALUATE MODELS ====
    # Train the models, see which best distinguishes between left and right hand imagery
    print("Training and evaluating models...")
    results, conf_matrices = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # ==== 6: VISUALIZE RESULTS ====
    # Create visual representations perfomance (accuracy, confusion matrices)
    print("Visualizing results...")
    visualize_results(results, conf_matrices, save_dir='results')
    
    print("EEG classifications completed! :)")


if __name__ == "__main__":
    main()