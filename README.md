 # EEG Classification Project

This project processes and classifies EEG data from the PhysioNet Motor Movement/Imagery Dataset. It identifies whether a subject is imagining left or right hand movement based on EEG signals.

![Project_Flowchart](https://github.com/user-attachments/assets/cdfc9e48-e609-4a05-9118-28348dad77da)

-> Flowchart created with draw.io

## Frequency Patterns

EEG (Electroencephalogram) signals are typically analyzed in different frequency bands, each associated with different mental states:

- Delta (0.5-4 Hz): Typically associated with deep sleep
- Theta (4-8 Hz): Often present during drowsiness or meditation
- Alpha (8-13 Hz): Common during relaxed wakefulness, especially with eyes closed
- Beta (13-30 Hz): Present during normal waking consciousness, especially active thinking
- Gamma (30-45 Hz): Associated with higher cognitive processing and cross-modal sensory processing

Motor imagery (imagining movement) typically produces changes in the alpha and beta bands, particularly over motor cortex regions of the brain.

## Data Source

The dataset comes from the BCI2000 instrumentation system:
- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)

### Citation

Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.

## Project Structure

eeg_classifier/
│
├── utilities/
│   ├── __init__.py          
│   ├── data_loader.py         # Data loading functions
│   ├── preprocessing.py       # Preprocessing functions
│   └── feature_extraction.py  # Feature extraction functions
│
├── models/
│   ├── __init__.py            
│   └── classifiers.py         # Model training & eval functions
│
├── visualization/
│   ├── __init__.py            
│   └── plots.py               # Visualization functions
│
├── config.py                  # Configuration params
├── main.py                    # Main script
└── README.md                  # You are here :)
 

## Setup

1. Install the required packages:

pip install numpy pandas matplotlib scipy scikit-learn mne tqdm


2. Update the BASE_DIR in config.py to point to the location of your PhysioNet dataset

## Using the pipeline

Run the main script to process the EEG data and train models:
    python main.py

## Pipeline Steps

1. Data Loading: Load EEG data from PhysioNet dataset
2. Preprocessing: Apply bandpass and notch filters
3. Feature Extraction: Extract time and frequency domain features
4. Model Training: Train SVM, Random Forest, and Neural Network models
5. Evaluation: Compare model performance with accuracy metrics & confusion matrices
6. Visualization: Generate plots to visualize model performance

## Results

Results are saved in the `results/` directory, including:
- Model comparison bar chart
- Confusion matrices for each model
