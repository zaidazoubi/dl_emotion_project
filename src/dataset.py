import os
import glob
import numpy as np
import librosa

# --- Configuration ---
# Use relative paths so it works no matter where you move the folder
# Assuming the script is run from the root project folder
RAW_DATA_PATH = os.path.join("data", "Audio_Speech_Actors_01-24")
PROCESSED_PATH = os.path.join("data", "processed")
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 173  # Calculated from your notebook analysis

# Emotion mapping (from RAVDESS filename documentation)
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    """
    Loads audio, extracts MFCC features, and pads/truncates to MAX_LEN.
    """
    try:
        # Load audio (librosa converts to mono by default)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Extract MFCCs (The "features" of the voice)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc = mfcc.T  # Transpose to shape (Time, Features)
        
        # Pad or Truncate to ensure fixed shape (MAX_LEN, N_MFCC)
        # This ensures every audio file produces a matrix of the exact same size
        if mfcc.shape[0] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:MAX_LEN, :]
            
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_data():
    print(f"Looking for data in: {RAW_DATA_PATH}")
    # Find all .wav files in the subfolders
    wav_files = glob.glob(os.path.join(RAW_DATA_PATH, "**", "*.wav"), recursive=True)
    
    if not wav_files:
        print("No files found! Please check that the 'data' folder is in the project root.")
        return

    print(f"Found {len(wav_files)} files. Starting feature extraction...")
    
    X = [] # Features
    y = [] # Labels
    
    for i, file in enumerate(wav_files):
        # Print progress every 100 files
        if i % 100 == 0: print(f"Processing {i}/{len(wav_files)}...")
        
        # Get label from filename (RAVDESS format: 03-01-06...)
        file_name = os.path.basename(file)
        parts = file_name.split("-")
        
        if len(parts) < 3: continue # Skip weird files
        
        emotion_code = parts[2]
        label = EMOTIONS.get(emotion_code)
        
        features = extract_features(file)
        
        if features is not None and label is not None:
            X.append(features)
            y.append(label)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Expand dimensions for CNN (Batch, Time, Features, Channels)
    # Resulting shape should be (1440, 173, 40, 1)
    X = X[..., np.newaxis]
    
    print(f"Extraction complete.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Save the processed data
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    np.save(os.path.join(PROCESSED_PATH, "X.npy"), X)
    np.save(os.path.join(PROCESSED_PATH, "y.npy"), y)
    print(f"Data saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    process_data()