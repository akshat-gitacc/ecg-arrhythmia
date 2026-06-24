# ECG Arrhythmia Detection

This was a two-person group project aimed at classifying ECG heartbeats as normal or abnormal using the MIT-BIH Arrhythmia Database from PhysioNet.

My contributions to the project included:

Built the ECG preprocessing pipeline using the WFDB library to load and parse MIT-BIH signal and annotation files.
Segmented raw ECG recordings into 360-sample heartbeat windows centered around annotated R-peaks.
Implemented binary label mapping and class handling logic for normal versus abnormal heartbeat classification.
Performed exploratory data analysis and generated visualizations including class distributions and ECG signal plots.
Engineered statistical features from ECG signals for classical machine learning approaches.
Developed and benchmarked Logistic Regression and Random Forest baseline models for comparison against the CNN architecture.

## Project Files

- `src/data/preprocess.py`: Extracts individual heartbeats centered around the R-peak (360 samples per beat) and labels them.
- `src/models/cnn_model.py`: The 1D CNN architecture in PyTorch.
- `src/models/classical_ml.py`: Trains and evaluates the baseline Logistic Regression and Random Forest models on handcrafted statistical features.
- `src/train/train_cnn.py`: Training script for the PyTorch CNN (handles data splits, class weights, and checkpointing).
- `src/train/demo_cnn.py`: A quick demo script to visualize sample beats and run inference using the trained CNN model.

## Results

I split the data into 70% training, 15% validation, and 15% testing sets. Here is how the models compared on the test set:

- **Logistic Regression**: 74% accuracy.
- **Random Forest**: 88% accuracy.
- **1D CNN**: 99.6% accuracy (F1-score of 0.99 for abnormal beats).

The CNN performs significantly better because it learns features directly from the raw heartbeat shape rather than relying on simple statistical aggregates.



