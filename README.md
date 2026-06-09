# Distorted-Visual-Sequence-Pattern-Recognition

A deep learning pipeline for recognizing character sequences from heavily distorted grayscale images.

---

## Overview

This project tackles the problem of reading text from images affected by noise, blur, overlapping symbols, shape deformation, and occlusion — conditions that make traditional OCR unreliable. The solution uses a **CRNN (Convolutional Recurrent Neural Network)** architecture trained end-to-end with **CTC loss**, capable of predicting variable-length character sequences from fixed-size image inputs.

---

## Architecture

```
Input Image (32 × 128 × 1)
        │
        ▼
  CNN Backbone
  ├── Conv(64)  → BN → ReLU → MaxPool(2×2)
  ├── Conv(128) → BN → ReLU → MaxPool(2×2)
  ├── Conv(256) × 2 → BN → ReLU → MaxPool(2×1)
  ├── Conv(512) × 2 → BN → ReLU → MaxPool(2×1)
  └── Conv(512, 4×3) → BN → ReLU → Dropout(0.25)
        │
        ▼
  Reshape + Dense(256, ReLU)
        │
        ▼
  Bidirectional LSTM (128 units) × 2
        │
        ▼
  Dense → Softmax (per time step)
        │
        ▼
  CTC Decode → Predicted Text
```

---

## Dataset

| Split | Contents |
|-------|----------|
| Train | Images + ground truth text labels (`train-labels.csv`) |
| Test  | Images only (predictions submitted as CSV) |

**Image specs:** Grayscale, resized to `32 × 128` px during preprocessing.

**Distortion types present in the data:**
- Background noise and visual artifacts
- Overlapping and deformed characters
- Gaussian blur
- Random occlusion patches
- Irregular spacing and alignment

Download the dataset: [Google Drive](https://drive.google.com/drive/folders/1lRUA-1uCCXfks8kpypFV-4f0UepWoLkU?usp=sharing)

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Image size | 32 × 128 |
| Batch size | 128 |
| Epochs | 15 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Validation split | 10% |
| RNN units | 128 |
| Loss function | CTC (Connectionist Temporal Classification) |

**Data augmentation (training only):**
- Random brightness scaling (±25%)

---

## Pipeline

```
1. Mount Google Drive & extract dataset zip
2. Build character vocabulary from training labels
3. Initialize CaptchaGenerator (Keras Sequence) for batched loading
4. Build CRNN model (training + prediction heads)
5. Train with CTC loss + ModelCheckpoint + CER callback
6. Evaluate on validation set (CER + Exact Match Accuracy)
7. Run greedy CTC decode on test images
8. Export predictions to CSV
```

---

## Evaluation

Submissions are scored on **Character Error Rate (CER)**, computed via Levenshtein distance:

```
CER = (Insertions + Deletions + Substitutions) / Total characters in ground truth
```

Lower CER = better. The notebook also tracks **Exact Match Accuracy** (percentage of sequences predicted perfectly).

---

## Output

The notebook generates a submission CSV in the required format:

```
image,prediction
test-0.png,AXU323
test-1.png,BT91KD
test-2.png,QWER12
```

---

## Requirements

```
tensorflow
keras
numpy
pandas
pillow
matplotlib
scikit-learn
tqdm
```

Run on **Google Colab** (GPU recommended). Mount your Drive and update `ZIP_PATH` to point to your dataset.

---

## Files

| File | Description |
|------|-------------|
| `Distorted_Visual_Sequence_Pattern_Recognition.ipynb` | Full solution notebook |
| `output.csv` | Test set predictions |
| `sample_images.png` | Grid of training samples with labels |
| `training_curves.png` | CTC loss and CER plots over epochs |
| `val_predictions.png` | Visual comparison of GT vs predicted on validation |
