# Complete Documentation - Multilabel Image Classification

## Table of Contents
1. [Architecture](#architecture)
2. [Training Strategy](#training-strategy)
3. [How It Works](#how-it-works)
4. [Technical Details](#technical-details)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## Architecture

### Model: EfficientNet-B0
- **Parameters:** 5.3M (5x smaller than ResNet-50)
- **Pretrained:** ImageNet weights
- **Classifier:** Dropout(0.3) → Linear(1280 → 4)
- **Output:** 4 logits (one per attribute)

### Why EfficientNet-B0?
- Faster training (2-3x vs ResNet-50)
- Better accuracy-to-size ratio
- Lower memory usage
- Modern architecture (2019)

---

## Training Strategy

### Two-Stage Fine-Tuning

**Stage 1: Classifier Head (5 epochs)**
- Freeze all backbone layers
- Train only dropout + linear layer
- Learning rate: 1e-3
- Purpose: Adapt classifier to new task

**Stage 2: Partial Fine-Tuning (15 epochs)**
- Freeze first 5 blocks (generic features)
- Unfreeze last 11 blocks (task-specific)
- Learning rate: 1e-4 (lower for stability)
- Early stopping: patience=5

### Anti-Overfitting Techniques
1. **Dropout (0.3)** - Randomly drops neurons
2. **Weight Decay (1e-4)** - L2 regularization
3. **Data Augmentation** - 8 different transforms
4. **Early Stopping** - Stops when val loss plateaus
5. **LR Scheduling** - Reduces LR when stuck
6. **Partial Freezing** - Only trains relevant layers

---

## How It Works

### 1. Data Loading (`dataset.py`)

**Handles NA Values:**
```python
For each label:
  if value == "NA":
    label = 0.0
    mask = 0.0   # This position ignored in loss
  else:
    label = float(value)
    mask = 1.0   # This position used in loss
```

**Data Augmentation (Training):**
- Resize to 256x256, random crop to 224x224
- Random horizontal flip (50%)
- Random vertical flip (20%)
- Color jitter (brightness, contrast, saturation, hue)
- Random rotation (±20°)
- Random affine (translate 10%)
- Random erasing (20% - cutout)
- Normalize with ImageNet stats

**Validation:** Only resize to 224x224 and normalize

### 2. Model Architecture (`model.py`)

```
Input Image [3, 224, 224]
    ↓
EfficientNet-B0 Backbone (frozen initially)
    ↓
Features [1280]
    ↓
Dropout(p=0.3)
    ↓
Linear(1280 → 4)
    ↓
Output Logits [4]
    ↓
Sigmoid (during inference)
    ↓
Probabilities [4]
    ↓
Threshold at 0.5
    ↓
Binary Predictions [4]
```

### 3. Loss Function

**Masked BCE with Logits:**
```python
loss = BCEWithLogitsLoss(outputs, labels)  # Per-element
loss = loss * mask                          # Zero out NA
loss = loss.sum() / mask.sum()             # Average over known
```

**Class Imbalance Weights:**
- Attr1: 0.13 (common positive)
- Attr2: 0.24 (moderately common)
- Attr3: 0.89 (balanced)
- Attr4: 11.96 (rare positive - heavily weighted)

### 4. Training Loop (`train.py`)

**Each Epoch:**
1. Set model to train mode
2. For each batch:
   - Forward pass → get logits
   - Compute masked loss
   - Backward pass → compute gradients
   - Update weights
   - Track loss for plotting
3. Validate on val set
4. Compute accuracy metrics
5. Save if best model
6. Check early stopping

**Metrics Tracked:**
- Loss (for optimization)
- Hamming Accuracy (per-label accuracy)
- Exact Match Accuracy (all labels correct)
- Per-Attribute Accuracy (for each of 4 attributes)

### 5. Inference (`inference.py`)

```python
1. Load trained model
2. Load image and preprocess
3. Forward pass → logits
4. Apply sigmoid → probabilities
5. Threshold at 0.5 → binary
6. Return indices where prediction = 1
```

---

## Technical Details

### Hyperparameters

```python
CONFIG = {
    'batch_size': 32,           # Larger = faster training
    'val_split': 0.2,           # 80/20 train/val split
    'seed': 42,                 # Reproducibility
    
    'stage1_epochs': 5,         # Classifier training
    'stage1_lr': 1e-3,          # Initial learning rate
    
    'stage2_epochs': 15,        # Fine-tuning
    'stage2_lr': 1e-4,          # Lower LR for stability
    
    'dropout': 0.3,             # Regularization
    'weight_decay': 1e-4,       # L2 penalty
    'early_stop_patience': 5,   # Epochs to wait
}
```

### Optimizer: AdamW
- Decoupled weight decay (proper L2)
- Adaptive learning rates per parameter
- Momentum with bias correction

### Learning Rate Scheduling
- ReduceLROnPlateau
- Reduces LR by 0.5x when val loss plateaus
- Patience: 2 epochs
- Helps escape local minima

### Data Split
- Train: 780 images (80%)
- Validation: 195 images (20%)
- Random split with fixed seed (reproducible)

### Training Time
- CPU: ~25-30 minutes
- GPU: ~5-8 minutes

### Model Size
- Parameters: 5.3M
- File size: ~21MB
- Memory usage: ~500MB during training

---

## Configuration

### Adjust Training Speed

**Faster (less accurate):**
```python
'stage1_epochs': 3,
'stage2_epochs': 8,
'batch_size': 64,
```

**Slower (more accurate):**
```python
'stage1_epochs': 8,
'stage2_epochs': 20,
'batch_size': 16,
```

### Adjust Overfitting/Underfitting

**If Overfitting (val loss increases):**
```python
'dropout': 0.4,              # Increase
'weight_decay': 5e-4,        # Increase
'stage2_lr': 5e-5,           # Decrease
'early_stop_patience': 3,    # Decrease
```

**If Underfitting (both losses high):**
```python
'dropout': 0.2,              # Decrease
'weight_decay': 1e-5,        # Decrease
'stage2_lr': 2e-4,           # Increase
'stage2_epochs': 20,         # Increase
```

### Adjust Inference Threshold

```python
# In inference.py, change threshold
predict(image_path, threshold=0.7)  # More conservative
predict(image_path, threshold=0.3)  # More aggressive
```

---

## Troubleshooting

### Issue: Out of Memory
**Solution:**
```python
CONFIG['batch_size'] = 16  # or 8
```

### Issue: Training Too Slow
**Solutions:**
- Use GPU instead of CPU
- Increase batch size (if memory allows)
- Reduce epochs
- Use smaller image size (224 → 192)

### Issue: Poor Accuracy
**Solutions:**
- Train longer (more epochs)
- Lower dropout (0.3 → 0.2)
- Higher learning rate (1e-4 → 2e-4)
- More data augmentation
- Collect more training data

### Issue: Overfitting
**Solutions:**
- Increase dropout (0.3 → 0.4)
- Increase weight decay (1e-4 → 5e-4)
- More aggressive augmentation
- Early stopping with lower patience
- Reduce model capacity

### Issue: Model Not Learning
**Check:**
1. Data paths are correct
2. Labels are loaded properly
3. Learning rate not too low
4. Batch size not too small
5. Gradients are flowing (not all frozen)

### Issue: Inference Predictions Wrong
**Solutions:**
- Train longer for better accuracy
- Adjust threshold per attribute
- Check if model is loaded correctly
- Verify preprocessing matches training

---

## File Structure

```
project/
├── data/
│   ├── images/images/      # Dataset images
│   ├── labels.txt          # Annotations
│   ├── dataset.py          # Data loading (120 lines)
│   ├── model.py            # Model definition (50 lines)
│   └── train.py            # Training pipeline (270 lines)
├── inference.py            # Prediction script (70 lines)
├── model.pth               # Trained weights (~21MB)
├── loss_curve.png          # Training visualization
├── README.md               # Quick start guide
├── DOCUMENTATION.md        # This file
└── requirements.txt        # Dependencies
```

---

## Code Walkthrough

### dataset.py
- `get_transforms()` - Creates augmentation pipeline
- `MultilabelDataset` - Custom dataset class
  - `_load_labels()` - Parses labels.txt, handles NA
  - `__getitem__()` - Returns (image, label, mask)
- `get_dataloaders()` - Creates train/val loaders

### model.py
- `build_model()` - Creates EfficientNet-B0 with custom head
- `unfreeze_model()` - Unfreezes layers for fine-tuning
- `get_pos_weights()` - Returns class imbalance weights

### train.py
- `masked_bce_loss()` - Loss function with NA handling
- `compute_accuracy()` - Calculates metrics
- `train_one_epoch()` - One training epoch
- `validate()` - Validation with metrics
- `plot_loss_curve()` - Generates required plot
- `main()` - Full training pipeline

### inference.py
- `predict()` - Loads model and predicts attributes
- Command-line interface for easy testing

---

## Performance Expectations

### Accuracy Targets
- **Hamming Accuracy:** 70-85% (per-label)
- **Exact Match:** 40-60% (all 4 correct)
- **Per-Attribute:** 60-90% (varies by attribute)

### Why Not Higher?
- Small dataset (~975 images)
- NA values reduce training data
- Class imbalance (some attributes rare)
- Multilabel is harder than single-label

### Improvements for Production
1. Collect more training data
2. Use test-time augmentation
3. Ensemble multiple models
4. Optimize thresholds per attribute
5. Use focal loss for extreme imbalance
6. Add attention mechanisms
7. Cross-validation for robustness

---

## Key Formulas

### BCE with Logits Loss
```
L = -[y·log(σ(x)) + (1-y)·log(1-σ(x))]
where σ(x) = 1/(1+e^(-x))
```

### Masked Loss
```
L_masked = Σ(L · mask) / Σ(mask)
```

### Hamming Accuracy
```
Acc = (correct predictions) / (total known labels)
```

### Exact Match Accuracy
```
Acc = (samples with all labels correct) / (total samples)
```

---

## Dependencies

```
torch>=2.0.0          # Deep learning framework
torchvision>=0.15.0   # Pretrained models
Pillow>=9.0.0         # Image loading
matplotlib>=3.5.0     # Plotting
```

Install:
```bash
pip install torch torchvision pillow matplotlib
```

---

## Assignment Requirements Met

✅ Training code produces model weights
✅ Loss curve with exact labels (iteration_number, training_loss, Aimonk_multilabel_problem)
✅ Inference code prints attribute list
✅ Uses PyTorch framework
✅ Uses established architecture (EfficientNet-B0)
✅ Fine-tunes ImageNet weights
✅ Handles NA values (masking)
✅ Addresses class imbalance (positive weights)
✅ Clean and modular code
✅ Comprehensive documentation

---

## Summary

This implementation provides a complete solution for multilabel image classification with:
- Modern efficient architecture
- Proper handling of missing data
- Class imbalance correction
- Anti-overfitting techniques
- Comprehensive metrics
- Clean, modular code
- Full documentation

The model trains in ~25 minutes on CPU and achieves reasonable accuracy for a challenging multilabel problem with limited data.
