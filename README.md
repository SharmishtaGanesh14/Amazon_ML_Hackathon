# Amazon ML Challenge 2025 - Multimodal Product Price Prediction

A sophisticated multimodal machine learning pipeline that predicts product prices using both textual descriptions and product images. This solution combines advanced NLP techniques, computer vision, and ensemble modeling to achieve accurate price predictions.

## Competition Overview

This repository contains my solution for the Amazon ML Challenge 2025, which focused on predicting product prices using multimodal data (text + images). The challenge required handling real-world e-commerce data with varying quality text descriptions and product images.

## Project Structure & File Descriptions

### Core Pipeline Files

#### `enhanced_multimodal_context_pipeline.py`
**Purpose**: Feature engineering and context vector generation  
**What it does**:
- Extracts text from product images using EasyOCR with advanced preprocessing
- Combines original product descriptions with OCR-extracted text
- Generates SBERT semantic embeddings for text understanding
- Applies TF-IDF with SVD dimensionality reduction for traditional NLP features
- Extracts EfficientNet-V2-S features from product images
- Processes numeric features (quantities, weights, volumes, pack counts)
- Outputs compressed `.npz` files with combined multimodal features

**Key Classes**:
- `EfficientOCR`: OCR processing with image enhancement
- `EnhancedTextFeatureExtractor`: Multi-source text processing
- `OfflineImageFeatureExtractor`: CNN-based image feature extraction
- `MultimodalEncoder`: Neural network for feature fusion

#### `multimodal_pricing_pipeline.py`
**Purpose**: Model training and price prediction  
**What it does**:
- Loads preprocessed features from context vector files
- Trains LightGBM gradient boosting model with GPU acceleration
- Trains custom MLP with SMAPE loss optimization
- Creates ensemble predictions using Ridge regression stacker
- Outputs final price predictions for submission

**Key Classes**:
- `SMAPELoss`: Custom loss function for price prediction
- `MultimodalMLP`: Deep neural network with batch normalization
- Ensemble prediction functions for model combination

### Advanced Variations

#### `optimized2.py`
**Purpose**: Hyperparameter-optimized pipeline with automated tuning  
**What it does**:
- Implements Optuna-based hyperparameter optimization for both LightGBM and neural networks
- Includes advanced logging with TensorBoard and Weights & Biases integration
- Features tunable neural architecture with dynamic layer configuration
- Performs systematic optimization with 50-100+ trials per model type
- Saves comprehensive training metrics and best hyperparameters

**Key Features**:
- `TunableMultimodalMLP`: Dynamic neural architecture
- `MetricsLogger`: Advanced experiment tracking
- Automated hyperparameter search functions
- JSON-based configuration management

#### `model_with_noocr_fushion.py`
**Purpose**: Simplified model without OCR complexity, focusing on advanced fusion  
**What it does**:
- Processes text and image features without OCR extraction
- Implements residual neural architecture with skip connections
- Uses advanced preprocessing with RobustScaler + QuantileTransformer
- Trains models in log-space for better price prediction convergence
- Employs GradientBoostingRegressor as nonlinear meta-learner

**Key Features**:
- `OptimizedMLP`: Residual architecture with skip connections
- Dual-stage feature scaling approach
- Stratified data splitting based on price buckets
- Nonlinear ensemble stacking

## Dependencies & Setup

### Required Packages
```bash
pip install numpy pandas scikit-learn lightgbm torch torchvision
pip install sentence-transformers easyocr opencv-python pillow tqdm
pip install transformers datasets optuna wandb tensorboard
```

### Pre-trained Models Required
- **SBERT Model**: `all-MiniLM-L6-v2` (place in `./all-MiniLM-L6-v2_local/`)
- **EfficientNet Weights**: `efficientnet_v2_s.pth` (place in `./pretrained_weights/`)

### Dataset Structure
```
dataset/
├── train.csv                 # Training data with sample_id, price, descriptions
├── test.csv                  # Test data for predictions  
├── train_images/             # Training product images
├── test_images/              # Test product images
└── generated_files/          # Output feature files
    ├── train_context_vectors.npz
    ├── test_context_vectors.npz
    └── *_predictions.csv
```

## Usage Instructions

### Method 1: Standard Pipeline (Recommended)
```bash
# Step 1: Generate multimodal features
python enhanced_multimodal_context_pipeline.py

# Step 2: Train models and generate predictions
python multimodal_pricing_pipeline.py
```

### Method 2: Hyperparameter Optimized Pipeline
```bash
# Run optimized pipeline with automated tuning
python optimized2.py
```

### Method 3: No-OCR Simplified Pipeline
```bash
# Run without OCR complexity
python model_with_noocr_fushion.py
```

## Output Files

| File | Description |
|------|-------------|
| `multimodal_predictions.csv` | Standard pipeline predictions |
| `optimized_multimodal_predictions.csv` | Hyperparameter-tuned predictions |
| `train_context_vectors.npz` | Training features (text + image + OCR) |
| `test_context_vectors.npz` | Test features for prediction |
| `best_hyperparameters.json` | Optimal hyperparameters from tuning |
| `training_metrics.json` | Training history and validation scores |

## Model Architecture Overview

### Two-Stage Pipeline

**Stage 1: Feature Engineering**
- **OCR Text Extraction**: EasyOCR with image preprocessing (CLAHE, denoising, scaling)
- **Advanced Text Processing**: Combines original descriptions with OCR-extracted text
- **SBERT Embeddings**: Semantic text representations using Sentence Transformers
- **TF-IDF + SVD**: Traditional NLP features with dimensionality reduction
- **Image Features**: EfficientNet-V2-S for deep visual representations
- **Numeric Features**: Extracted quantities, weights, volumes, and pack counts

**Stage 2: Price Prediction**
- **LightGBM Model**: Gradient boosting with GPU acceleration
- **Deep Neural Network**: Custom MLP with batch normalization and dropout
- **SMAPE Loss Function**: Optimized for symmetric percentage error
- **Ensemble Learning**: Ridge regression stacker for final predictions

## Code Highlights

### Efficient OCR Processing
```python
class EfficientOCR:
    def preprocess_image_for_ocr(self, image_path):
        # CLAHE enhancement + denoising + scaling
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced)
```

### Custom SMAPE Loss
```python
class SMAPELoss(nn.Module):
    def forward(self, y_pred, y_true):
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
        numerator = torch.abs(y_pred - y_true)
        return torch.mean(numerator / denominator) * 100
```

### Residual Neural Architecture
```python
class OptimizedMLP(nn.Module):
    def forward(self, x):
        x1 = self.act(self.bn1(self.fc1(x)))           # Base layer
        x2 = self.act(self.bn2(self.fc2(x1))) + 0.2*x1 # Residual connection
        x3 = self.act(self.fc3(x2)) + 0.2*x2           # Residual connection
        out = torch.clamp(self.fc4(x3).squeeze(1), min=1e-6)
        return out
```

## Model Performance

The ensemble approach combines:
- **LightGBM**: Handles tabular features and feature interactions effectively
- **Deep Neural Network**: Captures complex non-linear patterns in multimodal data
- **Meta-learner**: Ridge/GradientBoosting stacker for optimal weight combination

Performance is evaluated using SMAPE (Symmetric Mean Absolute Percentage Error), which is well-suited for price prediction tasks.

## Technical Innovations

1. **Multimodal Feature Fusion**: Seamlessly combines text, image, and numeric features
2. **OCR Enhancement**: Extracts additional context from product images
3. **Custom Loss Functions**: SMAPE optimization specifically for price prediction
4. **Advanced Preprocessing**: Robust scaling, outlier handling, and feature engineering
5. **Ensemble Architecture**: Multiple stacking strategies with automated hyperparameter tuning
6. **Memory Optimization**: Efficient batch processing and garbage collection
7. **Residual Connections**: Skip connections for better gradient flow in deep networks

## Experimental Variations

| Model Variant | Key Innovation | Complexity | OCR | Hyperparameter Tuning | Ensemble Method |
|---------------|---------------|------------|-----|---------------------|-----------------|
| **Enhanced Multimodal** | OCR + Advanced Feature Engineering | High | ✅ | Manual | Ridge Stacker |
| **Hyperparameter Optimized** | Automated Optuna Tuning | Very High | ✅ | Automated (100+ trials) | Ridge Stacker |
| **No-OCR Fusion** | Residual Architecture + Advanced Scaling | Medium | ❌ | Manual | GradientBoosting Stacker |

## Reproducibility

- **Fixed Random Seeds**: All models use consistent seeds (NumPy: 42, PyTorch: 42)
- **Deterministic Operations**: Reproducible results across different runs
- **Configuration Management**: JSON-based hyperparameter storage and loading
- **Version Control**: Clear dependency specifications and environment setup
- **Comprehensive Logging**: Training metrics, validation curves, model checkpoints

## Quick Start

1. **Clone repository and install dependencies**:
   ```bash
   git clone [your-repo-url]
   cd amazon-ml-challenge-2025
   pip install -r requirements.txt
   ```

2. **Download pre-trained models**:
   - SBERT model: `all-MiniLM-L6-v2`
   - EfficientNet weights: `efficientnet_v2_s.pth`

3. **Prepare dataset** in required structure (see Dataset Structure above)

4. **Run standard pipeline**:
   ```bash
   python enhanced_multimodal_context_pipeline.py
   python multimodal_pricing_pipeline.py
   ```

5. **Check predictions** in generated CSV files

## Future Improvements

- [ ] Implement cross-validation for more robust validation
- [ ] Add data augmentation techniques for images
- [ ] Experiment with transformer-based vision models (ViT)
- [ ] Implement advanced ensemble techniques (stacking, blending)
- [ ] Add hyperparameter optimization using Bayesian methods
- [ ] Integrate real-time inference capabilities

## Contact

Feel free to reach out for discussions about multimodal ML, price prediction, ensemble methods, or competition strategies!

---
*Developed for Amazon ML Challenge 2025 - Showcasing advanced multimodal machine learning techniques*
