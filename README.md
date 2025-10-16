# Amazon ML Challenge 2025 - Multimodal Product Price Prediction

A sophisticated multimodal machine learning pipeline that predicts product prices using both textual descriptions and product images. This solution combines advanced NLP techniques, computer vision, and ensemble modeling to achieve accurate price predictions.

## 🏆 Competition Overview

This repository contains my solution for the Amazon ML Challenge 2025, which focused on predicting product prices using multimodal data (text + images). The challenge required handling real-world e-commerce data with varying quality text descriptions and product images.

## 🚀 Approach & Architecture

### Two-Stage Pipeline

**Stage 1: Feature Engineering (`enhanced_multimodal_context_pipeline.py`)**
- **OCR Text Extraction**: Uses EasyOCR with image preprocessing (CLAHE, denoising, scaling)
- **Advanced Text Processing**: Combines original descriptions with OCR-extracted text
- **SBERT Embeddings**: Semantic text representations using Sentence Transformers
- **TF-IDF + SVD**: Traditional NLP features with dimensionality reduction
- **Image Features**: EfficientNet-V2-S for deep visual representations
- **Numeric Features**: Extracted quantities, weights, volumes, and pack counts

**Stage 2: Price Prediction (`multimodal_pricing_pipeline.py`)**
- **LightGBM Model**: Gradient boosting with GPU acceleration
- **Deep Neural Network**: Custom MLP with batch normalization and dropout
- **SMAPE Loss Function**: Optimized for symmetric percentage error
- **Ensemble Learning**: Ridge regression stacker for final predictions

## 🛠 Dependencies

```bash
pip install numpy pandas scikit-learn lightgbm torch torchvision
pip install sentence-transformers easyocr opencv-python pillow tqdm
pip install transformers datasets
```

### Pre-trained Models Required

- **SBERT Model**: `all-MiniLM-L6-v2` (place in `./all-MiniLM-L6-v2_local/`)
- **EfficientNet Weights**: `efficientnet_v2_s.pth` (place in `./pretrained_weights/`)

## 📁 Dataset Structure

```
dataset/
├── train.csv                 # Training data with sample_id, price, descriptions
├── test.csv                  # Test data for predictions  
├── train_images/             # Training product images
├── test_images/              # Test product images
└── generated_files/          # Output feature files
    ├── train_context_vectors.npz
    ├── test_context_vectors.npz
    └── multimodal_predictions.csv
```

## 🔧 Usage

### Step 1: Feature Engineering
```bash
python enhanced_multimodal_context_pipeline.py
```
This generates:
- `train_context_vectors.npz`: Combined text+image+OCR features for training
- `test_context_vectors.npz`: Features for test set

### Step 2: Model Training & Prediction
```bash
python multimodal_pricing_pipeline.py
```
This outputs:
- `multimodal_predictions.csv`: Final ensemble predictions

## 🎯 Key Features

### Advanced Text Processing
- **Multi-source text combination**: Original descriptions + OCR content
- **Regex-based feature extraction**: Pack counts, weights, volumes
- **Semantic embeddings**: SBERT for contextual understanding
- **Traditional NLP**: TF-IDF with SVD dimensionality reduction

### Computer Vision
- **Robust image preprocessing**: Handles missing/corrupted images
- **EfficientNet feature extraction**: State-of-the-art CNN architecture
- **OCR integration**: Extracts text from product images

### Machine Learning
- **Ensemble approach**: LightGBM + Deep Neural Network
- **SMAPE optimization**: Custom loss function for price prediction
- **Advanced regularization**: Dropout, batch normalization, early stopping
- **GPU acceleration**: Optimized for CUDA when available

## 📊 Model Performance

The ensemble model combines:
- **LightGBM**: Handles tabular features and interactions effectively
- **Deep Neural Network**: Captures complex non-linear patterns
- **Ridge Stacker**: Meta-learner for optimal weight combination

Performance is evaluated using SMAPE (Symmetric Mean Absolute Percentage Error), which is well-suited for price prediction tasks.

## 🔍 Code Highlights

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

## 💡 Technical Innovations

1. **Multimodal Feature Fusion**: Seamlessly combines text, image, and numeric features
2. **OCR Enhancement**: Extracts additional context from product images
3. **Robust Preprocessing**: Handles real-world data quality issues
4. **Ensemble Architecture**: Combines tree-based and neural approaches
5. **Memory Optimization**: Efficient batch processing and garbage collection

## 🚀 Future Improvements

- [ ] Implement cross-validation for more robust validation
- [ ] Add data augmentation for images
- [ ] Experiment with transformer-based vision models
- [ ] Implement advanced ensemble techniques (stacking, blending)
- [ ] Add hyperparameter optimization using Optuna

## 📧 Contact

Feel free to reach out for discussions about multimodal ML, price prediction, or competition strategies!

---
*Developed for Amazon ML Challenge 2025 - Showcasing advanced multimodal machine learning techniques*
