# Amazon ML Challenge 2025 - Multimodal Product Price Prediction

A sophisticated multimodal machine learning pipeline that predicts product prices using both textual descriptions and product images. This solution combines advanced NLP techniques, computer vision, and ensemble modeling to achieve accurate price predictions.

## 🏆 Competition Overview

This repository contains my solution for the Amazon ML Challenge 2025, which focused on predicting product prices using multimodal data (text + images). The challenge required handling real-world e-commerce data with varying quality text descriptions and product images.

## 📁 Project Structure & File Descriptions

### 🔧 Core Pipeline Files

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

### 🚀 Advanced Variations

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

## 🛠 Dependencies & Setup

### Required Packages
