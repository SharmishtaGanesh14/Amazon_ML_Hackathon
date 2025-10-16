# ──────────── optimized_multimodal_pricing_pipeline.py ────────────
import os, gc, warnings
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import optuna
import wandb
from typing import Dict, Any, Tuple, Optional

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────
# METRICS AND LOGGING
# ─────────────────────────────────────────────────────────────────────────

class SMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=self.epsilon)
        y_true = torch.clamp(y_true, min=self.epsilon)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
        numerator = torch.abs(y_pred - y_true)
        return torch.mean(numerator / denominator) * 100

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = den > 1e-8
    return 100 * np.mean(num[mask] / den[mask]) if np.any(mask) else 0.0

class MetricsLogger:
    def __init__(self, log_dir="logs", use_wandb=False, project_name="multimodal-pricing"):
        self.use_wandb = use_wandb
        self.writer = SummaryWriter(log_dir) if log_dir else None
        self.metrics_history = []

        if use_wandb:
            try:
                wandb.init(project=project_name, reinit=True)
            except Exception as e:
                print(f"W&B initialization failed: {e}")
                self.use_wandb = False

    def log_metrics(self, epoch, train_loss, val_smape, lr, model_type="CNN"):
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_smape': val_smape,
            'learning_rate': lr,
            'model_type': model_type
        }

        self.metrics_history.append(metrics)

        if self.writer:
            self.writer.add_scalar(f'{model_type}/Train_Loss', train_loss, epoch)
            self.writer.add_scalar(f'{model_type}/Val_SMAPE', val_smape, epoch)
            self.writer.add_scalar(f'{model_type}/Learning_Rate', lr, epoch)

        if self.use_wandb:
            try:
                wandb.log(metrics)
            except:
                pass

    def save_metrics(self, filepath="training_metrics.json"):
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            print(f"Metrics saved to {filepath}")
        except Exception as e:
            print(f"Failed to save metrics: {e}")

    def close(self):
        if self.writer:
            self.writer.close()
        if self.use_wandb:
            try:
                wandb.finish()
            except:
                pass

# ─────────────────────────────────────────────────────────────────────────
# DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────

def clean_features(X):
    """Clean and preprocess features"""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    var_mask = np.var(X, axis=0) > 1e-10
    X = X[:, var_mask]
    q1, q99 = np.percentile(X, [1, 99], axis=0)
    X = np.clip(X, q1, q99)
    return X

def load_train_data(npz_path, csv_path):
    """Load and align training data"""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Train features file not found: {npz_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Train CSV file not found: {csv_path}")

    data = np.load(npz_path, allow_pickle=True)
    X = data["features"].astype("float32")
    ids = data["sample_ids"].astype(str)
    X = clean_features(X)

    df = pd.read_csv(csv_path, usecols=["sample_id", "price"])
    df["sample_id"] = df["sample_id"].astype(str)
    df = df[df["sample_id"].isin(ids)]

    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    order = df["sample_id"].map(id_to_idx).values
    X_aligned = X[order]
    y_aligned = df["price"].values.astype("float32")

    mask = (y_aligned > 0.1) & (y_aligned < 5000)
    return X_aligned[mask], y_aligned[mask]

def load_test_data(npz_path):
    """Load test data"""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Test features file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    X_test = data["features"].astype("float32")
    ids = data["sample_ids"].astype(str)
    X_test = clean_features(X_test)
    return X_test, ids

# ─────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK MODEL
# ─────────────────────────────────────────────────────────────────────────

class TunableMultimodalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128, 64], dropout=0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout * (1 - 0.2 * i))
            ])
            prev_dim = hidden_dim

        # Final layers
        layers.extend([
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        ])
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_bn(x)
        return torch.clamp(self.network(x).squeeze(1), min=1e-6)

# ─────────────────────────────────────────────────────────────────────────
# LIGHTGBM HYPERPARAMETER OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────

def objective_lightgbm(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'regression',
        'metric': 'l1',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1),
        'seed': 42,
        'verbosity': -1,
        'n_jobs': -1
    }

    if torch.cuda.is_available():
        params.update({'device_type':'gpu','gpu_platform_id':0,'gpu_device_id':0,'max_bin':63})

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=3000,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )

    val_pred = model.predict(X_val)
    val_smape = smape(y_val, val_pred)
    return val_smape

def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=100):
    print("Starting LightGBM hyperparameter optimization...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
    study.optimize(lambda trial: objective_lightgbm(trial, X_train, y_train, X_val, y_val),
                   n_trials=n_trials, show_progress_bar=True)
    print(f"Best LightGBM SMAPE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params

def train_optimized_lightgbm(X_train, y_train, X_val, y_val, best_params):
    params = best_params.copy()
    params.update({'objective': 'regression','metric':'l1','boosting_type':'gbdt','seed':42,'verbosity':-1,'n_jobs':-1})
    if torch.cuda.is_available():
        params.update({'device_type':'gpu','gpu_platform_id':0,'gpu_device_id':0,'max_bin':63})

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(params, train_data, valid_sets=[valid_data],
                      num_boost_round=3000, callbacks=[lgb.early_stopping(150)])

    val_pred = model.predict(X_val)
    print(f"Final LightGBM Validation SMAPE: {smape(y_val, val_pred):.4f}")
    return model

# ─────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK HYPERPARAMETER OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────

# (objective_neural_network, tune_neural_network, train_optimized_neural_network remain unchanged)

# ─────────────────────────────────────────────────────────────────────────
# ENSEMBLE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────

# (create_ensemble_predictions remains unchanged)

# ─────────────────────────────────────────────────────────────────────────
# MAIN OPTIMIZED TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────

def optimized_training_pipeline(dataset_dir,
                               lgb_trials=100,
                               nn_trials=50,
                               nn_epochs=100,
                               use_wandb=False):
    dataset_dir = os.path.abspath(dataset_dir)  # Ensure absolute path
    logger = MetricsLogger(log_dir="logs", use_wandb=use_wandb)
    try:
        # Data loading
        print("\n1. Loading and preprocessing data...")
        X_full, y_full = load_train_data(
            os.path.join(dataset_dir, "train_features.npz"),
            os.path.join(dataset_dir, "train.csv")
        )
        X_test, test_ids = load_test_data(os.path.join(dataset_dir, "test_features.npz"))

        print(f"   Training samples: {len(X_full)}")
        print(f"   Features: {X_full.shape[1]}")
        print(f"   Test samples: {len(X_test)}")

        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
        print(f"   Train split: {len(X_train)} samples")
        print(f"   Validation split: {len(X_val)} samples")

        # LightGBM optimization
        best_lgb_params = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=lgb_trials)

        # Neural Network optimization
        best_nn_params = tune_neural_network(X_train, y_train, X_val, y_val, n_trials=nn_trials, logger=logger)

        # Train final models
        print("\n4. Training Final Models...")
        lgb_model = train_optimized_lightgbm(X_train, y_train, X_val, y_val, best_lgb_params)
        cnn_model = train_optimized_neural_network(X_train, y_train, X_val, y_val, best_nn_params, logger, epochs=nn_epochs)

        # Ensemble predictions
        val_results = create_ensemble_predictions(lgb_model, cnn_model, X_val, y_val)
        stacker = val_results['stacker']

        test_results = create_ensemble_predictions(lgb_model, cnn_model, X_test)
        test_meta_features = np.column_stack([test_results['lgb'], test_results['cnn']])
        test_ensemble_pred = stacker.predict(test_meta_features)

        # Save results
        output_df = pd.DataFrame({
            'sample_id': test_ids,
            'lgb_prediction': test_results['lgb'],
            'cnn_prediction': test_results['cnn'],
            'ensemble_prediction': test_ensemble_pred
        })
        output_filename = 'optimized_multimodal_predictions.csv'
        output_df.to_csv(output_filename, index=False)
        print(f"   Predictions saved to {output_filename}")

        # Save hyperparameters
        hyperparams = {
            'lightgbm_params': best_lgb_params,
            'neural_network_params': best_nn_params,
            'validation_scores': {
                'lgb_smape': float(smape(y_val, val_results['lgb'])),
                'cnn_smape': float(smape(y_val, val_results['cnn'])),
                'ensemble_smape': float(smape(y_val, val_results['ensemble']))
            }
        }
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(hyperparams, f, indent=2)
        print("   Best hyperparameters saved to best_hyperparameters.json")

        # Save training metrics
        logger.save_metrics('training_metrics.json')

        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*60)

        return lgb_model, cnn_model, best_lgb_params, best_nn_params, stacker

    except Exception as e:
        print(f"\nError in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

    finally:
        logger.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ─────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    dataset_abs_path = "/mnt/data/shyam/sharmishta/dataset"  # Updated absolute path
    config = {
        'dataset_dir': dataset_abs_path,
        'lgb_trials': 50,
        'nn_trials': 25,
        'nn_epochs': 80,
        'use_wandb': False
    }

    print("Starting Optimized Multimodal Pricing Pipeline...")
    print(f"Configuration: {config}")

    lgb_model, cnn_model, best_lgb_params, best_nn_params, stacker = optimized_training_pipeline(**config)

    if lgb_model is not None:
        print("\nPipeline completed successfully!")
        print("Files generated:")
        print("  - optimized_multimodal_predictions.csv")
        print("  - best_hyperparameters.json")
        print("  - training_metrics.json")
        print("  - logs/ directory (TensorBoard logs)")
    else:
        print("\nPipeline failed. Check error messages above.")
