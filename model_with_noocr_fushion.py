# ──────────── optimized_multimodal_pricing_pipeline.py ────────────
import os, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ── SMAPE Loss Function ───────────────────────────────────────
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

def clean_features(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    var_mask = np.var(X, axis=0) > 1e-10
    X = X[:, var_mask]
    q1, q99 = np.percentile(X, [1, 99], axis=0)
    X = np.clip(X, q1, q99)
    return X

# ── Load Data ───────────────────────────────────────────────
def load_train_data(npz_path, csv_path):
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
    data = np.load(npz_path, allow_pickle=True)
    X_test = data["features"].astype("float32")
    ids = data["sample_ids"].astype(str)
    X_test = clean_features(X_test)
    return X_test, ids

# ── Optimized MLP ───────────────────────────────────────────
class OptimizedMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_bn(x)
        x1 = self.act(self.bn1(self.fc1(x)))  # residual
        x2 = self.act(self.bn2(self.fc2(x1))) + 0.2*x1
        x3 = self.act(self.fc3(x2)) + 0.2*x2
        out = torch.clamp(self.fc4(x3).squeeze(1), min=1e-6)
        return out

def train_multimodal_cnn(X_train, y_train, X_val, y_val, epochs=80, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = min(512, len(X_train)//10) if device.type=='cuda' else min(128,len(X_train)//10)

    # Scale numeric features separately
    scaler = RobustScaler()
    qt = QuantileTransformer(output_distribution="normal")
    X_train_scaled = qt.fit_transform(scaler.fit_transform(X_train)).astype(np.float32)
    X_val_scaled = qt.transform(scaler.transform(X_val)).astype(np.float32)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(np.log1p(y_train))),
                              batch_size=batch_size, shuffle=True, pin_memory=(device.type=='cuda'))
    X_val_tensor = torch.from_numpy(X_val_scaled).to(device)
    y_val_tensor = torch.from_numpy(np.log1p(y_val)).to(device)

    model = OptimizedMLP(X_train_scaled.shape[1]).to(device)
    loss_fn = SMAPELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=6, factor=0.5, min_lr=1e-6)

    best_smape = float('inf')
    best_state = None
    patience_counter = 0
    patience_limit = 12

    for epoch in range(1, epochs+1):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(torch.expm1(model(batch_X)), torch.expm1(batch_y))  # revert log
            loss.backward()
            optimizer.step()

        if epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = torch.expm1(model(X_val_tensor)).cpu().numpy()
            val_smape = smape(y_val, val_pred)
            print(f"Epoch {epoch:03d} - CNN Validation SMAPE: {val_smape:.4f}")
            scheduler.step(val_smape)
            if val_smape < best_smape:
                best_smape = val_smape
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience_limit:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.feature_scaler = (scaler, qt)
    model.eval()
    return model

# ── LightGBM with SMAPE focus ───────────────────────────────
def train_lightgbm(X_train, y_train, X_val=None, y_val=None):
    train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
    valid_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data) if X_val is not None else None
    params = {'objective':'regression','metric':'l1','boosting_type':'gbdt',
              'learning_rate':0.03,'num_leaves':127,'feature_fraction':0.8,
              'bagging_fraction':0.85,'bagging_freq':1,'min_data_in_leaf':25,
              'lambda_l1':0.05,'lambda_l2':0.05,'min_gain_to_split':0.01,'seed':42,'verbosity':-1}
    if torch.cuda.is_available():
        params.update({'device_type':'gpu','gpu_platform_id':0,'gpu_device_id':0,'max_bin':63})
    model = lgb.train(params, train_data, valid_sets=[valid_data] if valid_data else None,
                      num_boost_round=3000, callbacks=[lgb.early_stopping(150)] if valid_data else None)
    if X_val is not None and y_val is not None:
        val_pred = np.expm1(model.predict(X_val))
        print(f"LightGBM Validation SMAPE: {smape(y_val, val_pred):.4f}")
    return model

# ── Nonlinear Ensemble ─────────────────────────────────────
def create_ensemble_predictions(lgb_model, cnn_model, X_val, y_val=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lgb_pred = np.expm1(lgb_model.predict(X_val))
    scaler, qt = cnn_model.feature_scaler
    X_scaled = qt.transform(scaler.transform(X_val)).astype(np.float32)
    with torch.no_grad():
        cnn_pred = torch.expm1(cnn_model(torch.from_numpy(X_scaled).to(device))).cpu().numpy()

    meta_features = np.column_stack([lgb_pred, cnn_pred])
    # Use small GradientBoostingRegressor as nonlinear stacker
    stacker = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1)
    if y_val is not None:
        stacker.fit(meta_features, y_val)
    else:
        stacker.fit(meta_features, lgb_pred)
    ensemble_pred = stacker.predict(meta_features)

    if y_val is not None:
        print(f"Ensemble Validation SMAPE: {smape(y_val, ensemble_pred):.4f}")
    return {'lgb': lgb_pred, 'cnn': cnn_pred, 'ensemble': ensemble_pred}

# ── Main Pipeline ─────────────────────────────────────────────
def main():
    dataset_dir = "./dataset"
    X_full, y_full = load_train_data(os.path.join(dataset_dir,"train_features_no_ocr.npz"),
                                     os.path.join(dataset_dir,"train.csv"))
    X_test, test_ids = load_test_data(os.path.join(dataset_dir,"test_features_no_ocr.npz"))

    # Stratified split based on log-price buckets
    bins = np.log1p(y_full)
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42,
                                                      stratify=pd.qcut(bins, q=10, duplicates='drop'))

    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    cnn_model = train_multimodal_cnn(X_train, y_train, X_val, y_val)

    _ = create_ensemble_predictions(lgb_model, cnn_model, X_val, y_val)
    predictions = create_ensemble_predictions(lgb_model, cnn_model, X_test)

    output_df = pd.DataFrame({
        'sample_id': test_ids,
        'lgb_prediction': predictions['lgb'],
        'cnn_prediction': predictions['cnn'],
        'ensemble_prediction': predictions['ensemble']
    })
    output_df.to_csv("optimized_multimodal_predictions.csv", index=False)
    print("Predictions saved to optimized_multimodal_predictions.csv")

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()
