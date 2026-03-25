"""
model_t2.py — Skipper NDT — Tâche 2 : Largeur de la carte magnétique
======================================================================
Régression de width_m à partir de features physiques extraites des NPZ.

Pipeline :
  NPZ → features physiques (24 features) → StandardScaler → MLP PyTorch

Architecture MLP :
  Linear(24→128) → ReLU → Dropout(0.3)
  Linear(128→64) → ReLU → Dropout(0.2)
  Linear(64→32)  → ReLU
  Linear(32→1)   → régression

Objectif : MAE < 1m

Fichiers sauvegardés :
  task2_model.pth     — poids du MLP
  task2_scaler.pkl    — StandardScaler
  training_curves_t2.png
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = r"D:/Projet_skipper_RNDT/Projet_Skipper/Data_NDT/Training_database_float16"
CSV_PATH   = r"D:/Projet_skipper_RNDT/Projet_Skipper/Data_NDT/Training_database_float16/pipe_presence_width_detection_label.csv"
OUTPUT_DIR = r"D:/Projet_skipper_RNDT/Projet_Skipper/models"
SEED       = 42
EPOCHS     = 300
BATCH_SIZE = 32
LR         = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Extraction de features physiques
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(file_path: str) -> list:
    """
    Extrait 24 features physiques depuis un fichier NPZ.
    Ces features décrivent la forme du signal magnétique
    pour prédire width_m.
    """
    raw  = np.load(file_path)['data']
    data = raw.astype(np.float32)          # float16 → float32 avant nan_to_num
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    Bx   = data[:, :, 0]
    By   = data[:, :, 1]
    Bz   = data[:, :, 2]
    Norm = data[:, :, 3]
    mask = Norm > 0

    features = {}

    # Features globales Norm
    norm_valid = Norm[mask]
    features['norm_max']    = float(norm_valid.max())
    features['norm_mean']   = float(norm_valid.mean())
    features['norm_std']    = float(norm_valid.std())
    features['norm_median'] = float(np.median(norm_valid))

    # Features globales Bz
    bz_valid = Bz[mask]
    features['bz_max']   = float(bz_valid.max())
    features['bz_mean']  = float(bz_valid.mean())
    features['bz_std']   = float(bz_valid.std())
    features['bz_range'] = float(bz_valid.max() - bz_valid.min())

    # Features globales Bx
    bx_valid = Bx[mask]
    features['bx_max']   = float(bx_valid.max())
    features['bx_std']   = float(bx_valid.std())
    features['bx_range'] = float(bx_valid.max() - bx_valid.min())

    # Features globales By
    by_valid = By[mask]
    features['by_max']   = float(by_valid.max())
    features['by_std']   = float(by_valid.std())
    features['by_range'] = float(by_valid.max() - by_valid.min())

    # Dimensions spatiales
    features['n_valid_pixels'] = float(mask.sum())
    features['height']         = float(data.shape[0])
    features['width_pixels']   = float(data.shape[1])
    features['aspect_ratio']   = float(data.shape[0] / (data.shape[1] + 1e-6))

    # Profil moyen
    col_means = Norm.mean(axis=0)
    row_means = Norm.mean(axis=1)
    features['profile_width_cols'] = float((col_means > col_means.max() * 0.5).sum())
    features['profile_width_rows'] = float((row_means > row_means.max() * 0.5).sum())

    # Gradient spatial
    grad_x = np.gradient(Norm, axis=1)
    grad_y = np.gradient(Norm, axis=0)
    grad_m = np.sqrt(grad_x**2 + grad_y**2)
    features['grad_mean'] = float(grad_m[mask].mean())
    features['grad_max']  = float(grad_m[mask].max())

    # Ratio couverture + dimensions physiques
    features['coverage_ratio'] = float(mask.mean())
    features['height_m']       = float(data.shape[0] * 0.2)
    features['width_pixels_m'] = float(data.shape[1] * 0.2)

    return list(features.values())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Architecture MLP
# ─────────────────────────────────────────────────────────────────────────────

class WidthMLP(nn.Module):
    """MLP pour régression sur features physiques."""
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Courbes d'entraînement
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(history: dict, output_dir: str):
    BG, TEXT = '#0f1117', '#e2e8f0'
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    def style(ax, title):
        ax.set_facecolor('#1e2130')
        ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold')
        ax.tick_params(colors=TEXT)
        for sp in ax.spines.values(): sp.set_edgecolor('#374151')
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.legend(facecolor='#1e2130', labelcolor=TEXT)

    axes[0].plot(epochs_range, history['train_loss'], color='#3b82f6',
                 label='Train Loss', linewidth=2)
    axes[0].plot(epochs_range, history['val_loss'],   color='#ef4444',
                 label='Val Loss',   linewidth=2)
    style(axes[0], 'Loss (SmoothL1)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].plot(epochs_range, history['train_mae'], color='#3b82f6',
                 label='Train MAE', linewidth=2)
    axes[1].plot(epochs_range, history['val_mae'],   color='#ef4444',
                 label='Val MAE',   linewidth=2)
    axes[1].axhline(1.0, color='#10b981', linestyle='--', linewidth=2,
                    label='Objectif MAE < 1m')
    style(axes[1], 'MAE (metres)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (m)')

    fig.suptitle('Courbes Entraînement — MLP Tâche 2',
                 color=TEXT, fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves_t2.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Courbes → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train_t2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*55}")
    print("ENTRAÎNEMENT T2 — Features physiques + MLP")
    print(f"  Device : {DEVICE}")
    print(f"{'='*55}")

    # Chargement CSV — label=1 uniquement
    df = pd.read_csv(CSV_PATH, sep=';')
    if 'field_file' in df.columns:
        df = df.rename(columns={'field_file': 'filename'})
    df_t2 = df[df['label'] == 1].reset_index(drop=True)
    print(f"\nDataset T2 : {len(df_t2)} échantillons")
    print(f"width_m    : {df_t2['width_m'].min():.1f}m → "
          f"{df_t2['width_m'].max():.1f}m  "
          f"(moy={df_t2['width_m'].mean():.1f}m)")

    # Extraction features
    print("\nExtraction des features physiques...")
    X_list, y_list = [], []
    for i, row in df_t2.iterrows():
        fpath = os.path.join(DATA_DIR, row['filename'])
        try:
            X_list.append(extract_features(fpath))
            y_list.append(float(row['width_m']))
        except Exception as e:
            print(f"  ✗ {row['filename']} : {e}")
        if len(X_list) % 200 == 0 and len(X_list) > 0:
            print(f"  {len(X_list)}/{len(df_t2)} traités...")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"  Shape X : {X.shape}  ({X.shape[1]} features)")

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED)
    print(f"Split : Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # StandardScaler
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # DataLoaders
    def make_loader(X_, y_, shuffle):
        ds = TensorDataset(
            torch.tensor(X_, dtype=torch.float32),
            torch.tensor(y_, dtype=torch.float32))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    # Modèle
    n_features = X_train.shape[1]
    model      = WidthMLP(n_features).to(DEVICE)
    criterion  = nn.SmoothL1Loss()
    optimizer  = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)
    print(f"MLP : {sum(p.numel() for p in model.parameters()):,} params")

    # Entraînement
    model_path    = os.path.join(OUTPUT_DIR, 'task2_model.pth')
    best_val_loss = float('inf')
    history       = {'train_loss':[], 'val_loss':[], 'train_mae':[], 'val_mae':[]}

    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} "
          f"{'Train MAE':<12} {'Val MAE (m)'}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss, tr_preds, tr_labels = 0.0, [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            tr_preds.extend(out.detach().cpu().numpy())
            tr_labels.extend(yb.cpu().numpy())
        train_loss /= len(train_loader.dataset)
        train_mae   = mean_absolute_error(tr_labels, tr_preds)

        # Val
        model.eval()
        val_loss, v_preds, v_labels = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out     = model(xb)
                loss    = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                v_preds.extend(out.cpu().numpy())
                v_labels.extend(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_mae   = mean_absolute_error(v_labels, v_preds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            saved = " ← sauvegardé"
        else:
            saved = ""

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} "
                  f"{train_mae:<12.2f} {val_mae:.2f}m{saved}")

    # Courbes
    plot_curves(history, OUTPUT_DIR)

    # Évaluation finale
    print("\n--- ÉVALUATION FINALE (test set) ---")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(DEVICE))
            all_preds.extend(out.cpu().numpy())
            all_labels.extend(yb.numpy())
    mae = mean_absolute_error(all_labels, all_preds)
    print(f"  {'✓' if mae < 1.0 else '✗'} MAE : {mae:.4f}m  (objectif < 1m)")

    # Graphiques évaluation finale
    all_preds_arr  = np.array(all_preds)
    all_labels_arr = np.array(all_labels)
    erreurs        = np.abs(all_preds_arr - all_labels_arr)

    BG, TEXT = '#0f1117', '#e2e8f0'
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    def style(ax, title):
        ax.set_facecolor('#1e2130')
        ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold')
        ax.tick_params(colors=TEXT)
        for sp in ax.spines.values(): sp.set_edgecolor('#374151')
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.legend(facecolor='#1e2130', labelcolor=TEXT)

    # Scatter valeurs réelles vs prédites
    axes[0].scatter(all_labels_arr, all_preds_arr,
                    alpha=0.4, color='#3b82f6', s=15)
    min_v = min(all_labels_arr.min(), all_preds_arr.min())
    max_v = max(all_labels_arr.max(), all_preds_arr.max())
    axes[0].plot([min_v, max_v], [min_v, max_v],
                 color='#ef4444', linewidth=2, label='Parfait')
    style(axes[0], 'Valeurs Reelles vs Predites')
    axes[0].set_xlabel('Valeur réelle (m)')
    axes[0].set_ylabel('Valeur prédite (m)')

    # Distribution des erreurs absolues
    axes[1].hist(erreurs, bins=30, color='#3b82f6',
                 edgecolor='white', linewidth=0.4, alpha=0.85)
    axes[1].axvline(1.0, color='#ef4444', linestyle='--',
                    linewidth=2, label='Objectif 1m')
    axes[1].axvline(erreurs.mean(), color='#f59e0b', linestyle='--',
                    linewidth=2, label=f'MAE = {erreurs.mean():.2f}m')
    style(axes[1], 'Distribution des Erreurs Absolues')
    axes[1].set_xlabel('Erreur absolue (m)')
    axes[1].set_ylabel('Nombre')

    fig.suptitle('Evaluation Finale — MLP Tache 2',
                 color=TEXT, fontsize=14, fontweight='bold')
    plt.tight_layout()
    eval_path = os.path.join(OUTPUT_DIR, 'evaluation_t2.png')
    plt.savefig(eval_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Graphique évaluation → {eval_path}")

    # Sauvegarde scaler
    scaler_path = os.path.join(OUTPUT_DIR, 'task2_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  Modèle → {model_path}")
    print(f"  Scaler → {scaler_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_t2()