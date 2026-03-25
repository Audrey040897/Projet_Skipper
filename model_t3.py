"""
model_t3.py — Skipper NDT — Tâche 3 : Couverture suffisante
=============================================================
Classification binaire de la couverture du signal magnétique.

  0 = couverture insuffisante
  1 = couverture suffisante

Pipeline :
  NPZ → nan_to_num → normalisation locale → cv2.resize(64,64)
      → CNN 4 canaux (3 blocs Conv + GAP + FC)

Architecture CNN :
  Conv(4→32) + BN + ReLU + MaxPool
  Conv(32→64) + BN + ReLU + MaxPool
  Conv(64→128) + BN + ReLU + MaxPool
  GAP → FC(128→64) → Dropout(0.5) → FC(64→2)

Objectifs :
  Accuracy > 90%  ·  Recall > 85%

Dataset :
  data_tache_3.csv — contient field_file + label + coverage_type

Fichiers sauvegardés :
  task3_model.pth
  training_curves_t3.png
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score,
                              f1_score, confusion_matrix)
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = r"D:/Projet_skipper_RNDT/Projet_Skipper/Data_NDT/Training_database_float16"
CSV_PATH   = r"D:/Projet_skipper_RNDT/Projet_Skipper/Data_t3"
OUTPUT_DIR = r"D:/Projet_skipper_RNDT/Projet_Skipper/models"
IMG_SIZE   = (64, 64)
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-3
SEED       = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Prétraitement NPZ
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_npz(file_path: str, img_size=IMG_SIZE,
                   augment: bool = False) -> torch.Tensor:
    """
    Charge et prétraite un NPZ pour T3.
    Pipeline : nan_to_num → normalisation locale → resize 64×64 → tensor
    """
    data = np.load(file_path)['data'].astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)

    # Normalisation locale par canal
    for c in range(data.shape[2]):
        canal = data[:, :, c]
        cmin, cmax = canal.min(), canal.max()
        if cmax - cmin > 0:
            data[:, :, c] = (canal - cmin) / (cmax - cmin)
        else:
            data[:, :, c] = 0.0

    # Resize
    data_r = cv2.resize(data, (img_size[1], img_size[0]),
                        interpolation=cv2.INTER_LINEAR)

    # Augmentation (flip H/V)
    if augment:
        if np.random.rand() > 0.5:
            data_r = np.fliplr(data_r)
        if np.random.rand() > 0.5:
            data_r = np.flipud(data_r)

    return torch.tensor(data_r.copy()).permute(2, 0, 1)   # (4, 64, 64)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PipelineDataset(Dataset):
    """
    Dataset T3 — charge toutes les images en RAM au démarrage.
    Augmentation flip H/V à la volée sur le train.
    """
    def __init__(self, dataframe: pd.DataFrame, data_dir: str,
                 img_size=IMG_SIZE, augment: bool = False):
        self.images  = []
        self.labels  = []
        self.augment = augment

        print(f"  Chargement de {len(dataframe)} images en RAM...")
        for _, row in dataframe.iterrows():
            fpath = os.path.join(data_dir, row['field_file'])
            if not os.path.exists(fpath):
                continue
            self.images.append(preprocess_npz(fpath, img_size, augment=False))
            self.labels.append(int(row['label']))
            if len(self.images) % 300 == 0:
                print(f"    {len(self.images)}/{len(dataframe)} chargées...")
        print(f"  Chargement terminé : {len(self.images)} images")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.augment:
            if np.random.rand() > 0.5:
                img = torch.flip(img, dims=[2])   # flip horizontal
            if np.random.rand() > 0.5:
                img = torch.flip(img, dims=[1])   # flip vertical
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Architecture CNN
# ─────────────────────────────────────────────────────────────────────────────

class PipelineCNN(nn.Module):
    """CNN 4 canaux pour classification couverture (T3)."""
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(
            self.gap(self.block3(self.block2(self.block1(x)))))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Courbes d'entraînement
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(history: dict, output_dir: str):
    BG, TEXT = '#0f1117', '#e2e8f0'
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(BG)

    def style(ax, title):
        ax.set_facecolor('#1e2130')
        ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold')
        ax.tick_params(colors=TEXT)
        for sp in ax.spines.values(): sp.set_edgecolor('#374151')
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.legend(facecolor='#1e2130', labelcolor=TEXT)

    axes[0].plot(ep, history['train_loss'], color='#3b82f6',
                 label='Train Loss', linewidth=2)
    axes[0].plot(ep, history['val_loss'],   color='#ef4444',
                 label='Val Loss',   linewidth=2)
    style(axes[0], 'Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].plot(ep, history['train_acc'], color='#3b82f6',
                 label='Train Acc', linewidth=2)
    axes[1].plot(ep, history['val_acc'],   color='#ef4444',
                 label='Val Acc',   linewidth=2)
    axes[1].axhline(0.90, color='#10b981', linestyle='--',
                    linewidth=1.5, label='Objectif 90%')
    style(axes[1], 'Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')

    axes[2].plot(ep, history['val_recall'], color='#f59e0b',
                 label='Val Recall', linewidth=2)
    axes[2].plot(ep, history['val_f1'],     color='#8b5cf6',
                 label='Val F1',     linewidth=2)
    axes[2].axhline(0.85, color='#10b981', linestyle='--',
                    linewidth=1.5, label='Objectif Recall 85%')
    style(axes[2], 'Recall & F1')
    axes[2].set_xlabel('Epoch')

    fig.suptitle('Courbes Entraînement — CNN Tâche 3',
                 color=TEXT, fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves_t3.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Courbes → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train_t3():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*55}")
    print("ENTRAÎNEMENT T3 — CNN couverture")
    print(f"  Device : {DEVICE}")
    print(f"{'='*55}")
    CSV_PATH= r"D:/Projet_skipper_RNDT/Projet_Skipper\Data_t3/Training_data_inspection_validation_float16/pipe_detection_label.csv"
    # Chargement CSV
    df = pd.read_csv(CSV_PATH, sep=',')

    # Garder uniquement les fichiers existants
    df['exists'] = df['field_file'].apply(
        lambda x: os.path.exists(os.path.join(DATA_DIR, x)))
    df = df[df['exists']].reset_index(drop=True)

    print(f"\nDataset T3 : {len(df)} fichiers trouvés")
    print(f"  label=0 (insuffisant) : {(df['label']==0).sum()}")
    print(f"  label=1 (suffisant)   : {(df['label']==1).sum()}")

    # Split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df['label'])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED,
        stratify=temp_df['label'])
    print(f"Split : Train={len(train_df)} | Val={len(val_df)} | "
          f"Test={len(test_df)}")

    # Datasets
    print("\nChargement train...")
    train_ds = PipelineDataset(train_df, DATA_DIR, augment=True)
    print("Chargement val...")
    val_ds   = PipelineDataset(val_df,   DATA_DIR, augment=False)
    print("Chargement test...")
    test_ds  = PipelineDataset(test_df,  DATA_DIR, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # Modèle
    model      = PipelineCNN().to(DEVICE)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters(), lr=LR)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5)
    print(f"\nCNN T3 : {sum(p.numel() for p in model.parameters()):,} params")

    model_path    = os.path.join(OUTPUT_DIR, 'task3_model.pth')
    best_val_loss = float('inf')
    history       = {
        'train_loss':[], 'val_loss':[],
        'train_acc':[], 'val_acc':[],
        'val_recall':[], 'val_f1':[],
    }

    print(f"\n{'Epoch':<8} {'Tr Loss':<10} {'Va Loss':<10} "
          f"{'Tr Acc':<10} {'Va Acc':<10} {'Va Rec':<10} Va F1")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * images.size(0)
            preds       = out.argmax(dim=1)
            tr_correct += (preds == labels).sum().item()
            tr_total   += images.size(0)
        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total

        # Val
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                out    = model(images)
                loss   = criterion(out, labels)
                va_loss    += loss.item() * images.size(0)
                preds       = out.argmax(dim=1)
                va_correct += (preds == labels).sum().item()
                va_total   += images.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        va_loss  /= va_total
        va_acc    = va_correct / va_total
        va_recall = recall_score(all_labels, all_preds, zero_division=0)
        va_f1     = f1_score(all_labels, all_preds, zero_division=0)
        scheduler.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), model_path)
            saved = " ✓"
        else:
            saved = ""

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['val_recall'].append(va_recall)
        history['val_f1'].append(va_f1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:<8} {tr_loss:<10.4f} {va_loss:<10.4f} "
                  f"{tr_acc:<10.3f} {va_acc:<10.3f} "
                  f"{va_recall:<10.3f} {va_f1:.3f}{saved}")

    # Courbes
    plot_curves(history, OUTPUT_DIR)

    # Évaluation finale
    print("\n--- ÉVALUATION FINALE (test set) ---")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            out   = model(images.to(DEVICE))
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc    = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1     = f1_score(all_labels, all_preds, zero_division=0)

    print(f"  {'✓' if acc>0.90 else '✗'} Accuracy : {acc:.4f}  (> 0.90)")
    print(f"  {'✓' if recall>0.85 else '✗'} Recall   : {recall:.4f}  (> 0.85)")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  → {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_t3()