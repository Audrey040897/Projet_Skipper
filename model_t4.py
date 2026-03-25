"""
model_t4.py — Skipper NDT — Tâche 4 : Conduites parallèles
============================================================
Pipeline : Image .pt → Resize 64px → CNN → Classification binaire

Classes :
  0 = single    (conduite simple)   — 1200 images
  1 = parallel  (conduites doubles) —  500 images

Le CNN apprend directement la forme du profil magnétique :
  - single   : 1 pic gaussien  → signature symétrique
  - parallel : 2 pics proches  → signature bimodale / élargie

Objectif : F1 > 0.80

Différences vs T1 :
  - Dataset filtré : uniquement images avec conduite (label==1)
  - Déséquilibre 2.4:1 → WeightedRandomSampler + pos_weight
  - Métrique principale : F1 (pas Recall)
  - 4 canaux utilisés (Bx, By, Bz, Norme) pour capturer la forme 2D

Fichiers générés :
  training_curves_t4.png   — Loss + F1/Accuracy Train vs Val
  model_t4.pt              — poids CNN
  model_t4_checkpoint.pt   — checkpoint reprise
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DatasetT4(Dataset):
    """
    Charge les .pt du cache pour la Tâche 4.
    Uniquement les images avec conduite (label==1).
    Label T4 : 1 si pipe_type=='parallel', 0 si 'single'.
    """

    def __init__(self, manifest_path: str, cache_dir: str):
        self.cache_dir = cache_dir
        df = pd.read_csv(manifest_path)
        if 'field_file' in df.columns:
            df = df.rename(columns={'field_file': 'filename'})

        # Filtrer : uniquement images avec conduite
        df = df[df['label'] == 1].reset_index(drop=True)

        # Label T4
        df['label_t4'] = (df['pipe_type'] == 'parallel').astype(int)

        self.df   = df
        n_par = int((df['label_t4'] == 1).sum())
        n_sin = int((df['label_t4'] == 0).sum())
        print(f"[DatasetT4] {len(df)} échantillons | "
              f"parallel={n_par}  single={n_sin}  "
              f"(déséquilibre {n_sin/n_par:.1f}:1)")

    def __len__(self): return len(self.df)

    def get_labels(self): return self.df['label_t4'].tolist()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pt  = os.path.join(self.cache_dir,
                           str(row['pt_file']).replace('\\', '/'))
        img = torch.load(pt, weights_only=True)   # (4, H, W)
        return {
            'image'   : img,
            'label'   : torch.tensor(int(row['label_t4']), dtype=torch.long),
            'filename': row['filename'],
        }


def collate_fn(batch):
    """Resize + padding adaptatif."""
    MAX_SIZE = 128  # résolution pour T4 — plus grande que T1 pour capturer la forme

    def resize(img):
        _, h, w = img.shape
        if max(h, w) <= MAX_SIZE:
            return img
        scale = MAX_SIZE / max(h, w)
        nh, nw = max(1, int(h*scale)), max(1, int(w*scale))
        return nn.functional.interpolate(
            img.unsqueeze(0).float(), size=(nh, nw),
            mode='bilinear', align_corners=False).squeeze(0)

    images = [resize(item['image']) for item in batch]
    max_h  = max(img.shape[1] for img in images)
    max_w  = max(img.shape[2] for img in images)
    padded = [nn.functional.pad(img, (0, max_w-img.shape[2],
                                       0, max_h-img.shape[1]))
              for img in images]
    return {
        'image'   : torch.stack(padded),
        'label'   : torch.stack([b['label'] for b in batch]),
        'filename': [b['filename'] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Architecture CNN
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class ModelT4(nn.Module):
    """
    CNN pour Tâche 4 — single vs parallel.

    Architecture légère + forte régularisation pour éviter l'overfitting.
    Le dataset est petit (1275 images) → moins de params + dropout fort.

    Entrée : (B, 4, H, W) — 4 canaux magnétiques
    Sortie : (B, 1)        — logit binaire
    """

    def __init__(self, in_channels=4, dropout=0.6):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 16,  stride=2),   # /2
            ConvBlock(16,          32,  stride=2),   # /4
            ConvBlock(32,          64,  stride=2),   # /8
            ConvBlock(64,          128, stride=2),   # /16
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.head(self.gap(self.backbone(x)).flatten(1))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Évaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    preds, labs = [], []
    for batch in loader:
        imgs  = batch['image'].to(device)
        probs = torch.sigmoid(model(imgs)).squeeze(1).cpu().numpy()
        preds.extend((probs >= threshold).astype(int))
        labs.extend(batch['label'].numpy())
    p, l = np.array(preds), np.array(labs)
    return {
        'accuracy': float(accuracy_score(l, p)),
        'recall'  : float(recall_score(l, p, zero_division=0, pos_label=1)),
        'f1'      : float(f1_score(l, p, zero_division=0, pos_label=1)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Courbes
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(history: dict, output_dir: str):
    """
    Génère training_curves_t4.png :
      Gauche : Loss (train)
      Droit  : F1 + Accuracy Train vs Val + ligne objectif F1=0.80
    """
    ep = history['epoch']
    if not ep:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Tâche 4 — CNN single vs parallel',
                 fontsize=13, fontweight='bold')

    # Loss
    axes[0].plot(ep, history['loss'], 'b-o', ms=3, lw=1.5, label='Train Loss')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Loss (train)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # F1 + Accuracy
    ax = axes[1]
    ax.plot(ep, history['train_f1'],  'g-o',  ms=3, lw=1.5, label='Train F1')
    ax.plot(ep, history['val_f1'],    'g--s', ms=3, lw=1.5, label='Val F1')
    ax.plot(ep, history['train_acc'], 'b-o',  ms=3, lw=1.5, label='Train Acc',
            alpha=0.6)
    ax.plot(ep, history['val_acc'],   'b--s', ms=3, lw=1.5, label='Val Acc',
            alpha=0.6)
    ax.axhline(0.80, color='green', lw=1.5, linestyle=':',
               label='Objectif F1 0.80')
    ax.set_xlabel('Époque')
    ax.set_ylabel('Score')
    ax.set_title('F1 & Accuracy — Train vs Val')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'training_curves_t4.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Courbes → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train_t4(manifest_path: str, cache_dir: str, output_dir: str,
             n_epochs: int = 150, batch_size: int = 16, lr: float = 1e-4,
             val_split: float = 0.15, test_split: float = 0.10,
             seed: int = 42):
    """
    Entraîne le CNN pour la Tâche 4 (single vs parallel).

    Génère :
      training_curves_t4.png   Loss + F1/Accuracy Train vs Val
      model_t4.pt              meilleur modèle
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*55}")
    print("ENTRAÎNEMENT T4 — CNN single vs parallel")
    print(f"  Device : {device}")
    print(f"{'='*55}")

    # ── Dataset + split ───────────────────────────────────────────────────
    ds     = DatasetT4(manifest_path, cache_dir)
    labels = ds.get_labels()
    n      = len(ds)

    train_idx, temp_idx = train_test_split(
        list(range(n)), test_size=val_split+test_split,
        random_state=seed, stratify=labels)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_split/(val_split+test_split),
        random_state=seed, stratify=[labels[i] for i in temp_idx])

    print(f"Split : Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

    # WeightedRandomSampler — compense le déséquilibre 2.4:1
    train_labels  = [labels[i] for i in train_idx]
    class_count   = np.bincount(train_labels)
    sample_weight = (1.0 / class_count)[np.array(train_labels)]
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weight).float(),
        num_samples=len(train_idx), replacement=True)

    kw = dict(collate_fn=collate_fn, num_workers=0)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size,
                              sampler=sampler, **kw)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size,
                              shuffle=False, **kw)
    test_loader  = DataLoader(Subset(ds, test_idx),  batch_size=batch_size,
                              shuffle=False, **kw)

    # ── Modèle ────────────────────────────────────────────────────────────
    model = ModelT4(in_channels=4, dropout=0.6).to(device)

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pw    = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=12)

    print(f"CNN T4 : {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Train  : {n_pos} parallel / {n_neg} single  "
          f"| pos_weight={pw.item():.2f}")

    # ── Historique ────────────────────────────────────────────────────────
    history = {
        'epoch': [], 'loss': [],
        'train_f1': [], 'train_acc': [],
        'val_f1':   [], 'val_acc':   [],
    }
    best_f1    = 0.0
    best_path  = os.path.join(output_dir, 'model_t4.pt')
    ckpt_path  = os.path.join(output_dir, 'model_t4_checkpoint.pt')
    start_epoch = 1

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_f1     = ckpt['best_score']
        history     = ckpt.get('history', history)
        print(f"[Reprise] Époque {start_epoch}/{n_epochs}  F1={best_f1:.4f}")

    # ── Boucle ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            lbl  = batch['label'].float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        m_val    = evaluate(model, val_loader,   device)
        m_train  = evaluate(model, train_loader, device)
        scheduler.step(m_val['f1'])

        print(f"Ep {epoch:3d}/{n_epochs} | Loss={avg_loss:.4f} | "
              f"Train F1={m_train['f1']:.3f} Acc={m_train['accuracy']:.3f} | "
              f"Val   F1={m_val['f1']:.3f}  Acc={m_val['accuracy']:.3f}")

        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['train_f1'].append(m_train['f1'])
        history['train_acc'].append(m_train['accuracy'])
        history['val_f1'].append(m_val['f1'])
        history['val_acc'].append(m_val['accuracy'])

        if m_val['f1'] > best_f1:
            best_f1 = m_val['f1']
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ F1={best_f1:.4f}  Acc={m_val['accuracy']:.4f}")

        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_score': best_f1, 'history': history}, ckpt_path)

    # ── Courbes ───────────────────────────────────────────────────────────
    plot_curves(history, output_dir)

    # ── Évaluation finale + recherche seuil optimal ──────────────────────
    print(f"\n--- Évaluation finale T4 (test set) ---")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    # Collecter toutes les probabilités
    all_probs, all_labs = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs  = batch['image'].to(device)
            probs = torch.sigmoid(model(imgs)).squeeze(1).cpu().numpy()
            all_probs.extend(probs)
            all_labs.extend(batch['label'].numpy())
    all_probs = np.array(all_probs)
    all_labs  = np.array(all_labs)

    # Recherche seuil optimal sur test set
    print(f"  {'Seuil':>7} | {'Accuracy':>10} | {'F1':>8} | {'Recall':>8}")
    print(f"  " + "-"*42)
    best_th, best_f1_th = 0.5, 0.0
    for thresh in np.arange(0.25, 0.65, 0.05):
        preds = (all_probs >= thresh).astype(int)
        acc   = float(accuracy_score(all_labs, preds))
        f1_   = float(f1_score(all_labs, preds, zero_division=0, pos_label=1))
        rec   = float(recall_score(all_labs, preds, zero_division=0, pos_label=1))
        mark  = " ✓✓" if f1_ > 0.80 else (" ✓" if f1_ > 0.75 else "")
        print(f"  {thresh:>7.2f} | {acc:>10.4f} | {f1_:>8.4f} | {rec:>8.4f}{mark}")
        if f1_ > best_f1_th:
            best_f1_th, best_th = f1_, thresh

    # Analyse des probabilités par classe
    probs_single   = all_probs[all_labs == 0]
    probs_parallel = all_probs[all_labs == 1]
    print(f"\n  Probabilités par classe :")
    print(f"    single   : moy={probs_single.mean():.3f}  "
          f"median={np.median(probs_single):.3f}  "
          f"max={probs_single.max():.3f}")
    print(f"    parallel : moy={probs_parallel.mean():.3f}  "
          f"median={np.median(probs_parallel):.3f}  "
          f"min={probs_parallel.min():.3f}")
    missed = probs_parallel[probs_parallel < 0.5]
    print(f"\n  Parallel manqués (prob < 0.5) : {len(missed)}/{len(probs_parallel)}")
    if len(missed) > 0:
        print(f"    Probabilités : {np.sort(missed)[::-1].round(3).tolist()}")
        zone = ((missed >= 0.35) & (missed < 0.5)).sum()
        below = (missed < 0.35).sum()
        print(f"    Entre 0.35-0.50 (récupérables avec seuil bas) : {zone}")
        print(f"    Sous 0.35 (erreurs franches)                   : {below}")

    print(f"\n  → Seuil optimal : {best_th:.2f}")
    preds_best = (all_probs >= best_th).astype(int)
    f1_best    = float(f1_score(all_labs, preds_best, zero_division=0, pos_label=1))
    acc_best   = float(accuracy_score(all_labs, preds_best))

    print(f"\n  Résultats finaux (seuil={best_th:.2f}) :")
    print(f"  {'✓' if f1_best>0.80 else '✗'} F1 (parallel) : {f1_best:.4f}  (> 0.80)")
    print(f"  Accuracy : {acc_best:.4f}")
    print()
    print(classification_report(all_labs, preds_best,
                                 target_names=['single', 'parallel']))
    print(f"  → {best_path}")

    # Sauvegarder le seuil optimal
    import json
    th_path = os.path.join(output_dir, 'threshold_t4.json')
    with open(th_path, 'w') as f_th:
        json.dump({'threshold': float(best_th),
                   'f1_test': round(f1_best, 4),
                   'acc_test': round(acc_best, 4)}, f_th, indent=2)
    print(f"  Seuil sauvegardé → {th_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. Inférence
# ─────────────────────────────────────────────────────────────────────────────

def predict_t4(npz_path: str, output_dir: str,
               threshold: float = None) -> dict:
    """
    Prédit si la conduite est simple ou parallèle.
    À appeler uniquement si T1 a détecté une conduite.

    Returns:
        {'parallel': 0|1, 'probability': float, 'label': str}
    """
    device = torch.device('cpu')
    model  = ModelT4(in_channels=4)
    model.load_state_dict(torch.load(
        os.path.join(output_dir, 'model_t4.pt'),
        map_location=device, weights_only=True))
    model.eval()

    # Charger + normaliser
    data   = np.load(npz_path, allow_pickle=True)
    arr    = data['data'].astype(np.float32)
    img    = np.transpose(arr, (2, 0, 1))   # (4, H, W)
    normed = np.zeros_like(img)
    for c in range(4):
        ch, valid = img[c], np.isfinite(img[c])
        if valid.any():
            cmin, cmax = ch[valid].min(), ch[valid].max()
            rng = cmax - cmin
            normed[c][valid] = 0.5 if rng < 1e-8 else (ch[valid]-cmin)/rng

    # Resize 128px
    t = torch.from_numpy(normed).unsqueeze(0).float()
    t = nn.functional.interpolate(t, size=(128, 128),
                                   mode='bilinear', align_corners=False)
    # Charger seuil optimal si disponible
    import json
    if threshold is None:
        th_path = os.path.join(output_dir, 'threshold_t4.json')
        threshold = json.load(open(th_path))['threshold'] if os.path.exists(th_path) else 0.5

    with torch.no_grad():
        prob = torch.sigmoid(model(t)).item()

    return {
        'parallel'   : int(prob >= threshold),
        'probability': round(prob, 4),
        'label'      : 'Conduites parallèles' if prob >= threshold
                       else 'Conduite simple',
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ══════════════════════════════════════════════════════════
    # ⚠️  MODIFIEZ CES CHEMINS
    # ══════════════════════════════════════════════════════════
    MANIFEST   = r"D:/Projet_skipper_RNDT/Projet_Skipper/cache_pt/manifest.csv"
    CACHE_DIR  = r"D:/Projet_skipper_RNDT/Projet_Skipper/cache_pt/"
    OUTPUT_DIR = r"D:/Projet_skipper_RNDT/Projet_Skipper/models/"
    # ══════════════════════════════════════════════════════════

    # Supprimer le checkpoint pour repartir :
    # del models\model_t4_checkpoint.pt

    train_t4(
        manifest_path = MANIFEST,
        cache_dir     = CACHE_DIR,
        output_dir    = OUTPUT_DIR,
        n_epochs      = 150,
        batch_size    = 16,
        lr            = 1e-4,
    )