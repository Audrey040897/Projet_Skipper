"""
model.py — Skipper NDT · CNN mutualisé Tâche 1 + Tâche 2
===========================================================

ARCHITECTURE :
  Backbone CNN partagé (4 blocs Conv + BatchNorm + ReLU + stride 2)
  → Global Average Pooling  →  vecteur fixe (B, 256)
  → Tête T1 : classification binaire   (conduite présente ?)
  → Tête T2 : régression continue      (largeur en mètres)

DONNÉES RÉELLES CONFIRMÉES :
  - Entrée      : (B, 4, H, W) float32 · dimensions variables
  - T1 labels   : 1 751 positifs / 1 184 négatifs  → class_weight = 1.48
  - T2 labels   : 2.0m → 154.8m · moyenne 36.9m    → présents seulement si label=1
  - Batch size  : 4 max (images jusqu'à 2791×1861 px)

LOSSES :
  T1 : BCEWithLogitsLoss(pos_weight=1.48)
  T2 : SmoothL1Loss  (plus robuste aux outliers que MSE)
       appliquée uniquement sur les échantillons label=1
  Total = loss_T1 + lambda_T2 * loss_T2

OBJECTIFS :
  T1 : Accuracy > 92%  ·  Recall > 95%
  T2 : MAE < 1 mètre
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score


# ─────────────────────────────────────────────────────────────────────────────
# 1. Prétraitement (autonome — pas besoin d'importer dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(path: str) -> np.ndarray:
    """NPZ (H,W,4) float16 → (4,H,W) float32."""
    data = np.load(path, allow_pickle=True)
    arr  = data['data'].astype(np.float32)
    return np.transpose(arr, (2, 0, 1))


def preprocess(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """NaN → 0 + normalisation Min-Max locale par canal."""
    normed = np.zeros_like(img)
    for c in range(img.shape[0]):
        ch    = img[c]
        valid = np.isfinite(ch)
        if not valid.any():
            continue
        cmin, cmax = ch[valid].min(), ch[valid].max()
        if cmax - cmin < eps:
            normed[c][valid] = 0.5
        else:
            normed[c][valid] = (ch[valid] - cmin) / (cmax - cmin)
    return normed


def augment(img: np.ndarray) -> np.ndarray:
    """Flips aléatoires — préserve la physique magnétique."""
    if np.random.rand() > 0.5:
        img = img[:, :, ::-1].copy()
    if np.random.rand() > 0.5:
        img = img[:, ::-1, :].copy()
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset T1 + T2
# ─────────────────────────────────────────────────────────────────────────────

class SkipperDatasetT1T2(Dataset):
    """
    Dataset pour Tâche 1 (détection) + Tâche 2 (régression largeur).

    Peut charger :
      - Fichiers NPZ bruts   (use_cache=False)
      - Tenseurs .pt en cache (use_cache=True, beaucoup plus rapide)

    Pour use_cache=True, img_dir doit pointer vers le sous-dossier
    'original/' ou 'aug_X/' généré par save_augmented.py.
    Utilisez le manifest.csv généré comme csv_path.
    """

    def __init__(self, csv_path: str, img_dir: str,
                 do_augment: bool = False, use_cache: bool = False):

        self.img_dir    = img_dir
        self.do_augment = do_augment
        self.use_cache  = use_cache

        df = pd.read_csv(csv_path, sep=';' if ';' in open(csv_path).readline() else ',')

        # Normaliser le nom de la colonne fichier
        for col in ['field_file', 'filename', 'original_file']:
            if col in df.columns:
                df = df.rename(columns={col: 'filename'})
                break

        # T2 : width_m absent si no_pipe → on le gardera comme NaN (masqué dans la loss)
        self.df = df.reset_index(drop=True)

        n_pos = int((df['label'] == 1).sum())
        n_neg = int((df['label'] == 0).sum())
        print(f"[Dataset T1+T2] {len(df)} échantillons")
        print(f"  T1 : {n_pos} positifs / {n_neg} négatifs")
        w2_valid = df['width_m'].notna().sum()
        print(f"  T2 : {w2_valid} labels width_m valides")

    def __len__(self):
        return len(self.df)

    def get_labels_t1(self) -> list:
        return self.df['label'].tolist()

    def __getitem__(self, idx: int) -> dict:
        row   = self.df.iloc[idx]
        fname = row['filename']

        if self.use_cache:
            # manifest.csv a une colonne 'pt_file' avec le chemin relatif
            # ex : "original/sample_00000_perfect_straight_clean_field.pt"
            pt_col   = 'pt_file' if 'pt_file' in row.index else 'filename'
            pt_fname = str(row[pt_col]).replace('\\', '/').replace('\\\\', '/')
            img = torch.load(os.path.join(self.img_dir, pt_fname),
                             weights_only=True)
        else:
            img = preprocess(load_npz(os.path.join(self.img_dir, fname)))
            if self.do_augment:
                img = augment(img)
            img = torch.from_numpy(img.copy())

        sample = {
            'image'   : img,
            'filename': fname,
            'label_t1': torch.tensor(int(row['label']), dtype=torch.long),
        }

        # T2 : NaN si no_pipe (masqué dans la loss)
        if pd.notna(row.get('width_m')):
            sample['label_t2'] = torch.tensor(float(row['width_m']),
                                               dtype=torch.float32)
        else:
            sample['label_t2'] = None

        return sample


# Taille maximale des images après redimensionnement
# Valeur de 512 : bon compromis vitesse/qualité sur CPU
# Augmenter à 768 ou 1024 si vous avez un GPU
MAX_IMG_SIZE = 256  # 256 sur CPU (~3 min/époque), 512 sur GPU


def resize_img(img: torch.Tensor, max_size: int = MAX_IMG_SIZE) -> torch.Tensor:
    """
    Redimensionne une image (C, H, W) en conservant le ratio d'aspect.
    La plus grande dimension est ramenée à max_size.
    Utilise l'interpolation bilinéaire (préserve mieux les patterns que nearest).
    """
    _, h, w = img.shape
    if max(h, w) <= max_size:
        return img   # déjà assez petite, pas besoin de resize

    scale = max_size / max(h, w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    return nn.functional.interpolate(
        img.unsqueeze(0),                # (1, C, H, W)
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)                         # (C, new_h, new_w)


def collate_t1t2(batch: list) -> dict:
    """
    Resize toutes les images à MAX_IMG_SIZE (ratio conservé)
    puis padding adaptatif à la taille max du batch.

    Pourquoi resize ici et pas dans __getitem__ ?
    Le resize dans le collate permet de l'appliquer après le chargement
    du cache .pt, sans modifier les fichiers sauvegardés.
    """
    # 1. Resize chaque image
    # Images déjà resize dans le cache — chargement direct
    images = [item['image'] for item in batch]

    # 2. Padding à la taille max du batch (après resize, dimensions proches)
    max_h  = max(img.shape[1] for img in images)
    max_w  = max(img.shape[2] for img in images)

    padded = [
        nn.functional.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))
        for img in images
    ]

    result = {
        'image'   : torch.stack(padded),
        'filename': [item['filename'] for item in batch],
        'label_t1': torch.stack([item['label_t1'] for item in batch]),
    }

    # T2 : remplace None par -1.0 (sentinelle → masqué dans la loss)
    t2_vals = [item['label_t2'] for item in batch]
    result['label_t2'] = torch.stack([
        v if v is not None else torch.tensor(-1.0) for v in t2_vals
    ])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Architecture CNN
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d(3×3, stride) → BatchNorm2d → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SkipperCNN(nn.Module):
    """
    CNN mutualisé Tâche 1 + Tâche 2.

    Backbone :
      4 blocs ConvBlock avec stride=2 → downsampling ×16
      Global Average Pooling → vecteur (B, 256)

    Têtes :
      head_t1 : FC(256→128→1) + Dropout(0.4)  → logit binaire
      head_t2 : FC(256→128→1) + Dropout(0.3)  → valeur continue (largeur m)

    Args:
        in_channels : canaux d'entrée (4 pour Bx/By/Bz/Norme)
        base_ch     : canaux du 1er bloc (doublés à chaque bloc)
        dropout_t1  : dropout tête T1
        dropout_t2  : dropout tête T2
    """

    def __init__(self, in_channels: int = 4, base_ch: int = 32,
                 dropout_t1: float = 0.4, dropout_t2: float = 0.3):
        super().__init__()

        # ── Backbone partagé ─────────────────────────────────────────────────
        self.backbone = nn.Sequential(
            ConvBlock(in_channels,      base_ch,     stride=2),  # ÷2
            ConvBlock(base_ch,          base_ch * 2, stride=2),  # ÷4
            ConvBlock(base_ch * 2,      base_ch * 4, stride=2),  # ÷8
            ConvBlock(base_ch * 4,      base_ch * 8, stride=2),  # ÷16
        )
        feat_dim = base_ch * 8  # 256 avec base_ch=32

        # Global Average Pooling : (B, 256, H', W') → (B, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Tête T1 : détection conduite (classification binaire) ─────────────
        self.head_t1 = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_t1),
            nn.Linear(128, 1),         # logit → sigmoid pour probabilité
        )

        # ── Tête T2 : largeur carte magnétique (régression) ───────────────────
        self.head_t2 = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_t2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_t2 * 0.5),
            nn.Linear(128, 1),         # prédit dans [0,1] → ×155 = mètres
            nn.Sigmoid(),              # force la sortie dans [0,1]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialisation He pour les couches Conv, Xavier pour les FC."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x : (B, 4, H, W) — float32, normalisé [0,1]
        Returns:
            dict avec :
              'logit_t1' : (B, 1) logit binaire  (appliquer sigmoid)
              'pred_t2'  : (B, 1) largeur prédite en mètres
        """
        features = self.backbone(x)                      # (B, 256, H', W')
        pooled   = self.gap(features).flatten(1)         # (B, 256)

        return {
            'logit_t1': self.head_t1(pooled),            # (B, 1)
            'pred_t2' : self.head_t2(pooled),            # (B, 1)
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calcul des losses (mutualisées)
# ─────────────────────────────────────────────────────────────────────────────

def compute_losses(outputs: dict, batch: dict,
                   criterion_t1: nn.Module, criterion_t2: nn.Module,
                   lambda_t2: float = 1.0, device: torch.device = None) -> dict:
    """
    Calcule les losses T1 et T2 séparément puis la loss totale.

    T2 est masquée : on ne calcule la loss que sur les échantillons
    où label_t2 != -1 (c.-à-d. uniquement les images avec conduite).

    Args:
        outputs    : dict retourné par SkipperCNN.forward()
        batch      : dict du DataLoader (label_t1, label_t2)
        criterion_t1 : BCEWithLogitsLoss
        criterion_t2 : SmoothL1Loss
        lambda_t2  : pondération de la loss T2 dans la loss totale
        device     : torch.device

    Returns:
        dict avec loss_t1, loss_t2, loss_total (tous des scalaires)
    """
    # ── Loss T1 ──────────────────────────────────────────────────────────────
    label_t1 = batch['label_t1'].float().unsqueeze(1).to(device)
    loss_t1  = criterion_t1(outputs['logit_t1'], label_t1)

    # ── Loss T2 : masquée sur les no_pipe (label_t2 == -1) ───────────────────
    # Normalisation du target dans [0,1] : width_m / WIDTH_MAX
    WIDTH_MAX = 155.0
    label_t2  = batch['label_t2'].unsqueeze(1).to(device)  # (B, 1)
    mask_t2   = (label_t2 >= 0)                            # True si valeur réelle

    if mask_t2.sum() > 0:
        pred_masked  = outputs['pred_t2'][mask_t2]
        # Normaliser le label dans [0,1] pour que la loss soit comparable à T1
        label_masked = label_t2[mask_t2] / WIDTH_MAX
        loss_t2      = criterion_t2(pred_masked, label_masked)
    else:
        loss_t2 = torch.tensor(0.0, device=device)

    loss_total = loss_t1 + lambda_t2 * loss_t2

    return {
        'loss_t1'   : loss_t1,
        'loss_t2'   : loss_t2,
        'loss_total': loss_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Évaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device, threshold: float = 0.5) -> dict:
    """
    Calcule toutes les métriques T1 et T2 sur un DataLoader.

    Returns:
        dict avec accuracy, recall, f1 (T1) et mae, rmse (T2)
    """
    model.eval()
    preds_t1, labels_t1 = [], []
    preds_t2, labels_t2 = [], []

    WIDTH_MAX = 155.0
    for batch in loader:
        imgs = batch['image'].to(device)
        out  = model(imgs)

        # T1
        prob  = torch.sigmoid(out['logit_t1']).squeeze(1).cpu()
        pred  = (prob >= threshold).long().numpy()
        preds_t1.extend(pred)
        labels_t1.extend(batch['label_t1'].numpy())

        # T2 : dénormaliser la prédiction (modèle prédit dans [0,1])
        lbl_t2 = batch['label_t2'].numpy()
        # Dénormaliser : pred * WIDTH_MAX → mètres
        prd_t2 = out['pred_t2'].squeeze(1).cpu().numpy() * WIDTH_MAX
        mask   = lbl_t2 >= 0
        if mask.sum() > 0:
            preds_t2.extend(prd_t2[mask])
            labels_t2.extend(lbl_t2[mask])

    preds_t1  = np.array(preds_t1)
    labels_t1 = np.array(labels_t1)

    metrics = {
        'accuracy': accuracy_score(labels_t1, preds_t1),
        'recall'  : recall_score(labels_t1, preds_t1, zero_division=0),
        'f1'      : f1_score(labels_t1, preds_t1, zero_division=0),
    }

    if preds_t2:
        preds_t2  = np.array(preds_t2)
        labels_t2 = np.array(labels_t2)
        metrics['mae']  = float(np.abs(preds_t2 - labels_t2).mean())
        metrics['rmse'] = float(np.sqrt(((preds_t2 - labels_t2) ** 2).mean()))
    else:
        metrics['mae']  = float('nan')
        metrics['rmse'] = float('nan')

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train(
    csv_path    : str,
    img_dir     : str,
    output_dir  : str   = './models',
    use_cache   : bool  = False,
    n_epochs    : int   = 40,
    batch_size  : int   = 4,
    lr          : float = 3e-4,
    lambda_t2   : float = 0.5,
    val_split   : float = 0.15,
    test_split  : float = 0.10,
    seed        : int   = 42,
):
    """
    Entraîne SkipperCNN sur T1 + T2 et sauvegarde le meilleur modèle.

    Args:
        csv_path   : pipe_detection_label.csv  OU  manifest.csv
        img_dir    : dossier NPZ  OU  dossier cache_pt/
        output_dir : dossier de sauvegarde (.pt)
        use_cache  : True pour charger des .pt (beaucoup plus rapide)
        n_epochs   : nombre d'époques
        batch_size : 4 max recommandé (images très grandes)
        lr         : learning rate Adam
        lambda_t2  : poids de la loss T2 dans la loss totale
        val_split  : proportion validation
        test_split : proportion test
        seed       : reproductibilité
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Train] Device : {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    ds_train = SkipperDatasetT1T2(csv_path, img_dir,
                                   do_augment=True,  use_cache=use_cache)
    ds_eval  = SkipperDatasetT1T2(csv_path, img_dir,
                                   do_augment=False, use_cache=use_cache)

    n       = len(ds_train)
    indices = list(range(n))

    # Split stratifié sur T1 (label binaire)
    labels_all  = ds_train.get_labels_t1()
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_split + test_split,
        random_state=seed, stratify=labels_all
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_split / (val_split + test_split),
        random_state=seed,
        stratify=[labels_all[i] for i in temp_idx]
    )
    print(f"[Split] Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

    # ── WeightedRandomSampler pour équilibrer T1 ─────────────────────────────
    train_labels  = [labels_all[i] for i in train_idx]
    class_count   = np.bincount(train_labels)
    sample_weight = (1.0 / class_count)[np.array(train_labels)]
    sampler       = WeightedRandomSampler(
        torch.from_numpy(sample_weight).float(),
        num_samples=len(train_idx), replacement=True
    )

    kw = dict(collate_fn=collate_t1t2, pin_memory=torch.cuda.is_available(),
              num_workers=0)   # num_workers=0 sur Windows

    train_loader = DataLoader(Subset(ds_train, train_idx),
                              batch_size=batch_size, sampler=sampler, **kw)
    val_loader   = DataLoader(Subset(ds_eval,  val_idx),
                              batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(Subset(ds_eval,  test_idx),
                              batch_size=batch_size, shuffle=False, **kw)

    # ── Modèle ───────────────────────────────────────────────────────────────
    model   = SkipperCNN(in_channels=4, base_ch=32).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Modèle] {n_param:,} paramètres entraînables")

    # ── Losses ───────────────────────────────────────────────────────────────
    # T1 : pos_weight = n_négatifs / n_positifs
    n_pos   = sum(train_labels)
    n_neg   = len(train_labels) - n_pos
    pw      = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    crit_t1 = nn.BCEWithLogitsLoss(pos_weight=pw)
    crit_t2 = nn.SmoothL1Loss()   # robuste aux outliers (width_m varie de 2 à 155)
    print(f"[Loss] T1 pos_weight={pw.item():.3f} · lambda_T2={lambda_t2}")

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    best_score  = float('inf')
    best_path   = os.path.join(output_dir, 'skipper_t1t2.pt')
    ckpt_path   = os.path.join(output_dir, 'skipper_t1t2_checkpoint.pt')
    history     = []
    start_epoch = 1

    # Reprise automatique depuis le dernier checkpoint si interrompu
    if os.path.exists(ckpt_path):
        print(f"\n[Reprise] Checkpoint trouvé : {ckpt_path}")
        ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_score  = ckpt['best_score']
        history     = ckpt.get('history', [])
        print(f"  Reprise à l'époque {start_epoch}/{n_epochs} | meilleur score={best_score:.4f}")
    else:
        print(f"\n[Train] Démarrage époque 1/{n_epochs}")

    for epoch in range(start_epoch, n_epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        epoch_losses = {'loss_t1': 0, 'loss_t2': 0, 'loss_total': 0}

        for batch in train_loader:
            imgs = batch['image'].to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = compute_losses(out, batch, crit_t1, crit_t2,
                                  lambda_t2=lambda_t2, device=device)
            loss['loss_total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += loss[k].item()

        n_batches = len(train_loader)
        avg = {k: v / n_batches for k, v in epoch_losses.items()}

        # ── Validation ────────────────────────────────────────────────────────
        m = evaluate(model, val_loader, device)
        scheduler.step(avg['loss_total'])

        # Score composite : on optimise recall T1 ET mae T2 ensemble
        mae_val  = m['mae']  if not np.isnan(m['mae']) else 10.0
        score    = mae_val + (1.0 - m['recall'])

        log = (f"Ep {epoch:3d}/{n_epochs} | "
               f"Loss={avg['loss_total']:.4f} "
               f"(T1={avg['loss_t1']:.4f} T2={avg['loss_t2']:.4f}) | "
               f"T1: Acc={m['accuracy']:.3f} Rec={m['recall']:.3f} | "
               f"T2: MAE={mae_val:.2f}m")
        print(log)
        history.append({'epoch': epoch, **avg, **m})

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Meilleur modèle sauvegardé (score={best_score:.4f})")

        # Checkpoint à chaque époque → permet de reprendre si interruption
        torch.save({
            'epoch'          : epoch,
            'model_state'    : model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_score'     : best_score,
            'history'        : history,
        }, ckpt_path)

    # ── Évaluation finale ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ÉVALUATION FINALE — Test set")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    m = evaluate(model, test_loader, device)

    ok_acc    = "✓" if m['accuracy'] > 0.92 else "✗"
    ok_recall = "✓" if m['recall']   > 0.95 else "✗"
    ok_mae    = "✓" if m['mae']      < 1.0  else "✗"

    print(f"\n  Tâche 1 — Détection conduite")
    print(f"   {ok_acc} Accuracy : {m['accuracy']:.4f}  (objectif > 0.92)")
    print(f"   {ok_recall} Recall   : {m['recall']:.4f}  (objectif > 0.95)")
    print(f"   F1-Score : {m['f1']:.4f}")
    print(f"\n  Tâche 2 — Largeur carte")
    print(f"   {ok_mae} MAE  : {m['mae']:.4f} m  (objectif < 1.0 m)")
    print(f"   RMSE : {m['rmse']:.4f} m")
    print(f"\n  Modèle → {best_path}")
    print(f"{'='*60}")

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# 7. Inférence sur un fichier NPZ unique
# ─────────────────────────────────────────────────────────────────────────────

def predict(npz_path: str, model_path: str,
            threshold: float = 0.5) -> dict:
    """
    Prédit T1 + T2 sur un fichier NPZ.

    Args:
        npz_path   : chemin vers le fichier .npz
        model_path : chemin vers skipper_t1t2.pt
        threshold  : seuil de décision T1 (défaut 0.5)

    Returns:
        {
          'pipeline_present' : 0 ou 1,
          'probability'      : float [0,1],
          'map_width_m'      : float (mètres) — pertinent si pipeline_present=1,
          'label'            : 'Conduite détectée' ou 'Aucune conduite',
        }
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SkipperCNN(in_channels=4, base_ch=32)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.to(device).eval()

    img = preprocess(load_npz(npz_path))
    x   = torch.from_numpy(img.copy()).unsqueeze(0).to(device)

    with torch.no_grad():
        out   = model(x)
        prob  = torch.sigmoid(out['logit_t1']).item()
        pred  = int(prob >= threshold)
        width = out['pred_t2'].item() * 155.0  # dénormalisation → mètres

    return {
        'pipeline_present': pred,
        'probability'     : round(prob, 4),
        'map_width_m'     : round(width, 2),
        'label'           : 'Conduite détectée' if pred == 1 else 'Aucune conduite',
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Point d'entrée — modifiez les chemins ici
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ══════════════════════════════════════════════════════════════
    # ⚠️  MODIFIEZ CES CHEMINS
    # ══════════════════════════════════════════════════════════════

    # ── Chemins NPZ bruts (USE_CACHE=False obligatoire) ──────────────────
    CSV_PATH  = r"D:/Projet_skipper_RNDT/Projet_Skipper/cache_pt/manifest.csv"
    IMG_DIR   = r"D:/Projet_skipper_RNDT/Projet_Skipper/cache_pt/"
    USE_CACHE = True

    OUTPUT_DIR = r"D:/Projet_skipper_RNDT/Projet_Skipper/models/"

    # ══════════════════════════════════════════════════════════════
    # ⚠️  HYPERPARAMÈTRES
    # ══════════════════════════════════════════════════════════════
    N_EPOCHS   = 60  # 100 époques — laisser tourner toute la nuit sur CPU
    BATCH_SIZE = 16   # images 128px → batch plus grand possible     # 4 avec MAX_IMG_SIZE=256, réduire à 2 si erreur mémoire
    LR         = 3e-4
    LAMBDA_T2  = 5.0   # augmenté pour forcer T2 à converger
    # ══════════════════════════════════════════════════════════════

    model, history = train(
        csv_path   = CSV_PATH,
        img_dir    = IMG_DIR,
        output_dir = OUTPUT_DIR,
        use_cache  = USE_CACHE,
        n_epochs   = N_EPOCHS,
        batch_size = BATCH_SIZE,
        lr         = LR,
        lambda_t2  = LAMBDA_T2,
    )