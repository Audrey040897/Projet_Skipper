"""
inference.py — Skipper NDT — Script d'inférence final
======================================================
Unifie les 4 tâches :

  T1 — Détection conduite       : PCA (canal Bz) + KNN
       Acc=1.000  Recall=1.000

  T2 — Largeur carte (m)        : Features physiques + MLP (notebook prof)
       Modèles : task2_model.pth + task2_scaler.pkl

  T3 — Couverture suffisante    : CNN 4 canaux (notebook prof)
       Modèle  : task3_model.pth
       0 = insuffisant  |  1 = suffisant

  T4 — Conduites parallèles     : CNN 4 canaux (notre modèle)
       F1=0.795  Acc=0.888
       Modèle  : model_t4.pt

Usage :
  python inference.py --npz mon_image.npz
  python inference.py --dossier ./real_data/ --output_csv resultats.csv

Sortie :
  {
    "fichier"          : "real_data_00045.npz",
    "T1_conduite"      : 1,
    "T1_probabilite"   : 1.0,
    "T1_label"         : "Conduite détectée",
    "T2_largeur_m"     : 21.5,
    "T3_couverture"    : 1,
    "T3_label"         : "Couverture suffisante",
    "T4_parallel"      : 1,
    "T4_probabilite"   : 0.98,
    "T4_label"         : "Conduites parallèles"
  }

Note : T3 et T4 sont calculés uniquement si T1 détecte une conduite.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import cv2
from skimage.transform import resize as sk_resize


# ─────────────────────────────────────────────────────────────────────────────
# Chemins par défaut
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS_DIR = r"D:/Projet_skipper_RNDT/Projet_Skipper/models/"


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires communs
# ─────────────────────────────────────────────────────────────────────────────

def load_npz_raw(npz_path: str) -> np.ndarray:
    """Charge un NPZ → (H, W, 4) float32 avec NaN."""
    data = np.load(npz_path, allow_pickle=True)
    return data['data'].astype(np.float32)


def normalize_local_4ch(arr: np.ndarray) -> np.ndarray:
    """
    Normalisation locale par canal dans [0,1].
    NaN → 0. arr shape : (H, W, 4).
    """
    result = np.zeros_like(arr)
    for c in range(arr.shape[2]):
        ch    = arr[:, :, c].copy()
        valid = np.isfinite(ch)
        if valid.any():
            cmin, cmax = ch[valid].min(), ch[valid].max()
            if cmax - cmin > 0:
                ch[valid] = (ch[valid] - cmin) / (cmax - cmin)
            else:
                ch[valid] = 0.5
        ch[~valid] = 0.0
        result[:, :, c] = ch
    return result


# ─────────────────────────────────────────────────────────────────────────────
# T1 — Détection conduite (PCA + KNN)
# ─────────────────────────────────────────────────────────────────────────────

def predict_t1(npz_path: str, models_dir: str) -> dict:
    """
    Tâche 1 — Détection conduite.
    Pipeline : canal Bz → nan_to_num → resize 224px → PCA → KNN
    """
    knn    = joblib.load(os.path.join(models_dir, 'model_t1_knn.pkl'))
    pca    = joblib.load(os.path.join(models_dir, 'pca_t1.pkl'))
    th_cfg = json.load(open(os.path.join(models_dir, 'threshold_t1.json')))
    threshold = th_cfg.get('threshold', 0.5)

    arr   = load_npz_raw(npz_path)
    bz    = np.nan_to_num(arr[:, :, 2])
    bz_r  = sk_resize(bz, (224, 224), anti_aliasing=True).flatten().reshape(1, -1)
    vec   = pca.transform(bz_r)
    prob  = float(knn.predict_proba(vec)[0, 1])
    pred  = int(prob >= threshold)

    return {
        'T1_conduite'   : pred,
        'T1_probabilite': round(prob, 4),
        'T1_label'      : 'Conduite détectée' if pred else 'Aucune conduite',
    }


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Largeur carte : Features physiques + MLP (notebook prof)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features_t2(npz_path: str) -> list:
    """
    Extraction des features physiques du notebook du professeur.
    Identique à extract_features() dans model_t2.ipynb.
    """
    data = np.load(npz_path, allow_pickle=True)['data'].astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)

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
    thresh_c  = col_means.max() * 0.5
    thresh_r  = row_means.max() * 0.5
    features['profile_width_cols'] = float((col_means > thresh_c).sum())
    features['profile_width_rows'] = float((row_means > thresh_r).sum())

    # Gradient spatial
    grad_x = np.gradient(Norm, axis=1)
    grad_y = np.gradient(Norm, axis=0)
    grad_m = np.sqrt(grad_x**2 + grad_y**2)
    features['grad_mean'] = float(grad_m[mask].mean())
    features['grad_max']  = float(grad_m[mask].max())

    # Ratio couverture
    features['coverage_ratio'] = float(mask.mean())

    # Dimensions physiques
    features['height_m']       = float(data.shape[0] * 0.2)
    features['width_pixels_m'] = float(data.shape[1] * 0.2)

    return list(features.values())


class _WidthMLP(nn.Module):
    """Architecture MLP identique au notebook model_t2.ipynb."""
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


def predict_t2(npz_path: str, models_dir: str) -> dict:
    """
    Tâche 2 — Largeur carte magnétique.
    Pipeline : features physiques → StandardScaler → MLP
    Modèles  : task2_model.pth + task2_scaler.pkl
    """
    scaler     = joblib.load(os.path.join(models_dir, 'task2_scaler.pkl'))
    feat       = np.array(_extract_features_t2(npz_path), dtype=np.float32).reshape(1, -1)
    n_features = feat.shape[1]
    feat_sc    = scaler.transform(feat)

    model = _WidthMLP(n_features=n_features)
    model.load_state_dict(torch.load(
        os.path.join(models_dir, 'task2_model.pth'),
        map_location='cpu', weights_only=True))
    model.eval()

    with torch.no_grad():
        x      = torch.tensor(feat_sc, dtype=torch.float32)
        width_m = float(model(x).item())

    width_m = float(np.clip(width_m, 2.0, 155.0))
    return {'T2_largeur_m': round(width_m, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# T3 — Couverture suffisante : CNN 4 canaux (notebook prof)
# ─────────────────────────────────────────────────────────────────────────────

class _PipelineCNN(nn.Module):
    """Architecture CNN identique au notebook model_t3.ipynb."""
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
        return self.classifier(self.gap(self.block3(self.block2(self.block1(x)))))


def predict_t3(npz_path: str, models_dir: str) -> dict:
    """
    Tâche 3 — Couverture suffisante.
    Pipeline : NPZ → normalisation locale → resize 64×64 → CNN
    Modèle   : task3_model.pth
    0 = couverture insuffisante  |  1 = couverture suffisante
    """
    model = _PipelineCNN()
    model.load_state_dict(torch.load(
        os.path.join(models_dir, 'task3_model.pth'),
        map_location='cpu', weights_only=True))
    model.eval()

    # Prétraitement identique au notebook
    arr  = load_npz_raw(npz_path)             # (H, W, 4)
    data = np.nan_to_num(arr, nan=0.0)
    for c in range(data.shape[2]):
        canal = data[:, :, c]
        cmin, cmax = canal.min(), canal.max()
        if cmax - cmin > 0:
            data[:, :, c] = (canal - cmin) / (cmax - cmin)
        else:
            data[:, :, c] = 0.0

    # Resize 64×64 avec cv2 (comme le notebook)
    data_r = cv2.resize(data, (64, 64), interpolation=cv2.INTER_LINEAR)
    t = torch.tensor(data_r.copy()).permute(2, 0, 1).unsqueeze(0).float()  # (1,4,64,64)

    with torch.no_grad():
        logits = model(t)
        pred   = int(logits.argmax(dim=1).item())
        prob   = float(torch.softmax(logits, dim=1)[0, 1].item())

    return {
        'T3_couverture' : pred,
        'T3_probabilite': round(prob, 4),
        'T3_label'      : 'Couverture suffisante' if pred == 1
                          else 'Couverture insuffisante',
    }


# ─────────────────────────────────────────────────────────────────────────────
# T4 — Conduites parallèles : CNN 4 canaux (notre modèle)
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class _ModelT4(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            _ConvBlock(4, 16, stride=2),
            _ConvBlock(16, 32, stride=2),
            _ConvBlock(32, 64, stride=2),
            _ConvBlock(64, 128, stride=2),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.6),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.head(self.gap(self.backbone(x)).flatten(1))


def predict_t4(npz_path: str, models_dir: str) -> dict:
    """
    Tâche 4 — Conduites parallèles.
    Pipeline : NPZ → normalisation locale → resize 128×128 → CNN
    """
    th_path   = os.path.join(models_dir, 'threshold_t4.json')
    threshold = json.load(open(th_path))['threshold'] \
                if os.path.exists(th_path) else 0.5

    model = _ModelT4()
    model.load_state_dict(torch.load(
        os.path.join(models_dir, 'model_t4.pt'),
        map_location='cpu', weights_only=True))
    model.eval()

    arr    = load_npz_raw(npz_path)   # (H, W, 4)
    normed = normalize_local_4ch(arr)
    t = torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).float()
    t = nn.functional.interpolate(t, size=(128, 128),
                                   mode='bilinear', align_corners=False)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).item()

    pred = int(prob >= threshold)
    return {
        'T4_parallel'   : pred,
        'T4_probabilite': round(prob, 4),
        'T4_label'      : 'Conduites parallèles' if pred else 'Conduite simple',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────────────────────────────────────

def predict(npz_path: str, models_dir: str = DEFAULT_MODELS_DIR) -> dict:
    """
    Prédit les 4 tâches sur un fichier NPZ.

    Logique :
      T1 — toujours
      T2 — toujours
      T3 — seulement si T1=1 (conduite présente)
      T4 — seulement si T1=1 (conduite présente)
    """
    result = {'fichier': os.path.basename(npz_path)}

    # T1
    r1 = predict_t1(npz_path, models_dir)
    result.update(r1)

    # T2
    r2 = predict_t2(npz_path, models_dir)
    result.update(r2)

    # T3 + T4 uniquement si conduite détectée
    if r1['T1_conduite'] == 1:
        result.update(predict_t3(npz_path, models_dir))
        result.update(predict_t4(npz_path, models_dir))
    else:
        result.update({
            'T3_couverture' : None,
            'T3_probabilite': None,
            'T3_label'      : 'N/A (aucune conduite)',
            'T4_parallel'   : None,
            'T4_probabilite': None,
            'T4_label'      : 'N/A (aucune conduite)',
        })

    return result


def predict_batch(npz_dir: str, models_dir: str = DEFAULT_MODELS_DIR,
                  output_csv: str = None) -> pd.DataFrame:
    """Traite tous les fichiers real_data_*.npz d'un dossier."""
    from tqdm import tqdm

    files        = [f for f in os.listdir(npz_dir)
                    if f.endswith('.npz') and f.startswith('real_data')]
    with_pipe    = [f for f in files if 'no_pipe' not in f]
    without_pipe = [f for f in files if 'no_pipe' in f]

    print(f"[Batch] {len(files)} fichiers real_data trouvés")
    print(f"  Avec conduite (ground truth)  : {len(with_pipe)}")
    print(f"  Sans conduite (ground truth)  : {len(without_pipe)}")

    results = []
    for fname in tqdm(files, desc="Inférence"):
        path = os.path.join(npz_dir, fname)
        try:
            r = predict(path, models_dir)
            results.append(r)
        except Exception as e:
            print(f"  ✗ {fname} : {e}")
            results.append({'fichier': fname, 'erreur': str(e)})

    df = pd.DataFrame(results)

    if output_csv:
        df.to_csv(output_csv, index=False, sep=';')
        print(f"\n  Résultats → {output_csv}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Skipper NDT — Inférence T1 + T2 + T3 + T4')
    parser.add_argument('--npz',        type=str)
    parser.add_argument('--dossier',    type=str)
    parser.add_argument('--models_dir', type=str, default=DEFAULT_MODELS_DIR)
    parser.add_argument('--output_csv', type=str, default=None)
    args = parser.parse_args()

    if args.npz:
        print(f"\nAnalyse : {args.npz}")
        result = predict(args.npz, args.models_dir)

        print("=" * 55)
        print("RÉSULTATS")
        print("=" * 55)
        print(f"  Fichier  : {result['fichier']}")
        print()
        print(f"  T1 — Détection conduite")
        print(f"    {'✓' if result['T1_conduite'] else '○'} {result['T1_label']}"
              f"  (prob={result['T1_probabilite']})")
        print()
        print(f"  T2 — Largeur carte")
        print(f"    {result['T2_largeur_m']} m")
        print()
        print(f"  T3 — Couverture")
        print(f"    {result['T3_label']}"
              + (f"  (prob={result['T3_probabilite']})"
                 if result['T3_probabilite'] is not None else ''))
        print()
        print(f"  T4 — Type de conduite")
        print(f"    {result['T4_label']}"
              + (f"  (prob={result['T4_probabilite']})"
                 if result['T4_probabilite'] is not None else ''))
        print("=" * 55)
        print()
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.dossier:
        output = args.output_csv or os.path.join(
            args.dossier, 'resultats_inference.csv')
        df = predict_batch(args.dossier, args.models_dir, output)

        print(f"\n{'='*55}")
        print("RÉSUMÉ FINAL")
        print(f"{'='*55}")
        print(f"  Total traité              : {len(df)}")
        if 'T1_conduite' in df.columns:
            print(f"  T1 — Conduite détectée    : {int(df['T1_conduite'].sum())}")
            print(f"  T1 — Aucune conduite      : {int((df['T1_conduite']==0).sum())}")
        if 'T3_couverture' in df.columns:
            print(f"  T3 — Couverture suffisante : {int((df['T3_couverture']==1).sum())}")
            print(f"  T3 — Insuffisante          : {int((df['T3_couverture']==0).sum())}")
        if 'T4_parallel' in df.columns:
            print(f"  T4 — Parallèles            : {int((df['T4_parallel']==1).sum())}")
            print(f"  T4 — Simples               : {int((df['T4_parallel']==0).sum())}")
        if 'T2_largeur_m' in df.columns:
            print(f"  T2 — Largeur moy           : {df['T2_largeur_m'].mean():.1f}m")
        print(f"{'='*55}")

    else:
        parser.print_help()
        print()
        print("Exemples :")
        print("  python inference.py --npz real_data_00045.npz")
        print("  python inference.py --dossier ./Training_database_float16/")


if __name__ == '__main__':
    main()