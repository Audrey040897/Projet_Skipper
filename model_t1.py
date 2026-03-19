"""
model_t1.py — Skipper NDT — Tâche 1 : Détection de conduite
=============================================================
Approche du professeur adaptée :
  - Canal Bz uniquement (canal 2, signature verticale de la conduite)
  - Resize 224×224 (standard vision)
  - NaN → 0
  - Flatten → PCA → KNN

Pipeline :
  NPZ → canal Bz (H,W) → resize (224,224) → nan_to_num → flatten (50176,)
      → PCA(N_COMPONENTS) → KNN

Différence vs prof : on teste plusieurs N_COMPONENTS (2, 10, 20, 50, 100)
pour trouver le meilleur au lieu de fixer à 2.
PCA 2D est gardé pour la visualisation.

Objectifs :
  Accuracy > 92%  ·  Recall > 95%

Fichiers générés :
  pca_2d_visu_t1.png       — visualisation PCA 2D (comme le prof)
  training_curves_t1.png   — Accuracy + Recall vs K et vs N_COMPONENTS
  model_t1_knn.pkl         — meilleur KNN
  pca_t1.pkl               — PCA sklearn
  threshold_t1.json        — seuil + config optimale
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize as sk_resize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              recall_score, classification_report)


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres (fidèles à l'approche du prof)
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE    = 224   # resize identique au prof
CANAL       = 2     # Bz — canal 2 (comme le prof)
K_VALUES    = [3, 5, 7, 10, 15, 20]   # K à tester
N_PCA_LIST  = [2, 10, 20, 50, 100]    # N_COMPONENTS à tester


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement d'un NPZ → vecteur 1D (méthode du prof)
# ─────────────────────────────────────────────────────────────────────────────

def load_npz_vector(npz_path: str) -> np.ndarray:
    """
    Charge un NPZ et retourne le vecteur aplati du canal Bz.

    Pipeline (données brutes — comme le prof) :
      data['data'] → float32 → canal Bz → nan_to_num(0) → resize(224,224) → flatten

    PAS de normalisation : la PCA travaille sur les valeurs brutes nT.
    C'est la PCA elle-même qui centre les données (soustrait la moyenne).
    """
    data  = np.load(npz_path, allow_pickle=True)
    array = data['data'].astype(np.float32)       # (H, W, 4) valeurs brutes nT
    array = array[:, :, CANAL]                    # canal Bz → (H, W)
    array = np.nan_to_num(array, nan=0.0)         # NaN → 0 (seul prétraitement)
    array = sk_resize(array, (IMG_SIZE, IMG_SIZE),
                      anti_aliasing=True)          # (224, 224)
    return array.flatten()                         # (50176,)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Construction du dataset depuis le cache .pt ou les NPZ
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(manifest_path: str, npz_dir: str,
                  output_dir: str) -> tuple:
    """
    Construit X (N, 50176) et y (N,) depuis les fichiers NPZ.
    Cache dans full_data_t1.npz.

    IMPORTANT : PCA est calculée UNIQUEMENT sur X_train après le split
    pour éviter le data leakage (pas de fuite d'info test → train).

    Returns:
        X (N, IMG_SIZE²), y (N,)
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, 'full_data_t1.npz')

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        X, y = data['images'], data['labels']
        print(f"[Dataset] Cache chargé : {X.shape}")
        return X, y

    print(f"[Dataset] Construction depuis NPZ...")
    print(f"  Pipeline : canal Bz → nan_to_num → resize {IMG_SIZE}px → flatten (données brutes)")
    df = pd.read_csv(manifest_path)
    if 'field_file' in df.columns:
        df = df.rename(columns={'field_file': 'filename'})

    images, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Chargement"):
        npz_path = os.path.join(npz_dir, row['filename'])
        if not os.path.exists(npz_path):
            continue
        try:
            vec = load_npz_vector(npz_path)
            images.append(vec)
            labels.append(int(row['label']))
        except Exception as e:
            tqdm.write(f"  ✗ {row['filename']} : {e}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=int)

    np.savez(cache_path, images=X, labels=y)
    print(f"  Dataset sauvegardé → {cache_path}")
    print(f"  Shape : {X.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualisation PCA 2D 
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_2d(X: np.ndarray, y: np.ndarray, output_dir: str):
    """
    Visualisation PCA 2D identique au prof :
      classe 0 → rouge, classe 1 → jaune
    """
    print("[PCA 2D] Visualisation...")
    pca2   = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X)
    var    = pca2.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    class_0 = X_pca2[~y.astype(bool)]
    class_1 = X_pca2[ y.astype(bool)]
    ax.scatter(class_0[:, 0], class_0[:, 1],
               color='red',    label='Classe 0 (absent)',  alpha=0.4, s=8)
    ax.scatter(class_1[:, 0], class_1[:, 1],
               color='#FFD700', label='Classe 1 (présent)', alpha=0.4, s=8)
    ax.set_title('PCA 2D + KNN classification\n'
                 f'PC1={var[0]:.1f}%  PC2={var[1]:.1f}% variance')
    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pca_2d_visu_t1.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  PCA 2D → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Recherche meilleur N_PCA + K
# ─────────────────────────────────────────────────────────────────────────────

def search_best_config(X_train, y_train, X_val, y_val) -> dict:
    """
    Teste toutes les combinaisons (N_PCA, K) et retourne la meilleure.
    Critère : Recall maximal avec Accuracy > 0.85.
    """
    print(f"\n--- Recherche meilleur N_PCA × K ---")
    print(f"  {'N_PCA':>6} | {'K':>4} | {'Acc':>8} | {'Recall':>8} | {'F1':>8}")
    print("  " + "-" * 45)

    best_cfg   = None
    best_score = 0.0
    all_results = []

    for n_pca in N_PCA_LIST:
        # PCA fit UNIQUEMENT sur X_train → transform sur val/test séparément
        # Évite le data leakage : le test ne doit jamais influencer la PCA
        pca    = PCA(n_components=n_pca, random_state=42)
        Xtr_p  = pca.fit_transform(X_train)   # fit + transform train
        Xvl_p  = pca.transform(X_val)         # transform uniquement

        for k in K_VALUES:
            knn = KNeighborsClassifier(
                n_neighbors = k,
                weights     = 'distance',
                metric      = 'euclidean',
                n_jobs      = -1,
            )
            knn.fit(Xtr_p, y_train)

            # Seuil fixe 0.5
            probs  = knn.predict_proba(Xvl_p)[:, 1]
            best_th = 0.5
            preds   = (probs >= best_th).astype(int)
            acc   = float(accuracy_score(y_val, preds))
            rec   = float(recall_score(y_val, preds, zero_division=0))
            f1    = float(f1_score(y_val, preds, zero_division=0))

            mark = " ✓✓" if acc>0.92 and rec>0.95 else \
                   (" ✓"  if acc>0.90 or  rec>0.90 else "")
            print(f"  {n_pca:>6} | {k:>4} | {acc:>8.4f} | "
                  f"{rec:>8.4f} | {f1:>8.4f}{mark}")

            score = rec * 2 + acc
            all_results.append({
                'n_pca': n_pca, 'k': k, 'pca': pca, 'knn': knn,
                'threshold': best_th,
                'acc': acc, 'recall': rec, 'f1': f1, 'score': score,
            })
            if score > best_score:
                best_score = score
                best_cfg   = all_results[-1]

    print(f"\n  → Meilleur : N_PCA={best_cfg['n_pca']}  K={best_cfg['k']} | "
          f"Acc={best_cfg['acc']:.4f}  Rec={best_cfg['recall']:.4f}  "
          f"seuil={best_cfg['threshold']:.2f}")
    return best_cfg, all_results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Courbes
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(all_results: list, best_cfg: dict, output_dir: str):
    """
    Génère training_curves_t1.png :
      Gauche : Recall val vs K pour chaque N_PCA
      Droit  : Accuracy val vs K pour chaque N_PCA
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Tâche 1 — PCA + KNN  |  Recherche N_PCA × K',
                 fontsize=13, fontweight='bold')

    colors = ['b', 'g', 'r', 'orange', 'purple']

    for ax_idx, (metric, title, obj) in enumerate([
        ('recall',   'Recall vs K',   0.95),
        ('acc',      'Accuracy vs K', 0.92),
    ]):
        ax = axes[ax_idx]
        for i, n_pca in enumerate(N_PCA_LIST):
            rows = [r for r in all_results if r['n_pca'] == n_pca]
            rows.sort(key=lambda r: r['k'])
            ks   = [r['k']      for r in rows]
            vals = [r[metric]   for r in rows]
            lw   = 2.5 if n_pca == best_cfg['n_pca'] else 1.2
            ls   = '-' if n_pca == best_cfg['n_pca'] else '--'
            ax.plot(ks, vals, marker='o', ms=5, lw=lw, ls=ls,
                    color=colors[i % len(colors)],
                    label=f'PCA {n_pca} CP')

        ax.axhline(obj, color='black', lw=1.5, linestyle=':',
                   label=f'Objectif {obj}')
        ax.axvline(best_cfg['k'], color='red', lw=1.5, linestyle=':',
                   label=f"Meilleur K={best_cfg['k']}")
        ax.set_xlabel('K (nombre de voisins)')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves_t1.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Courbes → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entraînement principal
# ─────────────────────────────────────────────────────────────────────────────

def train_t1(manifest_path: str, npz_dir: str, output_dir: str,
             val_split: float = 0.15, test_split: float = 0.10,
             seed: int = 42):
    """
    Entraîne PCA + KNN pour la Tâche 1 (approche du professeur).

    Args:
        manifest_path : manifest.csv avec labels
        npz_dir       : dossier des fichiers .npz originaux
        output_dir    : dossier de sauvegarde

    Génère :
      pca_2d_visu_t1.png      visualisation PCA 2D
      training_curves_t1.png  Recall + Accuracy vs K pour chaque N_PCA
      model_t1_knn.pkl        meilleur KNN
      pca_t1.pkl              PCA avec le meilleur N_COMPONENTS
      threshold_t1.json       seuil + config optimale
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print("ENTRAÎNEMENT T1 — Pipeline PCA + KNN")
    print(f"  Canal : Bz (canal {CANAL})  |  "
          f"Resize : {IMG_SIZE}×{IMG_SIZE}px")
    print(f"{'='*55}")

    # ── Étape 1 : Dataset ─────────────────────────────────────────────────
    X, y = build_dataset(manifest_path, npz_dir, output_dir)
    print(f"\n[Dataset] {len(X)} échantillons | "
          f"positifs={y.sum()}  négatifs={(y==0).sum()}")

    # ── Étape 2 : Split ───────────────────────────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_split+test_split,
        random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_split/(val_split+test_split),
        random_state=seed, stratify=y_temp)

    print(f"Split : Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    # ── Étape 3 : Visualisation PCA 2D (comme le prof) ───────────────────
    plot_pca_2d(X_train, y_train, output_dir)

    # ── Étape 4 : Recherche meilleur N_PCA + K ────────────────────────────
    best_cfg, all_results = search_best_config(
        X_train, y_train, X_val, y_val)

    # ── Étape 5 : Courbes ─────────────────────────────────────────────────
    plot_curves(all_results, best_cfg, output_dir)

    # ── Étape 6 : Évaluation finale sur test set ──────────────────────────
    X_test_pca = best_cfg['pca'].transform(X_test)
    probs_test = best_cfg['knn'].predict_proba(X_test_pca)[:, 1]
    preds_test = (probs_test >= best_cfg['threshold']).astype(int)

    acc_test = float(accuracy_score(y_test, preds_test))
    rec_test = float(recall_score(y_test, preds_test, zero_division=0))
    f1_test  = float(f1_score(y_test, preds_test, zero_division=0))

    print(f"\n--- Évaluation finale T1 (test set) ---")
    print(f"  Config : N_PCA={best_cfg['n_pca']}  K={best_cfg['k']}  "
          f"seuil={best_cfg['threshold']:.2f}")
    print(f"  {'✓' if acc_test>0.92 else '✗'} Accuracy : {acc_test:.4f}  (> 0.92)")
    print(f"  {'✓' if rec_test>0.95 else '✗'} Recall   : {rec_test:.4f}  (> 0.95)")
    print(f"  F1-Score : {f1_test:.4f}")
    print()
    print(classification_report(y_test, preds_test,
                                 target_names=['absent', 'présent']))

    # ── Étape 7 : Sauvegarde ──────────────────────────────────────────────
    knn_path = os.path.join(output_dir, 'model_t1_knn.pkl')
    pca_path = os.path.join(output_dir, 'pca_t1.pkl')
    th_path  = os.path.join(output_dir, 'threshold_t1.json')

    joblib.dump(best_cfg['knn'], knn_path)
    joblib.dump(best_cfg['pca'], pca_path)
    with open(th_path, 'w') as f:
        json.dump({
            'threshold'  : float(best_cfg['threshold']),
            'model'      : 'KNN',
            'n_pca'      : best_cfg['n_pca'],
            'k'          : best_cfg['k'],
            'canal'      : CANAL,
            'img_size'   : IMG_SIZE,
            'acc_test'   : round(acc_test, 4),
            'recall_test': round(rec_test, 4),
            'f1_test'    : round(f1_test, 4),
        }, f, indent=2)

    print(f"  Modèle  → {knn_path}")
    print(f"  PCA     → {pca_path}")
    print(f"  Config  → {th_path}")

    return best_cfg['knn'], best_cfg['pca']


# ─────────────────────────────────────────────────────────────────────────────
# 7. Inférence
# ─────────────────────────────────────────────────────────────────────────────

def predict_t1(npz_path: str, output_dir: str) -> dict:
    """
    Prédit la présence de conduite sur un fichier NPZ.

    Pipeline :
      NPZ → canal Bz → nan_to_num → resize 224 → flatten → PCA → KNN

    Returns:
        {'pipeline_present': 0|1, 'probability': float, 'label': str}
    """
    knn = joblib.load(os.path.join(output_dir, 'model_t1_knn.pkl'))
    pca = joblib.load(os.path.join(output_dir, 'pca_t1.pkl'))
    cfg = json.load(open(os.path.join(output_dir, 'threshold_t1.json')))
    threshold = cfg['threshold']

    vec  = load_npz_vector(npz_path).reshape(1, -1)
    vec  = pca.transform(vec)
    prob = float(knn.predict_proba(vec)[0, 1])

    return {
        'pipeline_present': int(prob >= threshold),
        'probability'     : round(prob, 4),
        'label'           : 'Conduite détectée' if prob >= threshold
                            else 'Aucune conduite',
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ══════════════════════════════════════════════════════════
    # ⚠️  MODIFIEZ CES CHEMINS
    # ══════════════════════════════════════════════════════════
    MANIFEST   = r"D:/Projet_skipper_RNDT/Projet_Skipper/cache_pt/manifest.csv"
    NPZ_DIR    = r"D:/Projet_skipper_RNDT/Projet_Skipper/Data_NDT/Training_database_float16/"
    OUTPUT_DIR = r"D:/Projet_skipper_RNDT/Projet_Skipper/models/"
    # ══════════════════════════════════════════════════════════

    # Pour repartir de zéro :
    # del models\full_data_t1.npz

    train_t1(
        manifest_path = MANIFEST,
        npz_dir       = NPZ_DIR,
        output_dir    = OUTPUT_DIR,
    )