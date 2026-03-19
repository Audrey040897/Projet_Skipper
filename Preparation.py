"""
prepare_dataset.py — Skipper NDT
==================================
Script unique qui fait TOUT en une seule passe :

  Étape 1 : Calcul des statistiques globales Min/Max
            sur TOUT le dataset synthétique (hors real_data)

  Étape 2 : Pour chaque image NPZ :
              - Chargement (H,W,4) float16 → float32
              - Remplacement NaN → 0
              - Normalisation Min-Max GLOBALE → [0, 1]
              - Sauvegarde en tenseur .pt (4, H, W)

  Étape 3 : Génération du manifest.csv avec tous les labels

RÉSULTAT :
  cache_pt/
  ├── tensors/          ← un .pt par image NPZ
  │   ├── sample_00000_perfect_straight_clean_field.pt
  │   └── ...
  ├── manifest.csv      ← CSV enrichi avec chemin .pt + labels
  └── global_stats.json ← min/max globaux (pour l'inférence)

UTILISATION :
  python prepare_dataset.py
  (modifier les chemins en bas du fichier)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement brut d'un NPZ → (4, H, W) float32
# ─────────────────────────────────────────────────────────────────────────────

def load_npz_raw(path: str) -> np.ndarray:
    """
    Charge un fichier NPZ et retourne (4, H, W) float32 SANS normalisation.
    NaN conservés à ce stade (nécessaires pour calculer les stats globales).
    """
    data = np.load(path, allow_pickle=True)
    arr  = data['data'].astype(np.float32)   # float16 → float32 (évite overflow)
    return np.transpose(arr, (2, 0, 1))       # (H, W, 4) → (4, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Étape 1 : Calcul des stats globales Min/Max sur tout le dataset
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_stats(filenames: list, npz_dir: str) -> dict:
    """
    Parcourt tous les fichiers NPZ et calcule le min et max global
    par canal (en ignorant les NaN).

    Returns:
        {'global_min': [min_c0, min_c1, min_c2, min_c3],
         'global_max': [max_c0, max_c1, max_c2, max_c3]}
    """
    print("\n[Étape 1/2] Calcul des statistiques globales Min/Max...")
    print(f"  {len(filenames)} fichiers à analyser\n")

    # Initialisation
    global_min = np.full(4,  np.inf, dtype=np.float64)
    global_max = np.full(4, -np.inf, dtype=np.float64)
    n_nan_total = 0

    for fname in tqdm(filenames, desc="Stats globales"):
        path = os.path.join(npz_dir, fname)
        try:
            img = load_npz_raw(path)           # (4, H, W)
            n_nan_total += int(np.isnan(img).sum())

            for c in range(4):
                ch    = img[c]
                valid = ch[np.isfinite(ch)]
                if len(valid) == 0:
                    continue
                global_min[c] = min(global_min[c], float(valid.min()))
                global_max[c] = max(global_max[c], float(valid.max()))

        except Exception as e:
            print(f"  ⚠  Erreur sur {fname} : {e}")

    print(f"\n  NaN totaux rencontrés : {n_nan_total:,}")
    print(f"\n  Statistiques globales par canal :")
    canal_names = ['Bx', 'By', 'Bz', 'Norme']
    for c in range(4):
        print(f"    {canal_names[c]} : min={global_min[c]:.4f}  max={global_max[c]:.4f}")

    return {
        'global_min': global_min.tolist(),
        'global_max': global_max.tolist(),
        'canal_names': canal_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prétraitement final : NaN→0 + normalisation Min-Max globale
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(img: np.ndarray, global_min: list, global_max: list,
               eps: float = 1e-8) -> np.ndarray:
    """
    Applique le prétraitement final sur une image (4, H, W) :
      1. NaN / Inf → 0.0
      2. Normalisation Min-Max GLOBALE par canal → [0, 1]

    Args:
        img        : (4, H, W) float32
        global_min : liste de 4 valeurs (min global par canal)
        global_max : liste de 4 valeurs (max global par canal)

    Returns:
        (4, H, W) float32 dans [0, 1], sans NaN
    """
    normed = np.zeros_like(img)   # NaN remplacés par 0

    for c in range(4):
        ch      = img[c]
        valid   = np.isfinite(ch)
        cmin    = global_min[c]
        cmax    = global_max[c]
        rng     = cmax - cmin

        if rng < eps:
            normed[c][valid] = 0.5
        else:
            normed[c][valid] = (ch[valid] - cmin) / rng

        # Clamp dans [0, 1] au cas où
        normed[c] = np.clip(normed[c], 0.0, 1.0)

    return normed


# ─────────────────────────────────────────────────────────────────────────────
# 4. Étape 2 : Conversion NPZ → .pt avec normalisation globale
# ─────────────────────────────────────────────────────────────────────────────

# Taille cible après resize
# 128px : trop petit pour T2 (1px ≈ 1.4m → MAE < 1m impossible)
# 256px : 1px ≈ 0.7m → précision suffisante pour MAE < 1m
TARGET_SIZE = 256


def resize_tensor(img: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    """
    Redimensionne (4, H, W) en conservant le ratio d'aspect.
    La plus grande dimension est ramenée à target px.
    Utilise une interpolation bilinéaire simple via numpy.
    """
    import cv2   # pip install opencv-python
    C, H, W = img.shape
    scale   = target / max(H, W)
    new_h   = max(1, int(H * scale))
    new_w   = max(1, int(W * scale))

    resized = np.zeros((C, new_h, new_w), dtype=np.float32)
    for c in range(C):
        resized[c] = cv2.resize(img[c], (new_w, new_h),
                                interpolation=cv2.INTER_LINEAR)
    return resized


def convert_to_pt(filenames: list, npz_dir: str,
                  out_dir: str, stats: dict) -> list:
    """
    Convertit chaque NPZ en tenseur .pt normalisé ET redimensionné.
    Les tenseurs sont stockés à TARGET_SIZE px max → ultra rapides à charger.

    Returns:
        Liste des noms de fichiers .pt créés avec succès
    """
    tensors_dir = os.path.join(out_dir, 'tensors')
    os.makedirs(tensors_dir, exist_ok=True)

    global_min = stats['global_min']
    global_max = stats['global_max']

    print(f"\n[Étape 2/2] Conversion NPZ → .pt normalisés + resize {TARGET_SIZE}px...")
    print(f"  Dossier sortie : {tensors_dir}\n")

    success = []
    errors  = []

    for fname in tqdm(filenames, desc="Conversion"):
        npz_path = os.path.join(npz_dir, fname)
        pt_name  = os.path.splitext(fname)[0] + '.pt'
        pt_path  = os.path.join(tensors_dir, pt_name)

        # Passer si déjà converti (reprise possible)
        if os.path.exists(pt_path):
            success.append(pt_name)
            continue

        try:
            img = load_npz_raw(npz_path)              # (4, H, W) float32 avec NaN
            img = preprocess(img, global_min, global_max)  # NaN→0, normalisation
            img = resize_tensor(img, TARGET_SIZE)     # resize → (4, ≤128, ≤128)
            tensor = torch.from_numpy(img)            # (4, H', W') float32
            torch.save(tensor, pt_path)
            success.append(pt_name)

        except Exception as e:
            errors.append(fname)
            tqdm.write(f"  ✗ Erreur {fname} : {e}")

    print(f"\n  ✓ Convertis : {len(success)}/{len(filenames)}")
    if errors:
        print(f"  ✗ Erreurs   : {len(errors)} fichiers")
        for e in errors[:5]:
            print(f"      {e}")

    return success


# ─────────────────────────────────────────────────────────────────────────────
# 5. Génération du manifest.csv
# ─────────────────────────────────────────────────────────────────────────────

def build_manifest(df_csv: pd.DataFrame, pt_names_ok: list,
                   out_dir: str) -> pd.DataFrame:
    """
    Construit le manifest.csv en joignant les labels CSV
    avec les fichiers .pt générés.

    Colonnes du manifest :
      pt_file    : chemin relatif vers le .pt  (ex: tensors/sample_00000_...pt)
      filename   : nom du NPZ original
      label      : 0/1 présence conduite (T1)
      width_m    : largeur en mètres (T2) — NaN si no_pipe
      pipe_type  : 'single' ou 'parallel' (T4)
      task4_label: 0/1 (parallel=1)
      noisy      : True/False
      shape      : 'straight' ou 'curved'
      coverage_type : 'perfect', 'offset', 'missed', 'no_pipe'
    """
    # Index des .pt créés avec succès
    pt_set = set(pt_names_ok)

    rows = []
    skipped = 0

    for _, row in df_csv.iterrows():
        fname   = row['filename']
        pt_name = os.path.splitext(fname)[0] + '.pt'

        if pt_name not in pt_set:
            skipped += 1
            continue

        rows.append({
            'pt_file'      : os.path.join('tensors', pt_name),
            'filename'     : fname,
            'label'        : int(row['label']),
            'width_m'      : row['width_m'] if pd.notna(row.get('width_m')) else None,
            'pipe_type'    : row.get('pipe_type', ''),
            'task4_label'  : int(row.get('pipe_type', 'single') == 'parallel'),
            'noisy'        : bool(row.get('noisy', False)),
            'shape'        : row.get('shape', ''),
            'coverage_type': row.get('coverage_type', ''),
        })

    manifest = pd.DataFrame(rows)
    manifest_path = os.path.join(out_dir, 'manifest.csv')
    manifest.to_csv(manifest_path, index=False)

    print(f"\n  manifest.csv : {len(manifest)} lignes")
    if skipped:
        print(f"  ⚠  {skipped} fichiers ignorés (conversion échouée)")
    print(f"\n  Distribution des labels :")
    print(f"    T1 — label=1 : {(manifest['label']==1).sum()}"
          f"  label=0 : {(manifest['label']==0).sum()}")
    print(f"    T2 — width_m valides : {manifest['width_m'].notna().sum()}")
    print(f"    T4 — parallel : {(manifest['task4_label']==1).sum()}"
          f"  single : {(manifest['task4_label']==0).sum()}")

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def prepare(csv_path: str, npz_dir: str, out_dir: str):
    """
    Lance tout le pipeline de préparation du dataset.

    Args:
        csv_path : pipe_detection_label.csv
        npz_dir  : dossier contenant les fichiers .npz
        out_dir  : dossier de sortie pour les .pt et manifest
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Lecture du CSV ────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, sep=';')
    if 'field_file' in df.columns:
        df = df.rename(columns={'field_file': 'filename'})

    # Exclure les real_data (pas dans le CSV — uniquement synthétiques)
    df_synth = df[~df['filename'].str.startswith('real_data')].reset_index(drop=True)
    print(f"\n[prepare_dataset]")
    print(f"  CSV total      : {len(df)} lignes")
    print(f"  Synthétiques   : {len(df_synth)} lignes (real_data exclus)")

    filenames = df_synth['filename'].tolist()

    # ── Étape 1 : Stats globales ──────────────────────────────────────────────
    stats_path = os.path.join(out_dir, 'global_stats.json')

    if os.path.exists(stats_path):
        print(f"\n  Stats globales déjà calculées → chargement depuis {stats_path}")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        canal_names = ['Bx', 'By', 'Bz', 'Norme']
        for c in range(4):
            print(f"    {canal_names[c]} : min={stats['global_min'][c]:.4f}"
                  f"  max={stats['global_max'][c]:.4f}")
    else:
        stats = compute_global_stats(filenames, npz_dir)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Stats sauvegardées → {stats_path}")

    # ── Étape 2 : Conversion NPZ → .pt ───────────────────────────────────────
    pt_ok = convert_to_pt(filenames, npz_dir, out_dir, stats)

    # ── Étape 3 : Manifest ────────────────────────────────────────────────────
    print(f"\n[Étape 3/3 — Manifest]")
    manifest = build_manifest(df_synth, pt_ok, out_dir)

    # ── Résumé final ──────────────────────────────────────────────────────────
    tensors_dir = os.path.join(out_dir, 'tensors')
    n_pt  = len([f for f in os.listdir(tensors_dir) if f.endswith('.pt')])
    size  = sum(
        os.path.getsize(os.path.join(tensors_dir, f))
        for f in os.listdir(tensors_dir) if f.endswith('.pt')
    ) / 1e9

    print(f"\n{'='*55}")
    print(f"  Terminé !")
    print(f"  Fichiers .pt créés : {n_pt}")
    print(f"  Taille totale      : {size:.1f} Go")
    print(f"  Stats globales     : {stats_path}")
    print(f"  Manifest           : {os.path.join(out_dir, 'manifest.csv')}")
    print(f"{'='*55}")
    print(f"\nDans model.py, utilisez :")
    print(f"  CSV_PATH  = r\"{os.path.join(out_dir, 'manifest.csv')}\"")
    print(f"  IMG_DIR   = r\"{out_dir}\"")
    print(f"  USE_CACHE = True")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ══════════════════════════════════════════════════════════
    # ⚠️  MODIFIEZ CES 3 CHEMINS
    # ══════════════════════════════════════════════════════════
    CSV_PATH = r"D:/Projet_skipper_RNDT/Projet_Skipper/Data_NDT/Training_database_float16/pipe_presence_width_detection_label.csv"
    NPZ_DIR  = r"D:/Projet_skipper_RNDT/Projet_Skipper/Data_NDT/Training_database_float16/"
    OUT_DIR  = r"D:/Projet_skipper_RNDT/Projet_Skipper/cache_pt/"
    # ══════════════════════════════════════════════════════════

    prepare(CSV_PATH, NPZ_DIR, OUT_DIR)