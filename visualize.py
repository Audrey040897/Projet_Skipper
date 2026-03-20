"""
visualize.py — Skipper NDT — Visualisation des données
========================================================
Affiche une image NPZ sur les 4 canaux (Bx, By, Bz, Norme)
AVANT et APRÈS suppression des NaN.

Usage :
  python visualize.py --npz mon_image.npz
  python visualize.py --npz mon_image.npz --save
  python visualize.py  (utilise un fichier par défaut)

Fichier généré (si --save) :
  visualisation_canaux.png
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(npz_path: str):
    """Charge un NPZ et retourne (H, W, 4) float32."""
    data = np.load(npz_path, allow_pickle=True)
    return data['data'].astype(np.float32)   # (H, W, 4)


def preprocess(arr: np.ndarray) -> np.ndarray:
    """
    Supprime les NaN et normalise chaque canal dans [0, 1].
    NaN → 0 (hors-champ).
    """
    H, W, C = arr.shape
    result   = np.zeros_like(arr)
    for c in range(C):
        ch    = arr[:, :, c].copy()
        valid = np.isfinite(ch)
        if valid.any():
            cmin = ch[valid].min()
            cmax = ch[valid].max()
            rng  = cmax - cmin
            if rng > 1e-8:
                ch[valid] = (ch[valid] - cmin) / rng
            else:
                ch[valid] = 0.5
        ch[~valid] = 0.0   # NaN → 0
        result[:, :, c] = ch
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize(npz_path: str, save: bool = False,
              output_dir: str = '.') -> str:
    """
    Génère une figure avec 2 lignes × 4 colonnes :
      Ligne 1 : image BRUTE (avec NaN → blanc)
      Ligne 2 : image APRÈS traitement (NaN → 0, normalisée)

    Args:
        npz_path   : chemin vers le fichier NPZ
        save       : True pour sauvegarder en PNG
        output_dir : dossier de sauvegarde

    Returns:
        Chemin du PNG sauvegardé (ou None)
    """
    canal_names = ['Bx', 'By', 'Bz', 'Norme']
    fname       = os.path.basename(npz_path)

    # Chargement
    arr_brut = load_npz(npz_path)         # (H, W, 4) avec NaN
    arr_proc = preprocess(arr_brut)       # (H, W, 4) sans NaN, normalisé
    H, W, _  = arr_brut.shape

    nan_pct  = np.isnan(arr_brut[:, :, 3]).mean() * 100  # % NaN canal Norme
    label_t1 = "avec conduite" if "no_pipe" not in fname else "sans conduite"

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(
        f'{fname}  —  {H}×{W}px  |  NaN={nan_pct:.1f}%  |  {label_t1}',
        fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.25)

    for c, name in enumerate(canal_names):

        # Ligne 1 : brut avec NaN
        ax1 = fig.add_subplot(gs[0, c])
        ch_brut = arr_brut[:, :, c].copy()
        # NaN → blanc dans la colormap
        cmap = plt.cm.jet.copy()
        cmap.set_bad(color='white')
        ch_masked = np.ma.masked_invalid(ch_brut)
        im1 = ax1.imshow(ch_masked, cmap=cmap, aspect='auto')
        ax1.set_title(f'{name} — brut', fontsize=10, fontweight='bold')
        ax1.set_xlabel(f'NaN : {np.isnan(ch_brut).mean()*100:.1f}%',
                       fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Ligne 2 : après traitement
        ax2 = fig.add_subplot(gs[1, c])
        ch_proc = arr_proc[:, :, c]
        im2 = ax2.imshow(ch_proc, cmap='jet', aspect='auto',
                          vmin=0, vmax=1)
        ax2.set_title(f'{name} — traité', fontsize=10, fontweight='bold')
        ax2.set_xlabel('NaN → 0  |  normalisé [0,1]', fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Labels lignes
    fig.text(0.01, 0.75, 'AVANT\n(brut)',
             va='center', ha='left', fontsize=11,
             fontweight='bold', color='#c0392b',
             rotation=90)
    fig.text(0.01, 0.27, 'APRÈS\n(traité)',
             va='center', ha='left', fontsize=11,
             fontweight='bold', color='#27ae60',
             rotation=90)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_name = fname.replace('.npz', '_canaux.png')
        out_path = os.path.join(output_dir, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Sauvegardé → {out_path}")
        return out_path
    else:
        plt.savefig('visualisation_canaux.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Sauvegardé → visualisation_canaux.png")
        return 'visualisation_canaux.png'


# ─────────────────────────────────────────────────────────────────────────────
# Stats complémentaires
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(npz_path: str):
    """Affiche les statistiques de l'image dans le terminal."""
    arr   = load_npz(npz_path)
    fname = os.path.basename(npz_path)
    H, W, _ = arr.shape

    print(f"\n{'='*55}")
    print(f"  {fname}")
    print(f"  Shape : {H} × {W} px  "
          f"({H*0.2:.1f}m × {W*0.2:.1f}m)")
    print(f"{'='*55}")
    print(f"  {'Canal':>8} | {'Min':>10} | {'Max':>10} | "
          f"{'Moyenne':>10} | {'NaN%':>6}")
    print(f"  " + "-"*55)
    for c, name in enumerate(['Bx', 'By', 'Bz', 'Norme']):
        ch    = arr[:, :, c]
        valid = ch[np.isfinite(ch)]
        nan_p = np.isnan(ch).mean() * 100
        if len(valid) > 0:
            print(f"  {name:>8} | {valid.min():>10.3f} | "
                  f"{valid.max():>10.3f} | "
                  f"{valid.mean():>10.3f} | {nan_p:>5.1f}%")
        else:
            print(f"  {name:>8} | {'N/A':>10} | {'N/A':>10} | "
                  f"{'N/A':>10} | {nan_p:>5.1f}%")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Skipper NDT — Visualisation des 4 canaux avant/après traitement')
    parser.add_argument('--npz',        type=str,
                        help='Chemin vers le fichier NPZ')
    parser.add_argument('--save',       action='store_true',
                        help='Sauvegarder en PNG')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Dossier de sauvegarde')
    args = parser.parse_args()

    # Fichier par défaut si non spécifié
    npz_path = args.npz or (
        r"D:/Projet_skipper_RNDT/Projet_Skipper/Data_NDT/"
        r"Training_database_float16/sample_00125_perfect_straight_clean_field.npz"
    )

    if not os.path.exists(npz_path):
        print(f"Fichier non trouvé : {npz_path}")
        print("Utilisez : python visualize.py --npz chemin/vers/image.npz")
        exit(1)

    # Statistiques
    print_stats(npz_path)

    # Visualisation
    print("Génération de la figure (2 lignes × 4 canaux)...")
    visualize(npz_path, save=args.save or True,
              output_dir=args.output_dir or '.')