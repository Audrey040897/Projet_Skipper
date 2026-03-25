"""
dataset.py — Skipper NDT (version simplifiée)
===============================================
Lit les tenseurs .pt pré-normalisés générés par prepare_dataset.py.

Pas d'augmentation, pas de prétraitement à la volée.
Chaque __getitem__ = un simple torch.load() → ultra rapide.

Prérequis : avoir lancé prepare_dataset.py au moins une fois.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SkipperDataset(Dataset):
    """
    Dataset Skipper NDT — lit des tenseurs .pt pré-normalisés.

    Args:
        manifest_path : chemin vers manifest.csv généré par prepare_dataset.py
        cache_dir     : dossier racine du cache (contient le sous-dossier tensors/)
        tasks         : liste de tâches à retourner ['task1', 'task2', 'task4']
    """

    def __init__(self, manifest_path: str, cache_dir: str,
                 tasks: list = None):

        self.cache_dir = cache_dir
        self.tasks     = tasks or ['task1', 'task2', 'task4']

        df = pd.read_csv(manifest_path)
        self.df = df.reset_index(drop=True)

        n_pos = int((df['label'] == 1).sum())
        n_neg = int((df['label'] == 0).sum())
        print(f"\n[SkipperDataset] {len(df)} échantillons")
        print(f"  T1 — label=1 : {n_pos}  label=0 : {n_neg}")
        print(f"  T2 — width_m valides : {df['width_m'].notna().sum()}")
        print(f"  T4 — parallel : {(df['task4_label']==1).sum()}"
              f"  single : {(df['task4_label']==0).sum()}")

    def __len__(self):
        return len(self.df)

    def get_labels_t1(self) -> list:
        return self.df['label'].tolist()

    def __getitem__(self, idx: int) -> dict:
        row     = self.df.iloc[idx]
        pt_path = os.path.join(self.cache_dir,
                               str(row['pt_file']).replace('\\', '/'))

        # Chargement ultra-rapide du tenseur .pt
        img = torch.load(pt_path, weights_only=True)   # (4, H, W) float32

        sample = {
            'image'   : img,
            'filename': row['filename'],
        }

        if 'task1' in self.tasks:
            sample['task1'] = torch.tensor(int(row['label']), dtype=torch.long)

        if 'task2' in self.tasks:
            w = row['width_m']
            sample['task2'] = (torch.tensor(float(w), dtype=torch.float32)
                               if pd.notna(w) else None)

        if 'task4' in self.tasks:
            sample['task4'] = torch.tensor(int(row['task4_label']), dtype=torch.long)

        return sample


# ─────────────────────────────────────────────────────────────────────────────
# Collate : padding adaptatif + resize optionnel
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: list, max_size: int = 256) -> dict:
    """
    1. Resize chaque image (ratio conservé, max_size px sur la plus grande dim)
    2. Padding à la taille max du batch

    Args:
        max_size : taille max après resize (256 sur CPU, 512 sur GPU)
    """
    import torch.nn.functional as F

    def resize(img, max_size):
        _, h, w = img.shape
        if max(h, w) <= max_size:
            return img
        scale = max_size / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        return F.interpolate(img.unsqueeze(0), size=(nh, nw),
                             mode='bilinear', align_corners=False).squeeze(0)

    images = [resize(item['image'], max_size) for item in batch]
    max_h  = max(img.shape[1] for img in images)
    max_w  = max(img.shape[2] for img in images)

    padded = [
        torch.nn.functional.pad(img, (0, max_w - img.shape[2],
                                       0, max_h - img.shape[1]))
        for img in images
    ]

    result = {
        'image'   : torch.stack(padded),
        'filename': [item['filename'] for item in batch],
    }

    # T1
    if 'task1' in batch[0]:
        result['task1'] = torch.stack([item['task1'] for item in batch])

    # T2 — None → sentinelle -1
    if 'task2' in batch[0]:
        result['task2'] = torch.stack([
            item['task2'] if item['task2'] is not None
            else torch.tensor(-1.0)
            for item in batch
        ])

    # T4
    if 'task4' in batch[0]:
        result['task4'] = torch.stack([item['task4'] for item in batch])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(manifest_path: str, cache_dir: str,
                    tasks: list = None,
                    val_split: float = 0.15,
                    test_split: float = 0.10,
                    batch_size: int = 8,
                    max_size: int = 256,
                    num_workers: int = 0,
                    seed: int = 42) -> dict:
    """
    Crée les DataLoaders train / val / test.

    Args:
        manifest_path : manifest.csv généré par prepare_dataset.py
        cache_dir     : dossier racine du cache
        tasks         : tâches à inclure
        val_split     : proportion validation (défaut 15%)
        test_split    : proportion test (défaut 10%)
        batch_size    : taille de batch (8 avec cache .pt, 2-4 sans)
        max_size      : taille max image après resize (256 CPU, 512 GPU)
        num_workers   : 0 sur Windows, 2-4 sur Linux/Mac
        seed          : reproductibilité

    Returns:
        {'train': DataLoader, 'val': DataLoader, 'test': DataLoader,
         'class_weight_t1': Tensor, 'class_weight_t4': Tensor}
    """
    from functools import partial

    tasks = tasks or ['task1', 'task2', 'task4']
    ds    = SkipperDataset(manifest_path, cache_dir, tasks=tasks)
    n     = len(ds)

    labels_t1   = ds.get_labels_t1()
    train_idx, temp_idx = train_test_split(
        list(range(n)), test_size=val_split + test_split,
        random_state=seed, stratify=labels_t1
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_split / (val_split + test_split),
        random_state=seed,
        stratify=[labels_t1[i] for i in temp_idx]
    )
    print(f"\n[Split] Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

    # WeightedRandomSampler pour équilibrer T1 pendant le train
    train_labels  = [labels_t1[i] for i in train_idx]
    class_count   = np.bincount(train_labels)
    sample_weight = (1.0 / class_count)[np.array(train_labels)]
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weight).float(),
        num_samples=len(train_idx), replacement=True
    )

    # Poids de classe pour les losses
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    class_weight_t1 = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    df = ds.df.iloc[train_idx]
    n_par = (df['task4_label'] == 1).sum()
    n_sin = (df['task4_label'] == 0).sum()
    class_weight_t4 = torch.tensor([n_sin / n_par], dtype=torch.float32)

    print(f"  class_weight T1 = {class_weight_t1.item():.3f}")
    print(f"  class_weight T4 = {class_weight_t4.item():.3f}")

    collate = partial(collate_fn, max_size=max_size)
    kw = dict(collate_fn=collate, num_workers=num_workers,
              pin_memory=torch.cuda.is_available())

    return {
        'train'          : DataLoader(Subset(ds, train_idx), batch_size=batch_size,
                                      sampler=sampler, **kw),
        'val'            : DataLoader(Subset(ds, val_idx),   batch_size=batch_size,
                                      shuffle=False, **kw),
        'test'           : DataLoader(Subset(ds, test_idx),  batch_size=batch_size,
                                      shuffle=False, **kw),
        'class_weight_t1': class_weight_t1,
        'class_weight_t4': class_weight_t4,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ══════════════════════════════════════════════════
    # CHEMINS
    # ══════════════════════════════════════════════════
    MANIFEST  = r"D:/Projet_skipper_RNDT/Projet_Skipper/cache_pt/manifest.csv"
    CACHE_DIR = r"D:/Projet_skipper_RNDT/Projet_Skipper/cache_pt/"
    # ══════════════════════════════════════════════════

    print("=== Test SkipperDataset ===")
    loaders = get_dataloaders(MANIFEST, CACHE_DIR, batch_size=4, max_size=256)

    batch = next(iter(loaders['train']))
    print(f"\nBatch test :")
    print(f"  image  : {tuple(batch['image'].shape)}  dtype={batch['image'].dtype}")
    print(f"  task1  : {batch['task1']}")
    print(f"  task2  : {batch['task2']}")
    print(f"  task4  : {batch['task4']}")
    print(f"\n✓ Dataset prêt pour model.py")