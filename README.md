# Projet_Skipper
# Skipper NDT × HETIC — Détection de conduites souterraines

## Présentation du projet

Skipper NDT est une entreprise spécialisée dans l'inspection non destructive des réseaux souterrains. Ce projet, réalisé en collaboration avec HETIC, vise à **automatiser l'analyse de données de champ magnétique actif** pour détecter et caractériser des conduites souterraines.

Les données sont des images magnétiques au format `.npz` à **4 canaux** (Bx, By, Bz, Norme) capturées par un magnétomètre. Chaque image représente une zone du sol scannée. À partir de ces images, le pipeline de deep learning doit répondre à 4 questions :

> *Y a-t-il une conduite ? Quelle est la largeur de la carte ? La couverture est-elle suffisante ? Les conduites sont-elles parallèles ?*

---

## Les 4 tâches

### Tâche 1 — Détection de conduite
**Objectif** : Classifier chaque image comme "conduite présente" (1) ou "absente" (0).

**Approche** : Canal Bz uniquement → resize 224px → flatten → PCA (2 composantes) → KNN (K=3)

**Pourquoi ça marche** : Les images avec conduite présentent un signal Bz 300× plus fort (300-400 nT) que les images sans conduite (~1 nT). La PC1 capture directement cette différence d'amplitude — le KNN sépare les deux classes trivialement.

| Métrique | Résultat | Objectif |
|----------|----------|----------|
| Accuracy | 1.000 ✓ | > 0.92 |
| Recall   | 1.000 ✓ | > 0.95 |
| F1-Score | 1.000 ✓ | — |

**Fichiers générés** : `model_t1_knn.pkl` · `pca_t1.pkl` · `threshold_t1.json`

---

### Tâche 2 — Largeur de la carte magnétique
**Objectif** : Estimer la largeur physique de la zone de mesure en mètres (`width_m`).

**Approche** : Extraction de 24 features physiques (amplitude Bz, gradients, profil spatial, dimensions) → StandardScaler → MLP PyTorch (24→128→64→32→1)

**Note** : `width_m` est un paramètre interne du simulateur — MAE=5m est le meilleur atteignable avec les features disponibles.

| Métrique | Résultat |
|----------|----------|
| MAE test | ~ 2.4170m|

**Fichiers générés** : `task2_model.pth` · `task2_scaler.pkl`

---

### Tâche 3 — Couverture suffisante
**Objectif** : Classifier la couverture du signal comme "suffisante" (1) ou "insuffisante" (0).

**Approche** : 4 canaux → nan_to_num → normalisation locale → cv2.resize(64,64) → CNN (3 blocs Conv + GAP + FC)

| Métrique | Résultat | Objectif |
|----------|----------|----------|
| Accuracy | > 1.000 ✓ | > 0.90 |
| Recall   | > 1.000 ✓ | > 0.85 |
| F&-Score  | 1.0000 ✓  |  - |

**Fichiers générés** : `task3_model.pth`

---

### Tâche 4 — Conduites parallèles
**Objectif** : Détecter si la conduite est simple (0) ou parallèle (1).

**Approche** : 4 canaux → normalisation locale → resize 128px → CNN léger (4 blocs Conv + FC)

**Difficulté** : Contrairement à T1, single et parallel ont des amplitudes similaires — la différence est dans la **forme** du profil magnétique (1 pic vs 2 pics). Nécessite un CNN pour capturer cette signature spatiale.

| Métrique | Résultat | Objectif |
|----------|----------|----------|
| F1 (parallel) | 0.8000 X| > 0.80 |
| Accuracy | 0.888 ✓ | — |

**Fichiers générés** : `model_t4.pt` · `threshold_t4.json`

---

## Structure du projet

```
Projet_Skipper/
│
├── Data_NDT/
│   └── Training_database_float16/    ← 2935 fichiers .npz (synthétiques + real_data)
│
├── cache_pt/                         ← cache tenseurs PyTorch
│   ├── manifest.csv
│   └── tensors/
│
├── models/                           ← modèles entraînés
│   ├── model_t1_knn.pkl
│   ├── pca_t1.pkl
│   ├── threshold_t1.json
│   ├── task2_model.pth
│   ├── task2_scaler.pkl
│   ├── task3_model.pth
│   ├── model_t4.pt
│   └── threshold_t4.json
│
├── prepare_dataset.py                ← prétraitement + cache .pt
├── model_t1.py                       ← Tâche 1 : PCA + KNN
├── model_t2.py                       ← Tâche 2 : Features + MLP
├── model_t3.py                       ← Tâche 3 : CNN couverture
├── model_t4.py                       ← Tâche 4 : CNN parallèles
├── inference.py                      ← script de prédiction final
├── visualize.py                      ← visualisation des canaux NPZ
└── README.md
```

---

## Installation

### Prérequis
- Python 3.10 ou 3.11
- GPU optionnel (CUDA) — le code tourne sur CPU

### 1. Créer l'environnement virtuel

```powershell
python -m venv env
.\env\Scripts\activate
```

### 2. Installer les dépendances

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn xgboost joblib matplotlib seaborn
pip install scikit-image opencv-python tqdm scipy
```

> Pour GPU (CUDA 11.8) :
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Lancement des scripts

### Étape 0 — Prétraitement du dataset

Convertit les fichiers `.npz` en tenseurs `.pt` mis en cache pour accélérer l'entraînement.

```powershell
python prepare_dataset.py
```

Génère `cache_pt/manifest.csv` et `cache_pt/tensors/*.pt` (~61 Go).

---

### Étape 1 — Entraîner Tâche 1 (PCA + KNN)

```powershell
python model_t1.py
```

**Durée** : ~10 min (calcul PCA sur 2833 images) puis instantané pour KNN.

**Fichiers générés** :
```
models/model_t1_knn.pkl
models/pca_t1.pkl
models/threshold_t1.json
models/pca_variance_t1.png
models/training_curves_t1.png
```

---

### Étape 2 — Entraîner Tâche 2 (MLP largeur)

```powershell
python model_t2.py
```

**Durée** : ~20 min (extraction features) + ~10 min (300 époques MLP).

**Fichiers générés** :
```
models/task2_model.pth
models/task2_scaler.pkl
models/training_curves_t2.png
models/evaluation_t2.png
```

---

### Étape 3 — Entraîner Tâche 3 (CNN couverture)

> ⚠️ Nécessite le fichier `data_tache_3.csv` fourni par Skipper NDT.

```powershell
python model_t3.py
```

**Durée** : ~30 min (chargement RAM) + ~1h (100 époques CNN).

**Fichiers générés** :
```
models/task3_model.pth
models/training_curves_t3.png
```

---

### Étape 4 — Entraîner Tâche 4 (CNN parallèles)

```powershell
python model_t4.py
```

**Durée** : ~3h (200 époques CNN sur CPU).

> Pour reprendre un entraînement interrompu, relancez simplement la même commande — le checkpoint est automatiquement rechargé.

**Fichiers générés** :
```
models/model_t4.pt
models/threshold_t4.json
models/training_curves_t4.png
```

---

### Visualisation des données

Affiche les 4 canaux d'une image avant et après traitement des NaN.

```powershell
# Sur un fichier spécifique
python visualize.py --npz "Data_NDT/Training_database_float16/real_data_00045.npz"

# Sauvegarder l'image
python visualize.py --npz mon_image.npz --save --output_dir ./figures/
```

---

## Inférence

### Fichier unique

```powershell
python inference.py --npz "Data_NDT/Training_database_float16/real_data_00045.npz"
```

**Sortie :**
```
==================================================
RÉSULTATS
==================================================
  Fichier  : real_data_00045.npz

  T1 — Détection conduite
    ✓ Conduite détectée  (prob=1.0)

  T2 — Largeur carte
    21.48 m

  T3 — Couverture
    Couverture suffisante  (prob=0.92)

  T4 — Type de conduite
    Conduites parallèles  (prob=1.0)
==================================================
```

### Batch sur tous les real_data

```powershell
python inference.py \
  --dossier "Data_NDT/Training_database_float16/" \
  --output_csv "resultats_real_data.csv"
```
ou
```
python inference.py 
    --dossier "D:/Projet_skipper_RNDT/Projet_Skipper/Data_NDT/Training_database_float16/" 
    --output_csv "D:/Projet_skipper_RNDT/Projet_Skipper/resultats_real_data.csv"
```
**Sortie terminal :**
```
[Batch] 102 fichiers real_data trouvés
  Avec conduite (ground truth)  : 51
  Sans conduite (ground truth)  : 51

==================================================
RÉSUMÉ FINAL
==================================================
  Total traité              : 102
  T1 — Conduite détectée    : 51
  T1 — Aucune conduite      : 51
  T3 — Couverture suffisante : 44
  T3 — Insuffisante          : 7
  T4 — Parallèles            : 12
  T4 — Simples               : 39
  T2 — Largeur moy           : 22.4m
==================================================
```

**CSV généré** avec une ligne par fichier :

| fichier | T1_conduite | T1_label | T2_largeur_m | T3_couverture | T3_label | T4_parallel | T4_label |
|---------|-------------|----------|--------------|---------------|----------|-------------|----------|
| real_data_00045.npz | 1 | Conduite détectée | 21.48 | 1 | Couverture suffisante | 1 | Conduites parallèles |
| real_data_no_pipe_00008.npz | 0 | Aucune conduite | 28.02 | null | N/A | null | N/A |

---

## Logique du pipeline d'inférence

```
Pour chaque image NPZ :
  ├── T1 — Toujours exécuté
  ├── T2 — Toujours exécuté
  └── Si T1 = 1 (conduite détectée) :
        ├── T3 — Couverture
        └── T4 — Type de conduite
      Sinon :
        └── T3 et T4 → N/A
```

---

## Modèles et fichiers nécessaires

Tous les modèles doivent être dans le dossier `models/` avant de lancer l'inférence :

| Fichier | Tâche | Script d'entraînement |
|---------|-------|-----------------------|
| `model_t1_knn.pkl` | T1 | `model_t1.py` |
| `pca_t1.pkl` | T1 | `model_t1.py` |
| `threshold_t1.json` | T1 | `model_t1.py` |
| `task2_model.pth` | T2 | `model_t2_prof.py` |
| `task2_scaler.pkl` | T2 | `model_t2_prof.py` |
| `task3_model.pth` | T3 | `model_t3.py` |
| `model_t4.pt` | T4 | `model_t4.py` |
| `threshold_t4.json` | T4 | `model_t4.py` |

---

## Résultats globaux

| Tâche | Modèle | Métrique principale | Résultat | Objectif |
|-------|--------|---------------------|----------|----------|
| T1 — Détection | PCA + KNN | Recall | **1.000** ✓ | > 0.95 |
| T2 — Largeur | Features + MLP | MAE | **~5.0m** ⚠ | < 1.0m |
| T3 — Couverture | CNN 4 canaux | Recall | **> 0.85** ✓ | > 0.85 |
| T4 — Parallèles | CNN 4 canaux | F1 | **0.795** ✓ | > 0.80 |

> **Note T2** : L'objectif MAE < 1m n'est pas atteignable car `width_m` est un paramètre interne du simulateur physique — il n'est pas extractible avec certitude depuis les pixels de l'image seule.
