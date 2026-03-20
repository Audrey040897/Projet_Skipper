import numpy as np
import torch
import torch.nn as nn
import os
import sys
import joblib

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH  = "task2_model.pth"
SCALER_PATH = "task2_scaler.pkl"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# EXTRACTION DE FEATURES PHYSIQUES (identique à l'entraînement)
# ============================================================
def extract_features(file_path):
    data = np.load(file_path)['data'].astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)

    Bx   = data[:, :, 0]
    By   = data[:, :, 1]
    Bz   = data[:, :, 2]
    Norm = data[:, :, 3]
    mask = Norm > 0

    features = {}

    norm_valid = Norm[mask]
    features['norm_max']    = norm_valid.max()
    features['norm_mean']   = norm_valid.mean()
    features['norm_std']    = norm_valid.std()
    features['norm_median'] = np.median(norm_valid)

    bz_valid = Bz[mask]
    features['bz_max']   = bz_valid.max()
    features['bz_mean']  = bz_valid.mean()
    features['bz_std']   = bz_valid.std()
    features['bz_range'] = bz_valid.max() - bz_valid.min()

    bx_valid = Bx[mask]
    features['bx_max']   = bx_valid.max()
    features['bx_std']   = bx_valid.std()
    features['bx_range'] = bx_valid.max() - bx_valid.min()

    by_valid = By[mask]
    features['by_max']   = by_valid.max()
    features['by_std']   = by_valid.std()
    features['by_range'] = by_valid.max() - by_valid.min()

    features['n_valid_pixels'] = mask.sum()
    features['height']         = data.shape[0]
    features['width_pixels']   = data.shape[1]
    features['aspect_ratio']   = data.shape[0] / (data.shape[1] + 1e-6)

    col_means = Norm.mean(axis=0)
    row_means = Norm.mean(axis=1)

    threshold = col_means.max() * 0.5
    features['profile_width_cols'] = (col_means > threshold).sum()

    threshold2 = row_means.max() * 0.5
    features['profile_width_rows'] = (row_means > threshold2).sum()

    grad_x = np.gradient(Norm, axis=1)
    grad_y = np.gradient(Norm, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    features['grad_mean'] = grad_magnitude[mask].mean()
    features['grad_max']  = grad_magnitude[mask].max()

    features['coverage_ratio'] = mask.mean()
    features['height_m']       = data.shape[0] * 0.2
    features['width_pixels_m'] = data.shape[1] * 0.2

    return np.array(list(features.values()), dtype=np.float32)

# ============================================================
# ARCHITECTURE MLP (identique à l'entraînement)
# ============================================================
class WidthMLP(nn.Module):
    def __init__(self, n_features):
        super(WidthMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================
def predict(file_path):
    """
    Prend en entrée un chemin vers un fichier .npz
    Retourne : largeur de carte en mètres (float)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    # Extraction des features
    features = extract_features(file_path)

    # Normalisation avec le scaler sauvegardé
    scaler   = joblib.load(SCALER_PATH)
    features = scaler.transform(features.reshape(1, -1))

    # Chargement du modèle
    n_features = features.shape[1]
    model      = WidthMLP(n_features).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Prédiction
    tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = model(tensor).item()

    # Clamp entre 2 et 155m (plage physique)
    pred = float(np.clip(pred, 2.0, 155.0))

    print(f"Fichier         : {file_path}")
    print(f"Width_m prédit  : {pred:.2f} m")

    return pred

# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage   : python inference_task2.py <chemin_vers_fichier.npz>")
        print("Exemple : python inference_task2.py Data_SNDT/sample_00015_perfect_straight_clean_field.npz")
        sys.exit(1)

    file_path  = sys.argv[1]
    prediction = predict(file_path)
    print(f"\nRésultat final : {prediction:.2f} m")