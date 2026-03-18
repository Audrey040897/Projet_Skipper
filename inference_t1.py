import numpy as np
import torch
import torch.nn as nn
import cv2
import sys
import os

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE   = (64, 64)
MODEL_PATH = "task1_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ARCHITECTURE CNN (identique à l'entraînement)
# ============================================================
class PipelineCNN(nn.Module):
    def __init__(self):
        super(PipelineCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)

# ============================================================
# PREPROCESSING (identique à l'entraînement)
# ============================================================
def preprocess_npz(file_path, img_size=IMG_SIZE):
    data = np.load(file_path)['data'].astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)
    for c in range(data.shape[2]):
        canal = data[:, :, c]
        c_min, c_max = canal.min(), canal.max()
        if c_max - c_min > 0:
            data[:, :, c] = (canal - c_min) / (c_max - c_min)
        else:
            data[:, :, c] = 0.0
    data_resized = cv2.resize(data, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    tensor = torch.tensor(data_resized).permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)
    return tensor

# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================
def predict(file_path):
    """
    Prend en entrée un chemin vers un fichier .npz
    Retourne : 0 (pas de conduite) ou 1 (conduite présente)
    """
    # Vérification que le fichier existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    # Chargement du modèle
    model = PipelineCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Preprocessing
    tensor = preprocess_npz(file_path).to(DEVICE)

    # Prédiction
    with torch.no_grad():
        output = model(tensor)
        probas = torch.softmax(output, dim=1)
        pred   = output.argmax(dim=1).item()

    prob_0 = probas[0][0].item() * 100
    prob_1 = probas[0][1].item() * 100

    print(f"Fichier    : {file_path}")
    print(f"Prédiction : {pred}  ({'Conduite présente' if pred == 1 else 'Pas de conduite'})")
    print(f"Confiance  : Classe 0 = {prob_0:.1f}%  |  Classe 1 = {prob_1:.1f}%")

    return pred

# ============================================================
# POINT D'ENTRÉE — usage : python inference.py mon_fichier.npz
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python inference.py <chemin_vers_fichier.npz>")
        print("Exemple : python inference.py Data_SNDT/sample_00015_perfect_straight_clean_field.npz")
        sys.exit(1)

    file_path = sys.argv[1]
    prediction = predict(file_path)
    print(f"\nRésultat final : {prediction}")