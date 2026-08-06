"""
recognize_number_cv.py (elexir.py)
------------------------------------
Reconnaissance de nombre dans une image de jeu via template matching OpenCV.

Pipeline :
  1. Segmentation par projection verticale → une région par chiffre
  2. Crop du chiffre (pixels blancs/dorés uniquement) dans chaque région
  3. Matching crop vs crop pour chaque template

Structure attendue :
    image_elexir/
        0/  ← une ou plusieurs images de référence du chiffre 0
        1/
        ...
        9/

Dépendances :
    pip install opencv-python numpy

Changement clé vs version d'origine : recognize_number()/NumberRecognizer.recognize()
acceptent maintenant soit un chemin, soit un array numpy déjà en mémoire (frame
croppé). Avant, l'appelant (Adb.py::get_elexir) devait écrire le crop sur disque
à CHAQUE frame juste pour que cette fonction le relise avec cv2.imread — un
aller-retour disque inutile dans une boucle qui tourne en continu.
"""

import cv2
import numpy as np
from pathlib import Path


# ─── Configuration ────────────────────────────────────────────────────────────

TEMPLATES_DIR  = Path("assets/image_elexir")
UPSCALE        = 3      # agrandissement avant traitement
WHITE_THRESH   = 170    # seuil pour isoler le texte blanc/doré
COL_EMPTY_PCT  = 0.05   # colonne "vide" si < 5% de pixels blancs
BORDER_PCT     = 0.10   # ignorer les gaps dans les 10% de bords
MIN_GAP_WIDTH  = 6      # largeur minimale (px upscalés) pour un vrai gap
CROP_PAD       = 4      # padding autour du chiffre lors du crop


# ─── Chargement des templates ─────────────────────────────────────────────────

def load_templates(templates_dir: Path) -> dict[int, list[np.ndarray]]:
    """
    Charge et pré-croppe les images de référence pour chaque chiffre (0-9).
    Retourne {chiffre: [crop_gris, ...]}
    """
    templates = {}
    for digit in range(10):
        folder = templates_dir / str(digit)
        if not folder.exists():
            continue
        imgs = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            for p in sorted(folder.glob(ext)):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    upscaled = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE,
                                          interpolation=cv2.INTER_CUBIC)
                    cropped = crop_to_digit(upscaled)
                    imgs.append(cropped)
        if imgs:
            templates[digit] = imgs
    if not templates:
        raise FileNotFoundError(
            f"Aucun template trouvé dans '{templates_dir}'. "
            "Vérifiez que le chemin est correct."
        )
    return templates


# ─── Crop sur les pixels du chiffre ───────────────────────────────────────────

def crop_to_digit(gray: np.ndarray) -> np.ndarray:
    """
    Coupe l'image pour ne garder que la bounding-box des pixels clairs
    (le chiffre blanc/doré), avec un petit padding.
    Fallback : retourne l'image entière.
    """
    _, binary = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return gray
    x, y, w, h = cv2.boundingRect(coords)
    x1 = max(0, x - CROP_PAD)
    y1 = max(0, y - CROP_PAD)
    x2 = min(gray.shape[1], x + w + CROP_PAD)
    y2 = min(gray.shape[0], y + h + CROP_PAD)
    return gray[y1:y2, x1:x2]


# ─── Segmentation par projection verticale ────────────────────────────────────

def segment_by_projection(gray: np.ndarray) -> list[np.ndarray]:
    """
    Découpe l'image en régions via projection verticale.
    Chaque région contient exactement un chiffre.
    Fallback : image entière si aucun vrai gap central trouvé.
    """
    h, w = gray.shape
    border = int(w * BORDER_PCT)

    _, binary = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)
    col_sum = binary.sum(axis=0) / 255
    threshold_col = h * COL_EMPTY_PCT

    gaps_mask = col_sum < threshold_col
    in_gap, gap_ranges, start = False, [], 0
    for i, v in enumerate(gaps_mask):
        if v and not in_gap:
            start = i; in_gap = True
        elif not v and in_gap:
            gap_ranges.append((start, i)); in_gap = False
    if in_gap:
        gap_ranges.append((start, w))

    center_gaps = [
        (s, e) for s, e in gap_ranges
        if s > border and e < w - border and (e - s) >= MIN_GAP_WIDTH
    ]

    if not center_gaps:
        return [gray]

    cuts = [0] + [(s + e) // 2 for s, e in center_gaps] + [w]
    return [gray[:, cuts[i]:cuts[i+1]] for i in range(len(cuts) - 1)
            if cuts[i+1] > cuts[i]]


# ─── Matching crop vs crop ────────────────────────────────────────────────────

def match_digit(
    region: np.ndarray,
    templates: dict[int, list[np.ndarray]],
) -> int:
    """
    Croppe la région puis la compare à chaque template (déjà croppé).
    Retourne le chiffre avec le meilleur score TM_CCOEFF_NORMED.
    """
    region_crop = crop_to_digit(region)

    best_digit = -1
    best_score = -1.0

    for digit, crops in templates.items():
        for tmpl_crop in crops:
            # Redimensionner le template croppé à la taille du crop de la région
            resized = cv2.resize(tmpl_crop, (region_crop.shape[1], region_crop.shape[0]))
            score = float(cv2.matchTemplate(region_crop, resized, cv2.TM_CCOEFF_NORMED).max())
            if score > best_score:
                best_score = score
                best_digit = digit

    return best_digit


# ─── Chargement image : chemin OU array déjà en mémoire ───────────────────────

def _load_gray(image: "str | Path | np.ndarray") -> np.ndarray:
    """Accepte un chemin (str/Path) ou un array BGR déjà en mémoire."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Image introuvable : '{image}'")
    else:
        img = image
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── Fonction principale ───────────────────────────────────────────────────────

def recognize_number(
    image: "str | Path | np.ndarray",
    templates: dict[int, list[np.ndarray]] | None = None,
    templates_dir: Path = TEMPLATES_DIR,
) -> int:
    """
    Reconnaît le nombre entier affiché dans une image de jeu.

    Args:
        image        : chemin vers l'image, OU array BGR déjà en mémoire
                        (évite un aller-retour disque dans une boucle temps réel).
        templates    : Templates pré-chargés (optionnel).
        templates_dir: Chemin vers le dossier des templates si non fournis.

    Returns:
        Le nombre reconnu (int).
    """
    if templates is None:
        templates = load_templates(templates_dir)

    gray = _load_gray(image)
    gray = cv2.resize(gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    regions = segment_by_projection(gray)
    digits = [match_digit(region, templates) for region in regions]

    return int("".join(str(d) for d in digits))


# ─── Wrapper avec cache des templates ─────────────────────────────────────────

class NumberRecognizer:
    """
    Charge les templates une seule fois, reconnaît autant d'images que voulu.

    Exemple :
        recognizer = NumberRecognizer("image_elexir")
        n = recognizer.recognize("screenshot.png")       # depuis un fichier
        n = recognizer.recognize(frame_crop_numpy_array)  # depuis la RAM
    """

    def __init__(self, templates_dir: str | Path = TEMPLATES_DIR):
        self._templates = load_templates(Path(templates_dir))
        print(f"✅ Templates chargés : {sorted(self._templates.keys())}")

    def recognize(self, image: "str | Path | np.ndarray") -> int:
        return recognize_number(image, templates=self._templates)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python recognize_number_cv.py <image_path> [templates_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    tmpl_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else TEMPLATES_DIR

    recognizer = NumberRecognizer(tmpl_dir)
    print(recognizer.recognize(img_path))