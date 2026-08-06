# Insta_Proxy

Bot d'automatisation pour un mode de jeu mobile type "auto-battler" (board 20
cases, banc, fusions/étoiles, achat de cartes à l'élixir). Le bot pilote le
téléphone via ADB (tap/drag), lit l'écran par vision (classifieurs
Keras/TensorFlow pour les unités, OCR pour l'élixir/l'état de la partie), et
décide des achats/ventes/placements via un petit solveur (`merge_solveur.py`).

> **Statut : première version, en cours de rodage.**
> Le bot tourne mais certains comportements sont encore approximatifs :
> désynchronisation possible entre l'état interne et le plateau réel dans
> certains cas rares, dépendance forte à la qualité de la classification
> vision (modèles entraînés sur un jeu de données limité), et aux temps de
> réponse ADB qui peuvent faire rater une fenêtre de déploiement sur un
> appareil lent. À utiliser en observant les logs, pas en mode "et j'oublie".

## Architecture

- `Adb.py` — boucle principale, capture d'écran, orchestration des actions.
- `adb_utils.py` — primitives ADB (tap, drag, capture, connexion device).
- `zones_config.py` — source unique des coordonnées écran (cases, boutons).
- `card_classifier.py` — chargement/inférence des modèles Keras.
- `elexir.py` — reconnaissance du nombre d'élixir (template matching).
- `merge_solveur.py` — logique de décision (ventes/achats/fusions/placement).
- `grid_calibrator.py` — outil interactif pour calibrer les zones écran.
- `modelcreate.py` — entraînement des classifieurs.
- `action.py`, `jsonverif.py` — scripts de test/vérification annexes.

## Installation

```bash
python -m venv venv_bot
venv_bot\Scripts\activate      # Windows
pip install -r requirements.txt
```

Dépendances système à installer séparément (voir `requirements.txt`) :
`adb` (Android Platform Tools), `tesseract-ocr`, `scrcpy`.

## Usage

```bash
python Adb.py
```

Lance un menu interactif : jeu automatique en continu, mode pas-à-pas
("debug", avec aperçu live des prédictions), ou enregistrement vidéo scrcpy
de la session.

## Limitations connues

- Les modèles de classification (`model/*.keras`) sont spécifiques à une
  résolution/calibration d'écran donnée — `grid_calibrator.py` est à relancer
  si tu changes d'appareil.
- Le classifieur board ne renvoie qu'un nom d'unité par case, pas son étoile
  (fusion) — l'étoile réelle n'est pas toujours récupérable après une fusion
  faite par le jeu pendant un combat non observé par le bot.
