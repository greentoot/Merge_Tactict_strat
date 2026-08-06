"""
zones_config.py
----------------
Source unique de vérité pour toutes les zones/coordonnées écran.

Avant : ZONES était copié-collé dans Adb.py ET grid_calibrator.py (20 lignes
dupliquées). Recalibrer une case obligeait à modifier les deux fichiers en
même temps — source d'erreurs. Maintenant : un seul endroit à éditer, généré
directement par grid_calibrator.py (touche S → coller la sortie ici).
"""

# Zones du board : (row_start, row_end, col_start, col_end)
# positions 0-9 = ligne avant, 10-19 = ligne arrière
ZONES = [
    (1402, 1425, 280, 302),
    (1402, 1425, 418, 440),
    (1401, 1423, 557, 579),
    (1401, 1423, 701, 723),
    (1401, 1423, 845, 867),
    (1492, 1515, 210, 232),
    (1492, 1515, 350, 372),
    (1492, 1515, 487, 510),
    (1492, 1515, 631, 654),
    (1495, 1518, 772, 795),
    (1593, 1616, 279, 302),
    (1593, 1616, 422, 444),
    (1596, 1619, 562, 584),
    (1599, 1622, 703, 726),
    (1599, 1622, 844, 867),
    (1692, 1715, 204, 227),
    (1695, 1718, 350, 372),
    (1695, 1718, 490, 512),
    (1695, 1718, 631, 654),
    (1692, 1715, 775, 798),
]

# Centres ADB de chaque case board : (x=col_center, y=row_center)
BOARD_COORDS = [((z[2] + z[3]) // 2, (z[0] + z[1]) // 2) for z in ZONES]

# Centres des 3 cartes "choix" proposées chaque tour
CHOIX_COORDS = [
    (297, 2230),
    (513, 2230),
    (729, 2230),
]

# 5 cases du banc (x=120, cell_width=138, y_center=1927)
BANC_COORDS = [(120 + i * 138 + 69, 1927) for i in range(5)]

# Bouton "vendre" visible après long-press sur une unité
SELL_BUTTON = (777, 960)

# ── Zones de lecture (crop) pour l'OCR / classification ─────────────────────
CHOIX_CROP_X, CHOIX_CROP_Y, CHOIX_CROP_W, CHOIX_CROP_H = 210, 2118, 174, 225
CHOIX_X_STEP = 216

ELEXIR_CROP = (2220, 2322, 888, 984)     # y1, y2, x1, x2
GAME_STATE_CROP = (342, 483, 12, 444)    # y1, y2, x1, x2
BANC_CROP = (1869, 1986, 120, 138)       # y1, y2, x1_start, largeur_cellule

DRAG_DURATION_MS = 300
