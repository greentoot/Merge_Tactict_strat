"""
test_actions.py
----------------
Teste individuellement chaque action ADB "haut niveau" (vendre / acheter /
placer / tap brut) sans dépendre de la vision (CardClassifier, OCR, elexir)
ni de Reflexion. Utilise exactement les mêmes primitives et les mêmes
coordonnées que Adb.py (adb_utils + zones_config) — donc si un swipe est mal
calibré ici, il l'est aussi en jeu, et inversement.

But : pouvoir dire "vends la case board 3" ou "place ce qui est sur le banc 1
sur la case 7" et voir EXACTEMENT le tap/swipe qui serait exécuté en vrai
par le bot, sans avoir à attendre un vrai tour de jeu détecté par la vision.

Usage :
    python test_actions.py
    (menu interactif, voir ci-dessous)

Ou en import direct :
    from test_actions import test_tap, test_drag, test_vendre, test_placer, test_acheter
    test_placer("banc0", 7)
"""

import time

from adb_utils import adb_drag, adb_long_press, adb_tap, test_adb_connection
from zones_config import (
    BANC_COORDS,
    BOARD_COORDS,
    CHOIX_COORDS,
    SELL_BUTTON,
)

# ─── Registre des zones nommées ────────────────────────────────────────────────
# Mêmes coordonnées que celles utilisées en vrai par VisualMaking dans Adb.py.
# "b0".."b19" = cases board, "banc0".."banc4" = cases banc,
# "c0".."c2" = cartes choix, "vendre" = bouton vendre.

def _build_zones() -> dict[str, tuple[int, int]]:
    zones: dict[str, tuple[int, int]] = {}
    for i, coord in enumerate(BOARD_COORDS):
        zones[f"b{i}"] = coord
    for i, coord in enumerate(BANC_COORDS):
        zones[f"banc{i}"] = coord
    for i, coord in enumerate(CHOIX_COORDS):
        zones[f"c{i}"] = coord
    zones["vendre"] = SELL_BUTTON
    return zones


ZONES = _build_zones()


def list_zones() -> None:
    print("Zones disponibles :")
    print("  board : b0..b19")
    print("  banc  : banc0..banc4")
    print("  choix : c0..c2")
    print("  bouton: vendre")
    print("  (ou tape directement 'x,y' pour des coordonnées libres)")


def zone_coord(name: str) -> tuple[int, int]:
    """Résout un nom de zone ('b3', 'banc1', ...) ou des coords brutes ('444,1422')."""
    name = name.strip()
    if name in ZONES:
        return ZONES[name]
    if "," in name:
        x_str, y_str = name.split(",", 1)
        return int(x_str.strip()), int(y_str.strip())
    raise ValueError(f"Zone inconnue : '{name}'. Tape 'list' pour voir les zones dispo.")


# ─── Actions unitaires ──────────────────────────────────────────────────────────
# Chacune reproduit EXACTEMENT la séquence adb faite par la méthode
# correspondante dans Adb.py (VisualMaking._vendre_unite / _acheter_carte /
# _placer_unite), sans la partie "vision / mise à jour d'état interne".

def test_tap(zone: str, delay: float = 0.3) -> None:
    """Reproduit un simple adb_tap sur une zone nommée ou des coords libres."""
    x, y = zone_coord(zone)
    print(f"  👉 TAP {zone} → ({x},{y})")
    adb_tap(x, y, delay=delay)


def test_drag(src: str, dst: str) -> None:
    """Reproduit un adb_drag entre deux zones (ex: banc0 → b7)."""
    sx, sy = zone_coord(src)
    dx, dy = zone_coord(dst)
    print(f"  👉 DRAG {src}({sx},{sy}) → {dst}({dx},{dy})")
    adb_drag(sx, sy, dx, dy)


def test_vendre(zone: str) -> None:
    """Reproduit _vendre_unite : tap sur l'unité, puis tap sur le bouton vendre.
    `zone` = où se trouve l'unité à vendre (ex: 'b3' ou 'banc1')."""
    x, y = zone_coord(zone)
    print(f"  🔴 VENDRE l'unité en {zone} @ ({x},{y})")
    adb_tap(x, y)
    time.sleep(0.25)
    print(f"  🔴 → tap bouton VENDRE @ {SELL_BUTTON}")
    adb_tap(*SELL_BUTTON)


def test_acheter(choix_idx: int) -> None:
    """Reproduit _acheter_carte : tap sur le slot de carte choisi (0, 1 ou 2)."""
    if not (0 <= choix_idx < len(CHOIX_COORDS)):
        raise ValueError(f"choix_idx doit être entre 0 et {len(CHOIX_COORDS) - 1}")
    x, y = CHOIX_COORDS[choix_idx]
    print(f"  🟢 ACHETER carte slot {choix_idx} @ ({x},{y})")
    adb_tap(x, y)


def test_placer(src_zone: str, dest_board_pos: int) -> None:
    """Reproduit _placer_unite : drag depuis src_zone (case board ou banc)
    vers la case board dest_board_pos."""
    if not (0 <= dest_board_pos < len(BOARD_COORDS)):
        raise ValueError(f"dest_board_pos doit être entre 0 et {len(BOARD_COORDS) - 1}")
    sx, sy = zone_coord(src_zone)
    dx, dy = BOARD_COORDS[dest_board_pos]
    print(f"  📦 PLACER {src_zone}({sx},{sy}) → case b{dest_board_pos}({dx},{dy})")
    adb_drag(sx, sy, dx, dy)


# ─── CLI interactive ────────────────────────────────────────────────────────────

MENU = """
══ TEST ACTIONS ADB ═══════════════════════════════════
  1) tap    <zone>            (ex: b3 / banc1 / c0 / vendre / 444,1422)
  2) drag   <src> <dst>       (ex: banc0 b7)
  3) vendre <zone>            (tap unité + tap bouton vendre)
  3b) vendre_lp <zone>        (long-press unité + tap bouton vendre)
  4) acheter <idx 0-2>
  5) placer <src> <dest_pos>  (ex: banc0 7)
  l) lister les zones
  q) quitter
════════════════════════════════════════════════════════
"""


def _cli() -> None:
    if not test_adb_connection():
        print("⚠️  Continuer quand même ? Les commandes adb échoueront sans device.")

    print(MENU)
    while True:
        raw = input("> ").strip()
        if not raw:
            continue
        if raw in ("q", "quit", "exit"):
            break
        if raw in ("l", "list"):
            list_zones()
            continue
        if raw in ("h", "help", "?"):
            print(MENU)
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        try:
            if cmd == "tap" and len(parts) == 2:
                test_tap(parts[1])
            elif cmd == "drag" and len(parts) == 3:
                test_drag(parts[1], parts[2])
            elif cmd == "vendre" and len(parts) == 2:
                test_vendre(parts[1])
            elif cmd == "acheter" and len(parts) == 2:
                test_acheter(int(parts[1]))
            elif cmd == "placer" and len(parts) == 3:
                test_placer(parts[1], int(parts[2]))
            else:
                print("  ⚠️  Commande invalide. Tape 'h' pour l'aide.")
                continue
        except ValueError as e:
            print(f"  ⚠️  {e}")
            continue

        time.sleep(0.1)


if __name__ == "__main__":
    _cli()
