"""
jsonverif.py
------------
Vérifie l'intégrité de statistique.json : nombre d'entrées et liste des
traits (types) uniques utilisés par les unités.

Avant : ce fichier contenait aussi run_adb_command / adb_tap / adb_long_press /
autoclikers — du code ADB sans rapport avec la vérification JSON, déjà
dupliqué ailleurs (voir adb_utils.py). Retiré : ça n'appartenait pas ici et
prêtait à confusion sur le rôle du script.
"""

import json

STATS_FILE = "statistique.json"


def main() -> None:
    with open(STATS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = data["statistique"]
    print(f"{len(stats)} entrées")

    traits = set()
    for key, unit in stats.items():
        # Ignore les entrées séparateurs (ex. "_2_ELIXIRS": "──────")
        if isinstance(unit, dict) and "traits" in unit:
            traits.update(unit["traits"])

    print(sorted(traits))


if __name__ == "__main__":
    main()