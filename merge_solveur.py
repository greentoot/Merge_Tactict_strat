import json
from collections import Counter
from itertools import combinations

# ─── Constantes ────────────────────────────────────────────────────────────────

BOARD_QT           = 20
STAR_MULT          = {1: 1, 2: 2, 3: 4, 4: 8}
SYNERGY_THRESHOLDS = {2: 2, 3: 5, 5: 9}
MAX_VENTES         = 2

PORTEE_LONGUE = 3        # portee >= 3 → arrière
FRONT_SLOTS   = range(0,  10)   # positions 0-9  : première ligne
BACK_SLOTS    = range(10, 20)   # positions 10-19 : deuxième ligne

MAX_BOARD_PAR_ROUND = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 6}

def board_limit(round_state: int) -> int:
    return MAX_BOARD_PAR_ROUND.get(round_state, 6)

def sell_value(unit: str, star: int, stats: dict) -> int:
    """Élixir récupéré : cout_base × unités investies (★1=1×, ★★=2×, ...)."""
    return (stats[unit]["cout"]-1) * STAR_MULT[star]


# ─── Moteur ────────────────────────────────────────────────────────────────────

class Reflexion:

    def __init__(self, source="statistique.json"):
        if isinstance(source, dict):
            self.data = source
        else:
            with open(source, "r") as f:
                self.data = json.load(f)
        self.stats = self.data["statistique"]
        self.board: dict[int, tuple[str, int]] = {}
        self.banc:  list[tuple[str, int]]      = []
        self.round_state: int = 6  # max par défaut


    # ── État ──────────────────────────────────────────────────────────────────

    def set_etat(self,
                 board: dict[int, tuple[str, int]] | list,
                 banc:  list[tuple[str, int]] = None) -> None:
        """
        Met à jour l'état interne depuis le plan exécuté par le code appelant.

        board : dict {position: (unit, star)}  OU liste [(unit,star)|None] longueur BOARD_QT
        banc  : liste [(unit, star)]  (optionnel, vide par défaut)
        """
        if isinstance(board, list):
            self.board = {i: v for i, v in enumerate(board) if v is not None}
        else:
            self.board = dict(board)
        self.banc = list(banc) if banc else []

    def get_all_units(self) -> list[tuple[str, int]]:
        return list(self.board.values()) + list(self.banc)

    # ── Point d'entrée unique ─────────────────────────────────────────────────
    def solution(self, choix: list[str], elexir: int, round_state: int = 6) -> dict:
            elexir = int(elexir)
            limit  = board_limit(round_state)
            best   = None

            for ventes in self._iter_ventes():
                gain      = sum(sell_value(u, s, self.stats) for u, s in ventes)
                budget    = elexir + gain
                pool_base = self._pool_sans(ventes)

                for achat in [None] + [u for u in choix
                                    if u in self.stats
                                    and self.stats[u]["cout"] <= budget]:
                    pool    = pool_base + ([(achat, 1)] if achat else [])
                    merged  = self._fusions(pool)
                    compo, banc_sim = self._compo_optimale(merged, limit)  # ← limit
                    val     = self._score(compo)
                    fusions = self._diff_fusions(pool, merged)

                    if best is None or val > best["valeur"]:
                        best = {
                            "ventes":       list(ventes),
                            "gain_ventes":  gain,
                            "achat":        achat,
                            "fusions":      fusions,
                            "compo":        compo,
                            "banc_sim":     banc_sim,
                            "valeur":       val,
                        }
            return best
    # ── Helpers privés ────────────────────────────────────────────────────────

    def _iter_ventes(self):
        """
        Énumère toutes les combinaisons de 0 à MAX_VENTES unités à vendre.

        Avant : `uniques = list(dict.fromkeys(self.get_all_units()))` dédupliquait
        les tuples (unite, etoile) identiques AVANT de faire les combinaisons.
        Conséquence : si le board contenait 2x Knight★1, il était impossible de
        considérer "vendre les 2 Knight★1 ensemble" — la combo n'existait tout
        simplement pas dans `uniques`. Le solveur ratait donc de vraies options
        de vente dès qu'on possédait plusieurs copies identiques d'une unité.

        Correction : on combine sur les INDICES de la liste complète (donc les
        doublons sont bien représentés), puis on déduplique seulement les
        RÉSULTATS identiques (même multiset de vente) pour ne pas répéter
        inutilement une combinaison équivalente issue de positions différentes.
        """
        units = self.get_all_units()
        yield []
        seen = set()
        for n in range(1, MAX_VENTES + 1):
            for combo in combinations(range(len(units)), n):
                vente = tuple(sorted(units[i] for i in combo))
                if vente not in seen:
                    seen.add(vente)
                    yield list(vente)

    def _pool_sans(self, ventes: list) -> list:
        pool = self.get_all_units()
        for u in ventes:
            pool.remove(u)
        return pool

    def _fusions(self, units: list) -> list:
        """Applique toutes les fusions récursivement (2×(u,s) → 1×(u,s+1))."""
        pool, changed = list(units), True
        while changed:
            changed = False
            count   = Counter(pool)
            for (u, s), qty in count.items():
                if qty >= 2 and s < 4:
                    pool.remove((u, s)); pool.remove((u, s))
                    pool.append((u, s + 1))
                    changed = True
                    break
        return pool

    def _compo_optimale(self, units: list, limit: int = 6) -> tuple:
            valides = [(u, s) for u, s in units if u in self.stats]

            def score(x): return self._score_unite(*x)

            front = sorted(
                [x for x in valides if self.stats[x[0]]["portee"] < PORTEE_LONGUE],
                key=score, reverse=True
            )
            back = sorted(
                [x for x in valides if self.stats[x[0]]["portee"] >= PORTEE_LONGUE],
                key=score, reverse=True
            )

            front_slots = list(FRONT_SLOTS)
            back_slots  = list(BACK_SLOTS)

            compo:    dict[int, tuple] = {}
            overflow: list[tuple]     = []

            for unit in front:
                if len(compo) >= limit:          # ← vérif limite
                    overflow.append(unit)
                    continue
                if front_slots:
                    compo[front_slots.pop(0)] = unit
                elif back_slots:
                    compo[back_slots.pop(0)] = unit
                else:
                    overflow.append(unit)

            for unit in back:
                if len(compo) >= limit:          # ← vérif limite
                    overflow.append(unit)
                    continue
                if back_slots:
                    compo[back_slots.pop(0)] = unit
                elif front_slots:
                    compo[front_slots.pop(0)] = unit
                else:
                    overflow.append(unit)

            return compo, overflow
    def _score_unite(self, unit: str, star: int) -> float:
        return self.stats[unit]["note"] * star if unit in self.stats else 0.0

    def _synergies(self, units: list) -> float:
        trait_count: Counter = Counter(
            trait
            for u, _ in units if u in self.stats
            for trait in self.stats[u]["traits"]
        )
        bonus = 0.0
        for count in trait_count.values():
            for seuil in sorted(SYNERGY_THRESHOLDS, reverse=True):
                if count >= seuil:
                    bonus += SYNERGY_THRESHOLDS[seuil]
                    break
        return bonus

    def _score(self, board: dict) -> float:
        units = list(board.values())
        return sum(self._score_unite(u, s) for u, s in units) + self._synergies(units)

    def _diff_fusions(self, before: list, after: list) -> list[str]:
        cb, ca = Counter(before), Counter(after)
        return [f"{u} {'★'*s}" for (u, s) in ca
                if s > 1 and ca[(u, s)] > cb.get((u, s), 0)]


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    cerv = Reflexion()

    cerv.set_etat(board={
        0: ("Barbarians", 1),
        1: ("Knight",     1),
        2: ("Wizzard",    2),
    }, banc=[])

    plan = cerv.solution(
        choix  = ["ArcherQueen", "Valkyrie", "Princess", "Musketeer"],
        elexir = 4
    )

    stars = {1:"★",2:"★★",3:"★★★",4:"★★★★"}
    print("══ PLAN DE JEU ═══════════════════════════════════════")
    if plan["ventes"]:
        for u, s in plan["ventes"]:
            print(f"  VENDRE  {u} {stars[s]}  (+{sell_value(u, s, cerv.stats)}é)")
    else:
        print("  Ventes : aucune")
    print(f"  ACHETER : {plan['achat'] or '(rien)'}")
    print(f"  Fusions : {plan['fusions'] or 'aucune'}")
    print(f"  Score   : {plan['valeur']:.1f}")

    print("\n══ COMPO (avant=0-9 / arrière=10-19) ════════════════")
    for pos, (u, s) in sorted(plan["compo"].items()):
        portee = cerv.stats[u]["portee"]
        zone   = "avant  " if portee < PORTEE_LONGUE else "arrière"
        print(f"  [{pos:02d}] {zone}  portée={portee}  {stars[s]:<4} {u}")

    if plan["banc_sim"]:
        print(f"\n  Banc : {', '.join(f'{u}{stars[s]}' for u,s in plan['banc_sim'])}")


if __name__ == "__main__":
    main()