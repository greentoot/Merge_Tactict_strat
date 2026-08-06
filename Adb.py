import datetime
import os
import re
import subprocess
import time
from collections import deque

import cv2
import numpy as np
import pytesseract

from adb_utils import (
    adb_drag,
    adb_tap,
    capture_phone_screen,
    test_adb_connection,
)
from card_classifier import CardClassifier
from elexir import NumberRecognizer
from merge_solveur import Reflexion
from zones_config import (
    BANC_COORDS,
    BANC_CROP,
    BOARD_COORDS,
    CHOIX_COORDS,
    CHOIX_CROP_H,
    CHOIX_CROP_W,
    CHOIX_CROP_X,
    CHOIX_CROP_Y,
    CHOIX_X_STEP,
    ELEXIR_CROP,
    GAME_STATE_CROP,
    SELL_BUTTON,
    ZONES,
)

# ──────────────────────────────────────────────────────────────────────────────
# Modèles
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH  = "model/clash_classifiersaison9.keras"
MODEL_PATH2 = "model/clash_3D_classifiersaison9.keras"
MODEL_PATH3 = "model/clash_3D_board_classifiersaison9.keras"
LABELS_PATH  = "model/clash_classifiersiason9.json"
LABELS_PATH2 = "model/clash_3D_classifiersiason9.json"
LABELS_PATH3 = "model/clash_3D_board_classifiersiason9.json"

IMG_H, IMG_W   = 64, 52     # cartes "choix"
IMG_H1, IMG_W2 = 117, 138   # board / banc (vue 3D)

DEBUG_SAVE_SCREENSHOT = "assets/debug/screen1.png"

# Pause entre deux captures d'écran en boucle continue (secondes) :
# évite de spammer ADB inutilement quand le jeu n'est pas en phase "déploiement".
LOOP_THROTTLE_S = 0.2

CHOIX_ANIM_DELAY_S = 0.2

# Délai de stabilité (secondes) exigé avant qu'un changement vu par la vision
# (case qui devient vide, ou case occupée par une unité non trackée) soit
# répercuté sur self.last_board. Évite qu'une frame isolée mal classifiée
# (flou d'animation, etc.) fasse disparaître ou halluciner une unité dans
# l'état interne. Voir VisualMaking._reconcile_board_with_vision.
BOARD_VISION_CONFIRM_S = 0.4

# ── Debug visuel temps réel ─────────────────────────────────────────────────
# Le flux vidéo du téléphone est géré par scrcpy (fluide, faible latence).
DEBUG_WINDOW_NAME = "Board - prédictions live"
RECENT_UPDATE_WINDOW = 1.5   # secondes pendant lesquelles une case "flashe" en vert
BOARD_CELL_PX = 90           # taille d'une case dans le panneau matrice
STRIP_CELL_PX = 78           # taille d'une case choix/banc/vente
SCRCPY_CMD = ["scrcpy", "--window-title", "Téléphone (scrcpy)", "--always-on-top"]

PREDICT_INTERVAL_S = 0.5     # fréquence des inférences modèle dans l'aperçu live (coûteux → throttle)
TAP_FLASH_DURATION = 0.7     # durée d'affichage du flash rouge sur une case tapée
TAP_MATCH_RADIUS = 45        # tolérance (px) pour associer un tap à une case

SCRCPY_RECORD_DIR = "assets/debug/records"  # vidéos du mode 'screen record'


# ──────────────────────────────────────────────────────────────────────────────
# VisualMaking
# ──────────────────────────────────────────────────────────────────────────────

class VisualMaking:
    def __init__(self, cerveau):
        self.cerveau = cerveau
        self.frame = None
        self.display_frame = None
        self.round_state: list[str] = []

        self.recognizer = NumberRecognizer("assets/image_elexir")
        self.classifier_card     = CardClassifier(MODEL_PATH,  LABELS_PATH,  IMG_H, IMG_W)
        self.classifier_board    = CardClassifier(MODEL_PATH2, LABELS_PATH2, IMG_H1, IMG_W2)
        self.classifier_board_3d = CardClassifier(MODEL_PATH3, LABELS_PATH3, IMG_H1, IMG_W2)

        # last_board : {position: (nom_unite, etoile)} 
        self.last_board: dict[int, tuple[str, int]] = {}
        # banc_interne : [(nom_unite, etoile), ...]
        self.banc_interne: list[tuple[str, int]] = []
        self.plan = None

        # ── État pour la vue debug temps réel ───────────────────────────────
        self.tap_history: deque[tuple[int, int, float]] = deque(maxlen=30)
        self.last_prediction_board: dict[int, str] = {}   # dernière prédiction vision par case
        self.pos_last_update: dict[int, float] = {}        # timestamp de dernier changement par case
        self._grid_layout = None                            # cache du layout déduit de ZONES
        self.debug_view_active = False                      # active/désactive le rafraîchissement live
        self._scrcpy_proc: subprocess.Popen | None = None   # process scrcpy pour la vue téléphone
        self._last_predict_ts = 0.0                          # throttle des inférences modèle

    # ── Détection visuelle ────────────────────────────────────────────────────

    def get_choix(self, frame) -> list[str]:
        choix = []
        for i in range(3):
            left = CHOIX_CROP_X + i * CHOIX_X_STEP
            crop = frame[CHOIX_CROP_Y:CHOIX_CROP_Y + CHOIX_CROP_H, left:left + CHOIX_CROP_W]
            resize = cv2.resize(crop, (IMG_W, IMG_H))
            choix.append(self.classifier_card.predict(resize))
        return choix

    def get_elexir(self, frame) -> int:
        y1, y2, x1, x2 = ELEXIR_CROP
        crop = frame[y1:y2, x1:x2]
        return self.recognizer.recognize(crop) 

    def get_game_state(self, frame) -> str:
        y1, y2, x1, x2 = GAME_STATE_CROP
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        texte = pytesseract.image_to_string(th, config="--psm 6").strip()
        self.round_state = re.findall(r'\d+', texte)
        return texte

    def get_game_banc(self, frame) -> list[str]:
        y1, y2, x1, cell_width = BANC_CROP
        banc = []
        for i in range(5):
            crop = frame[y1:y2, x1 + i * cell_width:x1 + (i + 1) * cell_width]
            resize = cv2.resize(crop, (IMG_W2, IMG_H1))
            banc.append(self.classifier_board.predict(resize))
        return banc

    def get_board_visuel(self, frame, positions: list[int] | None = None) -> dict[int, str]:
        """
        positions=None → scanne les 20 cases (comportement d'origine, utilisé
        pour la lecture "vérité terrain" à chaque frame).
        positions=[...] → ne fait tourner le classifieur (coûteux, un
        forward CNN par case) que sur ce sous-ensemble. Utilisé par
        _detecter_position_achetee, qui n'a besoin de re-classifier que les
        cases vides avant l'achat — pas les 20 — pour limiter le temps
        passé hors de la fenêtre de déploiement pendant un achat.
        """
        board = {}
        now = time.time()
        indices = positions if positions is not None else range(len(ZONES))
        for pos in indices:
            r1, r2, c1, c2 = ZONES[pos]
            crop = frame[r1:r2, c1:c2]
            resize = cv2.resize(crop, (IMG_W2, IMG_H1))
            nom = self.classifier_board_3d.predict(resize)
            board[pos] = nom

            if self.last_prediction_board.get(pos) != nom:
                self.pos_last_update[pos] = now
            self.last_prediction_board[pos] = nom

        return board

    def _reconcile_board_with_vision(self, board_vision: dict[int, str]) -> None:
        """
        Recale self.last_board (état simulé en interne, celui que voit le
        solveur) sur ce que la vision voit réellement à l'instant présent.

        Avant : self.last_board n'était mis à jour que par les propres
        actions du bot (achat détecté, placement, fusion) et, en fin de
        tour, entièrement remplacé par plan["compo"] — mais UNIQUEMENT si
        aucun combat n'avait démarré pendant le tour. Dès qu'un tour était
        interrompu par le combat (donc la plupart du temps), l'état interne
        restait figé pour toujours : les unités mortes/déplacées côté jeu
        n'étaient jamais retirées, et une unité pourtant bien visible (ex.
        Archers en case 17) pouvait rester absente de last_board si elle
        n'était jamais passée par un achat/placement détecté par le bot
        lui-même. Le solveur raisonnait alors sur un board fantôme.

        Maintenant : à chaque lecture (get_info_app, donc à chaque frame en
        phase déploiement), on compare board_vision à last_board et on
        corrige les écarts.

        Limites assumées :
        - Le classifieur ne renvoie qu'un nom d'unité par case, pas
          l'étoile. Si la case correspond déjà au nom suivi en interne, on
          ne touche à rien (l'étoile trackée est probablement la bonne).
          Si une unité "nouvelle" apparaît, on essaie de la retrouver
          ailleurs (autre case, banc) pour récupérer son étoile réelle ;
          sinon on part sur ★1 par défaut (imprécis en cas de fusion faite
          par le jeu pendant le combat, mais mieux que rien).
        - Pour éviter qu'une frame isolée mal classifiée (flou d'animation)
          ne fasse disparaître/halluciner une unité, on n'agit que si le
          changement est stable depuis BOARD_VISION_CONFIRM_S (déjà tracké
          par get_board_visuel via pos_last_update).
        """
        now = time.time()

        for pos, nom in board_vision.items():
            vide = not nom or nom.lower() in ("vide", "empty", "")
            connu = self.last_board.get(pos)

            if connu is not None and connu[0] == nom:
                continue  # cohérent avec l'état interne, rien à faire

            stable = (now - self.pos_last_update.get(pos, 0)) >= BOARD_VISION_CONFIRM_S
            if not stable:
                continue  # lecture trop récente, on attend confirmation

            if vide:
                if connu is not None:
                    print(f"  ⚠️  Case {pos} vue vide par la vision mais trackée en "
                          f"interne ({connu[0]}★{connu[1]}) → retirée "
                          f"(morte/déplacée/vendue au combat)")
                    del self.last_board[pos]
                continue

            # La case contient une unité que l'état interne ne connaît pas à
            # cette position (soit rien, soit un autre nom qu'avant).
            etoile = 1
            source = None
            for autre_pos, (u, s) in list(self.last_board.items()):
                if autre_pos != pos and u == nom:
                    etoile, source = s, ("board", autre_pos)
                    break
            if source is None:
                for i, (u, s) in enumerate(self.banc_interne):
                    if u == nom:
                        etoile, source = s, ("banc", i)
                        break

            if source is not None:
                kind, ref = source
                if kind == "board":
                    del self.last_board[ref]
                    print(f"  🔄 Case {pos} : {nom} détecté par vision, "
                          f"récupéré depuis case {ref} (★{etoile})")
                else:
                    self.banc_interne.pop(ref)
                    print(f"  🔄 Case {pos} : {nom} détecté par vision, "
                          f"récupéré depuis le banc (★{etoile})")
            else:
                print(f"  ➕ Case {pos} : {nom} détecté par vision, non trackée "
                      f"en interne → ajoutée ★1 (fallback, étoile potentiellement "
                      f"inconnue)")

            self.last_board[pos] = (nom, etoile)

    def get_info_app(self):
        frame = self.display_frame
        choix = self.get_choix(frame)
        elexir = self.get_elexir(frame)
        banc_noms = self.get_game_banc(frame)
        board = self.get_board_visuel(frame)
        self._reconcile_board_with_vision(board)
        return choix, elexir, banc_noms, board

    def update_display(self):
        frame = capture_phone_screen()
        if frame is not None:
            self.display_frame = frame

    # ── Helpers de recherche de position ──────────────────────────────────────

    def _trouver_coord_unite(self, unite: str, etoile: int) -> tuple[int, int] | None:
        for pos, (u, s) in self.last_board.items():
            if u == unite and s == etoile:
                return BOARD_COORDS[pos]
        for i, (u, s) in enumerate(self.banc_interne):
            if u == unite and s == etoile and i < len(BANC_COORDS):
                return BANC_COORDS[i]
        return None

    def _pos_board(self, unite: str, etoile: int) -> int | None:
        for pos, (u, s) in self.last_board.items():
            if u == unite and s == etoile:
                return pos
        return None

    def _retirer_unite(self, unite: str, etoile: int):
        for i, (u, s) in enumerate(self.banc_interne):
            if u == unite and s == etoile:
                self.banc_interne.pop(i)
                return
        for pos, (u, s) in list(self.last_board.items()):
            if u == unite and s == etoile:
                del self.last_board[pos]
                return

    def _index_choix(self, carte: str, choix: list[str]) -> int | None:
        for i, c in enumerate(choix):
            if c == carte:
                return i
        return None

    # ── Debug visuel temps réel ────────────────────────────────────────────────

    def _cluster_1d(self, values: list[float], tol: float = 20.0) -> list[float]:
        vals = sorted(values)
        clusters = [[vals[0]]]
        for v in vals[1:]:
            if v - clusters[-1][-1] <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [sum(c) / len(c) for c in clusters]

    def _bin_index(self, value: float, bin_centers: list[float]) -> int:
        return min(range(len(bin_centers)), key=lambda i: abs(bin_centers[i] - value))

    def _infer_grid_layout(self):

        if self._grid_layout is not None:
            return self._grid_layout

        centers = []
        for pos, (r1, r2, c1, c2) in enumerate(ZONES):
            cy = (r1 + r2) / 2
            cx = (c1 + c2) / 2
            centers.append((pos, cx, cy))

        row_bins = self._cluster_1d([cy for _, _, cy in centers])
        col_bins = self._cluster_1d([cx for _, cx, _ in centers])

        grid = {}
        for pos, cx, cy in centers:
            row = self._bin_index(cy, row_bins)
            col = self._bin_index(cx, col_bins)
            grid[pos] = (row, col)

        self._grid_layout = (grid, len(row_bins), len(col_bins))
        return self._grid_layout

    def _tap(self, x: int, y: int):
        self.tap_history.append((x, y, time.time()))
        adb_tap(x, y)
        self._refresh_debug_window()

    def _drag(self, sx: int, sy: int, dx: int, dy: int):
        now = time.time()
        self.tap_history.append((sx, sy, now))
        self.tap_history.append((dx, dy, now))
        adb_drag(sx, sy, dx, dy)
        self._refresh_debug_window()

    def _ensure_show_taps(self, enable: bool = True):
        val = "1" if enable else "0"
        try:
            subprocess.run(
                ["adb", "shell", "settings", "put", "system", "show_touches", val],
                check=False,
            )
        except FileNotFoundError:
            print("    'adb' introuvable dans le PATH — impossible d'activer show_touches.")

    def _launch_scrcpy(self):
        if self._scrcpy_proc is not None and self._scrcpy_proc.poll() is None:
            return  # déjà en cours
        try:
            self._scrcpy_proc = subprocess.Popen(SCRCPY_CMD)
            print("   scrcpy lancé (vue téléphone en direct).")
        except FileNotFoundError:
            self._scrcpy_proc = None
            print("    scrcpy introuvable — installe-le et vérifie qu'il est dans le PATH.")

    def _stop_scrcpy(self):
        if self._scrcpy_proc is not None and self._scrcpy_proc.poll() is None:
            self._scrcpy_proc.terminate()
        self._scrcpy_proc = None

    def _new_record_path(self) -> str:
        os.makedirs(SCRCPY_RECORD_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(SCRCPY_RECORD_DIR, f"record_{ts}.mp4")

    def _finaliser_enregistrement(self, proc, path):

        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _screen_record(self):

        path = self._new_record_path()
        cmd = SCRCPY_CMD + ["--record", path]

        try:
            proc = subprocess.Popen(cmd)
        except FileNotFoundError:
            print("    scrcpy introuvable — installe-le et vérifie qu'il est dans le PATH.")
            return

        print(f"   Enregistrement en cours → {path}")
        print("  L'IA joue automatiquement. Ctrl+C dans ce terminal pour tout arrêter.")

        try:
            self._boucle_continue()
        except KeyboardInterrupt:
            print("\n    Arrêt demandé.")
        finally:
            self._finaliser_enregistrement(proc, path)
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"   Vidéo enregistrée : {path} ({size_mb:.1f} Mo)")
            else:
                print("    Aucun fichier vidéo trouvé — l'enregistrement a peut-être échoué.")

    def _tap_flash_intensity(self, cx: float, cy: float, radius: float = TAP_MATCH_RADIUS) -> float:
        now = time.time()
        best = 0.0
        for (x, y, t) in self.tap_history:
            age = now - t
            if age > TAP_FLASH_DURATION:
                continue
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist <= radius:
                intensity = max(0.0, 1.0 - age / TAP_FLASH_DURATION)
                best = max(best, intensity)
        return best

    def _draw_cell(self, panel, x0, y0, cell_px, label_top, label_main, label_sub,
                    base_border, flash_intensity):
        if flash_intensity > 0:
            border_color = (0, 0, 255)
            thickness = 3
            fill = np.array([0, 0, 90], dtype=np.uint8)
            region = panel[y0 + 2:y0 + cell_px - 2, x0 + 2:x0 + cell_px - 2]
            region[:] = cv2.addWeighted(region, 1 - 0.5 * flash_intensity,
                                         np.full_like(region, fill), 0.5 * flash_intensity, 0)
        else:
            border_color = base_border
            thickness = 2

        cv2.rectangle(panel, (x0 + 2, y0 + 2), (x0 + cell_px - 2, y0 + cell_px - 2), border_color, thickness)
        if label_top:
            cv2.putText(panel, label_top, (x0 + 4, y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        if label_main:
            cv2.putText(panel, label_main[:10], (x0 + 4, y0 + cell_px // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "-", (x0 + cell_px // 2 - 4, y0 + cell_px // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        if label_sub:
            cv2.putText(panel, label_sub, (x0 + 4, y0 + cell_px // 2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 215, 255), 1, cv2.LINE_AA)

    def _build_board_matrix_panel(self, cell_px: int = BOARD_CELL_PX):
        grid, n_rows, n_cols = self._infer_grid_layout()
        panel = np.full((max(n_rows, 1) * cell_px, max(n_cols, 1) * cell_px, 3), 30, dtype=np.uint8)
        now = time.time()

        for pos, (row, col) in grid.items():
            x0, y0 = col * cell_px, row * cell_px

            nom, etoile = self.last_board.get(pos, (None, None))
            if nom is None:
                pred_nom = self.last_prediction_board.get(pos)
                if pred_nom and pred_nom.lower() not in ("vide", "empty", ""):
                    nom, etoile = pred_nom, 1

            recent = (now - self.pos_last_update.get(pos, 0)) < RECENT_UPDATE_WINDOW
            base_border = (60, 220, 60) if recent else (90, 90, 90)

            cx, cy = BOARD_COORDS[pos] if pos < len(BOARD_COORDS) else (x0 + cell_px / 2, y0 + cell_px / 2)
            flash = self._tap_flash_intensity(cx, cy)

            self._draw_cell(
                panel, x0, y0, cell_px,
                label_top=str(pos),
                label_main=nom,
                label_sub=("*" * (etoile or 1)) if nom else "",
                base_border=base_border,
                flash_intensity=flash,
            )

        return panel

    def _build_side_strip_panel(self, width: int, cell_px: int = STRIP_CELL_PX):

        n_cols = max(len(CHOIX_COORDS), len(BANC_COORDS)) + 1  # +1 pour le bouton vendre
        row_h = cell_px + 26  # place pour le label de ligne au-dessus
        panel = np.full((row_h * 2, max(width, n_cols * cell_px), 3), 20, dtype=np.uint8)

        cv2.putText(panel, "CHOIX (achat)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        for i, (cx, cy) in enumerate(CHOIX_COORDS):
            x0, y0 = i * cell_px, 22
            flash = self._tap_flash_intensity(cx, cy)
            self._draw_cell(panel, x0, y0, cell_px, label_top=f"c{i}", label_main="", label_sub="",
                             base_border=(90, 90, 90), flash_intensity=flash)

        y_offset = row_h
        cv2.putText(panel, "BANC", (6, y_offset + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        for i, (cx, cy) in enumerate(BANC_COORDS):
            x0, y0 = i * cell_px, y_offset + 22
            nom, etoile = self.banc_interne[i] if i < len(self.banc_interne) else (None, None)
            flash = self._tap_flash_intensity(cx, cy)
            self._draw_cell(panel, x0, y0, cell_px, label_top=f"b{i}", label_main=nom,
                             label_sub=("*" * (etoile or 1)) if nom else "",
                             base_border=(90, 90, 90), flash_intensity=flash)

        # Bouton vendre, tout à droite
        sx, sy = SELL_BUTTON
        vx0 = panel.shape[1] - cell_px - 4
        flash = self._tap_flash_intensity(sx, sy)
        self._draw_cell(panel, vx0, y_offset + 22, cell_px, label_top="VENDRE", label_main="", label_sub="",
                         base_border=(120, 60, 60), flash_intensity=flash)

        return panel

    def _refresh_debug_window(self):
        if not self.debug_view_active:
            return

        board_panel = self._build_board_matrix_panel()
        strip_panel = self._build_side_strip_panel(width=board_panel.shape[1])

        header_h = 30
        header = np.full((header_h, board_panel.shape[1], 3), 15, dtype=np.uint8)
        if self.tap_history:
            x, y, t = self.tap_history[-1]
            age = time.time() - t
            color = (0, 0, 255) if age < TAP_FLASH_DURATION else (140, 140, 140)
            cv2.putText(header, f"Dernier tap : ({x},{y})  il y a {age:.1f}s",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(header, "Aucun tap pour l'instant", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

        def _match_width(img, target_w):
            if img.shape[1] == target_w:
                return img
            pad = np.zeros((img.shape[0], target_w - img.shape[1], 3), dtype=np.uint8)
            return np.hstack([img, pad]) if img.shape[1] < target_w else img[:, :target_w]

        w = max(header.shape[1], board_panel.shape[1], strip_panel.shape[1])
        combined = np.vstack([_match_width(header, w), _match_width(board_panel, w), _match_width(strip_panel, w)])

        cv2.imshow(DEBUG_WINDOW_NAME, combined)
        cv2.waitKey(1)

    def _live_preview(self) -> str:
        print("  🎥 Aperçu en direct — 'c' : lancer le tour · 'q' : quitter · 's' : screenshot")
        while True:
            self.update_display()

            now = time.time()
            if self.display_frame is not None and (now - self._last_predict_ts) >= PREDICT_INTERVAL_S:
                self.get_board_visuel(self.display_frame)
                self._last_predict_ts = now

            self._refresh_debug_window()

            key = cv2.waitKey(30) & 0xFF
            if key == ord('c'):
                return "continue"
            if key == ord('q'):
                return "quit"
            if key == ord('s') and self.display_frame is not None:
                cv2.imwrite(DEBUG_SAVE_SCREENSHOT, self.display_frame)
                print(f"  📸 Sauvegardé : {DEBUG_SAVE_SCREENSHOT}")

    # ── Actions ADB haut niveau ───────────────────────────────────────────────

    def _vendre_unite(self, unite: str, etoile: int):
        coord = self._trouver_coord_unite(unite, etoile)
        if coord is None:
            print(f"    Impossible de localiser {unite}★{etoile} pour vendre")
            return
        cx, cy = coord
        print(f"   VENDRE {unite}★{etoile} @ ({cx},{cy})")
        self._tap(cx, cy)
        time.sleep(0.35)
        self._tap(*SELL_BUTTON)
        self._retirer_unite(unite, etoile)

    def _detecter_position_achetee(self, board_avant: dict[int, str]) -> int | None:

        time.sleep(0.5)  # laisser l'animation de placement se terminer
        self.update_display()
        if self.display_frame is None:
            return None

        # Seule une case VIDE avant l'achat peut accueillir la nouvelle
        # unité. Avant : on reclassifiait les 20 cases (20 forwards CNN) à
        # chaque achat pour n'en retenir qu'une — c'est justement ce genre
        # de travail inutile qui allonge le temps passé hors de la fenêtre
        # de déploiement et fait rater des ventes/achats suivants une fois
        # le combat lancé. On ne réévalue maintenant que les candidates.
        candidates = [
            pos for pos in range(len(ZONES))
            if not board_avant.get(pos) or board_avant.get(pos, "").lower() in ("vide", "empty", "")
        ]
        if not candidates:
            return None
        board_apres = self.get_board_visuel(self.display_frame, positions=candidates)

        for pos, nom in board_apres.items():
            if nom and nom.lower() not in ("vide", "empty", ""):
                return pos
        return None

    def _acheter_carte(self, carte: str, choix: list[str]) -> list[str]:
        idx = self._index_choix(carte, choix)
        if idx is None:
            print(f"    {carte} introuvable dans les choix {choix}")
            return choix
        cx, cy = CHOIX_COORDS[idx]
        print(f"   ACHETER {carte} (slot {idx}) @ ({cx},{cy})")

        # Snapshot du board juste avant l'achat, pour détecter ensuite la
        # case qui vient d'apparaître.
        board_avant = (
            self.get_board_visuel(self.display_frame)
            if self.display_frame is not None
            else dict(self.last_prediction_board)
        )

        self._tap(cx, cy)

        pos_achetee = self._detecter_position_achetee(board_avant)
        if pos_achetee is not None:
            self.last_board[pos_achetee] = (carte, 1)
            self.pos_last_update[pos_achetee] = time.time()
            print(f"  📍 {carte} auto-placée en case {pos_achetee} (détecté par vision)")
        else:
            print("    Position d'achat non détectée par la vision, fallback banc")
            self.banc_interne.append((carte, 1))

        # Laisse l'animation de renouvellement des cartes "choix" se
        time.sleep(CHOIX_ANIM_DELAY_S)
        self.update_display()
        nouveaux_choix = self.get_choix(self.display_frame)
        print(f"  ↳ Nouveaux choix : {nouveaux_choix}")
        return nouveaux_choix

    def _placer_unite(self, unite: str, etoile: int, dest_pos: int):
        if self.last_board.get(dest_pos) == (unite, etoile):
            return  # déjà en place

        src = self._trouver_coord_unite(unite, etoile)
        if src is None:
            print(f"    {unite}★{etoile} introuvable pour placement")
            return

        dx, dy = BOARD_COORDS[dest_pos]
        sx, sy = src
        print(f"   PLACER {unite}★{etoile}  ({sx},{sy}) → case {dest_pos} ({dx},{dy})")

        old_pos = self._pos_board(unite, etoile)
        if old_pos is not None:
            del self.last_board[old_pos]
        else:
            for i, (u, s) in enumerate(self.banc_interne):
                if u == unite and s == etoile:
                    self.banc_interne.pop(i)
                    break

        self._drag(sx, sy, dx, dy)
        self.last_board[dest_pos] = (unite, etoile)
        self.pos_last_update[dest_pos] = time.time()

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def _phase_deploiement(self) -> bool:

        self.update_display()
        if self.display_frame is None:
            return False
        return "déploiement" in self.get_game_state(self.display_frame)

    def jouer_solution(self, choix: list[str], elexir: int, banc_noms: list[str]):

        noms_connus = {u: s for u, s in self.banc_interne}
        self.banc_interne = [
            (nom, noms_connus.get(nom, 1))
            for nom in banc_noms
            if nom and nom.lower() not in ("vide", "empty", "")
        ]

        print(self.banc_interne)
        print(self.last_board)

        self.cerveau.set_etat(board=self.last_board, banc=self.banc_interne)

        round_state = 6
        if self.round_state:
            try:
                round_state = max(1, min(6, int(self.round_state[0])))
            except (IndexError, ValueError):
                round_state = 6

        plan = self.cerveau.solution(choix=choix, elexir=int(elexir), round_state=round_state)
        if plan is None:
            print("  Aucun plan trouvé.")
            return
        self.plan = plan

        stars = {1: "★", 2: "★★", 3: "★★★", 4: "★★★★"}

        def _print_plan(p):
            print("══ PLAN ══════════════════════════════════════════════")
            print(f"  Ventes  : {[(u, stars[s]) for u, s in p['ventes']] or 'aucune'}")
            print(f"  Achat   : {p['achat'] or '(rien)'}")
            print(f"  Fusions : {p['fusions'] or 'aucune'}")
            print(f"  Score   : {p['valeur']:.1f}")
            print("══════════════════════════════════════════════════════")

        _print_plan(plan)

        # Avant : le code supposait qu'on pouvait encore vendre/acheter une
        # fois le combat démarré ("on continue à vendre/acheter..."). En
        # pratique (retour terrain) c'est faux : dès que le combat est
        # réellement lancé, vendre échoue (unité déjà engagée, ex. Princess)
        # et acheter échoue aussi (ex. ArcherQueen déjà sur le terrain) — le
        # tap part mais ne fait rien d'utile, et le plan suivant est calculé
        # sur des hypothèses fausses. Maintenant : dès que
        # _phase_deploiement() devient False, on arrête net les ventes et
        # les achats restants (on ne tente plus aucun tap boutique), on ne
        # continue que ce qui était déjà prévu pour la fin de tour
        # (placements, déjà coupés séparément plus bas).
        combat_started = False
        for unite, etoile in plan["ventes"]:
            if not self._phase_deploiement():
                combat_started = True
                print("    Combat démarré : ventes restantes abandonnées "
                      "(vendre ne fonctionne plus une fois le combat "
                      "commencé).")
                break
            self._vendre_unite(unite, etoile)

        achats_faits: list[str] = []
        while not combat_started and plan.get("achat"):
            if not self._phase_deploiement():
                combat_started = True
                print("    Combat démarré : achats restants abandonnés "
                      "(acheter ne fonctionne plus une fois le combat "
                      "commencé).")
                break

            carte = plan["achat"]
            choix = self._acheter_carte(carte, choix)
            achats_faits.append(carte)
            elexir -= self.cerveau.stats[carte]["cout"]

            if plan["fusions"]:
                print(f"   Fusions attendues : {plan['fusions']}")
                time.sleep(0.15)
                for dest_pos, (u, s) in plan["compo"].items():

                    board_pos_trouve = None
                    for old_pos, (bu, bs) in self.last_board.items():
                        if bu == u and bs < s:
                            board_pos_trouve = old_pos
                            break

                    for i, (bu, bs) in enumerate(self.banc_interne):
                        if bu == u and bs < s:
                            print(f"   Fusion banc : {u} ★{bs}→★{s} (absorbée)")
                            self.banc_interne.pop(i)
                            break

                    if board_pos_trouve is not None:
                        print(f"   Fusion : {u} → ★{s} (conserve la case {board_pos_trouve})")
                        self.last_board[board_pos_trouve] = (u, s)
                    elif (u, s) not in self.last_board.values() and (u, s) not in self.banc_interne:
                        self.banc_interne.append((u, s))

            self.cerveau.set_etat(board=self.last_board, banc=self.banc_interne)
            nouveau_plan = self.cerveau.solution(choix=choix, elexir=int(elexir), round_state=round_state)
            if nouveau_plan is None:
                break
            plan = nouveau_plan
            _print_plan(plan)

        print(f"  🛒 Achats du tour : {achats_faits or 'aucun'}")

        if combat_started:
            print("    Placements coupés (combat déjà démarré) — les unités "
                  "restantes du banc seront placées au tour suivant.")
        else:
            compo_cible = plan["compo"]
            du_banc = [
                (pos, u, s) for pos, (u, s) in compo_cible.items()
                if self._pos_board(u, s) is None
            ]

            for dest_pos, unite, etoile in du_banc:
                if not self._phase_deploiement():
                    combat_started = True
                    print(f"    Combat démarré, {unite} reste sur le banc (inactive ce round).")
                    break
                self._placer_unite(unite, etoile, dest_pos)

        if not combat_started:
            self.last_board = dict(plan["compo"])
            now = time.time()
            for pos in self.last_board:
                self.pos_last_update[pos] = now
            print("   Tour terminé. Board synchronisé.")
        else:
            print("    Tour interrompu par le combat — état interne conservé tel quel.")
        self._refresh_debug_window()

    # ── Boucles ──────────────────────────────────────────────────────────────

    def _boucle_continue(self):
        while True:
            self.update_display()
            self._refresh_debug_window()
            if self.display_frame is None:
                time.sleep(0.5)
                continue
            if "déploiement" in self.get_game_state(self.display_frame):
                choix, elexir, banc, board = self.get_info_app()
                print(choix, elexir, banc, board)
                self.jouer_solution(choix, elexir, banc)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(LOOP_THROTTLE_S)  # évite de saturer ADB avec des captures en rafale

    def start_calibration(self):
        print("\nInstructions:")
        print("  1. Cliquez sur l'écran")
        print("=" * 70 + "\n")
        choix_mode = input(
            "ENTRÉE pour démarrer, 'debug' pour le mode pas-à-pas, "
            "'rec' pour jouer + enregistrer une vidéo scrcpy : "
        ).strip()

        if choix_mode == "rec":
            self._screen_record()
            return

        if choix_mode != "debug":
            try:
                self._boucle_continue()
            except KeyboardInterrupt:
                print("\n    Arrêt demandé.")
            return


        self.debug_view_active = True
        self._ensure_show_taps(True)
        self._launch_scrcpy()
        cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            while True:
                action = self._live_preview()
                if action == "quit":
                    return

                self.update_display()
                if self.display_frame is not None and "déploiement" in self.get_game_state(self.display_frame):
                    choix, elexir, banc, board = self.get_info_app()
                    self.jouer_solution(choix, elexir, banc)

                suite = input("r (recommencer), c (continuer en boucle), s (screenshot), q (quitter) : ").strip()
                if suite == "s":
                    frame = capture_phone_screen()
                    if frame is not None:
                        cv2.imwrite(DEBUG_SAVE_SCREENSHOT, frame)
                        print(f"  📸 Sauvegardé : {DEBUG_SAVE_SCREENSHOT}")
                elif suite == "c":
                    self.debug_view_active = False
                    try:
                        self._boucle_continue()
                    except KeyboardInterrupt:
                        print("\n    Arrêt demandé.")
                    return
                elif suite == "q":
                    return
        finally:
            self.debug_view_active = False
            self._ensure_show_taps(False)
            self._stop_scrcpy()
            cv2.destroyWindow(DEBUG_WINDOW_NAME)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    test_adb_connection()
    cerveau = Reflexion()
    make = VisualMaking(cerveau)
    make.start_calibration()


if __name__ == "__main__":
    main()