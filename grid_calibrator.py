"""
grid_calibrator.py
-------------------
Outil de calibration des zones écran (tap, éditeur interactif drag/resize/zoom).

Changement vs version d'origine : les primitives ADB (run_adb_command,
capture_phone_screen, test_adb_connection) et les ZONES viennent maintenant
de adb_utils.py / zones_config.py au lieu d'être redéfinies ici à l'identique.
"""

import cv2
import numpy as np

from adb_utils import capture_phone_screen, run_adb_command, test_adb_connection
from zones_config import ZONES

# ──────────────────────────────────────────────
# ÉDITEUR INTERACTIF
# ──────────────────────────────────────────────

HANDLE_SIZE = 7
WIN = "Editeur de zones"


class ZoneEditor:
    def __init__(self, frame, zones_raw):
        self.orig = frame
        h_orig, w_orig = frame.shape[:2]

        # Échelle de base pour tenir dans 1200x900
        self.base_scale = min(1.0, 1200 / w_orig, 900 / h_orig)
        self.disp_w = int(w_orig * self.base_scale)
        self.disp_h = int(h_orig * self.base_scale)

        # Zoom / pan
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.panning = False
        self.pan_start = None

        # Zones en coords IMAGE ORIGINALE
        self.zones = []
        for (zy1, zy2, zx1, zx2) in zones_raw:
            self.zones.append({'x1': zx1, 'y1': zy1, 'x2': zx2, 'y2': zy2, 'sel': False})

        # Drag / resize
        self.drag_active = False
        self.drag_start = None
        self.drag_offsets = []
        self.resize_zone = None
        self.resize_handle = None
        self.resize_orig = None

    # ── conversion coords ──────────────────────

    def _total_scale(self):
        return self.base_scale * self.zoom

    def _orig_to_disp(self, x, y):
        s = self._total_scale()
        return (int(x * s) + self.pan_x, int(y * s) + self.pan_y)

    def _disp_to_orig(self, dx, dy):
        s = self._total_scale()
        return ((dx - self.pan_x) / s, (dy - self.pan_y) / s)

    def _to_phone(self, z):
        return (int(z['y1']), int(z['y2']), int(z['x1']), int(z['x2']))

    # ── poignées ──────────────────────────────

    def _handles_disp(self, z):
        x1d, y1d = self._orig_to_disp(z['x1'], z['y1'])
        x2d, y2d = self._orig_to_disp(z['x2'], z['y2'])
        mxd, myd = (x1d + x2d) // 2, (y1d + y2d) // 2
        return {
            'tl': (x1d, y1d), 'tc': (mxd, y1d), 'tr': (x2d, y1d),
            'ml': (x1d, myd), 'mr': (x2d, myd),
            'bl': (x1d, y2d), 'bc': (mxd, y2d), 'br': (x2d, y2d),
        }

    def _hit_handle(self, z, mx, my):
        for name, (hx, hy) in self._handles_disp(z).items():
            if abs(mx - hx) <= HANDLE_SIZE and abs(my - hy) <= HANDLE_SIZE:
                return name
        return None

    def _hit_zone(self, mx, my):
        ox, oy = self._disp_to_orig(mx, my)
        for z in reversed(self.zones):
            if z['x1'] <= ox <= z['x2'] and z['y1'] <= oy <= z['y2']:
                return z
        return None

    def _selected(self):
        return [z for z in self.zones if z['sel']]

    # ── dessin ────────────────────────────────

    def _draw(self):
        s = self._total_scale()
        w_z = int(self.orig.shape[1] * s)
        h_z = int(self.orig.shape[0] * s)
        zoomed = cv2.resize(self.orig, (w_z, h_z), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        src_x = max(0, -self.pan_x)
        src_y = max(0, -self.pan_y)
        dst_x = max(0, self.pan_x)
        dst_y = max(0, self.pan_y)
        copy_w = min(w_z - src_x, self.disp_w - dst_x)
        copy_h = min(h_z - src_y, self.disp_h - dst_y)

        if copy_w > 0 and copy_h > 0:
            canvas[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
                zoomed[src_y:src_y + copy_h, src_x:src_x + copy_w]

        for z in self.zones:
            x1d, y1d = self._orig_to_disp(z['x1'], z['y1'])
            x2d, y2d = self._orig_to_disp(z['x2'], z['y2'])
            color = (0, 0, 255) if z['sel'] else (180, 180, 180)
            thick = 2 if z['sel'] else 1
            cv2.rectangle(canvas, (x1d, y1d), (x2d, y2d), color, thick)

            if z['sel']:
                for (hx, hy) in self._handles_disp(z).values():
                    cv2.rectangle(canvas,
                                  (hx - HANDLE_SIZE, hy - HANDLE_SIZE),
                                  (hx + HANDLE_SIZE, hy + HANDLE_SIZE),
                                  (0, 0, 255), -1)

        legend = [
            "Molette: zoom | Clic milieu / Ctrl+glisser: pan",
            "Clic: select | Shift+clic: multi | Glisser: deplacer",
            "Dbl-clic: resize | Clic droit: deselect | D: suppr | A: tout | R: recapturer | S: resume | ESC: quitter",
        ]
        for i, txt in enumerate(legend):
            cv2.putText(canvas, txt, (6, self.disp_h - 10 - (len(legend) - 1 - i) * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)

        cv2.putText(canvas, f"{len(self.zones)} zones | {len(self._selected())} sel | zoom x{self.zoom:.1f}",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1)

        cv2.imshow(WIN, canvas)

    # ── zoom centré sur le curseur ─────────────

    def _apply_zoom(self, mx, my, factor):
        old_zoom = self.zoom
        new_zoom = max(0.5, min(10.0, self.zoom * factor))
        if new_zoom == old_zoom:
            return
        self.pan_x = int(mx - (mx - self.pan_x) * new_zoom / old_zoom)
        self.pan_y = int(my - (my - self.pan_y) * new_zoom / old_zoom)
        self.zoom = new_zoom
        self._draw()

    # ── callback souris ───────────────────────

    def mouse_callback(self, event, mx, my, flags, param):
        ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)

        if event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.15 if flags > 0 else 1 / 1.15
            self._apply_zoom(mx, my, factor)
            return

        # Clic milieu ou Ctrl+clic gauche → pan
        if event == cv2.EVENT_MBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and ctrl):
            self.panning = True
            self.pan_start = (mx - self.pan_x, my - self.pan_y)
            return
        if event == cv2.EVENT_MBUTTONUP or (event == cv2.EVENT_LBUTTONUP and self.panning):
            self.panning = False
            return
        if event == cv2.EVENT_MOUSEMOVE and self.panning:
            self.pan_x = mx - self.pan_start[0]
            self.pan_y = my - self.pan_start[1]
            self._draw()
            return

        # Double-clic → sélectionne la zone pour resize
        if event == cv2.EVENT_LBUTTONDBLCLK:
            hit = self._hit_zone(mx, my)
            if hit:
                for z in self.zones:
                    z['sel'] = False
                hit['sel'] = True
            self._draw()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (mx, my)

            # Poignée ?
            for z in self._selected():
                h = self._hit_handle(z, mx, my)
                if h:
                    self.resize_zone = z
                    self.resize_handle = h
                    self.resize_orig = dict(z)
                    self.drag_active = True
                    return

            hit = self._hit_zone(mx, my)
            if hit:
                if shift:
                    hit['sel'] = not hit['sel']
                else:
                    if not hit['sel']:
                        for z in self.zones:
                            z['sel'] = False
                    hit['sel'] = True
                sel = self._selected()
                self.drag_start = (mx, my)
                self.drag_offsets = [
                    (z, mx - self._orig_to_disp(z['x1'], z['y1'])[0],
                        my - self._orig_to_disp(z['x1'], z['y1'])[1],
                        mx - self._orig_to_disp(z['x2'], z['y2'])[0],
                        my - self._orig_to_disp(z['x2'], z['y2'])[1])
                    for z in sel
                ]
                self.drag_active = True
            else:
                if not shift:
                    for z in self.zones:
                        z['sel'] = False
                self.drag_active = False
            self._draw()

        elif event == cv2.EVENT_MOUSEMOVE:
            if not self.drag_active or self.drag_start is None:
                return

            s = self._total_scale()

            if self.resize_zone is not None:
                z = self.resize_zone
                ro = self.resize_orig
                h = self.resize_handle
                dx = (mx - self.drag_start[0]) / s
                dy = (my - self.drag_start[1]) / s
                if 'l' in h: z['x1'] = min(ro['x1'] + dx, z['x2'] - 2)
                if 'r' in h: z['x2'] = max(ro['x2'] + dx, z['x1'] + 2)
                if 't' in h: z['y1'] = min(ro['y1'] + dy, z['y2'] - 2)
                if 'b' in h: z['y2'] = max(ro['y2'] + dy, z['y1'] + 2)
            else:
                for z, ox1, oy1, ox2, oy2 in self.drag_offsets:
                    nx1d = mx - ox1
                    ny1d = my - oy1
                    nx2d = mx - ox2
                    ny2d = my - oy2
                    ox1r, oy1r = self._disp_to_orig(nx1d, ny1d)
                    ox2r, oy2r = self._disp_to_orig(nx2d, ny2d)
                    z['x1'], z['y1'] = ox1r, oy1r
                    z['x2'], z['y2'] = ox2r, oy2r
            self._draw()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_active = False
            self.resize_zone = None
            self.resize_handle = None
            self.resize_orig = None
            self.drag_offsets = []
            self._draw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            for z in self.zones:
                z['sel'] = False
            self._draw()

    # ── résumé ────────────────────────────────

    def print_summary(self):
        print("\n" + "=" * 60)
        print("📋 ZONES (format y1, y2, x1, x2) :")
        print("=" * 60)
        for z in self.zones:
            py1, py2, px1, px2 = self._to_phone(z)
            print(f"    ({py1}, {py2}, {px1}, {px2}),")
        print("=" * 60)

    # ── boucle principale ─────────────────────

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, self.disp_w, self.disp_h)
        cv2.setMouseCallback(WIN, self.mouse_callback)
        self._draw()
        print("\n🖱️  Éditeur ouvert.")
        print("   Molette = zoom | Ctrl+glisser ou clic milieu = pan")
        print("   S = coordonnées | ESC = quitter\n")

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            elif key in (ord('d'), ord('D')):
                self.zones = [z for z in self.zones if not z['sel']]
                self._draw()
            elif key in (ord('a'), ord('A')):
                for z in self.zones:
                    z['sel'] = True
                self._draw()
            elif key in (ord('s'), ord('S')):
                self.print_summary()
            elif key in (ord('r'), ord('R')):
                print("📸 Recapture...")
                f = capture_phone_screen()
                if f is not None:
                    self.orig = f
                    print("✅ OK")
                else:
                    print("❌ Échec")
                self._draw()
            elif key == ord('+') or key == ord('='):
                self._apply_zoom(self.disp_w // 2, self.disp_h // 2, 1.2)
            elif key == ord('-'):
                self._apply_zoom(self.disp_w // 2, self.disp_h // 2, 1 / 1.2)
            elif key == ord('0'):
                self.zoom = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self._draw()

            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

        cv2.destroyAllWindows()
        self.print_summary()


# ──────────────────────────────────────────────
# AUTRES FONCTIONS
# ──────────────────────────────────────────────

def editer_zones():
    print("\n📸 Capture de l'écran en cours...")
    frame = capture_phone_screen()
    if frame is None:
        print("❌ Impossible de capturer l'écran.")
        return
    ZoneEditor(frame, ZONES).run()


def afficher_zones():
    print("\n📸 Capture de l'écran en cours...")
    frame = capture_phone_screen()
    if frame is None:
        print("❌ Impossible de capturer l'écran.")
        return
    display = frame.copy()
    for (y1, y2, x1, x2) in ZONES:
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 1)
    h, w = display.shape[:2]
    scale = min(1.0, 1200 / w, 900 / h)
    if scale < 1.0:
        display = cv2.resize(display, (int(w * scale), int(h * scale)))
    cv2.imshow("Zones", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class GridCalibrator:
    def __init__(self):
        self.point_cliquer = []
        self.frame = None
        self.display_frame = None

    def tap_phone(self, x, y):
        run_adb_command(["shell", "input", "tap", str(x), str(y)])
        a = input("bonne endroit ? n/y ")
        if a == "y":
            b = input("correspondance bouton : ")
            self.save_coord((x, y), b)
            self.point_cliquer.append((x, y))
        else:
            print(f"  Tap à ({x}, {y}) annulé")

    def start_calibration(self):
        print("\nCliquez sur l'écran pour connaître les coordonnées. ESC pour quitter.")
        input("Appuyez sur ENTRÉE pour commencer...")
        self.frame = capture_phone_screen()
        if self.frame is None:
            print("❌ Impossible de capturer l'écran.")
            return
        cv2.namedWindow("Calibration Grid")
        cv2.setMouseCallback("Calibration Grid", self.mouse_callback)
        self.update_display()
        while True:
            cv2.imshow("Calibration Grid", self.display_frame)
            if cv2.waitKey(50) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h_orig, w_orig = self.frame.shape[:2]
            h_disp, w_disp = self.display_frame.shape[:2]
            self.tap_phone(int(x * w_orig / w_disp), int(y * h_orig / h_disp))
            self.update_display()

    def update_display(self):
        self.display_frame = self.frame.copy()
        h, w = self.display_frame.shape[:2]
        for elem in self.point_cliquer:
            cv2.circle(self.display_frame, (elem[0], elem[1]), 10, (0, 0, 255), -1)
        scale = min(1.0, 1200 / w, 800 / h)
        if scale < 1.0:
            self.display_frame = cv2.resize(self.display_frame, (int(w * scale), int(h * scale)))

    def save_coord(self, coord, corresp):
        with open("file.txt", "a") as f:
            f.write(f"{coord} {corresp}\n")


# ──────────────────────────────────────────────
# MENU
# ──────────────────────────────────────────────

def main():
    if not test_adb_connection():
        return
    print("\n" + "=" * 70)
    print("🎮 CALIBRATEUR DE GRILLE")
    print("=" * 70)
    print("  [1] - Nouvelle calibration (tap téléphone)")
    print("  [2] - Afficher les zones")
    print("  [3] - Éditeur interactif (drag / resize / zoom)")
    print("  [4] - Quitter")
    print("=" * 70)
    choice = input("\nVotre choix: ").strip()
    if choice == "1":
        GridCalibrator().start_calibration()
    elif choice == "2":
        afficher_zones()
    elif choice == "3":
        editer_zones()
    elif choice == "4":
        print("\n👋 Au revoir!")
    else:
        print("\n❌ Choix invalide")


if __name__ == "__main__":
    main()