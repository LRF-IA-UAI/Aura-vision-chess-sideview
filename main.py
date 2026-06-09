"""
Chess Vision Eval
=================
Live camera feed that detects chess pieces from an oblique angle (~60 deg)
and draws a top-down board diagram with their positions.

Controls
--------
  R     = Calibrar  (detectar borde rojo, guardar homografia y tablero vacio)
  I     = Analizar  (warpear frame, detectar piezas, mostrar diagrama)
  S     = Guardar   (guardar ultimo resultado a disco)
  +/-   = Ajustar margen del borde (si la grilla no alinea con las casillas)
  ESC   = Salir

Requires a chessboard with red border.
"""
import os
import sys
import time
import cv2
import numpy as np

from config import (BOARD_SIZE, CAMERA_INDEX, IMAGES_DIR, OUTPUT_DIR, MODELS_DIR)
from perspective import PerspectiveCorrector
from detector import PieceDetector
from board_display import draw_board


# ── Find YOLO model ──────────────────────────────────────────────────

def _find_model():
    candidates = [
        os.path.join(MODELS_DIR, "chess_model.pt"),
        os.path.join(MODELS_DIR, "best.pt"),
        os.path.join(os.path.dirname(__file__), "..", "DETECCION TABLERO", "chess_model.pt"),
        os.path.join(os.path.dirname(__file__), "..", "aura-chess-tracker", "chess_model.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


# ── HUD ──────────────────────────────────────────────────────────────

def _draw_hud(frame, status, calibrated, margin_pct):
    overlay = frame.copy()
    h = frame.shape[0]

    lines = [
        "R:Calibrar  I:Analizar  S:Guardar  +/-:Margen  ESC:Salir",
        f"Estado: {status}",
    ]
    if calibrated:
        if margin_pct > 0.0:
            lines[1] += f"  [CALIBRADO BORDE ROJO margen={margin_pct:.0%}]"
        else:
            lines[1] += "  [CALIBRADO 64 CASILLAS]"

    for i, txt in enumerate(lines):
        y = h - 15 - (len(lines) - 1 - i) * 26
        cv2.putText(overlay, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
    return overlay


# ── Main ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    corrector = PerspectiveCorrector()
    detector = PieceDetector(_find_model())

    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else CAMERA_INDEX
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la camara (indice {cam_index})")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("=" * 55)
    print("  CHESS VISION EVAL")
    print("=" * 55)
    print(f"YOLO: {'SI' if detector.has_yolo else 'NO (usando fallback ocupancia)'}")
    print("Controles:")
    print("  R = Calibrar (Intenta detectar las 64 casillas, o el borde rojo si falla)")
    print("  I = Analizar (detectar piezas y mostrar diagrama)")
    print("  S = Guardar resultado a disco")
    print("  +/- = Ajustar margen (solo aplica si usa borde rojo)")
    print("  ESC = Salir")
    print()

    # State
    cal_matrix = None
    cal_corners = None
    empty_warped = None
    last_diagram = None
    last_warped = None
    last_debug = None
    border_margin = corrector.border_margin
    status = "Sin calibrar (pulsa R con tablero vacio visible)"

    WIN_CAM = "Chess Vision Eval - Camara"
    WIN_DIAGRAM = "Tablero Detectado"
    WIN_DEBUG = "Debug - YOLO + Grilla"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame")
            break

        hud = _draw_hud(frame, status, cal_matrix is not None, border_margin)
        cv2.imshow(WIN_CAM, hud)

        key = cv2.waitKey(1) & 0xFF

        # ── ESC / Q ──────────────────────────────────────────────
        if key in (27, ord('q')):
            break

        # ── R : calibrate ────────────────────────────────────────
        elif key == ord('r'):
            try:
                corners, matrix, method_name = corrector.calibrate(frame)
                cal_matrix = matrix
                cal_corners = corners
                border_margin = corrector.border_margin

                empty_warped = corrector.warp(frame, matrix)

                status = f"Calibrado OK ({method_name}) - coloca piezas y pulsa I"
                print(f"[R] Calibracion exitosa ({method_name})")
                print(f"    Esquinas: {corners.astype(int).tolist()}")
                print("    Coloca las piezas y pulsa I para analizar")

                # Show reference with grid
                ref_show = empty_warped.copy()
                cs = BOARD_SIZE // 8
                for i in range(9):
                    cv2.line(ref_show, (i*cs, 0), (i*cs, BOARD_SIZE), (0, 255, 0), 1)
                    cv2.line(ref_show, (0, i*cs), (BOARD_SIZE, i*cs), (0, 255, 0), 1)
                cv2.putText(ref_show, f"Referencia ({method_name})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Referencia", ref_show)

            except ValueError as e:
                status = f"Error: {e}"
                print(f"[R] ERROR: {e}")

        # ── I : analyze ──────────────────────────────────────────
        elif key == ord('i'):
            if cal_matrix is None:
                status = "Primero calibrar con R"
                print("[I] Primero calibrar con R")
                continue

            # Warp current frame
            corrector.border_margin = border_margin
            warped = corrector.warp(frame, cal_matrix)
            last_warped = warped.copy()

            # Detect
            pieces, method = detector.detect(frame, warped, cal_matrix, empty_ref=empty_warped)

            white = sum(1 for c in pieces.values() if c.isupper())
            black = sum(1 for c in pieces.values() if c.islower())
            total = white + black
            status = f"{total} piezas ({white}B + {black}N) [{method}]"

            # Console output
            print(f"\n[I] === Analisis ({method}) ===")
            print(f"    Total: {total}  |  Blancas: {white}  |  Negras: {black}")

            if detector.last_raw:
                print(f"    Detecciones YOLO crudas: {len(detector.last_raw)}")
                for d in sorted(detector.last_raw, key=lambda x: (x.row, x.col)):
                    print(f"      {d.square}: {d.cls_name:15s}  conf={d.conf:.0%}  "
                          f"bbox=[{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}]")

            _print_board(pieces)

            # Draw diagram
            last_diagram = draw_board(pieces)
            cv2.imshow(WIN_DIAGRAM, last_diagram)

            # Debug: original image + YOLO boxes OR warped + occupancy diff
            last_debug = detector.draw_debug(frame, warped, cal_matrix, empty_ref=empty_warped)
            cv2.putText(last_debug, f"{method} | {total} piezas", (10, BOARD_SIZE - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(WIN_DEBUG, last_debug)

        # ── S : save ─────────────────────────────────────────────
        elif key == ord('s'):
            if last_diagram is None:
                print("[S] Nada que guardar")
                continue

            ts = time.strftime("%Y%m%d_%H%M%S")
            prefix = os.path.join(OUTPUT_DIR, f"eval_{ts}")

            cv2.imwrite(f"{prefix}_diagrama.jpg", last_diagram)
            cv2.imwrite(f"{prefix}_original.jpg", frame)
            if last_warped is not None:
                cv2.imwrite(f"{prefix}_warped.jpg", last_warped)
            if last_debug is not None:
                cv2.imwrite(f"{prefix}_debug.jpg", last_debug)

            print(f"[S] Guardado: {prefix}_*")
            status = f"Guardado: eval_{ts}"

        # ── +/- : adjust border margin ───────────────────────────
        elif key in (ord('+'), ord('='), 43):     # + key
            if border_margin > 0.0 or cal_matrix is None: # Only if using red border
                border_margin = min(0.20, border_margin + 0.01)
                status = f"Margen: {border_margin:.0%}  (pulsa R para recalibrar)"
                print(f"[+] Margen borde: {border_margin:.0%}")
            else:
                print("[+] El margen no aplica cuando se detectan las 64 casillas")

        elif key in (ord('-'), 45):                # - key
            if border_margin > 0.0 or cal_matrix is None:
                border_margin = max(0.00, border_margin - 0.01)
                status = f"Margen: {border_margin:.0%}  (pulsa R para recalibrar)"
                print(f"[-] Margen borde: {border_margin:.0%}")
            else:
                print("[-] El margen no aplica cuando se detectan las 64 casillas")

    cap.release()
    cv2.destroyAllWindows()


def _print_board(pieces):
    """Print ASCII board to console."""
    print("      a   b   c   d   e   f   g   h")
    print("    +---+---+---+---+---+---+---+---+")
    for r in range(8):
        rank = str(8 - r)
        cells = []
        for c in range(8):
            p = pieces.get((r, c))
            cells.append(f" {p} " if p else " . ")
        print(f"  {rank} |{'|'.join(cells)}|")
        print("    +---+---+---+---+---+---+---+---+")


if __name__ == "__main__":
    main()
