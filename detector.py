"""
Piece detection module.

Primary : YOLOv8 with chess_model.pt  (full piece identification)
Fallback: frame-diff occupancy vs empty board reference (color only)

Both return  dict[(row, col)] -> fen_char
  row 0 = rank 8 (top),  col 0 = file a (left)
  fen_char:  'K','Q','R','B','N','P'  (white)
             'k','q','r','b','n','p'  (black)
"""
import os
import cv2
import numpy as np
from config import BOARD_SIZE

# Accept several common class-name conventions from YOLO models
_CLASS_TO_FEN = {}
for _color, _prefix in [("white", ""), ("black", "")]:
    _upper = _color == "white"
    for _piece, _fen in [("king", "K"), ("queen", "Q"), ("rook", "R"),
                          ("bishop", "B"), ("knight", "N"), ("pawn", "P")]:
        _char = _fen if _upper else _fen.lower()
        _CLASS_TO_FEN[f"{_color}_{_piece}"] = _char      # white_king
        _CLASS_TO_FEN[f"{_color}-{_piece}"] = _char      # white-king
        _CLASS_TO_FEN[f"{_color} {_piece}"] = _char      # white king
        _CLASS_TO_FEN[f"{_color[0]}{_fen}"] = _char      # wK / bK


class Detection:
    """Single YOLO detection."""
    __slots__ = ('x1', 'y1', 'x2', 'y2', 'cls_name', 'fen', 'conf', 'row', 'col')

    def __init__(self, x1, y1, x2, y2, cls_name, fen, conf, row, col):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.cls_name = cls_name
        self.fen = fen
        self.conf = conf
        self.row, self.col = row, col

    @property
    def square(self):
        return f"{chr(ord('a') + self.col)}{8 - self.row}"


def _enhance_contrast(image):
    """Apply CLAHE to improve dark-piece-on-dark-square visibility."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


class PieceDetector:

    def __init__(self, model_path=None):
        self.model = None
        self.last_raw = []          # raw Detection list from last detect_yolo
        self.use_clahe = True       # contrast enhancement for dark pieces
        self._try_load(model_path)

    def _try_load(self, model_path):
        if not model_path or not os.path.exists(model_path):
            if model_path:
                print(f"[YOLO] Modelo no encontrado: {model_path}")
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"[YOLO] Modelo cargado: {model_path}")
            print(f"[YOLO] Clases: {list(self.model.names.values())}")
        except ImportError:
            print("[YOLO] ultralytics no instalado (pip install ultralytics)")
        except Exception as e:
            print(f"[YOLO] Error: {e}")

    @property
    def has_yolo(self):
        return self.model is not None

    # ------------------------------------------------------------------
    # YOLO detection
    # ------------------------------------------------------------------

    def detect_yolo(self, frame, cal_matrix, confidence=0.15):
        if self.model is None:
            return None

        input_img = _enhance_contrast(frame) if self.use_clahe else frame
        # iou=0.6 allows more overlapping bounding boxes to coexist without being suppressed
        results = self.model(input_img, conf=confidence, iou=0.6, verbose=False)

        raw = []
        best = {}     # (row,col) -> Detection (highest conf per cell)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, y2 - (y2 - y1) * 0.05

                # Map point using homography matrix
                pt_orig = np.array([[[cx, cy]]], dtype=np.float32)
                pt_warped = cv2.perspectiveTransform(pt_orig, cal_matrix)[0][0]
                wx, wy = pt_warped[0], pt_warped[1]

                col_float = wx / BOARD_SIZE * 8
                row_float = wy / BOARD_SIZE * 8

                col = int(col_float)
                row = int(row_float)

                cls_name = self.model.names[int(box.cls[0])]
                fen = _CLASS_TO_FEN.get(cls_name) or _CLASS_TO_FEN.get(cls_name.lower())
                conf = float(box.conf[0])

                if fen is None:
                    continue
                if not (0 <= row < 8 and 0 <= col < 8):
                    continue

                det = Detection(x1, y1, x2, y2, cls_name, fen, conf, row, col)
                raw.append(det)

                prev = best.get((row, col))
                if prev is None or conf > prev.conf:
                    best[(row, col)] = det

        self.last_raw = raw
        return {k: v.fen for k, v in best.items()}

    # ------------------------------------------------------------------
    # Frame-diff occupancy (fallback when no YOLO model)
    # ------------------------------------------------------------------

    def detect_occupancy(self, warped, empty_ref, diff_thresh=12, cell_thresh=0.10):
        gray_cur = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(empty_ref, cv2.COLOR_BGR2GRAY)

        h, w = gray_cur.shape
        cw, ch = w // 8, h // 8
        m = int(cw * 0.18)  # margin to skip grid lines
        pieces = {}

        print("    [Ocupancia] Analisis por celda:")
        for row in range(8):
            for col in range(8):
                y1, y2 = row * ch + m, (row + 1) * ch - m
                x1, x2 = col * cw + m, (col + 1) * cw - m

                cell_cur = gray_cur[y1:y2, x1:x2]
                cell_ref = gray_ref[y1:y2, x1:x2]

                # Method 1: frame-diff vs empty board
                diff = cv2.absdiff(cell_cur, cell_ref)
                change_pct = np.sum(diff > diff_thresh) / diff.size

                # Method 2: texture — pieces have more variance than bare squares
                std_cur = np.std(cell_cur.astype(float))
                std_ref = np.std(cell_ref.astype(float))
                std_increase = std_cur - std_ref

                # Cell is occupied if significant change OR big texture increase
                occupied = change_pct > cell_thresh or std_increase > 15

                if occupied:
                    # Determine piece color
                    # Compare cell brightness vs reference brightness
                    mean_cur = np.mean(cell_cur)
                    mean_ref = np.mean(cell_ref)

                    # White pieces make cells brighter, black pieces make them darker
                    # relative to the empty square
                    brightness_diff = mean_cur - mean_ref

                    # Also check absolute brightness
                    is_light_sq = (row + col) % 2 == 0
                    abs_thresh = 135 if is_light_sq else 95

                    if brightness_diff > 10 or mean_cur > abs_thresh:
                        color = 'P'  # white piece
                    else:
                        color = 'p'  # black piece

                    sq = f"{chr(ord('a') + col)}{8 - row}"
                    print(f"      {sq}: diff={change_pct:.0%} std+={std_increase:+.1f} "
                          f"brillo={mean_cur:.0f}(ref:{mean_ref:.0f}) -> {'Blanca' if color == 'P' else 'Negra'}")
                    pieces[(row, col)] = color

        self.last_raw = []
        return pieces

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def detect(self, frame, warped, cal_matrix, empty_ref=None):
        """Returns (pieces_dict, method_name)."""
        yolo_result = self.detect_yolo(frame, cal_matrix)
        if yolo_result:  # non-None AND non-empty
            return yolo_result, "YOLO"

        if yolo_result is not None and not yolo_result:
            print("    [YOLO] 0 detecciones — modelo no reconoce estas piezas")
            print("    [YOLO] Cayendo a deteccion por ocupancia (frame-diff)")

        if empty_ref is not None:
            return self.detect_occupancy(warped, empty_ref), "Ocupancia"

        return {}, "Ninguno"

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def draw_debug(self, frame, warped, cal_matrix, empty_ref=None):
        """Draw YOLO boxes / occupancy diff + grid overlay."""
        if self.last_raw:
            # YOLO detections on original frame
            vis = frame.copy()
            inv_matrix = np.linalg.inv(cal_matrix)

            # Draw grid points
            pts = []
            for r in range(9):
                for c in range(9):
                    wx = c * (BOARD_SIZE / 8)
                    wy = r * (BOARD_SIZE / 8)
                    pts.append([wx, wy])
            pts = np.array([pts], dtype=np.float32)
            orig_pts = cv2.perspectiveTransform(pts, inv_matrix)[0]

            # Draw vertical lines
            for c in range(9):
                for r in range(8):
                    p1 = tuple(orig_pts[r * 9 + c].astype(int))
                    p2 = tuple(orig_pts[(r + 1) * 9 + c].astype(int))
                    cv2.line(vis, p1, p2, (0, 255, 0), 1)
            # Draw horizontal lines
            for r in range(9):
                for c in range(8):
                    p1 = tuple(orig_pts[r * 9 + c].astype(int))
                    p2 = tuple(orig_pts[r * 9 + c + 1].astype(int))
                    cv2.line(vis, p1, p2, (0, 255, 0), 1)

            # YOLO boxes
            for det in self.last_raw:
                is_white = det.fen.isupper()
                color = (0, 200, 0) if is_white else (0, 140, 255)
                pt1 = (int(det.x1), int(det.y1))
                pt2 = (int(det.x2), int(det.y2))
                cv2.rectangle(vis, pt1, pt2, color, 2)
                label = f"{det.cls_name} {det.conf:.0%}"
                ly = max(int(det.y1) - 6, 20)
                cv2.putText(vis, label, (int(det.x1), ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

            # Square labels mapped to original frame
            for r in range(8):
                for c in range(8):
                    label = f"{chr(ord('a') + c)}{8 - r}"
                    wx = c * (BOARD_SIZE / 8) + 10
                    wy = r * (BOARD_SIZE / 8) + 20
                    pt = np.array([[[wx, wy]]], dtype=np.float32)
                    orig_pt = cv2.perspectiveTransform(pt, inv_matrix)[0][0]
                    cv2.putText(vis, label, (int(orig_pt[0]), int(orig_pt[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 0), 2)

            return vis

        else:
            # Occupancy on warped frame
            vis = warped.copy()
            h, w = vis.shape[:2]
            cw, ch = w // 8, h // 8

            # Grid
            for i in range(9):
                cv2.line(vis, (i * cw, 0), (i * cw, h), (0, 255, 0), 1)
                cv2.line(vis, (0, i * ch), (w, i * ch), (0, 255, 0), 1)

            # Square labels
            for r in range(8):
                for c in range(8):
                    label = f"{chr(ord('a') + c)}{8 - r}"
                    lx = c * cw + 4
                    ly = r * ch + 16
                    cv2.putText(vis, label, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

            if empty_ref is not None:
                # Occupancy: highlight cells that changed vs reference
                gray_c = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(empty_ref, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray_c, gray_r)

                m = int(cw * 0.18)
                for r in range(8):
                    for c in range(8):
                        y1, y2 = r * ch + m, (r + 1) * ch - m
                        x1, x2 = c * cw + m, (c + 1) * cw - m
                        cell_diff = diff[y1:y2, x1:x2]
                        change = np.sum(cell_diff > 12) / cell_diff.size

                        if change > 0.10 or np.std(gray_c[y1:y2, x1:x2].astype(float)) - np.std(gray_r[y1:y2, x1:x2].astype(float)) > 15:
                            # Tint occupied cell
                            overlay = vis.copy()
                            cv2.rectangle(overlay, (c * cw, r * ch),
                                          ((c + 1) * cw, (r + 1) * ch),
                                          (0, 0, 255), -1)
                            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                            # Show change %
                            cv2.putText(vis, f"{change:.0%}",
                                        (c * cw + cw // 4, r * ch + ch // 2 + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            return vis
