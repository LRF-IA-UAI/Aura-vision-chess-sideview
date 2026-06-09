"""
Renders a clean chess diagram from detected piece positions.
"""
import cv2
import numpy as np

# Board colors (BGR)
_LIGHT_SQ = (205, 220, 232)   # cream
_DARK_SQ  = (100, 143, 185)   # brown
_BG       = (40, 40, 40)      # background

_MARGIN_LEFT   = 30   # rank labels
_MARGIN_BOTTOM = 30   # file labels
_INFO_HEIGHT   = 50   # piece count + FEN


def draw_board(pieces, cell_px=80):
    """Draw a chess diagram with detected pieces.

    Args:
        pieces:  dict  {(row, col): fen_char}
                 row 0 = rank 8,  col 0 = file a
        cell_px: size of each square in pixels

    Returns:
        BGR image of the diagram.
    """
    board_px = cell_px * 8
    W = _MARGIN_LEFT + board_px
    H = board_px + _MARGIN_BOTTOM + _INFO_HEIGHT
    img = np.full((H, W, 3), _BG, dtype=np.uint8)

    ox = _MARGIN_LEFT   # board origin x
    oy = 0              # board origin y

    # --- squares ---
    for r in range(8):
        for c in range(8):
            color = _LIGHT_SQ if (r + c) % 2 == 0 else _DARK_SQ
            x1 = ox + c * cell_px
            y1 = oy + r * cell_px
            cv2.rectangle(img, (x1, y1), (x1 + cell_px, y1 + cell_px), color, -1)

    # --- pieces ---
    for (r, c), fen in pieces.items():
        cx = ox + c * cell_px + cell_px // 2
        cy = oy + r * cell_px + cell_px // 2
        _draw_piece(img, cx, cy, fen, cell_px)

    # --- grid lines ---
    for i in range(9):
        x = ox + i * cell_px
        y = oy + i * cell_px
        cv2.line(img, (x, oy), (x, oy + board_px), (50, 50, 50), 1)
        cv2.line(img, (ox, y), (ox + board_px, y), (50, 50, 50), 1)

    # --- rank labels  (8 .. 1) ---
    for r in range(8):
        label = str(8 - r)
        ly = oy + r * cell_px + cell_px // 2 + 5
        cv2.putText(img, label, (6, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # --- file labels  (a .. h) ---
    for c in range(8):
        label = chr(ord('a') + c)
        lx = ox + c * cell_px + cell_px // 2 - 6
        cv2.putText(img, label, (lx, board_px + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # --- info bar ---
    white = sum(1 for ch in pieces.values() if ch.isupper())
    black = sum(1 for ch in pieces.values() if ch.islower())
    fen = _build_fen(pieces)

    y_info = board_px + _MARGIN_BOTTOM + 5
    cv2.putText(img, f"Blancas: {white}   Negras: {black}",
                (ox, y_info + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(img, fen[:60],
                (ox, y_info + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    return img


# ------------------------------------------------------------------

def _draw_piece(img, cx, cy, fen_char, cell_px):
    """Draw a single piece at (cx, cy)."""
    is_white = fen_char.isupper()

    radius = int(cell_px * 0.36)
    bg   = (240, 240, 240) if is_white else (30, 30, 30)
    rim  = (80, 80, 80)    if is_white else (180, 180, 180)
    text_col = (0, 0, 0)   if is_white else (255, 255, 255)

    cv2.circle(img, (cx, cy), radius, bg, -1)
    cv2.circle(img, (cx, cy), radius, rim, 2)

    letter = fen_char.upper()
    scale = cell_px / 65.0
    thick = max(2, int(scale * 1.5))
    sz = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
    tx = cx - sz[0] // 2
    ty = cy + sz[1] // 2
    cv2.putText(img, letter, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_col, thick)


def _build_fen(pieces):
    """Build FEN position string from pieces dict."""
    rows = []
    for r in range(8):
        row_str = ""
        empty = 0
        for c in range(8):
            p = pieces.get((r, c))
            if p is None:
                empty += 1
            else:
                if empty:
                    row_str += str(empty)
                    empty = 0
                row_str += p
        if empty:
            row_str += str(empty)
        rows.append(row_str)
    return "/".join(rows)
