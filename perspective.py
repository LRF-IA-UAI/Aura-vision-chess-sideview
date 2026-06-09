"""
Perspective correction via red border detection + homography.
Separates calibration (one-time) from warping (per-frame).
"""
import cv2
import numpy as np
from config import BOARD_SIZE, BORDER_MARGIN, RED_LOWER1, RED_UPPER1, RED_LOWER2, RED_UPPER2


class PerspectiveCorrector:

    def __init__(self, board_size=BOARD_SIZE, border_margin=BORDER_MARGIN):
        self.board_size = board_size
        self.border_margin = border_margin
        self.dst_points = np.array([
            [0, 0],
            [board_size, 0],
            [board_size, board_size],
            [0, board_size]
        ], dtype=np.float32)

    def _order_corners(self, pts):
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        # Calculate centroid
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        
        # Calculate angles around centroid
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        
        # Sort by angle (clockwise)
        idx = np.argsort(angles)
        pts_sorted = pts[idx]
        
        # Determine the "top-left" as the one closest to image origin (min x+y)
        sums = pts_sorted.sum(axis=1)
        tl_idx = np.argmin(sums)
        
        # Reorder so top-left is first
        ordered = np.roll(pts_sorted, -tl_idx, axis=0)
        return ordered

    # ------------------------------------------------------------------
    # Red border detection (Fallback)
    # ------------------------------------------------------------------

    def detect_red_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array(RED_LOWER1), np.array(RED_UPPER1))
        m2 = cv2.inRange(hsv, np.array(RED_LOWER2), np.array(RED_UPPER2))
        mask = cv2.bitwise_or(m1, m2)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        return mask

    def find_red_corners(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        img_area = mask.shape[0] * mask.shape[1]
        if cv2.contourArea(largest) < img_area * 0.01:
            return None

        peri = cv2.arcLength(largest, True)
        
        # Try to approximate polygon with increasing tolerance
        for eps_mult in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]:
            approx = cv2.approxPolyDP(largest, eps_mult * peri, True)
            if len(approx) == 4:
                return self._order_corners(approx.reshape(4, 2).astype(np.float32))

        # Fallback: manually find extreme points of the contour
        pts = largest.reshape(-1, 2)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()  # y - x
        
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]  # min(y - x) -> max x, min y
        bl = pts[np.argmax(d)]  # max(y - x) -> min x, max y
        
        corners = np.array([tl, tr, br, bl], dtype=np.float32)
        return self._order_corners(corners)

    # ------------------------------------------------------------------
    # Calibration  (call once with empty board)
    # ------------------------------------------------------------------

    def calibrate(self, image):
        """Detect internal chessboard corners or red border, compute perspective matrix.

        Returns (corners, matrix, method_string) or raises ValueError.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for brown/white boards
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # We will try normal, enhanced, and scaled down versions
        # High resolution images can sometimes confuse the corner detector
        images_to_try = [gray, gray_enhanced]
        scales_to_try = [1.0, 0.5, 0.75]
        
        flags_fast = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        flags_slow = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        
        ret, corners = False, None
        
        for img in images_to_try:
            for scale in scales_to_try:
                if scale != 1.0:
                    scaled = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                else:
                    scaled = img
                    
                ret, corners = cv2.findChessboardCorners(scaled, (7, 7), flags_fast)
                if not ret:
                    ret, corners = cv2.findChessboardCorners(scaled, (7, 7), flags_slow)
                    
                if ret:
                    if scale != 1.0:
                        corners = corners / scale
                    break
            if ret:
                break
        
        if ret:
            # Refine corners for sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # The 4 extreme internal corners (for a 7x7 grid) are at indices 0, 6, 48, 42
            pts = np.array([corners[0][0], corners[6][0], corners[48][0], corners[42][0]], dtype=np.float32)
            ordered_pts = self._order_corners(pts)
            
            sq = self.board_size / 8.0
            ideal_pts = np.array([
                [sq, sq],
                [7 * sq, sq],
                [7 * sq, 7 * sq],
                [sq, 7 * sq]
            ], dtype=np.float32)
            
            matrix = cv2.getPerspectiveTransform(ordered_pts, ideal_pts)
            self.border_margin = 0.0  # Perfect internal mapping needs no margin
            return ordered_pts, matrix, "casillas (64 cuadrados)"
            
        # =================================================================
        # FALLBACK: RED BORDER
        # =================================================================
        print("[R] Casillas no detectadas, intentando calibracion por borde rojo...")
        mask = self.detect_red_mask(image)
        red_corners = self.find_red_corners(mask)
        if red_corners is not None:
            matrix = cv2.getPerspectiveTransform(red_corners, self.dst_points)
            self.border_margin = BORDER_MARGIN  # Reset to default config margin for red border
            return red_corners, matrix, "borde rojo"

        raise ValueError("No se pudieron detectar ni las casillas ni el borde rojo. Asegurate de que haya buena iluminacion y que el tablero sea visible.")

    # ------------------------------------------------------------------
    # Warp  (call per frame with saved matrix)
    # ------------------------------------------------------------------

    def warp(self, image, matrix):
        """Warp image to top-down view using saved matrix. 
        Crops if using red border margin, otherwise direct map."""
        warped = cv2.warpPerspective(image, matrix, (self.board_size, self.board_size))
        m = int(self.board_size * self.border_margin)
        if m > 0:
            cropped = warped[m:self.board_size - m, m:self.board_size - m]
            return cv2.resize(cropped, (self.board_size, self.board_size))
        return warped
