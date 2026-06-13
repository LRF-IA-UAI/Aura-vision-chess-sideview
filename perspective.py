"""
Perspective correction via robust chessboard corner detection + homography.
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
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        idx = np.argsort(angles)
        pts_sorted = pts[idx]
        sums = pts_sorted.sum(axis=1)
        tl_idx = np.argmin(sums)
        return np.roll(pts_sorted, -tl_idx, axis=0)

    def _find_chessboard_robust(self, gray):
        """Try to find chessboard even if rotated (diagonal photos) or low contrast."""
        (h, w) = gray.shape[:2]
        cX, cY = (w // 2, h // 2)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # We try different angles because OpenCV's detector fails on diagonal grids
        angles_to_try = [0, 45, -45, 20, -20]
        
        for img in [gray, enhanced]:
            for angle in angles_to_try:
                if angle == 0:
                    rotated = img
                    M_inv = None
                else:
                    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
                    cos = np.abs(M[0, 0])
                    sin = np.abs(M[0, 1])
                    nW = int((h * sin) + (w * cos))
                    nH = int((h * cos) + (w * sin))
                    M[0, 2] += (nW / 2) - cX
                    M[1, 2] += (nH / 2) - cY
                    rotated = cv2.warpAffine(img, M, (nW, nH))
                    
                    M_inv = cv2.getRotationMatrix2D((nW // 2, nH // 2), -angle, 1.0)
                    M_inv[0, 2] += cX - (nW // 2)
                    M_inv[1, 2] += cY - (nH // 2)

                for scale in [1.0, 0.75, 0.5]:
                    if scale != 1.0:
                        scaled = cv2.resize(rotated, (0, 0), fx=scale, fy=scale)
                    else:
                        scaled = rotated
                        
                    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
                    ret, corners = cv2.findChessboardCorners(scaled, (7, 7), flags)
                    if not ret:
                        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                        ret, corners = cv2.findChessboardCorners(scaled, (7, 7), flags)
                        
                    if ret:
                        if scale != 1.0:
                            corners = corners / scale
                            
                        # Refine corners
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners = cv2.cornerSubPix(rotated, corners, (11, 11), (-1, -1), criteria)
                        
                        # Transform back if rotated
                        if M_inv is not None:
                            corners_shape = corners.shape
                            corners = corners.reshape(-1, 2)
                            ones = np.ones((corners.shape[0], 1))
                            points_3d = np.hstack([corners, ones])
                            transformed = M_inv.dot(points_3d.T).T
                            corners = transformed.reshape(corners_shape).astype(np.float32)
                            
                        return True, corners
        return False, None

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
        if cv2.contourArea(largest) < (mask.shape[0] * mask.shape[1] * 0.01):
            return None
        peri = cv2.arcLength(largest, True)
        for eps_mult in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]:
            approx = cv2.approxPolyDP(largest, eps_mult * peri, True)
            if len(approx) == 4:
                return self._order_corners(approx.reshape(4, 2).astype(np.float32))
        pts = largest.reshape(-1, 2)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        corners = np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)
        return self._order_corners(corners)

    def calibrate(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = self._find_chessboard_robust(gray)
        
        if ret:
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
            self.border_margin = 0.0
            return ordered_pts, matrix, "casillas (rotacion inv.)"
            
        print("[R] Casillas no detectadas, intentando calibracion por borde rojo...")
        mask = self.detect_red_mask(image)
        red_corners = self.find_red_corners(mask)
        if red_corners is not None:
            matrix = cv2.getPerspectiveTransform(red_corners, self.dst_points)
            self.border_margin = BORDER_MARGIN
            return red_corners, matrix, "borde rojo"

        raise ValueError("No se pudieron detectar ni las casillas ni el borde rojo. Asegurate de que la iluminacion sea buena.")

    def warp(self, image, matrix):
        warped = cv2.warpPerspective(image, matrix, (self.board_size, self.board_size))
        m = int(self.board_size * self.border_margin)
        if m > 0:
            cropped = warped[m:self.board_size - m, m:self.board_size - m]
            return cv2.resize(cropped, (self.board_size, self.board_size))
        return warped
