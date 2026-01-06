"""
Corner Detection and Grid Overlay Test
Detects the 4 corners of the game board, marks them, and overlays a 5x8 grid.
Press 'q' to quit, 's' to save a snapshot.
"""

import cv2
import numpy as np
import os
from datetime import datetime

# Grid configuration
ROWS, COLS = 5, 8
WARP_W, WARP_H = 2000, 1250

def order_points(pts):
    """Sort 4 points into TL, TR, BR, BL order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect

def find_board_corners(frame_bgr):
    """
    Detect outer border (largest 4-corner contour) and return the corner points.
    Returns: (corners, contour) or (None, None) if not found
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    
    for c in contours[:20]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 0.15 * frame_area:
                corners = approx.reshape(4, 2)
                return corners, c
    
    return None, None

def draw_corners_and_grid(frame_bgr, corners):
    """
    Draw corner markers and the 5x8 grid overlay on the frame.
    """
    overlay = frame_bgr.copy()
    
    if corners is None:
        # No corners detected - show message
        cv2.putText(overlay, "No board detected - looking for corners...", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return overlay, None
    
    # Order the corners: TL, TR, BR, BL
    ordered_corners = order_points(corners.astype("float32"))
    
    # Draw corner markers with labels
    corner_labels = ["TL", "TR", "BR", "BL"]
    corner_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Blue, Red, Yellow
    
    for i, (corner, label, color) in enumerate(zip(ordered_corners, corner_labels, corner_colors)):
        x, y = int(corner[0]), int(corner[1])
        # Draw circle at corner
        cv2.circle(overlay, (x, y), 15, color, -1)
        cv2.circle(overlay, (x, y), 20, color, 3)
        # Draw label
        cv2.putText(overlay, label, (x + 25, y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Draw the board outline
    pts = ordered_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], True, (0, 255, 255), 3)
    
    # Now draw the 5x8 grid lines using perspective
    # We need to interpolate points along each edge
    tl, tr, br, bl = ordered_corners
    
    # Draw horizontal lines (ROWS + 1 lines)
    for r in range(ROWS + 1):
        ratio = r / ROWS
        # Interpolate left edge
        left_pt = tl + ratio * (bl - tl)
        # Interpolate right edge
        right_pt = tr + ratio * (br - tr)
        
        pt1 = (int(left_pt[0]), int(left_pt[1]))
        pt2 = (int(right_pt[0]), int(right_pt[1]))
        cv2.line(overlay, pt1, pt2, (255, 255, 255), 2)
    
    # Draw vertical lines (COLS + 1 lines)
    for c in range(COLS + 1):
        ratio = c / COLS
        # Interpolate top edge
        top_pt = tl + ratio * (tr - tl)
        # Interpolate bottom edge
        bottom_pt = bl + ratio * (br - bl)
        
        pt1 = (int(top_pt[0]), int(top_pt[1]))
        pt2 = (int(bottom_pt[0]), int(bottom_pt[1]))
        cv2.line(overlay, pt1, pt2, (255, 255, 255), 2)
    
    # Draw cell numbers in each cell
    for r in range(ROWS):
        for c in range(COLS):
            # Calculate cell center using bilinear interpolation
            r_ratio = (r + 0.5) / ROWS
            c_ratio = (c + 0.5) / COLS
            
            # Top edge point
            top_pt = tl + c_ratio * (tr - tl)
            # Bottom edge point  
            bottom_pt = bl + c_ratio * (br - bl)
            # Center point
            center = top_pt + r_ratio * (bottom_pt - top_pt)
            
            cell_num = r * COLS + c + 1
            cx, cy = int(center[0]), int(center[1])
            
            # Draw cell number
            cv2.putText(overlay, str(cell_num), (cx - 15, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return overlay, ordered_corners

def warp_board(frame_bgr, corners):
    """
    Warp the detected board to a fixed-size rectangle.
    """
    if corners is None:
        return None
    
    ordered_corners = order_points(corners.astype("float32"))
    
    dst = np.array([
        [0, 0],
        [WARP_W - 1, 0],
        [WARP_W - 1, WARP_H - 1],
        [0, WARP_H - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (WARP_W, WARP_H))
    
    return warped

def draw_grid_on_warped(warped_bgr):
    """
    Draw the 5x8 grid on the warped (top-down) image.
    """
    if warped_bgr is None:
        return None
    
    overlay = warped_bgr.copy()
    H, W = overlay.shape[:2]
    cell_w = W / COLS
    cell_h = H / ROWS
    
    # Draw grid lines
    for r in range(ROWS + 1):
        y = int(r * cell_h)
        cv2.line(overlay, (0, y), (W, y), (255, 255, 255), 2)
    
    for c in range(COLS + 1):
        x = int(c * cell_w)
        cv2.line(overlay, (x, 0), (x, H), (255, 255, 255), 2)
    
    # Draw cell numbers
    for r in range(ROWS):
        for c in range(COLS):
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            cell_num = r * COLS + c + 1
            cv2.putText(overlay, str(cell_num), (cx - 20, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    return overlay

def main():
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get camera resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    # Warm up camera
    print("Warming up camera...")
    for _ in range(10):
        cap.read()
    
    # Capture a single frame
    print("Capturing image...")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not capture frame")
        return
    
    # Rotate 180 degrees (matching your camera setup)
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    # Find corners
    corners, contour = find_board_corners(frame)
    
    # Draw overlay on original frame
    overlay_frame, ordered_corners = draw_corners_and_grid(frame, corners)
    
    # Add status text
    if corners is not None:
        status = "Board DETECTED - Corners marked"
        print(status)
    else:
        status = "Board NOT detected"
        print(status)
    
    cv2.putText(overlay_frame, status, (10, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if corners is not None else (0, 0, 255), 2)
    
    # Save the overlay image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overlay_filename = f"corner_overlay_{timestamp}.jpg"
    cv2.imwrite(overlay_filename, overlay_frame)
    print(f"Saved: {overlay_filename}")
    
    # Save warped view if corners detected
    if corners is not None:
        warped = warp_board(frame, corners)
        warped_with_grid = draw_grid_on_warped(warped)
        warped_filename = f"corner_warped_{timestamp}.jpg"
        cv2.imwrite(warped_filename, warped_with_grid)
        print(f"Saved: {warped_filename}")
    
    print("Done!")

if __name__ == "__main__":
    main()
