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

def find_board_corners(frame_bgr, debug=False):
    """
    Detect outer border using GREEN tape detection (HSV color space).
    Returns: (corners, contour, debug_image) or (None, None, debug_image) if not found
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
    # Green color range in HSV (adjust if needed)
    # Hue: 35-85 covers most greens
    # Saturation: 40-255 (not too pale)
    # Value: 40-255 (not too dark)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    green_mask = cv2.dilate(green_mask, kernel, iterations=2)
    
    # Create debug image showing the green mask
    debug_img = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
    
    # Find contours in the green mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    
    # Draw all contours on debug image
    for i, c in enumerate(contours[:10]):
        color = (0, 255, 0) if i == 0 else (0, 165, 255)
        cv2.drawContours(debug_img, [c], -1, color, 2)
    
    # Try to find a 4-corner contour (the green tape border)
    for c in contours[:20]:
        peri = cv2.arcLength(c, True)
        # Try different approximation tolerances
        for tolerance in [0.02, 0.03, 0.04, 0.05, 0.06]:
            approx = cv2.approxPolyDP(c, tolerance * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                # Check if it's a significant portion of the frame
                if area > 0.08 * frame_area:
                    corners = approx.reshape(4, 2)
                    # Draw found corners on debug
                    for pt in corners:
                        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 10, (255, 0, 255), -1)
                    return corners, c, debug_img
    
    # Fallback: If no 4-corner found, use minimum bounding rectangle of largest contour
    if contours and cv2.contourArea(contours[0]) > 0.08 * frame_area:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        corners = np.int32(box)
        cv2.drawContours(debug_img, [corners], -1, (255, 0, 0), 3)
        return corners, contours[0], debug_img
    
    return None, None, debug_img

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
    print("\nLive corner detection mode")
    print("Controls:")
    print("  q - Quit")
    print("  s - Save snapshot")
    print("  w - Toggle warped view")
    print("  d - Toggle debug view (shows edge detection)")
    
    show_warped = False
    show_debug = True  # Start with debug on to see what's happening
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Rotate 180 degrees (matching your camera setup)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Find corners (with debug image)
        corners, contour, debug_img = find_board_corners(frame, debug=True)
        
        # Draw overlay on original frame
        overlay_frame, ordered_corners = draw_corners_and_grid(frame, corners)
        
        # Add status text
        if corners is not None:
            status = "Board DETECTED - Corners marked"
            color = (0, 255, 0)
        else:
            status = "Searching for board corners..."
            color = (0, 0, 255)
        
        cv2.putText(overlay_frame, status, (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(overlay_frame, "q=quit, s=save, w=warped, d=debug",
                    (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show main view
        cv2.imshow('Corner Detection - Press Q to quit', overlay_frame)
        
        # Show debug view (edge detection) if enabled
        if show_debug:
            debug_resized = cv2.resize(debug_img, (frame_width, frame_height))
            cv2.putText(debug_resized, "DEBUG: Edge Detection + Contours", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(debug_resized, "Green=largest contour, Orange=other contours, Pink=corners", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Debug: Edge Detection', debug_resized)
        
        # Show warped view if enabled and corners detected
        if show_warped and corners is not None:
            warped = warp_board(frame, corners)
            warped_with_grid = draw_grid_on_warped(warped)
            display_warped = cv2.resize(warped_with_grid, (800, 500))
            cv2.imshow('Warped Board View', display_warped)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Closing...")
            break
        elif key == ord('s'):
            # Save snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            overlay_filename = f"corner_overlay_{timestamp}.jpg"
            cv2.imwrite(overlay_filename, overlay_frame)
            print(f"Saved: {overlay_filename}")
            
            # Save debug image
            debug_filename = f"corner_debug_{timestamp}.jpg"
            cv2.imwrite(debug_filename, debug_img)
            print(f"Saved: {debug_filename}")
            
            if corners is not None:
                warped = warp_board(frame, corners)
                warped_with_grid = draw_grid_on_warped(warped)
                warped_filename = f"corner_warped_{timestamp}.jpg"
                cv2.imwrite(warped_filename, warped_with_grid)
                print(f"Saved: {warped_filename}")
        elif key == ord('w'):
            show_warped = not show_warped
            if not show_warped:
                cv2.destroyWindow('Warped Board View')
            print(f"Warped view: {'ON' if show_warped else 'OFF'}")
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow('Debug: Edge Detection')
            print(f"Debug view: {'ON' if show_debug else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
