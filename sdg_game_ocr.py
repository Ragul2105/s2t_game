import tkinter as tk
import random
import time
import threading
import cv2
import numpy as np
import pytesseract
import re
import os
from datetime import datetime

# ----------------------------
# GAME CONFIG
# ----------------------------
ROWS, COLS = 5, 8
ROUND_SECONDS = 60

# Warped board size (higher = better OCR; adjust if your system is slow)
WARP_W, WARP_H = 2000, 1250

# Camera index (0 usually works on Mac)
CAM_INDEX_DEFAULT = 0

# Training data folder
TRAINING_DATA_DIR = "training_data"

# SDG-ish colors (approx). You can edit.
SDG_COLORS = {
    1:  "#E5243B", 2:  "#DDA63A", 3:  "#4C9F38", 4:  "#C5192D",
    5:  "#FF3A21", 6:  "#26BDE2", 7:  "#FCC30B", 8:  "#A21942",
    9:  "#FD6925", 10: "#DD1367", 11: "#FD9D24", 12: "#BF8B2E",
    13: "#3F7E44", 14: "#0A97D9", 15: "#56C02B", 16: "#00689D",
    17: "#19486A"
}

# ----------------------------
# VISION / OCR HELPERS
# ----------------------------
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

def find_board_and_warp(frame_bgr, require_border=False):
    """
    Detect outer border (largest 4-corner contour) and warp to fixed size.
    If no border is detected and require_border=False, uses the full frame resized.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    board_quad = None
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]

    for c in contours[:20]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 0.15 * frame_area:
                board_quad = approx.reshape(4, 2)
                break

    if board_quad is None:
        if require_border:
            return None
        # No border detected - use full frame with slight crop and resize
        h, w = frame_bgr.shape[:2]
        # Crop 5% from edges to remove any camera frame artifacts
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        cropped = frame_bgr[margin_y:h-margin_y, margin_x:w-margin_x]
        # Resize to standard warp dimensions
        warped = cv2.resize(cropped, (WARP_W, WARP_H), interpolation=cv2.INTER_CUBIC)
        return warped

    rect = order_points(board_quad)
    dst = np.array([
        [0, 0],
        [WARP_W - 1, 0],
        [WARP_W - 1, WARP_H - 1],
        [0, WARP_H - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (WARP_W, WARP_H))
    return warped

def preprocess_for_ocr(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    # invert if looks too bright
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)

    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return th

def ocr_number_from_roi(roi_bgr):
    th = preprocess_for_ocr(roi_bgr)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(th, config=config)

    m = re.search(r"\d{1,2}", text)
    if not m:
        return None
    val = int(m.group())
    if 1 <= val <= 17:
        return val
    return None

def recognize_cell(cell_bgr):
    """
    Tiles may be rotated or shifted, so number can appear in different corners.
    Try multiple sub-ROIs and majority vote.
    """
    h, w = cell_bgr.shape[:2]
    rois = [
        (0, 0, int(0.50*w), int(0.50*h)),                    # TL
        (int(0.50*w), 0, w, int(0.50*h)),                    # TR
        (0, int(0.50*h), int(0.50*w), h),                    # BL
        (int(0.50*w), int(0.50*h), w, h),                    # BR
        (int(0.20*w), 0, int(0.80*w), int(0.45*h)),          # top-center
    ]

    candidates = []
    for (x1, y1, x2, y2) in rois:
        roi = cell_bgr[y1:y2, x1:x2]
        val = ocr_number_from_roi(roi)
        if val is not None:
            candidates.append(val)

    if not candidates:
        return None
    return max(set(candidates), key=candidates.count)

def split_and_recognize(warped_bgr):
    H, W = warped_bgr.shape[:2]
    cell_w = W / COLS
    cell_h = H / ROWS
    preds = [[None for _ in range(COLS)] for _ in range(ROWS)]

    for r in range(ROWS):
        for c in range(COLS):
            x1 = int(round(c * cell_w))
            y1 = int(round(r * cell_h))
            x2 = int(round((c + 1) * cell_w))
            y2 = int(round((r + 1) * cell_h))

            # inset reduces bleed into neighbor cell
            pad_x = int(0.06 * (x2 - x1))
            pad_y = int(0.06 * (y2 - y1))
            crop = warped_bgr[y1+pad_y:y2-pad_y, x1+pad_x:x2-pad_x]
            preds[r][c] = recognize_cell(crop)

    return preds

def draw_overlay(warped_bgr, preds, expected):
    out = warped_bgr.copy()
    H, W = out.shape[:2]
    cell_w = W / COLS
    cell_h = H / ROWS

    for r in range(ROWS):
        for c in range(COLS):
            x1 = int(round(c * cell_w))
            y1 = int(round(r * cell_h))
            x2 = int(round((c + 1) * cell_w))
            y2 = int(round((r + 1) * cell_h))

            pred = preds[r][c]
            exp = expected[r][c]
            ok = (pred == exp)

            # green-ish box if correct, red-ish if wrong (we'll just use white lines + text marks)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)

            txt = f"P:{pred if pred is not None else '?'} E:{exp} {'OK' if ok else 'X'}"
            cv2.putText(out, txt, (x1 + 10, y1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def score(preds, expected):
    correct = 0
    total = ROWS * COLS
    for r in range(ROWS):
        for c in range(COLS):
            if preds[r][c] == expected[r][c]:
                correct += 1
    return correct, total

def sharpness_score(img_bgr):
    """Higher = sharper. Used to pick the best frame from a burst."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def capture_best_frame(cam_index=0, burst=6):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Check macOS Camera permissions.")

    frames = []
    # Warm-up a bit
    for _ in range(10):
        cap.read()

    for _ in range(burst):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        time.sleep(0.05)

    cap.release()

    if not frames:
        raise RuntimeError("No frames captured from camera.")

    frames.sort(key=sharpness_score, reverse=True)
    return frames[0]

# ----------------------------
# GAME UI (Tkinter)
# ----------------------------
class SDGGameApp:
    def __init__(self, root, cam_index=0):
        self.root = root
        self.cam_index = cam_index

        self.root.title("SDG Tile Game (OCR MVP)")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)

        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.info_text = self.canvas.create_text(
            20, 20, anchor="nw", fill="white",
            font=("Helvetica", 26, "bold"),
            text=""
        )

        self.timer_text = self.canvas.create_text(
            20, 70, anchor="nw", fill="white",
            font=("Helvetica", 40, "bold"),
            text=""
        )

        self.status_text = self.canvas.create_text(
            20, 130, anchor="nw", fill="white",
            font=("Helvetica", 22),
            text="Press N to start. ESC to quit."
        )

        self.grid_items = []  # store drawn rectangles/text ids
        self.expected = None
        self.predictions = None  # store OCR predictions
        self.last_score = None  # store last score
        self.show_results = False  # toggle between expected and results view
        self.round_running = False
        self.end_time = None

        # Create training data directory
        os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("n", lambda e: self.start_new_round())
        self.root.bind("N", lambda e: self.start_new_round())
        self.root.bind("r", lambda e: self.start_new_round())
        self.root.bind("R", lambda e: self.start_new_round())
        self.root.bind("t", lambda e: self.toggle_results_view())
        self.root.bind("T", lambda e: self.toggle_results_view())

        self.root.bind("<Configure>", lambda e: self.redraw())

        self.redraw()

    def quit(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass
        self.root.destroy()

    def start_new_round(self):
        if self.round_running:
            return

        # Reset results view
        self.show_results = False
        self.predictions = None
        self.last_score = None

        # Generate random expected layout (numbers 1..17 with repetition)
        nums = [random.randint(1, 17) for _ in range(ROWS * COLS)]
        self.expected = [nums[r*COLS:(r+1)*COLS] for r in range(ROWS)]

        self.round_running = True
        self.end_time = time.time() + ROUND_SECONDS

        self.canvas.itemconfig(self.status_text, text="Match the pattern on the screen using physical tiles!")
        self.draw_grid(self.expected)
        self.update_timer_loop()

    def redraw(self):
        # Refresh UI layout when window size changes
        if self.show_results and self.predictions is not None:
            self.draw_results_grid()
        elif self.expected is not None:
            self.draw_grid(self.expected)

    def toggle_results_view(self):
        """Toggle between expected pattern and results comparison view."""
        if self.predictions is None:
            return
        self.show_results = not self.show_results
        self.redraw()

    def update_timer_loop(self):
        if not self.round_running:
            return

        remaining = int(self.end_time - time.time())
        if remaining < 0:
            remaining = 0

        self.canvas.itemconfig(self.info_text, text="SDG Tile Pattern (5 x 8)")
        self.canvas.itemconfig(self.timer_text, text=f"Time Left: {remaining:02d}s")

        if remaining == 0:
            self.round_running = False
            self.canvas.itemconfig(self.status_text, text="Time up! Capturing & scoring... (check results window)")
            # Run capture+OCR in another thread so UI doesn't freeze
            threading.Thread(target=self.process_scoring, daemon=True).start()
            return

        # tick again after 200ms
        self.root.after(200, self.update_timer_loop)

    def clear_grid(self):
        for item in self.grid_items:
            self.canvas.delete(item)
        self.grid_items = []

    def draw_grid(self, grid):
        self.clear_grid()

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # Layout region
        top_margin = 180
        side_margin = 60
        bottom_margin = 60

        grid_w = w - 2 * side_margin
        grid_h = h - top_margin - bottom_margin

        cell_w = grid_w / COLS
        cell_h = grid_h / ROWS

        # Draw
        for r in range(ROWS):
            for c in range(COLS):
                x1 = side_margin + c * cell_w
                y1 = top_margin + r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                val = grid[r][c]
                color = SDG_COLORS.get(val, "#444444")

                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color, outline="white", width=2
                )
                self.grid_items.append(rect)

                # Big number
                num_txt = self.canvas.create_text(
                    (x1 + x2) / 2, (y1 + y2) / 2 - 10,
                    text=str(val),
                    fill="white",
                    font=("Helvetica", int(min(cell_w, cell_h) * 0.35), "bold")
                )
                self.grid_items.append(num_txt)

                # Small hint
                hint_txt = self.canvas.create_text(
                    (x1 + x2) / 2, y2 - 18,
                    text="SDG",
                    fill="white",
                    font=("Helvetica", 14, "bold")
                )
                self.grid_items.append(hint_txt)

    def draw_results_grid(self):
        """Draw a comparison grid showing expected vs found values."""
        self.clear_grid()

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # Layout region
        top_margin = 180
        side_margin = 60
        bottom_margin = 60

        grid_w = w - 2 * side_margin
        grid_h = h - top_margin - bottom_margin

        cell_w = grid_w / COLS
        cell_h = grid_h / ROWS

        correct, total = self.last_score if self.last_score else (0, ROWS * COLS)

        # Draw header
        header = self.canvas.create_text(
            w / 2, top_margin - 40,
            text=f"RESULTS: {correct}/{total} correct  |  Press T to toggle view  |  Press N for new round",
            fill="yellow",
            font=("Helvetica", 20, "bold")
        )
        self.grid_items.append(header)

        # Draw comparison grid
        for r in range(ROWS):
            for c in range(COLS):
                x1 = side_margin + c * cell_w
                y1 = top_margin + r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                exp_val = self.expected[r][c]
                pred_val = self.predictions[r][c]
                is_correct = (exp_val == pred_val)

                # Green if correct, red if wrong
                if is_correct:
                    bg_color = "#2E7D32"  # Green
                    outline_color = "#4CAF50"
                else:
                    bg_color = "#C62828"  # Red
                    outline_color = "#EF5350"

                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=bg_color, outline=outline_color, width=3
                )
                self.grid_items.append(rect)

                # Show expected value
                exp_txt = self.canvas.create_text(
                    (x1 + x2) / 2, y1 + cell_h * 0.25,
                    text=f"Exp: {exp_val}",
                    fill="white",
                    font=("Helvetica", int(min(cell_w, cell_h) * 0.18), "bold")
                )
                self.grid_items.append(exp_txt)

                # Show found/predicted value
                pred_display = str(pred_val) if pred_val is not None else "?"
                found_txt = self.canvas.create_text(
                    (x1 + x2) / 2, y1 + cell_h * 0.55,
                    text=f"Got: {pred_display}",
                    fill="yellow" if not is_correct else "white",
                    font=("Helvetica", int(min(cell_w, cell_h) * 0.22), "bold")
                )
                self.grid_items.append(found_txt)

                # Show status
                status_txt = self.canvas.create_text(
                    (x1 + x2) / 2, y1 + cell_h * 0.82,
                    text="✓" if is_correct else "✗",
                    fill="white",
                    font=("Helvetica", int(min(cell_w, cell_h) * 0.2), "bold")
                )
                self.grid_items.append(status_txt)

    def process_scoring(self):
        """
        What happens at time end:
        1) capture best frame from camera
        2) detect board border + warp to top-down (or use full frame if no border)
        3) split into 40 cells (5x8)
        4) OCR each cell to predict 1..17
        5) compare with expected and score
        6) display results in Tkinter UI
        7) save images with timestamps for training
        """
        try:
            frame = capture_best_frame(self.cam_index, burst=8)
        except Exception as e:
            self.root.after(0, lambda: self.canvas.itemconfig(self.status_text, text=f"Camera error: {e}. Press R to retry."))
            return

        # Will use full frame if no border detected
        warped = find_board_and_warp(frame, require_border=False)

        preds = split_and_recognize(warped)
        correct, total = score(preds, self.expected)

        # Store predictions and score for UI display
        self.predictions = preds
        self.last_score = (correct, total)

        # Print grids to terminal
        print("\nEXPECTED (5x8):")
        for row in self.expected:
            print(row)

        print("\nPREDICTED (5x8):")
        for row in preds:
            print(row)

        print(f"\nSCORE: {correct}/{total}")

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save images with timestamps for training data
        frame_path = os.path.join(TRAINING_DATA_DIR, f"frame_{timestamp}.jpg")
        warped_path = os.path.join(TRAINING_DATA_DIR, f"warped_{timestamp}.jpg")
        overlay_path = os.path.join(TRAINING_DATA_DIR, f"overlay_{timestamp}.jpg")
        
        overlay = draw_overlay(warped, preds, self.expected)
        cv2.imwrite(frame_path, frame)
        cv2.imwrite(warped_path, warped)
        cv2.imwrite(overlay_path, overlay)
        
        # Save metadata (expected and predictions) as text file for training
        metadata_path = os.path.join(TRAINING_DATA_DIR, f"metadata_{timestamp}.txt")
        with open(metadata_path, "w") as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Score: {correct}/{total}\n\n")
            f.write("Expected:\n")
            for row in self.expected:
                f.write(str(row) + "\n")
            f.write("\nPredicted:\n")
            for row in preds:
                f.write(str(row) + "\n")

        print(f"\nSaved training data to: {TRAINING_DATA_DIR}/")
        print(f"  - {frame_path}")
        print(f"  - {warped_path}")
        print(f"  - {overlay_path}")
        print(f"  - {metadata_path}")

        # Show results in Tkinter UI (must be done from main thread)
        self.show_results = True
        self.root.after(0, self._update_ui_after_scoring, correct, total)

    def _update_ui_after_scoring(self, correct, total):
        """Update the UI after scoring is complete (called from main thread)."""
        self.canvas.itemconfig(self.info_text, text=f"RESULTS: {correct}/{total} correct")
        self.canvas.itemconfig(self.timer_text, text="")
        self.canvas.itemconfig(
            self.status_text,
            text=f"Score: {correct}/{total}. Press T to toggle view. Press N for new round. Images saved to {TRAINING_DATA_DIR}/"
        )
        self.draw_results_grid()

def main():
    global ROUND_SECONDS
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=CAM_INDEX_DEFAULT, help="Camera index (default 0).")
    parser.add_argument("--seconds", type=int, default=ROUND_SECONDS, help="Round duration in seconds.")
    args = parser.parse_args()

    ROUND_SECONDS = args.seconds

    root = tk.Tk()
    app = SDGGameApp(root, cam_index=args.cam)
    root.mainloop()

if __name__ == "__main__":
    main()