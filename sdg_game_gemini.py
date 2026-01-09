import tkinter as tk
import random
import time
import threading
import cv2
import numpy as np
import os
import json
import base64
import re
from datetime import datetime
import requests
from PIL import Image, ImageTk
import fitz  # PyMuPDF for PDF to image conversion

# ----------------------------
# GAME CONFIG
# ----------------------------
ROWS, COLS = 5, 8
ROUND_SECONDS = 60

# Warped board size (higher = better recognition)
WARP_W, WARP_H = 2000, 1250

# Camera index (0 usually works on Mac)
CAM_INDEX_DEFAULT = 0

# Training data folder
TRAINING_DATA_DIR = "training_data_gemini"

# Leaderboard file
LEADERBOARD_FILE = "leaderboard.json"

# Google Cloud Vision API Key - Set via environment variable or directly here
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# SDG-ish colors (approx). You can edit.
SDG_COLORS = {
    1:  "#E5243B", 2:  "#DDA63A", 3:  "#4C9F38", 4:  "#C5192D",
    5:  "#FF3A21", 6:  "#26BDE2", 7:  "#FCC30B", 8:  "#A21942",
    9:  "#FD6925", 10: "#DD1367", 11: "#FD9D24", 12: "#BF8B2E",
    13: "#3F7E44", 14: "#0A97D9", 15: "#56C02B", 16: "#00689D",
    17: "#19486A"
}

# SDG Images folder
IMAGES_FOLDER = "images"

# SDG PDF filenames mapping (number to filename)
SDG_PDF_FILES = {
    1: "1.pdf",
    2: "2.pdf",
    3: "3.pdf",
    4: "4.pdf",
    5: "5.pdf",
    6: "6.pdf",
    7: "7.pdf",
    8: "8.pdf",
    9: "9.pdf",
    10: "10.pdf",
    11: "11.pdf",
    12: "12.pdf",
    13: "13.pdf",
    14: "14.pdf",
    15: "15.pdf",
    16: "16.pdf",
    17: "17.pdf"
}

def load_sdg_images_from_pdfs():
    """
    Load all SDG PDF files and convert them to PIL images.
    Returns a dictionary mapping SDG number (1-17) to PIL Image.
    """
    sdg_images = {}
    
    for num, filename in SDG_PDF_FILES.items():
        pdf_path = os.path.join(IMAGES_FOLDER, filename)
        if os.path.exists(pdf_path):
            try:
                # Open PDF with PyMuPDF
                doc = fitz.open(pdf_path)
                if len(doc) > 0:
                    page = doc[0]  # Get first page
                    # Render at higher resolution for better quality
                    mat = fitz.Matrix(3, 3)  # 3x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    sdg_images[num] = img
                doc.close()
            except Exception as e:
                print(f"Error loading PDF {filename}: {e}")
        else:
            print(f"Warning: PDF not found: {pdf_path}")
    
    return sdg_images

# ----------------------------
# LEADERBOARD HELPERS
# ----------------------------
def load_leaderboard():
    """Load leaderboard from JSON file. Returns list of score entries."""
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading leaderboard: {e}")
    return []

def save_leaderboard(leaderboard):
    """Save leaderboard to JSON file."""
    try:
        with open(LEADERBOARD_FILE, "w") as f:
            json.dump(leaderboard, f, indent=2)
    except Exception as e:
        print(f"Error saving leaderboard: {e}")

def add_score_to_leaderboard(player_name, correct, total):
    """Add a new score to leaderboard and keep top 10."""
    leaderboard = load_leaderboard()
    percentage = round(correct / total * 100, 2)
    
    entry = {
        "name": player_name,
        "correct": correct,
        "total": total,
        "percentage": percentage,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    leaderboard.append(entry)
    # Sort by percentage (descending), then by correct count (descending)
    leaderboard.sort(key=lambda x: (x["percentage"], x["correct"]), reverse=True)
    # Keep only top 10
    leaderboard = leaderboard[:10]
    
    save_leaderboard(leaderboard)
    return leaderboard

# ----------------------------
# GEMINI VISION API HELPER
# ----------------------------
def image_to_base64(cv2_image):
    """Convert OpenCV BGR image to base64 encoded JPEG."""
    _, buffer = cv2.imencode('.jpg', cv2_image)
    return base64.b64encode(buffer).decode('utf-8')

def recognize_grid_with_gemini(warped_bgr):
    """
    Send the warped image to Gemini Vision API and get the 5x8 grid of SDG numbers.
    Gemini analyzes the image and identifies which SDG tile (1-17) is in each cell.
    Returns a 5x8 list of lists with integers 1-17 or None for empty/unrecognized cells.
    """
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not set. Please set the GOOGLE_API_KEY environment variable "
            "or edit the GOOGLE_API_KEY variable in this script."
        )
    
    # Convert image to base64
    image_base64 = image_to_base64(warped_bgr)
    
    # Gemini API endpoint (using gemini-1.5-flash for speed, or gemini-1.5-pro for accuracy)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
    
    # Prompt for Gemini to analyze the SDG tiles grid
    prompt = """Analyze this image of a 5x8 grid containing UN Sustainable Development Goal (SDG) tiles.

The grid has 5 rows and 8 columns. Each cell may contain an SDG tile (numbered 1-17) or be empty.

SDG tiles are colorful squares with:
- A number (1-17) 
- An icon representing the goal
- Distinctive colors (e.g., SDG 1 is red, SDG 5 is orange/red, SDG 14 is blue, SDG 15 is green, etc.)

Look at each cell position and identify which SDG number (1-17) is present. If a cell is empty or has no tile, use null.

IMPORTANT: Return ONLY a JSON array with exactly 5 rows and 8 columns, nothing else. Format:
[[row0_col0, row0_col1, ..., row0_col7], [row1_col0, ..., row1_col7], ..., [row4_col0, ..., row4_col7]]

Each value should be an integer 1-17 or null if empty.

Example response for a grid where only bottom-left has SDG 5:
[[null,null,null,null,null,null,null,null],[null,null,null,null,null,null,null,null],[null,null,null,null,null,null,null,null],[null,null,null,null,null,null,null,null],[5,null,null,null,null,null,null,null]]

Now analyze the image and return the JSON array:"""

    # Request payload for Gemini
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.8,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract the text response from Gemini
        if "candidates" not in result or len(result["candidates"]) == 0:
            print("No response from Gemini API")
            return [[None for _ in range(COLS)] for _ in range(ROWS)]
        
        text_response = result["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
        print(f"Gemini response: {text_response[:200]}...")
        
        # Parse the JSON array from the response
        # Clean up the response - remove markdown code blocks if present
        text_response = text_response.strip()
        if text_response.startswith("```"):
            # Remove markdown code block markers
            lines = text_response.split("\n")
            text_response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text_response = text_response.strip()
        
        # Parse JSON
        try:
            grid = json.loads(text_response)
        except json.JSONDecodeError:
            # Try to extract JSON array using regex
            import re
            match = re.search(r'\[\s*\[.*\]\s*\]', text_response, re.DOTALL)
            if match:
                grid = json.loads(match.group())
            else:
                print(f"Could not parse Gemini response as JSON: {text_response}")
                return [[None for _ in range(COLS)] for _ in range(ROWS)]
        
        # Validate and clean the grid
        if not isinstance(grid, list) or len(grid) != ROWS:
            print(f"Invalid grid dimensions: expected {ROWS} rows, got {len(grid) if isinstance(grid, list) else 'non-list'}")
            return [[None for _ in range(COLS)] for _ in range(ROWS)]
        
        cleaned_grid = []
        for r, row in enumerate(grid):
            if not isinstance(row, list) or len(row) != COLS:
                print(f"Invalid row {r}: expected {COLS} cols, got {len(row) if isinstance(row, list) else 'non-list'}")
                cleaned_grid.append([None for _ in range(COLS)])
                continue
            
            cleaned_row = []
            for val in row:
                if val is None or val == "null":
                    cleaned_row.append(None)
                elif isinstance(val, int) and 1 <= val <= 17:
                    cleaned_row.append(val)
                elif isinstance(val, str):
                    try:
                        num = int(val)
                        cleaned_row.append(num if 1 <= num <= 17 else None)
                    except ValueError:
                        cleaned_row.append(None)
                else:
                    cleaned_row.append(None)
            cleaned_grid.append(cleaned_row)
        
        # Count detected tiles
        detected = sum(1 for row in cleaned_grid for cell in row if cell is not None)
        print(f"Gemini detected {detected} SDG tiles in the grid")
        
        return cleaned_grid
        
    except requests.exceptions.RequestException as e:
        print(f"Gemini API request error: {e}")
        return [[None for _ in range(COLS)] for _ in range(ROWS)]
    except Exception as e:
        print(f"Gemini API error: {e}")
        import traceback
        traceback.print_exc()
        return [[None for _ in range(COLS)] for _ in range(ROWS)]

# Keep old function as fallback
def recognize_grid_with_vision_api(warped_bgr):
    """Fallback: Use Google Cloud Vision TEXT_DETECTION API."""
    return recognize_grid_with_gemini(warped_bgr)

# ----------------------------
# VISION HELPERS
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

def warp_board(frame_bgr, corners):
    """
    Warp the detected board to a fixed-size rectangle.
    """
    if corners is None:
        # No corners - use full frame resized
        warped = cv2.resize(frame_bgr, (WARP_W, WARP_H), interpolation=cv2.INTER_CUBIC)
        # Rotate 180 degrees
        warped = cv2.rotate(warped, cv2.ROTATE_180)
        return warped
    
    ordered_corners = order_points(corners.astype("float32"))
    
    dst = np.array([
        [0, 0],
        [WARP_W - 1, 0],
        [WARP_W - 1, WARP_H - 1],
        [0, WARP_H - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (WARP_W, WARP_H))
    
    # Rotate 180 degrees
    warped = cv2.rotate(warped, cv2.ROTATE_180)
    
    return warped

def draw_grid_on_warped(warped_bgr):
    """
    Draw the 5x8 grid on the warped (top-down) image with cell position labels.
    This helps Gemini identify which cell each number belongs to.
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
        cv2.line(overlay, (0, y), (W, y), (255, 255, 255), 3)
    
    for c in range(COLS + 1):
        x = int(c * cell_w)
        cv2.line(overlay, (x, 0), (x, H), (255, 255, 255), 3)
    
    # Draw cell position labels (row, col) in corner of each cell
    for r in range(ROWS):
        for c in range(COLS):
            x = int(c * cell_w) + 5
            y = int((r + 1) * cell_h) - 10
            label = f"[{r},{c}]"
            cv2.putText(overlay, label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return overlay

def find_board_and_warp(frame_bgr, require_border=False):
    """
    Detect outer border (largest 4-corner contour) and warp to fixed size.
    If no border is detected and require_border=False, uses the full frame resized.
    The result is rotated 180 degrees.
    Returns: (warped_image, warped_with_grid)
    """
    corners, _ = find_board_corners(frame_bgr)
    
    # Warp the board (includes 180 degree rotation)
    warped = warp_board(frame_bgr, corners)
    
    # Create version with grid overlay for Gemini
    warped_with_grid = draw_grid_on_warped(warped)
    
    return warped, warped_with_grid

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

            # green-ish box if correct, red-ish if wrong
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

        self.root.title("SDG Tile Game (Google Cloud Vision)")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)

        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Load SDG images from PDFs
        print("Loading SDG images from PDFs...")
        self.sdg_pil_images = load_sdg_images_from_pdfs()
        print(f"Loaded {len(self.sdg_pil_images)} SDG images")
        
        # Dictionary to store resized PhotoImage objects (will be populated on draw)
        self.sdg_tk_images = {}
        self.current_cell_size = (0, 0)  # Track current cell size for resizing

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
        self.predictions = None  # store Gemini predictions
        self.last_score = None  # store last score
        self.gemini_raw_response = None  # store raw Gemini response for training
        self.show_results = False  # toggle between expected and results view
        self.round_running = False
        self.end_time = None
        
        # Player info
        self.player_name = None
        self.show_leaderboard = True  # Start with leaderboard view

        # Create training data directory
        os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("n", lambda e: self.prompt_player_name())
        self.root.bind("N", lambda e: self.prompt_player_name())
        self.root.bind("r", lambda e: self.prompt_player_name())
        self.root.bind("R", lambda e: self.prompt_player_name())
        self.root.bind("h", lambda e: self.show_home_screen())
        self.root.bind("H", lambda e: self.show_home_screen())

        self.root.bind("<Configure>", lambda e: self.redraw())

        # Show leaderboard on startup
        self.show_home_screen()

    def quit(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass
        self.root.destroy()

    def show_home_screen(self):
        """Show the leaderboard/home screen."""
        if self.round_running:
            return
        self.show_leaderboard = True
        self.show_results = False
        self.expected = None
        self.player_name = None
        self.draw_leaderboard()

    def draw_leaderboard(self):
        """Draw the leaderboard screen."""
        self.clear_grid()
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        # Title
        title = self.canvas.create_text(
            w / 2, 100,
            text="🏆 SDG TILE GAME - LEADERBOARD 🏆",
            fill="#FFD700",
            font=("Helvetica", 64, "bold"),
            anchor="center"
        )
        self.grid_items.append(title)
        
        # Load leaderboard data
        leaderboard = load_leaderboard()
        
        # Leaderboard table header
        header_y = 220
        headers = ["RANK", "PLAYER", "SCORE", "PERCENTAGE", "DATE"]
        header_x_positions = [w * 0.1, w * 0.3, w * 0.5, w * 0.65, w * 0.85]
        
        for i, header in enumerate(headers):
            header_txt = self.canvas.create_text(
                header_x_positions[i], header_y,
                text=header,
                fill="#00BFFF",
                font=("Helvetica", 32, "bold"),
                anchor="center"
            )
            self.grid_items.append(header_txt)
        
        # Separator line
        line = self.canvas.create_line(
            w * 0.05, header_y + 40, w * 0.95, header_y + 40,
            fill="#00BFFF", width=3
        )
        self.grid_items.append(line)
        
        # Leaderboard entries
        if leaderboard:
            for i, entry in enumerate(leaderboard[:10]):
                row_y = header_y + 90 + i * 70
                
                # Rank with medal for top 3
                rank_text = f"{i + 1}"
                if i == 0:
                    rank_text = "🥇 1"
                    rank_color = "#FFD700"
                elif i == 1:
                    rank_text = "🥈 2"
                    rank_color = "#C0C0C0"
                elif i == 2:
                    rank_text = "🥉 3"
                    rank_color = "#CD7F32"
                else:
                    rank_color = "white"
                
                rank_txt = self.canvas.create_text(
                    header_x_positions[0], row_y,
                    text=rank_text,
                    fill=rank_color,
                    font=("Helvetica", 30, "bold"),
                    anchor="center"
                )
                self.grid_items.append(rank_txt)
                
                # Player name
                name_txt = self.canvas.create_text(
                    header_x_positions[1], row_y,
                    text=entry.get("name", "Unknown")[:15],
                    fill="white",
                    font=("Helvetica", 30),
                    anchor="center"
                )
                self.grid_items.append(name_txt)
                
                # Score
                score_txt = self.canvas.create_text(
                    header_x_positions[2], row_y,
                    text=f"{entry.get('correct', 0)}/{entry.get('total', 40)}",
                    fill="white",
                    font=("Helvetica", 30),
                    anchor="center"
                )
                self.grid_items.append(score_txt)
                
                # Percentage
                pct_txt = self.canvas.create_text(
                    header_x_positions[3], row_y,
                    text=f"{entry.get('percentage', 0)}%",
                    fill="#00FF00" if entry.get('percentage', 0) >= 80 else "#FFFF00" if entry.get('percentage', 0) >= 50 else "#FF6347",
                    font=("Helvetica", 30, "bold"),
                    anchor="center"
                )
                self.grid_items.append(pct_txt)
                
                # Date
                date_txt = self.canvas.create_text(
                    header_x_positions[4], row_y,
                    text=entry.get("timestamp", "")[:10],
                    fill="#AAAAAA",
                    font=("Helvetica", 26),
                    anchor="center"
                )
                self.grid_items.append(date_txt)
        else:
            # No entries message
            no_entries = self.canvas.create_text(
                w / 2, h / 2,
                text="No scores yet! Be the first to play!",
                fill="#888888",
                font=("Helvetica", 40),
                anchor="center"
            )
            self.grid_items.append(no_entries)
        
        # Update status text
        self.canvas.itemconfig(self.info_text, text="")
        self.canvas.itemconfig(self.timer_text, text="")
        self.canvas.itemconfig(
            self.status_text,
            text="Press N to start a new game  |  ESC to quit"
        )

    def prompt_player_name(self):
        """Prompt for player name before starting the game."""
        if self.round_running:
            return
        
        # Create a simple dialog to get player name
        from tkinter import simpledialog
        
        name = simpledialog.askstring(
            "Player Name",
            "Enter your name:",
            parent=self.root
        )
        
        if name and name.strip():
            self.player_name = name.strip()
            self.show_leaderboard = False
            self.start_new_round()
        else:
            # User cancelled or entered empty name
            pass

    def start_new_round(self):
        if self.round_running:
            return

        # Reset results view
        self.show_results = False
        self.show_leaderboard = False
        self.predictions = None
        self.last_score = None
        self.gemini_raw_response = None

        # Generate random expected layout (numbers 1..17, each number max 3 times)
        # With 40 cells and 17 numbers, we need at least 40/3 ≈ 14 different numbers
        # Each number can appear 1-3 times, total must equal 40
        available = []
        for num in range(1, 18):
            available.extend([num] * 3)  # Each number available up to 3 times
        
        random.shuffle(available)
        nums = available[:ROWS * COLS]  # Take exactly 40 numbers
        random.shuffle(nums)  # Shuffle again for good distribution
        
        self.expected = [nums[r*COLS:(r+1)*COLS] for r in range(ROWS)]

        # Print the generated matrix to console
        print("\n" + "="*50)
        print("GENERATED 5x8 MATRIX (Expected Pattern):")
        print("="*50)
        for r, row in enumerate(self.expected):
            print(f"Row {r}: {row}")
        print("="*50 + "\n")

        self.round_running = True
        self.end_time = time.time() + ROUND_SECONDS

        self.canvas.itemconfig(self.status_text, text="Match the pattern on the screen using physical tiles!")
        self.draw_grid(self.expected)
        self.update_timer_loop()

    def redraw(self):
        # Refresh UI layout when window size changes
        if self.show_leaderboard:
            self.draw_leaderboard()
        elif self.show_results and self.predictions is not None and self.last_score is not None:
            correct, total = self.last_score
            percentage = round(correct / total * 100, 1)
            self.draw_score_screen(correct, total, percentage)
        elif self.expected is not None:
            self.draw_grid(self.expected)

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
            self.canvas.itemconfig(self.status_text, text="Time up! Capturing & sending to Vision API... Please wait.")
            # Run capture+Gemini in another thread so UI doesn't freeze
            threading.Thread(target=self.process_scoring, daemon=True).start()
            return

        # tick again after 200ms
        self.root.after(200, self.update_timer_loop)

    def clear_grid(self):
        for item in self.grid_items:
            self.canvas.delete(item)
        self.grid_items = []

    def resize_sdg_images(self, cell_w, cell_h):
        """
        Resize all SDG images to fit the cell size.
        Caches resized images to avoid repeated resizing.
        """
        new_size = (int(cell_w), int(cell_h))
        
        # Only resize if cell size changed
        if new_size == self.current_cell_size and self.sdg_tk_images:
            return
        
        self.current_cell_size = new_size
        self.sdg_tk_images = {}
        
        padding = 4  # Small padding inside cell
        img_w = max(1, int(cell_w - padding * 2))
        img_h = max(1, int(cell_h - padding * 2))
        
        for num, pil_img in self.sdg_pil_images.items():
            try:
                # Resize maintaining aspect ratio
                resized = pil_img.copy()
                resized.thumbnail((img_w, img_h), Image.Resampling.LANCZOS)
                self.sdg_tk_images[num] = ImageTk.PhotoImage(resized)
            except Exception as e:
                print(f"Error resizing SDG {num} image: {e}")

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

        # Resize SDG images if needed
        self.resize_sdg_images(cell_w, cell_h)

        # Draw
        for r in range(ROWS):
            for c in range(COLS):
                x1 = side_margin + c * cell_w
                y1 = top_margin + r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                val = grid[r][c]
                color = SDG_COLORS.get(val, "#444444")

                # Draw cell background
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color, outline="white", width=2
                )
                self.grid_items.append(rect)

                # Draw SDG image if available
                if val in self.sdg_tk_images:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    img_item = self.canvas.create_image(
                        center_x, center_y,
                        image=self.sdg_tk_images[val],
                        anchor="center"
                    )
                    self.grid_items.append(img_item)
                else:
                    # Fallback: show number if image not available
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
        1) show 5 second countdown for user to move away from the board
        2) capture best frame from camera
        3) detect board border + warp to top-down (or use full frame if no border)
        4) send warped image to Gemini API
        5) parse Gemini response to get 5x8 grid
        6) compare with expected and score
        7) display results in Tkinter UI
        8) save images and Gemini response for training
        """
        # Show 5 second countdown for user to move away from the board
        for countdown in range(5, 0, -1):
            self.root.after(0, lambda c=countdown: self.canvas.itemconfig(
                self.status_text, 
                text=f"Step away from the board... capturing in {c} seconds"
            ))
            self.root.after(0, lambda c=countdown: self.canvas.itemconfig(
                self.timer_text,
                text=f"📸 {c}"
            ))
            time.sleep(1)
        
        self.root.after(0, lambda: self.canvas.itemconfig(self.status_text, text="Capturing image..."))
        self.root.after(0, lambda: self.canvas.itemconfig(self.timer_text, text="📸"))
        
        try:
            frame = capture_best_frame(self.cam_index, burst=8)
        except Exception as e:
            self.root.after(0, lambda: self.canvas.itemconfig(self.status_text, text=f"Camera error: {e}. Press R to retry."))
            return

        # Detect corners, warp board, and get grid-overlaid version
        warped, warped_with_grid = find_board_and_warp(frame, require_border=False)

        # Send the RAW warped image to Gemini API (not the grid overlay)
        # Gemini will intelligently analyze the SDG tiles and determine positions
        try:
            preds = recognize_grid_with_gemini(warped)
        except Exception as e:
            self.root.after(0, lambda: self.canvas.itemconfig(self.status_text, text=f"Gemini API error: {e}. Press R to retry."))
            return

        correct, total = score(preds, self.expected)

        # Store predictions and score for UI display
        self.predictions = preds
        self.last_score = (correct, total)

        # Print grids to terminal
        print("\nEXPECTED (5x8):")
        for row in self.expected:
            print(row)

        print("\nGEMINI PREDICTED (5x8):")
        for row in preds:
            print(row)

        print(f"\nSCORE: {correct}/{total}")

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save images with timestamps for training data
        frame_path = os.path.join(TRAINING_DATA_DIR, f"frame_{timestamp}.jpg")
        warped_path = os.path.join(TRAINING_DATA_DIR, f"warped_{timestamp}.jpg")
        warped_grid_path = os.path.join(TRAINING_DATA_DIR, f"warped_grid_{timestamp}.jpg")
        overlay_path = os.path.join(TRAINING_DATA_DIR, f"overlay_{timestamp}.jpg")
        
        overlay = draw_overlay(warped, preds, self.expected)
        cv2.imwrite(frame_path, frame)
        cv2.imwrite(warped_path, warped)
        cv2.imwrite(warped_grid_path, warped_with_grid)
        cv2.imwrite(overlay_path, overlay)
        
        # Save comprehensive training data as JSON
        training_data = {
            "timestamp": timestamp,
            "score": {
                "correct": correct,
                "total": total,
                "percentage": round(correct / total * 100, 2)
            },
            "expected_grid": self.expected,
            "vision_api_prediction": preds,
            "comparison": []
        }
        
        # Add cell-by-cell comparison for training analysis
        for r in range(ROWS):
            for c in range(COLS):
                cell_data = {
                    "row": r,
                    "col": c,
                    "expected": self.expected[r][c],
                    "predicted": preds[r][c],
                    "correct": self.expected[r][c] == preds[r][c]
                }
                training_data["comparison"].append(cell_data)
        
        # Save as JSON for easy parsing
        json_path = os.path.join(TRAINING_DATA_DIR, f"training_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(training_data, f, indent=2)
        
        # Also save human-readable metadata
        metadata_path = os.path.join(TRAINING_DATA_DIR, f"metadata_{timestamp}.txt")
        with open(metadata_path, "w") as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Score: {correct}/{total} ({round(correct/total*100, 1)}%)\n\n")
            f.write("Expected Grid:\n")
            for row in self.expected:
                f.write(str(row) + "\n")
            f.write("\nVision API Predicted Grid:\n")
            for row in preds:
                f.write(str(row) + "\n")
            f.write("\nCell-by-cell Analysis:\n")
            for r in range(ROWS):
                for c in range(COLS):
                    exp = self.expected[r][c]
                    pred = preds[r][c]
                    status = "✓" if exp == pred else "✗"
                    f.write(f"  [{r},{c}] Expected: {exp:2d} | Got: {str(pred) if pred else '?':>2} | {status}\n")

        print(f"\nSaved training data to: {TRAINING_DATA_DIR}/")
        print(f"  - {frame_path}")
        print(f"  - {warped_path}")
        print(f"  - {overlay_path}")
        print(f"  - {json_path}")
        print(f"  - {metadata_path}")

        # Show results in Tkinter UI (must be done from main thread)
        self.show_results = True
        self.root.after(0, self._update_ui_after_scoring, correct, total)

    def _update_ui_after_scoring(self, correct, total):
        """Update the UI after scoring is complete (called from main thread)."""
        percentage = round(correct / total * 100, 1)
        
        # Save score to leaderboard
        if self.player_name:
            add_score_to_leaderboard(self.player_name, correct, total)
            print(f"Score saved for {self.player_name}: {correct}/{total} ({percentage}%)")
        
        # Print detailed results to console
        print("\n" + "="*60)
        print(f"RESULTS FOR: {self.player_name}")
        print("="*60)
        print(f"SCORE: {correct}/{total} ({percentage}%)")
        print("\nEXPECTED vs PREDICTED COMPARISON:")
        print("-"*60)
        for r in range(ROWS):
            exp_row = self.expected[r]
            pred_row = self.predictions[r]
            matches = ["✓" if exp_row[c] == pred_row[c] else "✗" for c in range(COLS)]
            print(f"Row {r}: Exp {exp_row}")
            print(f"        Got {pred_row}")
            print(f"        {' '.join(f'{m:>2}' for m in matches)}")
        print("="*60 + "\n")
        
        # Draw big score on screen
        self.draw_score_screen(correct, total, percentage)
        
        # Return to home screen after 10 seconds
        self.root.after(10000, self.show_home_screen)

    def draw_score_screen(self, correct, total, percentage):
        """Draw a big score display in the center of the screen."""
        self.clear_grid()
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        # Hide default text items
        self.canvas.itemconfig(self.info_text, text="")
        self.canvas.itemconfig(self.timer_text, text="")
        self.canvas.itemconfig(self.status_text, text="")
        
        # Player name
        player_txt = self.canvas.create_text(
            w / 2, h * 0.15,
            text=f"🎮 {self.player_name} 🎮",
            fill="#00BFFF",
            font=("Helvetica", 56, "bold"),
            anchor="center"
        )
        self.grid_items.append(player_txt)
        
        # Big score number
        score_txt = self.canvas.create_text(
            w / 2, h * 0.45,
            text=f"{correct}/{total}",
            fill="#FFFFFF",
            font=("Helvetica", 180, "bold"),
            anchor="center"
        )
        self.grid_items.append(score_txt)
        
        # Percentage with color based on performance
        if percentage >= 80:
            pct_color = "#00FF00"  # Green
            emoji = "🏆"
        elif percentage >= 60:
            pct_color = "#FFFF00"  # Yellow
            emoji = "👍"
        elif percentage >= 40:
            pct_color = "#FFA500"  # Orange
            emoji = "💪"
        else:
            pct_color = "#FF6347"  # Red
            emoji = "🎯"
        
        pct_txt = self.canvas.create_text(
            w / 2, h * 0.68,
            text=f"{emoji} {percentage}% {emoji}",
            fill=pct_color,
            font=("Helvetica", 90, "bold"),
            anchor="center"
        )
        self.grid_items.append(pct_txt)
        
        # Status message
        status_txt = self.canvas.create_text(
            w / 2, h * 0.88,
            text="Returning to leaderboard in 10 seconds...  |  Press H for home",
            fill="#888888",
            font=("Helvetica", 28),
            anchor="center"
        )
        self.grid_items.append(status_txt)

def main():
    global ROUND_SECONDS, GOOGLE_API_KEY
    import argparse
    parser = argparse.ArgumentParser(description="SDG Tile Game with Google Cloud Vision")
    parser.add_argument("--cam", type=int, default=CAM_INDEX_DEFAULT, help="Camera index (default 0).")
    parser.add_argument("--seconds", type=int, default=ROUND_SECONDS, help="Round duration in seconds.")
    parser.add_argument("--api-key", type=str, default=None, help="Google Cloud Vision API key (or set GOOGLE_API_KEY env var).")
    args = parser.parse_args()

    ROUND_SECONDS = args.seconds
    
    if args.api_key:
        GOOGLE_API_KEY = args.api_key
    
    # Validate API key is set
    if not GOOGLE_API_KEY:
        print("ERROR: Google Cloud Vision API key not set!")
        print("Set it via:")
        print("  1. Environment variable: export GOOGLE_API_KEY='your-api-key'")
        print("  2. Command line: --api-key 'your-api-key'")
        print("  3. Edit GOOGLE_API_KEY in this script")
        return

    root = tk.Tk()
    app = SDGGameApp(root, cam_index=args.cam)
    root.mainloop()

if __name__ == "__main__":
    main()
