"""
Camera Zoom Capture Script
Captures images while progressively zooming out, with 2 second delays.
"""

import cv2
import time
import os

def preview_camera():
    """Open camera at 100% zoom and keep it open until user presses 'q'"""
    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get the original frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera resolution: {frame_width}x{frame_height}")
    print("Camera is open at 100% zoom (full frame)")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Rotate frame 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        cv2.imshow('Camera Preview - Press Q to quit', frame)
        
        # Check for 'q' key press (wait 1ms)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing camera...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

def capture_with_zoom():
    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Give camera time to warm up
    time.sleep(1)
    
    # Get the original frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    # Define zoom levels (1.0 = full zoom in, smaller = zoomed out)
    # We start zoomed in and progressively zoom out
    zoom_levels = [0.4, 0.6, 0.8, 1.0]  # 40%, 60%, 80%, 100% of frame (zooming out)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Starting capture sequence...")
    print("Press 'q' to quit early\n")
    
    for i, zoom in enumerate(zoom_levels, start=1):
        # Read a frame from the camera
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error: Could not read frame {i}")
            continue
        
        # Calculate crop dimensions based on zoom level
        # Lower zoom value = more zoomed in (smaller crop area)
        crop_width = int(frame_width * zoom)
        crop_height = int(frame_height * zoom)
        
        # Calculate the starting point for center crop
        x_start = (frame_width - crop_width) // 2
        y_start = (frame_height - crop_height) // 2
        
        # Crop the frame (simulating zoom)
        cropped_frame = frame[y_start:y_start + crop_height, x_start:x_start + crop_width]
        
        # Resize back to original dimensions for consistent output
        zoomed_frame = cv2.resize(cropped_frame, (frame_width, frame_height))
        
        # Save the image
        filename = os.path.join(output_dir, f"test-zoom-{i}.jpg")
        cv2.imwrite(filename, zoomed_frame)
        
        zoom_percentage = int(zoom * 100)
        print(f"Captured: test-zoom-{i}.jpg (Zoom level: {zoom_percentage}% of frame)")
        
        # Display the frame (optional)
        cv2.imshow('Zoom Capture', zoomed_frame)
        
        # Wait for 2 seconds (or break if 'q' is pressed)
        if i < len(zoom_levels):
            print(f"Waiting 2 seconds before next capture...")
            key = cv2.waitKey(2000)
            if key == ord('q'):
                print("Capture cancelled by user")
                break
    
    print("\nCapture complete!")
    print(f"Images saved in: {output_dir}")
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    preview_camera()  # Run preview mode - press 'q' to quit
