"""
Camera capture module for acquiring and preprocessing image frames.
"""
import cv2
import numpy as np
import time
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Camera:
    """Camera interface for capturing and preprocessing frames."""
    
    def __init__(self, camera_id=None, width=None, height=None, fps=None):
        """Initialize camera with specified parameters or use config defaults."""
        self.camera_id = camera_id if camera_id is not None else config.CAMERA_ID
        self.width = width if width is not None else config.CAMERA_WIDTH
        self.height = height if height is not None else config.CAMERA_HEIGHT
        self.fps = fps if fps is not None else config.FRAME_RATE
        
        self.cap = None
        self.is_open = False
    
    def open(self):
        """Open the camera connection and configure it."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Check if camera is opened successfully
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera with ID {self.camera_id}")
            
            # Read a test frame to confirm camera is working
            ret, _ = self.cap.read()
            if not ret:
                raise Exception("Camera opened but failed to read frame")
            
            self.is_open = True
            print(f"Camera opened successfully: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"Error opening camera: {str(e)}")
            self.close()
            return False
    
    def read(self):
        """Read a frame from the camera."""
        if not self.is_open:
            print("Cannot read: Camera is not open")
            return False, None
        
        return self.cap.read()
    
    def close(self):
        """Close the camera connection."""
        if self.cap is not None:
            self.cap.release()
        self.is_open = False
        print("Camera closed")
    
    def preprocess_frame(self, frame):
        """
        Apply standard preprocessing to the frame.
        Returns dictionary with various processed versions.
        """
        if frame is None:
            return None
        
        # Create results dictionary
        results = {
            'original': frame.copy()
        }
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results['gray'] = gray
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray, 
            config.BLUR_KERNEL_SIZE, 
            0
        )
        results['blurred'] = blurred
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            blurred, 
            config.CANNY_THRESHOLD1, 
            config.CANNY_THRESHOLD2
        )
        results['edges'] = edges
        
        # HSV color space for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results['hsv'] = hsv
        
        return results


def display_frame(frame, window_name="Frame", scale=1.0):
    """Display a frame with optional scaling."""
    if scale != 1.0:
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (width, height))
    
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1)


# Simple demo/test function
def test_camera():
    """Test camera functionality with live preview."""
    camera = Camera()
    
    if not camera.open():
        print("Failed to open camera. Exiting...")
        return
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret:
                print("Failed to read frame. Exiting...")
                break
            
            # Process the frame
            processed = camera.preprocess_frame(frame)
            
            # Display original and processed frames
            display_frame(processed['original'], "Original")
            display_frame(processed['gray'], "Grayscale")
            display_frame(processed['edges'], "Edges")
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera() 