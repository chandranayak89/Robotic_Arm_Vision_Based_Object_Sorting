"""
Color-based object detection module.
Uses HSV color thresholds to identify objects by color.
"""
import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ColorDetector:
    """Detects objects based on their color using HSV color space."""
    
    def __init__(self, color_ranges=None):
        """
        Initialize with color range thresholds.
        If none provided, uses the defaults from config.
        """
        self.color_ranges = color_ranges if color_ranges else config.COLOR_RANGES
    
    def detect(self, hsv_frame):
        """
        Detect objects by color in HSV frame.
        Returns a dictionary with color keys and corresponding masks.
        """
        if hsv_frame is None:
            return None
        
        results = {}
        
        # Process each color
        for color_name, ranges in self.color_ranges.items():
            # Create mask for each color
            lower = np.array(ranges["lower"])
            upper = np.array(ranges["upper"])
            mask = cv2.inRange(hsv_frame, lower, upper)
            
            # Handle red which wraps around in HSV space
            if "second_lower" in ranges and "second_upper" in ranges:
                second_lower = np.array(ranges["second_lower"])
                second_upper = np.array(ranges["second_upper"])
                second_mask = cv2.inRange(hsv_frame, second_lower, second_upper)
                # Combine the two red masks
                mask = cv2.bitwise_or(mask, second_mask)
            
            # Store the mask
            results[color_name] = mask
        
        return results
    
    def find_objects(self, hsv_frame, min_area=None):
        """
        Detect objects and find their contours and centroids.
        Returns a list of detected objects with color, contour, and centroid.
        """
        if hsv_frame is None:
            return []
        
        # Use the minimum area from config if not specified
        if min_area is None:
            min_area = config.CONTOUR_MIN_AREA
        
        # Get color masks
        color_masks = self.detect(hsv_frame)
        
        objects = []
        
        # Process each color mask
        for color_name, mask in color_masks.items():
            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Process each contour
            for contour in contours:
                # Filter out small contours
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                
                # Calculate center of the contour (centroid)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    # Fallback if moments method fails
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = x + w // 2, y + h // 2
                
                # Create object with color, contour and centroid
                detected_object = {
                    "color": color_name,
                    "contour": contour,
                    "centroid": (cx, cy),
                    "area": area
                }
                
                objects.append(detected_object)
        
        return objects
    
    def visualize(self, frame, objects):
        """
        Visualize detected objects in the frame.
        Draws contours and centroids.
        """
        if frame is None or not objects:
            return frame
        
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Process each detected object
        for obj in objects:
            # Get color in BGR format for visualization
            color_name = obj["color"]
            if color_name == "red":
                color_bgr = (0, 0, 255)  # Red in BGR
            elif color_name == "green":
                color_bgr = (0, 255, 0)  # Green in BGR
            elif color_name == "blue":
                color_bgr = (255, 0, 0)  # Blue in BGR
            elif color_name == "yellow":
                color_bgr = (0, 255, 255)  # Yellow in BGR
            else:
                color_bgr = (255, 255, 255)  # White for unknown colors
            
            # Draw contour
            cv2.drawContours(vis_frame, [obj["contour"]], -1, color_bgr, 2)
            
            # Draw centroid
            cv2.circle(vis_frame, obj["centroid"], 5, color_bgr, -1)
            
            # Add label with color and area
            cx, cy = obj["centroid"]
            label = f"{color_name} ({obj['area']:.0f})"
            cv2.putText(
                vis_frame, 
                label, 
                (cx - 20, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color_bgr, 
                2
            )
        
        return vis_frame


# Test function
def test_color_detector():
    """Test the color detector with a camera feed."""
    from camera.capture import Camera
    
    camera = Camera()
    detector = ColorDetector()
    
    if not camera.open():
        print("Failed to open camera. Exiting...")
        return
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret:
                print("Failed to read frame. Exiting...")
                break
            
            # Preprocess the frame
            processed = camera.preprocess_frame(frame)
            
            # Detect objects by color
            objects = detector.find_objects(processed['hsv'])
            
            # Visualize the results
            vis_frame = detector.visualize(frame, objects)
            
            # Display the results
            cv2.imshow("Color Detection", vis_frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_color_detector() 