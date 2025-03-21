"""
Shape-based object detection module.
Detects objects based on their geometric shapes.
"""
import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ShapeDetector:
    """Detects objects based on their geometric shapes."""
    
    def __init__(self, shape_params=None):
        """
        Initialize with shape detection parameters.
        If none provided, uses the defaults from config.
        """
        self.shape_params = shape_params if shape_params else config.SHAPE_PARAMS
    
    def detect_shape(self, contour):
        """
        Detect the shape of a contour.
        Returns the shape name (circle, rectangle, triangle, or unknown).
        """
        # Check if the contour is valid
        if contour is None or len(contour) < 3:
            return "unknown"
        
        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Approximate the contour to a simpler polygon
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        # Get the number of vertices in the approximated contour
        num_vertices = len(approx)
        
        # Check for circle using circularity
        # Circularity = 4*pi*area/perimeter^2 (1.0 for perfect circle)
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Detect circle
        if circularity > 0.8:
            # Additional check with min/max radius
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if (radius >= self.shape_params["circle"]["min_radius"] and 
                radius <= self.shape_params["circle"]["max_radius"]):
                return "circle"
        
        # Detect rectangle (4 vertices or close to it)
        elif num_vertices == 4 or num_vertices == 5:
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Check if aspect ratio is within range for rectangle
            if (aspect_ratio >= self.shape_params["rectangle"]["min_aspect_ratio"] and 
                aspect_ratio <= self.shape_params["rectangle"]["max_aspect_ratio"]):
                return "rectangle"
        
        # Detect triangle (3 vertices)
        elif num_vertices == 3:
            return "triangle"
        
        # If none of the above, it's an unknown shape
        return "unknown"
    
    def find_objects(self, edges_frame, min_area=None):
        """
        Detect objects and their shapes.
        Returns a list of detected objects with shape, contour, and centroid.
        """
        if edges_frame is None:
            return []
        
        # Use the minimum area from config if not specified
        if min_area is None:
            min_area = config.CONTOUR_MIN_AREA
        
        # Find contours in the edges image
        contours, _ = cv2.findContours(
            edges_frame, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        objects = []
        
        # Process each contour
        for contour in contours:
            # Filter out small contours
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Detect the shape
            shape_name = self.detect_shape(contour)
            
            # Calculate center of the contour (centroid)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # Fallback if moments method fails
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2
            
            # Create object with shape, contour and centroid
            detected_object = {
                "shape": shape_name,
                "contour": contour,
                "centroid": (cx, cy),
                "area": area
            }
            
            objects.append(detected_object)
        
        return objects
    
    def visualize(self, frame, objects):
        """
        Visualize detected shapes in the frame.
        Draws contours and centroids.
        """
        if frame is None or not objects:
            return frame
        
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Process each detected object
        for obj in objects:
            # Get color based on shape
            shape_name = obj["shape"]
            if shape_name == "circle":
                color_bgr = (0, 255, 0)  # Green for circles
            elif shape_name == "rectangle":
                color_bgr = (0, 0, 255)  # Red for rectangles
            elif shape_name == "triangle":
                color_bgr = (255, 0, 0)  # Blue for triangles
            else:
                color_bgr = (255, 255, 255)  # White for unknown shapes
            
            # Draw contour
            cv2.drawContours(vis_frame, [obj["contour"]], -1, color_bgr, 2)
            
            # Draw centroid
            cv2.circle(vis_frame, obj["centroid"], 5, color_bgr, -1)
            
            # Add label with shape and area
            cx, cy = obj["centroid"]
            label = f"{shape_name} ({obj['area']:.0f})"
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
def test_shape_detector():
    """Test the shape detector with a camera feed."""
    from camera.capture import Camera
    
    camera = Camera()
    detector = ShapeDetector()
    
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
            
            # Detect objects by shape
            objects = detector.find_objects(processed['edges'])
            
            # Visualize the results
            vis_frame = detector.visualize(frame, objects)
            
            # Display the original and edges frames for reference
            cv2.imshow("Original", frame)
            cv2.imshow("Edges", processed['edges'])
            
            # Display the results
            cv2.imshow("Shape Detection", vis_frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_shape_detector() 