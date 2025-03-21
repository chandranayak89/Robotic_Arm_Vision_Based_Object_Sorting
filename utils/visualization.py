"""
Visualization utility functions.
Provides functions for creating debug displays and UI elements.
"""
import cv2
import numpy as np
import time


def create_debug_frame(frames_dict, layout=None, size=None):
    """
    Create a debug frame from multiple input frames.
    
    Args:
        frames_dict: Dictionary of named frames to include
        layout: Tuple of (rows, cols) for grid layout, auto-calculated if None
        size: Size of the output frame (width, height), calculated if None
        
    Returns:
        Combined debug frame
    """
    if not frames_dict:
        return None
    
    # Ensure all frames are color (3 channels)
    processed_frames = {}
    for name, frame in frames_dict.items():
        if frame is None:
            # Create an empty frame with text showing it's none
            height, width = 240, 320  # Default size
            empty_frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                empty_frame, 
                f"{name} - None", 
                (20, height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            processed_frames[name] = empty_frame
        elif len(frame.shape) == 2:
            # Convert grayscale to color
            processed_frames[name] = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:
            # Already color
            processed_frames[name] = frame.copy()
        else:
            # Unknown format, create empty frame
            height, width = 240, 320  # Default size
            empty_frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                empty_frame, 
                f"{name} - Invalid format", 
                (20, height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            processed_frames[name] = empty_frame
    
    # Add frame names
    for name, frame in processed_frames.items():
        cv2.putText(
            frame, 
            name, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
    
    # Calculate grid layout if not provided
    if layout is None:
        n = len(processed_frames)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        layout = (rows, cols)
    else:
        rows, cols = layout
    
    # Get the first frame size as reference
    first_frame = next(iter(processed_frames.values()))
    h, w = first_frame.shape[:2]
    
    # Resize frames to the same size if they differ
    for name, frame in processed_frames.items():
        if frame.shape[:2] != (h, w):
            processed_frames[name] = cv2.resize(frame, (w, h))
    
    # Calculate the output frame size
    if size is None:
        output_width = w * cols
        output_height = h * rows
        size = (output_width, output_height)
    else:
        output_width, output_height = size
        # Calculate the size of each cell in the grid
        cell_width = output_width // cols
        cell_height = output_height // rows
        # Resize all frames to fit
        for name, frame in processed_frames.items():
            processed_frames[name] = cv2.resize(frame, (cell_width, cell_height))
        # Update h, w for the loop below
        h, w = cell_height, cell_width
    
    # Create the output frame
    output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Add frames to the grid
    frames_list = list(processed_frames.items())
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(frames_list):
                name, frame = frames_list[idx]
                y_start = i * h
                y_end = y_start + h
                x_start = j * w
                x_end = x_start + w
                output_frame[y_start:y_end, x_start:x_end] = frame
    
    return output_frame


def add_fps_counter(frame, fps, position=(20, 70)):
    """
    Add an FPS counter to a frame.
    
    Args:
        frame: The input frame
        fps: FPS value to display
        position: Position of the text
        
    Returns:
        Frame with FPS counter added
    """
    if frame is None:
        return None
    
    # Create a copy to avoid modifying the original
    result = frame.copy()
    
    # Add FPS text
    cv2.putText(
        result,
        f"FPS: {fps:.1f}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    return result


def add_timestamp(frame, position=(20, 110)):
    """
    Add a timestamp to a frame.
    
    Args:
        frame: The input frame
        position: Position of the text
        
    Returns:
        Frame with timestamp added
    """
    if frame is None:
        return None
    
    # Create a copy to avoid modifying the original
    result = frame.copy()
    
    # Get current time
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Add timestamp text
    cv2.putText(
        result,
        current_time,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    return result


def add_detection_info(frame, objects, position=(20, 150)):
    """
    Add detection info to a frame.
    
    Args:
        frame: The input frame
        objects: List of detected objects
        position: Starting position of the text
        
    Returns:
        Frame with detection info added
    """
    if frame is None:
        return None
    
    # Create a copy to avoid modifying the original
    result = frame.copy()
    
    # Add number of objects
    cv2.putText(
        result,
        f"Objects: {len(objects)}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    # List object properties (if any)
    y_offset = position[1]
    for i, obj in enumerate(objects[:5]):  # Limit to first 5 objects to avoid cluttering
        y_offset += 30
        
        # Create object description based on available properties
        description = f"Object {i+1}: "
        
        if "color" in obj:
            description += f"Color: {obj['color']}, "
            
        if "shape" in obj:
            description += f"Shape: {obj['shape']}, "
            
        if "class" in obj:
            description += f"Class: {obj['class']}, "
            
        if "area" in obj:
            description += f"Area: {obj['area']:.0f}"
            
        # Add each object's description
        cv2.putText(
            result,
            description,
            (position[0], y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    
    # Indicate if more objects were detected but not shown
    if len(objects) > 5:
        y_offset += 30
        cv2.putText(
            result,
            f"... and {len(objects) - 5} more",
            (position[0], y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    
    return result


# Test function to demonstrate visualization utilities
def test_visualization():
    """Test visualization functionality with sample frames."""
    # Create sample frames
    img1 = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (270, 190), (0, 0, 255), -1)
    
    img2 = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(img2, (160, 120), 100, (0, 255, 0), -1)
    
    img3 = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.line(img3, (0, 0), (320, 240), (255, 0, 0), 5)
    
    img4 = np.zeros((240, 320), dtype=np.uint8)  # Grayscale
    cv2.putText(img4, "Grayscale", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    # Create a debug frame
    frames_dict = {
        "Rectangle": img1,
        "Circle": img2, 
        "Line": img3,
        "Text": img4
    }
    
    debug_frame = create_debug_frame(frames_dict)
    
    # Add FPS and timestamp
    debug_frame = add_fps_counter(debug_frame, 30.5)
    debug_frame = add_timestamp(debug_frame)
    
    # Add detection info
    sample_objects = [
        {"color": "red", "shape": "circle", "area": 1256.0},
        {"color": "blue", "shape": "rectangle", "area": 4000.0},
        {"class": "screw", "confidence": 0.98}
    ]
    debug_frame = add_detection_info(debug_frame, sample_objects)
    
    # Display the result
    cv2.imshow("Debug Visualization", debug_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_visualization() 