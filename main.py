"""
Vision-based Robotic Arm Object Sorting System
Main application entry point

This module combines the camera, object detection, and arm control
modules to create a complete object sorting system.
"""
import cv2
import numpy as np
import time
import argparse
import os
import sys

# Import custom modules
import config
from camera.capture import Camera
from object_detection.color_detector import ColorDetector
from object_detection.shape_detector import ShapeDetector
from object_detection.dl_detector import DeepLearningDetector
from arm_control.controller import ArmController, SimulatedArmController
from calibration.calibrate import CameraCalibrator
from utils.visualization import (
    create_debug_frame, 
    add_fps_counter, 
    add_timestamp,
    add_detection_info
)


class ObjectSortingSystem:
    """Main class for the vision-based object sorting system."""
    
    def __init__(self, use_simulated_arm=True, detection_method="color"):
        """
        Initialize the object sorting system.
        
        Args:
            use_simulated_arm: Whether to use a simulated arm instead of real hardware
            detection_method: Detection method to use ("color", "shape", "dl", or "all")
        """
        self.detection_method = detection_method
        self.use_simulated_arm = use_simulated_arm
        
        # Initialize components
        self.camera = None
        self.color_detector = None
        self.shape_detector = None
        self.dl_detector = None
        self.arm_controller = None
        self.calibrator = None
        
        # Frame processing variables
        self.current_frame = None
        self.processed_frames = {}
        self.detected_objects = []
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        self.processing_active = False
        
        # Calibration status
        self.is_calibrated = False
    
    def initialize(self):
        """Initialize all system components."""
        print("\nInitializing Object Sorting System")
        print("==================================")
        
        # Initialize camera
        print("\n1. Initializing camera...")
        self.camera = Camera()
        if not self.camera.open():
            print("Failed to initialize camera. Exiting...")
            return False
        
        # Initialize detectors
        print("\n2. Initializing object detectors...")
        
        if self.detection_method in ["color", "all"]:
            print("- Initializing color detector")
            self.color_detector = ColorDetector()
        
        if self.detection_method in ["shape", "all"]:
            print("- Initializing shape detector")
            self.shape_detector = ShapeDetector()
        
        if self.detection_method in ["dl", "all"]:
            print("- Initializing deep learning detector")
            self.dl_detector = DeepLearningDetector()
            
            # Create models directory if it doesn't exist
            models_dir = os.path.dirname(config.MODEL_PATH)
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
        
        # Initialize calibration
        print("\n3. Loading camera calibration...")
        self.calibrator = CameraCalibrator()
        self.is_calibrated = self.calibrator.load_calibration()
        
        if not self.is_calibrated:
            print("No calibration found. System will use uncalibrated coordinates.")
            print("Run calibration with: python calibration/calibrate.py")
        
        # Initialize arm controller
        print("\n4. Initializing robotic arm...")
        if self.use_simulated_arm:
            print("Using simulated arm controller")
            self.arm_controller = SimulatedArmController()
        else:
            print(f"Connecting to arm on port {config.ARM_PORT}")
            self.arm_controller = ArmController()
        
        if not self.arm_controller.connect():
            print("Failed to connect to arm. Exiting...")
            return False
        
        # Move arm to home position
        print("Moving arm to home position...")
        self.arm_controller.home()
        
        print("\nSystem initialization complete!")
        return True
    
    def shutdown(self):
        """Shutdown all system components."""
        print("\nShutting down the system...")
        
        # Close the camera
        if self.camera is not None:
            self.camera.close()
        
        # Disconnect the arm
        if self.arm_controller is not None:
            self.arm_controller.disconnect()
        
        # Destroy all OpenCV windows
        cv2.destroyAllWindows()
        
        print("System shutdown complete.")
    
    def process_frame(self):
        """
        Process a single frame from the camera.
        Returns True if successful, False otherwise.
        """
        # Read a frame from the camera
        ret, self.current_frame = self.camera.read()
        if not ret:
            print("Failed to read frame from camera")
            return False
        
        # Preprocess the frame
        self.processed_frames = self.camera.preprocess_frame(self.current_frame)
        
        # Undistort the frame if calibration is available
        if self.is_calibrated:
            self.processed_frames['undistorted'] = self.calibrator.undistort_image(
                self.current_frame
            )
        
        # Reset detected objects
        self.detected_objects = []
        
        # Detect objects based on the selected method
        if self.detection_method in ["color", "all"] and self.color_detector is not None:
            color_objects = self.color_detector.find_objects(self.processed_frames['hsv'])
            self.detected_objects.extend(color_objects)
            
            # Create visualization
            self.processed_frames['color_detection'] = self.color_detector.visualize(
                self.current_frame.copy(), 
                color_objects
            )
        
        if self.detection_method in ["shape", "all"] and self.shape_detector is not None:
            shape_objects = self.shape_detector.find_objects(self.processed_frames['edges'])
            self.detected_objects.extend(shape_objects)
            
            # Create visualization
            self.processed_frames['shape_detection'] = self.shape_detector.visualize(
                self.current_frame.copy(), 
                shape_objects
            )
        
        if self.detection_method in ["dl", "all"] and self.dl_detector is not None:
            dl_objects = self.dl_detector.find_objects(self.current_frame)
            self.detected_objects.extend(dl_objects)
            
            # Create visualization
            self.processed_frames['dl_detection'] = self.dl_detector.visualize(
                self.current_frame.copy(), 
                dl_objects
            )
        
        # Calculate FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        return True
    
    def create_debug_view(self):
        """Create a debug view with all processed frames."""
        # Select frames to include in the debug view
        debug_frames = {
            "Original": self.current_frame
        }
        
        # Add method-specific frames
        if self.detection_method in ["color", "all"]:
            debug_frames["HSV"] = self.processed_frames.get('hsv')
            debug_frames["Color Detection"] = self.processed_frames.get('color_detection')
            
        if self.detection_method in ["shape", "all"]:
            debug_frames["Edges"] = self.processed_frames.get('edges')
            debug_frames["Shape Detection"] = self.processed_frames.get('shape_detection')
            
        if self.detection_method in ["dl", "all"]:
            debug_frames["DL Detection"] = self.processed_frames.get('dl_detection')
            
        if self.is_calibrated:
            debug_frames["Undistorted"] = self.processed_frames.get('undistorted')
        
        # Create the debug frame
        debug_frame = create_debug_frame(debug_frames)
        
        # Add FPS counter and timestamp
        debug_frame = add_fps_counter(debug_frame, self.fps)
        debug_frame = add_timestamp(debug_frame)
        
        # Add detection info
        debug_frame = add_detection_info(debug_frame, self.detected_objects)
        
        return debug_frame
    
    def sort_object(self, object_info):
        """
        Sort an object using the robotic arm.
        Returns True if the object was sorted successfully.
        """
        if self.arm_controller is None:
            print("Cannot sort: Arm controller not initialized")
            return False
        
        # If calibration is available, convert pixel coordinates to world coordinates
        if self.is_calibrated and "centroid" in object_info:
            pixel_x, pixel_y = object_info["centroid"]
            
            # Assume objects are at a fixed distance from the camera
            # In a real system, this would be determined by a depth sensor or stereo vision
            z_distance = 300  # mm
            
            # Convert to world coordinates
            world_coords = self.calibrator.pixel_to_world(pixel_x, pixel_y, z_distance)
            
            if world_coords:
                # Update centroid with world coordinates
                object_info["world_position"] = world_coords
        
        # Queue the sorting operation on the arm
        return self.arm_controller.sort_object(object_info)
    
    def run(self):
        """
        Run the main processing loop.
        This is the entry point for the application.
        """
        if not self.initialize():
            print("Failed to initialize system. Exiting...")
            self.shutdown()
            return
        
        print("\nStarting main processing loop...")
        print("Press 'q' to quit, 's' to sort current objects")
        
        # Start performance timer
        self.start_time = time.time()
        self.processing_active = True
        
        try:
            while self.processing_active:
                # Process the current frame
                if not self.process_frame():
                    print("Frame processing failed")
                    break
                
                # Create debug view
                debug_frame = self.create_debug_view()
                
                # Display the debug view
                cv2.imshow("Object Sorting System", debug_frame)
                
                # Process key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Quit the application
                    print("User requested to quit")
                    break
                    
                elif key == ord('s'):
                    # Sort detected objects
                    print(f"\nSorting {len(self.detected_objects)} objects...")
                    
                    for i, obj in enumerate(self.detected_objects):
                        print(f"Sorting object {i+1}...")
                        if self.sort_object(obj):
                            print(f"Object {i+1} sorted successfully")
                        else:
                            print(f"Failed to sort object {i+1}")
                    
                    print("Sorting complete")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        finally:
            # Clean up resources
            self.shutdown()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vision-based Robotic Arm Object Sorting System"
    )
    
    parser.add_argument(
        "--detection", 
        choices=["color", "shape", "dl", "all"],
        default="color",
        help="Object detection method to use"
    )
    
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="Use simulated arm instead of real hardware"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run the sorting system
    system = ObjectSortingSystem(
        use_simulated_arm=args.simulated,
        detection_method=args.detection
    )
    
    system.run() 