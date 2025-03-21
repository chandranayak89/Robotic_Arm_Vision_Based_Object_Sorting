"""
Camera calibration module.
Maps pixel coordinates to real-world coordinates for precise arm positioning.
"""
import cv2
import numpy as np
import sys
import os
import time

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from camera.capture import Camera


class CameraCalibrator:
    """
    Camera calibration class.
    Performs calibration for mapping from pixel to real-world coordinates.
    """
    
    def __init__(self, grid_size=None, square_size=None, num_samples=None):
        """
        Initialize calibration parameters.
        If not provided, uses the defaults from config.
        """
        self.grid_size = grid_size if grid_size else config.CALIBRATION_GRID_SIZE
        self.square_size = square_size if square_size else config.CALIBRATION_SQUARE_SIZE
        self.num_samples = num_samples if num_samples else config.CALIBRATION_SAMPLES
        
        # Calibration matrices
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
        # Flag to indicate if calibration is done
        self.is_calibrated = False
    
    def _create_object_points(self):
        """
        Create 3D points for the calibration grid.
        Returns a set of 3D points that represent the grid.
        """
        # Initialize object points
        objp = np.zeros((self.grid_size[0] * self.grid_size[1], 3), np.float32)
        
        # Fill with grid coordinates (x, y, 0)
        objp[:, :2] = np.mgrid[0:self.grid_size[0], 0:self.grid_size[1]].T.reshape(-1, 2)
        
        # Scale by square size (to mm)
        objp *= self.square_size
        
        return objp
    
    def calibrate_camera(self, camera=None):
        """
        Perform camera calibration using a checkerboard grid.
        
        Args:
            camera: Camera object, will create a new one if not provided
            
        Returns:
            True if calibration was successful
        """
        # Use existing or create new camera
        own_camera = camera is None
        if own_camera:
            camera = Camera()
            if not camera.open():
                print("Failed to open camera for calibration")
                return False
        
        try:
            print("\nCamera Calibration")
            print("=================")
            print(f"Looking for {self.grid_size[0]}x{self.grid_size[1]} grid")
            print(f"Square size: {self.square_size} mm")
            print(f"Collecting {self.num_samples} samples")
            print("\nPlace the calibration grid in view of the camera")
            print("Press 'c' to capture a sample, 'q' to quit, 'space' to skip current frame")
            
            # Prepare object points
            objp = self._create_object_points()
            
            # Arrays to store object points and image points
            objpoints = []  # 3D points in real world space
            imgpoints = []  # 2D points in image plane
            
            # Counter for saved samples
            sample_count = 0
            
            while sample_count < self.num_samples:
                # Capture frame
                ret, frame = camera.read()
                if not ret:
                    print("Failed to read frame")
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Create display frame
                display_frame = frame.copy()
                
                # Add instructions on the display frame
                instructions = f"Samples: {sample_count}/{self.num_samples}"
                cv2.putText(
                    display_frame, 
                    instructions, 
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Add keyboard instructions
                key_instructions = "c: capture, space: skip, q: quit"
                cv2.putText(
                    display_frame, 
                    key_instructions, 
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Look for chessboard corners
                ret_chess, corners = cv2.findChessboardCorners(
                    gray, 
                    self.grid_size, 
                    None
                )
                
                # If found, add object and image points
                if ret_chess:
                    # Draw the corners on the display frame
                    cv2.drawChessboardCorners(
                        display_frame, 
                        self.grid_size, 
                        corners, 
                        ret_chess
                    )
                    
                    # Add status message
                    status = "Grid detected - press 'c' to capture"
                    cv2.putText(
                        display_frame, 
                        status, 
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                else:
                    # Add status message
                    status = "No grid detected"
                    cv2.putText(
                        display_frame, 
                        status, 
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 0, 255), 
                        2
                    )
                
                # Display the resulting frame
                cv2.imshow('Calibration', display_frame)
                
                # Wait for key press
                key = cv2.waitKey(1) & 0xFF
                
                # Handle key press
                if key == ord('q'):
                    print("Calibration aborted by user")
                    break
                    
                elif key == ord('c') and ret_chess:
                    # Refine corner locations for better accuracy
                    corners2 = cv2.cornerSubPix(
                        gray, 
                        corners, 
                        (11, 11), 
                        (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    
                    # Add the points
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    
                    # Increment sample count
                    sample_count += 1
                    
                    print(f"Sample {sample_count}/{self.num_samples} captured")
                    
                    # Provide visual feedback
                    cv2.putText(
                        display_frame, 
                        "CAPTURED", 
                        (display_frame.shape[1]//2 - 80, display_frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, 
                        (0, 255, 0), 
                        3
                    )
                    cv2.imshow('Calibration', display_frame)
                    cv2.waitKey(500)  # Pause to show the capture confirmation
            
            # If we have enough samples, calibrate
            if sample_count >= 3:  # Need at least 3 samples for meaningful calibration
                print("\nCalculating calibration parameters...")
                
                # Perform camera calibration
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, 
                    imgpoints, 
                    gray.shape[::-1], 
                    None, 
                    None
                )
                
                if ret:
                    # Store calibration results
                    self.camera_matrix = mtx
                    self.dist_coeffs = dist
                    self.rvecs = rvecs
                    self.tvecs = tvecs
                    self.is_calibrated = True
                    
                    # Save calibration parameters
                    self._save_calibration()
                    
                    print("Calibration successful!")
                    print(f"Camera matrix:\n{mtx}")
                    print(f"Distortion coefficients:\n{dist}")
                    
                    # Calculate and print reprojection error
                    mean_error = self._calculate_reprojection_error(
                        objpoints, 
                        imgpoints, 
                        rvecs, 
                        tvecs, 
                        mtx, 
                        dist
                    )
                    print(f"Reprojection error: {mean_error}")
                    
                    return True
                else:
                    print("Calibration failed!")
                    return False
            else:
                print("\nNot enough samples for calibration!")
                return False
                
        finally:
            # Close the camera if we opened it
            if own_camera and camera is not None:
                camera.close()
            
            cv2.destroyAllWindows()
    
    def _save_calibration(self):
        """Save calibration parameters to file."""
        if not self.is_calibrated:
            print("Cannot save: No calibration data available")
            return False
        
        # Create calibration directory if it doesn't exist
        cal_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        
        # Save parameters
        np.save(os.path.join(cal_dir, 'camera_matrix.npy'), self.camera_matrix)
        np.save(os.path.join(cal_dir, 'dist_coeffs.npy'), self.dist_coeffs)
        
        print(f"Calibration saved to {cal_dir}")
        return True
    
    def load_calibration(self):
        """Load calibration parameters from file."""
        cal_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check if calibration files exist
        mtx_path = os.path.join(cal_dir, 'camera_matrix.npy')
        dist_path = os.path.join(cal_dir, 'dist_coeffs.npy')
        
        if os.path.exists(mtx_path) and os.path.exists(dist_path):
            try:
                self.camera_matrix = np.load(mtx_path)
                self.dist_coeffs = np.load(dist_path)
                self.is_calibrated = True
                
                print("Calibration loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading calibration: {str(e)}")
                return False
        else:
            print("Calibration files not found")
            return False
    
    def _calculate_reprojection_error(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        """
        Calculate the reprojection error from calibration.
        Lower error means better calibration.
        """
        mean_error = 0
        for i in range(len(objpoints)):
            # Project 3D points to 2D
            imgpoints2, _ = cv2.projectPoints(
                objpoints[i], 
                rvecs[i], 
                tvecs[i], 
                mtx, 
                dist
            )
            
            # Calculate error
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
            
        return mean_error / len(objpoints)
    
    def undistort_image(self, image):
        """
        Remove lens distortion from an image.
        Returns the undistorted image.
        """
        if not self.is_calibrated:
            print("Cannot undistort: No calibration data available")
            return image
        
        return cv2.undistort(
            image, 
            self.camera_matrix, 
            self.dist_coeffs, 
            None, 
            self.camera_matrix
        )
    
    def pixel_to_world(self, pixel_x, pixel_y, z_distance=0):
        """
        Convert pixel coordinates to world coordinates.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates in the image
            z_distance: Distance from camera to object in mm
            
        Returns:
            (x, y, z) world coordinates in mm
        """
        if not self.is_calibrated:
            print("Cannot convert: No calibration data available")
            return None
        
        # Convert pixel to normalized coordinates
        u = pixel_x
        v = pixel_y
        
        # Get camera matrix elements
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Convert to normalized image coordinates
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        
        # Calculate real-world coordinates (simplified pinhole model)
        x = x_norm * z_distance
        y = y_norm * z_distance
        z = z_distance
        
        return (x, y, z)
    
    def world_to_pixel(self, world_x, world_y, world_z):
        """
        Convert world coordinates to pixel coordinates.
        
        Args:
            world_x, world_y, world_z: World coordinates in mm
            
        Returns:
            (pixel_x, pixel_y) coordinates in the image
        """
        if not self.is_calibrated:
            print("Cannot convert: No calibration data available")
            return None
        
        # Get camera matrix elements
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Check for division by zero
        if world_z == 0:
            print("Warning: Z coordinate is zero, cannot project to image plane")
            return None
        
        # Calculate pixel coordinates
        pixel_x = int((world_x / world_z) * fx + cx)
        pixel_y = int((world_y / world_z) * fy + cy)
        
        return (pixel_x, pixel_y)


# Test function
def test_calibration():
    """Run the camera calibration process."""
    calibrator = CameraCalibrator()
    
    # Check if calibration already exists
    if calibrator.load_calibration():
        print("Found existing calibration. Running test with undistortion...")
        
        # Test with live camera feed
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
                
                # Undistort the frame
                undistorted = calibrator.undistort_image(frame)
                
                # Display the original and undistorted frames
                cv2.imshow('Original', frame)
                cv2.imshow('Undistorted', undistorted)
                
                # Test coordinate conversion
                height, width = frame.shape[:2]
                center_pixel = (width // 2, height // 2)
                
                # Convert center pixel to world coordinates (assuming 500mm distance)
                world_coords = calibrator.pixel_to_world(
                    center_pixel[0], 
                    center_pixel[1], 
                    500
                )
                
                # Convert back to pixel coordinates
                if world_coords:
                    pixel_coords = calibrator.world_to_pixel(
                        world_coords[0], 
                        world_coords[1], 
                        world_coords[2]
                    )
                    
                    print(f"Center pixel: {center_pixel}")
                    print(f"World coordinates: {world_coords}")
                    print(f"Reprojected pixel: {pixel_coords}")
                    
                    # Draw markers on the undistorted image
                    cv2.circle(undistorted, center_pixel, 5, (0, 255, 0), -1)
                    if pixel_coords:
                        cv2.circle(undistorted, pixel_coords, 5, (0, 0, 255), -1)
                    
                    cv2.imshow('Undistorted with markers', undistorted)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            camera.close()
            cv2.destroyAllWindows()
    else:
        # Run new calibration
        print("No calibration found. Running calibration process...")
        calibrator.calibrate_camera()


if __name__ == "__main__":
    test_calibration() 