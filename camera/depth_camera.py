"""
Depth camera module for 3D object positioning.
Supports various depth sensing devices (Intel RealSense, stereo cameras, etc.)
"""
import cv2
import numpy as np
import time
import sys
import os
from enum import Enum

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Optional imports for specific depth cameras
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Intel RealSense SDK not found. RealSense cameras will not be available.")

try:
    from pyk4a import PyK4A, Config as K4AConfig
    AZURE_KINECT_AVAILABLE = True
except ImportError:
    AZURE_KINECT_AVAILABLE = False
    print("Azure Kinect SDK not found. Kinect cameras will not be available.")


class DepthCameraType(Enum):
    """Enumeration of supported depth camera types."""
    REALSENSE = "realsense"
    AZURE_KINECT = "azure_kinect"
    STEREO = "stereo"
    SIMULATION = "simulation"


class DepthCamera:
    """Depth camera interface for acquiring RGB and depth frames."""
    
    def __init__(self, camera_type=None, device_id=0, config=None):
        """
        Initialize depth camera with specified parameters.
        
        Args:
            camera_type: Type of depth camera to use (from DepthCameraType enum)
            device_id: Device ID or index
            config: Additional configuration parameters
        """
        # Use config defaults if not specified
        self.camera_type = camera_type or config.DEPTH_CAMERA_TYPE
        self.device_id = device_id
        self.config = config or {}
        
        # Initialize device-specific members
        self.pipeline = None
        self.profile = None
        self.align = None
        self.device = None
        
        # Camera parameters
        self.width = self.config.get("width", config.CAMERA_WIDTH)
        self.height = self.config.get("height", config.CAMERA_HEIGHT)
        self.fps = self.config.get("fps", config.FRAME_RATE)
        
        # Depth parameters
        self.depth_scale = 0.001  # Default: 1mm per unit
        
        # Camera state
        self.is_open = False
        
        # Initialize stereo cameras if using stereo type
        if self.camera_type == DepthCameraType.STEREO:
            self.left_camera = None
            self.right_camera = None
            self.stereo_matcher = None
        
        # For simulation mode
        if self.camera_type == DepthCameraType.SIMULATION:
            self.color_image = None
            self.depth_image = None
            self.simulation_frame_count = 0
    
    def open(self):
        """
        Open and configure the depth camera.
        Returns True if successful, False otherwise.
        """
        try:
            if self.camera_type == DepthCameraType.REALSENSE:
                return self._open_realsense()
            elif self.camera_type == DepthCameraType.AZURE_KINECT:
                return self._open_azure_kinect()
            elif self.camera_type == DepthCameraType.STEREO:
                return self._open_stereo()
            elif self.camera_type == DepthCameraType.SIMULATION:
                return self._open_simulation()
            else:
                print(f"Unsupported camera type: {self.camera_type}")
                return False
        except Exception as e:
            print(f"Error opening depth camera: {str(e)}")
            self.close()
            return False
    
    def _open_realsense(self):
        """Open and configure an Intel RealSense camera."""
        if not REALSENSE_AVAILABLE:
            print("Intel RealSense SDK not available")
            return False
        
        try:
            # Create pipeline and configure
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable streams
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start the pipeline
            self.profile = self.pipeline.start(config)
            
            # Get depth scale for converting depth units to meters
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Create alignment object
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            # Wait for stable frames
            for _ in range(5):
                self.pipeline.wait_for_frames()
            
            self.is_open = True
            print(f"RealSense camera opened: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"Error opening RealSense camera: {str(e)}")
            return False
    
    def _open_azure_kinect(self):
        """Open and configure a Microsoft Azure Kinect camera."""
        if not AZURE_KINECT_AVAILABLE:
            print("Azure Kinect SDK not available")
            return False
        
        try:
            # Configure and start the device
            k4a_config = K4AConfig()
            k4a_config.color_resolution = self._get_k4a_resolution()
            k4a_config.depth_mode = self._get_k4a_depth_mode()
            k4a_config.camera_fps = self._get_k4a_fps()
            
            self.device = PyK4A(k4a_config)
            self.device.start()
            
            # Wait for stable frames
            for _ in range(5):
                self.device.get_capture()
            
            self.is_open = True
            print(f"Azure Kinect camera opened: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"Error opening Azure Kinect camera: {str(e)}")
            return False
    
    def _get_k4a_resolution(self):
        """Convert resolution settings to K4A resolution constants."""
        from pyk4a import ColorResolution
        # Map common resolutions to K4A constants
        if self.width == 1280 and self.height == 720:
            return ColorResolution.RES_720P
        elif self.width == 1920 and self.height == 1080:
            return ColorResolution.RES_1080P
        elif self.width == 2560 and self.height == 1440:
            return ColorResolution.RES_1440P
        elif self.width == 3840 and self.height == 2160:
            return ColorResolution.RES_2160P
        else:
            # Default to 720p
            self.width = 1280
            self.height = 720
            return ColorResolution.RES_720P
    
    def _get_k4a_depth_mode(self):
        """Convert depth settings to K4A depth mode constants."""
        from pyk4a import DepthMode
        # Choose appropriate depth mode based on configuration
        depth_mode = self.config.get("depth_mode", "NFOV_UNBINNED")
        if depth_mode == "NFOV_UNBINNED":
            return DepthMode.NFOV_UNBINNED
        elif depth_mode == "NFOV_2X2BINNED":
            return DepthMode.NFOV_2X2BINNED
        elif depth_mode == "WFOV_2X2BINNED":
            return DepthMode.WFOV_2X2BINNED
        elif depth_mode == "WFOV_UNBINNED":
            return DepthMode.WFOV_UNBINNED
        else:
            return DepthMode.NFOV_UNBINNED
    
    def _get_k4a_fps(self):
        """Convert FPS settings to K4A FPS constants."""
        from pyk4a import FPS
        # Map FPS to K4A constants
        if self.fps == 5:
            return FPS.FPS_5
        elif self.fps == 15:
            return FPS.FPS_15
        elif self.fps == 30:
            return FPS.FPS_30
        else:
            # Default to 30fps
            self.fps = 30
            return FPS.FPS_30
    
    def _open_stereo(self):
        """Open and configure stereo cameras for depth calculation."""
        try:
            # Open left and right cameras
            self.left_camera = cv2.VideoCapture(self.device_id)
            self.right_camera = cv2.VideoCapture(self.device_id + 1)
            
            # Check if cameras opened successfully
            if not self.left_camera.isOpened() or not self.right_camera.isOpened():
                print("Failed to open stereo cameras")
                return False
            
            # Set camera properties
            for cam in [self.left_camera, self.right_camera]:
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cam.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Create stereo matcher
            stereo_method = self.config.get("stereo_method", "SGBM")
            if stereo_method == "SGBM":
                self.stereo_matcher = cv2.StereoSGBM_create(
                    minDisparity=0,
                    numDisparities=16*16,
                    blockSize=11,
                    P1=8 * 3 * 11**2,
                    P2=32 * 3 * 11**2,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32
                )
            else:  # BM method
                self.stereo_matcher = cv2.StereoBM_create(
                    numDisparities=16*16,
                    blockSize=11
                )
            
            # Read test frames to confirm setup
            ret_left, _ = self.left_camera.read()
            ret_right, _ = self.right_camera.read()
            
            if not ret_left or not ret_right:
                print("Failed to read frames from stereo cameras")
                return False
            
            # Set calibration parameters
            self.stereo_baseline = self.config.get("stereo_baseline", 0.12)  # 12cm between cameras
            self.stereo_focal_length = self.config.get("stereo_focal_length", 600)  # pixels
            self.depth_scale = self.stereo_baseline * self.stereo_focal_length
            
            self.is_open = True
            print(f"Stereo cameras opened: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"Error opening stereo cameras: {str(e)}")
            return False
    
    def _open_simulation(self):
        """Set up a simulation depth camera (for testing without hardware)."""
        try:
            # Create empty color and depth images
            self.color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.depth_image = np.zeros((self.height, self.width), dtype=np.uint16)
            
            # Draw some sample objects in the scene
            cv2.circle(self.color_image, (self.width//3, self.height//2), 50, (0, 0, 255), -1)  # Red circle
            cv2.rectangle(self.color_image, (2*self.width//3-40, self.height//2-40), 
                         (2*self.width//3+40, self.height//2+40), (0, 255, 0), -1)  # Green square
            
            # Create corresponding depth values (darker = closer)
            cv2.circle(self.depth_image, (self.width//3, self.height//2), 50, 5000, -1)  # Circle at 0.5m
            cv2.rectangle(self.depth_image, (2*self.width//3-40, self.height//2-40), 
                         (2*self.width//3+40, self.height//2+40), 8000, -1)  # Square at 0.8m
            
            # Set background to far distance
            mask = self.depth_image == 0
            self.depth_image[mask] = 10000  # Background at 1m
            
            self.is_open = True
            print("Simulation depth camera initialized")
            return True
            
        except Exception as e:
            print(f"Error initializing simulation camera: {str(e)}")
            return False
    
    def read(self):
        """
        Read a frame from the depth camera.
        Returns (success, color_frame, depth_frame, aligned_depth)
        """
        if not self.is_open:
            print("Cannot read: Depth camera is not open")
            return False, None, None, None
        
        try:
            if self.camera_type == DepthCameraType.REALSENSE:
                return self._read_realsense()
            elif self.camera_type == DepthCameraType.AZURE_KINECT:
                return self._read_azure_kinect()
            elif self.camera_type == DepthCameraType.STEREO:
                return self._read_stereo()
            elif self.camera_type == DepthCameraType.SIMULATION:
                return self._read_simulation()
            else:
                return False, None, None, None
        except Exception as e:
            print(f"Error reading from depth camera: {str(e)}")
            return False, None, None, None
    
    def _read_realsense(self):
        """Read frames from Intel RealSense camera."""
        # Wait for a coherent pair of frames
        frames = self.pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Create depth colormap for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        return True, color_image, depth_image, depth_colormap
    
    def _read_azure_kinect(self):
        """Read frames from Azure Kinect camera."""
        # Get capture
        capture = self.device.get_capture()
        
        # Get color and depth images
        color_image = capture.color
        depth_image = capture.depth
        
        # Convert depth to aligned color space
        transformed_depth = capture.transformed_depth
        
        # Create depth colormap for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(transformed_depth, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        return True, color_image, depth_image, depth_colormap
    
    def _read_stereo(self):
        """Read and process frames from stereo cameras."""
        # Read frames from both cameras
        ret_left, left_frame = self.left_camera.read()
        ret_right, right_frame = self.right_camera.read()
        
        if not ret_left or not ret_right:
            return False, None, None, None
        
        # Convert to grayscale for disparity calculation
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity map
        disparity = self.stereo_matcher.compute(left_gray, right_gray)
        
        # Convert disparity to depth
        # depth = baseline * focal_length / disparity
        valid_disparity = disparity > 0
        depth_image = np.zeros_like(disparity, dtype=np.uint16)
        depth_image[valid_disparity] = (self.depth_scale / disparity[valid_disparity]).astype(np.uint16)
        
        # Create depth colormap for visualization
        norm_disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(norm_disparity.astype(np.uint8), cv2.COLORMAP_JET)
        
        return True, left_frame, depth_image, depth_colormap
    
    def _read_simulation(self):
        """Generate simulated depth camera frames."""
        # Create a copy of the static images
        color_image = self.color_image.copy()
        depth_image = self.depth_image.copy()
        
        # Animate the objects slightly to simulate movement
        self.simulation_frame_count += 1
        offset_x = int(10 * np.sin(self.simulation_frame_count / 10.0))
        offset_y = int(5 * np.cos(self.simulation_frame_count / 15.0))
        
        # Create affine transformation matrix
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        
        # Apply the transformation
        color_image = cv2.warpAffine(self.color_image, M, (self.width, self.height))
        depth_image = cv2.warpAffine(self.depth_image, M, (self.width, self.height))
        
        # Create depth colormap for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.01), 
            cv2.COLORMAP_JET
        )
        
        # Add timestamp
        cv2.putText(
            color_image,
            f"Simulation Frame #{self.simulation_frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Sleep to simulate camera frame rate
        time.sleep(1.0 / self.fps)
        
        return True, color_image, depth_image, depth_colormap
    
    def close(self):
        """Close the depth camera connection."""
        try:
            if self.camera_type == DepthCameraType.REALSENSE and self.pipeline:
                self.pipeline.stop()
            elif self.camera_type == DepthCameraType.AZURE_KINECT and self.device:
                self.device.stop()
            elif self.camera_type == DepthCameraType.STEREO:
                if self.left_camera:
                    self.left_camera.release()
                if self.right_camera:
                    self.right_camera.release()
            
            self.is_open = False
            print("Depth camera closed")
            
        except Exception as e:
            print(f"Error closing depth camera: {str(e)}")
    
    def get_point_cloud(self, color_image, depth_image):
        """
        Generate a colored point cloud from color and depth images.
        
        Args:
            color_image: RGB image (H x W x 3)
            depth_image: Depth image in mm (H x W)
            
        Returns:
            points: (N x 3) array of 3D points
            colors: (N x 3) array of corresponding colors
        """
        if color_image is None or depth_image is None:
            return None, None
        
        # Get image dimensions
        height, width = depth_image.shape
        
        # Create meshgrid of pixel coordinates
        pixel_x, pixel_y = np.meshgrid(np.arange(width), np.arange(height))
        pixel_x = pixel_x.flatten()
        pixel_y = pixel_y.flatten()
        
        # Get depth values
        z = depth_image.flatten() * self.depth_scale
        
        # Filter out invalid depth values
        valid_depth = z > 0
        pixel_x = pixel_x[valid_depth]
        pixel_y = pixel_y[valid_depth]
        z = z[valid_depth]
        
        # Use camera intrinsics to compute 3D coordinates
        # For a calibrated camera, use the actual intrinsics
        # For simplicity, we use approximate values
        fx = self.width / 2  # approx. focal length
        fy = self.height / 2
        cx = self.width / 2  # principal point
        cy = self.height / 2
        
        x = (pixel_x - cx) * z / fx
        y = (pixel_y - cy) * z / fy
        
        # Combine into points array
        points = np.column_stack((x, y, z))
        
        # Get corresponding colors
        colors = color_image.reshape(-1, 3)[valid_depth]
        
        return points, colors
    
    def get_depth_at_point(self, depth_image, x, y, window_size=5):
        """
        Get the depth value at a specific point, averaged over a small window.
        
        Args:
            depth_image: Depth image
            x, y: Pixel coordinates
            window_size: Size of window to average over
            
        Returns:
            Depth value in meters
        """
        # Check if coordinates are within image bounds
        height, width = depth_image.shape
        if x < 0 or x >= width or y < 0 or y >= height:
            return 0.0
        
        # Define region to average over
        x_min = max(0, x - window_size // 2)
        x_max = min(width, x + window_size // 2 + 1)
        y_min = max(0, y - window_size // 2)
        y_max = min(height, y + window_size // 2 + 1)
        
        # Extract region
        region = depth_image[y_min:y_max, x_min:x_max]
        
        # Filter out zero values (invalid measurements)
        valid_depths = region[region > 0]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Calculate median depth (more robust than mean)
        depth_mm = np.median(valid_depths)
        
        # Convert to meters
        depth_m = depth_mm * self.depth_scale
        
        return depth_m
    
    def get_object_dimensions(self, depth_image, contour, color_image=None):
        """
        Estimate physical dimensions of an object from its contour and depth.
        
        Args:
            depth_image: Depth image
            contour: Object contour points
            color_image: Optional color image for visualization
            
        Returns:
            Dictionary with estimated width, height, and depth in mm
        """
        if len(contour) < 5:
            return None
        
        # Get bounding rectangle of contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Get rotated rectangle (minimum area)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Get depth at contour centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        
        # Get depth at object center
        depth_m = self.get_depth_at_point(depth_image, centroid_x, centroid_y)
        
        if depth_m <= 0:
            return None
        
        # Calculate physical dimensions using depth
        # Conversion factor depends on camera intrinsics and depth
        # For a calibrated camera, use the actual intrinsics
        # For simplicity, we use approximate values
        fx = self.width / 2  # approx. focal length
        fy = self.height / 2
        
        # Convert pixel dimensions to physical dimensions
        physical_width = w * depth_m / fx * 1000  # in mm
        physical_height = h * depth_m / fy * 1000  # in mm
        
        # Estimate depth of object (thickness)
        # Find min and max depth within contour
        mask = np.zeros_like(depth_image)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        masked_depth = depth_image.copy()
        masked_depth[mask == 0] = 0
        
        # Filter out zeros
        valid_depths = masked_depth[masked_depth > 0]
        if len(valid_depths) > 0:
            min_depth = np.percentile(valid_depths, 5)  # 5th percentile
            max_depth = np.percentile(valid_depths, 95)  # 95th percentile
            physical_depth = (max_depth - min_depth) * self.depth_scale * 1000  # in mm
        else:
            physical_depth = 0
        
        # Visualize measurements if color image is provided
        if color_image is not None:
            vis_img = color_image.copy()
            cv2.drawContours(vis_img, [box], 0, (0, 255, 0), 2)
            cv2.circle(vis_img, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            
            # Add dimension text
            text_offset = 20
            cv2.putText(
                vis_img,
                f"W: {physical_width:.1f}mm, H: {physical_height:.1f}mm",
                (x, y - text_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            cv2.putText(
                vis_img,
                f"Depth: {physical_depth:.1f}mm",
                (x, y - text_offset * 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            cv2.putText(
                vis_img,
                f"Distance: {depth_m*1000:.0f}mm",
                (x, y - text_offset * 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            cv2.imshow("Object Dimensions", vis_img)
        
        return {
            "width": physical_width,
            "height": physical_height,
            "depth": physical_depth,
            "distance": depth_m * 1000  # in mm
        }


# Test function
def test_depth_camera():
    """Test depth camera functionality."""
    # First test if RealSense SDK is available
    if REALSENSE_AVAILABLE:
        camera_type = DepthCameraType.REALSENSE
    else:
        # Fall back to simulation mode
        camera_type = DepthCameraType.SIMULATION
    
    # Create and initialize the camera
    depth_cam = DepthCamera(camera_type=camera_type)
    
    if not depth_cam.open():
        print("Failed to open depth camera")
        return
    
    try:
        while True:
            # Read frames
            ret, color_image, depth_image, depth_colormap = depth_cam.read()
            
            if not ret:
                print("Failed to read frame")
                break
            
            # Display the images
            cv2.imshow('Color Image', color_image)
            cv2.imshow('Depth Image', depth_colormap)
            
            # Generate point cloud (for demonstration)
            points, colors = depth_cam.get_point_cloud(color_image, depth_image)
            
            if points is not None:
                print(f"Generated point cloud with {len(points)} points")
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        depth_cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_depth_camera() 