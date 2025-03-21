"""
Configuration file for the Robotic Arm Vision-Based Object Sorting system.
Edit these parameters to match your hardware setup and requirements.
"""

# Camera settings
CAMERA_ID = 0  # Camera device ID (usually 0 for built-in webcam)
CAMERA_WIDTH = 640  # Camera resolution width
CAMERA_HEIGHT = 480  # Camera resolution height
FRAME_RATE = 30  # Camera frame rate

# Depth camera settings
DEPTH_CAMERA_TYPE = "simulation"  # Options: "realsense", "azure_kinect", "stereo", "simulation"
DEPTH_CAMERA_ENABLED = False  # Set to True to enable depth sensing
DEPTH_MIN_DISTANCE = 100  # Minimum distance in mm to consider valid
DEPTH_MAX_DISTANCE = 2000  # Maximum distance in mm to consider valid
POINT_CLOUD_DENSITY = 2  # Downsampling factor for point cloud (higher = less dense)

# Robot arm settings
ARM_PORT = "COM3"  # Serial port for arm connection (change according to your setup)
ARM_BAUDRATE = 9600  # Serial baud rate
ARM_TIMEOUT = 1.0  # Serial timeout in seconds

# Path planning settings
PATH_PLANNING_ENABLED = False  # Set to True to enable advanced path planning
PATH_PLANNING_METHOD = "rrt"  # Options: "rrt", "rrt_star", "prm"
OBSTACLE_AVOIDANCE_ENABLED = True  # Set to True to enable obstacle avoidance
OBSTACLE_MARGIN = 20  # Margin around obstacles in mm
MAX_PLANNING_ATTEMPTS = 5  # Maximum number of planning attempts before fallback
SMOOTHING_ENABLED = True  # Apply path smoothing to trajectories
TRAJECTORY_POINTS = 10  # Number of points to generate in trajectory

# Workspace dimensions (in mm)
WORKSPACE_WIDTH = 500
WORKSPACE_HEIGHT = 400
WORKSPACE_DEPTH = 300

# Bin positions (x, y, z) in mm from arm base
BINS = {
    "red": (100, 100, 50),
    "green": (100, 200, 50),
    "blue": (200, 100, 50),
    "yellow": (200, 200, 50),
    "circle": (300, 100, 50),
    "rectangle": (300, 200, 50),
    "triangle": (400, 100, 50),
    "unknown": (400, 200, 50)
}

# HSV color thresholds
COLOR_RANGES = {
    "red": {
        "lower": (0, 120, 70),
        "upper": (10, 255, 255),
        "second_lower": (170, 120, 70),  # Red wraps around in HSV
        "second_upper": (180, 255, 255)
    },
    "green": {
        "lower": (35, 100, 70),
        "upper": (85, 255, 255)
    },
    "blue": {
        "lower": (100, 150, 70),
        "upper": (130, 255, 255)
    },
    "yellow": {
        "lower": (20, 100, 100),
        "upper": (35, 255, 255)
    }
}

# Shape detection parameters
SHAPE_PARAMS = {
    "circle": {"min_radius": 10, "max_radius": 100},
    "rectangle": {"min_aspect_ratio": 0.8, "max_aspect_ratio": 1.2},
    "triangle": {"approx_vertices": 3}
}

# Deep learning model settings
MODEL_PATH = "models/object_classifier.pth"
MODEL_CLASSES = ["screw", "nut", "bolt", "washer"]
MODEL_INPUT_SIZE = (224, 224)  # Input size for neural network

# Neural network training parameters
TRAINING_ENABLED = False  # Set to True to enable model training
TRAINING_DATA_PATH = "training_data"  # Directory with training data
VALIDATION_SPLIT = 0.2  # Fraction of data to use for validation
BATCH_SIZE = 32  # Training batch size
NUM_EPOCHS = 50  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate for optimizer
SAVE_CHECKPOINTS = True  # Save checkpoints during training
DATA_AUGMENTATION = True  # Apply data augmentation to training data
MODEL_ARCHITECTURE = "mobilenet_v2"  # Base model architecture

# Multi-camera settings
MULTI_CAMERA_ENABLED = False  # Set to True to use multiple cameras
CAMERA_POSITIONS = [
    {"id": 0, "name": "main", "position": (0, 0, 0), "rotation": (0, 0, 0)},
    {"id": 1, "name": "side", "position": (300, 0, 0), "rotation": (0, 90, 0)},
    {"id": 2, "name": "top", "position": (0, 0, 300), "rotation": (90, 0, 0)}
]
FUSION_METHOD = "weighted"  # Method to fuse multiple camera detections

# GUI settings
GUI_ENABLED = True  # Set to True to use the GUI interface
GUI_THEME = "dark"  # GUI theme (dark or light)
GUI_SIZE = (1280, 720)  # GUI window size
UI_UPDATE_INTERVAL = 50  # UI update interval in milliseconds
ENABLE_VISUALIZATIONS = True  # Enable advanced visualizations

# Image processing parameters
BLUR_KERNEL_SIZE = (5, 5)
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150
CONTOUR_MIN_AREA = 100

# Calibration settings
CALIBRATION_GRID_SIZE = (9, 6)  # Number of inner corners in calibration checkerboard
CALIBRATION_SQUARE_SIZE = 25.0  # Size of calibration squares in mm
CALIBRATION_SAMPLES = 10  # Number of calibration samples to capture

# Debug settings
DEBUG_MODE = True  # Enable debug visualization
DEBUG_WINDOW_SIZE = (1280, 720)  # Size of debug window 