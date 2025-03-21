"""
Deep learning-based object detection module.
Uses pre-trained models to classify objects.
"""
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DeepLearningDetector:
    """Detects objects using deep learning models."""
    
    def __init__(self, model_path=None, classes=None, input_size=None):
        """
        Initialize with model parameters.
        If not provided, uses the defaults from config.
        """
        self.model_path = model_path if model_path else config.MODEL_PATH
        self.classes = classes if classes else config.MODEL_CLASSES
        self.input_size = input_size if input_size else config.MODEL_INPUT_SIZE
        
        # Initialize model as None (will be loaded on demand)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """Get the transforms for preprocessing images for the model."""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_model(self):
        """
        Load the pre-trained model.
        Returns True if successful, False otherwise.
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                print("Using a pre-trained MobileNet model instead")
                
                # Use a pre-trained MobileNet as fallback
                self.model = torchvision.models.mobilenet_v2(pretrained=True)
                # Replace the classifier to match our number of classes
                num_classes = len(self.classes)
                self.model.classifier[1] = torch.nn.Linear(
                    self.model.classifier[1].in_features,
                    num_classes
                )
            else:
                # Load the model from file
                self.model = torch.load(self.model_path, map_location=self.device)
            
            # Move model to the appropriate device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image_for_model(self, image):
        """
        Preprocess an image for the model.
        Converts OpenCV BGR to PIL RGB format.
        """
        # Convert BGR to RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Apply transforms
        tensor_image = self.transform(pil_image)
        
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)
        
        return tensor_image
    
    def classify_image(self, image):
        """
        Classify an entire image.
        Returns the predicted class index and probability.
        """
        if self.model is None:
            if not self.load_model():
                return -1, 0.0
        
        # Preprocess the image
        tensor_image = self.preprocess_image_for_model(image)
        tensor_image = tensor_image.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(tensor_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get the predicted class
            _, predicted = torch.max(outputs, 1)
            confidence = probabilities[0][predicted.item()].item()
        
        return predicted.item(), confidence
    
    def find_objects(self, frame, min_confidence=0.5):
        """
        Find and classify objects in the frame.
        Current implementation classifies the entire image, assuming one main object.
        For multiple object detection, a dedicated object detection model would be needed.
        
        Returns a list with a single object classification.
        """
        if frame is None:
            return []
        
        # Classify the entire image
        class_idx, confidence = self.classify_image(frame)
        
        # If confidence is too low or classification failed, return empty list
        if class_idx < 0 or confidence < min_confidence:
            return []
        
        # Get class name
        class_name = self.classes[class_idx] if class_idx < len(self.classes) else "unknown"
        
        # Create an object in the center of the frame
        height, width = frame.shape[:2]
        cx, cy = width // 2, height // 2
        
        # Create a central contour (circle shape)
        radius = min(width, height) // 4
        contour = np.array([[
            [cx + int(radius * np.cos(angle)), cy + int(radius * np.sin(angle))]
            for angle in np.linspace(0, 2*np.pi, 20, endpoint=False)
        ]], dtype=np.int32)
        
        # Create object with classification result
        detected_object = {
            "class": class_name,
            "contour": contour,
            "centroid": (cx, cy),
            "confidence": confidence
        }
        
        return [detected_object]
    
    def visualize(self, frame, objects):
        """
        Visualize detected objects in the frame.
        Draws bounding boxes with class names and confidence.
        """
        if frame is None or not objects:
            return frame
        
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Process each detected object
        for obj in objects:
            # Get contour
            contour = obj["contour"]
            
            # Draw contour in green
            cv2.drawContours(vis_frame, [contour], -1, (0, 255, 0), 2)
            
            # Add label with class and confidence
            class_name = obj["class"]
            confidence = obj["confidence"]
            cx, cy = obj["centroid"]
            
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(
                vis_frame, 
                label, 
                (cx - 50, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
        
        return vis_frame


# Test function
def test_dl_detector():
    """Test the deep learning detector with a camera feed."""
    from camera.capture import Camera
    
    camera = Camera()
    detector = DeepLearningDetector()
    
    # Create models directory if it doesn't exist
    models_dir = os.path.dirname(config.MODEL_PATH)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    if not camera.open():
        print("Failed to open camera. Exiting...")
        return
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret:
                print("Failed to read frame. Exiting...")
                break
            
            # Detect objects using deep learning
            objects = detector.find_objects(frame)
            
            # Visualize the results
            vis_frame = detector.visualize(frame, objects)
            
            # Display the results
            cv2.imshow("Deep Learning Detection", vis_frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_dl_detector() 