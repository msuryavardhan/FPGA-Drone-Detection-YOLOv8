import cv2
import numpy as np
import onnxruntime as ort
import time
from typing import List, Tuple, Optional

class DroneDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Initialize the drone detector with ONNX model
        
        Args:
            model_path: Path to the ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]  # 640
        self.input_width = self.input_shape[3]   # 640
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_shape}")
        print(f"Input name: {self.input_name}")
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO inference
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image tensor
        """
        # Resize image to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to CHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def postprocess_output(self, outputs: np.ndarray, original_shape: Tuple[int, int]) -> List[dict]:
        """
        Postprocess YOLO output to get bounding boxes and confidences
        
        Args:
            outputs: Raw model output
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detection dictionaries
        """
        predictions = outputs[0]  # Shape: (1, 84, 8400) for YOLOv8
        predictions = np.transpose(predictions, (0, 2, 1))  # Shape: (1, 8400, 84)
        
        # Extract boxes and scores
        boxes = predictions[0, :, :4]  # x_center, y_center, width, height
        scores = predictions[0, :, 4:]  # class scores
        
        # Get class with highest score
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence threshold
        valid_indices = confidences > self.conf_threshold
        boxes = boxes[valid_indices]
        confidences = confidences[valid_indices]
        class_ids = class_ids[valid_indices]
        
        if len(boxes) == 0:
            return []
        
        # Convert from center format to corner format
        x_centers, y_centers, widths, heights = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_centers - widths / 2
        y1 = y_centers - heights / 2
        x2 = x_centers + widths / 2
        y2 = y_centers + heights / 2
        
        # Scale to original image size
        orig_h, orig_w = original_shape
        x1 = x1 * orig_w / self.input_width
        y1 = y1 * orig_h / self.input_height
        x2 = x2 * orig_w / self.input_width
        y2 = y2 * orig_h / self.input_height
        
        # Apply NMS
        boxes_corners = np.column_stack([x1, y1, x2, y2])
        indices = cv2.dnn.NMSBoxes(
            boxes_corners.tolist(), 
            confidences.tolist(), 
            self.conf_threshold, 
            self.iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # Format results
        detections = []
        for i in indices.flatten():
            detection = {
                'bbox': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                'confidence': float(confidences[i]),
                'class_id': int(class_ids[i]),
                'class_name': 'Drone'
            }
            detections.append(detection)
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Run inference on a single image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detection dictionaries
        """
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess_output(outputs, image.shape[:2])
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image

def main():
    """Main function to run drone detection on camera"""
    # Initialize detector
    model_path = "best.onnx"  # Path to your ONNX model
    detector = DroneDetector(model_path, conf_threshold=0.5, iou_threshold=0.45)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use default camera (0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Starting drone detection on camera...")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Run detection
            detections = detector.detect(frame)
            
            # Draw detections
            result_frame = detector.draw_detections(frame, detections)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                print(f"FPS: {fps:.2f}")
            
            # Display detection count
            cv2.putText(result_frame, f"Detections: {len(detections)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Drone Detection", result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"drone_detection_{timestamp}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"Frame saved as {filename}")
    
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

if __name__ == "__main__":
    main()
