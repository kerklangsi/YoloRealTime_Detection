
import cv2
from typing import Optional, Tuple, List, Dict
from detection.detector import YOLODetector
import numpy as np

class ImageHandler:
	def __init__(self, detector: YOLODetector):
		self.detector = detector

	def load_image(self, image_path: str) -> Optional[np.ndarray]:
		image = cv2.imread(image_path)
		if image is None:
			print(f"Error: Could not load image {image_path}")
			return None
		return image

	def detect_image(self, image_path: str, conf_threshold: float = 0.5, nms_threshold: float = 0.4) -> Tuple[Optional[np.ndarray], List[Dict]]:
		image = self.load_image(image_path)
		if image is None:
			return None, []
		self.detector.reset_current_counts()
		annotated_frame, detections = self.detector.detect_objects(image, conf_threshold, nms_threshold)
		return annotated_frame, detections
