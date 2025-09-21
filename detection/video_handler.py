
import cv2
import time
from typing import Optional, Tuple, List, Dict, Union
from detection.detector import YOLODetector
import numpy as np

class VideoHandler:
	def __init__(self, detector: YOLODetector):
		self.detector = detector
		self.cap = None
		self.is_running = False

	def open_source(self, source: Union[int, str]) -> bool:
		self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if isinstance(source, int) else 0)
		if not self.cap.isOpened():
			print(f"Error: Could not open video source {source}")
			return False
		return True

	def release(self):
		if self.cap:
			self.cap.release()
			self.cap = None

	def process_stream(self, conf_threshold: float = 0.5, nms_threshold: float = 0.4, resize: Tuple[int, int] = (1024, 768), max_frames: Optional[int] = None) -> List[List[Dict]]:
		if self.cap is None:
			print("Error: No video source opened.")
			return []
		self.is_running = True
		frame_count = 0
		all_detections = []
		while self.is_running and self.cap.isOpened():
			ret, frame = self.cap.read()
			if not ret:
				break
			frame = cv2.resize(frame, resize)
			self.detector.reset_current_counts()
			annotated_frame, detections = self.detector.detect_objects(frame, conf_threshold, nms_threshold)
			all_detections.append(detections)
			frame_count += 1
			if max_frames and frame_count >= max_frames:
				break
		self.release()
		return all_detections
