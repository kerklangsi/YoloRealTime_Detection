import os
import cv2
import time
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO

class YOLODetector:
	def __init__(self):
		self.model, self.model_path, self.class_names = None, None, []
		self.detection_stats, self.session_start_time = {}, None
		self.total_detections, self.is_running = 0, False
		self.device_str = "Unknown"

	def scan_available_models(self) -> List[str]:
		model_files, current_dir = [], Path(".")
		for pt_file in current_dir.glob("*.pt"):
			model_files.append(str(pt_file))
		return sorted(model_files)

	def load_model(self, model_path: str) -> bool:
		if not os.path.exists(model_path) or not model_path.endswith(".pt"):
			print(f"Invalid model: {model_path}")
			return False
		try:
			self.model = YOLO(model_path)
			self.model_path = model_path
			import torch
			if torch.cuda.is_available():
				self.model.to("cuda")
				self.device_str = f"GPU ({torch.cuda.get_device_name(torch.cuda.current_device())})"
			else:
				self.model.to("cpu")
				self.device_str = "CPU"
			self.class_names = list(self.model.names.values())
			self.reset_source()
			return True
		except Exception as e:
			print(f"Error loading model: {e}")
			return False

	def detect_objects(self, frame: np.ndarray, conf_threshold: float = 0.5, nms_threshold: float = 0.4) -> Tuple[np.ndarray, List[Dict]]:
		if self.model is None:
			return frame, []
		try:
			results = self.model(frame, conf=conf_threshold, iou=nms_threshold)
			annotated_frame, detections = self._process_results(frame, results[0])
			return annotated_frame, detections
		except Exception as e:
			print(f"Error during detection: {e}")
			return frame, []

	def _process_results(self, frame: np.ndarray, result) -> Tuple[np.ndarray, List[Dict]]:
		detections, annotated_frame = [], frame.copy()
		if result.boxes is not None:
			boxes = result.boxes.xyxy.cpu().numpy()
			confidences = result.boxes.conf.cpu().numpy()
			class_ids = result.boxes.cls.cpu().numpy().astype(int)
			for box, conf, class_id in zip(boxes, confidences, class_ids):
				x1, y1, x2, y2 = box.astype(int)
				if 0 <= class_id < len(self.class_names):
					class_name = self.class_names[class_id]
				else:
					class_name = f"class_{class_id}"
				self._update_detection_stats(class_name, conf)
				cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
				label = f"{class_name}: {conf:.2f}"
				label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
				cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
				cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
				detections.append({'class': class_name,'confidence': float(conf),'bbox': [int(x1), int(y1), int(x2), int(y2)]})
		return annotated_frame, detections

	def _update_detection_stats(self, class_name: str, confidence: float):
		if class_name not in self.detection_stats:
			self.detection_stats[class_name] = {'total_count': 0,'current_count': 0,'avg_confidence': 0.0,'confidence_sum': 0.0}
		stats = self.detection_stats[class_name]
		stats['total_count'] += 1
		stats['current_count'] += 1
		stats['confidence_sum'] += confidence
		stats['avg_confidence'] = stats['confidence_sum'] / stats['total_count']
		self.total_detections += 1

	def get_current_detections(self) -> List[Dict]:
		detections = []
		for class_name, stats in self.detection_stats.items():
			if stats['current_count'] > 0:
				detections.append({'class': class_name, 'confidence': stats['avg_confidence'], 'count': stats['current_count'],
								   'totalEncountered': stats['total_count']})
		return detections

	def reset_current_counts(self):
		[stats.update({'current_count': 0}) for stats in self.detection_stats.values()]

	def reset_source(self):
		self.detection_stats = {}
		self.total_detections = 0
		self.session_start_time = datetime.now()

	def get_session_stats(self) -> Dict:
		if self.session_start_time is None:
			self.session_start_time = datetime.now()
		session_time = datetime.now() - self.session_start_time
		return {'modelName': os.path.basename(self.model_path) if self.model_path else 'No model', 'totalDetections': self.total_detections,
				 'sessionTime': str(session_time).split('.')[0],
				 'classStats': {name: stats['total_count'] for name, stats in self.detection_stats.items()}}

	def save_statistics(self, filename: Optional[str] = None) -> str:
		if filename is None:
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			filename = f"yolo_stats_{timestamp}.json"
		stats_data = {
			'session_info': self.get_session_stats(), 'detailed_stats': self.detection_stats, 'export_time': datetime.now().isoformat()}
		with open(filename, 'w') as f:
			json.dump(stats_data, f, indent=2)
			print(f"Statistics saved to: {filename}")
		return filename
