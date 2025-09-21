import os
import cv2
import time
import threading
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from gui.left_panel import setup_left_panel
from gui.center_panel import setup_center_panel
from gui.right_panel import setup_right_panel
from PIL import Image, ImageTk
try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None

from detection.detector import YOLODetector
from detection.image_handler import ImageHandler
from detection.video_handler import VideoHandler

class YOLOGui:
	def __init__(self):
		self.root = tk.Tk()
		self.root.title("YOLO Object Detection GUI")
		self.root.geometry("1480x630")
		self.root.minsize(1480, 630)
		self.root.configure(bg="#ffffff")

		# Load icon for window panel
		icon_path = os.path.join(os.path.dirname(__file__), '../icons/YoloRealTime_Detection.ico')
		try:
			self.root.iconbitmap(icon_path)
		except Exception:
			pass
		else:
			self.window_img_tk = None

		self.detector = YOLODetector()
		self.image_handler = ImageHandler(self.detector)
		self.video_handler = VideoHandler(self.detector)
		self.cap = None
		self.is_running = False
		self.current_source = None
		self.processing_thread = None
		self.selected_source = tk.StringVar(value="media_device")
		self.selected_model = tk.StringVar()
		self.custom_model_path = tk.StringVar()
		self.rtsp_url = tk.StringVar()
		self.conf_threshold = tk.DoubleVar(value=0.5)
		self.nms_threshold = tk.DoubleVar(value=0.4)
		self.selected_media_device = tk.StringVar()
		self.setup_gui()
		self.refresh_media_devices()
		self.refresh_models()

	def setup_gui(self):
		self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
		self.paned_window.pack(fill=tk.BOTH, expand=True)
		self.paned_window.bind('<B1-Motion>', lambda e: "break")
		self.paned_window.bind('<Button-1>', lambda e: "break")
		self.left_frame = ttk.Frame(self.paned_window, width=400, style="LeftPanel.TFrame")
		self.left_frame.pack_propagate(True)
		self.paned_window.add(self.left_frame, weight=0)
		setup_left_panel(self, self.left_frame)
		self.center_frame = ttk.Frame(self.paned_window, style="CenterPanel.TFrame")
		self.center_frame.pack_propagate(True)
		self.paned_window.add(self.center_frame, weight=1)
		setup_center_panel(self, self.center_frame)
		self.right_frame = ttk.Frame(self.paned_window, width=350, style="RightPanel.TFrame")
		self.right_frame.pack_propagate(True)
		self.paned_window.add(self.right_frame, weight=0)
		setup_right_panel(self, self.right_frame)

	def update_conf_label(self, event=None):
		percent = int(self.conf_threshold.get() * 100)
		self.conf_label.config(text=f"{percent}%")

	def update_nms_label(self, event=None):
		percent = int(self.nms_threshold.get() * 100)
		self.nms_label.config(text=f"{percent}%")

	def refresh_media_devices(self):
		self.media_device_combo['values'] = ["None"]
		self.selected_media_device.set("None")
		if self.selected_source.get() == "media_device":
			devices = []
			if sys.platform == 'win32' and FilterGraph is not None:
				try:
					graph = FilterGraph()
					devices = graph.get_input_devices()
				except Exception as e:
					print(f"Error getting media devices with pygrabber: {e}")
			if not devices:
				for i in range(5):
					cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
					if cap.isOpened():
						devices.append(f"Device {i}")
						cap.release()
			if devices:
				self.media_device_combo['values'] = ["None"] + devices
				self.media_device_combo.current(0)
				self.selected_media_device.set("None")

	def on_source_change(self):
		# Always present attributes, no need for hasattr
		self.rtsp_frame.pack_forget()
		self.file_frame.pack_forget()
		self.media_device_frame.pack_forget()
		source = self.selected_source.get()
		if source == "rtsp":
			self.rtsp_frame.pack(fill=tk.X, pady=5)
		elif source == "video":
			self.file_frame.pack(fill=tk.X, pady=5)
			for child in self.file_frame.winfo_children():
				if isinstance(child, ttk.Button):
					if child.cget('text') == 'Select Video File':
						child.pack(fill=tk.X, pady=2)
					else:
						child.pack_forget()
			self.selected_file_label.pack_forget()
			self.selected_file_label.pack(fill=tk.X, pady=(2, 0))
		elif source == "image":
			self.file_frame.pack(fill=tk.X, pady=5)
			for child in self.file_frame.winfo_children():
				if isinstance(child, ttk.Button):
					if child.cget('text') == 'Select Image File':
						child.pack(fill=tk.X, pady=2)
					else:
						child.pack_forget()
			self.selected_file_label.pack_forget()
			self.selected_file_label.pack(fill=tk.X, pady=(2, 0))
		elif source == "media_device":
			self.media_device_frame.pack(fill=tk.X, pady=5)
			self.refresh_media_devices()
			# Show preview if a device is selected
			selected = self.selected_media_device.get()
			if selected and selected != "None":
				try:
					idx = int(selected.split()[-1])
					cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
					ret, frame = cap.read()
					if ret and frame is not None:
						self.display_frame(frame)
					cap.release()
				except Exception:
					pass

	def refresh_models(self):
		models_dir = Path(__file__).resolve().parent.parent / "models"
		models = list(models_dir.glob("*.pt"))
		if hasattr(self, 'model_listbox'):
			self.model_listbox.delete(0, tk.END)
			for model in models:
				self.model_listbox.insert(tk.END, model.name)
			if models and not self.selected_model.get():
				self.model_listbox.selection_set(0)
				self.on_model_select(None)

	def select_video_file(self):
		filename = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")])
		if filename:
			self.current_source = filename
			self.selected_file_label.config(text=filename)
			messagebox.showinfo("Success", f"Video file selected: {os.path.basename(filename)}")

	def select_image_file(self):
		filename = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
		if filename:
			self.current_source = filename
			self.selected_file_label.config(text=filename)
			frame = cv2.imread(filename)
			if frame is not None:
				self.display_frame(frame)
			else:
				messagebox.showerror("Error", f"Could not load image: {filename}")

	def process_image(self, image_path):
		if self.detector.model is None:
			messagebox.showerror("Error", "Please load a model first")
			return
		try:
			annotated_frame, detections = self.image_handler.detect_image(
				image_path,
				conf_threshold=self.conf_threshold.get(),
				nms_threshold=self.nms_threshold.get()
			)
			if annotated_frame is None:
				messagebox.showerror("Error", f"Could not load image: {image_path}")
				return
			self.display_frame(annotated_frame)
			self.update_detection_display()
			messagebox.showinfo("Success", f"Processed image with {len(detections)} detections")
		except Exception as e:
			messagebox.showerror("Error", f"Error processing image: {e}")

	def start_detection(self):
		if self.detector.model is None:
			messagebox.showerror("Error", "Please load a model first")
			return
		if self.is_running:
			return
		source_type = self.selected_source.get()
		try:
			if source_type == "media_device":
				selected = self.selected_media_device.get()
				if not selected or selected == "None":
					messagebox.showerror("Error", "No media device selected")
					return
				device_index = None
				if selected.startswith("Device "):
					try:
						device_index = int(selected.split(" ")[1])
					except Exception:
						device_index = 0
				else:
					devices = self.media_device_combo['values']
					if selected in devices:
						idx = devices.index(selected) - 1
						if idx >= 0:
							device_index = idx
				if device_index is None:
					messagebox.showerror("Error", f"Could not resolve device index for: {selected}")
					return
				if not self.video_handler.open_source(device_index):
					messagebox.showerror("Error", f"Could not open video source: {selected}")
					return
				self.cap = self.video_handler.cap
				self.is_running = True; self.start_button.config(state=tk.DISABLED)
				self.stop_button.config(state=tk.NORMAL); self.status_label.config(text="Status: Running")
				source_text = self.selected_media_device.get()
				self.source_display_label.config(text=f"Source: {source_text}"); self.source_path_label.config(text=f"Source Path: {source_text}")
				self.processing_thread = threading.Thread(target=self.process_video, daemon=True); self.processing_thread.start()
			elif source_type == "rtsp":
				rtsp_url = self.rtsp_url.get().strip()
				if not rtsp_url:
					messagebox.showerror("Error", "Please enter RTSP URL")
					return
				if not self.video_handler.open_source(rtsp_url):
					messagebox.showerror("Error", "Could not open video source")
					return
				self.cap = self.video_handler.cap
				self.is_running = True; self.start_button.config(state=tk.DISABLED)
				self.stop_button.config(state=tk.NORMAL); self.status_label.config(text="Status: Running")
				self.source_display_label.config(text=f"Source: {rtsp_url}"); self.source_path_label.config(text=f"Source Path: {rtsp_url}")
				self.processing_thread = threading.Thread(target=self.process_video, daemon=True); self.processing_thread.start()
			elif source_type == "video":
				if not self.current_source:
					messagebox.showerror("Error", "Please select a video file")
					return
				if not self.video_handler.open_source(self.current_source):
					messagebox.showerror("Error", "Could not open video source")
					return
				self.cap = self.video_handler.cap
				self.is_running = True; self.start_button.config(state=tk.DISABLED)
				self.stop_button.config(state=tk.NORMAL); self.status_label.config(text="Status: Running")
				self.source_display_label.config(text=f"Source: {self.current_source}"); self.source_path_label.config(text=f"Source Path: {self.current_source}")
				self.processing_thread = threading.Thread(target=self.process_video, daemon=True); self.processing_thread.start()
			elif source_type == "image":
				if not self.current_source:
					messagebox.showerror("Error", "Please select an image file")
					return
				annotated_frame, detections = self.image_handler.detect_image(
					self.current_source,
					conf_threshold=self.conf_threshold.get(),
					nms_threshold=self.nms_threshold.get()
				)
				if annotated_frame is None:
					messagebox.showerror("Error", f"Could not load image: {self.current_source}")
					return
				self.display_frame(annotated_frame)
				self.update_detection_display()
				messagebox.showinfo("Success", f"Processed image with {len(detections)} detections")
			else:
				messagebox.showerror("Error", "Unknown source type")
				return
		except Exception as e:
			messagebox.showerror("Error", f"Error starting detection: {e}")

	def stop_detection(self):
		self.is_running = False
		if self.cap: self.cap.release()
		self.start_button.config(state=tk.NORMAL); self.stop_button.config(state=tk.DISABLED)
		self.status_label.config(text="Status: Stopped"); self.fps_label.config(text="FPS: 0")

	def process_video(self):
		fps_counter = 0; fps_start_time = time.time()
		resize_width, resize_height = 1024, 768
		if self.cap is not None:
			source_fps = self.cap.get(cv2.CAP_PROP_FPS)
			if not source_fps or source_fps <= 0 or source_fps > 120: source_fps = 30.0
		else: source_fps = 30.0
		frame_time = 1.0 / source_fps
		while self.is_running and self.cap and self.cap.isOpened():
			start_time = time.time(); ret, frame = self.cap.read()
			if not ret: break
			frame = cv2.resize(frame, (resize_width, resize_height)); self.detector.reset_current_counts()
			annotated_frame, detections = self.detector.detect_objects(frame, self.conf_threshold.get(), self.nms_threshold.get())
			self.root.after(0, lambda f=annotated_frame: self.display_frame(f))
			self.root.after(0, self.update_detection_display); fps_counter += 1
			if fps_counter % 30 == 0:
				elapsed = time.time() - fps_start_time; fps = 30 / elapsed if elapsed > 0 else 0
				self.root.after(0, lambda f=fps: self.fps_label.config(text=f"FPS: {f:.1f}")); fps_start_time = time.time()
			elapsed_time = time.time() - start_time; sleep_time = frame_time - elapsed_time
			if sleep_time > 0: time.sleep(sleep_time)
		self.root.after(0, self.stop_detection)

	def display_frame(self, frame):
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); label_width = self.video_label.winfo_width()
		label_height = self.video_label.winfo_height()
		if label_width > 1 and label_height > 1:
			h, w = frame_rgb.shape[:2]; aspect_ratio = w / h; max_height = label_height - 10; new_height = max_height
			new_width = int(new_height * aspect_ratio)
			if new_width > label_width - 10: new_width = label_width - 10; new_height = int(new_width / aspect_ratio)
			frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
		else: frame_resized = frame_rgb
		pil_image = Image.fromarray(frame_resized); photo = ImageTk.PhotoImage(pil_image)
		self.video_label.configure(image=photo, text=""); self._current_photo = photo

	def on_model_select(self, event):
		if hasattr(self, 'model_listbox'):
			selection = self.model_listbox.curselection()
			if selection:
				model_name = self.model_listbox.get(selection[0])
				models_dir = Path(__file__).resolve().parent.parent / "models"
				model_path = str(models_dir / model_name)
				self.selected_model.set(model_path)
				if self.detector.load_model(model_path):
					if hasattr(self, 'model_label'):
						self.model_label.config(text=f"Model: {model_name}")
					self.update_gpu_status()
					messagebox.showinfo("Success", f"Model loaded: {model_name}")
				else:
					messagebox.showerror("Error", f"Failed to load model: {model_name}")

	def load_custom_model_dialog(self):
		filename = filedialog.askopenfilename(title="Select YOLO Model File", filetypes=[("YOLO model files", "*.pt")])
		if filename:
			self.custom_model_path.set(filename)
			if self.detector.load_model(filename):
				if hasattr(self, 'model_label'):
					self.model_label.config(text=f"Model: {os.path.basename(filename)}")
				self.update_gpu_status()
				messagebox.showinfo("Success", f"Custom model loaded: {filename}")
				if hasattr(self, 'model_listbox') and filename not in self.model_listbox.get(0, tk.END):
					self.model_listbox.insert(tk.END, filename)
					idx = self.model_listbox.get(0, tk.END).index(filename)
					self.model_listbox.selection_clear(0, tk.END)
					self.model_listbox.selection_set(idx)
					self.model_listbox.see(idx)
					self.selected_model.set(filename)
			else:
				messagebox.showerror("Error", f"Failed to load custom model: {filename}")

	def update_gpu_status(self):
		device_str = getattr(self.detector, 'device_str', None)
		if device_str: self.gpu_status_label.config(text=f"Device: {device_str}")
		else: self.gpu_status_label.config(text="Device: Unknown")

	def save_statistics(self):
		try: filename = self.detector.save_statistics(); messagebox.showinfo("Success", f"Statistics saved to: {filename}")
		except Exception as e: messagebox.showerror("Error", f"Error saving statistics: {e}")

	def clear_source(self):
		self.detector.reset_source()
		self.update_detection_display()
		self.selected_source.set("media_device")
		self.selected_media_device.set("")
		self.rtsp_url.set("")
		self.current_source = None
		self.selected_file_label.config(text="No file selected")
		self.on_source_change()
		self.source_display_label.config(text="Source: None")
		self.source_path_label.config(text="Source Path: None")
		self.video_label.config(image='', text="Video output will appear here", bg="#000000", fg="white")
		self._current_photo = None
		messagebox.showinfo("Success", "source selection reset")

	def reset_statistics(self):
		self.detector.reset_source()
		self.update_detection_display()
		messagebox.showinfo("Success", "Statistics have been cleared.")

	def update_detection_display(self):
		for item in self.detection_tree.get_children():
			self.detection_tree.delete(item)
		detections = self.detector.get_current_detections()
		for class_name, stats in self.detector.detection_stats.items():
			current_count = stats['current_count']
			total_count = stats['total_count']
			avg_conf = stats['avg_confidence']
			self.detection_tree.insert('', 'end', values=(class_name, current_count, total_count, f"{avg_conf:.2f}"))
		session_stats = self.detector.get_session_stats()
		self.session_time_label.config(text=f"Session Time: {session_stats['sessionTime']}")
		self.total_detections_label.config(text=f"Total Detections: {session_stats['totalDetections']}")

	def run(self):
		self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
		self.root.mainloop()

	def on_closing(self):
		if self.is_running:
			self.stop_detection()
		self.root.destroy()
