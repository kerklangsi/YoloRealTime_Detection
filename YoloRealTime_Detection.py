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
from PIL import Image, ImageTk
from ultralytics import YOLO
try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None

class YOLODetector:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.class_names = []
        self.detection_stats = {}
        self.session_start_time = None
        self.total_detections = 0
        self.is_running = False
        self.device_str = "Unknown"
    
    def scan_available_model(self) -> List[str]:
        model_files = []
        # Use absolute path for reliability
        for model_dir in [Path(__file__).parent / "model", Path(__file__).parent / "models"]:
            if model_dir.exists():
                for pt_file in model_dir.glob("*.pt"):
                    model_files.append(pt_file.name)
        return sorted(model_files)
    
    def load_model(self, model_path: str) -> bool:
        candidate_paths = [Path(model_path)]
        if not Path(model_path).is_absolute():
            candidate_paths.append(Path("./model") / model_path)
        full_path = None
        for path in candidate_paths:
            if path.exists() and str(path).endswith(".pt"):
                full_path = path
                break
        if full_path is None:
            print(f"Invalid model: {model_path}")
            return False
        try:
            self.model = YOLO(str(full_path))
            self.model_path = str(full_path)
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
        detections = []
        annotated_frame = frame.copy()
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
                detections.append({'class': class_name, 'confidence': float(conf), 'bbox': [int(x1), int(y1), int(x2), int(y2)]})
        return annotated_frame, detections
    
    def _update_detection_stats(self, class_name: str, confidence: float):
        if class_name not in self.detection_stats:
            self.detection_stats[class_name] = {
                'total_count': 0, 'current_count': 0, 'avg_confidence': 0.0, 'confidence_sum': 0.0
            }
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
                detections.append({
                    'class': class_name, 'confidence': stats['avg_confidence'],
                    'count': stats['current_count'], 'totalEncountered': stats['total_count']
                })
        return detections
    
    def reset_current_counts(self):
        for stats in self.detection_stats.values():
            stats.update({'current_count': 0})

    def reset_source(self):
        self.detection_stats = {}
        self.total_detections = 0
        self.session_start_time = datetime.now()
    
    def get_session_stats(self) -> Dict:
        if self.session_start_time is None:
            self.session_start_time = datetime.now()
        session_time = datetime.now() - self.session_start_time
        return {
            'modelName': os.path.basename(self.model_path) if self.model_path else 'No model',
            'totalDetections': self.total_detections, 'sessionTime': str(session_time).split('.')[0],
            'classStats': {name: stats['total_count'] for name, stats in self.detection_stats.items()}
        }
    
    def save_statistics(self, filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"yolo_stats_{timestamp}.json"
        stats_data = {
            'session_info': self.get_session_stats(), 'detailed_stats': self.detection_stats,
            'export_time': datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(stats_data, f, indent=2)
            print(f"Statistics saved to: {filename}")
            return filename

class YOLOGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO Object Detection GUI")
        self.root.geometry("1480x630")
        self.root.minsize(1480, 630)
        self.root.configure(bg="#ffffff")
        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception as e:
                print(f"Could not set icon: {e}")
        self.detector = YOLODetector()
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
        self.refresh_model()
    
    def setup_gui(self):
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        self.paned_window.bind('<B1-Motion>', lambda e: "break")
        self.paned_window.bind('<Button-1>', lambda e: "break")
        self.left_frame = ttk.Frame(self.paned_window, width=400)
        self.left_frame.pack_propagate(True)
        self.paned_window.add(self.left_frame, weight=0)
        self.setup_left_panel(self.left_frame)
        self.center_frame = ttk.Frame(self.paned_window)
        self.center_frame.pack_propagate(True)
        self.paned_window.add(self.center_frame, weight=1)
        self.setup_center_panel(self.center_frame)
        self.right_frame = ttk.Frame(self.paned_window, width=350)
        self.right_frame.pack_propagate(True)
        self.paned_window.add(self.right_frame, weight=0)
        self.setup_right_panel(self.right_frame)
    
    def setup_left_panel(self, parent):
        left_frame = ttk.LabelFrame(parent, text="Source Selection", padding=10)
        left_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 2), pady=(0, 0))
        ttk.Label(left_frame, text="Input Source:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        sources = ["Media Device", "RTSP Stream", "Video File", "Image File"]
        values = ["media_device", "rtsp", "video", "image"]
        for text, value in zip(sources, values):
            ttk.Radiobutton(left_frame, text=text, variable=self.selected_source, value=value, command=self.on_source_change).pack(anchor=tk.W, pady=2)
        self.source_frame = ttk.Frame(left_frame)
        self.source_frame.pack(fill=tk.X, pady=10)
        self.rtsp_frame = ttk.Frame(self.source_frame)
        ttk.Label(self.rtsp_frame, text="RTSP URL:").pack(anchor=tk.W)
        ttk.Entry(self.rtsp_frame, textvariable=self.rtsp_url, width=30).pack(fill=tk.X, pady=2)
        self.media_device_frame = ttk.Frame(self.source_frame)
        ttk.Label(self.media_device_frame, text="Select Media Device:").pack(anchor=tk.W)
        self.media_device_combo = ttk.Combobox(self.media_device_frame, textvariable=self.selected_media_device, state="readonly")
        self.media_device_combo.pack(fill=tk.X, pady=2)
        self.file_frame = ttk.Frame(self.source_frame)
        for txt, cmd in [("Select Video File", self.select_video_file), ("Select Image File", self.select_image_file)]:
            ttk.Button(self.file_frame, text=txt, command=cmd).pack(fill=tk.X, pady=2)
        self.selected_file_label = ttk.Label(self.file_frame, text="No file selected", foreground="blue")
        self.selected_file_label.pack(fill=tk.X, pady=2)
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="Select Model:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.model_listbox = tk.Listbox(left_frame, height=5, exportselection=False)
        self.model_listbox.pack(fill=tk.X, pady=2)
        self.model_listbox.bind('<<ListboxSelect>>', self.on_model_select)
        ttk.Button(left_frame, text="Load Custom Model", command=self.load_custom_model_dialog).pack(fill=tk.X, pady=2)
        ttk.Entry(left_frame, textvariable=self.custom_model_path, width=30, state='readonly').pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, text="Detection Settings:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        scale_width = 200
        label_width = 18
        for label, var, cmd, val in [
            ("Confidence Threshold:", self.conf_threshold, self.update_conf_label, "50%"),
            ("NMS Threshold:", self.nms_threshold, self.update_nms_label, "40%")
        ]:
            frame = ttk.Frame(left_frame)
            frame.pack(fill=tk.X, pady=2)
            frame.columnconfigure(0, minsize=label_width*8)
            frame.columnconfigure(1, weight=1)
            ttk.Label(frame, text=label).grid(row=0, column=0, sticky=tk.W)
            scale = ttk.Scale(frame, from_=0.0, to=1.0, variable=var, orient=tk.HORIZONTAL, command=cmd, length=scale_width)
            scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
            lbl = ttk.Label(frame, text=val)
            lbl.grid(row=0, column=2, sticky=tk.E)
            if label.startswith("Confidence"):
                self.conf_scale, self.conf_label = scale, lbl
            else:
                self.nms_scale, self.nms_label = scale, lbl
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        self.start_button = ttk.Button(left_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(fill=tk.X, pady=2)
        self.stop_button = ttk.Button(left_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Clear source", command=self.clear_source).pack(fill=tk.X, pady=2)
        self.on_source_change()
    
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
        for frame in [self.rtsp_frame, self.file_frame, self.media_device_frame]:
            frame.pack_forget()
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
    
    def refresh_model(self):
        self.model_listbox.delete(0, tk.END)
        model_names = self.detector.scan_available_model()
        for name in model_names:
            self.model_listbox.insert(tk.END, name)
        if model_names and not self.selected_model.get():
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
            frame = cv2.imread(image_path)
            if frame is None:
                messagebox.showerror("Error", f"Could not load image: {image_path}")
                return
            self.detector.reset_current_counts()
            annotated_frame, detections = self.detector.detect_objects(frame, self.conf_threshold.get(), self.nms_threshold.get())
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
                self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", f"Could not open video source: {selected}")
                    return
                # Display first frame immediately
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.display_frame(frame)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.is_running = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.status_label.config(text="Status: Running")
                source_text = self.selected_media_device.get()
                self.source_display_label.config(text=f"Source: {source_text}")
                self.source_path_label.config(text=f"Source Path: {source_text}")
                self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
                self.processing_thread.start()
            elif source_type == "rtsp":
                rtsp_url = self.rtsp_url.get().strip()
                if not rtsp_url:
                    messagebox.showerror("Error", "Please enter RTSP URL")
                    return
                self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not open video source")
                    return
                # Display first frame immediately
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.display_frame(frame)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.is_running = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.status_label.config(text="Status: Running")
                self.source_display_label.config(text=f"Source: {rtsp_url}")
                self.source_path_label.config(text=f"Source Path: {rtsp_url}")
                self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
                self.processing_thread.start()
            elif source_type == "video":
                if not self.current_source:
                    messagebox.showerror("Error", "Please select a video file")
                    return
                self.cap = cv2.VideoCapture(self.current_source)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not open video source")
                    return
                # Display first frame immediately to reduce perceived delay
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.display_frame(frame)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame for processing
                self.is_running = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.status_label.config(text="Status: Running")
                self.source_display_label.config(text=f"Source: {self.current_source}")
                self.source_path_label.config(text=f"Source Path: {self.current_source}")
                self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
                self.processing_thread.start()
            elif source_type == "image":
                if not self.current_source:
                    messagebox.showerror("Error", "Please select an image file")
                    return
                frame = cv2.imread(self.current_source)
                if frame is None:
                    messagebox.showerror("Error", f"Could not load image: {self.current_source}")
                    return
                self.detector.reset_current_counts()
                annotated_frame, detections = self.detector.detect_objects(frame, self.conf_threshold.get(), self.nms_threshold.get())
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
        if self.cap: 
            self.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        self.fps_label.config(text="FPS: 0")
    
    def process_video(self):
        fps_counter = 0
        fps_start_time = time.time()
        resize_width, resize_height = 1024, 768
        if self.cap is not None:
            source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not source_fps or source_fps <= 0 or source_fps > 120:
                source_fps = 30.0
        else:
            source_fps = 30.0
        frame_time = 1.0 / source_fps
        while self.is_running and self.cap and self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (resize_width, resize_height))
            self.detector.reset_current_counts()
            annotated_frame, detections = self.detector.detect_objects(frame, self.conf_threshold.get(), self.nms_threshold.get())
            self.root.after(0, lambda f=annotated_frame: self.display_frame(f))
            self.root.after(0, self.update_detection_display)
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - fps_start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                self.root.after(0, lambda f=fps: self.fps_label.config(text=f"FPS: {f:.1f}"))
                fps_start_time = time.time()
            elapsed_time = time.time() - start_time
            sleep_time = frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.root.after(0, self.stop_detection)
    
    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        if label_width > 1 and label_height > 1:
            h, w = frame_rgb.shape[:2]
            aspect_ratio = w / h
            max_height = label_height - 10
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
            if new_width > label_width - 10:
                new_width = label_width - 10
            new_height = int(new_width / aspect_ratio)
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        else:
            frame_resized = frame_rgb
        pil_image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(pil_image)
        self.video_label.configure(image=photo, text="")
        self._current_photo = photo
    
    def setup_center_panel(self, parent):
        center_frame = parent
        center_frame.config(height=520)
        center_frame.pack_propagate(False)
        self.source_display_frame = ttk.LabelFrame(center_frame, text="Source Display")  
        self.source_display_frame.pack(fill=tk.BOTH, expand=True)
        self.source_display_label = ttk.Label(self.source_display_frame, text="Source: None", font=('Arial', 12, 'bold'))
        self.source_display_label.pack(anchor=tk.NW, pady=(0, 10))
        self.video_box_frame = ttk.Frame(self.source_display_frame)
        self.video_box_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = tk.Label(
            self.video_box_frame, text="Video output will appear here", bg="#000000", fg="white", font=("Arial", 16), anchor="center"
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.bottom_status_frame = ttk.Frame(center_frame)
        self.bottom_status_frame.pack(fill=tk.X, pady=(10, 0))
        self.bottom_status_frame.columnconfigure(0, weight=1)
        self.bottom_status_frame.columnconfigure(1, weight=1)
        self.bottom_status_frame.columnconfigure(2, weight=1)
        self.status_label = ttk.Label(self.bottom_status_frame, text="Status: Ready")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        self.source_path_label = ttk.Label(self.bottom_status_frame, text="Source: None")
        self.source_path_label.grid(row=0, column=1)
        self.fps_label = ttk.Label(self.bottom_status_frame, text="FPS: 0")
        self.fps_label.grid(row=0, column=2, sticky=tk.E)
    
    def setup_right_panel(self, parent):
        right_frame = ttk.LabelFrame(parent, text="Detection Results")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        stats_frame = ttk.LabelFrame(right_frame, text="Session Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        self.model_label = ttk.Label(stats_frame, text="Model: None")
        self.model_label.pack(anchor=tk.W)
        self.gpu_status_label = ttk.Label(stats_frame, text="Device: Unknown")
        self.gpu_status_label.pack(anchor=tk.W)
        self.session_time_label = ttk.Label(stats_frame, text="Session Time: 00:00:00")
        self.session_time_label.pack(anchor=tk.W)
        self.total_detections_label = ttk.Label(stats_frame, text="Total Detections: 0")
        self.total_detections_label.pack(anchor=tk.W)
        export_frame = ttk.Frame(stats_frame)
        export_frame.pack(fill=tk.X, pady=5)
        ttk.Button(export_frame, text="Save Stats", command=self.save_statistics).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_frame, text="Clear Stats", command=self.reset_statistics).pack(side=tk.LEFT)
        ttk.Label(right_frame, text="Object Classes:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        list_frame = ttk.Frame(right_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        columns = ('Class', 'Current', 'Total', 'Confidence')
        self.detection_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        for col in columns:
            self.detection_tree.heading(col, text=col)
            self.detection_tree.column(col, width=80)
        tree_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.detection_tree.yview)
        self.detection_tree.configure(yscrollcommand=tree_scrollbar.set)
        self.detection_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def on_model_select(self, event):
        selection = self.model_listbox.curselection()
        if selection:
            model_name = self.model_listbox.get(selection[0])
            self.selected_model.set(model_name)
            # Use absolute path for both model and models
            for model_dir in [Path(__file__).parent / "model", Path(__file__).parent / "models"]:
                candidate_path = model_dir / model_name
                if candidate_path.exists():
                    if self.detector.load_model(str(candidate_path)):
                        self.model_label.config(text=f"Model: {model_name}")
                        self.update_gpu_status()
                        messagebox.showinfo("Success", f"Model loaded: {model_name}")
                        return
            messagebox.showerror("Error", f"Failed to load model: {model_name}")

    def load_custom_model_dialog(self):
        filename = filedialog.askopenfilename(title="Select YOLO Model File", filetypes=[("YOLO model files", "*.pt")])
        if filename:
            self.custom_model_path.set(filename)
            if self.detector.load_model(filename):
                self.model_label.config(text=f"Model:{os.path.basename(filename)}")
                self.update_gpu_status()
                messagebox.showinfo("Success", f"Custom model loaded: {filename}")
                if filename not in self.model_listbox.get(0, tk.END): 
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
        if device_str:
            self.gpu_status_label.config(text=f"Device: {device_str}")
        else:
            self.gpu_status_label.config(text="Device: Unknown")

    def save_statistics(self):
        try:
            filename = self.detector.save_statistics()
            messagebox.showinfo("Success", f"Statistics saved to: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving statistics: {e}")
    
    def clear_source(self):
        self.detector.reset_source()
        self.update_detection_display()
        self.selected_source.set("media_device")
        self.selected_media_device.set(""); self.rtsp_url.set("")
        self.current_source = None
        self.selected_file_label.config(text="No file selected")
        self.on_source_change()
        self.refresh_model()
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
            total_count = stats['total_count']; avg_conf = stats['avg_confidence']
            self.detection_tree.insert('', 'end', values=(class_name, current_count, total_count, f"{avg_conf:.2f}"))
        session_stats = self.detector.get_session_stats()
        self.session_time_label.config(text=f"Session Time: {session_stats['sessionTime']}")
        self.total_detections_label.config(text=f"Total Detections: {session_stats['totalDetections']}")
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.stop_detection() if self.is_running else None
        self.root.destroy()

if __name__ == "__main__":
    YOLOGui().run()