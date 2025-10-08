import os,cv2,time,threading,sys,json,torch,socket,subprocess,signal
import numpy as np
import tkinter as tk
from pathlib import Path
from datetime import datetime
from typing import Dict,List,Optional,Tuple
from tkinter import ttk,filedialog,messagebox
from PIL import Image,ImageTk
from ultralytics import YOLO
from pygrabber.dshow_graph import FilterGraph

# Model Loading
class YOLOModel:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.class_names = []
        self.device_str = "UNKNOWN"
    # Scan for available YOLO models in a directory
    def scan_available_models(self, model_dir: Optional[Path] = None) -> List[str]:
        if model_dir is None:
            model_dir = Path(".")
        return sorted([str(pt_file) for pt_file in model_dir.glob("*.pt")])
    # Load a YOLO model
    def load_model(self, model_path: str) -> bool:
        if not os.path.exists(model_path) or not model_path.endswith(".pt"):
            print(f"Invalid model: {model_path}")
            return False
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            if torch.cuda.is_available():
                self.model.to("cuda")
                self.device_str = f"GPU ({torch.cuda.get_device_name(torch.cuda.current_device())})"
            else:
                self.model.to("cpu")
                self.device_str = "CPU"
            self.class_names = list(self.model.names.values())
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Statistics management class
class YOLOStats:
    def __init__(self):
        self.detection_stats = {}
        self.total_detections = 0
        self.session_start = None
    # Reset statistics
    def reset(self):
        self.detection_stats = {}
        self.total_detections = 0
        self.session_start = datetime.now()
    # Update detection statistics
    def update_detection_stats(self, class_name: str, confidence: float, bbox: Optional[List[int]] = None):
        if class_name not in self.detection_stats:
            self.detection_stats[class_name] = {
                'total_count': 0,
                'current_count': 0,
                'avg_confidence': 0.0,
                'confidence_sum': 0.0,
                'last_bboxes': []
            }
        stats = self.detection_stats[class_name]
        stats['current_count'] += 1
        stats['confidence_sum'] += confidence
        stats['avg_confidence'] = stats['confidence_sum'] / (stats['total_count'] + 1)
        is_new_location = True
        if bbox is not None:
            # Check if this bbox is close to any previous bbox (center distance < 100px)
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for prev_bbox in stats['last_bboxes']:
                px1, py1, px2, py2 = prev_bbox
                pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                if abs(cx - pcx) < 100 and abs(cy - pcy) < 100:
                    is_new_location = False
                    break
            stats['last_bboxes'].append(bbox)
            if len(stats['last_bboxes']) > 20:
                stats['last_bboxes'] = stats['last_bboxes'][-20:]
        if is_new_location:
            stats['total_count'] += 1
            self.total_detections += 1
    # Reset current detection counts
    def reset_counts(self):
        [stats.update({'current_count': 0}) for stats in self.detection_stats.values()]
    # Get current detections
    def get_current_detections(self) -> List[Dict]:
        detections = []
        for class_name, stats in self.detection_stats.items():
            if stats['current_count'] > 0:
                detections.append({'class': class_name, 'confidence': stats['avg_confidence'], 'count': stats['current_count'], 'totalEncountered': stats['total_count']})
        return detections
    # Get session statistics
    def get_session_stats(self, model_path: Optional[str] = None) -> Dict:
        if self.session_start is None:
            self.session_start = datetime.now()
        session_time = datetime.now() - self.session_start
        return {
            'modelName': os.path.basename(model_path) if model_path else 'No model',
            'totalDetections': self.total_detections,
            'sessionTime': str(session_time).split('.')[0],
            'classStats': {name: stats['total_count'] for name, stats in self.detection_stats.items()}
        }
    # Save statistics to JSON file
    def save_statistics(self, model_path: Optional[str] = None, filename: Optional[str] = None) -> str:
        save_dir = Path(__file__).resolve().parent / "save"
        save_dir.mkdir(exist_ok=True)
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = str(save_dir / f"yolo_stats_{timestamp}.json")
        def convert_stats(stats):
            out = {}
            for k, v in stats.items():
                out[k] = {}
                for sk, sv in v.items():
                    if isinstance(sv, np.floating):
                        out[k][sk] = float(sv)
                    else:
                        out[k][sk] = sv
            return out
        stats_data = {
            'session_info': self.get_session_stats(model_path),
            'detailed_stats': convert_stats(self.detection_stats),
            'export_time': datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(stats_data, f, indent=2)
            print(f"Statistics saved to: {filename}")
        return filename

# Streamlined detector class
class YOLODetector:
    def __init__(self):
        self.model_manager = YOLOModel()
        self.stats_manager = YOLOStats()
        self.running = False
    # Load a YOLO model
    def load_model(self, model_path: str) -> bool:
        loaded = self.model_manager.load_model(model_path)
        if loaded:
            self.stats_manager.reset()
        return loaded
    # Perform object detection on a frame
    def objects(self, frame: np.ndarray, conf_threshold: float = 0.5, nms_threshold: float = 0.4) -> Tuple[np.ndarray, List[Dict]]:
        model = self.model_manager.model
        class_names = self.model_manager.class_names
        if model is None:
            return frame, []
        try:
            results = model(frame, conf=conf_threshold, iou=nms_threshold)
            annotated_frame, detections = self.process_results(frame, results[0], class_names)
            return annotated_frame, detections
        except Exception as e:
            print(f"Error during detection: {e}")
            return frame, []
    # Process detection results and annotate frame
    def process_results(self, frame: np.ndarray, result, class_names: List[str]) -> Tuple[np.ndarray, List[Dict]]:
        detections, annotated_frame = [], frame.copy()
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"class_{class_id}"
                self.stats_manager.update_detection_stats(class_name, conf, [int(x1), int(y1), int(x2), int(y2)])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                detections.append({'class': class_name,'confidence': float(conf), 'bbox': [int(x1), int(y1), int(x2), int(y2)]})
        return annotated_frame, detections
    # Reset current detection counts
    def reset_counts(self):
        self.stats_manager.reset_counts()
    # Reset statistics
    def reset_source(self):
        self.stats_manager.reset()
    # Get current detections
    def get_current_detections(self) -> List[Dict]:
        return self.stats_manager.get_current_detections()
    # Get session statistics
    def get_session_stats(self) -> Dict:
        return self.stats_manager.get_session_stats(self.model_manager.model_path)
    # Save statistics to JSON file
    def save_statistics(self, filename: Optional[str] = None) -> str:
        return self.stats_manager.save_statistics(self.model_manager.model_path, filename)

# GUI source management class
class YOLOSource:
    def __init__(self, gui):
        self.gui = gui
    # Initialize media device list
    def refresh_media_devices(self):
        self.gui.media_device_combo['values'] = ["None"]
        self.gui.selected_device.set("None")
        if self.gui.selected_source.get() == "media_device":
            devices = []
            if sys.platform == 'win32' and FilterGraph is not None:
                graph = FilterGraph()
                devices = graph.get_input_devices()
            if not devices:
                for i in range(5):
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        devices.append(f"Device {i}")
                        cap.release()
            if devices:
                self.gui.media_device_combo['values'] = ["None"] + devices
                self.gui.media_device_combo.current(0)
                self.gui.selected_device.set("None")
    # Initialize model list
    def refresh_models(self):
        model_dir = Path(__file__).resolve().parent / "model"
        models = list(model_dir.glob("*.pt"))
        self.gui.model_listbox.delete(0, tk.END)
        # Show all available models in the folder
        for model_path in models:
            self.gui.model_listbox.insert(tk.END, model_path.name)
        if models and not self.gui.selected_model.get():
            self.gui.model_listbox.selection_set(0)
            self.gui.model_select(None)
    # Select video file
    def select_video_file(self):
        filename = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")])
        if filename:
            self.gui.current_source = filename
            self.gui.file_label.config(text=filename)
            messagebox.showinfo("Success", f"Video file selected: {os.path.basename(filename)}")
    # Select image file
    def select_image_file(self):
        filename = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if filename:
            self.gui.current_source = filename
            self.gui.file_label.config(text=filename)
            frame = cv2.imread(filename)
            if frame is not None:
                self.gui.yolo_process.display_frame(frame)
            else:
                messagebox.showerror("Error", f"Could not load image: {filename}")
    def select_folder(self):
        foldername = filedialog.askdirectory(title="Select Folder")
        if foldername:
            self.gui.current_source = foldername
            self.gui.file_label.config(text=foldername)
            messagebox.showinfo("Success", f"Folder selected: {os.path.basename(foldername)}")
    # Handle source type change
    def on_source_change(self):
        for frame in [self.gui.rtsp_frame, self.gui.file_frame, self.gui.media_device_frame]:
            frame.pack_forget()
        source = self.gui.selected_source.get()
        # Display frame from the RTSP stream
        if source == "rtsp":
            self.gui.rtsp_frame.pack(fill=tk.X, pady=5)
        # Display frame from the selected video file
        elif source == "video":
            self.gui.file_frame.pack(fill=tk.X, pady=5)
            for child in self.gui.file_frame.winfo_children():
                if isinstance(child, ttk.Button):
                    if child.cget('text') == 'Select Video File':
                        child.pack(fill=tk.X, pady=2)
                    else:
                        child.pack_forget()
            self.gui.file_label.pack_forget()
            self.gui.file_label.pack(fill=tk.X, pady=(2, 0))
        # Display frame from the selected image file
        elif source == "image":
            self.gui.file_frame.pack(fill=tk.X, pady=5)
            for child in self.gui.file_frame.winfo_children():
                if isinstance(child, ttk.Button):
                    if child.cget('text') == 'Select Image File':
                        child.pack(fill=tk.X, pady=2)
                    else:
                        child.pack_forget()
            self.gui.file_label.pack_forget()
            self.gui.file_label.pack(fill=tk.X, pady=(2, 0))
        # Display only the folder selection button for 'Folder' source
        elif source == "folder":
            self.gui.file_frame.pack(fill=tk.X, pady=5)
            for child in self.gui.file_frame.winfo_children():
                if isinstance(child, ttk.Button):
                    if child.cget('text') == 'Select Folder':
                        child.pack(fill=tk.X, pady=2)
                    else:
                        child.pack_forget()
            self.gui.file_label.pack_forget()
            self.gui.file_label.pack(fill=tk.X, pady=(2, 0))
        # Display frame from the selected folder
        elif source == "folder":
            self.gui.file_frame.pack(fill=tk.X, pady=5)
            folder = self.gui.current_source
            if folder and os.path.isdir(folder):
                images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
                if images:
                    self.gui.file_label.config(text=f"{len(images)} images found")
                else:
                    self.gui.file_label.config(text="No images found")
        # Display frame from the selected media device
        elif source == "media_device":
            self.gui.media_device_frame.pack(fill=tk.X, pady=5)
            self.refresh_media_devices()
            selected = self.gui.selected_device.get()
            if selected and selected != "None":
                idx = int(selected.split()[-1])
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.gui.display_frame(frame)
                cap.release()
    # Handle model selection
    def model_select(self, event):
        selection = self.gui.model_listbox.curselection()
        if selection:
            model_name = self.gui.model_listbox.get(selection[0])
            # If model_name is a full path, use it directly otherwise, build path from model_dir
            if os.path.isabs(model_name) and os.path.exists(model_name):
                model_path = model_name
            else:
                model_dir = Path(__file__).resolve().parent / "model"
                model_path = str(model_dir / model_name)
            self.gui.selected_model.set(model_name)
            # Attempt to load the selected model
            if self.gui.detector.load_model(model_path):
                self.gui.model_label.config(text=f"Model: {os.path.basename(model_path)}")
                self.gui.stats_panel.update_gpu_status()
                messagebox.showinfo("Success", f"Model loaded: {os.path.basename(model_path)}")
            else:
                messagebox.showerror("Error", f"Failed to load model: {model_name}")
    # Load custom model from file dialog
    def load_custom_model(self):
        model_dir = Path(__file__).resolve().parent / "model"
        filename = filedialog.askopenfilename(title="Select YOLO Model File", initialdir=str(model_dir), filetypes=[("YOLO model files", "*.pt")])
        if filename:
            self.gui.custom_model_path.set(filename)
            if self.gui.detector.load_model(filename):
                self.gui.model_label.config(text=f"Model: {os.path.basename(filename)}")
                self.gui.stats_panel.update_gpu_status()
                messagebox.showinfo("Success", f"Custom model loaded: {os.path.basename(filename)}")
                # Always insert full path for custom models
                if filename not in self.gui.model_listbox.get(0, tk.END):
                    self.gui.model_listbox.insert(tk.END, filename)
                idx = self.gui.model_listbox.get(0, tk.END).index(filename)
                self.gui.model_listbox.selection_clear(0, tk.END)
                self.gui.model_listbox.selection_set(idx)
                self.gui.model_listbox.see(idx)
                self.gui.selected_model.set(filename)
            else:
                messagebox.showerror("Error", f"Failed to load custom model: {os.path.basename(filename)}")
# GUI detection management class
class YOLODetection:
    def __init__(self, gui):
        self.gui = gui
        self.cap = None
        self.running = False
        self.processing_thread = None
    # Start detection based on selected source
    def start_detection(self):
        if self.gui.detector.model_manager.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        if self.running:
            return
        source_type = self.gui.selected_source.get()
        try:
            # Handle different source types
            if source_type == "media_device":
                selected = self.gui.selected_device.get()
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
                    devices = self.gui.media_device_combo['values']
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
                self.running = True
                self.gui.yolo_process.cap = self.cap
                self.gui.yolo_process.running = True
                self.gui.disable_source()
                self.gui.start_button.config(state=tk.DISABLED)
                self.gui.stop_button.config(state=tk.NORMAL)
                self.gui.status_label.config(text="Status: Running")
                source_text = self.gui.selected_device.get()
                self.gui.display_label.config(text=f"Source: {source_text}")
                self.gui.path_label.config(text=f"Source Path: {source_text}")
                self.processing_thread = threading.Thread(target=self.gui.yolo_process.process_video, daemon=True)
                self.processing_thread.start()
            # RTSP stream
            elif source_type == "rtsp":
                rtsp_url = self.gui.rtsp_url.get().strip()
                if not rtsp_url:
                    messagebox.showerror("Error", "Please enter RTSP URL")
                    return
                self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) 
                if not self.cap.isOpened():
                    messagebox.showerror("Error", f"Could not open RTSP stream: {rtsp_url}")
                    return
                self.running = True
                self.gui.yolo_process.cap = self.cap
                self.gui.yolo_process.running = True
                self.gui.disable_source()
                self.gui.start_button.config(state=tk.DISABLED)
                self.gui.stop_button.config(state=tk.NORMAL)
                self.gui.status_label.config(text="Status: Running")
                self.gui.display_label.config(text=f"Source: {rtsp_url}")
                self.gui.path_label.config(text=f"Source Path: {rtsp_url}")
                self.processing_thread = threading.Thread(target=self.gui.yolo_process.process_video, daemon=True)
                self.processing_thread.start()
            # Video file
            elif source_type == "video":
                if not self.gui.current_source:
                    messagebox.showerror("Error", "Please select a video file")
                    return
                self.cap = cv2.VideoCapture(self.gui.current_source)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not open video source")
                    return
                self.running = True
                self.gui.yolo_process.cap = self.cap
                self.gui.yolo_process.running = True
                self.gui.disable_source()
                self.gui.start_button.config(state=tk.DISABLED)
                self.gui.stop_button.config(state=tk.NORMAL)
                self.gui.status_label.config(text="Status: Running")
                self.gui.display_label.config(text=f"Source: {self.gui.current_source}")
                self.gui.path_label.config(text=f"Source Path: {self.gui.current_source}")
                self.processing_thread = threading.Thread(target=self.gui.yolo_process.process_video, daemon=True)
                self.processing_thread.start()
            # Image file
            elif source_type == "image":
                if not self.gui.current_source:
                    messagebox.showerror("Error", "Please select an image file")
                    return
                frame = cv2.imread(self.gui.current_source)
                if frame is None:
                    messagebox.showerror("Error", f"Could not load image: {self.gui.current_source}")
                    return
                self.gui.detector.reset_counts()
                annotated_frame, detections = self.gui.detector.objects(frame, self.gui.conf_threshold.get(), self.gui.nms_threshold.get())
                self.gui.yolo_process.display_frame(annotated_frame)
                self.gui.detection_display()
                messagebox.showinfo("Success", f"Processed image with {len(detections)} detections")
            # Image folder
            elif source_type == "folder":
                folder = self.gui.current_source
                if not folder or not os.path.isdir(folder):
                    messagebox.showerror("Error", "Please select a valid image folder")
                    return
                images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
                if not images:
                    messagebox.showerror("Error", "No images found in folder")
                    return
                result_dir = os.path.join(folder, "result")
                os.makedirs(result_dir, exist_ok=True)
                total = len(images)
                processed_count = 0
                self.gui.yolo_process.running = True
                self.gui.disable_source()
                self.gui.start_button.config(state=tk.DISABLED)
                self.gui.stop_button.config(state=tk.NORMAL)
                self.gui.status_label.config(text="Status: Running")
                self.processing_thread = threading.Thread(target=lambda: self.gui.yolo_process.process_folder(images, folder, result_dir, total), daemon=True)
                self.processing_thread.start()
            else:
                messagebox.showerror("Error", "Unknown source type")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Error starting detection: {e}")
    # Stop detection and release resources
    def stop_detection(self):
        if not self.running:
            return
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.gui.start_button.config(state=tk.NORMAL)
        self.gui.stop_button.config(state=tk.DISABLED)
        self.gui.clear_video_image()
        self.gui.status_label.config(text="Status: Stopped")
        self.gui.fps_label.config(text="FPS: 0")
        self.gui.enable_source()
# processing source class
class YOLOProcess:
    def __init__(self, gui):
        self.gui = gui
        self.cap = None
        self.gui.detection_manager.running = False
        self.processing_thread = None
    # Video processing loop
    def process_video(self):
        fps_counter, fps_start_time = 0, time.time()
        resize_width, resize_height = 1024, 768
        if self.cap is not None:
            source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not source_fps or source_fps <= 0 or source_fps > 120:
                source_fps = 30.0
        else:
            source_fps = 30.0
        frame_time = 1.0 / source_fps
        while self.gui.detection_manager.running and self.cap and self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (resize_width, resize_height))
            self.gui.detector.reset_counts()
            annotated_frame, detections = self.gui.detector.objects(frame, self.gui.conf_threshold.get(), self.gui.nms_threshold.get())
            self.gui.root.after(0, lambda f=annotated_frame: self.display_frame(f))
            self.gui.root.after(0, self.gui.detection_display)
            # Skip 2 frames
            for _ in range(2):
                if self.cap.isOpened():
                    self.cap.read()
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - fps_start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                self.gui.root.after(0, lambda f=fps: self.gui.fps_label.config(text=f"FPS: {f:.1f}"))
                fps_start_time = time.time()
            elapsed_time = time.time() - start_time
            sleep_time = frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        if self.gui.detection_manager.running:
            self.gui.root.after(0, self.gui.detection_manager.stop_detection)
    # Display frame in video label
    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_width = self.gui.video_label.winfo_width()
        label_height = self.gui.video_label.winfo_height()
        if label_width > 1 and label_height > 1:
            h, w = frame_rgb.shape[:2]
            aspect_ratio = w / h
            max_height = label_height - 10
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
            source_type = self.gui.selected_source.get()
            # Use source_type to determine correct label
            if source_type == "rtsp":
                # RTMP or RTSP
                if hasattr(self.gui, 'rtmp_enabled') and self.gui.rtmp_enabled.get() and self.gui.rtmp_url.get():
                    url = self.gui.rtmp_url.get().strip()
                else:
                    url = self.gui.rtsp_url.get().strip()
                self.gui.display_label.config(text=f"Source: {url}")
                self.gui.path_label.config(text=f"Source Path: {url}")
            elif source_type == "video" or source_type == "image" or source_type == "folder":
                if self.gui.current_source is not None:
                    file_name = os.path.basename(self.gui.current_source)
                    source_path = self.gui.current_source
                else:
                    file_name = "None"
                    source_path = "None"
                self.gui.display_label.config(text=f"Source: {file_name}")
                self.gui.path_label.config(text=f"Source Path: {source_path}")
            elif source_type == "media_device":
                device = self.gui.selected_device.get()
                self.gui.display_label.config(text=f"Source: {device}")
                self.gui.path_label.config(text=f"Source Path: {device}")
            else:
                self.gui.display_label.config(text="Source: None")
                self.gui.path_label.config(text="Source Path: None")
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        else:
            frame_resized = frame_rgb
        pil_image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(pil_image)
        self.gui.video_label.configure(image=photo)
        self.gui.current_photo = photo
    # Process image folder
    def process_folder(self, images, folder, result_dir, total):
        processed_count = 0
        self.gui.detection_manager.running = True
        for idx, img_name in enumerate(images):
            if not self.gui.detection_manager.running:
                break
            img_path = os.path.join(folder, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            self.gui.detector.reset_counts()
            annotated_frame, detections = self.gui.detector.objects(frame, self.gui.conf_threshold.get(), self.gui.nms_threshold.get())
            # Save result image with object classes drawn
            result_path = os.path.join(result_dir, img_name)
            cv2.imwrite(result_path, annotated_frame)
            processed_count += 1
            # Overlay percentage and annotation count for GUI preview only
            preview_frame = annotated_frame.copy()
            percent = int((idx+1)/total*100)
            percent_text = f"{percent}%"
            count_text = f"{idx+1}/{total}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, thickness = 1.0, 2
            # Percentage (top left)
            pt_x, pt_y = 10, 50
            text_size, _ = cv2.getTextSize(percent_text, font, font_scale, thickness)
            bg_rect = (pt_x-5, pt_y-text_size[1]-5, pt_x+text_size[0]+5, pt_y+10)
            cv2.rectangle(preview_frame, (bg_rect[0], bg_rect[1]), (bg_rect[2], bg_rect[3]), (0,0,0), -1)
            cv2.putText(preview_frame, percent_text, (pt_x, pt_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
            # Count (top right)
            img_w = preview_frame.shape[1]
            text_size2, _ = cv2.getTextSize(count_text, font, font_scale, thickness)
            ct_x, ct_y = img_w - text_size2[0] - 10, 50
            bg_rect2 = (ct_x-5, ct_y-text_size2[1]-5, ct_x+text_size2[0]+5, ct_y+10)
            cv2.rectangle(preview_frame, (bg_rect2[0], bg_rect2[1]), (bg_rect2[2], bg_rect2[3]), (0, 0, 0), -1)
            cv2.putText(preview_frame, count_text, (ct_x, ct_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            # Preview in GUI
            self.display_frame(preview_frame)
            self.gui.root.update_idletasks()
        def show_result():
            if processed_count == total:
                messagebox.showinfo("Success", f"Processed {processed_count} images. Results saved in result.")
            else:
                messagebox.showinfo("Stopped", f"Stopped early. {processed_count} images processed and saved in result.")
        self.gui.root.after(0, show_result)
    # Update RTMP URL
    def update_rtmp_url(self):
        mediamtx_path = os.path.join(os.path.dirname(__file__), 'mediamtx', 'mediamtx.exe')
        if self.gui.rtmp_enabled.get():
            # RTMP enabled: start mediamtx and set RTMP URL
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
            except Exception:
                ip = "127.0.0.1"
            rtsp_url = f"rtmp://{ip}:1935/live"
            self.gui.rtsp_url.set(rtsp_url)
            # Start mediamtx only if not already running
            if not hasattr(self.gui, 'mediamtx_proc') or self.gui.mediamtx_proc is None:
                try:
                    mediamtx_dir = os.path.join(os.path.dirname(__file__), 'mediamtx')
                    creationflags = 0
                    if sys.platform == 'win32':
                        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                    self.gui.mediamtx_proc = subprocess.Popen(
                        f'"{mediamtx_path}"', shell=True, cwd=mediamtx_dir, creationflags=creationflags)
                except Exception as e:
                    messagebox.showerror("mediamtx Error", f"Could not start mediamtx: {e}")
        else:
            # RTMP disabled: stop mediamtx and clear RTSP URL
            self.gui.rtsp_url.set("")
            if hasattr(self.gui, 'mediamtx_proc') and self.gui.mediamtx_proc:
                if sys.platform == 'win32':
                    self.gui.mediamtx_proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.gui.mediamtx_proc.send_signal(signal.SIGINT)
                self.gui.mediamtx_proc = None
            messagebox.showinfo("Success", "RTMP stopped")
# GUI statistics panel class
class YOLOStatsPanel:
    def __init__(self, gui):
        self.gui = gui
    def update_gpu_status(self):
        device_str = getattr(self.gui.detector.model_manager, 'device_str', None)
        if device_str and device_str != "Unknown":
            self.gui.gpu_label.config(text=f"Device: {device_str}")
        else:
            self.gui.gpu_label.config(text="Device: CPU")
    # Save statistics to file
    def save_statistics(self):
        try:
            filename = self.gui.detector.save_statistics()
            messagebox.showinfo("Success", f"Statistics saved to: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving statistics: {e}")
    # Reset statistics
    def reset_statistics(self):
        self.gui.detector.reset_source()
        self.gui.detection_display()
        messagebox.showinfo("Success", "Statistics have been cleared.")
# Main GUI class
class YOLOGui:
    def __init__(self):
        # Initialize main window and variables
        self.root = tk.Tk()
        self.root.title("YOLO Object Detection GUI")
        icon_path = Path(__file__).resolve().parent / "icon.ico"
        self.root.iconbitmap(str(icon_path))
        self.root.geometry("1480x730")
        self.root.minsize(1480, 750)
        self.root.configure(bg="#ffffff")
        self.cap, self.running = None, False
        self.current_source, self.processing_thread = None, None
        # Initialize detector and GUI components
        self.detector = YOLODetector()
        self.selected_source = tk.StringVar(value="media_device")
        self.selected_model = tk.StringVar()
        self.custom_model_path = tk.StringVar()
        self.rtsp_url = tk.StringVar()
        self.rtmp_url = tk.StringVar()
        self.rtmp_enabled = tk.BooleanVar(value=False)
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.nms_threshold = tk.DoubleVar(value=0.4)
        self.selected_device = tk.StringVar()
        self.source_manager = YOLOSource(self)
        self.detection_manager = YOLODetection(self)
        self.stats_panel = YOLOStatsPanel(self)
        self.yolo_process = YOLOProcess(self)
        self.setup_gui()
        # After GUI setup, refresh media devices and models
        self.source_manager.refresh_media_devices()
        self.source_manager.refresh_models()
    # Setup main GUI layout
    def setup_gui(self):
        # Setup main GUI layout: left, center, right panels
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        self.paned_window.bind('<B1-Motion>', lambda e: "break")
        self.paned_window.bind('<Button-1>', lambda e: "break")
        # Left Panel
        self.left_frame = ttk.Frame(self.paned_window, width=350)
        self.left_frame.pack_propagate(True)
        self.paned_window.add(self.left_frame, weight=0)
        self.setup_left_panel(self.left_frame)
        # Center Panel
        self.center_frame = ttk.Frame(self.paned_window)
        self.center_frame.pack_propagate(True)
        self.paned_window.add(self.center_frame, weight=1)
        self.setup_center_panel(self.center_frame)
        # Right Panel
        self.right_frame = ttk.Frame(self.paned_window, width=350)
        self.right_frame.pack_propagate(True)
        self.paned_window.add(self.right_frame, weight=0)
        self.setup_right_panel(self.right_frame)
    # Setup left panel components
    def setup_left_panel(self, parent):
        left_frame = ttk.LabelFrame(parent, text="Source Selection", padding=10)
        left_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 2), pady=(0, 0))
        # Input source selection
        ttk.Label(left_frame, text="Input Source:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        sources = ["Media Device", "RTSP Stream", "Video File", "Image File", "Folder"]
        values = ["media_device", "rtsp", "video", "image", "folder"]
        self.source_live_buttons = []
        for text, value in zip(sources, values):
            rb = ttk.Radiobutton(left_frame, text=text, variable=self.selected_source,value=value, command=self.source_manager.on_source_change)
            rb.pack(anchor=tk.W, pady=2)
            self.source_live_buttons.append(rb)
        self.source_frame = ttk.Frame(left_frame)
        self.source_frame.pack(fill=tk.X, pady=10)
        # RTSP input
        self.rtsp_frame = ttk.Frame(self.source_frame)
        rtsp_row = ttk.Frame(self.rtsp_frame)
        rtsp_row.pack(fill=tk.X)
        ttk.Label(rtsp_row, text="RTSP URL:").pack(side=tk.LEFT, anchor=tk.W)
        self.rtmp_check = ttk.Checkbutton(rtsp_row, text="RTMP", variable=self.rtmp_enabled, command=self.yolo_process.update_rtmp_url)
        self.rtmp_check.pack(side=tk.RIGHT, anchor=tk.E, padx=(10, 0))
        ttk.Entry(self.rtsp_frame, textvariable=self.rtsp_url, width=30).pack(fill=tk.X, pady=3)
        # Media device input
        self.media_device_frame = ttk.Frame(self.source_frame)
        ttk.Label(self.media_device_frame, text="Select Media Device:").pack(anchor=tk.W)
        self.media_device_combo = ttk.Combobox(
            self.media_device_frame, textvariable=self.selected_device,state="readonly")
        self.media_device_combo.pack(fill=tk.X, pady=3)
        # File input (video/image/folder)
        self.file_frame = ttk.Frame(self.source_frame)
        for txt, cmd in [
            ("Select Video File", self.source_manager.select_video_file),
            ("Select Image File", self.source_manager.select_image_file),
            ("Select Folder", self.source_manager.select_folder)
        ]:
            ttk.Button(self.file_frame, text=txt, command=cmd).pack(fill=tk.X)
        self.file_label = ttk.Label(self.file_frame, text="No file selected", foreground="blue")
        self.file_label.pack(fill=tk.X, pady=2)
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X)
        # Model selection
        ttk.Label(left_frame, text="Select Model:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        model_dir = Path(__file__).resolve().parent / "model"
        ttk.Label(left_frame, text=f"{model_dir}", font=('Arial', 10)).pack(anchor=tk.W, pady=(0, 2))
        self.model_listbox = tk.Listbox(left_frame, height=5, exportselection=False)
        self.model_listbox.pack(fill=tk.X, pady=2)
        self.model_listbox.bind('<<ListboxSelect>>', self.model_select)
        ttk.Button(left_frame, text="Load Custom Model", command=self.load_custom_model).pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, textvariable=self.custom_model_path, foreground="blue").pack(fill=tk.X, pady=2)
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X)
        # Detection settings
        ttk.Label(left_frame, text="Detection Settings:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        scale_width, label_width = 150, 18
        for label, var, cmd, val in [
            ("Confidence Threshold:", self.conf_threshold, self.update_conf, "50%"),
            ("NMS Threshold:", self.nms_threshold, self.update_nms, "40%")
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
        # Detection control buttons
        self.start_button = ttk.Button(left_frame, text="Start Detection", command=self.detection_manager.start_detection)
        self.start_button.pack(fill=tk.X, pady=5)
        self.stop_button = ttk.Button(left_frame, text="Stop Detection", command=self.detection_manager.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Clear source", command=self.clear_source).pack(fill=tk.X, pady=5)
        self.source_manager.on_source_change()
    # Setup center panel with video display and status
    def setup_center_panel(self, parent):
        self.source_display_frame = ttk.LabelFrame(parent, text="Source Display")  
        self.source_display_frame.pack(fill=tk.BOTH, expand=True)
        self.display_label = ttk.Label(self.source_display_frame, text="Source: None", font=('Arial', 12, 'bold'))
        self.display_label.pack(anchor=tk.NW, pady=(0, 10))
        self.video_box_frame = ttk.Frame(self.source_display_frame)
        self.video_box_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = tk.Label(
            self.video_box_frame, text="Video output will appear here", bg="#000000", fg="white", font=("Arial", 16), anchor="center"
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)
        # Bottom status/info frame: shows status, source path, and FPS
        self.bottom_status_frame = ttk.Frame(parent)
        self.bottom_status_frame.pack(fill=tk.X, pady=(10, 0))
        self.bottom_status_frame.columnconfigure(0, weight=1)
        self.bottom_status_frame.columnconfigure(1, weight=1)
        self.bottom_status_frame.columnconfigure(2, weight=1)
        # Status labels
        self.status_label = ttk.Label(self.bottom_status_frame, text="Status: Ready")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        self.path_label = ttk.Label(self.bottom_status_frame, text="Source Path: None")
        self.path_label.grid(row=0, column=1)
        self.fps_label = ttk.Label(self.bottom_status_frame, text="FPS: 0")
        self.fps_label.grid(row=0, column=2, sticky=tk.E)
    # Setup right panel with detection results and statistics
    def setup_right_panel(self, parent):
        right_frame = ttk.LabelFrame(parent, text="Detection Results")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        stats_frame = ttk.LabelFrame(right_frame, text="Session Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        self.model_label = ttk.Label(stats_frame, text="Model: None")
        self.model_label.pack(anchor=tk.W)
        self.gpu_label = ttk.Label(stats_frame, text="Device: Unknown")
        self.gpu_label.pack(anchor=tk.W)
        self.session_label = ttk.Label(stats_frame, text="Session Time: 00:00:00")
        self.session_label.pack(anchor=tk.W)
        self.detections_label = ttk.Label(stats_frame, text="Total Detections: 0")
        self.detections_label.pack(anchor=tk.W)
        export_frame = ttk.Frame(stats_frame)
        export_frame.pack(fill=tk.X, pady=5)
        # Buttons to save and clear stats
        ttk.Button(export_frame, text="Save Stats", command=self.stats_panel.save_statistics).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_frame, text="Clear Stats", command=self.stats_panel.reset_statistics).pack(side=tk.LEFT)
        ttk.Label(right_frame, text="Object Classes:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        list_frame = ttk.Frame(right_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        # Detection results Treeview
        columns = ('Class', 'Current', 'Total', 'Confidence')
        self.detection_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        for col in columns:
            self.detection_tree.heading(col, text=col)
            self.detection_tree.column(col, width=80, anchor='center')
        tree_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.detection_tree.yview)
        self.detection_tree.configure(yscrollcommand=tree_scrollbar.set)
        self.detection_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # A gap below the Treeview
        gap_frame = ttk.Frame(right_frame, height=10)
        gap_frame.pack(fill=tk.X, pady=(0, 10))
    # Update detection results display
    def detection_display(self):
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)
        detections = self.detector.get_current_detections()
        for class_name, stats in self.detector.stats_manager.detection_stats.items():
            current_count = stats['current_count']
            total_count = stats['total_count']
            avg_conf = stats['avg_confidence']
            self.detection_tree.insert('', 'end', values=(class_name, current_count, total_count, f"{avg_conf:.2f}"))
        session_stats = self.detector.get_session_stats()
        self.session_label.config(text=f"Session Time: {session_stats['sessionTime']}")
        self.detections_label.config(text=f"Total Detections: {session_stats['totalDetections']}")
    # Clear source selection and reset display
    def clear_source(self):
        self.detection_manager.stop_detection()
        self.detector.reset_source()
        self.detection_display()
        self.selected_source.set("media_device")
        self.selected_device.set("")
        self.rtsp_url.set(""); self.rtmp_url.set("")
        self.current_source = None
        self.file_label.config(text="No file selected")
        self.source_manager.on_source_change()
        self.display_label.config(text="Source: None")
        self.path_label.config(text="Source Path: None")
        self.clear_video_image(); self.enable_source()
        self.rtmp_enabled.set(False)
        if hasattr(self, 'mediamtx_proc') and self.mediamtx_proc:
            if sys.platform == 'win32':
                self.mediamtx_proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.mediamtx_proc.send_signal(signal.SIGINT)
            self.mediamtx_proc = None
    # Model selection
    def model_select(self, event):
        self.source_manager.model_select(event)
    # load custom model
    def load_custom_model(self):
        self.source_manager.load_custom_model()  
    # Update confidence threshold label
    def update_conf(self, event=None):
        percent = int(self.conf_threshold.get() * 100)
        self.conf_label.config(text=f"{percent}%")
    # Update NMS threshold label
    def update_nms(self, event=None):
        percent = int(self.nms_threshold.get() * 100)
        self.nms_label.config(text=f"{percent}%")
    # Disable source selection live buttons
    def disable_source(self):
        for rb in getattr(self, 'source_live_buttons', []):
            rb.config(state=tk.DISABLED)
    # Clear video image
    def clear_video_image(self):
        self.video_label.config(image='', text="Video output will appear here", bg="#000000", fg="white")
        self.current_photo = None
    # Enable source selection live buttons
    def enable_source(self):
        for rb in getattr(self, 'source_live_buttons', []):
            rb.config(state=tk.NORMAL)
    # Start the Tkinter main loop
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    def on_closing(self):
        if self.detection_manager.running:
            self.detection_manager.stop_detection()
        self.root.destroy()
# Run the GUI application
if __name__ == "__main__":
    YOLOGui().run()