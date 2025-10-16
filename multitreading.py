import os
import cv2
import time
import threading
import sys
import json
import torch
import socket
import subprocess
import numpy as np
import tkinter as tk
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None

# --- CORE LOGIC CLASSES ---

class YOLOModel:
    """Manages loading and handling of the YOLO model."""
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.model_path: Optional[str] = None
        self.class_names: List[str] = []
        self.device_str: str = "UNKNOWN"

    def load_model(self, model_path: str) -> bool:
        """Loads a YOLO model from the given path."""
        if not os.path.exists(model_path) or not model_path.endswith(".pt"):
            messagebox.showerror("Error", f"Invalid or non-existent model file: {model_path}")
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
            messagebox.showerror("Model Load Error", f"Error loading model: {e}")
            return False

class YOLOStats:
    """Manages detection statistics."""
    def __init__(self):
        self.detection_stats: Dict[str, Dict] = {}
        self.total_detections: int = 0
        self.session_start: Optional[datetime] = None

    def reset(self):
        """Resets all statistics for a new session."""
        self.detection_stats = {}
        self.total_detections = 0
        self.session_start = datetime.now()

    def update_detection_stats(self, class_name: str, confidence: float, bbox: List[int]):
        """Updates stats for a detected object."""
        if class_name not in self.detection_stats:
            self.detection_stats[class_name] = {
                'total_count': 0, 'current_count': 0, 'avg_confidence': 0.0,
                'confidence_sum': 0.0, 'last_bboxes': []
            }
        stats = self.detection_stats[class_name]
        stats['current_count'] += 1
        stats['confidence_sum'] += confidence
        stats['avg_confidence'] = stats['confidence_sum'] / (stats['total_count'] + stats['current_count'])

        # Check if detection is in a new location to update total_count
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        is_new_location = all(abs(cx - ((px1 + px2) // 2)) >= 100 or abs(cy - ((py1 + py2) // 2)) >= 100
                              for px1, py1, px2, py2 in stats['last_bboxes'])

        stats['last_bboxes'].append(bbox)
        stats['last_bboxes'] = stats['last_bboxes'][-20:]  # Keep last 20 bboxes

        if is_new_location:
            stats['total_count'] += 1
            self.total_detections += 1

    def reset_counts(self):
        """Resets only the current count for each class."""
        for stats in self.detection_stats.values():
            stats['current_count'] = 0

    def get_current_detections(self) -> List[Dict]:
        """Returns a list of classes detected in the current frame."""
        return [{'class': name, 'count': stats['current_count'], 'total': stats['total_count'], 'confidence': stats['avg_confidence']}
                for name, stats in self.detection_stats.items() if stats['current_count'] > 0]

    def get_session_stats(self, model_path: Optional[str] = None) -> Dict:
        """Returns a summary of the current session's statistics."""
        if self.session_start is None:
            self.session_start = datetime.now()
        session_time = datetime.now() - self.session_start
        return {
            'modelName': os.path.basename(model_path) if model_path else 'No model',
            'totalDetections': self.total_detections,
            'sessionTime': str(session_time).split('.')[0],
            'classStats': {name: stats['total_count'] for name, stats in self.detection_stats.items()}
        }

    def save_statistics(self, model_path: Optional[str] = None) -> str:
        """Saves session statistics to a JSON file."""
        save_dir = Path(__file__).resolve().parent / "save"
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = save_dir / f"yolo_stats_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_stats(stats):
            return {k: {sk: float(sv) if isinstance(sv, np.floating) else sv for sk, sv in v.items()} for k, v in stats.items()}

        stats_data = {
            'session_info': self.get_session_stats(model_path),
            'detailed_stats': convert_stats(self.detection_stats),
            'export_time': datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(stats_data, f, indent=2)
        return str(filename)

class YOLODetector:
    """Main class to orchestrate model, stats, and detection processing."""
    def __init__(self):
        self.model_manager = YOLOModel()
        self.stats_manager = YOLOStats()

    def load_model(self, model_path: str) -> bool:
        """Loads a model and resets stats."""
        if self.model_manager.load_model(model_path):
            self.stats_manager.reset()
            return True
        return False

    def detect_objects(self, frame: np.ndarray, conf: float, nms: float) -> Tuple[np.ndarray, List[Dict]]:
        """Performs object detection on a single frame."""
        model = self.model_manager.model
        if model is None:
            return frame, []
        
        try:
            results = model(frame, conf=conf, iou=nms, verbose=False)
            return self._process_results(frame, results[0])
        except Exception as e:
            print(f"Error during detection: {e}")
            return frame, []

    def _process_results(self, frame: np.ndarray, result) -> Tuple[np.ndarray, List[Dict]]:
        """Processes detection results and annotates the frame."""
        detections = []
        annotated_frame = frame.copy()
        class_names = self.model_manager.class_names

        if result.boxes is not None:
            for box, conf, cls_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = map(int, box)
                class_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
                
                self.stats_manager.update_detection_stats(class_name, conf, [x1, y1, x2, y2])
                
                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                detections.append({'class': class_name, 'confidence': float(conf), 'bbox': [x1, y1, x2, y2]})
        return annotated_frame, detections

# --- GUI HELPER CLASSES ---

class YOLOSource:
    """Manages GUI source selection logic."""
    def __init__(self, gui: 'YOLOGui'):
        self.gui = gui

    def refresh_media_devices(self):
        """Refreshes the list of available media devices."""
        devices = []
        if FilterGraph:
            try:
                devices = FilterGraph().get_input_devices()
            except Exception:
                pass # Fallback to probing if pygrabber fails
        if not devices:
            for i in range(5):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    devices.append(f"Device {i}")
                    cap.release()
        
        self.gui.media_device_combo['values'] = ["None"] + devices
        self.gui.selected_device.set("None")

    def refresh_models(self):
        """Refreshes the list of available .pt models."""
        model_dir = Path(__file__).resolve().parent / "model"
        model_dir.mkdir(exist_ok=True)
        models = sorted([p.name for p in model_dir.glob("*.pt")])
        self.gui.model_listbox.delete(0, tk.END)
        for model_name in models:
            self.gui.model_listbox.insert(tk.END, model_name)
        if models and not self.gui.selected_model.get():
            self.gui.model_listbox.selection_set(0)
            self.gui.model_select(None)
            
    def _select_file(self, title: str, filetypes: List[Tuple[str, str]]):
        """Generic file selection dialog."""
        filename = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if filename:
            self.gui.current_source = filename
            self.gui.file_label.config(text=os.path.basename(filename))
            return True
        return False
        
    def select_video_file(self):
        if self._select_file("Select Video File", [("Video files", "*.mp4 *.avi *.mov *.mkv")]):
             messagebox.showinfo("Success", f"Video file selected: {os.path.basename(self.gui.current_source)}")

    def select_image_file(self):
        if self._select_file("Select Image File", [("Image files", "*.jpg *.jpeg *.png *.bmp")]):
            frame = cv2.imread(self.gui.current_source)
            if frame is not None:
                self.gui.yolo_process.display_frame(frame)
            else:
                messagebox.showerror("Error", f"Could not load image: {self.gui.current_source}")

    def select_folder(self):
        """Folder selection dialog."""
        foldername = filedialog.askdirectory(title="Select Folder")
        if foldername:
            self.gui.current_source = foldername
            self.gui.file_label.config(text=os.path.basename(foldername))
            messagebox.showinfo("Success", f"Folder selected: {os.path.basename(self.gui.current_source)}")

    def on_source_change(self, _=None):
        """Handles UI changes when the source type is switched."""
        source_map = {
            "rtsp": (self.gui.rtsp_frame, None, None),
            "media_device": (self.gui.media_device_frame, None, None),
            "video": (self.gui.file_frame, self.select_video_file, "Select Video File"),
            "image": (self.gui.file_frame, self.select_image_file, "Select Image File"),
            "folder": (self.gui.file_frame, self.select_folder, "Select Folder"),
        }
        
        for frame, _, _ in source_map.values():
            if frame: frame.pack_forget()

        source_type = self.gui.selected_source.get()
        if source_type in source_map:
            frame_to_show, cmd, text = source_map[source_type]
            if frame_to_show:
                frame_to_show.pack(fill=tk.X, pady=5)
                if frame_to_show == self.gui.file_frame:
                    self.gui.file_button.config(text=text, command=cmd)
                    self.gui.file_label.pack(fill=tk.X, pady=(2, 0))

        if source_type == "media_device":
            self.refresh_media_devices()

class YOLODetection:
    """Manages the start/stop logic for the detection process."""
    def __init__(self, gui: 'YOLOGui'):
        self.gui = gui
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None
        
    def _start_video_processing(self, cap: cv2.VideoCapture, source_name: str):
        """Helper to initialize and start video processing thread."""
        if not cap or not cap.isOpened():
            messagebox.showerror("Error", f"Could not open video source: {source_name}")
            return

        self.cap = cap
        self.running = True
        self.gui.yolo_process.cap = self.cap
        self.gui.yolo_process.running = True

        self.gui.set_ui_state(running=True)
        self.processing_thread = threading.Thread(target=self.gui.yolo_process.process_video, daemon=True)
        self.processing_thread.start()

    def start_detection(self):
        """Starts detection based on the selected source."""
        if self.gui.detector.model_manager.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        if self.running:
            return

        source_type = self.gui.selected_source.get()
        
        try:
            if source_type == "media_device":
                selected = self.gui.selected_device.get()
                if not selected or selected == "None":
                    messagebox.showerror("Error", "No media device selected")
                    return
                # Determine device index from selection
                device_index = next((i for i, dev in enumerate(self.gui.media_device_combo['values']) if dev == selected), 0) -1
                if device_index < 0: return

                cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
                self._start_video_processing(cap, selected)

            elif source_type == "rtsp":
                rtsp_url = self.gui.rtsp_url.get().strip()
                if not rtsp_url:
                    messagebox.showerror("Error", "Please enter RTSP URL")
                    return
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                self._start_video_processing(cap, rtsp_url)

            elif source_type == "video":
                if not self.gui.current_source or not os.path.exists(self.gui.current_source):
                    messagebox.showerror("Error", "Please select a valid video file")
                    return
                cap = cv2.VideoCapture(self.gui.current_source)
                self._start_video_processing(cap, os.path.basename(self.gui.current_source))

            elif source_type == "image":
                if not self.gui.current_source or not os.path.exists(self.gui.current_source):
                    messagebox.showerror("Error", "Please select an image file")
                    return
                frame = cv2.imread(self.gui.current_source)
                if frame is None:
                    messagebox.showerror("Error", f"Could not load image: {self.gui.current_source}")
                    return
                
                self.gui.detector.stats_manager.reset_counts()
                annotated_frame, detections = self.gui.detector.detect_objects(frame, self.gui.conf_threshold.get(), self.gui.nms_threshold.get())
                self.gui.yolo_process.display_frame(annotated_frame)
                self.gui.update_detection_display()
                
                # Save annotated result
                photo_dir = Path(__file__).resolve().parent / "photo"
                photo_dir.mkdir(exist_ok=True)
                result_path = photo_dir / Path(self.gui.current_source).name
                cv2.imwrite(str(result_path), annotated_frame)
                messagebox.showinfo("Success", f"Processed image with {len(detections)} detections.")
            
            elif source_type == "folder":
                folder = self.gui.current_source
                if not folder or not os.path.isdir(folder):
                    messagebox.showerror("Error", "Please select a valid image folder")
                    return
                
                self.gui.set_ui_state(running=True)
                self.processing_thread = threading.Thread(target=self.gui.yolo_process.process_folder, daemon=True)
                self.processing_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Error starting detection: {e}")

    def stop_detection(self):
        """Stops the detection process and releases resources."""
        if not self.running:
            return
        self.running = False
        self.gui.yolo_process.running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.gui.set_ui_state(running=False)
        self.gui.clear_video_image()

class YOLOProcess:
    """Handles the actual video/image processing loops."""
    def __init__(self, gui: 'YOLOGui'):
        self.gui = gui
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False

    def process_video(self):
        """Main loop for processing video frames from a stream."""
        fps_counter, fps_start_time = 0, time.time()
        source_fps = (self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30.0) or 30.0
        frame_time = 1.0 / source_fps if 0 < source_fps < 120 else 1/30.0

        while self.running and self.cap and self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            
            self.gui.detector.stats_manager.reset_counts()
            annotated_frame, _ = self.gui.detector.detect_objects(frame, self.gui.conf_threshold.get(), self.gui.nms_threshold.get())
            
            self.gui.root.after(0, self.display_frame, annotated_frame)
            self.gui.root.after(0, self.gui.update_detection_display)
            
            # Frame skipping (read 2, process 1)
            for _ in range(2):
                if self.cap and self.cap.isOpened(): self.cap.grab()
            
            # FPS Calculation
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                self.gui.root.after(0, lambda f=fps: self.gui.fps_label.config(text=f"FPS: {f:.1f}"))
                fps_counter, fps_start_time = 0, time.time()

            # Maintain source FPS
            elapsed = time.time() - start_time
            if frame_time - elapsed > 0:
                time.sleep(frame_time - elapsed)

        if self.running:
            self.gui.root.after(0, self.gui.detection_manager.stop_detection)

    def process_folder(self):
        """Processes all images in a selected folder."""
        folder = self.gui.current_source
        images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if not images:
            messagebox.showerror("Error", "No images found in folder")
            self.gui.root.after(0, self.gui.detection_manager.stop_detection)
            return

        result_dir = Path(__file__).resolve().parent / "photo" / os.path.basename(folder)
        result_dir.mkdir(parents=True, exist_ok=True)
        total = len(images)
        
        for i, img_name in enumerate(images):
            if not self.running:
                break
            img_path = os.path.join(folder, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            self.gui.detector.stats_manager.reset_counts()
            annotated_frame, _ = self.gui.detector.detect_objects(frame, self.gui.conf_threshold.get(), self.gui.nms_threshold.get())
            cv2.imwrite(str(result_dir / img_name), annotated_frame)

            # Add progress overlay for display
            progress_text = f"Processing: {i+1}/{total} ({(i+1)/total*100:.1f}%)"
            cv2.putText(annotated_frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated_frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            self.display_frame(annotated_frame)
            self.gui.root.update_idletasks()
        
        messagebox.showinfo("Success", f"Processed {i+1} images. Results saved in 'photo' folder.")
        self.gui.root.after(0, self.gui.detection_manager.stop_detection)

    def display_frame(self, frame: np.ndarray):
        """Resizes and displays a frame in the GUI."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target_w, target_h = self.gui.video_label.winfo_width(), self.gui.video_label.winfo_height()
        if target_w <= 1 or target_h <= 1: return # Avoid division by zero

        h, w, _ = frame_rgb.shape
        aspect_ratio = w / h
        
        if target_w / target_h > aspect_ratio:
            new_h = target_h
            new_w = int(new_h * aspect_ratio)
        else:
            new_w = target_w
            new_h = int(new_w / aspect_ratio)

        resized_frame = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        pil_image = Image.fromarray(resized_frame)
        photo = ImageTk.PhotoImage(pil_image)
        self.gui.video_label.config(image=photo)
        self.gui.current_photo = photo # Keep a reference to avoid garbage collection

# --- MAIN GUI CLASS ---

class YOLOGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO Object Detection GUI")
        try:
            icon_path = Path(__file__).resolve().parent / "icon.ico"
            if icon_path.exists(): self.root.iconbitmap(str(icon_path))
        except tk.TclError:
            pass # Ignore if icon fails to load
            
        self.root.geometry("1500x700")
        self.root.minsize(1200, 600)

        # Core components
        self.detector = YOLODetector()
        self.source_manager = YOLOSource(self)
        self.detection_manager = YOLODetection(self)
        self.yolo_process = YOLOProcess(self)
        self.mediamtx_proc: Optional[subprocess.Popen] = None
        self.current_source: Optional[str] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        # Style and variables
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom styles to remove grey background
        self.style.configure('.', background='white', foreground='black')
        self.style.configure('TFrame', background='white')
        self.style.configure('TLabel', background='white', foreground='black')
        self.style.configure('TRadiobutton', background='white', foreground='black')
        self.style.configure('TCheckbutton', background='white', foreground='black')
        self.style.configure('TLabelFrame', background='white')
        self.style.configure('TLabelFrame.Label', background='white', foreground='black')
        self.root.configure(bg='white')
        
        self._setup_variables()
        
        # Build UI
        self._setup_gui()
        
        # Initial state
        self.source_manager.refresh_models()
        self.source_manager.on_source_change()

    def _setup_variables(self):
        """Initializes all Tkinter variables."""
        self.selected_source = tk.StringVar(value="media_device")
        self.selected_model = tk.StringVar()
        self.rtsp_url = tk.StringVar()
        self.rtmp_enabled = tk.BooleanVar(value=False)
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.nms_threshold = tk.DoubleVar(value=0.4)
        self.selected_device = tk.StringVar()
        self.auto_save_frame = tk.BooleanVar(value=False)

    def _setup_gui(self):
        """Builds the main GUI layout."""
        # Main layout: Paned window
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Panel (Controls)
        self.left_frame = ttk.Frame(main_pane, width=350)
        self._setup_left_panel(self.left_frame)
        main_pane.add(self.left_frame, weight=0)

        # Center Panel (Video Display)
        center_frame = ttk.Frame(main_pane)
        self._setup_center_panel(center_frame)
        main_pane.add(center_frame, weight=1)

        # Right Panel (Stats)
        right_frame = ttk.Frame(main_pane, width=350)
        self._setup_right_panel(right_frame)
        main_pane.add(right_frame, weight=0)
        
    def _setup_left_panel(self, parent: ttk.Frame):
        """Builds the control panel on the left."""
        # Source Selection
        source_frame = ttk.LabelFrame(parent, text="Source Selection", padding=10)
        source_frame.pack(fill=tk.X, pady=5)
        
        sources = {"Media Device": "media_device", "RTSP Stream": "rtsp", "Video File": "video", "Image File": "image", "Folder": "folder"}
        for text, value in sources.items():
            ttk.Radiobutton(source_frame, text=text, variable=self.selected_source, value=value, command=self.source_manager.on_source_change).pack(anchor=tk.W)

        # Dynamic Source Options Frame
        self.rtsp_frame = ttk.Frame(source_frame)
        ttk.Label(self.rtsp_frame, text="RTSP URL:").pack(anchor=tk.W)
        ttk.Entry(self.rtsp_frame, textvariable=self.rtsp_url).pack(fill=tk.X, expand=True)

        self.media_device_frame = ttk.Frame(source_frame)
        ttk.Label(self.media_device_frame, text="Select Device:").pack(anchor=tk.W)
        self.media_device_combo = ttk.Combobox(self.media_device_frame, textvariable=self.selected_device, state="readonly")
        self.media_device_combo.pack(fill=tk.X, expand=True)
        
        self.file_frame = ttk.Frame(source_frame)
        self.file_button = ttk.Button(self.file_frame, text="Select File...")
        self.file_button.pack(fill=tk.X)
        self.file_label = ttk.Label(self.file_frame, text="No file selected", foreground="black")
        self.file_label.pack(fill=tk.X)

        # Model Selection
        model_frame = ttk.LabelFrame(parent, text="Model Selection", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        self.model_listbox = tk.Listbox(model_frame, height=5, exportselection=False, bg="white", fg="black", selectbackground="#cce6ff")
        self.model_listbox.pack(fill=tk.X, expand=True, pady=2)
        self.model_listbox.bind('<<ListboxSelect>>', self.model_select)
        ttk.Button(model_frame, text="Load Custom Model", command=self.load_custom_model).pack(fill=tk.X, pady=2)

        # Detection Settings
        settings_frame = ttk.LabelFrame(parent, text="Detection Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Confidence Threshold
        conf_row = ttk.Frame(settings_frame)
        conf_row.pack(fill=tk.X, pady=2)
        ttk.Label(conf_row, text="Confidence:", width=12).pack(side=tk.LEFT)
        ttk.Scale(conf_row, from_=0.0, to=1.0, variable=self.conf_threshold, orient=tk.HORIZONTAL, command=self.update_threshold_labels).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.conf_label = ttk.Label(conf_row, text="50%", width=5)
        self.conf_label.pack(side=tk.LEFT)

        # NMS Threshold
        nms_row = ttk.Frame(settings_frame)
        nms_row.pack(fill=tk.X, pady=2)
        ttk.Label(nms_row, text="NMS:", width=12).pack(side=tk.LEFT)
        ttk.Scale(nms_row, from_=0.0, to=1.0, variable=self.nms_threshold, orient=tk.HORIZONTAL, command=self.update_threshold_labels).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.nms_label = ttk.Label(nms_row, text="40%", width=5)
        self.nms_label.pack(side=tk.LEFT)
        
        # Control Buttons
        control_frame = ttk.Frame(parent, padding=10)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.start_button = ttk.Button(control_frame, text="Start Detection", command=self.detection_manager.start_detection)
        self.start_button.pack(fill=tk.X, pady=2)
        self.stop_button = ttk.Button(control_frame, text="Stop Detection", command=self.detection_manager.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)

    def _setup_center_panel(self, parent: ttk.Frame):
        """Builds the video display panel."""
        display_frame = ttk.LabelFrame(parent, text="Live Feed", padding=10)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(display_frame, bg="black", text="Video output will appear here", fg="white", font=("Arial", 16))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Status Bar
        status_bar = ttk.Frame(parent, padding=(5, 2))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = ttk.Label(status_bar, text="Status: Ready")
        self.status_label.pack(side=tk.LEFT)
        self.fps_label = ttk.Label(status_bar, text="FPS: 0.0")
        self.fps_label.pack(side=tk.RIGHT)

    def _setup_right_panel(self, parent: ttk.Frame):
        """Builds the statistics and results panel."""
        # Session Stats
        stats_frame = ttk.LabelFrame(parent, text="Session Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=5)
        self.model_label = ttk.Label(stats_frame, text="Model: None")
        self.model_label.pack(anchor=tk.W)
        self.gpu_label = ttk.Label(stats_frame, text="Device: Unknown")
        self.gpu_label.pack(anchor=tk.W)
        self.session_label = ttk.Label(stats_frame, text="Session Time: 00:00:00")
        self.session_label.pack(anchor=tk.W)
        self.detections_label = ttk.Label(stats_frame, text="Total Detections: 0")
        self.detections_label.pack(anchor=tk.W)

        # Action Buttons
        actions_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=5)
        ttk.Button(actions_frame, text="Save Stats", command=self.save_statistics).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Open Results Folder", command=lambda: self.open_folder("photo")).pack(fill=tk.X, pady=2)
        ttk.Checkbutton(actions_frame, text="Auto-Save Frame on Detection", variable=self.auto_save_frame).pack(anchor=tk.W, pady=2)
        
        # Detection Results Treeview
        results_frame = ttk.LabelFrame(parent, text="Detection Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        columns = ('Class', 'Current', 'Total', 'Confidence')
        self.detection_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)
        for col in columns:
            self.detection_tree.heading(col, text=col)
            self.detection_tree.column(col, width=80, anchor='center')
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.detection_tree.yview)
        self.detection_tree.configure(yscrollcommand=scrollbar.set)
        
        self.detection_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def run(self):
        """Starts the Tkinter main loop."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    # --- GUI Callbacks and Helpers ---

    def model_select(self, _=None):
        """Handles model selection from the listbox."""
        selection = self.model_listbox.curselection()
        if not selection: return
        
        model_name = self.model_listbox.get(selection[0])
        model_path = str(Path(__file__).resolve().parent / "model" / model_name)
        
        if self.detector.load_model(model_path):
            self.selected_model.set(model_name)
            self.model_label.config(text=f"Model: {model_name}")
            self.gpu_label.config(text=f"Device: {self.detector.model_manager.device_str}")
            messagebox.showinfo("Success", f"Model '{model_name}' loaded successfully.")
        else:
            self.selected_model.set("")
            self.model_label.config(text="Model: None")

    def load_custom_model(self):
        """Opens a file dialog to load a custom .pt model."""
        model_path = filedialog.askopenfilename(
            title="Select YOLO Model", filetypes=[("PyTorch Models", "*.pt")]
        )
        if model_path and self.detector.load_model(model_path):
            model_name = os.path.basename(model_path)
            self.model_label.config(text=f"Model: {model_name}")
            self.gpu_label.config(text=f"Device: {self.detector.model_manager.device_str}")
            # Add to listbox if not present
            if model_name not in self.model_listbox.get(0, tk.END):
                self.model_listbox.insert(tk.END, model_name)
            messagebox.showinfo("Success", f"Custom model '{model_name}' loaded.")

    def update_detection_display(self):
        """Updates the detection treeview and session stats labels."""
        self.detection_tree.delete(*self.detection_tree.get_children())
        detections = self.detector.stats_manager.get_current_detections()
        
        for det in sorted(detections, key=lambda x: x['class']):
            self.detection_tree.insert('', 'end', values=(det['class'], det['count'], det['total'], f"{det['confidence']:.2f}"))

        # Auto-save frame logic
        if self.auto_save_frame.get() and detections:
            self.save_frame()
            
        session_stats = self.detector.stats_manager.get_session_stats(self.detector.model_manager.model_path)
        self.session_label.config(text=f"Session Time: {session_stats['sessionTime']}")
        self.detections_label.config(text=f"Total Detections: {session_stats['totalDetections']}")
        
    def save_statistics(self):
        """Saves session stats to a file."""
        try:
            filename = self.detector.stats_manager.save_statistics(self.detector.model_manager.model_path)
            messagebox.showinfo("Success", f"Statistics saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save statistics: {e}")

    def save_frame(self):
        """Saves the current displayed frame as an image."""
        if not self.current_photo: return
        
        save_dir = Path(__file__).resolve().parent / "photo"
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = save_dir / f"capture_{timestamp}.png"
        
        # Convert PhotoImage back to a savable format
        image = self.current_photo._PhotoImage__photo.subsample(1, 1)
        image.write(str(filename), format="png")
        
    def open_folder(self, folder_name: str):
        """Opens a specified subfolder in the file explorer (Windows only)."""
        path = Path(__file__).resolve().parent / folder_name
        path.mkdir(exist_ok=True)
        os.startfile(path)

    def update_threshold_labels(self, _=None):
        """Updates the percentage labels for threshold scales."""
        self.conf_label.config(text=f"{int(self.conf_threshold.get() * 100)}%")
        self.nms_label.config(text=f"{int(self.nms_threshold.get() * 100)}%")
        
    def set_ui_state(self, running: bool):
        """Enables/disables UI elements based on running state."""
        state = tk.DISABLED if running else tk.NORMAL
        self.start_button.config(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL if running else tk.DISABLED)
        self.status_label.config(text=f"Status: {'Running' if running else 'Stopped'}")
        
        for widget in self.left_frame.winfo_children():
            self._toggle_widget_state_recursively(widget, state)

    def _toggle_widget_state_recursively(self, parent_widget, state):
        """Recursively sets the state for all child widgets that support it."""
        # A tuple of widget classes that are known to have a 'state' option.
        STATEFUL_WIDGETS = (
            ttk.Button, ttk.Radiobutton, ttk.Checkbutton,
            ttk.Entry, ttk.Combobox, ttk.Scale,
            tk.Listbox
        )
        
        if isinstance(parent_widget, STATEFUL_WIDGETS):
            if parent_widget != self.stop_button:
                try:
                    parent_widget.configure(state=state)
                except tk.TclError:
                    pass

        for child in parent_widget.winfo_children():
            self._toggle_widget_state_recursively(child, state)

    def clear_video_image(self):
        """Resets the video display label."""
        self.video_label.config(image='', text="Video output will appear here")
        self.current_photo = None
        self.fps_label.config(text="FPS: 0.0")

    def on_closing(self):
        """Handles the application closing event."""
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
            if self.detection_manager.running:
                self.detection_manager.stop_detection()
            self.root.destroy()

if __name__ == "__main__":
    app = YOLOGui()
    app.run()

