import os,cv2,time,threading,sys,json,torch,socket,subprocess
import numpy as np
import tkinter as tk
from pathlib import Path
from datetime import datetime
from typing import Dict,List,Optional,Tuple
from tkinter import ttk,filedialog,messagebox
from PIL import Image,ImageTk
from ultralytics import YOLO
import queue # Added for thread-safe task management
try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None

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
        # Update running average confidence
        current_total = stats['total_count']
        current_sum = stats['avg_confidence'] * current_total
        new_total = current_total + 1
        stats['avg_confidence'] = (current_sum + confidence) / new_total
        stats['confidence_sum'] += confidence
        
        # Logic to check if a detection is a "new" encounter (imperfect, but preserved)
        is_new_location = True
        if bbox is not None:
            # Check if this bbox is close to any previous bbox (center distance < 100px)
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for prev_bbox in stats['last_bboxes']:
                px1, py1, px2, py2 = prev_bbox
                pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                # Calculate distance squared to avoid sqrt
                dist_sq = (cx - pcx)**2 + (cy - pcy)**2 
                if dist_sq < 10000: # 100px threshold squared
                    is_new_location = False
                    break
            stats['last_bboxes'].append(bbox)
            if len(stats['last_bboxes']) > 20:
                stats['last_bboxes'] = stats['last_bboxes'][-20:]
        
        # Only increment total_count if it seems like a new object/location encounter
        if is_new_location:
            stats['total_count'] += 1
            self.total_detections += 1
    # Reset current detection counts (needed for per-frame updates in video)
    def reset_counts(self):
        [stats.update({'current_count': 0}) for stats in self.detection_stats.values()]
    # Get current detections
    def get_current_detections(self) -> List[Dict]:
        detections = []
        for class_name, stats in self.detection_stats.items():
            # Only include classes that are currently detected (current_count > 0)
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
            # NOTE: self.stats_manager.reset_counts() is handled by YOLOProcess before calling this for video streams
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
                # Update stats with individual detection
                self.stats_manager.update_detection_stats(class_name, conf, [int(x1), int(y1), int(x2), int(y2)])
                
                # Draw bounding box and label
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

# Worker thread for batch detection
class Worker(threading.Thread):
    def __init__(self, task_queue, result_list, model_path, conf_threshold, nms_threshold):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.result_list = result_list
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.daemon = True # Allows program to exit if threads are running

        # Each worker thread gets its own detector and model instance
        self.detector = YOLODetector()
        self.detector.load_model(model_path)

    def run(self):
        while True:
            try:
                # Get task from queue (waits up to 1 second)
                img_path = self.task_queue.get(timeout=1)
            except queue.Empty:
                # No more tasks, thread exits
                break
                
            try:
                # Perform detection (assuming single image processing)
                frame = cv2.imread(img_path)
                if frame is None:
                    self.result_list.append((img_path, "Error: Could not load image"))
                    continue

                # Run detection
                _, detections = self.detector.objects(
                    frame, self.conf_threshold, self.nms_threshold
                )
                
                # Simple summary of results
                detection_summary = f"Detections: {len(detections)}"
                if detections:
                    # Get class counts for a more detailed summary
                    class_counts = {}
                    for d in detections:
                        class_counts[d['class']] = class_counts.get(d['class'], 0) + 1
                    detection_summary += " (" + ", ".join([f"{c}: {n}" for c, n in class_counts.items()]) + ")"
                
                self.result_list.append((img_path, detection_summary))

            except Exception as e:
                self.result_list.append((img_path, f"Error processing: {e}"))
                
            finally:
                self.task_queue.task_done()

# Manager class for multi-threaded batch processing
class MultiThreadDetectionManager:
    def __init__(self, model_path, conf_threshold, nms_threshold):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        # Limit threads to a reasonable number, e.g., 8 or CPU count
        self.num_threads = min(8, os.cpu_count() or 1) 
        self.task_queue = queue.Queue()
        self.results = []
        self.workers = []
        
    def run(self, sources: List[str]):
        self.results = []
        self.workers = []

        # 1. Populate the task queue with image paths
        for src in sources:
            self.task_queue.put(src)

        # 2. Create and start worker threads
        for _ in range(self.num_threads):
            worker = Worker(self.task_queue, self.results, self.model_path, 
                            self.conf_threshold, self.nms_threshold)
            self.workers.append(worker)
            worker.start()

        # 3. Wait for all tasks to be completed
        self.task_queue.join() 
        
        # 4. Wait for all workers to finish their last task and exit gracefully
        for worker in self.workers:
            # Join with a timeout to prevent potential deadlocks if a thread is stuck
            worker.join(timeout=2) 

        return self.results

# GUI source management class
class YOLOSource:
# ... (rest of YOLOSource is unchanged)
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
                # Fallback check for common indices
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
            self.model_select(None)
    # Select video file
    def select_video_file(self):
        filename = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")])
        if filename:
            self.gui.current_source = filename
            self.gui.file_label.config(text=os.path.basename(filename))
            messagebox.showinfo("Success", f"Video file selected: {os.path.basename(filename)}")
    # Select image file
    def select_image_file(self):
        filename = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if filename:
            self.gui.current_source = filename
            self.gui.file_label.config(text=os.path.basename(filename))
            frame = cv2.imread(filename)
            if frame is not None:
                self.gui.yolo_process.display_frame(frame)
            else:
                messagebox.showerror("Error", f"Could not load image: {filename}")
    def select_folder(self):
        foldername = filedialog.askdirectory(title="Select Folder")
        if foldername:
            self.gui.current_source = foldername
            self.gui.file_label.config(text=os.path.basename(foldername))
            messagebox.showinfo("Success", f"Folder selected: {os.path.basename(foldername)}")
    # Handle source type change
    def on_source_change(self):
        # Hide all source-specific frames initially
        for frame in [self.gui.rtsp_frame, self.gui.file_frame, self.gui.media_device_frame]:
            frame.pack_forget()
        source = self.gui.selected_source.get()
        # Display frame from the RTSP stream
        if source == "rtsp":
            self.gui.rtsp_frame.pack(fill=tk.X, pady=5)
        # Handle file-based sources (video, image, folder)
        elif source in ["video", "image", "folder"]:
            self.gui.file_frame.pack(fill=tk.X, pady=5)
            # Map source type to button text
            button_texts = {
                "video": "Select Video File",
                "image": "Select Image File",
                "folder": "Select Folder"
            }
            selected_text = button_texts[source]
            
            # Show only the relevant button
            for child in self.gui.file_frame.winfo_children():
                if isinstance(child, ttk.Button):
                    if child.cget('text') == selected_text:
                        child.pack(fill=tk.X, pady=2)
                    else:
                        child.pack_forget()
            
            # Update file label visibility
            self.gui.file_label.pack_forget()
            self.gui.file_label.pack(fill=tk.X, pady=(2, 0))

            # Update file label content based on selection
            if self.gui.current_source and os.path.isdir(self.gui.current_source):
                if source == "folder":
                    images = [f for f in os.listdir(self.gui.current_source) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
                    if images:
                        self.gui.file_label.config(text=f"{len(images)} images found in {os.path.basename(self.gui.current_source)}")
                    else:
                        self.gui.file_label.config(text=f"No images found in {os.path.basename(self.gui.current_source)}")
                else: # video or image selected
                    self.gui.file_label.config(text=os.path.basename(self.gui.current_source))
            elif self.gui.current_source and not os.path.isdir(self.gui.current_source):
                self.gui.file_label.config(text=os.path.basename(self.gui.current_source))
            else:
                self.gui.file_label.config(text="No file/folder selected")

        # Display frame from the selected media device
        elif source == "media_device":
            self.gui.media_device_frame.pack(fill=tk.X, pady=5)
            self.refresh_media_devices()
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
    
    # Helper to set common UI state for starting a live/video stream
    def _start_capture_stream_ui_update(self, source_text: str):
        self.running = True
        self.gui.yolo_process.cap = self.cap
        self.gui.yolo_process.running = True
        self.gui.disable_source()
        self.gui.start_button.config(state=tk.DISABLED)
        self.gui.stop_button.config(state=tk.NORMAL)
        self.gui.status_label.config(text="Status: Running")
        self.gui.display_label.config(text=f"Source: {source_text}")
        self.gui.path_label.config(text=f"Source Path: {source_text}")
        self.processing_thread = threading.Thread(target=self.gui.yolo_process.process_video, daemon=True)
        self.processing_thread.start()

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
                # Device Index Resolution Logic (kept simple)
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
                    messagebox.showerror("Error", f"Could not open device source: {selected}")
                    return
                self._start_capture_stream_ui_update(selected)
            # RTSP stream
            elif source_type == "rtsp":
                rtsp_url = self.gui.rtsp_url.get().strip()
                if not rtsp_url:
                    messagebox.showerror("Error", "Please enter RTSP URL")
                    return
                rtsp_tcp = f"{rtsp_url}?rtsp_transport=tcp"
                self.cap = cv2.VideoCapture(rtsp_tcp, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                if not self.cap.isOpened():
                    messagebox.showerror("Error", f"Could not open RTSP stream: {rtsp_url}")
                    return
                self._start_capture_stream_ui_update(rtsp_url)
            # Video file
            elif source_type == "video":
                if not self.gui.current_source:
                    messagebox.showerror("Error", "Please select a video file")
                    return
                self.cap = cv2.VideoCapture(self.gui.current_source)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not open video source")
                    return
                self._start_capture_stream_ui_update(self.gui.current_source)
            # Image file (single-frame processing)
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
                # Save annotated result in photo/<image_name>
                image_path = Path(self.gui.current_source)
                image_name = image_path.name
                photo_dir = Path(__file__).resolve().parent / "photo"
                photo_dir.mkdir(parents=True, exist_ok=True)
                result_path = photo_dir / image_name
                cv2.imwrite(str(result_path), annotated_frame)
                messagebox.showinfo("Success", f"Processed image with {len(detections)} detections.")
            # Image folder (batch processing)
            elif source_type == "folder":
                folder = self.gui.current_source
                if not folder or not os.path.isdir(folder):
                    messagebox.showerror("Error", "Please select a valid image folder")
                    return
                images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
                if not images:
                    messagebox.showerror("Error", "No images found in folder")
                    return
                
                photo_dir = Path(__file__).resolve().parent / "photo"
                folder_name = os.path.basename(folder)
                result_dir = photo_dir / folder_name
                result_dir.mkdir(parents=True, exist_ok=True)
                total = len(images)
                
                # Set running state before starting thread
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
        self.gui.yolo_process.running = False # Stop the processing loop
        # Release video capture safely
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                pass
            self.cap = None
        self.gui.start_button.config(state=tk.NORMAL)
        self.gui.stop_button.config(state=tk.DISABLED)
        self.gui.clear_video_image()
        self.gui.status_label.config(text="Status: Stopped")
        self.gui.fps_label.config(text="FPS: 0")
        self.gui.enable_source()

# processing source class
class YOLOProcess:
# ... (rest of YOLOProcess is unchanged)
    def __init__(self, gui):
        self.gui = gui
        self.cap = None
        self.running = False
        self.processing_thread = None
    # Video processing loop
    def process_video(self):
        fps_counter, fps_start_time = 0, time.time()
        source_fps = 30.0 # Default fallback
        if self.cap is not None:
            # Safely check for source FPS
            try:
                source_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if not source_fps or source_fps <= 0 or source_fps > 120:
                    source_fps = 30.0
            except Exception:
                source_fps = 30.0
                
        frame_time = 1.0 / source_fps
        while self.gui.detection_manager.running and self.cap and self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Loop video for file source
                if self.gui.selected_source.get() == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        break
                else:
                    break
            
            # Reset current counts for new frame detection (needed for 'Current' column)
            self.gui.detector.reset_counts()
            
            annotated_frame, detections = self.gui.detector.objects(frame, self.gui.conf_threshold.get(), self.gui.nms_threshold.get())
            self.gui.root.after(0, lambda f=annotated_frame: self.display_frame(f))
            self.gui.root.after(0, self.gui.detection_display)
            
            # Skip 2 frames safely (for performance enhancement on lower frame rates)
            for _ in range(2):
                if self.cap.isOpened():
                    ret_skip, _ = self.cap.read()
                    if not ret_skip:
                        break
                        
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
            # Auto-stop if loop broke (e.g., file end, stream error)
            self.gui.root.after(0, self.gui.detection_manager.stop_detection)
            
    # Display frame in video label
    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target_w = self.gui.video_label.winfo_width()
        target_h = self.gui.video_label.winfo_height()
        if target_w == 0 or target_h == 0:
            return
            
        h, w = frame_rgb.shape[:2]
        src_aspect = w / h
        disp_aspect = target_w / target_h
        
        # Calculate new dimensions while maintaining aspect ratio
        if src_aspect > disp_aspect:
            new_w = target_w
            new_h = int(target_w / src_aspect)
        else:
            new_h = target_h
            new_w = int(target_h * src_aspect)
            
        frame_scaled = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calculate padding to center the image
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        
        # Add black borders for centering
        frame_resized = cv2.copyMakeBorder(
            frame_scaled, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        pil_image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(pil_image)
        self.gui.video_label.configure(image=photo)
        self.gui.current_photo = photo
        
        # Update source display labels
        source_type = self.gui.selected_source.get()
        source_path = ""
        source_name = "None"
        
        if source_type == "rtsp":
            url = self.gui.rtmp_url.get().strip() if (hasattr(self.gui, 'rtmp_enabled') and self.gui.rtmp_enabled.get() and self.gui.rtmp_url.get()) else self.gui.rtsp_url.get().strip()
            source_path = url
            source_name = url
        elif source_type in ("video", "image", "folder") and self.gui.current_source:
            source_path = self.gui.current_source
            source_name = os.path.basename(source_path)
        elif source_type == "media_device":
            device = self.gui.selected_device.get()
            source_path = device
            source_name = device

        self.gui.display_label.config(text=f"Source: {source_name}")
        self.gui.path_label.config(text=f"Source Path: {source_path}")

    # Process image folder
    def process_folder(self, images, folder, result_dir, total):
        processed_count = 0
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
            result_path = str(result_dir / img_name)
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
            cv2.rectangle(preview_frame, (pt_x-5, pt_y-text_size[1]-5), (pt_x+text_size[0]+5, pt_y+10), (0,0,0), -1)
            cv2.putText(preview_frame, percent_text, (pt_x, pt_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
            
            # Count (top right)
            img_w = preview_frame.shape[1]
            text_size2, _ = cv2.getTextSize(count_text, font, font_scale, thickness)
            ct_x, ct_y = img_w - text_size2[0] - 10, 50
            cv2.rectangle(preview_frame, (ct_x-5, ct_y-text_size2[1]-5), (ct_x+text_size2[0]+5, ct_y+10), (0, 0, 0), -1)
            cv2.putText(preview_frame, count_text, (ct_x, ct_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            self.display_frame(preview_frame)
            self.gui.root.update_idletasks() # Force UI update

        def show_result():
            if processed_count == total:
                messagebox.showinfo("Success", f"Processed {processed_count} images. Results saved in {result_dir.name} folder.")
            else:
                messagebox.showinfo("Stopped", f"Stopped early. {processed_count} images processed and saved in {result_dir.name} folder.")
                
        # Cleanup
        self.gui.root.after(0, self.gui.detection_manager.stop_detection)
        self.gui.root.after(0, show_result)
        
    # Update RTMP URL (used for local RTMP push from media device/RTSP source)
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
                    messagebox.showinfo("Success", f"RTMP address set at {rtsp_url}. Push your stream here.")
                except Exception as e:
                    messagebox.showerror("mediamtx Error", f"Could not start mediamtx: {e}")
        else:
            # RTMP disabled: stop mediamtx and clear RTSP URL
            self.gui.rtsp_url.set("")
            if hasattr(self.gui, 'mediamtx_proc') and self.gui.mediamtx_proc:
                if sys.platform == 'win32':
                    try:
                        # Attempt to gracefully stop mediamtx
                        subprocess.run(["taskkill", "/F", "/IM", "mediamtx.exe"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                        )
                    except subprocess.CalledProcessError:
                        pass
            self.gui.mediamtx_proc = None
            messagebox.showinfo("Success", "RTMP stopped")

# GUI statistics panel class
class YOLOStatsPanel:
# ... (rest of YOLOStatsPanel is unchanged)
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
            messagebox.showinfo("Success", f"Statistics saved to save stats folder as {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving statistics: {e}")
    # Reset statistics
    def reset_statistics(self):
        self.gui.detector.reset_source()
        self.gui.detection_display()
        messagebox.showinfo("Success", "Statistics have been cleared.")
    # Save current frame with detection
    def save_frame(self):
        frame = getattr(self.gui, 'current_photo', None)
        detections = self.gui.detector.get_current_detections()
        if frame is None or not detections:
            return
            
        source_name = "auto"
        if hasattr(self.gui, 'current_source') and self.gui.current_source:
            source_path = Path(self.gui.current_source)
            source_name = source_path.stem
        
        save_dir = Path(__file__).resolve().parent / "photo" / source_name
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = str(save_dir / f"{timestamp}.png")
        
        # Use a safe way to save the PhotoImage content (if available)
        try:
            # tkinter PhotoImage object may have a write method
            frame._PhotoImage__photo.write(filename, format="png")
        except Exception:
            # Fallback (may not work in all environments)
            messagebox.showerror("Save Error", "Cannot save frame: PhotoImage object structure unsupported.")

    # Open save folder in file explorer
    def open_save_folder(self):
        save_folder = Path(__file__).resolve().parent / "save"
        save_folder.mkdir(exist_ok=True)
        subprocess.Popen(f'explorer "{save_folder}"')
    # Open results folder in file explorer
    def open_photo_folder(self):
        photo_folder = Path(__file__).resolve().parent / "photo"
        photo_folder.mkdir(exist_ok=True)
        subprocess.Popen(f'explorer "{photo_folder}"')
    # Open model folder in file explorer
    def open_model_folder(self):
        model_folder = Path(__file__).resolve().parent / "model"
        model_folder.mkdir(exist_ok=True)
        subprocess.Popen(f'explorer "{model_folder}"')

# Main GUI class
class YOLOGui:
    def __init__(self):
# ... (rest of YOLOGui __init__ is unchanged)
        # Initialize main window and variables
        self.root = tk.Tk()
        self.root.title("YOLO Object Detection GUI")
        # NOTE: icon.ico is assumed to be present for the icon to load
        icon_path = Path(__file__).resolve().parent / "icon.ico"
        if icon_path.exists():
            self.root.iconbitmap(str(icon_path))
            
        self.root.geometry("1500x700")
        self.root.minsize(1500, 700)
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
        self.current_photo = None # Variable to hold the reference to the PhotoImage
        
        # Managers
        self.source_manager = YOLOSource(self)
        self.detection_manager = YOLODetection(self)
        self.stats_panel = YOLOStatsPanel(self)
        self.yolo_process = YOLOProcess(self)
        
        self.setup_gui()
        
        # After GUI setup, refresh media devices and models
        self.source_manager.refresh_media_devices()
        self.source_manager.refresh_models()
        self.source_manager.on_source_change() # Ensure correct frame is shown on startup

    # Setup main GUI layout
    def setup_gui(self):
# ... (rest of YOLOGui setup_gui is unchanged)
        # Setup main GUI layout: left, center, right panels
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        # Prevent manual resizing of panels to maintain structure
        self.paned_window.bind('<B1-Motion>', lambda e: "break")
        self.paned_window.bind('<Button-1>', lambda e: "break")
        
        # Left Panel (Source/Model Controls)
        self.left_frame = ttk.Frame(self.paned_window, width=350)
        self.left_frame.pack_propagate(True)
        self.paned_window.add(self.left_frame, weight=0)
        self.setup_left_panel(self.left_frame)
        
        # Center Panel (Video Display)
        self.center_frame = ttk.Frame(self.paned_window)
        self.center_frame.pack_propagate(True)
        self.paned_window.add(self.center_frame, weight=1)
        self.setup_center_panel(self.center_frame)
        
        # Right Panel (Stats/Results)
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
        
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Model selection
        ttk.Label(left_frame, text="Select Model:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        model_dir = Path(__file__).resolve().parent / "model"
        ttk.Label(left_frame, text=f"{model_dir}", font=('Arial', 8)).pack(anchor=tk.W, pady=(0, 2))
        self.model_listbox = tk.Listbox(left_frame, height=5, exportselection=False)
        self.model_listbox.pack(fill=tk.X, pady=2)
        self.model_listbox.bind('<<ListboxSelect>>', self.model_select)
        
        ttk.Button(left_frame, text="Load Custom Model", command=self.load_custom_model).pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, textvariable=self.custom_model_path, foreground="blue").pack(fill=tk.X, pady=2)
        
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
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
        ttk.Button(left_frame, text="Clear Source", command=self.clear_source).pack(fill=tk.X, pady=5)
        
        # Multi-thread detection button
        self.multi_thread_button = ttk.Button(left_frame, text="Multi-Thread Batch Process (Folder)", command=self.run_multi_thread_detection)
        self.multi_thread_button.pack(fill=tk.X, pady=5)

    def run_multi_thread_detection(self):
        # Only suitable for batch processing (Image Folder)
        if self.selected_source.get() != "folder":
            messagebox.showerror("Error", "Multi-Thread Detection is only supported for 'Folder' source (batch image processing).")
            return

        folder = self.current_source
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid image folder for batch processing.")
            return

        sources = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
        
        if not sources:
            messagebox.showerror("Error", "No images found in the selected folder.")
            return

        model_path = None
        if self.selected_model.get():
            model_dir = Path(__file__).resolve().parent / "model"
            if os.path.isabs(self.selected_model.get()) and os.path.exists(self.selected_model.get()):
                model_path = self.selected_model.get()
            else:
                model_path = str(model_dir / self.selected_model.get())
        else:
            messagebox.showerror("Error", "Please select a model.")
            return
            
        # UI update for running state
        self.status_label.config(text="Status: Multi-Thread Running...")
        self.root.update_idletasks()
        
        try:
            # Instantiate and run manager
            manager = MultiThreadDetectionManager(
                model_path, 
                self.conf_threshold.get(), 
                self.nms_threshold.get()
            )
            # Run the manager in a separate thread to prevent GUI lockup
            threading.Thread(target=lambda: self._start_multi_thread_process(manager, sources), daemon=True).start()

        except Exception as e:
            self.status_label.config(text="Status: Ready")
            messagebox.showerror("Error", f"Multi-thread setup failed: {e}")
            
    def _start_multi_thread_process(self, manager, sources):
        """Worker function to run the batch processing manager."""
        results = manager.run(sources)
        
        # Display results back in the main thread
        self.root.after(0, lambda: self._show_multi_thread_results(results))
        self.root.after(0, lambda: self.status_label.config(text="Status: Ready"))

    def _show_multi_thread_results(self, results):
        """Displays the summary of multi-thread batch results."""
        if not results:
            messagebox.showinfo("Multi-Thread Results", "No results generated.")
            return
            
        # Compile a summary of results
        total_processed = len(results)
        total_errors = sum(1 for _, res in results if res.startswith("Error"))
        
        # Aggregate class counts across all images
        overall_stats = {}
        for path, summary in results:
            if not summary.startswith("Error"):
                # Extract class counts from summary string, e.g., 'Detections: 3 (person: 2, car: 1)'
                try:
                    stats_part = summary.split('(')[-1].replace(')', '')
                    if stats_part:
                        parts = stats_part.split(', ')
                        for part in parts:
                            if ':' in part:
                                class_name, count = part.split(': ')
                                overall_stats[class_name] = overall_stats.get(class_name, 0) + int(count)
                except Exception:
                    # Ignore parsing errors for simple presentation
                    pass

        summary_lines = [
            f"--- Batch Summary ---",
            f"Folder: {os.path.basename(self.current_source)}",
            f"Images Processed: {total_processed}",
            f"Processing Errors: {total_errors}",
            f"--- Total Objects Encountered ---",
        ]
        
        # Format overall object stats
        if overall_stats:
            for class_name, count in sorted(overall_stats.items(), key=lambda item: item[1], reverse=True):
                summary_lines.append(f"  - {class_name}: {count}")
        else:
            summary_lines.append("  (No objects detected)")
            
        summary_text = "\n".join(summary_lines)

        messagebox.showinfo("Multi-Thread Detection Summary", summary_text)

    # Setup center panel with video display and status
    def setup_center_panel(self, parent):
# ... (rest of YOLOGui setup_center_panel is unchanged)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=0)
        parent.grid_columnconfigure(0, weight=1)
        
        self.source_display = ttk.LabelFrame(parent, text="Source Display")
        self.source_display.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.display_label = ttk.Label(self.source_display, text="Source: None", font=('Arial', 12, 'bold'))
        self.display_label.pack(anchor=tk.NW, padx=5, pady=5)
        
        self.video_box = ttk.Frame(self.source_display)
        self.video_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video/Image Output Label
        self.video_label = tk.Label(
            self.video_box, text="Video output will appear here", bg="#000000", fg="white", font=("Arial", 16), anchor="center"
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Bottom status/info frame: shows status, source path, and FPS
        self.bottom_status_frame = ttk.Frame(parent)
        self.bottom_status_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
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
# ... (rest of YOLOGui setup_right_panel is unchanged)
        right_frame = ttk.LabelFrame(parent, text="Detection Results")
        right_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 2), pady=(0, 0))
        
        # Session Statistics
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
        self.save_var = tk.BooleanVar(value=False)
        self.auto_checkbox = ttk.Checkbutton(export_frame, text="Auto Save Frame", variable=self.save_var)
        self.auto_checkbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Add buttons to open folders
        ttk.Button(stats_frame, text="Open Save Stats Folder", command=self.stats_panel.open_save_folder).pack(fill=tk.X, pady=2)
        ttk.Button(stats_frame, text="Open Results Folder", command=self.stats_panel.open_photo_folder).pack(fill=tk.X, pady=2)
        ttk.Button(stats_frame, text="Open Model Folder", command=self.stats_panel.open_model_folder).pack(fill=tk.X, pady=2)
        
        # Detection results List
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

    # Update detection results display
    def detection_display(self):
# ... (rest of YOLOGui detection_display is unchanged)
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)
            
        detections = self.detector.get_current_detections()
        
        # Auto-save logic preparation
        if not hasattr(self, 'last_count'):
            self.last_count = 0
            
        main_count = 0
        for detection in detections:
            class_name = detection['class']
            current_count = detection['count']
            total_count = detection['totalEncountered']
            avg_conf = detection['confidence']
            self.detection_tree.insert('', 'end', values=(class_name, current_count, total_count, f"{avg_conf:.2f}"))
            
            # Use max current count for simple auto-save logic
            if current_count > main_count:
                main_count = current_count
                
        # Only save frame if auto-save is enabled
        if not hasattr(self, 'last_save'):
            self.last_save = 0
            
        if hasattr(self, 'save_var') and self.save_var.get():
            # Debounce: only save if count increases and 2.0s has passed since last save
            now = time.time()
            if main_count > 0 and main_count > self.last_count:
                if now - getattr(self, 'last_save', 0) > 2.0:
                    self.stats_panel.save_frame()
                    self.last_save = now
                    
        self.last_count = main_count
        
        session_stats = self.detector.get_session_stats()
        self.session_label.config(text=f"Session Time: {session_stats['sessionTime']}")
        self.detections_label.config(text=f"Total Detections: {session_stats['totalDetections']}")

    # Clear source selection and reset display
    def clear_source(self):
# ... (rest of YOLOGui clear_source is unchanged)
        if not messagebox.askokcancel("Clear Source", "Are you sure you want to clear the current source and reset the display?"):
            return
            
        self.detection_manager.stop_detection()
        self.detector.reset_source()
        self.detection_display()
        
        # Reset source variables
        self.selected_source.set("media_device")
        self.selected_device.set("None")
        self.rtsp_url.set("")
        self.current_source = None
        self.file_label.config(text="No file selected")
        
        # Reset UI
        self.source_manager.on_source_change()
        self.display_label.config(text="Source: None")
        self.path_label.config(text="Source Path: None")
        self.clear_video_image()
        self.enable_source()
        
        # Stop mediamtx if running
        self.rtmp_enabled.set(False)
        self.rtsp_url.set("")
        if hasattr(self, 'mediamtx_proc') and self.mediamtx_proc:
            if sys.platform == 'win32':
                try:
                    subprocess.run(["taskkill", "/F", "/IM", "mediamtx.exe"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                    )
                except subprocess.CalledProcessError:
                    pass
            self.mediamtx_proc = None
            
    # Model selection wrapper
    def model_select(self, event):
        self.source_manager.model_select(event)
        
    # load custom model wrapper
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
        
    # Disable source selection buttons
    def disable_source(self):
        for rb in getattr(self, 'source_live_buttons', []):
            rb.config(state=tk.DISABLED)
            
    # Clear video image
    def clear_video_image(self):
        self.video_label.config(image='', text="Video output will appear here", bg="#000000", fg="white")
        self.current_photo = None
        
    # Enable source selection buttons
    def enable_source(self):
        for rb in getattr(self, 'source_live_buttons', []):
            rb.config(state=tk.NORMAL)
            
    # Start the Tkinter main loop
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
            if self.detection_manager.running:
                self.detection_manager.stop_detection()
            
            # Ensure mediamtx is stopped on exit
            if hasattr(self, 'mediamtx_proc') and self.mediamtx_proc:
                if sys.platform == 'win32':
                    try:
                        subprocess.run(["taskkill", "/F", "/IM", "mediamtx.exe"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                        )
                    except subprocess.CalledProcessError:
                        pass
                        
            self.root.destroy()

# Run the GUI application
if __name__ == "__main__":
    YOLOGui().run()
