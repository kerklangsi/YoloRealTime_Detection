# GitHub Copilot Instructions

## Project Overview

This is a Windows-first YOLO real-time object detection GUI application built with Python/Tkinter. The application provides a user-friendly interface for running YOLO models on various input sources (webcams, RTSP streams, video files, images) without requiring users to write code.

## Architecture & Key Components

### Core Class Hierarchy
- `YOLODetector`: Main orchestrator combining model management and statistics
- `YOLOModel`: Handles model loading, device selection (GPU/CPU), and YOLO inference
- `YOLOStats`: Manages detection statistics, JSON export, and session tracking
- `YOLOGui`: Main Tkinter GUI with three-panel layout (left controls, center video, right stats)

### Manager Classes (Composition Pattern)
The GUI uses composition with specialized manager classes:
- `YOLOSource`: Handles input source selection and device enumeration
- `YOLODetection`: Manages detection lifecycle and video processing threads
- `YOLOStatsPanel`: Coordinates statistics display and export operations

### Threading Model
- **Main Thread**: Tkinter GUI and user interactions
- **Processing Thread**: Video frame processing loop (`YOLODetection.process_video()`)
- **Thread Communication**: Uses `root.after()` for thread-safe GUI updates

## Development Workflows

### Running the Application
```powershell
python YoloRealTime_Detection.py
```

### Building Executable
Use the provided PowerShell script for PyInstaller builds:
```powershell
.\pyinstaller.ps1
```
This script:
- Auto-detects Python files and icons
- Reads hidden imports from `include.txt`
- Includes all folders as data (excluding `.git`, `__pycache__`)
- Creates a windowed executable in `dist/` folder

### Model Management
- Place `.pt` YOLO model files in `model/` directory
- Default model: `model/HotsportDetector_yolo11.pt`
- Custom models loaded via file dialog are temporarily added to model list

## Project-Specific Patterns

### Error Handling Strategy
- GUI operations use `messagebox.showerror()` for user-friendly error reporting
- Detection errors are logged to console but don't crash the GUI
- Resource cleanup (camera/video capture) uses try/catch blocks

### Device Detection (Windows-Specific)
```python
# Preferred method using pygrabber (Windows DirectShow)
graph = FilterGraph()
devices = graph.get_input_devices()

# Fallback: OpenCV device enumeration
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
```

### Statistics Export Pattern
JSON exports include:
- Session metadata (model name, duration, total detections)
- Per-class statistics (count, confidence averages)
- ISO timestamp for export tracking
- Files saved to `save/` directory with timestamp naming

### GUI Layout Convention
Three-panel design using `ttk.PanedWindow`:
- **Left (350px)**: Source selection, model loading, detection controls
- **Center (flexible)**: Video display with status bar
- **Right (350px)**: Detection results tree and statistics

## Critical Integration Points

### YOLO Model Integration
- Uses Ultralytics YOLO library (`ultralytics.YOLO`)
- Automatic GPU detection: `torch.cuda.is_available()`
- Model inference: `model(frame, conf=threshold, iou=nms_threshold)`

### Video Source Handling
- **Media Device**: Uses OpenCV with DirectShow backend (`cv2.CAP_DSHOW`)
- **RTSP**: Direct URL to `cv2.VideoCapture()`
- **Files**: Standard OpenCV file reading
- Frame resizing: Fixed 1024x768 for processing, dynamic for display

### Threading Communication
```python
# Safe GUI updates from processing thread
self.gui.root.after(0, lambda f=frame: self.display_frame(f))
self.gui.root.after(0, self.gui.detection_display)
```

## Build System Notes

### PyInstaller Configuration
- `--noconsole`: Windowed application (no command prompt)
- `--onedir`: Creates a folder distribution (not single file)
- `--contents-directory "bin"`: Moves libraries to bin/ subfolder
- Auto-includes: All project folders as data bundles

### Required Hidden Imports
Listed in `include.txt`:
- `ultralytics` (YOLO framework)
- `torch` (PyTorch)
- `cv2` (OpenCV)
- `numpy`, `PIL` (image processing)
- `pygrabber.dshow_graph` (Windows camera detection)

## File Structure Conventions

```
YoloRealTime_Detection/
├── YoloRealTime_Detection.py  # Single-file application
├── model/                     # YOLO .pt model files
├── save/                      # JSON statistics exports
├── pyinstaller.ps1           # Build automation script
├── include.txt               # PyInstaller hidden imports
└── icon.ico                  # Application icon
```