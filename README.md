
# YOLO Real-Time Detection GUI

GitHub release (latest by version) – [Releases](https://github.com/kerklangsi/YoloRealTime_Detection/releases)  
GitHub repo – [YoloRealTime_Detection](https://github.com/kerklangsi/YoloRealTime_Detection)

A portable **Windows GUI tool** for real-time object detection using YOLO models. Supports detection from webcams, RTSP streams, video files, and images. No coding required just run and use!

---

## 📥 Download & Install

* Download the latest release from the [Releases section](https://github.com/kerklangsi/YoloRealTime_Detection/releases).
* Or clone the repository and install dependencies:

```powershell
git clone https://github.com/kerklangsi/YoloRealTime_Detection.git
cd YoloRealTime_Detection
python "YoloRealTime_Detection.py"
pip install -r requirements.txt
```

---

## ✨ Features

- Real-time object detection using YOLO models (`.pt` files)
- Supports media devices (webcams), RTSP streams, video files, and image files
- Interactive GUI (Tkinter)
- Displays detection results, statistics, and session info
- Export detection statistics to JSON
- GPU/CPU device selection (automatic) nvidia 
- Model selection and custom model loading

---

## ⚙ Requirements

- Microsoft Windows 10/11
- Python 3.8+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (installed via requirements)
- [PyTorch](https://pytorch.org/) (installed via requirements)
- [OpenCV](https://opencv.org/) (installed via requirements)
- [Pillow](https://python-pillow.org/)
- [pygrabber](https://pypi.org/project/pygrabber/) (for webcam support)
- [mediamtx](https://github.com/mediamtx/mediamtx) (for RTMP support)

---

## 📖 Usage

1. Run the GUI application:
   ```powershell
   python main.py
   ```
2. Select the input source (Media Device, RTSP, Video File, Image File).
3. Choose or load a YOLO model (`.pt` file).
4. Adjust detection settings (confidence and NMS thresholds).
5. Start detection. Results and statistics will be displayed in the GUI.
6. Export statistics as needed.

### 🖼 View Saved Annotated Images

* All annotated detection results (from image or folder processing) are automatically saved in the `photo` folder inside your project directory.
* To view saved images:
  - Use the "Open Photo Folder" button in the right panel of the GUI, or
  - Manually open the `photo` folder in Windows Explorer.
* Each processed image is saved as `<image_name>` in `photo`.
* For folder processing, results are saved in subfolders named after the source folder.

### 📁 View Model and Stats Folders

* The `model` folder contains your YOLO model files (`.pt`).
  - Use the "Open Model Folder" button in the GUI, or manually open the `model` folder to manage your models.
* The `stats` folder contains exported detection statistics (e.g., JSON files).
  - Use the "Open Stats Folder" button in the GUI, or manually open the `stats` folder to view or export statistics.

---

Each feature is self-contained and guides you with prompts specific to its task.

---


### 🖼 Example Image


Below are example screenshots and sample outputs from the YOLO Real-Time Detection GUI. Each image shows a different feature or input source:

**Example 1: Default GUI layout**
![Default GUI](https://github.com/kerklangsi/YoloRealTime_Detection/blob/main/example/GUI/example_1.png?raw=true)

| Media Device (webcam/camera) detection | RTSP/RTMP stream detection |
|:----------------------------------------------------:|:------------------------------------------------------:|
| ![Media Device](https://github.com/kerklangsi/YoloRealTime_Detection/blob/main/example/GUI/example_2.png?raw=true) | ![RTSP Stream](https://github.com/kerklangsi/YoloRealTime_Detection/blob/main/example/GUI/example_3.png?raw=true) |
| Video file detection | Image file detection |
| ![Video File](https://github.com/kerklangsi/YoloRealTime_Detection/blob/main/example/GUI/example_4.png?raw=true) | ![Image File](https://github.com/kerklangsi/YoloRealTime_Detection/blob/main/example/GUI/example_5.png?raw=true)
| Folder batch image detection| Open Folder to view results |
| ![Folder Batch](https://github.com/kerklangsi/YoloRealTime_Detection/blob/main/example/GUI/example_6.png?raw=true) | ![Folder View](https://github.com/kerklangsi/YoloRealTime_Detection/blob/main/example/GUI/example_7.png?raw=true)

**Example 9: Annotated image result**
| Example image | Example Result |
|:----------------------------------------------------:|:------------------------------------------------------:|
| ![Annotated Image](https://github.com/kerklangsi/YoloRealTime_Detection/blob/main/example/Image/CCTV%20video_0004.png?raw=true) | ![Annotated Photo](https://github.com/kerklangsi/YoloRealTime_Detection/blob/main/photo/Image/CCTV%20video_0004.png?raw=true) |

**Example 10: Example video file (CCTV video)**
| <video src="https://github.com/user-attachments/assets/813d8233-92af-40ad-a425-f73ecfa821c4"></video> |
| <video src="https://github.com/user-attachments/assets/fc7f5b80-f4d6-4776-bb58-75d2f57b3b60"></video> |

## 🙌 Credits

See [CREDITS.md](CREDITS.md) for full acknowledgments.

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.