
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
pip install -r requirements.txt
```

---

## ✨ Features

- Real-time object detection using YOLO models (`.pt` files)
- Supports media devices (webcams), RTSP streams, video files, and image files
- Interactive GUI (Tkinter)
- Displays detection results, statistics, and session info
- Export detection statistics to JSON
- GPU/CPU device selection (automatic)
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

Each feature is self-contained and guides you with prompts specific to its task.

---

## 🙌 Credits

See [CREDITS.md](CREDITS.md) for full acknowledgments.

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.