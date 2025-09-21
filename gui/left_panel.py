import tkinter as tk
from tkinter import ttk

def setup_left_panel(self, parent):
    left_frame = ttk.LabelFrame(parent, text="Source Selection", padding=10)
    left_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 2), pady=(0, 0))
    ttk.Label(left_frame, text="Input Source:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
    sources = ["Media Device", "RTSP Stream", "Video File", "Image File"]
    values = ["media_device", "rtsp", "video", "image"]
    for text, value in zip(sources, values):
        ttk.Radiobutton(left_frame, text=text, variable=self.selected_source, value=value,
                        command=self.on_source_change).pack(anchor=tk.W, pady=2)
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