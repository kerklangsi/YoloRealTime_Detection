import tkinter as tk
from tkinter import ttk

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