import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from yolov4.utils_v4 import detect_image, detect_video, Load_Yolo_model
from yolov4.configs import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import time

class YoloApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv4 Object Detection")
        
        self.image_path = None
        self.video_path = None
        self.output_dir = "./detected"
        self.yolo = Load_Yolo_model()  # Încarcă modelul YOLO
        self.image_size = (500, 500)  # Dimensiunea imaginii pentru afișare
        
        # Creează directorul de ieșire dacă nu există
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        
        # Creează un cadru pentru butoane
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10, side=tk.TOP)
        
        self.load_image_button = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_image_button.pack(side=tk.LEFT, padx=5)
        
        self.detect_image_button = tk.Button(button_frame, text="Apply YOLOv4", command=self.apply_yolo_image)
        self.detect_image_button.pack(side=tk.LEFT, padx=5)
        
        self.load_and_detect_video_button = tk.Button(button_frame, text="Apply YOLOv4 on Video", command=self.load_and_apply_yolo_video)
        self.load_and_detect_video_button.pack(side=tk.LEFT, padx=5)

        self.load_csv_button = tk.Button(button_frame, text="Load CSV Data", command=self.load_csv_data)
        self.load_csv_button.pack(side=tk.LEFT, padx=5)
        
        self.stat_type = tk.StringVar(root)
        self.stat_type.set("Select Statistic")
        self.stat_menu = tk.OptionMenu(button_frame, self.stat_type, "Accuracy", "Precision", "Recall", "F1 Score", command=self.update_plot)
        self.stat_menu.pack(side=tk.LEFT, padx=5)
        
        # Cadru pentru afișarea imaginii
        image_frame = tk.Frame(root)
        image_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.image_label = tk.Label(image_frame)
        self.image_label.pack()

        self.time_label = tk.Label(image_frame, text="")
        self.time_label.pack()

        # Cadru pentru graficul de statistici
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.metrics_df = None  # DataFrame pentru metrici

    # Funcție pentru încărcarea unei imagini
    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize(self.image_size, Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
    
    # Funcție pentru aplicarea modelului YOLO pe o imagine
    def apply_yolo_image(self):
        if not self.image_path:
            return
        
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        output_image_path = os.path.join(self.output_dir, f"{image_name}_pred.jpg")

        start_time = time.time()
        detect_image(self.yolo, self.image_path, output_image_path, input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255, 0, 0))
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        self.time_label.config(text=f"Detection Time: {elapsed_time:.2f} seconds")

        result_image = Image.open(output_image_path)
        result_image = result_image.resize(self.image_size, Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(result_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
    
    # Funcție pentru aplicarea modelului YOLO pe un videoclip
    def load_and_apply_yolo_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if self.video_path:
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_video_path = os.path.join(self.output_dir, f"{video_name}_pred.avi")
            
            detect_video(self.yolo, self.video_path, output_video_path, input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))

    # Funcție pentru încărcarea datelor dintr-un fișier CSV
    def load_csv_data(self):
        csv_path = filedialog.askopenfilename(
            title="Select CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if csv_path:
            self.metrics_df = pd.read_csv(csv_path)
            self.update_plot()
    
    # Funcție pentru generarea graficului
    def generate_plots(self, stat_type):
        if self.metrics_df is None:
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))

        classes = self.metrics_df["Class"]
        values = self.metrics_df[stat_type]

        ax.bar(classes, values, color='b')
        ax.set_title(f'{stat_type} per Class', fontsize=10)
        ax.set_xlabel('Class', fontsize=8)
        ax.set_ylabel(stat_type, fontsize=8)
        ax.set_xticks(classes)
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        plt.tight_layout()

        return fig

    # Funcție pentru actualizarea graficului
    def update_plot(self, event=None):
        if self.metrics_df is None:
            return
        
        metric = self.stat_type.get()
        fig = self.generate_plots(metric)
        
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloApp(root)
    root.mainloop()
