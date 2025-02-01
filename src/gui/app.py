import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttk  # Modern theme library
from ttkbootstrap.constants import *
import cv2
import os
import sys
import threading
import torch
import numpy as np
from PIL import Image, ImageTk
import time
# Add the project root directory to Python's module search path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import the model
from models.model import InterpolationModel

class VideoInterpolationApp:
    def __init__(self, root):
        self.root = root
        self.style = ttk.Style(theme='darkly')  # Modern theme
        self.root.title("Decadence - AI Video Interpolator")
        self.root.geometry("1000x750")
        
        # Configure theme colors
        self.style.configure("TButton", font=('Helvetica', 11))
        self.style.configure("TLabel", font=('Helvetica', 10))
        self.style.configure("Header.TLabel", font=('Helvetica', 12, 'bold'))
        
        # Model and settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_ui()
        self.processing = False
        self.total_frames = 0
        self.processed_frames = 0

    def initialize_ui(self):
        """Create and arrange UI components with modern styling"""
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header Section
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=10)
        ttk.Label(header_frame, text="AI Video Interpolator", style="Header.TLabel").pack(side=tk.LEFT)
        
        # Settings Panel
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding=15)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Model Selection
        ttk.Label(settings_frame, text="Model:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.model_combobox = ttk.Combobox(settings_frame, values=self.get_available_models())
        self.model_combobox.grid(row=0, column=1, padx=5, sticky=tk.EW)
        self.model_combobox.current(0)
        
        # Interpolation Cycles
        ttk.Label(settings_frame, text="Cycles:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.cycle_spinbox = ttk.Spinbox(settings_frame, from_=1, to=4, width=3)
        self.cycle_spinbox.grid(row=0, column=3, padx=5, sticky=tk.W)
        self.cycle_spinbox.set(1)
        
        # File Selection
        self.btn_select = ttk.Button(
            main_frame, 
            text="Select Video File", 
            command=self.select_file,
            bootstyle=PRIMARY
        )
        self.btn_select.pack(fill=tk.X, pady=10)
        
        # Preview
        self.preview_label = ttk.Label(main_frame)
        self.preview_label.pack(pady=10)
        
        # Progress Bar
        self.progress = ttk.Progressbar(
            main_frame, 
            orient=HORIZONTAL, 
            length=400, 
            mode='determinate',
            bootstyle=(STRIPED, SUCCESS)
        )
        self.progress.pack(fill=tk.X, pady=10)
        
        # Stats Panel
        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill=tk.X, pady=5)
        self.time_label = ttk.Label(stats_frame, text="Elapsed: 00:00 | Remaining: --:--")
        self.time_label.pack(side=tk.LEFT)
        self.fps_label = ttk.Label(stats_frame, text="Processing FPS: 0.00")
        self.fps_label.pack(side=tk.RIGHT)
        
        # Process Button
        self.btn_process = ttk.Button(
            main_frame, 
            text="Start Processing", 
            command=self.start_processing,
            bootstyle=(SUCCESS, OUTLINE)
        )
        self.btn_process.pack(fill=tk.X, pady=10)

    def get_available_models(self):
        return [f for f in os.listdir("models") if f.endswith(".pth")]

    def select_file(self):
        self.input_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if self.input_path:
            self.show_preview(self.input_path)
            self.btn_process.configure(bootstyle=SUCCESS)

    def show_preview(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize((640, 360), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_label.config(image=imgtk)
            self.preview_label.image = imgtk
        cap.release()

    def start_processing(self):
        if not self.input_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return

        # Initialize processing
        self.processing = True
        self.processed_frames = 0
        self.total_frames = 0
        self.start_time = time.time()
        
        # Start processing thread
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        try:
            # Load model
            model = InterpolationModel().to(self.device)
            model.load_state_dict(torch.load(f"models/{self.model_combobox.get()}"))
            model.eval()

            # Calculate total work
            original_frames = self.extract_frames()
            total_cycles = int(self.cycle_spinbox.get())
            self.total_frames = len(original_frames) * (2 ** total_cycles)
            
            # Process frames
            current_frames = original_frames
            for cycle in range(total_cycles):
                current_frames = self.interpolate_frames(current_frames, model)
            
            # Write output
            self.write_video(current_frames)
            
            messagebox.showinfo("Success", "Processing completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.processing = False
            self.clean_temp_files()

    def extract_frames(self):
        """Smooth progress implementation for frame extraction"""
        cap = cv2.VideoCapture(self.input_path)
        frames = []
        
        while cap.isOpened() and self.processing:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            # Update progress
            self.processed_frames += 1
            self.update_progress()
        
        cap.release()
        return frames

    def interpolate_frames(self, frames, model):
        """Smooth progress implementation for interpolation"""
        new_frames = []
        
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i+1]
            
            # Process frame pair
            interpolated = self.process_frame_pair(frame1, frame2, model)
            new_frames.extend([frame1, interpolated])
            
            # Update progress for both frames
            self.processed_frames += 2
            self.update_progress()
        
        new_frames.append(frames[-1])
        return new_frames

    def process_frame_pair(self, frame1, frame2, model):
        """Process a single frame pair with progress tracking"""
        tensor1 = self.frame_to_tensor(frame1)
        tensor2 = self.frame_to_tensor(frame2)
        
        with torch.no_grad():
            output = model(tensor1, tensor2)
        
        return self.tensor_to_frame(output)

    def update_progress(self):
        """Smooth real-time progress updates with time estimation"""
        elapsed = time.time() - self.start_time
        progress = self.processed_frames / self.total_frames
        fps = self.processed_frames / elapsed if elapsed > 0 else 0
        
        # Update progress bar
        self.progress['value'] = progress * 100
        
        # Update time estimates
        remaining = (elapsed / progress - elapsed) if progress > 0 else 0
        self.time_label.config(
            text=f"Elapsed: {self.format_time(elapsed)} | Remaining: {self.format_time(remaining)}"
        )
        self.fps_label.config(text=f"Processing FPS: {fps:.2f}")
        
        self.root.update_idletasks()

    def format_time(self, seconds):
        """Convert seconds to MM:SS format"""
        if seconds < 0: return "--:--"
        return f"{int(seconds//60):02d}:{int(seconds%60):02d}"

    def write_video(self, frames):
        """Final video writing with progress updates"""
        cap = cv2.VideoCapture(self.input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) * (2 ** int(self.cycle_spinbox.get()))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.processed_frames += 1
            self.update_progress()
        
        out.release()

    def frame_to_tensor(self, frame):
        """Convert numpy frame to normalized tensor"""
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def tensor_to_frame(self, tensor):
        """Convert tensor back to numpy frame"""
        tensor = tensor.squeeze().cpu().permute(1, 2, 0) * 255
        return tensor.numpy().astype(np.uint8)

    def clean_temp_files(self):
        """Clean temporary directories"""
        for folder in ["temp/original_frames", "temp/interpolated_frames"]:
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))

if __name__ == "__main__":
    root = ttk.Window()
    app = VideoInterpolationApp(root)
    root.mainloop()