import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import sys
import threading
import torch
import numpy as np
from PIL import Image, ImageTk
# Add the project root directory to Python's module search path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import the model
from models.model import InterpolationModel

class VideoInterpolationApp:
    def __init__(self, root):
        self.root = root
<<<<<<< HEAD
<<<<<<< HEAD
        self.style = ttk.Style(theme='darkly')  # Modern theme
        self.root.title("Decadence - AI Video Interpolator")
        self.root.geometry("1000x750")
        
        # Configure theme colors
        self.style.configure("TButton", font=('Helvetica', 11))
        self.style.configure("TLabel", font=('Helvetica', 10))
        self.style.configure("Header.TLabel", font=('Helvetica', 12, 'bold'))
=======
        self.root.title("AI Video Interpolator")
        self.root.geometry("900x650")
        self.root.iconbitmap("assets/logo_no_bg.ico")  # Add your icon file
>>>>>>> parent of 75f4239 (Update app.py)
=======
        self.root.title("AI Video Interpolator")
        self.root.geometry("900x650")
        self.root.iconbitmap("assets/logo_no_bg.ico")  # Add your icon file
>>>>>>> parent of 75f4239 (Update app.py)
        
        # Model and settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.available_models = self._get_available_models()
        self.num_cycles = 1  # Default: 1 cycle (2x FPS)
        
        # UI styling
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, font=("Helvetica", 10))
        self.style.configure("TLabel", font=("Helvetica", 10))
        
        # Widgets
        self.create_widgets()
        self.processing = False
        self.input_path = ""
        self.output_path = ""
        
        # Temporary directories
        self.original_frames_dir = "temp/original_frames"
        self.interpolated_frames_dir = "temp/interpolated_frames"
        os.makedirs(self.original_frames_dir, exist_ok=True)

    def _get_available_models(self):
        models_dir = "models"
        return [f for f in os.listdir(models_dir) if f.endswith(".pth")]

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        # Model selection
        ttk.Label(main_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_combobox = ttk.Combobox(main_frame, values=self.available_models)
        self.model_combobox.grid(row=0, column=1, padx=10, pady=5, sticky=tk.EW)
        self.model_combobox.current(0)
        
        # Interpolation cycles
        ttk.Label(main_frame, text="Interpolation Cycles:").grid(row=1, column=0, sticky=tk.W)
        self.cycle_spinbox = ttk.Spinbox(main_frame, from_=1, to=3, width=5)
        self.cycle_spinbox.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
        self.cycle_spinbox.set(1)
        
        # File selection
        self.btn_select = ttk.Button(main_frame, text="Select Video File", command=self.select_file)
        self.btn_select.grid(row=2, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        # Preview
        self.preview_label = ttk.Label(main_frame)
        self.preview_label.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        # Process button
        self.btn_process = ttk.Button(main_frame, text="Process Video", command=self.start_processing)
        self.btn_process.grid(row=5, column=0, columnspan=2, pady=10, sticky=tk.EW)

    def select_file(self):
        self.input_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        if self.input_path:
            self.show_preview(self.input_path)

    def show_preview(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize((640, 360))
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_label.config(image=imgtk)
            self.preview_label.image = imgtk
        cap.release()

    def start_processing(self):
        if not self.input_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return
        
        # Load selected model
        model_name = self.model_combobox.get()
        self.model = InterpolationModel().to(self.device)
        try:
            self.model.load_state_dict(torch.load(f"models/{model_name}", map_location=self.device))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return
        self.model.eval()
        
        # Get number of cycles
        self.num_cycles = int(self.cycle_spinbox.get())
        
        # Get output path
        self.output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi")]
        )
        
        if self.output_path:
            self.processing = True
            threading.Thread(target=self.process_video).start()

    def process_video(self):
        try:
            total_steps = 10 + (self.num_cycles * 80) + 10  # 10% extraction, 80% per cycle, 10% writing
            step = 0
            
            # Step 1: Extract frames (10%)
            original_frames, original_res = self.extract_frames(self.input_path)
            step += 10
            self.update_progress(step / total_steps * 100)
            
            # Step 2: Interpolate over multiple cycles
            interpolated_frames = original_frames.copy()
            for cycle in range(self.num_cycles):
                interpolated_frames = self.generate_interpolated_frames(interpolated_frames)
                step += 80 / self.num_cycles
                self.update_progress(step / total_steps * 100)
            
            # Step 3: Write video (10%)
            self.create_video(interpolated_frames, original_res)
            self.update_progress(100)
            
            messagebox.showinfo("Success", "Video processing completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.clean_temp_files()
            self.processing = False

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        original_frames = []
        idx = 0
        while cap.isOpened() and self.processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(frame)
            idx += 1
        
        cap.release()
        return original_frames, (width, height)

    def generate_interpolated_frames(self, frames):
        interpolated_frames = []
        for i in range(len(frames)-1):
            frame1 = frames[i]
            frame2 = frames[i+1]
            interpolated_frame = self.run_model(frame1, frame2)
            interpolated_frames.extend([frame1, interpolated_frame])
        interpolated_frames.append(frames[-1])
        return interpolated_frames

    def run_model(self, frame1, frame2):
        frame1_tensor = self.preprocess(frame1).to(self.device)
        frame2_tensor = self.preprocess(frame2).to(self.device)
        with torch.no_grad():
            interpolated_tensor = self.model(frame1_tensor, frame2_tensor)
        return self.postprocess(interpolated_tensor)

    def preprocess(self, frame):
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)

    def postprocess(self, tensor):
        tensor = tensor.squeeze().cpu().permute(1, 2, 0) * 255
        return tensor.numpy().astype(np.uint8)

    def create_video(self, frames, original_res):
        width, height = original_res
        cap = cv2.VideoCapture(self.input_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calculate new FPS: original_fps * (2^num_cycles)
        new_fps = original_fps * (2 ** self.num_cycles)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, new_fps, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

    def update_progress(self, value):
        self.root.after(10, lambda val=value: self.progress.configure(value=val))

    def clean_temp_files(self):
        for folder in [self.original_frames_dir, self.interpolated_frames_dir]:
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))

    def on_closing(self):
        if self.processing:
            if messagebox.askokcancel("Quit", "Processing in progress. Are you sure you want to quit?"):
                self.processing = False
                self.root.destroy()
        else:
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoInterpolationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()