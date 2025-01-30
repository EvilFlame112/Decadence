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
        self.root.title("Video Frame Interpolation")
        self.root.geometry("800x600")
        
        # Model setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InterpolationModel().to(self.device)
        self.model.load_state_dict(torch.load("models/interpolator.pth", map_location=self.device))
        self.model.eval()
        
        # UI Elements
        self.create_widgets()
        self.processing = False
        self.input_path = ""
        self.output_path = ""
        
        # Temporary directories
        self.original_frames_dir = "temp/original_frames"
        self.interpolated_frames_dir = "temp/interpolated_frames"
        os.makedirs(self.original_frames_dir, exist_ok=True)
        os.makedirs(self.interpolated_frames_dir, exist_ok=True)

    def create_widgets(self):
        # File selection
        self.btn_select = ttk.Button(self.root, text="Select Video File", command=self.select_file)
        self.btn_select.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)
        
        # Start button
        self.btn_process = ttk.Button(self.root, text="Process Video", command=self.start_processing)
        self.btn_process.pack(pady=10)
        
        # Preview window
        self.preview_label = ttk.Label(self.root)
        self.preview_label.pack(pady=10)

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
        
        self.output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi")]
        )
        
        if self.output_path:
            self.processing = True
            threading.Thread(target=self.process_video).start()

    def process_video(self):
        try:
            # Step 1: Extract original frames
            original_frames = self.extract_frames(self.input_path)
            
            # Step 2: Interpolate frames
            interpolated_frames = self.generate_interpolated_frames(original_frames)
            
            # Step 3: Create new video
            self.create_video(interpolated_frames)
            
            messagebox.showinfo("Success", "Video processing completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.clean_temp_files()
            self.processing = False
    
    def run_model(self, frame1, frame2):
        """Run the model on two input frames."""
        # Preprocess frames
        frame1_tensor = self.preprocess(frame1).to(self.device)
        frame2_tensor = self.preprocess(frame2).to(self.device)
        
        # Generate interpolated frame
        with torch.no_grad():
            interpolated_tensor = self.model(frame1_tensor, frame2_tensor)
        
        # Postprocess and return
        return self.postprocess(interpolated_tensor)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        original_frames = []
        idx = 0
        while cap.isOpened() and self.processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(frame)
            
            # Save frame for potential debugging
            cv2.imwrite(os.path.join(self.original_frames_dir, f"frame_{idx:04d}.png"), frame)
            idx += 1
            
            self.update_progress(idx/frame_count * 50)  # First 50% progress
        
        cap.release()
        return original_frames

    def generate_interpolated_frames(self, original_frames):
        interpolated_frames = []
        
        for i in range(len(original_frames)-1):
            # Get consecutive frames
            frame1 = original_frames[i]
            frame2 = original_frames[i+1]
            
            # Generate interpolated frame using run_model
            interpolated_frame = self.run_model(frame1, frame2)
            
            # Append original and interpolated frames
            interpolated_frames.append(frame1)
            interpolated_frames.append(interpolated_frame)
        
        # Add last frame
        interpolated_frames.append(original_frames[-1])
        return interpolated_frames

    def preprocess(self, frame):
        # Resize frame to match model input (256x256)
        frame = cv2.resize(frame, (256, 256))
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)  # Add batch dimension

    def postprocess(self, tensor):
        tensor = tensor.squeeze().cpu().permute(1, 2, 0) * 255
        return tensor.numpy().astype(np.uint8)

    def create_video(self, frames):
        if not frames:
            return
        
        # Get video properties from input video
        cap = cv2.VideoCapture(self.input_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Calculate new FPS to maintain the same duration
        new_fps = original_fps * 2  # Double FPS for smooth playback
        
        # Get frame dimensions (all frames are resized to 256x256)
        height, width = 256, 256
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, new_fps, (width, height))
        
        # Write all interpolated frames
        for frame in frames:
            frame = cv2.resize(frame, (width, height))  # Ensure correct size
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        
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