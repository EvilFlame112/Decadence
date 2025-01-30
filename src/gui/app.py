import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from src.inference.interpolate import interpolate

class InterpolationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vimeo90K Interpolator")
        self.frame1_path = ""
        self.frame2_path = ""

        # UI Elements
        self.btn_load_frame1 = tk.Button(root, text="Load Frame 1", command=lambda: self.load_frame(1))
        self.btn_load_frame2 = tk.Button(root, text="Load Frame 2", command=lambda: self.load_frame(2))
        self.btn_interpolate = tk.Button(root, text="Interpolate", command=self.run_interpolation)
        self.label_result = tk.Label(root)

        self.btn_load_frame1.pack()
        self.btn_load_frame2.pack()
        self.btn_interpolate.pack()
        self.label_result.pack()

    def load_frame(self, frame_id):
        path = filedialog.askopenfilename(title=f"Select Frame {frame_id}")
        if frame_id == 1:
            self.frame1_path = path
        else:
            self.frame2_path = path

    def run_interpolation(self):
        if not self.frame1_path or not self.frame2_path:
            return
        interpolated = interpolate(self.frame1_path, self.frame2_path)
        interpolated_rgb = cv2.cvtColor(interpolated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(interpolated_rgb)
        img_tk = ImageTk.PhotoImage(img)
        self.label_result.config(image=img_tk)
        self.label_result.image = img_tk

# Run the app
root = tk.Tk()
app = InterpolationApp(root)
root.mainloop()