import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QProgressBar, QFileDialog, QMessageBox, QTextEdit, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
# Add the project root directory to Python's module search path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import the model
from models.model import InterpolationModel


class VideoInterpolationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Video Interpolator")
        self.setGeometry(100, 100, 1200, 900)  # Window size
        self.center_window()  # Center the window on launch

        # Initialize model and settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.available_models = self._get_available_models()
        self.num_cycles = 1

        # Theme settings
        self.dark_mode = True
        self.set_dark_theme()

        # Create UI components
        self.init_ui()
        self.processing = False
        self.input_path = ""
        self.output_path = ""

        # Setup temporary directories
        self.original_frames_dir = "temp/original_frames"
        self.interpolated_frames_dir = "temp/interpolated_frames"
        os.makedirs(self.original_frames_dir, exist_ok=True)

    def center_window(self):
        """Center the window on the user's screen."""
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def _get_available_models(self):
        models_dir = "models"
        return [f for f in os.listdir(models_dir) if f.endswith(".pth")]

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Left panel for controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)

        # Application title
        title_label = QLabel("Decadence AI Video Interpolator")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #fff;")
        control_layout.addWidget(title_label)

        # Model selection
        control_layout.addWidget(QLabel("Select Model:"))
        self.model_combobox = QComboBox()
        self.model_combobox.addItems(self.available_models)
        self.model_combobox.setStyleSheet("QComboBox { border-radius: 10px; padding: 5px; background-color: #444; color: #fff; }")
        control_layout.addWidget(self.model_combobox)

        # Interpolation cycles
        control_layout.addWidget(QLabel("Interpolation Cycles (1-3):"))
        self.cycle_spinbox = QSpinBox()
        self.cycle_spinbox.setRange(1, 3)
        self.cycle_spinbox.setValue(1)
        self.cycle_spinbox.setStyleSheet("QSpinBox { border-radius: 10px; padding: 5px; background-color: #444; color: #fff; }")
        control_layout.addWidget(self.cycle_spinbox)

        # File selection button
        self.btn_select = QPushButton("Select Video File")
        self.btn_select.clicked.connect(self.select_file)
        self.btn_select.setStyleSheet("QPushButton { border-radius: 10px; padding: 10px; background-color: #4CAF50; color: #fff; }"
                                     "QPushButton:hover { background-color: #45a049; }")
        control_layout.addWidget(self.btn_select)

        # Process button
        self.btn_process = QPushButton("Process Video")
        self.btn_process.clicked.connect(self.start_processing)
        self.btn_process.setStyleSheet("QPushButton { border-radius: 10px; padding: 10px; background-color: #2196F3; color: #fff; }"
                                      "QPushButton:hover { background-color: #1e88e5; }")
        control_layout.addWidget(self.btn_process)
        
        # Theme toggle button with icon
        self.btn_theme = QPushButton()
        self.btn_theme.setIcon(QIcon("assets/logo_no_bg.ico"))  # Add a small icon for theme switcher
        self.btn_theme.setToolTip("Switch Theme")
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_theme.setStyleSheet("QPushButton { border-radius: 10px; padding: 10px; background-color: #444; }")
        self.btn_theme.setMaximumWidth(50)
        control_layout.addWidget(self.btn_theme)

        

        # Add control panel to main layout
        main_layout.addWidget(control_panel, stretch=1)

        # Right panel for preview and console
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Preview label for video/image
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #555; border-radius: 10px;")
        right_layout.addWidget(self.preview_label, stretch=3)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setStyleSheet("QProgressBar { border-radius: 5px; background-color: #444; }"
                                   "QProgressBar::chunk { background-color: #2196F3; border-radius: 5px; }")
        right_layout.addWidget(self.progress)

        # Console view
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #555; color: #fff; border-radius: 10px; padding: 10px;")
        right_layout.addWidget(self.console, stretch=1)

        # Add right panel to main layout
        main_layout.addWidget(right_panel, stretch=3)

        # Set main widget
        self.setCentralWidget(main_widget)

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.set_dark_theme()
        else:
            self.set_light_theme()

    def set_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QLabel {
                color: #333;
            }
            QWidget {
                background-color: #333;
                color: #fff;
            }
            QPushButton {
                border-radius: 10px;
                padding: 10px;
            }
            QComboBox, QSpinBox {
                border-radius: 10px;
                padding: 5px;
                background-color: #444;
                color: #fff;
            }
            QProgressBar {
                border-radius: 5px;
                background-color: #444;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 5px;
            }
            QTextEdit {
                background-color: #333;
                color: #fff;
                border-radius: 10px;
                padding: 10px;
            }
        """)

    def set_light_theme(self):
        """Apply light theme styling."""
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                color: #000;
            }
            QPushButton {
                border-radius: 10px;
                padding: 10px;
            }
            QComboBox, QSpinBox {
                border-radius: 10px;
                padding: 5px;
                background-color: #fff;
                color: #000;
            }
            QProgressBar {
                border-radius: 5px;
                background-color: #fff;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 5px;
            }
            QTextEdit {
                background-color: #fff;
                color: #000;
                border-radius: 10px;
                padding: 10px;
            }
        """)

    def log_to_console(self, message):
        """Log messages to the console."""
        self.console.append(message)
        self.console.ensureCursorVisible()

    def select_file(self):
        self.input_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if self.input_path:
            self.log_to_console(f"Selected video: {self.input_path}")
            self.show_preview(self.input_path)

    def show_preview(self, video_path):
        """Show the first frame of the selected video in the preview."""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            self.update_preview(img)
        cap.release()

    def update_preview(self, img):
        """Update the preview window with a resized and centered image."""
        # Resize image to fit the preview area
        preview_width = self.preview_label.width()
        preview_height = self.preview_label.height()
        img.thumbnail((preview_width, preview_height), Image.ANTIALIAS)

        # Convert PIL image to QPixmap
        qimage = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # Center the image in the preview area
        self.preview_label.setPixmap(pixmap)
        self.preview_label.setAlignment(Qt.AlignCenter)

    def start_processing(self):
        if not self.input_path:
            self.log_to_console("Error: Please select a video file first!")
            return

        # Load selected model
        model_name = self.model_combobox.currentText()
        try:
            self.model = InterpolationModel().to(self.device)
            self.model.load_state_dict(torch.load(f"models/{model_name}", map_location=self.device))
            self.model.eval()
            self.log_to_console(f"Loaded model: {model_name}")
        except Exception as e:
            self.log_to_console(f"Error: Failed to load model: {str(e)}")
            return

        # Get processing parameters
        self.num_cycles = self.cycle_spinbox.value()
        self.output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video", "", "MP4 Video (*.mp4);;AVI Video (*.avi)"
        )

        if self.output_path:
            self.processing = True
            self.btn_process.setEnabled(False)
            self.btn_select.setEnabled(False)
            self.worker = VideoProcessingThread(
                self.input_path, self.output_path, self.num_cycles, self.model, self.device
            )
            self.worker.progress_signal.connect(self.update_progress)
            self.worker.log_signal.connect(self.log_to_console)
            self.worker.finished_signal.connect(self.on_processing_finished)
            self.worker.start()

    def update_progress(self, value):
        self.progress.setValue(value)

    def on_processing_finished(self, success):
        self.processing = False
        self.btn_process.setEnabled(True)
        self.btn_select.setEnabled(True)

        if success:
            self.log_to_console("Processing complete! Displaying output video...")
            self.show_preview(self.output_path)
            QMessageBox.information(self, "Success", "Video processing completed!")
        else:
            QMessageBox.critical(self, "Error", "Processing failed. Check console for details.")

    def closeEvent(self, event):
        if self.processing:
            reply = QMessageBox.question(
                self, "Quit", "Processing in progress. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


class VideoProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)

    def __init__(self, input_path, output_path, num_cycles, model, device):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.num_cycles = num_cycles
        self.model = model
        self.device = device

    def run(self):
        try:
            self.log_signal.emit("Starting video processing...")
            self.progress_signal.emit(0)

            # Phase 1: Extract frames (30% of progress)
            self.log_signal.emit("Extracting frames...")
            original_frames, original_res = self.extract_frames(self.input_path)
            self.progress_signal.emit(30)

            # Phase 2: Interpolate frames (60% of progress)
            interpolated_frames = original_frames.copy()
            for cycle in range(self.num_cycles):
                self.log_signal.emit(f"Interpolating frames (Cycle {cycle + 1}/{self.num_cycles})...")
                interpolated_frames = self.generate_interpolated_frames(interpolated_frames)
                self.progress_signal.emit(30 + (60 * (cycle + 1) // self.num_cycles))

            # Phase 3: Save video (10% of progress)
            self.log_signal.emit("Saving output video...")
            self.create_video(interpolated_frames, original_res)
            self.progress_signal.emit(100)

            self.finished_signal.emit(True)
        except Exception as e:
            self.log_signal.emit(f"Error: Processing failed: {str(e)}")
            self.finished_signal.emit(False)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        original_frames = []
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(frame)
            idx += 1

            # Update progress during extraction
            self.progress_signal.emit(int(30 * (idx / frame_count)))

        cap.release()
        return original_frames, (width, height)

    def generate_interpolated_frames(self, frames):
        interpolated_frames = []
        total_pairs = len(frames) - 1

        for i in range(total_pairs):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            interpolated_frame = self.run_model(frame1, frame2)
            interpolated_frames.extend([frame1, interpolated_frame])

            # Update progress during interpolation
            self.progress_signal.emit(int(30 + (60 * (i + 1) / total_pairs)))

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

        new_fps = original_fps * (2 ** self.num_cycles)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, new_fps, (width, height))

        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoInterpolationApp()
    window.show()
    sys.exit(app.exec_())