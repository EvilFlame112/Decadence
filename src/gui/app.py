import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QSpinBox, QProgressBar, 
    QFileDialog, QMessageBox, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PIL import Image
# Add the project root directory to Python's module search path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import the model
from models.model import InterpolationModel

class VideoInterpolationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Decandence: AI Video Interpolator")
        self.setGeometry(100, 100, 1280, 900)
        
        # Set application icon
        app_icon = QIcon("assets/logo_dark.png")
        self.setWindowIcon(app_icon)
        QApplication.setWindowIcon(app_icon)  # This sets the taskbar icon
        
        self.center_window()

        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Theme settings
        self.dark_mode = True
        self.theme_icons = {
            True: QIcon("assets/sun.png"),    # Show sun when in dark mode (to switch to light)
            False: QIcon("assets/moon.png")   # Show moon when in light mode (to switch to dark)
        }

        # Initialize UI
        self.init_ui()
        self.processing = False
        self.input_path = ""
        self.output_path = ""

        # Setup temporary directories
        self.original_frames_dir = "temp/original_frames"
        self.interpolated_frames_dir = "temp/interpolated_frames"
        os.makedirs(self.original_frames_dir, exist_ok=True)

    def center_window(self):
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def _get_available_models(self):
        models_dir = "models"
        return [f for f in os.listdir(models_dir) if f.endswith(".pth")]

    def init_ui(self):
        # Main widget setup
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Top bar with title and theme toggle
        top_bar = QHBoxLayout()
        top_bar.setSpacing(10)

        # Title with icon
        title_container = QHBoxLayout()
        title_container.setSpacing(8)
        icon_label = QLabel()
        icon_label.setPixmap(QIcon("assets/logo_dark.png").pixmap(QSize(48, 48)))
        title_container.addWidget(icon_label)
        
        title = QLabel("Decadence")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title_container.addWidget(title)
        
        subtitle = QLabel("AI Video Frame Interpolation")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setStyleSheet("color: #666;")
        title_container.addWidget(subtitle)
        title_container.addStretch()
        top_bar.addLayout(title_container)

        # Theme toggle
        self.btn_theme = QPushButton()
        self.btn_theme.setIcon(self.theme_icons[self.dark_mode])
        self.btn_theme.setIconSize(QSize(16, 16))
        self.btn_theme.setFixedSize(40, 40)
        self.btn_theme.setToolTip("Toggle Light/Dark Mode")
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.style_theme_button(self.btn_theme)
        top_bar.addWidget(self.btn_theme)
        
        main_layout.addLayout(top_bar)

        # Main content area
        content = QHBoxLayout()
        content.setSpacing(15)

        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # Model section
        self.model_section = QWidget()
        self.model_section.setObjectName("model_section")
        model_layout = QVBoxLayout(self.model_section)
        model_layout.setContentsMargins(12, 12, 12, 12)
        model_layout.setSpacing(6)

        model_header = QLabel("Model Settings")
        model_header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        model_layout.addWidget(model_header)

        self.model_combobox = QComboBox()
        self.model_combobox.addItems(self._get_available_models())
        self.style_combo(self.model_combobox, "#4CAF50")
        model_layout.addWidget(self.model_combobox)

        cycles_label = QLabel("Interpolation Cycles")
        cycles_label.setFont(QFont("Segoe UI", 10))
        model_layout.addWidget(cycles_label)

        self.cycle_spinbox = QSpinBox()
        self.cycle_spinbox.setRange(1, 3)
        self.cycle_spinbox.setValue(1)
        self.style_spinbox(self.cycle_spinbox, "#2196F3")
        model_layout.addWidget(self.cycle_spinbox)

        self.style_section(self.model_section)
        left_layout.addWidget(self.model_section)

        # Action buttons
        self.button_section = QWidget()
        button_layout = QVBoxLayout(self.button_section)
        button_layout.setContentsMargins(12, 12, 12, 12)
        button_layout.setSpacing(8)

        self.btn_select = QPushButton("Select Input Video")
        self.btn_select.setIcon(QIcon("assets/folder.png"))
        self.style_action_button(self.btn_select, "primary")
        self.btn_select.clicked.connect(self.select_file)
        button_layout.addWidget(self.btn_select)

        self.btn_process = QPushButton("Start Processing")
        self.btn_process.setIcon(QIcon("assets/play.png"))
        self.style_action_button(self.btn_process, "success")
        self.btn_process.clicked.connect(self.start_processing)
        button_layout.addWidget(self.btn_process)

        self.style_section(self.button_section)
        left_layout.addWidget(self.button_section)
        left_layout.addStretch()

        

        content.addWidget(left_panel)

        # Right panel - Preview and Console
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        # Preview area
        self.preview_section = QWidget()
        preview_layout = QVBoxLayout(self.preview_section)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(400)
        preview_layout.addWidget(self.preview_label)
        
        self.style_section(self.preview_section)
        right_panel.addWidget(self.preview_section, 2)

        # Add progress bar to left panel
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.style_progress(self.progress, "#2196F3")
        right_panel.addWidget(self.progress)

        # Console area
        self.console_section = QWidget()
        console_layout = QVBoxLayout(self.console_section)
        console_layout.setContentsMargins(12, 12, 12, 12)

        console_header = QLabel("Console Output")
        console_header.setFont(QFont("Segoe UI", 11, QFont.Bold))
        console_layout.addWidget(console_header)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 10))
        self.style_console(self.console)
        console_layout.addWidget(self.console)
        
        self.style_section(self.console_section)
        right_panel.addWidget(self.console_section, 1)

        content.addLayout(right_panel, 1)
        main_layout.addLayout(content)
        
        self.setCentralWidget(main_widget)
        self.toggle_theme()

    def style_theme_button(self, button):
        bg_color = "#2D2D2D" if self.dark_mode else "#FFFFFF"
        hover_color = "#363636" if self.dark_mode else "#F0F0F0"
        border_color = "#404040" if self.dark_mode else "#E0E0E0"
        
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 20px;
                padding: 0px;
                margin: 4px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """)

    def style_section(self, widget):
        widget.setStyleSheet(f"""
            QWidget {{
                background-color: {self.get_theme_color('section_bg')};
                border: 1px solid {self.get_theme_color('border')};
                border-radius: 8px;
            }}
        """)

    def style_combo(self, widget, color):
        widget.setStyleSheet(f"""
            QComboBox {{
                border: 1px solid {color};
                border-radius: 4px;
                padding: 5px;
                background: {self.get_theme_color('input_bg')};
                color: {self.get_theme_color('text')};
                min-height: 30px;
                background-color: {color}10;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
                border-left: 1px solid {color};
            }}
            QComboBox::down-arrow {{
                image: url(assets/dropdown_{self.get_theme_color('arrow')}.png);
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.get_theme_color('input_bg')};
                color: {self.get_theme_color('text')};
                selection-background-color: {color}40;
                selection-color: {self.get_theme_color('text')};
            }}
        """)

    def style_spinbox(self, widget, color):
        widget.setStyleSheet(f"""
            QSpinBox {{
                border: 1px solid {color};
                border-radius: 4px;
                padding: 5px;
                background: {self.get_theme_color('input_bg')};
                color: {self.get_theme_color('text')};
                min-height: 30px;
                background-color: {color}10;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                background-color: transparent;
                border: none;
            }}
        """)

    def style_action_button(self, button, style):
        colors = {
            "primary": ("#2196F3", "#1976D2"),
            "success": ("#4CAF50", "#388E3C")
        }
        normal, hover = colors[style]
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {normal};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-height: 36px;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
            QPushButton:disabled {{
                background-color: #BDBDBD;
            }}
        """)

    def style_progress(self, progress, color):
        progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {self.get_theme_color('border')};
                border-radius: 4px;
                text-align: center;
                height: 8px;
                background-color: {self.get_theme_color('input_bg')};
            }}
            QProgressBar::chunk {{
                background-color: #2196F3;
                border-radius: 3px;
            }}
        """)

    def style_console(self, console):
        console.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid {self.get_theme_color('border')};
                border-radius: 4px;
                padding: 8px;
                background: {self.get_theme_color('console_bg')};
                color: {self.get_theme_color('text')};
            }}
        """)

    def get_theme_color(self, key):
        colors = {
            'dark': {
                'window_bg': '#1E1E1E',
                'section_bg': '#2D2D2D',
                'input_bg': '#363636',
                'console_bg': '#252526',
                'border': '#404040',
                'text': '#FFFFFF',
                'secondary_text': '#AAAAAA',
                'arrow': 'light'
            },
            'light': {
                'window_bg': '#F5F5F5',
                'section_bg': '#FFFFFF',
                'input_bg': '#FFFFFF',
                'console_bg': '#FAFAFA',
                'border': '#E0E0E0',
                'text': '#424242',
                'secondary_text': '#666666',
                'arrow': 'dark'
            }
        }
        return colors['dark' if self.dark_mode else 'light'][key]

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.btn_theme.setIcon(self.theme_icons[self.dark_mode])
        self.style_theme_button(self.btn_theme)
        
        # Update window background
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {self.get_theme_color('window_bg')};
            }}
        """)

        # Update all section widgets
        for widget in self.findChildren(QWidget):
            if widget.parent() and isinstance(widget.parent(), QMainWindow):
                widget.setStyleSheet(f"""
                    QWidget {{
                        background-color: {self.get_theme_color('window_bg')};
                        color: {self.get_theme_color('text')};
                    }}
                """)

        # Update labels
        for label in self.findChildren(QLabel):
            if "title" in str(label.objectName()).lower():
                label.setStyleSheet(f"color: {self.get_theme_color('text')};")
            else:
                label.setStyleSheet(f"color: {self.get_theme_color('secondary_text')};")

        # Refresh all styled components
        for section in [self.model_section, self.button_section, self.preview_section, self.console_section]:
            self.style_section(section)
        self.style_combo(self.model_combobox, "#4CAF50")
        self.style_spinbox(self.cycle_spinbox, "#2196F3")
        self.style_console(self.console)
        
        # Update preview background
        self.preview_label.setStyleSheet(f"""
            QLabel {{
                background-color: {self.get_theme_color('section_bg')};
                border-radius: 10px;
                border: 2px solid {self.get_theme_color('border')};
            }}
        """)

        # Update progress bar
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {self.get_theme_color('border')};
                border-radius: 4px;
                text-align: center;
                height: 8px;
                background-color: {self.get_theme_color('input_bg')};
            }}
            QProgressBar::chunk {{
                background-color: #2196F3;
                border-radius: 3px;
            }}
        """)

    def log_to_console(self, message):
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
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            self.update_preview(img)
        cap.release()

    def update_preview(self, img):
        preview_width = self.preview_label.width()
        preview_height = self.preview_label.height()
        img.thumbnail((preview_width, preview_height), Image.ANTIALIAS)
        qimage = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimage))

    def start_processing(self):
        if not self.input_path:
            self.log_to_console("Error: Please select a video file first!")
            return

        model_name = self.model_combobox.currentText()
        try:
            self.model = InterpolationModel().to(self.device)
            self.model.load_state_dict(torch.load(f"models/{model_name}", map_location=self.device))
            self.model.eval()
            self.log_to_console(f"Loaded model: {model_name}")
        except Exception as e:
            self.log_to_console(f"Error: Failed to load model: {str(e)}")
            return

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
            if reply == QMessageBox.Yes: event.accept()
            else: event.ignore()
        else: event.accept()

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
            
            original_frames, original_res = self.extract_frames(self.input_path)
            self.progress_signal.emit(30)
            
            interpolated_frames = original_frames.copy()
            for cycle in range(self.num_cycles):
                self.log_signal.emit(f"Interpolating frames (Cycle {cycle + 1}/{self.num_cycles})...")
                interpolated_frames = self.generate_interpolated_frames(interpolated_frames)
                self.progress_signal.emit(30 + (60 * (cycle + 1) // self.num_cycles))
            
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
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(frame)
            idx += 1
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
            self.progress_signal.emit(int(30 + (60 * (i + 1) / total_pairs)))
        
        interpolated_frames.append(frames[-1])
        return interpolated_frames

    def run_model(self, frame1, frame2):
        frame1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).float().to(self.device) / 255.0
        frame2_tensor = torch.from_numpy(frame2).permute(2, 0, 1).float().to(self.device) / 255.0
        with torch.no_grad():
            interpolated_tensor = self.model(frame1_tensor.unsqueeze(0), frame2_tensor.unsqueeze(0))
        output = interpolated_tensor.squeeze().cpu().permute(1, 2, 0) * 255
        return output.numpy().astype(np.uint8)

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