import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QProgressBar,
    QFileDialog, QMessageBox, QTextEdit, QDialog, QSizePolicy, QSlider # Added QSlider
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QCloseEvent # Added QCloseEvent
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PIL import Image, ImageQt
# Add the project root directory to Python's module search path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import the model
from models.model import InterpolationModel

# Helper function for asset paths relative to the script location or PyInstaller bundle
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        # Assume assets are copied relative to the _MEIPASS directory
        base_path = sys._MEIPASS
    except Exception:
        # Development mode: Navigate up from src/gui/app.py to the project root
        script_dir = os.path.dirname(__file__)
        base_path = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Construct the full path relative to the determined base path
    full_path = os.path.join(base_path, relative_path)
    # print(f"Resource path for '{relative_path}': {full_path}") # Debug print
    return full_path


class VideoInterpolationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Decadence: AI Video Interpolator")
        self.setGeometry(100, 100, 1100, 750) # Adjusted size

        # Set application icon using resource_path
        try:
            app_icon_path = resource_path("assets/logo_dark.png")
            if os.path.exists(app_icon_path):
                app_icon = QIcon(app_icon_path)
                self.setWindowIcon(app_icon)
                QApplication.setWindowIcon(app_icon) # This sets the taskbar icon
            else:
                print(f"Warning: App icon not found at {app_icon_path}")
        except Exception as e:
             print(f"Error loading app icon: {e}")

        self.center_window()

        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Theme settings using resource_path
        self.dark_mode = True
        try:
            sun_icon_path = resource_path("assets/sun.png")
            moon_icon_path = resource_path("assets/moon.png")
            self.theme_icons = {
                True: QIcon(sun_icon_path) if os.path.exists(sun_icon_path) else QIcon(),
                False: QIcon(moon_icon_path) if os.path.exists(moon_icon_path) else QIcon()
            }
            if not os.path.exists(sun_icon_path): print(f"Warning: Sun icon not found at {sun_icon_path}")
            if not os.path.exists(moon_icon_path): print(f"Warning: Moon icon not found at {moon_icon_path}")
        except Exception as e:
            print(f"Error loading theme icons: {e}")
            self.theme_icons = {True: QIcon(), False: QIcon()}


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
        main_widget.setObjectName("mainWidget") # Give the main widget an object name
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
        title.setObjectName("appTitle") 
        title_container.addWidget(title)

        subtitle = QLabel("AI Video Frame Interpolation")
        subtitle.setFont(QFont("Segoe UI", 11)) 
        subtitle.setObjectName("appSubtitle") 
        title_container.addWidget(subtitle)
        # Add some vertical space for subtitle
        title_vbox = QVBoxLayout()
        title_vbox.addWidget(title)
        title_vbox.addWidget(subtitle)
        title_vbox.setSpacing(0) # No space between title and subtitle
        title_container.addLayout(title_vbox)

        title_container.addStretch(1) # Push title group left
        top_bar.addLayout(title_container)

        # Theme toggle button
        self.btn_theme = QPushButton()
        self.btn_theme.setIcon(self.theme_icons[self.dark_mode])
        self.btn_theme.setIconSize(QSize(20, 20)) 
        self.btn_theme.setFixedSize(36, 36) 
        self.btn_theme.setToolTip("Toggle Light/Dark Mode")
        self.btn_theme.setObjectName("themeToggleButton") 
        self.btn_theme.clicked.connect(self.toggle_theme)
        # self.style_theme_button(self.btn_theme) # Styling handled centrally
        top_bar.addWidget(self.btn_theme)

        main_layout.addLayout(top_bar)
        # main_layout.addSpacing(10) # Remove extra spacing here

        # Main content area
        content = QHBoxLayout()
        content.setSpacing(15)

        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel") # Add object name
        left_panel.setFixedWidth(300) # Slightly wider
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10) # Increased spacing between sections

        # Model section
        self.model_section = QWidget()
        self.model_section.setObjectName("model_section") # Keep object name
        model_layout = QVBoxLayout(self.model_section)
        model_layout.setContentsMargins(15, 15, 15, 15) # More padding
        model_layout.setSpacing(10) # Increased spacing

        model_header = QLabel("Model Settings")
        model_header.setFont(QFont("Segoe UI", 11, QFont.Bold)) 
        model_header.setObjectName("modelHeaderLabel")
        model_layout.addWidget(model_header)

        self.model_combobox = QComboBox()
        self.model_combobox.addItems(self._get_available_models())
        self.model_combobox.setObjectName("modelCombobox")
        # self.style_combo(self.model_combobox, "#4CAF50") # Styling handled centrally
        model_layout.addWidget(self.model_combobox)

        cycles_label = QLabel("Interpolation Cycles")
        cycles_label.setFont(QFont("Segoe UI", 10)) # Keep font size
        cycles_label.setObjectName("cyclesLabel")
        model_layout.addWidget(cycles_label)

        self.cycle_spinbox = QSpinBox()
        self.cycle_spinbox.setRange(1, 3)
        self.cycle_spinbox.setValue(1)
        self.cycle_spinbox.setObjectName("cycleSpinbox")
        # self.style_spinbox(self.cycle_spinbox, "#2196F3") # Styling handled centrally
        model_layout.addWidget(self.cycle_spinbox)

        # self.style_section(self.model_section) # Styling handled centrally
        left_layout.addWidget(self.model_section)

        # Action buttons
        self.button_section = QWidget()
        self.button_section.setObjectName("button_section") # Add object name
        button_layout = QVBoxLayout(self.button_section)
        button_layout.setContentsMargins(15, 15, 15, 15) # More padding
        button_layout.setSpacing(12) # Increased spacing

        self.btn_select = QPushButton(" Select Input Video") # Add space for icon
        try:
            folder_icon_path = resource_path("assets/folder.png")
            if os.path.exists(folder_icon_path):
                self.btn_select.setIcon(QIcon(folder_icon_path))
            else: print(f"Warning: Folder icon not found at {folder_icon_path}")
        except Exception as e: print(f"Error loading folder icon: {e}")
        self.btn_select.setObjectName("selectButton") # Add object name
        self.btn_select.clicked.connect(self.select_file)
        button_layout.addWidget(self.btn_select)

        self.btn_process = QPushButton(" Start Processing") # Add space for icon
        try:
            play_icon_path = resource_path("assets/play.png")
            if os.path.exists(play_icon_path):
                self.btn_process.setIcon(QIcon(play_icon_path))
            else: print(f"Warning: Play icon not found at {play_icon_path}")
        except Exception as e: print(f"Error loading play icon: {e}")
        self.btn_process.setObjectName("processButton") # Add object name
        self.btn_process.clicked.connect(self.start_processing)
        button_layout.addWidget(self.btn_process)

        # self.style_section(self.button_section) # Styling handled centrally
        left_layout.addWidget(self.button_section)
        left_layout.addStretch()

        

        content.addWidget(left_panel)

        # Right panel - Preview and Console
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        # Preview area
        self.preview_section = QWidget()
        self.preview_section.setObjectName("preview_section") 
        preview_layout = QVBoxLayout(self.preview_section)
        preview_layout.setContentsMargins(15, 15, 15, 15) # More padding

        self.preview_label = QLabel("Load a video to see preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(300) # Adjusted height
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Make preview label expand
        preview_layout.addWidget(self.preview_label)

        # Playback controls
        # Playback controls layout (Slider + Button)
        playback_layout = QVBoxLayout() # Use QVBoxLayout for slider above button
        playback_layout.setSpacing(5)

        # Seek Slider
        self.preview_slider = QSlider(Qt.Horizontal)
        self.preview_slider.setObjectName("previewSlider")
        self.preview_slider.setRange(0, 0) # Initial range
        self.preview_slider.setEnabled(False) # Disabled initially
        self.preview_slider.sliderMoved.connect(self.preview_seek) # Seek when user moves slider
        self.preview_slider.valueChanged.connect(self.preview_slider_changed) # Update frame when value changes programmatically (less frequent)
        playback_layout.addWidget(self.preview_slider)

        # Button controls layout
        self.playback_controls = QHBoxLayout()
        self.playback_controls.setAlignment(Qt.AlignCenter) # Center controls
        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.setObjectName("playPauseButton") # Add object name
        self.btn_play_pause.setCheckable(True) # Make it a toggle button
        self.btn_play_pause.toggled.connect(self.toggle_playback)
        self.btn_play_pause.setEnabled(False) # Disabled initially
        self.playback_controls.addStretch() # Push button to center
        self.playback_controls.addWidget(self.btn_play_pause)
        self.playback_controls.addStretch() # Push button to center
        playback_layout.addLayout(self.playback_controls) # Add button layout below slider

        preview_layout.addLayout(playback_layout) # Add combined controls layout

        # self.style_section(self.preview_section) # Styling handled centrally
        right_panel.addWidget(self.preview_section, 2) # Give preview more space

        # Playback state
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.next_preview_frame)
        self.preview_cap = None
        self.preview_fps = 30 # Default FPS

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0) # Start at 0
        self.progress.setTextVisible(False) 
        self.progress.setObjectName("progressBar")
        # self.style_progress(self.progress, "#2196F3") # Styling handled centrally
        right_panel.addWidget(self.progress)

        # Console area
        self.console_section = QWidget()
        self.console_section.setObjectName("console_section") 
        console_layout = QVBoxLayout(self.console_section)
        console_layout.setContentsMargins(15, 15, 15, 10) # Adjusted padding
        console_layout.setSpacing(8) # Increased spacing

        console_header = QLabel("Console Output")
        console_header.setFont(QFont("Segoe UI", 11, QFont.Bold))
        console_header.setObjectName("consoleHeaderLabel")
        console_layout.addWidget(console_header)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        # self.console.setFont(QFont("Consolas", 10)) # Font set in stylesheet
        self.console.setObjectName("console") # Add object name
        # self.style_console(self.console) # Styling handled centrally
        console_layout.addWidget(self.console)

        # self.style_section(self.console_section) # Styling handled centrally
        right_panel.addWidget(self.console_section, 1) # Give console less relative space

        content.addLayout(right_panel, 3) # Adjust right panel stretch factor
        main_layout.addLayout(content)

        self.setCentralWidget(main_widget)
        self.update_styles() # Apply initial styles

    # Removed old individual styling methods as QSS handles this centrally now

    def get_theme_color(self, key):
        # --- REFINED THEME COLORS ---
        colors = {
            'dark': {
                'window_bg': '#2B2B2B', # Slightly lighter dark bg
                'section_bg': '#3C3C3C', # Darker section bg
                'input_bg': '#454545', # Lighter input bg
                'console_bg': '#252525', # Darker console
                'border': '#555555', # More prominent border
                'text': '#DCDCDC', # Slightly dimmer text
                'secondary_text': '#9E9E9E', # Dimmer secondary
                'title_text': '#FFFFFF',
                'arrow': 'light',
                'theme_button_bg': '#3C3C3C',
                'theme_button_hover': '#4A4A4A',
                'theme_button_border': '#555555',
                'primary_button_bg': '#0A6ED8', # Slightly different blue
                'primary_button_hover': '#085AB0',
                'success_button_bg': '#2A9D8F', # Teal green
                'success_button_hover': '#218074',
                'disabled_button_bg': '#4F4F4F',
                'disabled_button_text': '#888888',
                'progress_chunk': '#2A9D8F', # Match success button
                'combo_border': '#606060',
                'spin_border': '#606060',
                'preview_border': '#606060',
                'preview_bg': '#333333', # Darker preview bg
            },
            'light': {
                'window_bg': '#F5F5F5', # Off-white bg
                'section_bg': '#FFFFFF',
                'input_bg': '#FFFFFF',
                'console_bg': '#FAFAFA',
                'border': '#E0E0E0', # Softer border
                'text': '#424242', # Dark gray text
                'secondary_text': '#757575', # Medium gray
                'title_text': '#212121', # Near black
                'arrow': 'dark',
                'theme_button_bg': '#FFFFFF',
                'theme_button_hover': '#F0F0F0',
                'theme_button_border': '#E0E0E0',
                'primary_button_bg': '#0A6ED8',
                'primary_button_hover': '#085AB0',
                'success_button_bg': '#2A9D8F',
                'success_button_hover': '#218074',
                'disabled_button_bg': '#E0E0E0',
                'disabled_button_text': '#BDBDBD',
                'progress_chunk': '#2A9D8F',
                'combo_border': '#BDBDBD', # Lighter input border
                'spin_border': '#BDBDBD',
                'preview_border': '#E0E0E0',
                'preview_bg': '#FFFFFF',
            }
        }
        return colors['dark' if self.dark_mode else 'light'][key]

    def update_styles(self):
        # Centralized styling function using QSS
        theme = 'dark' if self.dark_mode else 'light'
        # Load arrow icons based on theme
        arrow_theme = self.get_theme_color('arrow')
        dropdown_arrow_path = resource_path(f"assets/dropdown_{arrow_theme}.png")
        up_arrow_path = resource_path(f"assets/up_arrow_{arrow_theme}.png")
        down_arrow_path = resource_path(f"assets/down_arrow_{arrow_theme}.png")

        dropdown_arrow_url = f"url({dropdown_arrow_path.replace(os.sep, '/')})" if os.path.exists(dropdown_arrow_path) else "url()"
        up_arrow_url = f"url({up_arrow_path.replace(os.sep, '/')})" if os.path.exists(up_arrow_path) else "url()"
        down_arrow_url = f"url({down_arrow_path.replace(os.sep, '/')})" if os.path.exists(down_arrow_path) else "url()"

        if not os.path.exists(dropdown_arrow_path): print(f"Warning: Dropdown arrow icon not found at {dropdown_arrow_path}")
        if not os.path.exists(up_arrow_path): print(f"Warning: Up arrow icon not found at {up_arrow_path}")
        if not os.path.exists(down_arrow_path): print(f"Warning: Down arrow icon not found at {down_arrow_path}")

        # Generate the stylesheet string
        style_sheet = f"""
            QMainWindow, QWidget#mainWidget {{ /* Apply window background to main window AND the central widget */
                background-color: {self.get_theme_color('window_bg')};
            }}
            QWidget {{ /* Default for other widgets unless overridden */
                color: {self.get_theme_color('text')};
                background-color: transparent; 
            }}
            QLabel {{
                background-color: transparent;
                color: {self.get_theme_color('secondary_text')};
            }}
            QLabel#appTitle {{
                color: {self.get_theme_color('title_text')};
                font-weight: bold;
            }}
             QLabel#appSubtitle {{
                color: {self.get_theme_color('secondary_text')};
            }}
            QLabel#modelHeaderLabel, QLabel#cyclesLabel, QLabel#consoleHeaderLabel {{
                 color: {self.get_theme_color('text')};
                 font-weight: bold;
                 padding-bottom: 2px; /* Space below labels */
            }}
            QPushButton#themeToggleButton {{
                background-color: {self.get_theme_color('theme_button_bg')};
                border: 1px solid {self.get_theme_color('theme_button_border')};
                border-radius: 18px; /* Round */
                padding: 0px;
            }}
            QPushButton#themeToggleButton:hover {{
                background-color: {self.get_theme_color('theme_button_hover')};
            }}
            /* Section Styling */
            QWidget#model_section, QWidget#button_section, 
            QWidget#preview_section, QWidget#console_section {{
                background-color: {self.get_theme_color('section_bg')};
                border: 1px solid {self.get_theme_color('border')};
                border-radius: 6px; /* Slightly less rounded */
            }}
            /* Input Styling */
            QComboBox#modelCombobox {{
                border: 1px solid {self.get_theme_color('combo_border')};
                border-radius: 4px;
                padding: 5px 8px;
                background-color: {self.get_theme_color('input_bg')};
                color: {self.get_theme_color('text')};
                min-height: 28px;
            }}
            QComboBox#modelCombobox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: {self.get_theme_color('combo_border')};
                border-left-style: solid; 
                border-top-right-radius: 3px; 
                border-bottom-right-radius: 3px;
                background-color: transparent;
            }}
            QComboBox#modelCombobox::down-arrow {{
                image: {dropdown_arrow_url};
                width: 12px;
                height: 12px;
            }}
            QComboBox#modelCombobox QAbstractItemView {{ /* Dropdown list */
                background-color: {self.get_theme_color('input_bg')};
                color: {self.get_theme_color('text')};
                border: 1px solid {self.get_theme_color('border')};
                selection-background-color: {self.get_theme_color('primary_button_bg')};
                selection-color: white; /* Text color on selection */
                padding: 4px;
            }}
            QSpinBox#cycleSpinbox {{
                border: 1px solid {self.get_theme_color('spin_border')};
                border-radius: 4px;
                padding: 5px 8px;
                background-color: {self.get_theme_color('input_bg')};
                color: {self.get_theme_color('text')};
                min-height: 28px;
            }}
            QSpinBox#cycleSpinbox::up-button, QSpinBox#cycleSpinbox::down-button {{
                 subcontrol-origin: border;
                 width: 16px;
                 border-left-width: 1px;
                 border-left-color: {self.get_theme_color('spin_border')};
                 border-left-style: solid;
                 border-radius: 0px;
                 background-color: transparent;
            }}
            QSpinBox#cycleSpinbox::up-button {{
                subcontrol-position: top right;
                border-top-right-radius: 3px;
            }}
            QSpinBox#cycleSpinbox::up-arrow {{
                image: {up_arrow_url};
                width: 10px; /* Adjust size as needed */
                height: 10px; /* Adjust size as needed */
            }}
            QSpinBox#cycleSpinbox::down-button {{
                subcontrol-position: bottom right;
                border-bottom-right-radius: 3px;
            }}
            QSpinBox#cycleSpinbox::down-arrow {{
                image: {down_arrow_url};
                width: 10px; /* Adjust size as needed */
                height: 10px; /* Adjust size as needed */
            }}
            /* Button Styling */
            QPushButton#selectButton, QPushButton#processButton {{
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-height: 32px;
                icon-size: 16px; /* Ensure icon size is consistent */
                text-align: center; /* Center text and icon */
            }}
             QPushButton#selectButton {{
                 background-color: {self.get_theme_color('primary_button_bg')};
             }}
             QPushButton#selectButton:hover {{
                 background-color: {self.get_theme_color('primary_button_hover')};
             }}
             QPushButton#selectButton:disabled {{
                 background-color: {self.get_theme_color('disabled_button_bg')};
                 color: {self.get_theme_color('disabled_button_text')};
             }}
             QPushButton#processButton {{
                 background-color: {self.get_theme_color('success_button_bg')};
             }}
             QPushButton#processButton:hover {{
                 background-color: {self.get_theme_color('success_button_hover')};
             }}
             QPushButton#processButton:disabled {{
                 background-color: {self.get_theme_color('disabled_button_bg')};
                 color: {self.get_theme_color('disabled_button_text')};
             }}
            QPushButton#playPauseButton {{ /* Style for Play/Pause */
                 background-color: {self.get_theme_color('primary_button_bg')};
                 color: white;
                 border: none;
                 border-radius: 4px;
                 padding: 6px 12px; /* Smaller padding */
                 font-weight: bold;
                 min-height: 28px;
                 min-width: 60px; /* Fixed width */
            }}
             QPushButton#playPauseButton:hover {{
                 background-color: {self.get_theme_color('primary_button_hover')};
             }}
             QPushButton#playPauseButton:checked {{ /* Style when 'Paused' */
                 background-color: {self.get_theme_color('success_button_bg')}; /* Use success color for pause */
             }}
             QPushButton#playPauseButton:checked:hover {{
                 background-color: {self.get_theme_color('success_button_hover')};
             }}
             QPushButton#playPauseButton:disabled {{
                 background-color: {self.get_theme_color('disabled_button_bg')};
                 color: {self.get_theme_color('disabled_button_text')};
             }}
            /* Progress Bar Styling */
            QProgressBar#progressBar {{
                border: 1px solid {self.get_theme_color('border')};
                border-radius: 5px; /* More rounded */
                text-align: center;
                height: 10px; 
                background-color: {self.get_theme_color('input_bg')};
                color: {self.get_theme_color('text')}; 
            }}
            QProgressBar#progressBar::chunk {{
                background-color: {self.get_theme_color('progress_chunk')};
                border-radius: 4px; /* Rounded chunk */
                margin: 1px; 
            }}
            /* Console Styling */
            QTextEdit#console {{
                border: 1px solid {self.get_theme_color('border')};
                border-radius: 4px;
                padding: 8px;
                background-color: {self.get_theme_color('console_bg')};
                color: {self.get_theme_color('text')};
                font-family: Consolas, Courier New, monospace; 
            }}
            /* Preview Label Styling */
            QLabel#previewLabel {{ 
                background-color: {self.get_theme_color('preview_bg')}; 
                border-radius: 4px;
                border: 1px dashed {self.get_theme_color('preview_border')}; 
                color: {self.get_theme_color('secondary_text')}; 
                font-style: italic; /* Italic placeholder text */
            }}
            /* Tooltip Styling */
            QToolTip {{
                color: {self.get_theme_color('text')};
                background-color: {self.get_theme_color('section_bg')};
                border: 1px solid {self.get_theme_color('border')};
                padding: 4px;
                border-radius: 3px;
            }}

            /* Video Player Dialog Styling */
            QDialog#videoPlayerWindow {{
                background-color: {self.get_theme_color('window_bg')};
            }}
            QDialog#videoPlayerWindow QLabel#videoTitleLabel {{
                color: {self.get_theme_color('text')};
                font-weight: bold;
                background-color: transparent;
            }}
             QDialog#videoPlayerWindow QLabel#videoLabel {{
                background-color: {self.get_theme_color('preview_bg')}; /* Use preview bg for consistency */
                border-radius: 4px;
                border: 1px solid {self.get_theme_color('preview_border')}; /* Use preview border */
                color: {self.get_theme_color('secondary_text')};
            }}
            QDialog#videoPlayerWindow QSlider#seekSlider::groove:horizontal {{
                border: 1px solid {self.get_theme_color('border')};
                height: 8px;
                background: {self.get_theme_color('input_bg')};
                margin: 2px 0;
                border-radius: 4px;
            }}
            QDialog#videoPlayerWindow QSlider#seekSlider::handle:horizontal {{
                background: {self.get_theme_color('primary_button_bg')};
                border: 1px solid {self.get_theme_color('primary_button_bg')};
                width: 14px;
                margin: -4px 0; /* handle is placed vertically centered */
                border-radius: 7px;
            }}
            QDialog#videoPlayerWindow QSlider#seekSlider::add-page:horizontal {{
                background: {self.get_theme_color('input_bg')};
                border-radius: 4px;
            }}
            QDialog#videoPlayerWindow QSlider#seekSlider::sub-page:horizontal {{
                background: {self.get_theme_color('progress_chunk')}; /* Use progress chunk color */
                border-radius: 4px;
            }}
            /* Inherit playPauseButton styling from main window via object name */

        """
        # Apply the stylesheet to the main window instance AND globally for dialogs
        QApplication.instance().setStyleSheet(style_sheet)
        # self.setStyleSheet(style_sheet) # Applying globally might be better for dialogs
        # Force style refresh on specific widgets if needed
        self.update()

    # Removed style_theme_button(self, button)
    # Removed style_section(self, widget)
    # Removed style_combo(self, widget, color)
    # Removed style_spinbox(self, widget, color)
    # Removed style_action_button(self, button, style)
    # Removed style_progress(self, progress, color)
    # Removed style_console(self, console)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        # Ensure theme icon exists before setting
        if self.theme_icons[self.dark_mode]:
            self.btn_theme.setIcon(self.theme_icons[self.dark_mode])
        self.update_styles() # Apply the updated styles

    def log_to_console(self, message):
        self.console.append(message)
        self.console.ensureCursorVisible()

    def select_file(self):
        self.input_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if self.input_path:
            self.log_to_console(f"Selected video: {self.input_path}")
            self.stop_playback() # Stop previous playback if any
            self.show_preview(self.input_path)
            self.btn_play_pause.setEnabled(True) # Enable play button
            self.btn_play_pause.setChecked(False) # Ensure it starts as 'Play'

    def show_preview(self, video_path):
        # Release previous capture if exists
        if self.preview_cap:
            self.preview_cap.release()
            self.preview_cap = None

        self.preview_cap = cv2.VideoCapture(video_path)
        if not self.preview_cap.isOpened():
            self.log_to_console(f"Error: Could not open video file for preview: {video_path}")
            self.preview_cap = None
            self.btn_play_pause.setEnabled(False)
            return

        self.preview_fps = self.preview_cap.get(cv2.CAP_PROP_FPS)
        if self.preview_fps <= 0:
            self.log_to_console("Warning: Could not get video FPS, defaulting to 30.")
            self.preview_fps = 30 # Default if FPS read fails

        # Get frame count for slider
        self.preview_frame_count = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.preview_frame_count > 0:
            self.preview_slider.setRange(0, self.preview_frame_count - 1)
            self.preview_slider.setEnabled(True)
        else:
            self.preview_slider.setRange(0, 0)
            self.preview_slider.setEnabled(False)
            self.log_to_console("Warning: Could not get video frame count.")

        # Show the first frame immediately
        self.preview_slider.setValue(0) # Ensure slider starts at 0
        self.next_preview_frame(show_first=True)

    def update_preview(self, frame_rgb):
        # Convert numpy array (RGB) to QPixmap for display
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        qimage = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Scale pixmap while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Only update if preview label exists and is valid
        if hasattr(self, 'preview_label') and self.preview_label:
            self.preview_label.setText("") # Clear placeholder text if showing frame
            self.preview_label.setPixmap(scaled_pixmap)
            self.preview_label.setAlignment(Qt.AlignCenter) # Keep centered

    def next_preview_frame(self, show_first=False):
        if not self.preview_cap or not self.preview_cap.isOpened():
            self.stop_playback()
            return

        ret, frame = self.preview_cap.read()
        if ret:
            # Update slider position
            current_frame_index = int(self.preview_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            # Block signals to prevent triggering valueChanged connection unnecessarily
            self.preview_slider.blockSignals(True)
            self.preview_slider.setValue(current_frame_index)
            self.preview_slider.blockSignals(False)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.update_preview(frame_rgb)
        else:
            # End of video or error, stop playback and reset
            if not show_first: # Don't log error if just showing first frame failed
                 self.log_to_console("Preview finished or failed to read frame.")
            self.stop_playback()
            # Optionally rewind to show the first frame again after playback ends
            if self.preview_cap and self.preview_cap.isOpened():
                self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.preview_slider.setValue(0) # Reset slider
                ret, frame = self.preview_cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.update_preview(frame_rgb)
                else:
                     # If even first frame fails, show placeholder
                     self.preview_label.setText("Could not load preview")
                     self.preview_label.setAlignment(Qt.AlignCenter)
            else:
                 self.preview_label.setText("Could not load preview")
                 self.preview_label.setAlignment(Qt.AlignCenter)


    def toggle_playback(self, checked):
        if checked: # Button is pressed -> Pause state
            self.start_playback()
            self.btn_play_pause.setText("Pause")
        else: # Button is not pressed -> Play state
            self.stop_playback()
            self.btn_play_pause.setText("Play")

    def start_playback(self):
        if self.preview_cap and self.preview_cap.isOpened() and not self.playback_timer.isActive():
            interval = int(1000 / self.preview_fps) # Interval in milliseconds
            self.playback_timer.start(interval)
            self.log_to_console("Preview playback started.")

    def stop_playback(self):
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self.log_to_console("Preview playback stopped.")
        # Ensure button is in 'Play' state visually if stopped externally
        if self.btn_play_pause.isChecked():
             self.btn_play_pause.setChecked(False) # This will trigger toggle_playback
             # self.btn_play_pause.setText("Play") # Text is set within toggle_playback

    def preview_seek(self, position):
        """Handles seeking when the user manually moves the slider."""
        if self.preview_cap and self.preview_cap.isOpened():
            was_playing = self.playback_timer.isActive()
            if was_playing:
                self.stop_playback() # Pause playback while seeking

            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = self.preview_cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_preview(frame_rgb)
                # Update slider value precisely after seek (in case set didn't land exactly)
                actual_pos = int(self.preview_cap.get(cv2.CAP_PROP_POS_FRAMES)) -1
                self.preview_slider.blockSignals(True)
                self.preview_slider.setValue(actual_pos)
                self.preview_slider.blockSignals(False)
            else:
                self.log_to_console(f"Could not seek to frame {position}")
                # Optionally reset to beginning or end? For now, just log.

            # Optional: Resume playback if it was active before seeking
            # if was_playing:
            #     self.start_playback()

    def preview_slider_changed(self, position):
        """Handles seeking when the slider value changes programmatically (e.g., during playback)."""
        # This is primarily to update the frame if the slider is set externally,
        # but seeking is mainly handled by sliderMoved for user interaction.
        # We might not need this if sliderMoved covers all cases, but let's keep it simple for now.
        # Avoid seeking if the timer is active, as next_preview_frame handles updates.
        if self.preview_cap and self.preview_cap.isOpened() and not self.playback_timer.isActive():
             self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, position)
             ret, frame = self.preview_cap.read()
             if ret:
                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                 self.update_preview(frame_rgb)


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
            self.progress.setValue(0) # Reset progress
            self.progress.setTextVisible(True) # Show percentage during processing
            self.worker = VideoProcessingThread(
                self.input_path, self.output_path, self.num_cycles, self.model, self.device
            )
            self.worker.progress_signal.connect(self.update_progress)
            self.worker.log_signal.connect(self.log_to_console)
            self.worker.finished_signal.connect(self.on_processing_finished)
            self.stop_playback() # Stop preview playback before processing
            self.worker.start()

    def update_progress(self, value):
        self.progress.setValue(value)
        # Optional: Update text format if needed, e.g., self.progress.setFormat(f"{value}%")

    def on_processing_finished(self, success):
        self.processing = False
        self.btn_process.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.progress.setTextVisible(False) # Hide percentage text after processing
        self.progress.setValue(100 if success else 0) # Show full or reset bar
        
        if success:
            self.log_to_console("Processing complete!")
            reply = QMessageBox.information(
                self, "Success", 
                "Video processing completed!\nDo you want to play the output video?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.play_output_video(self.output_path)
        else:
            QMessageBox.critical(self, "Error", "Processing failed. Check console for details.")
        # Reset preview state after processing
        self.btn_play_pause.setEnabled(False)
        self.preview_slider.setEnabled(False)
        self.preview_slider.setValue(0)
        self.stop_playback()
        if self.preview_cap:
            self.preview_cap.release()
            self.preview_cap = None
        self.preview_label.setText("Load a video to see preview")
        self.preview_label.setAlignment(Qt.AlignCenter)


    def closeEvent(self, event):
        self.stop_playback() # Ensure playback stops on close
        if self.preview_cap: # Release video capture resource
            self.preview_cap.release()

        if self.processing:
            reply = QMessageBox.question(
                self, "Quit", "Processing in progress. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes: event.accept()
            else: event.ignore()
        else: event.accept()

    def play_output_video(self, output_video_path):
        # Ensure the video file exists before trying to play
        if not os.path.exists(output_video_path):
            QMessageBox.warning(self, "File Not Found", f"Output video not found at:\n{output_video_path}")
            return

        # Pass both original input path and the new output path
        player_dialog = VideoPlayerWindow(self.input_path, output_video_path, self)
        player_dialog.exec_() # Show the dialog modally


class VideoPlayerWindow(QDialog):
    """A dialog window to play original and interpolated videos side-by-side."""
    def __init__(self, original_video_path, interpolated_video_path, parent=None):
        super().__init__(parent)
        self.original_video_path = original_video_path
        self.interpolated_video_path = interpolated_video_path
        self.original_cap = None
        self.interpolated_cap = None
        self.timer = QTimer(self)
        self.fps = 30 # Default

        self.init_ui()
        self.load_videos()

    def init_ui(self):
        self.setWindowTitle("Video Comparison Player")
        self.setObjectName("videoPlayerWindow") # Add object name for styling
        self.setMinimumSize(1000, 600) # Set a minimum size for side-by-side

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15) # Add padding
        main_layout.setSpacing(15) # Add spacing

        # Layout for side-by-side video display
        video_display_layout = QHBoxLayout()
        video_display_layout.setSpacing(15) # Add spacing between videos

        # Original video label
        original_label_container = QVBoxLayout()
        original_label_container.setAlignment(Qt.AlignCenter)
        original_label_container.setSpacing(5) # Space between title and video

        original_title = QLabel("Original Video")
        original_title.setAlignment(Qt.AlignCenter)
        original_title.setObjectName("videoTitleLabel") # Add object name
        original_label_container.addWidget(original_title)

        self.original_video_label = QLabel("Loading original video...")
        self.original_video_label.setAlignment(Qt.AlignCenter)
        self.original_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_video_label.setObjectName("videoLabel") # Add object name
        # self.original_video_label.setStyleSheet("background-color: black; color: white;") # Styling via QSS
        original_label_container.addWidget(self.original_video_label)
        video_display_layout.addLayout(original_label_container)


        # Interpolated video label
        interpolated_label_container = QVBoxLayout()
        interpolated_label_container.setAlignment(Qt.AlignCenter)
        interpolated_label_container.setSpacing(5) # Space between title and video

        interpolated_title = QLabel("Interpolated Video")
        interpolated_title.setAlignment(Qt.AlignCenter)
        interpolated_title.setObjectName("videoTitleLabel") # Add object name
        interpolated_label_container.addWidget(interpolated_title)

        self.interpolated_video_label = QLabel("Loading interpolated video...")
        self.interpolated_video_label.setAlignment(Qt.AlignCenter)
        self.interpolated_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.interpolated_video_label.setObjectName("videoLabel") # Add object name
        # self.interpolated_video_label.setStyleSheet("background-color: black; color: white;") # Styling via QSS
        interpolated_label_container.addWidget(self.interpolated_video_label)
        video_display_layout.addLayout(interpolated_label_container)

        main_layout.addLayout(video_display_layout)

        # Playback controls layout (Slider + Button)
        controls_container = QVBoxLayout()
        controls_container.setSpacing(5)

        # Seek Slider
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setObjectName("seekSlider") # Add object name for styling
        self.seek_slider.setRange(0, 0) # Initial range
        self.seek_slider.setEnabled(False) # Disabled initially
        self.seek_slider.sliderMoved.connect(self.seek_videos) # Seek when user moves slider
        # self.seek_slider.valueChanged.connect(self.slider_changed) # Optional: Handle programmatic changes
        controls_container.addWidget(self.seek_slider)

        # Button controls layout
        self.controls_layout = QHBoxLayout()
        self.controls_layout.setSpacing(10) # Add spacing to controls
        self.controls_layout.setAlignment(Qt.AlignCenter) # Center controls

        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.setCheckable(True)
        self.btn_play_pause.toggled.connect(self.toggle_playback)
        self.btn_play_pause.setEnabled(False) # Disabled until video loads
        self.btn_play_pause.setObjectName("playPauseButton") # Add object name

        self.controls_layout.addStretch()
        self.controls_layout.addWidget(self.btn_play_pause)
        self.controls_layout.addStretch()
        controls_container.addLayout(self.controls_layout) # Add buttons below slider

        main_layout.addLayout(controls_container) # Add combined controls layout

        self.timer.timeout.connect(self.next_frame)
        self.update_styles() # Apply initial styles

    def update_styles(self):
        # Call the parent's update_styles method to apply the theme
        if self.parent() and hasattr(self.parent(), 'update_styles'):
            self.parent().update_styles()
            # Re-apply specific styles for the dialog if needed, though parent's QSS should handle it
            # self.setStyleSheet(...) # Can add dialog-specific styles here if necessary

    def load_videos(self):
        if self.original_cap: self.original_cap.release()
        if self.interpolated_cap: self.interpolated_cap.release()

        # Attempt to open both video files
        self.original_cap = cv2.VideoCapture(self.original_video_path)
        self.interpolated_cap = cv2.VideoCapture(self.interpolated_video_path)

        # Check if both videos opened successfully
        original_opened = self.original_cap.isOpened()
        interpolated_opened = self.interpolated_cap.isOpened()

        if not original_opened:
            self.original_video_label.setText(f"Error: Could not open original video:\n{self.original_video_path}")
            self.btn_play_pause.setEnabled(False)
            if interpolated_opened: # Release interpolated cap if original failed
                self.interpolated_cap.release()
                self.interpolated_cap = None
            return

        if not interpolated_opened:
            self.interpolated_video_label.setText(f"Error: Could not open interpolated video:\n{self.interpolated_video_path}")
            self.btn_play_pause.setEnabled(False)
            if original_opened: # Release original cap if interpolated failed
                self.original_cap.release()
                self.original_cap = None
            return

        # Get original video FPS
        self.original_fps = self.original_cap.get(cv2.CAP_PROP_FPS)
        if self.original_fps <= 0: self.original_fps = 30 # Fallback FPS

        # Get interpolated video FPS
        self.interpolated_fps = self.interpolated_cap.get(cv2.CAP_PROP_FPS)
        if self.interpolated_fps <= 0: self.interpolated_fps = 30 # Fallback FPS

        # Use the interpolated video's FPS for playback timing and seek bar range
        self.fps = self.interpolated_fps

        # Get frame count for slider (use interpolated video's count)
        self.frame_count = int(self.interpolated_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_count > 0:
            self.seek_slider.setRange(0, self.frame_count - 1)
            self.seek_slider.setEnabled(True)
        else:
            self.seek_slider.setRange(0, 0)
            self.seek_slider.setEnabled(False)
            self.parent().log_to_console("Warning: Could not get interpolated video frame count for seek bar.")


        self.btn_play_pause.setEnabled(True)
        self.btn_play_pause.setChecked(False) # Start in 'Play' state
        self.btn_play_pause.setText("Play")
        self.seek_slider.setValue(0) # Ensure slider starts at 0
        self.next_frame(show_first=True) # Show the first frame

    def next_frame(self, show_first=False):
        # Ensure both video captures are open
        if not self.original_cap or not self.original_cap.isOpened() or \
           not self.interpolated_cap or not self.interpolated_cap.isOpened():
            self.stop_playback()
            return

        # Read the next frame from the interpolated video (timer is based on its FPS)
        ret_interp, frame_interp = self.interpolated_cap.read()

        if ret_interp:
            # Get current position (frame index and timestamp) of interpolated video
            current_interp_frame_index = int(self.interpolated_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            current_interp_msec = self.interpolated_cap.get(cv2.CAP_PROP_POS_MSEC)

            # Update slider position
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(current_interp_frame_index)
            self.seek_slider.blockSignals(False)

            # Seek the original video to the corresponding timestamp
            # Use set with MSEC; note that seeking might not be perfectly accurate
            self.original_cap.set(cv2.CAP_PROP_POS_MSEC, current_interp_msec)
            ret_orig, frame_orig = self.original_cap.read()

            # It's possible the read after seek fails or gets the wrong frame if seek isn't precise.
            # A potential improvement would be to read the frame *before* the target MSEC
            # and then read frames until the target MSEC is passed, but that adds complexity.
            # For now, we'll use the frame read immediately after the MSEC seek.

            frame_orig_rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB) if ret_orig else None
            frame_interp_rgb = cv2.cvtColor(frame_interp, cv2.COLOR_BGR2RGB) # Already checked ret_interp

            self.update_display(frame_orig_rgb, frame_interp_rgb)

            if not ret_orig and not show_first:
                 self.parent().log_to_console(f"Error reading original frame near timestamp {current_interp_msec:.2f} ms.")
                 # Keep playing interpolated video even if original fails? Yes.

        else:
            # End of interpolated video or error reading interpolated frame
            if not show_first: # Avoid logging error if just showing the first frame failed
                 self.parent().log_to_console("Interpolated video playback finished or failed to read frame.")
            self.stop_playback()
            # Rewind both videos to show first frame again if possible
            if self.original_cap and self.original_cap.isOpened():
                self.original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.interpolated_cap and self.interpolated_cap.isOpened():
                self.interpolated_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.seek_slider.setValue(0) # Reset slider

            # Attempt to show the first frame again after rewinding
            ret_orig, frame_orig = self.original_cap.read() if self.original_cap and self.original_cap.isOpened() else (False, None)
            ret_interp, frame_interp = self.interpolated_cap.read() if self.interpolated_cap and self.interpolated_cap.isOpened() else (False, None)

            frame_orig_rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB) if ret_orig else None
            frame_interp_rgb = cv2.cvtColor(frame_interp, cv2.COLOR_BGR2RGB) if ret_interp else None
            self.update_display(frame_orig_rgb, frame_interp_rgb)

            # If even first frame fails after rewind, show error text
            if not ret_orig:
                self.original_video_label.setText("Could not read original video frame.")
            if not ret_interp:
                self.interpolated_video_label.setText("Could not read interpolated video frame.")


    def update_display(self, frame_orig_rgb, frame_interp_rgb):
        # Update original video label
        if frame_orig_rgb is not None:
            height_orig, width_orig, channel_orig = frame_orig_rgb.shape
            bytes_per_line_orig = 3 * width_orig
            qimage_orig = QImage(frame_orig_rgb.data, width_orig, height_orig, bytes_per_line_orig, QImage.Format_RGB888)
            pixmap_orig = QPixmap.fromImage(qimage_orig)
            # Scale pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap_orig = pixmap_orig.scaled(self.original_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_video_label.setText("") # Clear placeholder
            self.original_video_label.setPixmap(scaled_pixmap_orig)
        else:
            # Optionally display placeholder if frame is None
            self.original_video_label.setText("Original frame unavailable")

        # Update interpolated video label
        if frame_interp_rgb is not None:
            height_interp, width_interp, channel_interp = frame_interp_rgb.shape
            bytes_per_line_interp = 3 * width_interp
            qimage_interp = QImage(frame_interp_rgb.data, width_interp, height_interp, bytes_per_line_interp, QImage.Format_RGB888)
            pixmap_interp = QPixmap.fromImage(qimage_interp)
            # Scale pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap_interp = pixmap_interp.scaled(self.interpolated_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.interpolated_video_label.setText("") # Clear placeholder
            self.interpolated_video_label.setPixmap(scaled_pixmap_interp)
        else:
            # Optionally display placeholder if frame is None
            self.interpolated_video_label.setText("Interpolated frame unavailable")


    def toggle_playback(self, checked):
        if checked: # Button is pressed -> Pause state
            self.start_playback()
            self.btn_play_pause.setText("Pause")
        else: # Button is not pressed -> Play state
            self.stop_playback()
            self.btn_play_pause.setText("Play")

    def start_playback(self):
        if self.original_cap and self.original_cap.isOpened() and \
           self.interpolated_cap and self.interpolated_cap.isOpened() and \
           not self.timer.isActive():
            interval = int(1000 / self.fps)
            self.timer.start(interval)

    def stop_playback(self):
        if self.timer.isActive():
            self.timer.stop()
        # Ensure button is visually in 'Play' state if stopped
        if self.btn_play_pause.isChecked():
            self.btn_play_pause.setChecked(False)
            self.btn_play_pause.setText("Play")

    def seek_videos(self, position):
        """Handles seeking when the user manually moves the slider (position is interpolated frame index)."""
        if not self.original_cap or not self.original_cap.isOpened() or \
           not self.interpolated_cap or not self.interpolated_cap.isOpened():
            return

        was_playing = self.timer.isActive()
        if was_playing:
            self.stop_playback() # Pause playback while seeking

        # Seek interpolated video first to get the target timestamp
        self.interpolated_cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret_interp, frame_interp = self.interpolated_cap.read()
        target_msec = self.interpolated_cap.get(cv2.CAP_PROP_POS_MSEC) # Get timestamp after seek

        # Seek original video to the timestamp
        self.original_cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
        ret_orig, frame_orig = self.original_cap.read()

        frame_orig_rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB) if ret_orig else None
        frame_interp_rgb = cv2.cvtColor(frame_interp, cv2.COLOR_BGR2RGB) if ret_interp else None

        self.update_display(frame_orig_rgb, frame_interp_rgb)

        # Update slider value precisely after seek (based on interpolated frame index)
        if ret_interp:
            actual_pos = int(self.interpolated_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(actual_pos)
            self.seek_slider.blockSignals(False)
        else:
             self.parent().log_to_console(f"Could not seek interpolated video to frame {position}")

        if not ret_orig:
             self.parent().log_to_console(f"Could not seek original video near timestamp {target_msec:.2f} ms.")


        # Optional: Resume playback if it was active before seeking
        # if was_playing:
        #     self.start_playback()


    def closeEvent(self, event: QCloseEvent):
        """Ensure resources are released when the dialog is closed."""
        self.stop_playback()
        if self.original_cap:
            self.original_cap.release()
            self.original_cap = None
        if self.interpolated_cap:
            self.interpolated_cap.release()
            self.interpolated_cap = None
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
