# Main application class
import cv2
import pickle
import numpy as np
import os
import threading
import time
import torch
from datetime import datetime
from tkinter import Tk, Label, Button, Frame, Canvas, Text, Scrollbar, OptionMenu, StringVar, IntVar, BooleanVar, \
    messagebox, ttk, BOTH, \
    TOP, LEFT, RIGHT, BOTTOM, X, Y, VERTICAL, HORIZONTAL, Toplevel
from PIL import Image, ImageTk

# Import our modules
from utils.gpu_utils import check_gpu_availability, gpu_adaptive_threshold, gpu_resize, diagnose_gpu
from utils.file_utils import ensure_directories_exist, load_parking_positions, save_parking_positions, save_log, \
    export_statistics
from detection.vehicle_detector import VehicleDetector
from detection.parking_detection import process_parking_frame, check_parking_space
from detection.vehicle_counting import detect_vehicles_traditional, detect_vehicles_ml, get_centroid

# Import UI modules
from ui.detection_tab import DetectionTab
from ui.setup_tab import SetupTab
from ui.log_tab import LogTab
from ui.stats_tab import StatsTab
from ui.reference_tab import ReferenceTab


class ParkingManagementSystem:
    DEFAULT_CONFIDENCE = 0.6
    DEFAULT_THRESHOLD = 500
    MIN_CONTOUR_SIZE = 40
    DEFAULT_OFFSET = 10
    DEFAULT_LINE_HEIGHT = 400

    def __init__(self, master):
        self.master = master
        self.master.title("Smart Parking Management System")
        self.master.geometry("1280x720")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize class variables
        self.running = False
        self.posList = []
        self.video_capture = None
        self.current_video = None
        self.vehicle_counter = 0
        self.matches = []  # For vehicle counting
        self.line_height = 400  # Default line height for vehicle detection
        self.min_contour_width = 40
        self.min_contour_height = 40
        self.offset = 10
        self.parking_threshold = 500  # Default threshold for parking space detection
        self.detection_mode = "parking"  # Default detection mode
        self.log_data = []  # For logging events

        # ML detection settings
        self.use_ml_detection = False
        self.ml_detector = None
        self.ml_confidence = self.DEFAULT_CONFIDENCE
        self.parking_threshold = self.DEFAULT_THRESHOLD
        self.min_contour_width = self.MIN_CONTOUR_SIZE
        self.min_contour_height = self.MIN_CONTOUR_SIZE
        self.offset = self.DEFAULT_OFFSET
        self.line_height = self.DEFAULT_LINE_HEIGHT

        # Thread safety
        self._cleanup_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.video_lock = threading.Lock()

        # GPU availability
        self.torch_gpu_available, self.cv_gpu_available = check_gpu_availability()

        # Video reference map and dimensions
        self.video_reference_map = {
            "sample5.mp4": "saming1.png",
            "Video.mp4": "videoImg.png",
            "carPark.mp4": "carParkImg.png",
            "0": "webcamImg.png",  # Default for webcam
            "newVideo1.mp4": "newRefImage1.png",
            "newVideo2.mp4": "newRefImage2.png"
        }

        # Reference dimensions
        self.reference_dimensions = {
            "carParkImg.png": (1280, 720),
            "videoImg.png": (1280, 720),
            "webcamImg.png": (640, 480),
            "newRefImage1.png": (1280, 720),
            "newRefImage2.png": (1920, 1080)
        }
        self.current_reference_image = "carParkImg.png"  # Default

        # Load resources
        self.config_dir = "config"
        self.log_dir = "logs"
        self.ensure_directories_exist()
        self.load_parking_positions()

        # Setup UI components
        self.setup_ui()

        # Start a monitoring thread to log data
        self.monitor_thread = threading.Thread(target=self.monitoring_thread, daemon=True)
        self.monitor_thread.start()

        # Diagnose GPU
        self.diagnose_gpu()

    def ensure_directories_exist(self):
        """Ensure necessary directories exist"""
        directories = [self.config_dir, self.log_dir]
        ensure_directories_exist(directories)

    def __del__(self):
        self.cleanup_resources()

    def cleanup_resources(self):
        with self._cleanup_lock:
            if hasattr(self, 'video_capture') and self.video_capture:
                self.video_capture.release()
            if hasattr(self, 'ml_detector') and self.ml_detector:
                del self.ml_detector
            torch.cuda.empty_cache()

    def load_parking_positions(self, reference_image=None):
        if reference_image is None:
            reference_image = self.current_reference_image

        positions = load_parking_positions(self.config_dir, reference_image, self.log_event)
        self.posList = positions

        # Update counters
        self.total_spaces = len(self.posList)
        self.free_spaces = 0
        self.occupied_spaces = self.total_spaces

    def setup_ui(self):
        """Set up the application's user interface"""
        # Create main container
        self.main_container = ttk.Notebook(self.master)
        self.main_container.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.detection_tab = Frame(self.main_container)
        self.setup_tab = Frame(self.main_container)
        self.log_tab = Frame(self.main_container)
        self.stats_tab = Frame(self.main_container)
        self.reference_tab = Frame(self.main_container)

        self.main_container.add(self.detection_tab, text="Detection")
        self.main_container.add(self.setup_tab, text="Setup")
        self.main_container.add(self.log_tab, text="Logs")
        self.main_container.add(self.stats_tab, text="Statistics")
        self.main_container.add(self.reference_tab, text="References")

        # Setup each tab
        self.detection_tab_controller = DetectionTab(self.detection_tab, self)
        self.setup_tab_controller = SetupTab(self.setup_tab, self)
        self.log_tab_controller = LogTab(self.log_tab, self)
        self.stats_tab_controller = StatsTab(self.stats_tab, self)
        self.reference_tab_controller = ReferenceTab(self.reference_tab, self)

        # Initialize status attributes
        self.total_spaces = len(self.posList)
        self.free_spaces = 0
        self.occupied_spaces = self.total_spaces
        self.status_info = self.detection_tab_controller.status_info

        # Initialize video processing attributes
        self.frame_count = 0
        self.frame_skip = 3
        self.frame_processing_thread = None

    # The rest of the methods would be implemented here, but for clarity
    # only a subset are shown in this example. Each method would be moved
    # to the appropriate module.

    def update_status_info(self):
        """Update the status information display"""
        if hasattr(self.detection_tab_controller, 'status_info'):
            status_text = f"Total Spaces: {self.total_spaces}\n"
            status_text += f"Free Spaces: {self.free_spaces}\n"
            status_text += f"Occupied: {self.occupied_spaces}\n"
            status_text += f"Vehicles Counted: {self.vehicle_counter}"

            # Update the status info label
            self.detection_tab_controller.status_info.config(text=status_text)

    def log_event(self, message):
        """Log an event with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        # Add to log data
        self.log_data.append(log_entry)

        # Update log display if it exists
        if hasattr(self, 'log_text'):
            self.log_text.config(state="normal")
            self.log_text.insert("end", log_entry + "\n")
            self.log_text.see("end")  # Auto-scroll to the end
            self.log_text.config(state="disabled")

    def diagnose_gpu(self):
        """Run GPU diagnostics and log results"""
        results = diagnose_gpu()
        for result in results:
            self.log_event(result)

    def on_closing(self):
        """Handle window closing event"""
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            self.running = False
            if self.video_capture is not None:
                self.video_capture.release()
            self.master.destroy()

    def monitoring_thread(self):
        """Background thread for monitoring and periodic logging"""
        while True:
            # Record stats every hour if detection is running
            if self.running:
                self.record_current_stats()

            # Sleep for an hour (3600 seconds)
            time.sleep(3600)

    # Add any remaining methods you need...
    def save_parking_spaces(self):
        """Save the current parking space positions"""
        try:
            if not self.current_reference_image:
                self.log_event("No reference image selected")
                return False

            if hasattr(self, 'posList'):
                success = save_parking_positions(
                    self.config_dir,
                    self.current_reference_image,
                    self.posList,
                    self.log_event
                )
                if success:
                    self.log_event(f"Saved {len(self.posList)} parking spaces")
                return success
            return False

        except Exception as e:
            self.log_event(f"Error saving parking spaces: {str(e)}")
            return False

    def clear_all_spaces(self):
        """Clear all defined parking spaces"""
        try:
            if hasattr(self, 'posList'):
                self.posList.clear()
                self.log_event("Cleared all parking spaces")

                # Update UI if setup tab exists
                if hasattr(self, 'setup_tab_controller'):
                    # Refresh canvas to remove drawn spaces
                    self.load_reference_image()

            return True

        except Exception as e:
            self.log_event(f"Error clearing parking spaces: {str(e)}")
            return False

    def associate_video_with_reference(self):
        """Associate a video source with the current reference image"""
        try:
            if not self.current_reference_image:
                self.log_event("No reference image selected")
                return False

            # Get list of available video sources
            video_sources = ["sample5.mp4", "Video.mp4", "0", "carPark.mp4",
                           "newVideo1.mp4", "newVideo2.mp4"]

            # Update video reference mapping
            if hasattr(self, 'detection_tab_controller'):
                selected_source = self.detection_tab_controller.video_source_var.get()
                self.video_reference_map[selected_source] = self.current_reference_image
                self.log_event(f"Associated video {selected_source} with {self.current_reference_image}")

                # Update reference tab if it exists
                if hasattr(self, 'reference_tab_controller'):
                    self.reference_tab_controller.populate_reference_tree()

            return True

        except Exception as e:
            self.log_event(f"Error associating video: {str(e)}")
            return False

    def on_mouse_down(self, event):
        """Handle mouse button press event for drawing parking spaces"""
        try:
            if hasattr(self, 'setup_tab_controller'):
                self.drawing = True
                self.start_x = event.x
                self.start_y = event.y
                self.current_rect = None

        except Exception as e:
            self.log_event(f"Error handling mouse down: {str(e)}")

    def on_mouse_move(self, event):
        """Handle mouse movement while drawing parking spaces"""
        try:
            if hasattr(self, 'setup_tab_controller') and self.drawing:
                if self.current_rect:
                    self.setup_tab_controller.setup_canvas.delete(self.current_rect)

                self.current_rect = self.setup_tab_controller.setup_canvas.create_rectangle(
                    self.start_x, self.start_y, event.x, event.y,
                    outline='green', width=2
                )

        except Exception as e:
            self.log_event(f"Error handling mouse move: {str(e)}")

    def on_mouse_up(self, event):
        """Handle mouse button release to complete parking space drawing"""
        try:
            if hasattr(self, 'setup_tab_controller') and self.drawing:
                self.drawing = False
                if self.current_rect:
                    self.setup_tab_controller.setup_canvas.delete(self.current_rect)

                # Add new parking space
                x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
                x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
                width = x2 - x1
                height = y2 - y1

                if width > 20 and height > 20:  # Minimum size check
                    self.posList.append([x1, y1, width, height])
                    self.setup_tab_controller.setup_canvas.create_rectangle(
                        x1, y1, x2, y2,
                        outline='green', width=2
                    )
                    self.log_event(f"Added parking space at ({x1}, {y1})")

        except Exception as e:
            self.log_event(f"Error handling mouse up: {str(e)}")

    def on_right_click(self, event):
        """Handle right click to delete parking spaces"""
        try:
            if hasattr(self, 'setup_tab_controller'):
                x, y = event.x, event.y
                for i, (px, py, w, h) in enumerate(self.posList):
                    if px <= x <= px + w and py <= y <= py + h:
                        self.posList.pop(i)
                        self.log_event(f"Removed parking space at ({px}, {py})")
                        self.load_reference_image()  # Refresh display
                        break

        except Exception as e:
            self.log_event(f"Error handling right click: {str(e)}")

    def load_reference_image(self, image_path=None):
        """Load the reference image for parking space setup"""
        try:
            if image_path:
                self.current_reference_image = image_path

            if not hasattr(self, 'current_reference_image') or not self.current_reference_image:
                self.log_event("No reference image selected")
                return False

            if not os.path.exists(self.current_reference_image):
                self.log_event(f"Reference image not found: {self.current_reference_image}")
                return False

            # Load the image
            img = cv2.imread(self.current_reference_image)
            if img is None:
                self.log_event("Failed to load reference image")
                return False

            # Store image dimensions
            if not hasattr(self, 'reference_dimensions'):
                self.reference_dimensions = {}
            self.reference_dimensions[self.current_reference_image] = (img.shape[1], img.shape[0])

            # Load associated parking positions
            if not hasattr(self, 'posList'):
                self.posList = []
            self.posList = load_parking_positions(self.config_dir, self.current_reference_image, self.log_event)

            # Update UI if setup tab exists
            if hasattr(self, 'setup_tab_controller'):
                # Update canvas with new image
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)

                self.setup_tab_controller.setup_canvas.config(width=img.shape[1], height=img.shape[0])
                self.setup_tab_controller.setup_canvas.create_image(0, 0, anchor="nw", image=img_tk)
                self.setup_tab_controller.setup_canvas.image = img_tk

            return True

        except Exception as e:
            self.log_event(f"Error loading reference image: {str(e)}")
            return False

    def browse_reference_image(self):
        """Browse and select a reference image file"""
        try:
            from tkinter import filedialog

            filetypes = [
                ("Image files", "*.png;*.jpg;*.jpeg"),
                ("All files", "*.*")
            ]

            filename = filedialog.askopenfilename(
                title="Select Reference Image",
                filetypes=filetypes,
                initialdir="."
            )

            if filename:
                # Update current reference image
                self.current_reference_image = filename
                self.log_event(f"Selected reference image: {filename}")

                # Load the selected image
                self.load_reference_image(filename)
                return True

            return False

        except Exception as e:
            self.log_event(f"Error browsing reference image: {str(e)}")
            return False

    def clear_log(self):
        """Clear the log display and log data"""
        try:
            if hasattr(self, 'log_tab_controller'):
                self.log_tab_controller.clear_display()
                if hasattr(self, 'log_data'):
                    self.log_data.clear()
                self.log_event("Log cleared")
                return True
            return False

        except Exception as e:
            print(f"Error clearing log: {str(e)}")
            return False

    def save_log(self):
        """Save the current log data to a file"""
        try:
            if hasattr(self, 'log_data') and self.log_data:
                filename = save_log(self.log_dir, self.log_data, self.log_event)
                if filename:
                    self.log_event(f"Log saved to {filename}")
                    return True
            else:
                self.log_event("No log data to save")
            return False

        except Exception as e:
            self.log_event(f"Error saving log: {str(e)}")
            return False

    def clear_statistics(self):
        """Clear all recorded parking statistics"""
        try:
            if hasattr(self, 'stats_tab_controller'):
                # Clear treeview
                for item in self.stats_tab_controller.stats_tree.get_children():
                    self.stats_tab_controller.stats_tree.delete(item)

                # Clear stored statistics if any
                if hasattr(self, 'stats_data'):
                    self.stats_data.clear()

                self.log_event("Statistics cleared")
                return True
            return False

        except Exception as e:
            self.log_event(f"Error clearing statistics: {str(e)}")
            return False

    def export_statistics(self):
        """Export parking statistics to a CSV file"""
        try:
            if hasattr(self, 'stats_data') and self.stats_data:
                filename = export_statistics(self.log_dir, self.stats_data, self.log_event)
                if filename:
                    self.log_event(f"Statistics exported to {filename}")
                    return True
            else:
                self.log_event("No statistics data to export")
            return False

        except Exception as e:
            self.log_event(f"Error exporting statistics: {str(e)}")
            return False

    def record_current_stats(self):
        """Record current parking statistics"""
        try:
            if not hasattr(self, 'stats_data'):
                self.stats_data = []

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stats = (
                timestamp,
                self.total_spaces,
                self.free_spaces,
                self.occupied_spaces,
                self.vehicle_counter
            )

            self.stats_data.append(stats)

            # Update stats tree if available
            if hasattr(self, 'stats_tab_controller'):
                self.stats_tab_controller.stats_tree.insert("", "end", values=stats)

            self.log_event(f"Recorded statistics at {timestamp}")
            return True

        except Exception as e:
            self.log_event(f"Error recording statistics: {str(e)}")
            return False