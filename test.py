# Enable OpenCV optimizations if available
import cv2
import pickle
import cvzone
import numpy as np
import os
import threading
import time
import torch
from torchvision.models import detection
from tkinter import Toplevel
from datetime import datetime
from tkinter import Tk, Label, Button, Frame, Canvas, Text, Scrollbar, OptionMenu, StringVar, IntVar, BooleanVar, messagebox, ttk, BOTH, \
    TOP, LEFT, RIGHT, BOTTOM, X, Y, VERTICAL, HORIZONTAL
from PIL import Image, ImageTk

try:
    cv2.setUseOptimized(True)
    print("OpenCV optimizations:", "Enabled" if cv2.useOptimized() else "Disabled")
except:
    print("OpenCV optimization settings unavailable")

# Add these near the start of your file, after imports
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # May improve performance
    torch.cuda.empty_cache()  # Clear cache at startup

class VehicleDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache first
            print(f"GPU Memory before model: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")

        try:
            # Load pre-trained model with updated API
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            self.model = detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            self.model.to(self.device)
            self.model.eval()

            # Verify model is on correct device
            print(f"Model device: {next(self.model.parameters()).device}")

            # Report memory usage after model load
            if torch.cuda.is_available():
                print(f"GPU Memory after model: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # COCO class names
        self.classes = [
            'background', 'person', 'bicycle', 'car', 'motorcycle',
            'airplane', 'bus', 'train', 'truck', 'boat'
        ]
        self.vehicle_classes = [2, 3, 5, 6, 7, 8]  # Indices of vehicle classes

        # Use smaller input size for inference (keeps aspect ratio)
        self.max_size = 480  # Lower this for more speed, raise for more accuracy

        # Warm up the model
        if torch.cuda.is_available():
            dummy_input = torch.zeros((1, 3, self.max_size, self.max_size), device=self.device)
            try:
                with torch.no_grad():
                    _ = self.model(dummy_input)
                print("Model warm-up completed")
            except Exception as e:
                print(f"Model warm-up failed: {str(e)}, continuing anyway")

    def detect_vehicles(self, frame):
        orig_h, orig_w = frame.shape[:2]

        # Resize to target size while maintaining aspect ratio
        scale = min(self.max_size / orig_h, self.max_size / orig_w)
        if scale < 1.0:
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        # Convert frame to tensor
        img = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            predictions = self.model(img)

        # Extract detections
        boxes = predictions[0]['boxes'].cpu().numpy().astype(int)
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # Filter by confidence and vehicle classes
        vehicle_detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score > self.confidence_threshold and label in self.vehicle_classes:
                x1, y1, x2, y2 = box

                # Scale back to original size if resized
                if scale < 1.0:
                    x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)

                vehicle_detections.append((x1, y1, x2, y2, score, label))

        return vehicle_detections
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
        # Add after other initializations in __init__
        self.use_ml_detection = False
        self.ml_detector = None
        self.ml_confidence = 0.6  # Default confidence threshold
        self.ml_confidence = self.DEFAULT_CONFIDENCE
        self.parking_threshold = self.DEFAULT_THRESHOLD
        self.min_contour_width = self.MIN_CONTOUR_SIZE
        self.min_contour_height = self.MIN_CONTOUR_SIZE
        self.offset = self.DEFAULT_OFFSET
        self.line_height = self.DEFAULT_LINE_HEIGHT
        self._cleanup_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.video_lock = threading.Lock()
        self.torch_gpu_available, self.cv_gpu_available = self.check_gpu_availability()
        self.diagnose_gpu()  # Add this line

        # Video reference map and dimensions - moved up before setup_ui()
        # In the __init__ method, update the video_reference_map
        self.video_reference_map = {
            "sample5.mp4": "saming1.png",
            "Video.mp4": "videoImg.png",
            "carPark.mp4": "carParkImg.png",
            "0": "webcamImg.png",  # Default for webcam
            "newVideo1.mp4": "newRefImage1.png",  # Add your new reference image
            "newVideo2.mp4": "newRefImage2.png"  # Add another reference image
        }

        # Also update reference_dimensions if you know the original dimensions
        self.reference_dimensions = {
            "carParkImg.png": (1280, 720),
            "videoImg.png": (1280, 720),
            "webcamImg.png": (640, 480),
            "newRefImage1.png": (1280, 720),  # Add dimensions for your new reference image
            "newRefImage2.png": (1920, 1080)  # Add dimensions for another reference image
        }
        self.current_reference_image = "carParkImg.png"  # Default - moved up

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
        # Original dimensions of reference images (for scaling)
        self.reference_dimensions = {
            "carParkImg.png": (1280, 720),  # Update with actual original dimensions
            "videoImg.png": (1280, 720),  # Update with actual original dimensions
            "webcamImg.png": (640, 480)  # Update with actual original dimensions
        }
        self.current_reference_image = "carParkImg.png"  # Default


    def ensure_directories_exist(self):
        """Ensure necessary directories exist"""
        for directory in [self.config_dir, self.log_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    def __del__(self):
        self.cleanup_resources()

    def update_data(self):
        with self.data_lock:
            # Update shared data here
            pass

    def process_video(self):
        with self.video_lock:
            # Process video frame here
            pass

    def cleanup_resources(self):
        with self._cleanup_lock:
            if hasattr(self, 'video_capture') and self.video_capture:
                self.video_capture.release()
            if hasattr(self, 'ml_detector') and self.ml_detector:
                del self.ml_detector
            torch.cuda.empty_cache()

    def load_parking_positions(self, reference_image=None):
        try:
            if reference_image is None:
                reference_image = self.current_reference_image
            pos_file = os.path.join(self.config_dir, f'CarParkPos_{os.path.splitext(reference_image)[0]}')

            if not os.path.exists(self.config_dir):
                raise FileNotFoundError(f"Config directory {self.config_dir} does not exist")

            if os.path.exists(pos_file):
                with open(pos_file, 'rb') as f:
                    self.posList = pickle.load(f)
            else:
                self.posList = []

        except FileNotFoundError as e:
            self.log_event(f"Position file not found: {str(e)}")
            messagebox.showwarning("Warning", "Position file not found. Starting with empty positions.")
        except PermissionError as e:
            self.log_event(f"Permission denied: {str(e)}")
            messagebox.showerror("Error", "Permission denied accessing position file")
        except Exception as e:
            self.log_event(f"Unexpected error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load parking positions: {str(e)}")
            self.total_spaces = 0
            self.free_spaces = 0
            self.occupied_spaces = 0
            self.total_spaces = len(self.posList)
            self.free_spaces = 0
            self.occupied_spaces = self.total_spaces

    def show_progress(self, message):
        self.status_label.config(text=message)
        self.master.update_idletasks()

    def long_running_operation(self):
        self.show_progress("Operation in progress...")
        try:
            # Perform operation
            pass
        finally:
            self.show_progress("Ready")

    def validate_config(self):
        if not 0 <= self.ml_confidence <= 1:
            raise ValueError("ML confidence must be between 0 and 1")
        if self.parking_threshold <= 0:
            raise ValueError("Parking threshold must be positive")
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def scale_positions_to_current_dimensions(self):
        """Scale parking positions based on current video dimensions"""
        if not hasattr(self, 'image_width') or not hasattr(self, 'image_height'):
            return

        if self.current_reference_image in self.reference_dimensions:
            ref_width, ref_height = self.reference_dimensions[self.current_reference_image]

            # Calculate scale factors
            width_scale = self.image_width / ref_width
            height_scale = self.image_height / ref_height

            # Scale all positions
            scaled_positions = []
            for x, y, w, h in self.posList:
                new_x = int(x * width_scale)
                new_y = int(y * height_scale)
                new_w = int(w * width_scale)
                new_h = int(h * height_scale)
                scaled_positions.append((new_x, new_y, new_w, new_h))

            self.posList = scaled_positions

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
        self.setup_detection_tab()
        self.setup_setup_tab()
        self.setup_log_tab()
        self.setup_stats_tab()
        self.setup_reference_tab()

    def setup_detection_tab(self):
        """Set up the detection tab UI with scrollable right panel"""
        # Main frame for detection
        self.detection_frame = Frame(self.detection_tab)
        self.detection_frame.pack(fill=BOTH, expand=True)

        # Left side - Video feed
        self.video_frame = Frame(self.detection_frame)
        self.video_frame.pack(side=LEFT, fill=BOTH, expand=True)

        self.video_canvas = Canvas(self.video_frame, bg='black')
        self.video_canvas.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # Right side - Controls and info with scrolling
        right_frame = Frame(self.detection_frame, width=300)
        right_frame.pack(side=RIGHT, fill=Y, padx=10, pady=10)
        right_frame.pack_propagate(False)

        # Add a canvas and scrollbar for scrolling
        control_canvas = Canvas(right_frame, width=280)
        control_canvas.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = Scrollbar(right_frame, orient=VERTICAL, command=control_canvas.yview)
        scrollbar.pack(side=RIGHT, fill=Y)

        control_canvas.configure(yscrollcommand=scrollbar.set)
        control_canvas.bind('<Configure>', lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))

        # Create the inner frame for controls
        self.control_frame = Frame(control_canvas)
        control_canvas_window = control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw", width=280)

        # Mode selection
        Label(self.control_frame, text="Detection Mode:", font=("Arial", 12)).pack(pady=(10, 5))
        self.mode_var = StringVar(value="parking")
        self.mode_menu = OptionMenu(self.control_frame, self.mode_var,
                                    "parking", "vehicle",
                                    command=self.switch_detection_mode)
        self.mode_menu.pack(fill=X, pady=5)

        # ML detection toggle
        self.ml_frame = Frame(self.control_frame)
        self.ml_frame.pack(fill=X, pady=5)
        self.ml_var = StringVar(value="Off")
        Label(self.ml_frame, text="ML Detection:").pack(side=LEFT)
        self.ml_toggle = OptionMenu(self.ml_frame, self.ml_var, "Off", "On", command=self.toggle_ml_detection)
        self.ml_toggle.pack(side=LEFT, padx=5)

        # ML confidence slider
        Label(self.control_frame, text="ML Confidence:").pack(anchor="w")
        self.ml_confidence_scale = ttk.Scale(self.control_frame, from_=0.1, to=0.9,
                                             orient=HORIZONTAL, value=0.6,
                                             command=self.update_ml_confidence)
        self.ml_confidence_scale.pack(fill=X, pady=5)
        self.ml_confidence_label = Label(self.control_frame, text=f"Value: 0.6")
        self.ml_confidence_label.pack(anchor="w")

        # Video source selection
        Label(self.control_frame, text="Video Source:", font=("Arial", 12)).pack(pady=(10, 5))
        self.video_source_var = StringVar(value="sample5.mp4")
        self.video_sources = ["sample5.mp4", "Video.mp4", "0", "carPark.mp4", "newVideo1.mp4", "newVideo2.mp4"]
        self.video_menu = OptionMenu(self.control_frame, self.video_source_var,
                                     *self.video_sources,
                                     command=self.switch_video_source)
        self.video_menu.pack(fill=X, pady=5)

        # Status information
        Label(self.control_frame, text="Status Information", font=("Arial", 14, "bold")).pack(pady=10)
        self.status_info = Label(self.control_frame,
                                 text="Total Spaces: 0\nFree Spaces: 0\nOccupied: 0\nVehicles Counted: 0",
                                 font=("Arial", 12), justify=LEFT)
        self.status_info.pack(pady=5, fill=X)

        # Status indicator
        self.status_label = Label(self.control_frame, text="Status: Stopped", fg="red", font=("Arial", 12))
        self.status_label.pack(pady=5, fill=X)

        # Buttons
        self.button_frame = Frame(self.control_frame)
        self.button_frame.pack(fill=X, pady=10)

        self.start_button = Button(self.button_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(fill=X, pady=5)

        self.stop_button = Button(self.button_frame, text="Stop Detection", command=self.stop_detection,
                                  state="disabled")
        self.stop_button.pack(fill=X, pady=5)

        self.reset_button = Button(self.button_frame, text="Reset Counters", command=self.reset_counters)
        self.reset_button.pack(fill=X, pady=5)

        # Advanced settings
        Label(self.control_frame, text="Settings", font=("Arial", 12, "bold")).pack(pady=(15, 5))

        # Threshold slider
        Label(self.control_frame, text="Detection Threshold:").pack(anchor="w")
        self.threshold_scale = ttk.Scale(self.control_frame, from_=100, to=1000,
                                         orient=HORIZONTAL, value=self.parking_threshold,
                                         command=self.update_threshold)
        self.threshold_scale.pack(fill=X, pady=5)
        self.threshold_label = Label(self.control_frame, text=f"Value: {self.parking_threshold}")
        self.threshold_label.pack(anchor="w")

        # Performance section
        Label(self.control_frame, text="Performance Settings", font=("Arial", 12, "bold")).pack(pady=(15, 5))

        # Frame skip control
        skip_frame = Frame(self.control_frame)
        skip_frame.pack(fill=X, pady=5)

        Label(skip_frame, text="ML Frame Skip:").pack(side=LEFT)
        self.frame_skip_var = IntVar(value=3)

        skip_options = [(1, "Every frame"), (2, "Every 2nd"), (3, "Every 3rd"), (5, "Every 5th"), (8, "Every 8th")]
        for val, text in skip_options:
            ttk.Radiobutton(skip_frame, text=text, variable=self.frame_skip_var, value=val,
                            command=lambda v=val: self.set_frame_skip_rate(v)).pack(anchor="w")

        # GPU status
        gpu_status = "Available" if torch.cuda.is_available() else "Not Available"
        gpu_color = "green" if torch.cuda.is_available() else "red"
        Label(self.control_frame, text=f"GPU: {gpu_status}", fg=gpu_color, font=("Arial", 10, "bold")).pack(pady=5)
        # Add this to your setup_detection_tab method after the GPU status label
        Button(self.control_frame, text="Test GPU", command=self.test_gpu).pack(pady=5)


        # Make sure the canvas can scroll to all controls
        self.control_frame.update_idletasks()  # Update geometry
        control_canvas.config(scrollregion=control_canvas.bbox("all"))

        # Define mousewheel scrolling function
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Add mousewheel scrolling
        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Store control_canvas reference for later use
        self.control_canvas = control_canvas

    def test_gpu(self):

        message = "GPU Status:\n"
        message += f"PyTorch GPU: {'Available' if torch.cuda.is_available() else 'Not Available'}\n"
        message += f"OpenCV GPU: {'Available' if self.cv_gpu_available else 'Not Available'}"

        messagebox.showinfo("GPU Test Results", message)

        # Create a dialog to show results
        dialog = Toplevel(self.master)
        dialog.title("GPU Test Results")
        dialog.geometry("500x400")
        dialog.resizable(True, True)

        # Create scrollable text area for results
        result_text = Text(dialog, wrap="word", height=20, width=60)
        result_text.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Add scrollbar
        scrollbar = Scrollbar(result_text)
        scrollbar.pack(side=RIGHT, fill=Y)
        result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=result_text.yview)

        # Add basic GPU info
        result_text.insert("end", f"PyTorch Version: {torch.__version__}\n")
        result_text.insert("end", f"CUDA Available: {torch.cuda.is_available()}\n")

        if torch.cuda.is_available():
            result_text.insert("end", f"GPU Device: {torch.cuda.get_device_name(0)}\n")
            result_text.insert("end", f"CUDA Version: {torch.version.cuda}\n")

            # Simple performance test with proper error handling
            result_text.insert("end", "\nRunning GPU speed test (smaller tensor size for safety)...\n")

            try:
                # Use smaller tensor size to avoid OOM errors
                tensor_size = 2000  # Reduced from 5000

                # Create test tensor on CPU
                start_time = time.time()
                cpu_tensor = torch.randn(tensor_size, tensor_size)
                cpu_time = time.time() - start_time
                result_text.insert("end", f"CPU tensor creation: {cpu_time:.4f} seconds\n")

                # Create test tensor on GPU with error handling
                try:
                    torch.cuda.empty_cache()  # Clear GPU memory first
                    start_time = time.time()
                    gpu_tensor = torch.randn(tensor_size, tensor_size, device='cuda')
                    torch.cuda.synchronize()  # Wait for GPU operations to complete
                    gpu_time = time.time() - start_time
                    result_text.insert("end", f"GPU tensor creation: {gpu_time:.4f} seconds\n")

                    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                    result_text.insert("end", f"GPU speedup: {speedup:.2f}x\n")

                    # Clean up to avoid memory issues
                    del gpu_tensor
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        result_text.insert("end", "GPU test failed: CUDA out of memory. Try reducing the workload.\n")
                    else:
                        result_text.insert("end", f"GPU test failed: {str(e)}\n")
                except Exception as e:
                    result_text.insert("end", f"GPU test failed: {str(e)}\n")
            except Exception as e:
                result_text.insert("end", f"Performance test failed: {str(e)}\n")

        # Disable editing
        result_text.config(state="disabled")

        # Close button
        Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

    def detect_vehicles_ml(self, frame):
        """Process frame with ML detection with better GPU error handling"""
        try:
            orig_h, orig_w = frame.shape[:2]

            # Resize to target size while maintaining aspect ratio
            scale = min(self.max_size / orig_h, self.max_size / orig_w)
            if scale < 1.0:
                new_h, new_w = int(orig_h * scale), int(orig_w * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            # Convert frame to tensor and move to appropriate device
            img = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

            try:
                img = img.to(self.device)
            except RuntimeError as e:
                # Handle out of memory or other device errors
                self.log_event(f"GPU error: {e}. Falling back to CPU")
                self.device = torch.device('cpu')
                self.model.to(self.device)
                img = img.to(self.device)

            with torch.no_grad():
                predictions = self.model(img)

            # Extract detections
            boxes = predictions[0]['boxes'].cpu().numpy().astype(int)
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

            # Filter by confidence and vehicle classes
            vehicle_detections = []
            for box, score, label in zip(boxes, scores, labels):
                if score > self.confidence_threshold and label in self.vehicle_classes:
                    x1, y1, x2, y2 = box

                    # Scale back to original size if resized
                    if scale < 1.0:
                        x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)

                    # Ensure coordinates are within frame bounds
                    x1 = max(0, min(x1, orig_w - 1))
                    y1 = max(0, min(y1, orig_h - 1))
                    x2 = max(0, min(x2, orig_w - 1))
                    y2 = max(0, min(y2, orig_h - 1))

                    vehicle_detections.append((x1, y1, x2, y2, score, label))

            return vehicle_detections

        except Exception as e:
            self.log_event(f"ML detection error: {str(e)}")
            return []  # Return empty list on error

    def add_performance_controls(self):
        """Add performance controls to the UI"""
        # Add to your setup_detection_tab method after other controls
        Label(self.control_frame, text="Performance Settings", font=("Arial", 12, "bold")).pack(pady=(15, 5))

        # Frame for performance controls
        perf_frame = Frame(self.control_frame)
        perf_frame.pack(fill=X, pady=5)

        # Frame skip rate for ML detection
        Label(perf_frame, text="ML Frame Skip:").pack(anchor="w")
        self.frame_skip_scale = ttk.Scale(perf_frame, from_=1, to=10,
                                          orient=HORIZONTAL, value=3,
                                          command=self.update_frame_skip)
        self.frame_skip_scale.pack(fill=X, pady=2)
        self.frame_skip_label = Label(perf_frame, text="3")
        self.frame_skip_label.pack(anchor="e")

    def update_frame_skip(self, value):
        """Update the frame skip rate for ML detection"""
        skip = int(float(value))
        self.frame_skip = skip
        self.frame_skip_label.config(text=str(skip))

    def set_frame_skip_rate(self, rate):
        """
        Set the frame skip rate for ML detection.
        A higher number means fewer frames are processed by ML, improving speed but reducing accuracy.

        Args:
            rate: Number of frames to skip (2 = process every other frame, 3 = process every third frame)
        """
        if not hasattr(self, 'frame_skip'):
            self.frame_skip = 3  # Default

        self.frame_skip = max(1, int(rate))
        self.log_event(f"Set ML frame skip rate to {self.frame_skip}")

    def setup_reference_tab(self):
        """Set up a new tab for reference image management"""
        # Reference tab frame
        self.reference_frame = Frame(self.reference_tab, padx=10, pady=10)
        self.reference_frame.pack(fill=BOTH, expand=True)

        # Header frame
        header_frame = Frame(self.reference_frame)
        header_frame.pack(fill=X, pady=5)

        Label(header_frame, text="Reference Images", font=("Arial", 14, "bold")).pack(side=LEFT)

        # Add buttons
        Button(header_frame, text="Add Reference", command=self.browse_reference_image).pack(side=RIGHT, padx=5)
        Button(header_frame, text="Associate Video", command=self.associate_video_with_reference).pack(side=RIGHT,
                                                                                                       padx=5)

        # Create Treeview for references
        ref_tree_frame = Frame(self.reference_frame)
        ref_tree_frame.pack(fill=BOTH, expand=True, pady=10)

        self.ref_tree = ttk.Treeview(ref_tree_frame, columns=("image", "dimensions", "associated_videos"))

        # Define column headings
        self.ref_tree.heading("#0", text="")
        self.ref_tree.heading("image", text="Reference Image")
        self.ref_tree.heading("dimensions", text="Dimensions")
        self.ref_tree.heading("associated_videos", text="Associated Videos")

        # Define column widths
        self.ref_tree.column("#0", width=0, stretch=False)
        self.ref_tree.column("image", width=200)
        self.ref_tree.column("dimensions", width=150)
        self.ref_tree.column("associated_videos", width=300)

        # Add scrollbar
        ref_vsb = ttk.Scrollbar(ref_tree_frame, orient=VERTICAL, command=self.ref_tree.yview)
        self.ref_tree.configure(yscrollcommand=ref_vsb.set)
        ref_vsb.pack(side=RIGHT, fill=Y)
        self.ref_tree.pack(side=LEFT, fill=BOTH, expand=True)

        # Preview frame
        preview_frame = Frame(self.reference_frame)
        preview_frame.pack(fill=BOTH, expand=True, pady=10)

        Label(preview_frame, text="Image Preview", font=("Arial", 12, "bold")).pack(pady=5)

        self.preview_canvas = Canvas(preview_frame, bg="black", height=300)
        self.preview_canvas.pack(fill=BOTH, expand=True)

        # Populate the reference tree
        self.populate_reference_tree()

        # Bind selection event
        self.ref_tree.bind("<<TreeviewSelect>>", self.on_reference_select)

    def populate_reference_tree(self):
        """Populate the reference image tree with data"""
        # Clear existing items
        for item in self.ref_tree.get_children():
            self.ref_tree.delete(item)

        # Add each reference image
        for ref_img in set(self.video_reference_map.values()):
            # Find associated videos
            associated = [vid for vid, img in self.video_reference_map.items() if img == ref_img]
            associated_str = ", ".join(associated)

            # Get dimensions
            dimensions = self.reference_dimensions.get(ref_img, "Unknown")
            if dimensions != "Unknown":
                dimensions_str = f"{dimensions[0]}x{dimensions[1]}"
            else:
                dimensions_str = "Unknown"

            # Insert into tree
            self.ref_tree.insert("", "end", values=(ref_img, dimensions_str, associated_str))

    def on_reference_select(self, event):
        """Handle reference image selection"""
        selection = self.ref_tree.selection()
        if selection:
            item = selection[0]
            ref_img = self.ref_tree.item(item, "values")[0]

            # Display the image in the preview canvas
            try:
                if os.path.exists(ref_img):
                    img = cv2.imread(ref_img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Resize for preview
                    preview_height = 300
                    ratio = preview_height / img.shape[0]
                    preview_width = int(img.shape[1] * ratio)

                    img = cv2.resize(img, (preview_width, preview_height))

                    # Convert to PhotoImage
                    img_pil = Image.fromarray(img)
                    img_tk = ImageTk.PhotoImage(image=img_pil)

                    # Update canvas
                    self.preview_canvas.config(width=preview_width, height=preview_height)
                    self.preview_canvas.create_image(0, 0, anchor="nw", image=img_tk)
                    self.preview_canvas.image = img_tk
            except Exception as e:
                self.log_event(f"Error previewing reference image: {str(e)}")

    def setup_setup_tab(self):
        """Set up the setup tab UI for defining parking spaces"""
        # Frame for setup controls
        self.setup_control_frame = Frame(self.setup_tab, padx=10, pady=10)
        self.setup_control_frame.pack(side=TOP, fill=X)

        Label(self.setup_control_frame, text="Parking Space Setup", font=("Arial", 14, "bold")).pack(side=LEFT, padx=10)

        self.setup_instructions = Label(self.setup_control_frame,
                                        text="Left-click and drag to draw spaces. Right-click to delete spaces.",
                                        font=("Arial", 10))
        self.setup_instructions.pack(side=LEFT, padx=10)

        calibration_frame = Frame(self.setup_control_frame)
        calibration_frame.pack(side=LEFT, padx=20)

        Label(calibration_frame, text="Calibration:").pack(side=LEFT)

        Button(calibration_frame, text="↑", command=lambda: self.shift_all_spaces(0, -5)).pack(side=LEFT)
        Button(calibration_frame, text="↓", command=lambda: self.shift_all_spaces(0, 5)).pack(side=LEFT)
        Button(calibration_frame, text="←", command=lambda: self.shift_all_spaces(-5, 0)).pack(side=LEFT)
        Button(calibration_frame, text="→", command=lambda: self.shift_all_spaces(5, 0)).pack(side=LEFT)

        self.save_spaces_button = Button(self.setup_control_frame, text="Save Spaces", command=self.save_parking_spaces)
        self.save_spaces_button.pack(side=RIGHT, padx=10)

        self.clear_spaces_button = Button(self.setup_control_frame, text="Clear All", command=self.clear_all_spaces)
        self.clear_spaces_button.pack(side=RIGHT, padx=10)

        # Add this to your setup_setup_tab method, near the save and clear buttons
        self.ref_image_var = StringVar(value=self.current_reference_image)
        Label(self.setup_control_frame, text="Reference Image:").pack(side=LEFT, padx=10)
        self.ref_image_menu = OptionMenu(self.setup_control_frame, self.ref_image_var,
                                         *list(self.video_reference_map.values()),
                                         command=self.load_reference_image)
        self.ref_image_menu.pack(side=LEFT, padx=10)

        # Add this at the end of the setup_setup_tab method
        self.associate_button = Button(self.setup_control_frame, text="Associate Video",
                                       command=self.associate_video_with_reference)
        self.associate_button.pack(side=RIGHT, padx=10)

        # Frame for the setup canvas
        self.setup_canvas_frame = Frame(self.setup_tab)
        self.setup_canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.setup_canvas = Canvas(self.setup_canvas_frame, bg='black')
        self.setup_canvas.pack(fill=BOTH, expand=True)

        # Setup mouse events
        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.current_rect = None

        self.setup_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.setup_canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.setup_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.setup_canvas.bind("<ButtonPress-3>", self.on_right_click)

        # Load reference image
        self.load_reference_image()

    def add_reference_image_button(self):
        """Add a button to the setup tab for adding new reference images"""
        self.add_ref_button = Button(self.setup_control_frame, text="Add Reference Image",
                                     command=self.browse_reference_image)
        self.add_ref_button.pack(side=RIGHT, padx=10)

    def browse_reference_image(self):
        """Browse for a new reference image and add it to the system"""
        from tkinter import filedialog

        # Open file dialog to select image
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )

        if file_path:
            # Get just the filename
            file_name = os.path.basename(file_path)

            # Check if the file is already in the working directory
            if not os.path.exists(file_name):
                # Copy the file to the working directory
                import shutil
                try:
                    shutil.copy(file_path, file_name)
                    self.log_event(f"Copied reference image {file_name} to working directory")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to copy reference image: {str(e)}")
                    return

            # Get image dimensions
            try:
                img = cv2.imread(file_name)
                height, width = img.shape[:2]

                # Add to reference dimensions
                self.reference_dimensions[file_name] = (width, height)

                # Update dropdown menu
                menu = self.ref_image_menu["menu"]
                menu.delete(0, "end")
                for ref_img in list(self.video_reference_map.values()) + [file_name]:
                    menu.add_command(label=ref_img,
                                     command=lambda value=ref_img: self.ref_image_var.set(
                                         value) or self.load_reference_image(value))

                # Select the new image
                self.ref_image_var.set(file_name)
                self.load_reference_image(file_name)

                self.log_event(f"Added reference image {file_name} ({width}x{height})")
                messagebox.showinfo("Success", f"Added reference image: {file_name}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to process reference image: {str(e)}")

    def associate_video_with_reference(self):
        """Associate a video source with a reference image"""
        # Create a simple dialog
        dialog = Toplevel(self.master)
        dialog.title("Associate Video with Reference Image")
        dialog.geometry("400x200")
        dialog.resizable(False, False)

        # Video source selection
        Label(dialog, text="Video Source:").pack(pady=(10, 5))
        video_var = StringVar(value=self.video_sources[0])
        video_dropdown = OptionMenu(dialog, video_var, *self.video_sources)
        video_dropdown.pack(fill=X, padx=20, pady=5)

        # Reference image selection
        Label(dialog, text="Reference Image:").pack(pady=(10, 5))
        ref_var = StringVar(value=list(self.video_reference_map.values())[0])
        ref_dropdown = OptionMenu(dialog, ref_var, *list(self.video_reference_map.values()) +
                                                    [img for img in os.listdir() if
                                                     img.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        ref_dropdown.pack(fill=X, padx=20, pady=5)

        # Button frame
        btn_frame = Frame(dialog)
        btn_frame.pack(fill=X, pady=20)

        # Cancel button
        Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=RIGHT, padx=20)

        # Associate button
        def do_associate():
            video = video_var.get()
            ref_img = ref_var.get()
            self.video_reference_map[video] = ref_img
            self.log_event(f"Associated video {video} with reference image {ref_img}")
            messagebox.showinfo("Success", f"Associated {video} with {ref_img}")
            dialog.destroy()

        Button(btn_frame, text="Associate", command=do_associate).pack(side=RIGHT, padx=20)

    def setup_log_tab(self):
        """Set up the log tab UI"""
        # Log tab frame
        self.log_frame = Frame(self.log_tab, padx=10, pady=10)
        self.log_frame.pack(fill=BOTH, expand=True)

        # Title and controls
        self.log_header = Frame(self.log_frame)
        self.log_header.pack(fill=X, pady=5)

        Label(self.log_header, text="System Logs", font=("Arial", 14, "bold")).pack(side=LEFT)

        self.clear_log_button = Button(self.log_header, text="Clear Log", command=self.clear_log)
        self.clear_log_button.pack(side=RIGHT, padx=5)

        self.save_log_button = Button(self.log_header, text="Save Log", command=self.save_log)
        self.save_log_button.pack(side=RIGHT, padx=5)

        # Log text area with scrollbar
        self.log_text_frame = Frame(self.log_frame)
        self.log_text_frame.pack(fill=BOTH, expand=True, pady=10)

        self.log_text = Text(self.log_text_frame, wrap="word", height=20)
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)

        self.log_scrollbar = Scrollbar(self.log_text_frame, command=self.log_text.yview)
        self.log_scrollbar.pack(side=RIGHT, fill=Y)

        self.log_text.config(yscrollcommand=self.log_scrollbar.set)
        self.log_text.config(state="disabled")

    def shift_all_spaces(self, dx, dy):
        """Shift all parking spaces by dx, dy"""
        for i in range(len(self.posList)):
            x, y, w, h = self.posList[i]
            self.posList[i] = (x + dx, y + dy, w, h)

        # Redraw spaces
        self.draw_parking_spaces()
        self.log_event(f"Shifted all spaces by ({dx}, {dy})")

    def setup_stats_tab(self):
        """Set up the statistics tab UI"""
        # Stats tab frame
        self.stats_frame = Frame(self.stats_tab, padx=10, pady=10)
        self.stats_frame.pack(fill=BOTH, expand=True)

        # Title
        Label(self.stats_frame, text="Parking Statistics", font=("Arial", 16, "bold")).pack(pady=10)

        # Statistics data
        self.stats_data_frame = Frame(self.stats_frame)
        self.stats_data_frame.pack(fill=BOTH, expand=True, pady=10)

        # Create Treeview for statistics
        self.stats_tree = ttk.Treeview(self.stats_data_frame,
                                       columns=("timestamp", "total", "free", "occupied", "vehicles"))

        # Define column headings
        self.stats_tree.heading("#0", text="")
        self.stats_tree.heading("timestamp", text="Timestamp")
        self.stats_tree.heading("total", text="Total Spaces")
        self.stats_tree.heading("free", text="Free Spaces")
        self.stats_tree.heading("occupied", text="Occupied Spaces")
        self.stats_tree.heading("vehicles", text="Vehicles Counted")

        # Define column widths
        self.stats_tree.column("#0", width=0, stretch=False)
        self.stats_tree.column("timestamp", width=200)
        self.stats_tree.column("total", width=100)
        self.stats_tree.column("free", width=100)
        self.stats_tree.column("occupied", width=100)
        self.stats_tree.column("vehicles", width=120)

        # Add scrollbar to treeview
        self.stats_vsb = ttk.Scrollbar(self.stats_data_frame, orient=VERTICAL, command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=self.stats_vsb.set)
        self.stats_vsb.pack(side=RIGHT, fill=Y)
        self.stats_tree.pack(side=LEFT, fill=BOTH, expand=True)

        # Statistics controls
        self.stats_control_frame = Frame(self.stats_frame)
        self.stats_control_frame.pack(fill=X, pady=10)

        Button(self.stats_control_frame, text="Clear Statistics", command=self.clear_statistics).pack(side=RIGHT,
                                                                                                      padx=5)
        Button(self.stats_control_frame, text="Export Statistics", command=self.export_statistics).pack(side=RIGHT,
                                                                                                        padx=5)
        Button(self.stats_control_frame, text="Record Current Stats", command=self.record_current_stats).pack(
            side=RIGHT, padx=5)

    # Event handlers and methods for setup tab
    def load_reference_image(self, image_name=None):
        """Load and display the reference image for parking space setup"""
        try:
            if image_name is None:
                image_name = self.current_reference_image
            else:
                self.current_reference_image = image_name

            self.ref_image_path = image_name
            if os.path.exists(self.ref_image_path):
                self.ref_img = cv2.imread(self.ref_image_path)
                if self.ref_img is None:
                    raise Exception("Could not load image file.")

                # Get original dimensions
                orig_height, orig_width = self.ref_img.shape[:2]

                # Store original dimensions if not already defined
                if image_name not in self.reference_dimensions:
                    self.reference_dimensions[image_name] = (orig_width, orig_height)

                # Resize to match the video dimensions if you know them
                if hasattr(self, 'image_width') and hasattr(self, 'image_height'):
                    self.ref_img = cv2.resize(self.ref_img, (self.image_width, self.image_height))

                self.ref_img = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2RGB)
                self.ref_img_pil = Image.fromarray(self.ref_img)
                self.ref_img_tk = ImageTk.PhotoImage(image=self.ref_img_pil)

                self.setup_canvas.config(width=self.image_width, height=self.image_height)
                self.image_id = self.setup_canvas.create_image(0, 0, anchor="nw", image=self.ref_img_tk)

                # Draw any existing parking spaces
                self.draw_parking_spaces()
            else:
                messagebox.showwarning("Warning", f"Reference image '{self.ref_image_path}' not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load reference image: {str(e)}")

    def draw_parking_spaces(self):
        """Draw the defined parking spaces on the setup canvas"""
        # First clear any existing rectangles
        self.setup_canvas.delete("parking_space")

        # Draw each parking space
        for i, (x, y, w, h) in enumerate(self.posList):
            rect_id = self.setup_canvas.create_rectangle(
                x, y, x + w, y + h,
                outline="magenta", width=2,
                tags=("parking_space", f"space_{i}")
            )

    def on_mouse_down(self, event):
        """Handle mouse down event for drawing parking spaces"""
        self.drawing = True
        self.start_x, self.start_y = event.x, event.y

        # Create a new rectangle
        self.current_rect = self.setup_canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="green", width=2, tags="current_rect"
        )

    def on_mouse_move(self, event):
        """Handle mouse move event while drawing parking spaces"""
        if self.drawing:
            # Update rectangle size
            self.setup_canvas.coords(self.current_rect,
                                     self.start_x, self.start_y, event.x, event.y)

    # Find the existing on_mouse_up method (around line 396)
    # Replace it with this improved version:
    def on_mouse_up(self, event):
        if self.drawing:
            self.drawing = False
            end_x, end_y = event.x, event.y

            # Calculate width and height
            width = abs(end_x - self.start_x)
            height = abs(end_y - self.start_y)

            # Ensure we have the top-left coordinates for storage
            x_pos = min(self.start_x, end_x)
            y_pos = min(self.start_y, end_y)

            # Only add if rectangle has meaningful size
            if width > 5 and height > 5:
                # Add to the list with the correct coordinates
                self.posList.append((x_pos, y_pos, width, height))

                # Update total spaces
                self.total_spaces = len(self.posList)
                self.occupied_spaces = self.total_spaces
                self.update_status_info()

                # Redraw all spaces
                self.draw_parking_spaces()

            # Remove the temporary rectangle
            self.setup_canvas.delete("current_rect")

    def on_right_click(self, event):
        """Handle right-click to delete a parking space"""
        x, y = event.x, event.y

        # Check if click is inside any parking space
        for i, (x1, y1, w, h) in enumerate(self.posList):
            if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                # Remove from the list
                self.posList.pop(i)

                # Update total spaces
                self.total_spaces = len(self.posList)
                self.occupied_spaces = self.total_spaces
                self.update_status_info()

                # Redraw all spaces
                self.draw_parking_spaces()
                break

    def save_parking_spaces(self):
        """Save the defined parking spaces to a file"""
        try:
            # Scale back to reference dimensions before saving
            if self.current_reference_image in self.reference_dimensions:
                ref_width, ref_height = self.reference_dimensions[self.current_reference_image]

                # Calculate scale factors (inverse of what we use for display)
                width_scale = ref_width / self.image_width
                height_scale = ref_height / self.image_height

                # Scale all positions back to reference dimensions
                reference_positions = []
                for x, y, w, h in self.posList:
                    ref_x = int(x * width_scale)
                    ref_y = int(y * height_scale)
                    ref_w = int(w * width_scale)
                    ref_h = int(h * height_scale)
                    reference_positions.append((ref_x, ref_y, ref_w, ref_h))

                save_positions = reference_positions
            else:
                save_positions = self.posList

            pos_file = os.path.join(self.config_dir, f'CarParkPos_{os.path.splitext(self.current_reference_image)[0]}')
            with open(pos_file, 'wb') as f:
                pickle.dump(save_positions, f)
            self.log_event(f"Saved {len(self.posList)} parking spaces for {self.current_reference_image}")
            messagebox.showinfo("Success",
                                f"Saved {len(self.posList)} parking spaces for {self.current_reference_image}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save parking spaces: {str(e)}")

    def clear_all_spaces(self):
        """Clear all defined parking spaces"""
        if messagebox.askyesno("Confirm", "Are you sure you want to delete all parking spaces?"):
            self.posList = []
            self.draw_parking_spaces()
            self.total_spaces = 0
            self.free_spaces = 0
            self.occupied_spaces = 0
            self.update_status_info()
            self.log_event("Cleared all parking spaces")

    # Detection methods
    def start_detection(self):
        """Start the detection process"""
        if not self.running:
            self.switch_video_source(self.video_source_var.get())

            if self.video_capture and self.video_capture.isOpened():
                self.running = True
                self.status_label.config(text="Status: Running", fg="green")
                self.start_button.config(state="disabled")
                self.stop_button.config(state="normal")
                self.log_event(f"Started {self.detection_mode} detection with source {self.current_video}")
                self.process_frame()
            else:
                messagebox.showerror("Error", "Could not open video source.")

    def stop_detection(self):
        """Stop the detection process"""
        self.running = False
        self.status_label.config(text="Status: Stopped", fg="red")
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.log_event(f"Stopped {self.detection_mode} detection")

    def reset_counters(self):
        """Reset the detection counters"""
        self.vehicle_counter = 0
        self.matches = []
        self.update_status_info()
        self.log_event("Reset counters")

    def switch_detection_mode(self, mode):
        """Switch between parking space detection and vehicle counting modes"""
        previous_mode = self.detection_mode
        self.detection_mode = mode
        self.log_event(f"Switched to {mode} detection mode")

        # Store current reference image before switching
        current_ref = self.current_reference_image

        # Update UI elements based on mode
        if mode == "parking":
            self.threshold_scale.config(from_=100, to=1000, value=self.parking_threshold)
            self.threshold_label.config(text=f"Value: {self.parking_threshold}")

            # Make sure we're using the proper reference image for the current video
            if self.current_video in self.video_reference_map:
                self.current_reference_image = self.video_reference_map[self.current_video]
                self.load_parking_positions(self.current_reference_image)

        else:  # vehicle mode
            self.threshold_scale.config(from_=10, to=100, value=20)  # Default for vehicle detection
            self.threshold_label.config(text=f"Value: 20")

            # Keep the current reference image, don't change it
            if previous_mode != mode and current_ref:
                self.current_reference_image = current_ref

    def reset_detection_parameters(self):
        """Reset detection parameters when switching sources"""
        if self.detection_mode == "vehicle":
            # Reset vehicle detection parameters
            self.matches = []
            if hasattr(self, 'prev_frame'):
                del self.prev_frame
        else:
            # Make sure we have the right parking positions
            self.load_parking_positions()

        # Update UI
        self.update_status_info()

    def toggle_ml_detection(self, value):
        """Toggle ML detection on/off"""
        if value == "On":
            if self.ml_detector is None:
                success = self.initialize_ml_detector()
                if not success:
                    self.ml_var.set("Off")
                    return
            self.use_ml_detection = True
            self.log_event("ML detection enabled")
        else:
            self.use_ml_detection = False
            self.log_event("ML detection disabled")

    def switch_video_source(self, source):
        """Switch the video source"""
        # Stop current detection if running
        was_running = self.running
        if was_running:
            self.stop_detection()

        # Close existing capture if any
        if self.video_capture is not None:
            self.video_capture.release()

        try:
            # Handle webcam (integer) or video file
            self.current_video = source
            if source == "0":
                self.video_capture = cv2.VideoCapture(0)  # Webcam
            else:
                self.video_capture = cv2.VideoCapture(source)  # Video file

            if not self.video_capture.isOpened():
                raise Exception(f"Could not open video source: {source}")

            # Get frame dimensions
            _, first_frame = self.video_capture.read()
            if first_frame is not None:
                self.image_height, self.image_width = first_frame.shape[:2]
                # Reset video to beginning
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.video_canvas.config(width=self.image_width, height=self.image_height)
            self.log_event(f"Switched to video source: {source}")

            # Update reference image based on video source
            if source in self.video_reference_map:
                new_ref_image = self.video_reference_map[source]
                self.current_reference_image = new_ref_image

                # Load the corresponding parking positions
                self.load_parking_positions(new_ref_image)

                # Update the setup tab with the new reference image
                self.load_reference_image(new_ref_image)

            # Restart if was running
            if was_running:
                self.start_detection()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to switch video source: {str(e)}")

        # At the end of the switch_video_source method, add:
        self.reset_detection_parameters()

    def initialize_ml_detector(self):
        try:
            # Show loading message
            loading_window = Toplevel(self.master)
            loading_window.title("Loading Model")
            loading_window.geometry("300x150")
            loading_window.resizable(False, False)
            loading_window.transient(self.master)
            loading_window.grab_set()

            # Add information
            gpu_info_label = Label(loading_window, text="")
            gpu_info_label.pack(pady=5)

            if torch.cuda.is_available():
                # Clear cache before loading model
                torch.cuda.empty_cache()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_info_label.config(text=f"Using GPU: {gpu_name}")
            else:
                gpu_info_label.config(text="Using CPU (GPU not available)")

            progress_label = Label(loading_window, text="Loading ML model...", font=("Arial", 12))
            progress_label.pack(pady=10)
            loading_window.update()

            self.log_event("Initializing ML detector...")

            # Initialize model with proper weights parameter
            try:
                from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
                self.ml_detector = VehicleDetector(confidence_threshold=self.ml_confidence)
                self.ml_detector.model = detection.fasterrcnn_resnet50_fpn(
                    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
                self.ml_detector.model.to(self.ml_detector.device)
                self.ml_detector.model.eval()
            except AttributeError:
                # Fall back to older API if using older PyTorch
                self.ml_detector = VehicleDetector(confidence_threshold=self.ml_confidence)

            # Update message
            progress_label.config(text="ML model loaded successfully!")
            loading_window.update()
            time.sleep(1)  # Let the user see success message

            # Close loading window
            loading_window.destroy()

            self.log_event("ML detector initialized successfully")
            return True
        except Exception as e:
            if 'loading_window' in locals() and loading_window.winfo_exists():
                loading_window.destroy()

            messagebox.showerror("ML Error", f"Failed to initialize ML detector: {str(e)}")
            self.log_event(f"ML detector initialization failed: {str(e)}")
            return False

    def update_threshold(self, value):
        """Update the detection threshold value"""
        threshold = int(float(value))
        if self.detection_mode == "parking":
            self.parking_threshold = threshold
        self.threshold_label.config(text=f"Value: {threshold}")

    def get_centroid(self, x, y, w, h):
        """Calculate centroid of a rectangle"""
        return x + w // 2, y + h // 2

    # Find the existing check_parking_space method (around line 592)
    # Replace it with this improved version:
    def check_parking_space(self, imgPro, img):
        """Process frame to check parking spaces with debugging info"""
        space_counter = 0
        for i, (x, y, w, h) in enumerate(self.posList):
            # Ensure coordinates are within image bounds
            if (y >= 0 and y + h < imgPro.shape[0] and
                    x >= 0 and x + w < imgPro.shape[1]):

                # Draw ID number for each space
                cv2.putText(img, str(i), (x + 5, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                imgCrop = imgPro[y:y + h, x:x + w]
                count = cv2.countNonZero(imgCrop)

                if count < self.parking_threshold:
                    color = (0, 255, 0)  # Green for free
                    space_counter += 1
                else:
                    color = (0, 0, 255)  # Red for occupied

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cvzone.putTextRect(img, str(count), (x, y + h - 3), scale=1, thickness=2, offset=0)

        # Update counters
        self.free_spaces = space_counter
        self.occupied_spaces = self.total_spaces - self.free_spaces
        return img

    def detect_vehicles(self, frame1, frame2):
        """Process frames to detect and count vehicles"""
        # Get difference between frames
        d = cv2.absdiff(frame1, frame2)
        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(grey, (5, 5), 0)

        _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw detection line
        line_y = self.line_height
        if line_y >= frame1.shape[0]:
            line_y = frame1.shape[0] - 50
        cv2.line(frame1, (0, line_y), (frame1.shape[1], line_y), (0, 255, 0), 2)

        # Process contours
        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            contour_valid = (w >= self.min_contour_width) and (h >= self.min_contour_height)

            if not contour_valid:
                continue

            cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

            centroid = self.get_centroid(x, y, w, h)
            self.matches.append(centroid)
            cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

        # Check for vehicles crossing the line
        for (x, y) in list(self.matches):
            if (line_y - self.offset) < y < (line_y + self.offset):
                self.vehicle_counter += 1
                self.matches.remove((x, y))

        # Display count
        cvzone.putTextRect(frame1, f"Vehicle Count: {self.vehicle_counter}", (10, 30),
                           scale=2, thickness=2, offset=10, colorR=(0, 200, 0))

        return frame1

    def check_gpu_availability(self):
        """Check and report GPU availability for PyTorch and OpenCV"""

        # Check PyTorch GPU
        torch_gpu_available = torch.cuda.is_available()
        if torch_gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
            self.log_event(f"PyTorch GPU available: {gpu_name} (Count: {gpu_count}, Memory: {gpu_mem:.2f}GB)")
        else:
            self.log_event("PyTorch GPU not available, using CPU")

        # Check OpenCV GPU (CUDA)
        cv_gpu_available = False
        try:
            cv_gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cv_gpu_count > 0:
                cv_gpu_available = True
                self.log_event(f"OpenCV CUDA enabled devices: {cv_gpu_count}")
            else:
                self.log_event("OpenCV CUDA not available")
        except:
            self.log_event("OpenCV CUDA support not compiled")

        return torch_gpu_available, cv_gpu_available

    def gpu_adaptive_threshold(self, img, max_value, adaptive_method, threshold_type, block_size, c):
        """GPU-accelerated adaptive threshold if available"""
        if hasattr(self, 'cv_gpu_available') and self.cv_gpu_available:
            try:
                # Upload to GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)

                # Process on GPU
                gpu_result = cv2.cuda.adaptiveThreshold(
                    gpu_img, max_value, adaptive_method, threshold_type, block_size, c)

                # Download result
                result = gpu_result.download()
                return result
            except Exception as e:
                self.log_event(f"GPU threshold error: {e}, falling back to CPU")

        # CPU fallback
        return cv2.adaptiveThreshold(img, max_value, adaptive_method, threshold_type, block_size, c)

    def gpu_resize(self, img, size):
        """GPU-accelerated resize if available"""
        if hasattr(self, 'cv_gpu_available') and self.cv_gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                gpu_resized = cv2.cuda.resize(gpu_img, size)
                return gpu_resized.download()
            except Exception as e:
                self.log_event(f"GPU resize error: {e}, falling back to CPU")

        return cv2.resize(img, size)

    def update_ml_confidence(self, value):
        """Update ML confidence threshold"""
        confidence = float(value)
        self.ml_confidence = confidence
        self.ml_confidence_label.config(text=f"Value: {confidence:.1f}")
        if self.ml_detector:
            self.ml_detector.confidence_threshold = confidence

    def process_frame(self):
        """Process video frames for the selected detection mode"""
        # Add at the beginning of the process_frame method:
        if not hasattr(self, 'image_width') or not hasattr(self, 'image_height'):
            self.image_width = 640
            self.image_height = 480
            self.log_event("Using default dimensions for display")

        if not self.running:
            return

        # Read frame from video
        success, img = self.video_capture.read()

        # Reset video if at end
        if not success:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, img = self.video_capture.read()
            if not success:
                self.status_label.config(text="Status: Video Error", fg="red")
                self.log_event("Video error occurred")
                self.stop_detection()
                return

        # Resize frame if necessary
        if img.shape[1] != self.image_width or img.shape[0] != self.image_height:
            img = cv2.resize(img, (self.image_width, self.image_height))

        # Process based on selected mode
        # In process_frame method, replace the parking detection part:
        if self.detection_mode == "parking":
            # Process for parking detection
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

            # Use GPU accelerated threshold if available
            imgThreshold = self.gpu_adaptive_threshold(
                imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

            # Check parking spaces
            img = self.check_parking_space(imgDilate, img)


        else:  # Vehicle detection mode

            if not hasattr(self, 'prev_frame'):
                self.prev_frame = img.copy()

            # Add frame skipping for ML detection

            if not hasattr(self, 'frame_count'):
                self.frame_count = 0

                self.frame_skip = 3  # Process every 3rd frame (adjust as needed)

                self.last_detections = []  # Store last successful detections

            self.frame_count += 1

            # Use ML detection if enabled

            if self.use_ml_detection and self.ml_detector:

                try:

                    # Only run ML detection on certain frames to improve performance

                    if self.frame_count % self.frame_skip == 0:

                        # Get vehicle detections

                        detections = self.ml_detector.detect_vehicles(img)

                        # Store for use in skipped frames

                        self.last_detections = detections

                    else:

                        # Use the last known detections for in-between frames

                        detections = self.last_detections

                    # Draw detection line

                    line_y = self.line_height

                    if line_y >= img.shape[0]:
                        line_y = img.shape[0] - 50

                    cv2.line(img, (0, line_y), (img.shape[1], line_y), (0, 255, 0), 2)

                    # Process detections

                    current_centroids = []

                    for x1, y1, x2, y2, score, label in detections:
                        # Draw bounding box

                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Calculate centroid

                        centroid_x = (x1 + x2) // 2

                        centroid_y = (y1 + y2) // 2

                        centroid = (centroid_x, centroid_y)

                        # Add to current centroids

                        current_centroids.append(centroid)

                        # Draw centroid

                        cv2.circle(img, centroid, 5, (0, 0, 255), -1)

                        # Label with confidence

                        label_text = f"{self.ml_detector.classes[label]}: {score:.2f}"

                        cv2.putText(img, label_text, (x1, y1 - 10),

                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Count vehicles crossing the line

                    for centroid in current_centroids:

                        if (line_y - self.offset) < centroid[1] < (line_y + self.offset):

                            if centroid not in self.matches:
                                self.vehicle_counter += 1

                                self.matches.append(centroid)

                    # Clean up old centroids

                    self.matches = [match for match in self.matches

                                    if
                                    any(np.linalg.norm(np.array(match) - np.array(c)) < 50 for c in current_centroids)]

                    # Display count

                    cvzone.putTextRect(img, f"Vehicle Count: {self.vehicle_counter}", (10, 30),

                                       scale=2, thickness=2, offset=10, colorR=(0, 200, 0))


                except Exception as e:

                    self.log_event(f"ML detection error: {str(e)}")

                    # Fallback to traditional method

                    img = self.detect_vehicles(img, self.prev_frame)

            else:

                # Traditional detection method

                img = self.detect_vehicles(img, self.prev_frame)

            self.prev_frame = img.copy()



        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL format and then to Tkinter format
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Display the image
        self.video_canvas.config(width=img.shape[1], height=img.shape[0])
        self.video_canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.video_canvas.image = img_tk  # Keep a reference to prevent garbage collection

        # Update status information
        self.update_status_info()

        if self.detection_mode == "parking" or not self.use_ml_detection:
            self.master.after(10, self.process_frame)  # Standard delay
        else:
            self.master.after(1, self.process_frame)  # Faster for ML detection

    def process_frame_optimized(self):
        """Optimized frame processing method with better GPU handling"""
        if not self.running:
            return

        try:
            # Use video_lock to prevent concurrency issues
            with self.video_lock:
                # Read frame from video
                success, img = self.video_capture.read()

            # Reset video if at end
            if not success:
                with self.video_lock:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, img = self.video_capture.read()
                if not success:
                    self.status_label.config(text="Status: Video Error", fg="red")
                    self.log_event("Video error occurred")
                    self.stop_detection()
                    return

            # Use GPU resize if available
            if img.shape[1] != self.image_width or img.shape[0] != self.image_height:
                img = self.gpu_resize(img, (self.image_width, self.image_height))

            # Process based on selected mode
            if self.detection_mode == "parking":
                img = self.process_parking_detection(img)
            else:  # Vehicle detection mode
                img = self.process_vehicle_detection(img)

            # Update UI with processed image
            self.update_display(img)

            # Schedule next frame with appropriate delay
            delay = 1 if self.use_ml_detection and self.detection_mode == "vehicle" else 10
            self.master.after(delay, self.process_frame_optimized)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                self.log_event("GPU memory error: falling back to CPU")
                if hasattr(self, 'ml_detector') and self.ml_detector:
                    self.ml_detector.device = torch.device('cpu')
                    if hasattr(self.ml_detector, 'model'):
                        self.ml_detector.model.to(self.ml_detector.device)
                self.master.after(10, self.process_frame_optimized)
            else:
                self.log_event(f"Error in frame processing: {str(e)}")
                self.master.after(100, self.process_frame_optimized)  # Retry with longer delay
        except Exception as e:
            self.log_event(f"Error in frame processing: {str(e)}")
            self.master.after(100, self.process_frame_optimized)  # Retry with longer delay

    def update_status_info(self):
        """Update the status information display"""
        if hasattr(self, 'status_info'):
            status_text = f"Total Spaces: {self.total_spaces}\n"
            status_text += f"Free Spaces: {self.free_spaces}\n"
            status_text += f"Occupied: {self.occupied_spaces}\n"
            status_text += f"Vehicles Counted: {self.vehicle_counter}"

            # Update the status info label
            self.status_info.config(text=status_text)

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

    def clear_log(self):
        """Clear the log display"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the log?"):
            self.log_text.config(state="normal")
            self.log_text.delete(1.0, "end")
            self.log_text.config(state="disabled")
            self.log_data = []
            self.log_event("Log cleared")

    def save_log(self):
        """Save the log to a file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.log_dir, f"parking_log_{timestamp}.txt")

            with open(filename, 'w') as f:
                for entry in self.log_data:
                    f.write(entry + "\n")

            messagebox.showinfo("Success", f"Log saved to {filename}")
            self.log_event(f"Log saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save log: {str(e)}")

    def record_current_stats(self):
        """Record current statistics to the stats view"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Insert at the beginning of the treeview
        self.stats_tree.insert("", 0, values=(
            timestamp,
            self.total_spaces,
            self.free_spaces,
            self.occupied_spaces,
            self.vehicle_counter
        ))

        self.log_event("Recorded current statistics")

    def clear_statistics(self):
        """Clear the statistics view"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all statistics?"):
            for item in self.stats_tree.get_children():
                self.stats_tree.delete(item)
            self.log_event("Statistics cleared")

    def export_statistics(self):
        """Export statistics to a CSV file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.log_dir, f"parking_stats_{timestamp}.csv")

            with open(filename, 'w') as f:
                f.write("Timestamp,Total Spaces,Free Spaces,Occupied Spaces,Vehicles Counted\n")

                for item in self.stats_tree.get_children():
                    values = self.stats_tree.item(item)["values"]
                    f.write(f"{values[0]},{values[1]},{values[2]},{values[3]},{values[4]}\n")

            messagebox.showinfo("Success", f"Statistics exported to {filename}")
            self.log_event(f"Statistics exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export statistics: {str(e)}")

    def monitoring_thread(self):
        """Background thread for monitoring and periodic logging"""
        while True:
            # Record stats every hour if detection is running
            if self.running:
                self.record_current_stats()

            # Sleep for an hour (3600 seconds)
            time.sleep(3600)

    def on_closing(self):
        """Handle window closing event"""
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            self.running = False
            if self.video_capture is not None:
                self.video_capture.release()
            self.master.destroy()

    def diagnose_gpu(self):
        """Run comprehensive GPU diagnostics"""
        try:
            # Check PyTorch GPU availability
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                cuda_version = torch.version.cuda

                self.log_event(f"PyTorch CUDA available: Yes")
                self.log_event(f"CUDA Version: {cuda_version}")
                self.log_event(f"GPU Device: {gpu_name}")
                self.log_event(f"GPU Count: {gpu_count}")
                self.log_event(f"Current GPU Device: {torch.cuda.current_device()}")

                # Test GPU memory
                try:
                    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

                    self.log_event(f"GPU Memory: Currently Allocated: {allocated:.2f}MB")
                    self.log_event(f"GPU Memory: Max Allocated: {max_allocated:.2f}MB")
                    self.log_event(f"GPU Memory: Total: {total:.2f}GB")

                    # Test simple GPU operation
                    try:
                        test_tensor = torch.tensor([1., 2., 3.], device='cuda')
                        self.log_event(f"GPU Test: Created test tensor on GPU: {test_tensor.device}")
                    except Exception as e:
                        self.log_event(f"GPU Test Failed: {str(e)}")

                except Exception as e:
                    self.log_event(f"GPU Memory Check Failed: {str(e)}")

            else:
                self.log_event("PyTorch CUDA not available")
                if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
                    self.log_event(f"PyTorch was built with CUDA: {torch.version.cuda}")

                # Check if CUDA is installed but not being found
                import subprocess
                try:
                    nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
                    self.log_event("NVIDIA GPU detected by system but not by PyTorch!")
                    self.log_event("This indicates a PyTorch/CUDA version mismatch")
                except:
                    self.log_event("NVIDIA driver tools (nvidia-smi) not found")

        except Exception as e:
            self.log_event(f"GPU Diagnostics failed: {str(e)}")

if __name__ == "__main__":
    root = Tk()
    app = ParkingManagementSystem(root)
    root.mainloop()