from tkinter import Frame, Label, Button, StringVar, OptionMenu, BooleanVar, Scale, HORIZONTAL, LEFT, RIGHT, TOP, X, Y, \
    BOTH
from PIL import Image, ImageTk
import cv2
import os  # Add os import for path handling


class DetectionTab:
    def __init__(self, parent, controller):
        self.parent = parent
        self.controller = controller

        # Create videos directory if it doesn't exist
        self.videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "videos"))
        if not os.path.exists(self.videos_dir):
            os.makedirs(self.videos_dir)
            print(f"Created videos directory at: {self.videos_dir}")

        # Setup status panel first
        self._setup_status_panel()

        # Initialize variables
        self.video_source_var = StringVar(value="carPark.mp4")
        self.detection_mode_var = StringVar(value="parking")
        self.use_ml_var = BooleanVar(value=False)
        self.running = False
        self.current_frame = None

        # Setup remaining UI components
        self._setup_control_panel()
        self._setup_video_panel()

        # Bind video source change
        self.video_source_var.trace('w', self._on_video_source_change)
        self.detection_mode_var.trace('w', self._on_detection_mode_change)

    def _setup_control_panel(self):
        """Setup the control panel with buttons and options"""
        control_frame = Frame(self.parent)
        control_frame.pack(side=TOP, fill=X, padx=5, pady=5)

        # Video source selection
        Label(control_frame, text="Video Source:").pack(side=LEFT, padx=5)
        video_sources = ["carPark.mp4", "Video.mp4", "sample5.mp4", "0", "newVideo1.mp4", "newVideo2.mp4"]
        OptionMenu(control_frame, self.video_source_var, *video_sources).pack(side=LEFT, padx=5)

        # Detection mode selection
        Label(control_frame, text="Mode:").pack(side=LEFT, padx=5)
        modes = ["parking", "counting"]
        OptionMenu(control_frame, self.detection_mode_var, *modes).pack(side=LEFT, padx=5)

        # ML Detection toggle
        self.ml_toggle = Button(control_frame, text="Toggle ML",
                              command=lambda: self.use_ml_var.set(not self.use_ml_var.get()))
        self.ml_toggle.pack(side=LEFT, padx=5)

        # Start/Stop button
        self.start_button = Button(control_frame, text="Start", command=self.toggle_detection)
        self.start_button.pack(side=RIGHT, padx=5)

    def _setup_video_panel(self):
        """Setup the video display panel"""
        self.video_frame = Frame(self.parent, bg='black')
        self.video_frame.pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=5)

        self.video_label = Label(self.video_frame, bg='black')
        self.video_label.pack(fill=BOTH, expand=True)

    def _setup_status_panel(self):
        """Setup the status information panel"""
        status_frame = Frame(self.parent)
        status_frame.pack(side=TOP, fill=X, padx=5, pady=5)

        # Initialize status_info before it's accessed
        self.status_info = Label(status_frame, text="Status: Ready", font=("Arial", 10))
        self.status_info.pack(side=LEFT, padx=5)

    def toggle_detection(self):
        """Toggle detection on/off"""
        try:
            if not self.running:
                self.start_button.config(text="Stop")
                self.running = True

                # Initialize video source
                source = self.video_source_var.get()
                if source == "0":
                    source = 0  # Webcam
                else:
                    # Use videos directory path
                    source = os.path.join(self.videos_dir, source)
                    if not os.path.exists(source):
                        raise Exception(f"Video file not found: {source}")

                # Start detection in controller
                self.controller.video_capture = cv2.VideoCapture(source)
                if not self.controller.video_capture.isOpened():
                    raise Exception(f"Failed to open video source: {source}")

                # Start processing
                self.controller.running = True
                self.process_video_frame()

            else:
                # Stop detection
                self.start_button.config(text="Start")
                self.running = False
                self.controller.running = False

                # Release video capture
                if self.controller.video_capture:
                    self.controller.video_capture.release()

                self.status_info.config(text="Detection stopped")

        except Exception as e:
            self.status_info.config(text=f"Error: {str(e)}")
            self.running = False
            self.controller.running = False

    def process_video_frame(self):
        """Process and display video frames"""
        if self.running and self.controller.video_capture:
            try:
                ret, frame = self.controller.video_capture.read()
                if ret:
                    # Convert frame to RGB for PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process frame based on detection mode
                    if self.detection_mode_var.get() == "parking":
                        processed_frame = self.controller.process_parking_frame(frame)
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    else:  # counting mode
                        processed_frame = self.controller.process_counting_frame(frame)
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                    # Convert to PIL Image and then to PhotoImage
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)

                    # Update display
                    self.video_label.config(image=imgtk)
                    self.video_label.image = imgtk  # Keep a reference!

                    # Schedule next frame
                    self.parent.after(30, self.process_video_frame)
                else:
                    # Video ended or failed to read frame
                    self.toggle_detection()
                    self.controller.log_event("Video ended or failed to read frame")

            except Exception as e:
                self.controller.log_event(f"Processing error: {str(e)}")
                self.toggle_detection()

    def _toggle_ml_detection(self):
        """Toggle ML detection mode"""
        try:
            current_state = self.use_ml_var.get()
            self.use_ml_var.set(not current_state)

            # Update button appearance
            self.ml_toggle.config(
                relief="sunken" if self.use_ml_var.get() else "raised",
                text="ML ON" if self.use_ml_var.get() else "ML OFF"
            )

            # Restart detection if running
            if self.running:
                self.toggle_detection()
                self.toggle_detection()

            self.status_info.config(
                text=f"ML Detection {'enabled' if self.use_ml_var.get() else 'disabled'}"
            )

        except Exception as e:
            self.status_info.config(text=f"Error toggling ML: {str(e)}")

    def update_status(self, text):
        """Update status display"""
        if hasattr(self, 'status_info'):
            self.status_info.config(text=text)

    def update_video_display(self, frame):
        """Update the video display with a new frame"""
        if hasattr(self, 'video_label'):
            self.video_label.config(image=frame)
            self.video_label.image = frame

    def _on_video_source_change(self, *args):
        """Handle video source change event"""
        try:
            # Stop current detection if running
            if self.running:
                self.toggle_detection()

            # Update reference image based on video source
            video_source = self.video_source_var.get()
            if video_source in self.controller.video_reference_map:
                ref_image = self.controller.video_reference_map[video_source]
                self.controller.current_reference_image = ref_image
                self.controller.load_parking_positions(ref_image)
                self.controller.log_event(f"Changed video source to {video_source}")
        except Exception as e:
            self.controller.log_event(f"Error changing video source: {str(e)}")

    def _on_detection_mode_change(self, *args):
        """Handle detection mode change event"""
        try:
            mode = self.detection_mode_var.get()
            # Stop current detection if running
            if self.running:
                self.toggle_detection()
            self.controller.detection_mode = mode
            self.controller.log_event(f"Changed detection mode to {mode}")
        except Exception as e:
            self.controller.log_event(f"Error changing detection mode: {str(e)}")