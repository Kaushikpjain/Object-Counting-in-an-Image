import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import time
from threading import Thread


class ObjectCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Counter")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f2f5")

        # Variables
        self.cap = None
        self.running = False
        self.min_area = 500
        self.threshold_value = 180
        self.blur_size = 5
        self.detection_color = (0, 255, 0)  # Green boxes for detected objects

        # GUI Setup
        self.setup_ui()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50")
        header_frame.pack(fill="x", padx=10, pady=10)

        self.header = tk.Label(
            header_frame,
            text="Object Counter",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        self.header.pack(pady=15)

        # Main Content
        main_frame = tk.Frame(self.root, bg="#f0f2f5")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left Panel (Controls)
        control_frame = tk.Frame(main_frame, bg="#ecf0f1", bd=2, relief="groove")
        control_frame.pack(side="left", fill="y", padx=(0, 10), pady=10)

        # Settings Label
        settings_label = tk.Label(
            control_frame,
            text="Detection Settings",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            fg="black"
        )
        settings_label.pack(pady=(10, 5))

        # Minimum Area Slider
        self.min_area_slider = self.create_slider(
            control_frame,
            "Minimum Object Area:",
            100, 2000, self.min_area,
            self.update_min_area
        )

        # Threshold Slider
        self.threshold_slider = self.create_slider(
            control_frame,
            "Threshold Value:",
            0, 255, self.threshold_value,
            self.update_threshold
        )

        # Blur Size Slider
        self.blur_slider = self.create_slider(
            control_frame,
            "Blur Size (odd numbers):",
            1, 15, self.blur_size,
            self.update_blur_size
        )

        # Buttons
        button_frame = tk.Frame(control_frame, bg="#ecf0f1")
        button_frame.pack(pady=20)

        self.btn_upload = ttk.Button(
            button_frame,
            text="Upload Image",
            command=self.upload_image,
            style="Accent.TButton"
        )
        self.btn_upload.pack(fill="x", pady=5)

        self.btn_webcam = ttk.Button(
            button_frame,
            text="Start Webcam",
            command=self.toggle_webcam,
            style="Accent.TButton"
        )
        self.btn_webcam.pack(fill="x", pady=5)

        self.btn_save = ttk.Button(
            button_frame,
            text="Save Result",
            command=self.save_result,
            state="disabled"
        )
        self.btn_save.pack(fill="x", pady=5)

        # Results in Control Panel
        self.count_label = tk.Label(
            button_frame,
            text="Objects Detected: 0",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            fg="black"
        )
        self.count_label.pack(pady=(10, 0))

        self.fps_label = tk.Label(
            button_frame,
            text="FPS: -",
            font=("Arial", 10),
            bg="#ecf0f1",
            fg="black"
        )
        self.fps_label.pack(pady=(2, 10))

        # Right Panel (Display)
        display_frame = tk.Frame(main_frame, bg="#bdc3c7")
        display_frame.pack(side="right", fill="both", expand=True, pady=10)

        self.canvas = tk.Canvas(display_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # Configure styles
        self.style = ttk.Style()
        self.style.configure("Accent.TButton", font=("Arial", 12), padding=10)

    def create_slider(self, parent, label_text, from_, to, initial, command):
        frame = tk.Frame(parent, bg="#ecf0f1")
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(
            frame,
            text=label_text,
            font=("Arial", 10),
            bg="#ecf0f1",
            fg="black"
        ).pack(anchor="w")

        slider = ttk.Scale(
            frame,
            from_=from_,
            to=to,
            value=initial,
            command=lambda v: command(int(float(v))),
            orient="horizontal"
        )
        slider.pack(fill="x")

        value_label = tk.Label(
            frame,
            text=str(initial),
            font=("Arial", 10),
            bg="#ecf0f1",
            fg="black",
            width=5
        )
        value_label.pack(side="right")

        return slider, value_label

    def update_min_area(self, value):
        self.min_area = value
        self.min_area_slider[1].config(text=str(value))

    def update_threshold(self, value):
        self.threshold_value = value
        self.threshold_slider[1].config(text=str(value))

    def update_blur_size(self, value):
        value = int(value)
        value = value + 1 if value % 2 == 0 else value
        self.blur_size = value
        self.blur_slider[1].config(text=str(value))

    def count_objects_in_frame(self, frame):
        start_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        good_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                good_contours.append(cnt)
                count += 1

        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        return count, good_contours, fps

    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        frame = cv2.resize(frame, (640, 480))
        count, contours, fps = self.count_objects_in_frame(frame)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.detection_color, 2)

        self.update_display(frame)
        self.count_label.config(text=f"Objects Detected: {count}")
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.btn_save.config(state="normal")
        self.last_result = frame

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.process_image(file_path)

    def toggle_webcam(self):
        if not self.running:
            self.start_webcam()
        else:
            self.stop_webcam()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam.")
            return

        self.running = True
        self.btn_webcam.config(text="Stop Webcam")
        self.btn_upload.config(state="disabled")
        self.btn_save.config(state="disabled")

        self.thread = Thread(target=self.webcam_loop, daemon=True)
        self.thread.start()

    def stop_webcam(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.btn_webcam.config(text="Start Webcam")
        self.btn_upload.config(state="normal")
        self.btn_save.config(state="normal")
        self.count_label.config(text="Webcam stopped")

    def webcam_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            count, contours, fps = self.count_objects_in_frame(frame)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.detection_color, 2)

            self.update_display(frame)
            self.count_label.config(text=f"Objects Detected: {count}")
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.last_result = frame

            time.sleep(0.01)

    def update_display(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height

        if canvas_ratio > img_ratio:
            new_height = canvas_height
            new_width = int(new_height * img_ratio)
        else:
            new_width = canvas_width
            new_height = int(new_width / img_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            anchor="center",
            image=self.photo
        )

    def save_result(self):
        if hasattr(self, 'last_result'):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", ".png"), ("JPEG Image", ".jpg"), ("All Files", ".*")]
            )
            if file_path:
                cv2.imwrite(file_path, self.last_result)
                messagebox.showinfo("Success", "Image saved successfully!")

    def on_closing(self):
        self.stop_webcam()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
