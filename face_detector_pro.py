"""
Face Detection Pro - Batch & Single Image Processing
Author: Yuseph Alvandi
Description: AI-powered face detection with both folder and single image support.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FaceDetectionApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Face Detection Pro")
        self.window.geometry("1100x800")
        self.window.configure(fg_color="#0a0a0a")
        
        # Single image
        self.image_path = None
        self.original_img = None
        self.current_img = None
        
        # Batch processing
        self.input_folder = ""
        self.output_folder = ""
        
        self.confidence = ctk.DoubleVar(value=0.5)
        self.mode = ctk.StringVar(value="single")
        
        # Path to Caffe model files
        self.proto_path = os.path.expanduser("~/python_projects/image_processing/models/deploy.prototxt")
        self.model_path = os.path.expanduser("~/python_projects/image_processing/models/res10_300x300_ssd_iter_140000.caffemodel")
        
        self.setup_ui()
    
    def setup_ui(self):
        header = ctk.CTkFrame(self.window, fg_color="transparent")
        header.pack(fill="x", pady=(20, 10), padx=30)
        ctk.CTkLabel(header, text="😊 Face Detection Pro", font=ctk.CTkFont(size=32, weight="bold"), text_color="#1E90FF").pack()
        ctk.CTkLabel(header, text="AI-powered face detection — Single image or batch folder", font=ctk.CTkFont(size=14), text_color="#AAAAAA").pack()
        
        content = ctk.CTkFrame(self.window, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=10)
        
        # Left panel
        left = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12, width=380)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        
        ctk.CTkLabel(left, text="Controls", font=ctk.CTkFont(size=18, weight="bold"), text_color="#1E90FF").pack(pady=(20, 15))
        
        # Mode selection
        mode_frame = ctk.CTkFrame(left, fg_color="transparent")
        mode_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkRadioButton(mode_frame, text="Single Image", variable=self.mode, value="single", command=self.toggle_mode).pack(side="left", padx=5)
        ctk.CTkRadioButton(mode_frame, text="Batch Folder", variable=self.mode, value="batch", command=self.toggle_mode).pack(side="left", padx=5)
        
        # Single image controls
        self.single_frame = ctk.CTkFrame(left, fg_color="transparent")
        ctk.CTkButton(self.single_frame, text="📂 Open Image", command=self.open_image, height=40, font=ctk.CTkFont(size=14)).pack(pady=10, padx=20, fill="x")
        
        self.file_label = ctk.CTkLabel(self.single_frame, text="No file selected", text_color="#888888", font=ctk.CTkFont(size=12))
        self.file_label.pack(pady=5)
        
        ctk.CTkButton(self.single_frame, text="💾 Save Result", command=self.save_single_result, height=40, fg_color="#2ECC71", font=ctk.CTkFont(size=14)).pack(pady=10, padx=20, fill="x")
        
        # Batch controls
        self.batch_frame = ctk.CTkFrame(left, fg_color="transparent")
        
        ctk.CTkButton(self.batch_frame, text="📁 Select Input Folder", command=self.select_input, height=35, font=ctk.CTkFont(size=13)).pack(pady=5, padx=20, fill="x")
        self.input_label = ctk.CTkLabel(self.batch_frame, text="Not selected", text_color="#888888", font=ctk.CTkFont(size=11))
        self.input_label.pack(pady=2)
        
        ctk.CTkButton(self.batch_frame, text="📁 Select Output Folder", command=self.select_output, height=35, font=ctk.CTkFont(size=13)).pack(pady=5, padx=20, fill="x")
        self.output_label = ctk.CTkLabel(self.batch_frame, text="Not selected", text_color="#888888", font=ctk.CTkFont(size=11))
        self.output_label.pack(pady=2)
        
        ctk.CTkButton(self.batch_frame, text="🚀 Process All Images", command=self.process_batch, height=40, fg_color="#E67E22", font=ctk.CTkFont(size=14)).pack(pady=10, padx=20, fill="x")
        
        # Confidence slider (shared)
        ctk.CTkLabel(left, text="Confidence Threshold", font=ctk.CTkFont(size=13, weight="bold"), text_color="#CCCCCC").pack(pady=(15, 5))
        ctk.CTkLabel(left, text="Higher = Stricter", font=ctk.CTkFont(size=10), text_color="#888888").pack()
        ctk.CTkSlider(left, from_=0.1, to=0.9, variable=self.confidence, width=280, command=self.on_confidence_change).pack()
        self.conf_label = ctk.CTkLabel(left, text="0.50", font=ctk.CTkFont(size=12), text_color="#1E90FF")
        self.conf_label.pack()
        
        self.face_count_label = ctk.CTkLabel(left, text="Faces: 0", font=ctk.CTkFont(size=16, weight="bold"), text_color="#4CAF50")
        self.face_count_label.pack(pady=(20, 5))
        
        self.status_label = ctk.CTkLabel(left, text="Ready", text_color="#4CAF50", font=ctk.CTkFont(size=11))
        self.status_label.pack(pady=10)
        
        # Right panel (Preview)
        right = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12)
        right.pack(side="right", fill="both", expand=True)
        
        self.preview_label = ctk.CTkLabel(right, text="No Image Loaded", font=ctk.CTkFont(size=16), text_color="#555555")
        self.preview_label.pack(expand=True)
        
        # Show single mode by default
        self.toggle_mode()
    
    def toggle_mode(self):
        if self.mode.get() == "single":
            self.batch_frame.pack_forget()
            self.single_frame.pack(fill="x", pady=10)
        else:
            self.single_frame.pack_forget()
            self.batch_frame.pack(fill="x", pady=10)
    
    # ========== SINGLE IMAGE ==========
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        
        self.image_path = path
        self.original_img = cv2.imread(path)
        self.file_label.configure(text=os.path.basename(path))
        self.status_label.configure(text="Image loaded. Adjust confidence if needed.", text_color="#4CAF50")
        self.detect_and_show()
    
    def detect_and_show(self):
        if self.original_img is None:
            return
        
        img = self.original_img.copy()
        faces = self.detect_faces(img)
        
        for (startX, startY, endX, endY) in faces:
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        self.current_img = img
        self.face_count_label.configure(text=f"Faces: {len(faces)}")
        
        # Show preview
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        preview_w = 600
        ratio = preview_w / pil_img.width
        preview_h = int(pil_img.height * ratio)
        pil_img = pil_img.resize((preview_w, preview_h))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.preview_label.configure(image=tk_img, text="")
        self.preview_label.image = tk_img
    
    def save_single_result(self):
        if self.current_img is None:
            messagebox.showerror("Error", "No processed image to save!")
            return
        
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if path:
            cv2.imwrite(path, self.current_img)
            self.status_label.configure(text=f"Saved: {os.path.basename(path)}", text_color="#4CAF50")
    
    # ========== BATCH PROCESSING ==========
    def select_input(self):
        self.input_folder = filedialog.askdirectory(title="Select Input Folder with Images")
        if self.input_folder:
            self.input_label.configure(text=f"✓ {os.path.basename(self.input_folder)}", text_color="#4CAF50")
    
    def select_output(self):
        self.output_folder = filedialog.askdirectory(title="Select Output Folder for Results")
        if self.output_folder:
            self.output_label.configure(text=f"✓ {os.path.basename(self.output_folder)}", text_color="#4CAF50")
    
    def process_batch(self):
        if not self.input_folder or not self.output_folder:
            messagebox.showerror("Error", "Please select both input and output folders!")
            return
        
        supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        total_faces = 0
        processed = 0
        
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(supported):
                try:
                    img = cv2.imread(os.path.join(self.input_folder, filename))
                    faces = self.detect_faces(img)
                    
                    for (startX, startY, endX, endY) in faces:
                        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    
                    output_path = os.path.join(self.output_folder, f"detected_{filename}")
                    cv2.imwrite(output_path, img)
                    
                    total_faces += len(faces)
                    processed += 1
                    self.status_label.configure(text=f"🔄 {filename}: {len(faces)} faces", text_color="#FFAA33")
                    self.window.update()
                    
                except Exception as e:
                    self.status_label.configure(text=f"⚠️ Skipped: {filename}", text_color="#FF5555")
                    self.window.update()
        
        self.status_label.configure(text=f"✅ Done! {processed} images, {total_faces} faces.", text_color="#4CAF50")
        messagebox.showinfo("Complete", f"Processed {processed} images.\nDetected {total_faces} face(s).")
    
    # ========== CORE DETECTION ENGINE ==========
    def detect_faces(self, img):
        """Returns list of (startX, startY, endX, endY) for each detected face."""
        net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        (h, w) = img.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence.get():
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
        
        return faces
    
    def on_confidence_change(self, value):
        self.conf_label.configure(text=f"{self.confidence.get():.2f}")
        if self.mode.get() == "single" and self.original_img is not None:
            self.detect_and_show()
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = FaceDetectionApp()
    app.run()