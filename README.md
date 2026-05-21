Face Detection Pro

An AI-powered desktop application for face detection in images; either single or batch processing with live preview.

Built with Python, OpenCV, and CustomTkinter. This application uses a deep neural network (Caffe SSD) trained on thousands of sample human faces. Appropriate for offline operation.


WHAT IT DOES

This tool detects human faces in images utilizing a pre-trained deep learning model. There are two modes available: single image with live preview, or batch folder processing for multiple images at once.

Adjust the detection confidence with a slider to set the desired sensitivity of the software for finding faces. Save the result when satisfied with the output.


HOW IT WORKS (THE SCIENCE)

A Caffe SSD deep neural network is utilized in developing Face Detection Pro. Caffe SSD is a convolutional neural network designed for detecting objects. The model has been trained on a large dataset of images of human beings and runs totally on your own system with no internet connection required.

Unlike older Haar Cascade methods (which frequently identify objects like balls or clothing as human faces), this DNN model is much more accurate and gives fewer false positives. It also detects faces at different angles, not just at front views.

This is applied machine vision and deep learning in a desktop application.


COMPETITIVE ADVANTAGE

Majority of online face detection tools require uploading (privacy risk). On the other hand, this application works offline.

- Deep learning model (not old Haar Cascade)
- Live preview with adjustable confidence
- Single image and folder processing (batch) modes
- No internet required
- No privacy concerns
- Built by a physicist specialized in optics and image processing


DEPENDENCIES

Python libraries (install once):
pip install opencv-python customtkinter pillow numpy

AI model files (one-time download, place in ~/models/):
- deploy.prototxt (28 KB)
- res10_300x300_ssd_iter_140000.caffemodel (10.7 MB)

Download both from the OpenCV GitHub repository.


HOW TO RUN

python face_detector_pro.py


AUTHOR

Yuseph Alvandi
PhD in Optics and Laser Physics
Python Developer and Image Processing Specialist

GitHub: https://github.com/YusephAlvandi


LICENSE

MIT License