Face Detection Pro

AI-powered face detection tool — single image or batch folder processing.

Built with Python, OpenCV, and CustomTkinter. Uses a deep learning model (Caffe SSD) for highly accurate face detection. Works completely offline.


WHAT IT DOES

Face Detection Pro detects human faces in images using a pre-trained deep neural network. It offers two modes:

1. Single Image — Open one image, preview the result live, adjust confidence, and save the output.

2. Batch Folder — Select a folder of images, process them all at once, and save the results automatically.


HOW IT WORKS

Unlike older tools that use Haar Cascade (which mistakes balls and elbows for faces), Face Detection Pro uses a Caffe SSD deep learning model — the same type of AI used in professional surveillance systems.

Compared to Haar Cascade:

- Haar Cascade: Low accuracy, many false positives, cannot detect profile faces, no confidence control

- Face Detection Pro (Caffe DNN): High accuracy, far fewer mistakes, often detects profile faces, adjustable confidence threshold


CONFIDENCE THRESHOLD

A single slider controls how strict the detection is:

- Lower (0.2-0.4): Find more faces, but may include some false detections

- Higher (0.6-0.9): Only return faces the model is very confident about


HOW TO RUN

1. Install dependencies:
pip install opencv-python customtkinter pillow numpy

2. Download the AI model files (one-time setup):
deploy.prototxt (28 KB)
res10_300x300_ssd_iter_140000.caffemodel (10.7 MB)

Place both files in a models folder.

3. Run the application:
python face_detector_pro.py


TECH STACK

Python 3.x, OpenCV, CustomTkinter, NumPy, Pillow


AUTHOR

Yuseph Alvandi
PhD in Atomic and Molecular Physics (Optics and Laser)
Python Developer and Image Processing Specialist

GitHub: https://github.com/YusephAlvandi


LICENSE

MIT License