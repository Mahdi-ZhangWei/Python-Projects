import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class FaceEyeSmileDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the Haar cascades
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

        # Set up the UI
        self.setWindowTitle("Face, Eye, and Smile Detection")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close_application)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.exit_button)
        self.central_widget.setLayout(layout)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.image_label.setText("Error: Unable to access the camera.")
            return

        # Set up a timer for real-time video processing
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.image_label.setText("Failed to grab frame. Exiting.")
            self.timer.stop()
            return

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around faces and detect eyes and smiles within each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

            # Detect smiles
            smiles = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

        # Convert the frame to QImage for display in QLabel
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = channel * width
        q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Display the frame in the QLabel
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def close_application(self):
        self.timer.stop()
        self.cap.release()
        self.close()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = FaceEyeSmileDetectionApp()
    window.show()
    sys.exit(app.exec_())
