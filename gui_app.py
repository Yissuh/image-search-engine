
import sys
import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import threading
import time
from datetime import datetime
from dataset_indexer import RotatedImageIndexer

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QGroupBox, QFileDialog, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QPixmap, QImage

class Communicate(QObject):
    update_status = pyqtSignal(str)

class CameraThread(QThread):
    frame_ready = pyqtSignal(QPixmap)
    error = pyqtSignal(str)

    def __init__(self, cap, get_label_size_func, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.get_label_size_func = get_label_size_func
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, _ = frame.shape
                    qimg = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    label_size = self.get_label_size_func()
                    if label_size.width() > 0 and label_size.height() > 0:
                        pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
                    self.frame_ready.emit(pixmap)
                else:
                    self.error.emit("Camera Status: Error - Unable to read frame")
            self.msleep(30)  # ~33 FPS

    def stop(self):
        self.running = False
        self.wait()

class ImageSearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Product Image Search System")
        self.resize(900, 750)
        self.comm = Communicate()
        self.comm.update_status.connect(self._update_indexer_status_main_thread)
        self.indexer_processor = RotatedImageIndexer(dataset_dir="")
        self.cap = None
        self.preview_running = False
        self.init_ui()

    def init_ui(self):
        tabs = QTabWidget()
        self.capture_tab = QWidget()
        self.indexer_tab = QWidget()
        tabs.addTab(self.capture_tab, "Product Capture")
        tabs.addTab(self.indexer_tab, "Dataset Indexer")
        self.setCentralWidget(tabs)
        self.setup_capture_tab()
        self.setup_indexer_tab()

    def setup_capture_tab(self):
        layout = QVBoxLayout()
        # Product Details
        details_group = QGroupBox("Product Details")
        details_layout = QFormLayout()
        self.entries = {}
        for field in ["Name", "Flavor", "Barcode"]:
            entry = QLineEdit()
            self.entries[field.lower()] = entry
            details_layout.addRow(f"{field}: ", entry)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        # Camera Controls
        controls_group = QGroupBox("Camera Controls")
        controls_layout = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([str(i) for i in range(3)])
        controls_layout.addWidget(QLabel("Camera:"))
        controls_layout.addWidget(self.camera_combo)
        test_btn = QPushButton("Test Camera")
        test_btn.clicked.connect(self.test_camera)
        controls_layout.addWidget(test_btn)
        self.status_label = QLabel("Camera Status: Not Connected")
        controls_layout.addWidget(self.status_label)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Capture Button
        capture_btn = QPushButton("Capture & Process")
        capture_btn.clicked.connect(self.capture_image)
        layout.addWidget(capture_btn)

        # Camera Preview
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group, stretch=1)

        self.capture_tab.setLayout(layout)

        # Add a placeholder image
        placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.putText(placeholder, "Camera Preview", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        img = Image.fromarray(placeholder.astype('uint8'))
        qimg = QImage(img.tobytes(), img.width, img.height, img.width*3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pixmap.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio))

        # Camera setup
        self.camera_index = 0
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        self.initialize_camera()

    def setup_indexer_tab(self):
        layout = QVBoxLayout()
        # Dataset Path
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Dataset Directory:"))
        self.dataset_path_entry = QLineEdit("./dataset")
        path_layout.addWidget(self.dataset_path_entry)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_dataset)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # Index/PKL file names
        file_layout = QHBoxLayout()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.index_path_entry = QLineEdit(f"dataset-{timestamp}.index")
        self.meta_path_entry = QLineEdit(f"dataset-{timestamp}.pkl")
        file_layout.addWidget(QLabel("Index File:"))
        file_layout.addWidget(self.index_path_entry)
        file_layout.addWidget(QLabel("Metadata PKL:"))
        file_layout.addWidget(self.meta_path_entry)
        layout.addLayout(file_layout)

        # Index Button
        index_btn = QPushButton("Index Dataset")
        index_btn.clicked.connect(self.run_indexer)
        layout.addWidget(index_btn)

        # Status Display
        self.indexer_status = QTextEdit()
        self.indexer_status.setReadOnly(True)
        layout.addWidget(self.indexer_status, stretch=1)

        self.indexer_tab.setLayout(layout)

    def change_camera(self, idx):
        self.camera_index = idx
        self.initialize_camera()

    def browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Directory", "./dataset")
        if folder:
            self.dataset_path_entry.setText(folder)

    def run_indexer(self):
        dataset_path = self.dataset_path_entry.text()
        index_path = self.index_path_entry.text()
        meta_path = self.meta_path_entry.text()
        if not os.path.isdir(dataset_path):
            QMessageBox.critical(self, "Error", f"Directory not found: {dataset_path}")
            return
        self.indexer_status.setPlainText(f"Starting indexing of {dataset_path}...\n")
        thread = threading.Thread(target=self._run_indexer_thread, args=(dataset_path, index_path, meta_path), daemon=True)
        thread.start()

    def _run_indexer_thread(self, dataset_path, index_path, meta_path):
        try:
            start_time = time.time()
            self.indexer_processor = RotatedImageIndexer(dataset_dir=dataset_path, index_path=index_path, meta_path=meta_path)
            self.indexer_processor.index_images()
            elapsed = time.time() - start_time
            self.comm.update_status.emit(f"\nIndexing completed in {elapsed:.2f} seconds.")
        except Exception as e:
            self.comm.update_status.emit(f"\nError during indexing: {str(e)}")

    def _update_indexer_status_main_thread(self, text):
        self.indexer_status.moveCursor(self.indexer_status.textCursor().End)
        self.indexer_status.insertPlainText(text)
        self.indexer_status.ensureCursorVisible()

    def test_camera(self):
        camera_index = self.camera_combo.currentIndex()
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                QMessageBox.information(self, "Camera Test", f"Camera {camera_index} is working!")
            else:
                QMessageBox.critical(self, "Camera Test", f"Camera {camera_index} failed to capture frame")
        else:
            QMessageBox.critical(self, "Camera Test", f"Failed to open camera {camera_index}")
        cap.release()

    def initialize_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.status_label.setText(f"Camera Status: Not Available (Camera {self.camera_index})")
            self.preview_running = False
            return
        self.status_label.setText(f"Camera Status: Connected (Camera {self.camera_index})")
        self.preview_running = True
        self.start_preview()

    def start_preview(self):
        if hasattr(self, 'camera_thread') and self.camera_thread is not None:
            self.camera_thread.stop()
        self.camera_thread = CameraThread(self.cap, self.preview_label.size, self)
        self.camera_thread.frame_ready.connect(self.on_frame_ready)
        self.camera_thread.error.connect(self.on_camera_error)
        self.camera_thread.start()

    def stop_preview(self):
        if hasattr(self, 'camera_thread') and self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None

    def on_frame_ready(self, pixmap):
        self.preview_label.setPixmap(pixmap)

    def on_camera_error(self, msg):
        self.status_label.setText(msg)

    # Remove update_camera_preview (now handled by thread)

    def capture_image(self):
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.critical(self, "Capture Error", "Camera is not connected")
            return
        product_name = self.entries['name'].text()
        flavor = self.entries['flavor'].text()
        barcode = self.entries['barcode'].text()
        folder_name = f"{product_name}_{flavor}_{barcode}".replace(' ', '_')
        output_dir = os.path.join('dataset', folder_name)
        os.makedirs(output_dir, exist_ok=True)
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = remove(rgb_frame)
            if not isinstance(processed, np.ndarray):
                processed = np.array(processed)
            for angle in range(0, 360, 45):
                rotated = self.rotate_image(processed, angle)
                if rotated.shape[2] == 4:
                    out_img = cv2.cvtColor(rotated, cv2.COLOR_RGBA2BGRA)
                else:
                    out_img = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
                filename = os.path.join(output_dir, f"{angle}_degrees.png")
                cv2.imwrite(filename, out_img)
            QMessageBox.information(self, "Success", f"Images saved to {output_dir}")
        else:
            QMessageBox.critical(self, "Capture Error", "Failed to capture image")

    def rotate_image(self, image, angle):
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        if image.shape[2] == 4:
            rotated = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        else:
            rotated = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        return rotated


    def closeEvent(self, event):
        self.preview_running = False
        self.stop_preview()
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    app = QApplication(sys.argv)
    window = ImageSearchApp()
    window.setStyleSheet("""
        QMainWindow {
            background-color: #f8f9fa;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #bbb;
            border-radius: 8px;
            margin-top: 10px;
        }
        QGroupBox:title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QLabel, QLineEdit, QTextEdit {
            font-size: 15px;
        }
        QPushButton {
            background-color: #0078d4;
            color: white;
            border-radius: 5px;
            padding: 6px 14px;
        }
        QPushButton:hover {
            background-color: #005fa3;
        }
    """)
    window.show()
    sys.exit(app.exec())

