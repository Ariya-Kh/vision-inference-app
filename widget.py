# This Python file uses the following encoding: utf-8
import cv2
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QFileDialog, QGraphicsScene
from PySide6.QtGui import QPixmap, QImage, QFont
from app.yolo.yolov8_runner import YOLOv8ONNXRunner
from app.yolo.yolo26_runner import YOLO26ONNXRunner
from app.rfdetr.rfdetr_runner import RFDETRRunner
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox
import numpy as np
import time
from app.gui.ui_form import Ui_Widget

# Important:
# You need to run the following command to generate the ui_form.py file
#     source ~/vision-app-env/bin/activate
#     pyside6-uic app/gui/form.ui -o app/gui/ui_form.py
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py

MODEL_CONFIG = {
    "YOLOv3": {
        "tasks": ["Detect"],
        "show_nms": True,
        "runner": YOLOv8ONNXRunner
    },
    "YOLOv5": {
        "tasks": ["Detect"],
        "show_nms": True,
        "runner": YOLOv8ONNXRunner
    },
    "YOLOv6": {
        "tasks": ["Detect"],
        "show_nms": True,
        "runner": YOLOv8ONNXRunner
    },
    "YOLOv8": {
        "tasks": ["Detect", "Segment"],
        "show_nms": True,
        "runner": YOLOv8ONNXRunner
    },
    "YOLOv9": {
        "tasks": ["Detect", "Segment"],
        "show_nms": True,
        "runner": YOLOv8ONNXRunner
    },
    "YOLOv10": {
        "tasks": ["Detect"],
        "show_nms": False,
        "runner": YOLO26ONNXRunner
    },
    "YOLO11": {
        "tasks": ["Detect", "Segment"],
        "show_nms": True,
        "runner": YOLOv8ONNXRunner
    },
    "YOLO12": {
        "tasks": ["Detect", "Segment"],
        "show_nms": True,
        "runner": YOLOv8ONNXRunner
    },
    "YOLO26": {
        "tasks": ["Detect", "Segment"],
        "show_nms": False,
        "runner": YOLO26ONNXRunner
    },
    "RFDETR": {
        "tasks": ["Detect", "Segment"],
        "show_nms": False,
        "runner": RFDETRRunner
    }
}


class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.scene_orig = QGraphicsScene(self)
        self.scene_res = QGraphicsScene(self)
        self.ui.labelProcess.setFont(QFont('Arial', 14))

        self.ui.graphicsView.setScene(self.scene_orig)
        self.ui.graphicsViewRes.setScene(self.scene_res)
        self.ui.btnRunInference.clicked.connect(self.run_inference)
        self.runner = None  # your runner
        self.ui.comboBoxModel.insertItems(0, ["YOLOv3", "YOLOv5", "YOLOv6", "YOLOv8", "YOLOv9", "YOLOv10", "YOLO11", "YOLO12", "YOLO26", "RFDETR"])
        # self.ui.comboBoxModel.
        self.ui.comboBoxTask.insertItems(0, ["Detect"])
        self.ui.comboBoxDevice.insertItems(0, ["GPU", "CPU"])
        self.ui.btnOpenImage.clicked.connect(self.open_image)
        self.ui.btnLoadModel.clicked.connect(self.load_model)

        self.ui.horizontalSliderConf.valueChanged.connect(self.slider_changed)
        self.ui.lineEditConf.editingFinished.connect(self.edit_changed)

        # initial value
        self.ui.horizontalSliderConf.setValue(25)
        self.ui.lineEditConf.setText("25")

        self.ui.horizontalSliderNMS.valueChanged.connect(self.slider_changed_nms)
        self.ui.comboBoxModel.currentTextChanged.connect(self.model_changed)
        self.ui.lineEditNMS.editingFinished.connect(self.edit_changed_nms)

        # initial value
        self.ui.horizontalSliderNMS.setValue(50)
        self.ui.lineEditNMS.setText("50")

        self.model_loaded = False
        self.current_image = None
        self.current_image_path = None


    def model_changed(self):
        model_name = self.ui.comboBoxModel.currentText()

        cfg = MODEL_CONFIG.get(model_name)

        self.ui.comboBoxTask.clear()
        self.ui.comboBoxTask.addItems(cfg["tasks"])

        self.ui.horizontalSliderNMS.setVisible(cfg["show_nms"])
        self.ui.lineEditNMS.setVisible(cfg["show_nms"])
        self.ui.labelNMS.setVisible(cfg["show_nms"])

        self.model_loaded = False
        self.ui.labelModelInfo.setText("")


    def slider_changed_nms(self, value):
        # int -> float
        self.ui.lineEditNMS.setText(f"{value}")


    def edit_changed_nms(self):
        value = int(self.ui.lineEditNMS.text())

        self.ui.horizontalSliderNMS.setValue(value)


    def slider_changed(self, value):
        # int -> float
        self.ui.lineEditConf.setText(f"{value}")


    def edit_changed(self):
        value = int(self.ui.lineEditConf.text())

        self.ui.horizontalSliderConf.setValue(value)



    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model",
            "",
            "*.onnx"
        )

        if not file_path:
            return

        model_name = self.ui.comboBoxModel.currentText()

        cfg = MODEL_CONFIG.get(model_name)
        if cfg is None:
            raise ValueError(f"Unsupported model: {model_name}")

        runner_cls = cfg.get("runner")
        if runner_cls is None:
            raise ValueError(f"No runner defined for model: {model_name}")

        self.runner = runner_cls()
        self.runner.load_model(file_path, True) if self.ui.comboBoxDevice.currentText() == "GPU" else self.runner.load_model(file_path, False)
        cuda_warning = self.runner.cuda_warning
        if cuda_warning:
            QMessageBox.warning(self, "CUDA Not Found. Using CPU instead.")
            self.ui.comboBoxDevice.setCurrentText("CPU")

        device_name = self.runner.get_device_name()

        device = self.ui.comboBoxDevice.currentText()
        self.ui.labelModelInfo.setText(f"Loaded Model: {file_path} | Device: {device} -> {device_name}")
        self.model_loaded = True


    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        self.current_image_path = file_path
        self.current_image = cv2.imread(file_path)  # BGR image for inference

        pixmap = QPixmap(file_path)

        self.scene_orig.clear()
        self.scene_res.clear()
        # Add pixmap
        self.scene_orig.addPixmap(pixmap)

        # Set scene rect to pixmap size
        self.scene_orig.setSceneRect(pixmap.rect())

        # Reset any previous zoom/transform
        self.ui.graphicsView.resetTransform()

        # Hide scrollbars
        self.ui.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Center alignment
        self.ui.graphicsView.setAlignment(Qt.AlignCenter)

        # Scale to fit view while keeping aspect ratio
        self.ui.graphicsView.fitInView(self.scene_orig.sceneRect(), Qt.KeepAspectRatio)


    def cv_to_pixmap(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w

        # BGR -> RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        qimg = QImage(
            rgb_image.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        return QPixmap.fromImage(qimg)


    def draw_mask_overlay(self, image, mask, color, alpha=0.4):
        """
        image : BGR uint8 image
        mask  : binary mask (0/1 or 0/255)
        color : (B, G, R)
        """

        overlay = image.copy()

        mask_bool = mask.astype(bool)

        overlay[mask_bool] = color

        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def visualize(self, detections):
        # Start from the original image
        res_image = self.current_image.copy()

        # Draw boxes only if detections exist
        if detections:
            for det in detections:
                x0, y0, x1, y1 = map(int, det.bbox)
                color = tuple(np.random.randint(0, 256, size=3).tolist())

                cv2.rectangle(res_image, (x0, y0), (x1, y1), color, 4)
                label = f"Class_{det.id}: {int(det.conf*100)}%"
                cv2.putText(
                    res_image,
                    label,
                    (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    color,
                    2,
                    lineType=cv2.LINE_AA
                )


                if self.ui.comboBoxTask.currentText() == "Segment":
                    res_image = self.draw_mask_overlay(res_image, det.mask, color)

        else:
            label = "No Detections"
            cv2.putText(
                res_image,
                label,
                (10 , 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 0, 0),
                4,
                lineType=cv2.LINE_AA
            )


        pixmap = self.cv_to_pixmap(res_image)
        self.scene_res.clear()

        self.scene_res.addPixmap(pixmap)

        # Set scene rect to pixmap size
        self.scene_res.setSceneRect(pixmap.rect())

        # Reset any previous zoom/transform
        self.ui.graphicsViewRes.resetTransform()

        # Center alignment
        self.ui.graphicsViewRes.setAlignment(Qt.AlignCenter)

        # Scale to fit view while keeping aspect ratio
        self.ui.graphicsViewRes.fitInView(self.scene_res.sceneRect(), Qt.KeepAspectRatio)


    def run_inference(self):
        if self.current_image is None:
            QMessageBox.warning(
                self,                 # parent widget
                "No image selected",    # dialog title
                "Please open an image first before running inference."  # message
            )
            return
        if not self.model_loaded:
            QMessageBox.warning(
                self,                 # parent widget
                "Model Not Loaded",    # dialog title
                "Please load a model first before running inference."  # message
            )
            return

        self.runner.conf_thresh = self.ui.horizontalSliderConf.value() / 100.0
        self.runner.nms_thresh = self.ui.horizontalSliderNMS.value() / 100.0
        self.runner.task = self.ui.comboBoxTask.currentText()

        start = time.time()
        results = self.runner.run(self.current_image)
        end = time.time()
        elapsed_ms = (end - start) * 1000  # convert seconds to milliseconds

        self.ui.labelProcess.setText(
            f"Process Time: {elapsed_ms:.2f} ms"
        )

        self.visualize(results)
        print("Running inference on image of shape:", self.current_image.shape)





