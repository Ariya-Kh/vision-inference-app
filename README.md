<h1>Vision Inference App</h1>

<p>
Desktop application for running deep learning computer vision models using
<strong>ONNX Runtime</strong> with a graphical interface built in
<strong>PySide6</strong>.
</p>

<hr>

<h2>Overview</h2>

<p>This application allows users to:</p>

<ul>
    <li>Load ONNX models</li>
    <li>Run detection or segmentation inference</li>
    <li>Visualize bounding boxes and masks</li>
    <li>Compare original and processed images</li>
    <li>Adjust confidence and NMS thresholds</li>
    <li>Run inference on CPU or CUDA (if available)</li>
</ul>

<hr>

<h2>Features</h2>

<ul>
    <li>Detection and instance segmentation support</li>
    <li>Modular model runner design</li>
    <li>ONNX Runtime backend</li>
    <li>CPU and GPU execution</li>
    <li>Mask overlay and contour visualization</li>
    <li>Real-time parameter adjustment</li>
    <li>Device and inference time display</li>
    <li>Two-view visualization (original + result)</li>
</ul>

<hr>

<h2>Supported Models</h2>

<h3>Detection</h3>
<ul>
    <li>YOLOv3</li>
    <li>YOLOv5</li>
    <li>YOLOv6</li>
    <li>YOLOv8</li>
    <li>YOLOv9</li>
    <li>YOLOv10</li>
    <li>YOLO11</li>
    <li>YOLO12</li>
    <li>YOLO26</li>
    <li>RF-DETR</li>
</ul>

<h3>Segmentation</h3>
<ul>
    <li>YOLOv8-Seg</li>
    <li>YOLOv9-Seg</li>
    <li>YOLO11-Seg</li>
    <li>YOLO12-Seg</li>
    <li>YOLO26-Seg</li>
    <li>RF-DETR-Seg</li>
</ul>

<hr>

<h2>Screenshots</h2>

<img width="1850" height="1053" alt="Screenshot from 2026-02-18 13-38-50" src="https://github.com/user-attachments/assets/1b589ce9-725f-417c-a74e-304f15c637e0" />
<img width="1850" height="1053" alt="Screenshot from 2026-02-18 13-42-37" src="https://github.com/user-attachments/assets/915e4956-5a2a-473a-8db1-15f77c8fcace" />

<hr>

<h2>Installation</h2>

<h3>Requirements</h3>

<ul>
    <li>Python</li>
    <li>OpenCV</li>
    <li>NumPy</li>
    <li>ONNX Runtime</li>
    <li>PySide6</li>
</ul>

<p>Install dependencies:</p>

<pre><code>pip install -r requirements.txt</code></pre>

<hr>

<h2>Model Loading Workflow</h2>

<ol>
    <li>Select model type from dropdown</li>
    <li>Load ONNX model</li>
    <li>Open image</li>
    <li>Run inference</li>
</ol>


<hr>

<h2>Unified Detection Structure</h2>

<pre><code>@dataclass
class Detection:
    mask: Optional[np.ndarray]
    bbox: tuple   # (x0, y0, x1, y1)
    conf: float
    id: int
</code></pre>

<p>
All model outputs are converted into this structure, keeping the UI independent
from model implementation.
</p>

<hr>

<h2>CUDA Support</h2>

<p>
CUDA is used automatically when available through ONNX Runtime GPU.
If CUDA is unavailable, inference falls back to CPU.
</p>

<hr>


