import onnxruntime as ort
from .base_runner import ModelRunner
import subprocess
import platform

class ONNXRunner(ModelRunner):
    def __init__(self):
        self.session = None

    def load_model(self, path, use_cuda=False):

        self.cuda_warning = False
        cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
        if use_cuda and not cuda_available:
            self.cuda_warning = True  # just set a flag
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda and cuda_available else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, providers=providers)


    def get_device_name(self):
        if self.session is None:
            return "No model"

        provider = self.session.get_providers()[0]

        # ---------- GPU ----------
        if provider == "CUDAExecutionProvider":
            try:
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    encoding="utf-8"
                )
                gpu_name = result.strip().split("\n")[0]
                return f"{gpu_name}"
            except Exception:
                return "GPU"

        # ---------- CPU ----------
        if provider == "CPUExecutionProvider":

            # Linux (best method)
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            return f"{cpu_name}"
            except Exception:
                pass

            # fallback
            cpu_name = platform.processor()
            return "CPU"

        return provider
