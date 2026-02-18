from abc import ABC, abstractmethod

class ModelRunner(ABC):
    def __init__(self):
        self.session = None
        self.input_size = (640, 640)
        self.conf_thresh = 0.1
        self.task = None
        self.cuda_warning = False  # track if CUDA was unavailable

    @abstractmethod
    def load_model(self, path, use_cuda=False):
        pass

    @abstractmethod
    def postprocess(self, outputs, params):
        pass

    @abstractmethod
    def run(self, image):
        """
        Run inference on an image (numpy array)
        Return dictionary: {"bboxes": [], "masks": [], "classes": []}
        """
        pass
