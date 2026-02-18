from .utils import process_mask, preprocess
from ..inference.utils import Detection
from ..inference.onnx_runner import ONNXRunner
import numpy as np
import cv2

class MaskRCNNRunner(ONNXRunner):
    def __init__(self):
        self.nms_thresh = 0.5

    def postprocess(self, outputs, params):
        """
        Model-specific postprocessing: convert outputs to standardized dict
        for YOLOv8. Other YOLO model    def __init__(self):
        """
        if outputs is None:
            return {"bboxes": [], "classes": [], "masks": []}

        ratio, dw, dh, width, height = params["ratio"], params["dw"], params["dh"], params["width"], params["height"]

        preds = outputs[0]

        preds = preds.transpose(0, 2, 1)
        preds = np.array(preds, dtype=np.float32)  # ensure numeric
        pred = preds[0]

        if self.task == "Segment":
            proto = outputs[1][0]
            last_idx = pred.shape[1]
            scores = pred[:, 4:last_idx - 32]
            mask_coeffs = pred[:, last_idx - 32:]
        else:
            scores = pred[:, 4:]

        boxes = pred[:, :4]
        classes = np.argmax(scores, axis=1)
        confs = scores[np.arange(scores.shape[0]), classes]

        mask = confs >= 0.1

        boxes = boxes[mask]
        classes = classes[mask]
        confs = confs[mask]

        if self.task == "Segment":
            mask_coeffs = mask_coeffs[mask]

        x = boxes[:, 0] - dw
        y = boxes[:, 1] - dh
        w = boxes[:, 2]
        h = boxes[:, 3]

        # optional clamp
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height)

        boxes[:, 0] = np.clip((x - 0.5 * w) * ratio, 0, width);
        boxes[:, 1] = np.clip((y - 0.5 * h) * ratio, 0, height);
        boxes[:, 2] = np.clip((x + 0.5 * w) * ratio, 0, width);
        boxes[:, 3] = np.clip((y + 0.5 * h) * ratio, 0, height);

        bboxes = boxes.tolist()
        classes = classes.tolist()
        confs = confs.tolist()


        indices = cv2.dnn.NMSBoxesBatched(bboxes, confs, classes, self.conf_thresh, self.nms_thresh)

        if len(indices) > 0:
            indices = np.array(indices).flatten()
        else:
            indices = []

        final_boxes = [bboxes[i] for i in indices]
        final_classes = [classes[i] for i in indices]
        final_confs = [confs[i] for i in indices]

        if self.task == "Segment":
            final_mask_coeffs = [mask_coeffs[i] for i in indices]
            final_masks = []

            for mask_coeff, bbox in zip(final_mask_coeffs, final_boxes):
                mask = process_mask(mask_coeff, proto, bbox, params, self.input_size)
                final_masks.append(mask)

            objects = [
                Detection(
                    mask=final_masks[i],
                    bbox=final_boxes[i],
                    conf=final_confs[i],
                    id=final_classes[i]
                )
                for i in range(len(final_boxes))
            ]
        else:
            objects = [
                Detection(
                    mask=None,
                    bbox=final_boxes[i],
                    conf=final_confs[i],
                    id=final_classes[i]
                )
                for i in range(len(final_boxes))
            ]

        return objects

    def run(self, image):
        if self.session is None:
            raise RuntimeError("Model not loaded")

        images = []
        images.append(image)
        meta = self.session.get_modelmeta()
        print(meta.custom_metadata_map)

        input_meta = self.session.get_inputs()[0]  # the first input tensor
        print(input_meta.shape[2])
        self.input_size = (input_meta.shape[2] or 640, input_meta.shape[3] or 640)

        input_tensor, params = preprocess(image, self.input_size)  # shared preprocessing

        outputs = self.session.run(None, {input_meta.name: input_tensor})

        return self.postprocess(outputs, params)


