from .utils import process_mask, preprocess
from ..inference.utils import Detection
from ..inference.onnx_runner import ONNXRunner
import numpy as np

class YOLO26ONNXRunner(ONNXRunner):

    def postprocess(self, outputs, params):
        """
        Model-specific postprocessing: convert outputs to standardized dict
        for YOLOv8. Other YOLO models would have their own _postprocess.
        """
        if outputs is None:
            return {"bboxes": [], "classes": [], "masks": []}


        ratio, dw, dh, width, height = params["ratio"], params["dw"], params["dh"], params["width"], params["height"]

        preds = outputs[0]  # assuming (1, N, 85)
        pred = preds[0]

        if self.task == "Segment":
            proto = outputs[1][0]
            last_idx = pred.shape[1]
            mask_coeffs = pred[:, last_idx - 32:]


        boxes = pred[:, :4]
        confs = pred[:, 4]
        classes = pred[:, 5]

        mask = confs >= self.conf_thresh

        boxes = boxes[mask]
        classes = classes[mask]
        confs = confs[mask]

        if self.task == "Segment":
            mask_coeffs = mask_coeffs[mask]

        x0 = boxes[:, 0] - dw;
        y0 = boxes[:, 1] - dh;
        x1 = boxes[:, 2] - dw;
        y1 = boxes[:, 3] - dh;

        # optional clamp
        boxes[:, 0] = np.clip(x0 * ratio, 0, width)
        boxes[:, 1] = np.clip(y0 * ratio, 0, height)
        boxes[:, 2] = np.clip(x1 * ratio, 0, width)
        boxes[:, 3] = np.clip(y1 * ratio, 0, height)


        bboxes = boxes.tolist()
        classes = classes.astype(int).tolist()
        confs = confs.tolist()

        final_boxes = bboxes
        final_classes = classes
        final_confs = confs

        if self.task == "Segment":
            final_mask_coeffs = mask_coeffs
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


