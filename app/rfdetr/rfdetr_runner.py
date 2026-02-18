# This Python file uses the following encoding: utf-8
from ..inference.utils import Detection
from ..inference.onnx_runner import ONNXRunner
import numpy as np
import cv2

class RFDETRRunner(ONNXRunner):

    def preprocess(self, image):
        h, w = image.shape[:2]
        input_size = self.input_size
        r = min(input_size[1] / h, input_size[0] / w)
        padw = round(w * r)
        padh = round(h * r)

        resized = cv2.resize(image, (padw, padh))

        dw = (input_size[0] - padw) / 2
        dh = (input_size[1] - padh) / 2

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114,114,114)
        )

        means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        stds  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        padded = (padded - means) / stds

        blob = cv2.dnn.blobFromImage(
            padded, scalefactor=1/255.0, size=input_size, swapRB=True
        )

        params = {
            "ratio": 1 / r,
            "dw": dw,
            "dh": dh,
            "width": w,
            "height": h
        }

        return blob, params

    def postprocess(self, outputs, params):
        """
        Model-specific postprocessing: convert outputs to standardized dict
        for YOLOv8. Other YOLO models would have their own _postprocess.
        """
        if outputs is None:
            return {"bboxes": [], "classes": [], "masks": []}


        ratio, dw, dh, width, height = params["ratio"], params["dw"], params["dh"], params["width"], params["height"]
        input_size = self.input_size
        dets = outputs[0]
        labels = outputs[1]

        if self.task == "Segment":
            masks = outputs[2][0]

        boxes = dets[0]
        scores = labels[0]
        scores = 1 / (1 + np.exp(-scores))  # shape (300, 91)

        classes = np.argmax(scores, axis=1)
        confs = scores[np.arange(scores.shape[0]), classes]

        mask = confs >= self.conf_thresh

        boxes = boxes[mask]
        classes = classes[mask]
        confs = confs[mask]

        if self.task == "Segment":
            masks = masks[mask]

        x = boxes[:, 0] * input_size[0]
        y = boxes[:, 1] * input_size[1]
        w = boxes[:, 2] * input_size[0]
        h = boxes[:, 3] * input_size[1]

        # optional clamp
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height)
        x0 = x - 0.5 * w;
        y0 = y - 0.5 * h;
        x1 = x + 0.5 * w;
        y1 = y + 0.5 * h;

        x0 = x0 - dw;
        y0 = y0 - dh;
        x1 = x1 - dw;
        y1 = y1 - dh;

        boxes[:, 0] = np.clip(x0 * ratio, 0, width);
        boxes[:, 1] = np.clip(y0 * ratio, 0, height);
        boxes[:, 2] = np.clip(x1 * ratio, 0, width);
        boxes[:, 3] = np.clip(y1 * ratio, 0, height);

        bboxes = boxes.tolist()
        classes = classes.astype(int).tolist()
        confs = confs.tolist()
        final_boxes = bboxes
        final_classes = classes
        final_confs = confs

        if self.task == "Segment":
            final_masks = []
            for mask, bbox in zip(masks, final_boxes):
                mask = self.process_mask(mask, bbox, params)
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


    def process_mask(self, mask, bbox, pparam):

        dw = pparam['dw']
        dh = pparam['dh']
        orig_w = pparam['width']
        orig_h = pparam['height']
        input_h, input_w = self.input_size

        mask_input = cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

        x0_pad, y0_pad = int(dw), int(dh)
        x1_pad, y1_pad = int(input_w - dw), int(input_h - dh)
        mask_unpad = mask_input[y0_pad:y1_pad, x0_pad:x1_pad]
        mask_orig = cv2.resize(mask_unpad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        x0, y0, x1, y1 = map(int, bbox)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, orig_w), min(y1, orig_h)
        mask_crop = mask_orig[y0:y1, x0:x1]

        full_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
        full_mask[y0:y1, x0:x1] = mask_crop

        binary_mask = (full_mask > 0.5).astype(np.uint8)

        return binary_mask

    def run(self, image):
        if self.session is None:
            raise RuntimeError("Model not loaded")

        images = []
        images.append(image)
        meta = self.session.get_modelmeta()

        input_meta = self.session.get_inputs()[0]  # the first input tensor
        self.input_size = (input_meta.shape[2] or 384, input_meta.shape[3] or 384)

        input_tensor, params = self.preprocess(image)  # shared preprocessing

        outputs = self.session.run(None, {input_meta.name: input_tensor})

        return self.postprocess(outputs, params)

