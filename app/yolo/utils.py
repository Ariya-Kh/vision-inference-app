import cv2
import numpy as np

def preprocess(image, input_size=(640, 640)):
    h, w = image.shape[:2]

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


def process_mask(mask_coeff, proto, bbox, pparam, input_size):

    dw = pparam['dw']
    dh = pparam['dh']
    orig_w = pparam['width']
    orig_h = pparam['height']
    input_h, input_w = input_size

    c, mh, mw = proto.shape
    mask = mask_coeff @ proto.reshape(c, -1)
    mask = 1.0 / (1.0 + np.exp(-mask))
    mask = mask.reshape(mh, mw)

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

