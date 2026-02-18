from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Detection:
    mask: Optional[np.ndarray]
    bbox: tuple   # (x0, y0, x1, y1)
    conf: float
    id: int
