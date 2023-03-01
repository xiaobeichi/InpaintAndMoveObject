"""Camera pose and ray generation utility functions."""

import enum
import types
import numpy as np

def intrinsic_matrix(fx: float,
                     fy: float,
                     cx: float,
                     cy: float,
                     xnp: types.ModuleType = np):
  """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
  return xnp.array([
      [fx, 0, cx],
      [0, fy, cy],
      [0, 0, 1.],
  ])


class ProjectionType(enum.Enum):
  """Camera projection type (standard perspective pinhole or fisheye model)."""
  PERSPECTIVE = 'perspective'
  FISHEYE = 'fisheye'