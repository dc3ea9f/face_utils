import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from face_alignment import FaceAlignment, LandmarksType
from .utils.bfm import load_lm3d
from .utils.align_batch import align_img_batch


class Preprocessor(nn.Module):
    def __init__(self, bfm_folder, **kwargs):
        super(Preprocessor, self).__init__()
        self.lm3d_std = load_lm3d(bfm_folder)
        self.device = 'cpu'
        self.kwargs = kwargs
        self.fa = FaceAlignment(LandmarksType._2D, device=self.device, **kwargs)
    
    def to(self, device):
        self.device = device
        self.fa = FaceAlignment(LandmarksType._2D, device=self.device, **self.kwargs)
        return self
    
    def cuda(self):
        self.device = 'cuda'
        self.fa = FaceAlignment(LandmarksType._2D, device=self.device, **self.kwargs)
        return self
    
    def cpu(self):
        self.device = 'cpu'
        self.fa = FaceAlignment(LandmarksType._2D, device=self.device, **self.kwargs)
        return self

    def extract_bboxs(self, x):
        """
        Extract bounding boxes from input frames.

        Args:
            x (torch.tensor): Input frames in (N, H, W, C)
        Returns:
            list[np.array]: Bounding boxes in (4, )
        """
        x = x.permute(0, 3, 1, 2).float().to(self.device)
        return self.fa.face_detector.detect_from_batch(x)
    
    def extract_keypoints(self, x):
        """
        Extract keypoints from input frames.

        Args:
            x (torch.tensor): Input frames in (N, H, W, C)
        Returns:
            list[np.array]: Keypoints in (68, 2)
        """
        x = x.permute(0, 3, 1, 2).float().to(self.device)
        return self.fa.get_landmarks_from_batch(x)

    def align_and_recrop(self, x):
        """
        Perform face alignment and recrop the input frames.

        Args:
            x (torch.tensor): Input frames in (N, H, W, C)
        Returns:
            torch.tensor: Transform parameters in (N, 5)
            torch.tensor: Aligned and recropped frames in (N, H, W, C)
            torch.tensor: Aligned and recropped landmarks in (N, 68, 2)
            torch.tensor: Aligned and recropped masks in (N, H, W)
        """
        keypoints = self.extract_keypoints(x)
        keypoints = np.stack([e if e is not None else -1 * np.ones((68, 2)) for e in keypoints])
        trans_params, new_images, new_landmarks, new_masks = align_img_batch(
            x,
            torch.from_numpy(keypoints).to(self.device),
            torch.from_numpy(self.lm3d_std).to(self.device),
        )
        return trans_params, new_images, new_landmarks, new_masks
    
    def prepare_input(self, x):
        """
        Convert input frames to the format required by the network. The device is not changed.

        Args:
            x (torch.tensor): Input frames in (N, H, W, C)
        Returns:
            torch.tensor: Normalized input frames in (N, C, H, W)
        """
        x = x.permute(0, 3, 1, 2).float()
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        x /= 255.
        return x
