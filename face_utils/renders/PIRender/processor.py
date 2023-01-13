import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from einops import rearrange


def obtain_seq_index(frame_index, semantic_radius, num_frames):
    seq = list(range(frame_index - semantic_radius, frame_index + semantic_radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq

class Preprocessor(nn.Module):
    def __init__(self, resolution=256, semantic_radius=13):
        super(Preprocessor, self).__init__()
        self.semantic_radius = semantic_radius
        self.resolution = resolution

    def prepare_image(self, img):
        """
        Convert input RGB image to PIRender input

        Args:
            img (np.ndarray): RGB image with shape (H, W, C)
        Returns:
            img (torch.Tensor): PIRender input with shape (1, C, H, W)
        """
        H, W, C = img.shape
        if H != self.resolution or W != self.resolution:
            img = Image.fromarray(img)
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
            img = np.array(img)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.
        img = (img - 0.5) / 0.5
        return img

    def prepare_coeffs(self, coeffs):
        """
        Prepare coeffs to PIRender input

        Args:
            coeffs (dict): coeffs from DeepFace3DReconModel, if 'crop' is not in coeffs, it will be generated
        Returns:
            semantics (torch.Tensor): PIRender input with shape (T, C, self.semantic_radius * 2 + 1)
        """
        result = []
        exp_coeff = coeffs['exp']
        angle_coeff = coeffs['angle']
        trans_coeff = coeffs['trans']
        if 'crop' in coeffs:
            crop_coeff = coeffs['crop']
        else:
            crop_coeff = np.concatenate([
                np.ones((exp_coeff.shape[0], 1)).astype(float),
                np.ones((exp_coeff.shape[0], 2)).astype(float) * 128,
            ], axis=1)
        semantic = np.concatenate([exp_coeff, angle_coeff, trans_coeff, crop_coeff], axis=1)
        num_frames = semantic.shape[0]

        for frame_index in range(num_frames):
            index = obtain_seq_index(frame_index, self.semantic_radius, num_frames)
            curr_semantic = semantic[index, ...]
            result.append(curr_semantic)
        result = torch.from_numpy(np.stack(result, axis=0)).permute(0, 2, 1)
        return result


class PostProcessor(nn.Module):
    def __init__(self):
        super(PostProcessor, self).__init__()
    
    def recover_video(self, x):
        """
        Convert PIRender output to video

        Args:
            x: (T, B, C, H, W) tensor
        Returns:
            torch.tensor: (T * B, H, W, C) in uint8
        """
        x = torch.clamp(x, -1, 1)
        x = rearrange(x, 't b c h w -> (t b) h w c')
        x = (x + 1) / 2.0 * 255
        x = x.byte()
        return x