import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

def extract_5p_batch(lms):
    lm_idx = torch.tensor([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5ps = torch.stack([lms[:, lm_idx[0], :], torch.mean(lms[:, lm_idx[[1, 2]], :], 1), torch.mean(
        lms[:, lm_idx[[3, 4]], :], 1), lms[:, lm_idx[5], :], lms[:, lm_idx[6], :]], dim=1)
    lm5ps = lm5ps[:, [1, 2, 0, 3, 4], :]
    return lm5ps


def POS_batch(xp, x):
    batch_size, _, npts = xp.shape

    A = torch.zeros([batch_size, 2*npts, 8]).to(xp.device)

    A[:, 0:2*npts-1:2, 0:3] = x.permute(1, 0)
    A[:, 0:2*npts-1:2, 3] = 1

    A[:, 1:2*npts:2, 4:7] = x.permute(1, 0)
    A[:, 1:2*npts:2, 7] = 1

    b = xp.permute(0, 2, 1).view(batch_size, 2*npts, 1)

    k, _, _, _ = torch.linalg.lstsq(A, b)

    R1 = k[:, 0:3]
    R2 = k[:, 4:7]
    sTx = k[:, 3]
    sTy = k[:, 7]
    s = (torch.linalg.norm(R1, dim=1) + torch.linalg.norm(R2, dim=1))/2
    t = torch.stack([sTx, sTy], dim=1)

    return t, s


def resize_n_crop_img_batch(images, landmarks, t, s, target_size=224, masks=None):
    device = images.device
    images = images.float()
    _, h0, w0, _ = images.shape
    w = (w0*s).int()
    h = (h0*s).int()
    left = (w/2 - target_size/2 + (t[:, 0] - w0/2)*s).int()
    right = left + target_size
    up = (h/2 - target_size/2 + (h0/2 - t[:, 1])*s).int()
    below = up + target_size

    new_images = torch.zeros([images.shape[0], target_size, target_size, 3])
    new_masks = torch.zeros([masks.shape[0], target_size, target_size, masks.shape[3]]) if masks is not None else None
    for idx, image in enumerate(images):
        image = Image.fromarray(image.cpu().numpy().astype(np.uint8))
        image = image.resize((w[idx].item(), h[idx].item()), Image.BICUBIC)
        image = image.crop((left[idx].item(), up[idx].item(), right[idx].item(), below[idx].item()))
        image = np.array(image)
        new_images[idx] = torch.from_numpy(image)
        if masks is not None:
            mask = masks[idx]
            mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
            mask = mask.resize((w[idx].item(), h[idx].item()), Image.BICUBIC)
            mask = mask.crop((left[idx].item(), up[idx].item(), right[idx].item(), below[idx].item()))
            mask = np.array(mask)
            new_masks[idx] = mask
    new_landmarks = torch.stack([
        landmarks[:, :, 0] - t[:, 0] + w0/2,
        landmarks[:, :, 1] - t[:, 1] + h0/2,
    ], dim=-1) * s.view(-1, 1, 1)
    new_landmarks = new_landmarks - torch.stack([w/2 - target_size/2, h/2 - target_size/2], dim=-1)
    new_images = new_images.to(device)
    new_masks = new_masks.to(device) if masks is not None else None
    return new_images, new_landmarks, new_masks


def align_img_batch(images, landmarks, landmark_3d, masks=None, target_size=224, rescale_factor=102.):
    _, h0, w0, _ = images.shape
    if landmarks.shape[1] != 5:
        landmarks_5p = extract_5p_batch(landmarks)
    else:
        landmarks_5p = landmarks
    
    t, s = POS_batch(landmarks_5p.permute(0, 2, 1), landmark_3d.permute(1, 0))
    s = rescale_factor / s

    new_images, new_landmarks, new_masks = resize_n_crop_img_batch(images, landmarks, t, s, target_size=target_size, masks=masks)
    new_images = new_images.byte()
    trans_params = torch.zeros([images.shape[0], 5]).to(images.device)
    trans_params[:, 0] = w0
    trans_params[:, 1] = h0
    trans_params[:, 2] = s.view(-1)
    trans_params[:, 3:] = t.view(-1, 2)
    return trans_params, new_images, new_landmarks, new_masks