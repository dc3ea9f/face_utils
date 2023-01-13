"""This script is to load 3D face model for Deep3DFaceRecon_pytorch
"""

import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat
from array import array
import os.path as osp
import torch


# load expression basis
def LoadExpBasis(bfm_folder='BFM'):
    n_vertex = 53215
    Expbin = open(osp.join(bfm_folder, 'Exp_Pca.bin'), 'rb')
    exp_dim = array('i')
    exp_dim.fromfile(Expbin, 1)
    expMU = array('f')
    expPC = array('f')
    expMU.fromfile(Expbin, 3*n_vertex)
    expPC.fromfile(Expbin, 3*exp_dim[0]*n_vertex)
    Expbin.close()

    expPC = np.array(expPC)
    expPC = np.reshape(expPC, [exp_dim[0], -1])
    expPC = np.transpose(expPC)

    expEV = np.loadtxt(osp.join(bfm_folder, 'std_exp.txt'))

    return expPC, expEV


# transfer original BFM09 to our face model
def transferBFM09(bfm_folder='BFM'):
    print('Transfer BFM09 to BFM_model_front......')
    original_BFM = loadmat(osp.join(bfm_folder, '01_MorphableModel.mat'))
    shapePC = original_BFM['shapePC']  # shape basis
    shapeEV = original_BFM['shapeEV']  # corresponding eigen value
    shapeMU = original_BFM['shapeMU']  # mean face
    texPC = original_BFM['texPC']  # texture basis
    texEV = original_BFM['texEV']  # eigen value
    texMU = original_BFM['texMU']  # mean texture

    expPC, expEV = LoadExpBasis()

    # transfer BFM09 to our face model

    idBase = shapePC*np.reshape(shapeEV, [-1, 199])
    idBase = idBase/1e5  # unify the scale to decimeter
    idBase = idBase[:, :80]  # use only first 80 basis

    exBase = expPC*np.reshape(expEV, [-1, 79])
    exBase = exBase/1e5  # unify the scale to decimeter
    exBase = exBase[:, :64]  # use only first 64 basis

    texBase = texPC*np.reshape(texEV, [-1, 199])
    texBase = texBase[:, :80]  # use only first 80 basis

    # our face model is cropped along face landmarks and contains only 35709 vertex.
    # original BFM09 contains 53490 vertex, and expression basis provided by Guo et al. contains 53215 vertex.
    # thus we select corresponding vertex to get our face model.

    index_exp = loadmat(osp.join(bfm_folder, 'BFM_front_idx.mat'))
    index_exp = index_exp['idx'].astype(np.int32) - 1  # starts from 0 (to 53215)

    index_shape = loadmat(osp.join(bfm_folder, 'BFM_exp_idx.mat'))
    index_shape = index_shape['trimIndex'].astype(
        np.int32) - 1  # starts from 0 (to 53490)
    index_shape = index_shape[index_exp]

    idBase = np.reshape(idBase, [-1, 3, 80])
    idBase = idBase[index_shape, :, :]
    idBase = np.reshape(idBase, [-1, 80])

    texBase = np.reshape(texBase, [-1, 3, 80])
    texBase = texBase[index_shape, :, :]
    texBase = np.reshape(texBase, [-1, 80])

    exBase = np.reshape(exBase, [-1, 3, 64])
    exBase = exBase[index_exp, :, :]
    exBase = np.reshape(exBase, [-1, 64])

    meanshape = np.reshape(shapeMU, [-1, 3])/1e5
    meanshape = meanshape[index_shape, :]
    meanshape = np.reshape(meanshape, [1, -1])

    meantex = np.reshape(texMU, [-1, 3])
    meantex = meantex[index_shape, :]
    meantex = np.reshape(meantex, [1, -1])

    # other info contains triangles, region used for computing photometric loss,
    # region used for skin texture regularization, and 68 landmarks index etc.
    other_info = loadmat(osp.join(bfm_folder, 'facemodel_info.mat'))
    frontmask2_idx = other_info['frontmask2_idx']
    skinmask = other_info['skinmask']
    keypoints = other_info['keypoints']
    point_buf = other_info['point_buf']
    tri = other_info['tri']
    tri_mask2 = other_info['tri_mask2']

    # save our face model
    savemat(osp.join(bfm_folder, 'BFM_model_front.mat'), {'meanshape': meanshape, 'meantex': meantex, 'idBase': idBase, 'exBase': exBase, 'texBase': texBase,
            'tri': tri, 'point_buf': point_buf, 'tri_mask2': tri_mask2, 'keypoints': keypoints, 'frontmask2_idx': frontmask2_idx, 'skinmask': skinmask})


# load landmarks for standard face, which is used for image preprocessing
def load_lm3d(bfm_folder):

    Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


def split_coeff(coeffs):
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }

def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

def extract_5p_batch(lms):
    lm_idx = torch.tensor([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5ps = torch.stack([lms[:, lm_idx[0], :], torch.mean(lms[:, lm_idx[[1, 2]], :], 1), torch.mean(
        lms[:, lm_idx[[3, 4]], :], 1), lms[:, lm_idx[5], :], lms[:, lm_idx[6], :]], dim=1)
    lm5ps = lm5ps[:, [1, 2, 0, 3, 4], :]
    return lm5ps

# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


def POS_batch(xp, x):
    batch_size, _, npts = xp.shape

    A = torch.zeros([batch_size, 2*npts, 8])

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

def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size
    
    img = img.resize((w, h), resample=Image.BICUBIC)
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.BICUBIC)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img, lm, mask

def resize_n_crop_img_batch(images, landmarks, t, s, target_size=224, masks=None):
    device = images.device
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
        image = Image.fromarray(image.cpu().numpy())
        image = image.resize((w[idx], h[idx]), resample=Image.BICUBIC)
        image = image.crop((left[idx].item(), up[idx].item(), right[idx].item(), below[idx].item()))
        new_images[idx] = torch.from_numpy(np.array(image))
        if masks is not None:
            mask = masks[idx]
            mask = Image.fromarray(mask.cpu().numpy())
            mask = mask.resize((w[idx], h[idx]), resample=Image.BICUBIC)
            new_masks[idx] = torch.from_numpy(np.array(mask))
    new_landmarks = torch.stack([
        landmarks[:, :, 0] - t[:, 0] + w0/2,
        landmarks[:, :, 1] - t[:, 1] + h0/2,
    ], dim=-1) * s.view(-1, 1, 1)
    new_landmarks = new_landmarks - torch.stack([w/2 - target_size/2, h/2 - target_size/2], dim=-1)
    new_images = new_images.to(device)
    new_masks = new_masks.to(device) if masks is not None else None
    return new_images, new_landmarks, new_masks


def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor/s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0].item(), t[1].item()])

    return trans_params, img_new, lm_new, mask_new


def align_img_batch(images, landmarks, landmark_3d, masks=None, target_size=224, rescale_factor=102.):
    _, h0, w0, _ = images.shape
    if landmarks.shape[1] != 5:
        landmarks_5p = extract_5p_batch(landmarks)
    else:
        landmarks_5p = landmarks
    
    t, s = POS_batch(landmarks_5p.permute(0, 2, 1), landmark_3d.permute(1, 0))
    s = rescale_factor / s

    new_images, new_landmarks, new_masks = resize_n_crop_img_batch(images, landmarks, t, s, target_size=target_size, masks=masks)
    trans_params = torch.zeros([images.shape[0], 5]).to(images.device)
    trans_params[:, 0] = w0
    trans_params[:, 1] = h0
    trans_params[:, 2] = s.view(-1)
    trans_params[:, 3:] = t.view(-1, 2)
    return trans_params, new_images, new_landmarks, new_masks