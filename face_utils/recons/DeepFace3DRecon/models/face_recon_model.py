import numpy as np
import torch
from .networks import define_net_recon
from .bfm_model import ParametricFaceModel
from ..utils.bfm import split_coeff

from scipy.io import savemat


class FaceReconModel(torch.nn.Module):
    def __init__(self, net_recon='resnet50', resume_path='checkpoints/init_model/resnet50-0676ba61.pth', bfm_folder='BFM/', bfm_model='BFM_model_front.mat'):
        """Initialize this model class.
        """
        super(FaceReconModel, self).__init__()
        self.net_recon = net_recon
        self.resume_path = resume_path
        self.bfm_folder = bfm_folder
        self.bfm_model = bfm_model

        self.net_recon = define_net_recon(
            net_recon=self.net_recon, use_last_fc=False,
        )

        self.net_recon.load_state_dict(torch.load(self.resume_path, map_location='cpu')['net_recon'])

    @torch.no_grad()
    def extract_coeffs(self, x):
        output_coeff = self.net_recon(x)
        pred_coeffs_dict = split_coeff(output_coeff)
        pred_coeffs = {key: value.cpu().numpy() for key, value in pred_coeffs_dict.items()}
        return pred_coeffs_dict
    
    def set_render(self, focal=1015., center=112., camera_d=10., z_near=5., z_far=15.):
        from util.nvdiffrast import MeshRenderer

        self.focal = focal
        self.center = center
        self.camera_d = camera_d
        self.z_near = z_near
        self.z_far = z_far

        fov = 2 * np.arctan(self.center / self.focal) * 180 / np.pi
        self.facemodel = ParametricFaceModel(
            bfm_folder=self.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model
        )

        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=self.z_near, zfar=self.z_far, rasterize_size=int(2 * self.center)
        )

    def get_mesh(self, name):
        import trimesh
        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.facemodel.face_buf.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
        return mesh
