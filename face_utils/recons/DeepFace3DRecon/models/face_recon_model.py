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
        return pred_coeffs_dict
    
    def init_render(self, focal=1015., center=112., camera_d=10., z_near=5., z_far=15.):
        global trimesh
        import trimesh
        global MeshRenderer
        from .nvdiffrast import MeshRenderer

        self.focal = focal
        self.center = center
        self.camera_d = camera_d
        self.z_near = z_near
        self.z_far = z_far

        fov = 2 * np.arctan(self.center / self.focal) * 180 / np.pi
        self.facemodel = ParametricFaceModel(
            bfm_folder=self.bfm_folder, camera_distance=self.camera_d, focal=self.focal, center=self.center,
            is_train=False, default_name=self.bfm_model
        ).to(next(self.parameters()).device)

        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=self.z_near, zfar=self.z_far, rasterize_size=int(2 * self.center)
        ).to(next(self.parameters()).device)

    @torch.no_grad()
    def get_meshs(self, coeffs):
        """
        Get the mesh from the given coefficients.

        Args:
            coeffs (torch.Tensor): The coefficients of the mesh, can be generated from `extract_coeffs`.
        Returns:
            list[trimesh.Trimesh]: The reconstructed meshs.
        """
        pred_vertex, pred_tex, pred_color, _ = self.facemodel.compute_for_render(coeffs)
        recon_shape = pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = self.camera_d - recon_shape[..., -1] # from camera space to world space
        face_model = self.facemodel.face_buf.cpu().numpy()

        result = []
        recon_shape = recon_shape.cpu().numpy()
        pred_color = pred_color.cpu().numpy()
        for rshape, rcolor in zip(recon_shape, pred_color):
            mesh = trimesh.Trimesh(vertices=rshape, faces=face_model, vertex_colors=np.clip(255. * rcolor, 0, 255).astype(np.uint8))
            result.append(mesh)
        return result
    
    @torch.no_grad()
    def get_mesh_images(self, coeffs, *, images=None):
        """
        Render the mesh to images with the given coefficients. If `images` is not None, the rendered mesh will overlay the input image.

        Args:
            coeffs (torch.Tensor): The coefficients of the mesh, can be generated from `extract_coeffs`.
            images (torch.Tensor, optional): The input images. The shape should be (N, 3, H, W). Defaults to None.
        Returns:
            np.ndarray: The rendered images. The shape is (N, H, W, 3).
        """
        pred_vertex, pred_tex, pred_color, _ = self.facemodel.compute_for_render(coeffs)
        pred_masks, _, pred_faces = self.renderer(pred_vertex, self.facemodel.face_buf, feat=pred_color)

        if images is not None:
            output_vis = pred_faces * pred_masks + (1 - pred_masks) * images
            output_vis = torch.clamp(output_vis, 0, 1)
            output_vis = output_vis.permute(0, 2, 3, 1).cpu().numpy() * 255
            output_vis = output_vis.astype(np.uint8)
            return output_vis

        pred_faces = pred_faces.permute(0, 2, 3, 1).cpu().numpy() * 255.
        pred_faces = pred_faces.astype(np.uint8)
        return pred_faces
