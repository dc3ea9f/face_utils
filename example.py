import numpy as np
import torch
from PIL import Image
from face_utils.utils import read_video, write_video
from face_utils.recons import DeepFace3DReconModel, DeepFace3DReconPreprocessor
from face_utils.renders import PIRenderFaceGenerator, PIRenderPreprocessor, PIRenderPostProcessor
from tqdm import tqdm


if __name__ == '__main__':
    video = read_video('example_input.mp4')

    model = DeepFace3DReconModel(
        resume_path='<Path to>/Deep3DFaceRecon_pytorch/checkpoints/face_recon_feat0.2_augment/epoch_20.pth',
        bfm_folder='<Path to>/Deep3DFaceRecon_pytorch/BFM/',
        bfm_model='BFM_model_front.mat',
    )
    recon_preprocessor = DeepFace3DReconPreprocessor(
        bfm_folder='<Path to>/Deep3DFaceRecon_pytorch/BFM/',
    ).cuda()

    video = torch.from_numpy(video).cuda()
    trans_params, new_video, new_landmarks, new_masks = recon_preprocessor.align_and_recrop(video)
    new_video = recon_preprocessor.prepare_input(new_video)

    model = model.cuda()
    model.eval()
    coeffs = model.extract_coeffs(new_video)
    coeffs = {key: value.cpu().numpy() for key, value in coeffs.items()}
    coeffs['crop'] = trans_params.cpu().numpy()[:, 2:]

    # ....
    # You may modify the coeffs through a talking head model
    # ....

    # difference of resolution
    model = PIRenderFaceGenerator()
    model.load_state_dict(torch.load('<Path to>/PIRender/result/face/epoch_00190_iteration_000400000_checkpoint.pt')['net_G_ema'])
    model.eval()
    
    preprocessor = PIRenderPreprocessor(semantic_radius=13)
    poseprocessor = PIRenderPostProcessor()
    semantics = preprocessor.prepare_coeffs(coeffs)
    img = preprocessor.prepare_image(video[0].cpu().numpy())

    model = model.cuda()
    semantics = semantics.cuda().float()
    img = img.cuda().float()

    result_imgs = []
    for semantic in tqdm(semantics):
        result_img = model(img, semantic.unsqueeze(0))
        result_imgs.append(result_img.cpu())
    result_imgs = torch.stack(result_imgs, dim=0)
    result_imgs = poseprocessor.recover_video(result_imgs)
    result_imgs = result_imgs.cpu().numpy()
    write_video('rendered_result.mp4', result_imgs)