import torch
from tqdm import tqdm
from face_utils.utils import read_video, write_video
from face_utils.recons import DeepFace3DReconModel, DeepFace3DReconPreprocessor
from face_utils.renders import PIRenderFaceGenerator, PIRenderPreprocessor, PIRenderPostProcessor


if __name__ == '__main__':
    video = read_video('example_input.mp4')

    # init deep3dface recon model for extracting coeffs
    model = DeepFace3DReconModel(
        resume_path='<Path to>/Deep3DFaceRecon_pytorch/checkpoints/face_recon_feat0.2_augment/epoch_20.pth',
        bfm_folder='<Path to>/Deep3DFaceRecon_pytorch/BFM/',
        bfm_model='BFM_model_front.mat',
    )

    # init deep3dface recon preprocessor for aligning / recropping / preparing input
    recon_preprocessor = DeepFace3DReconPreprocessor(
        bfm_folder='<Path to>/Deep3DFaceRecon_pytorch/BFM/',
    ).cuda()

    video = torch.from_numpy(video).cuda()
    trans_params, new_video, new_landmarks, new_masks = recon_preprocessor.align_and_recrop(video)
    new_video_tensor = recon_preprocessor.prepare_input(new_video)

    model = model.cuda()
    model.eval()
    coeffs = model.extract_coeffs(new_video_tensor)

    # if want to save the meshs / mesh images / mesh overlay on the input video
    model.init_render()

    # save the meshs
    meshs = model.get_meshs(coeffs)
    for i, mesh in enumerate(tqdm(meshs)):
        mesh.export(f'results/meshs/{i:04d}.obj')
    
    # save the mesh images
    face_images = model.get_mesh_images(coeffs)
    write_video('results/raw_mesh.mp4', face_images)

    # save the mesh overlay on the input video
    face_images = model.get_mesh_images(coeffs, images=new_video)
    write_video('results/raw_mesh_overlay.mp4', face_images)

    coeffs = {key: value.cpu().numpy() for key, value in coeffs.items()}
    coeffs['crop'] = trans_params.cpu().numpy()[:, 2:]


    # ....
    # You may modify the coeffs through a talking head model
    # ....


    # init PIRender model for rendering with the given coeffs
    model = PIRenderFaceGenerator()
    model.load_state_dict(torch.load('<Path to>/PIRender/result/face/epoch_00190_iteration_000400000_checkpoint.pt')['net_G_ema'])
    model.eval()

    preprocessor = PIRenderPreprocessor(semantic_radius=13)
    poseprocessor = PIRenderPostProcessor()
    semantics = preprocessor.prepare_coeffs(coeffs)

    # set identity image as the first frame of the input video
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
    write_video('results/rendered_result.mp4', result_imgs)