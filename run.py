import os
import sys
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video

# ============================================================
#  OPTION A — Zero123++ v1.2 avec vs sans UNet fine-tuné
#  OPTION B — SyncDreamer avec adaptateur 16→6 vues
# ============================================================

def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras

def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    frames = torch.cat(frames, dim=1)[0]
    return frames

# ============================================================
# OPTION B — Adaptateur SyncDreamer
# ============================================================
SYNCDREAMER_AZIMUTHS = np.arange(16) * (360.0 / 16)
INSTANTMESH_TARGET_AZIMUTHS = np.array([30, 90, 150, 210, 270, 330])

def select_syncdreamer_views(syncdreamer_grid: Image.Image) -> torch.Tensor:
    grid = np.asarray(syncdreamer_grid, dtype=np.float32) / 255.0
    H, W, _ = grid.shape
    h, w = H // 4, W // 4

    views = []
    for row in range(4):
        for col in range(4):
            view = grid[row*h:(row+1)*h, col*w:(col+1)*w]
            views.append(view)
    views = np.stack(views, axis=0)

    selected_indices = []
    for target in INSTANTMESH_TARGET_AZIMUTHS:
        diffs = np.abs(SYNCDREAMER_AZIMUTHS - target)
        diffs = np.minimum(diffs, 360 - diffs)
        best_idx = int(np.argmin(diffs))
        selected_indices.append(best_idx)

    print(f"[SyncDreamer] Azimuths cibles InstantMesh : {INSTANTMESH_TARGET_AZIMUTHS}")
    print(f"[SyncDreamer] Indices sélectionnés dans les 16 vues : {selected_indices}")
    print(f"[SyncDreamer] Azimuths sélectionnés : {SYNCDREAMER_AZIMUTHS[selected_indices]}")

    selected = views[selected_indices]

    selected_resized = []
    for v in selected:
        img = Image.fromarray((v * 255).astype(np.uint8)).resize((320, 320), Image.LANCZOS)
        selected_resized.append(np.asarray(img, dtype=np.float32) / 255.0)
    selected_resized = np.stack(selected_resized, axis=0)

    tensor = torch.from_numpy(selected_resized).permute(0, 3, 1, 2).float()
    return tensor

def load_syncdreamer():
    syncdreamer_root = "/content/SyncDreamer"
    try:
        sys.path.append(syncdreamer_root)
        from generate import load_model
        cfg = f"{syncdreamer_root}/configs/syncdreamer.yaml"
        ckpt = f"{syncdreamer_root}/ckpt/syncdreamer-pretrain.ckpt"
        old_cwd = os.getcwd()
        os.chdir(syncdreamer_root) # to resolve relative paths insie the model's init
        model = load_model(cfg, ckpt)
        os.chdir(old_cwd)
        return model
    except ImportError as e:
        raise ImportError(
            "SyncDreamer non installé. "
            "Cloner https://github.com/liuyuan-pal/SyncDreamer "
            "et placer le ckpt dans ckpt/syncdreamer-pretrain.ckpt"
        ) from e

###############################################################################
# Arguments
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('input_path', type=str, help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
parser.add_argument(
    '--diffusion_model',
    type=str,
    default='zero123plus_finetuned',
    choices=['zero123plus_finetuned', 'zero123plus_base', 'syncdreamer'],
    help='Quel modèle de diffusion utiliser'
)
args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration
###############################################################################
config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False
device = torch.device('cuda')

print(f'[Diffusion] Modèle choisi : {args.diffusion_model}')

if args.diffusion_model in ['zero123plus_finetuned', 'zero123plus_base']:
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )

    if args.diffusion_model == 'zero123plus_finetuned':
        print('[Diffusion] Chargement du UNet fine-tuné InstantMesh (fond blanc)...')
        if os.path.exists(infer_config.unet_path):
            unet_ckpt_path = infer_config.unet_path
        else:
            unet_ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename="diffusion_pytorch_model.bin",
                repo_type="model"
            )
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        pipeline.unet.load_state_dict(state_dict, strict=True)
        print('[Diffusion] UNet fine-tuné chargé ✓')
    else:
        print('[Diffusion] OPTION A : UNet de base Zero123++ v1.2 (sans fine-tuning)')

    pipeline = pipeline.to(device)
    syncdreamer_model = None
elif args.diffusion_model == 'syncdreamer':
    print('[Diffusion] OPTION B : Chargement de SyncDreamer...')
    pipeline = None
    syncdreamer_model = load_syncdreamer()
    syncdreamer_model = syncdreamer_model.to(device)
    print('[Diffusion] SyncDreamer chargé ✓')

print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(
        repo_id="TencentARC/InstantMesh",
        filename=f"{config_name.replace('-', '_')}.ckpt",
        repo_type="model"
    )
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)
model = model.to(device)

if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

output_subfolder = f"{config_name}_{args.diffusion_model}"
image_path = os.path.join(args.output_path, output_subfolder, 'images')
mesh_path  = os.path.join(args.output_path, output_subfolder, 'meshes')
video_path = os.path.join(args.output_path, output_subfolder, 'videos')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path,  exist_ok=True)
os.makedirs(video_path, exist_ok=True)

if os.path.isdir(args.input_path):
    input_files = [
        os.path.join(args.input_path, f)
        for f in os.listdir(args.input_path)
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.webp')
    ]
else:
    input_files = [args.input_path]

print(f'Total input images: {len(input_files)}')

###############################################################################
# Stage 1: Génération multi-vues
###############################################################################
rembg_session = None if args.no_rembg else rembg.new_session()
outputs = []

for idx, image_file in enumerate(input_files):
    name = os.path.basename(image_file).split('.')[0]
    print(f'[{idx+1}/{len(input_files)}] Imagining {name} ... (mode: {args.diffusion_model})')

    input_image = Image.open(image_file)
    if not args.no_rembg:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    if args.diffusion_model in ['zero123plus_finetuned', 'zero123plus_base']:
        output_image = pipeline(
            input_image,
            num_inference_steps=args.diffusion_steps,
        ).images[0]
        output_image.save(os.path.join(image_path, f'{name}.png'))

        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)

    elif args.diffusion_model == 'syncdreamer':
        sys.path.append("/content/SyncDreamer")
        from ldm.util import prepare_inputs

        with torch.no_grad():
            data = prepare_inputs(image_file, elevation=30)

            for k, v in data.items():
                data[k] = v.unsqueeze(0).cuda()
                data[k] = torch.repeat_interleave(data[k], 1, dim=0)

            x_sample = syncdreamer_model.sample(
                data,
                cfg_scale=2.0,
                batch_view_num=8,
            )

            x_sample = (torch.clamp(x_sample, max=1.0, min=-1.0) + 1) * 0.5

            x_sample = (
                x_sample.permute(0, 1, 3, 4, 2)
                .cpu()
                .numpy()
                * 255
            ).astype(np.uint8)

            # x_sample shape:
            # [B, 16, H, W, 3]

            views = [
                Image.fromarray(x_sample[0, i])
                for i in range(16)
            ]

            # Save horizontal grid
            output_grid = Image.fromarray(
                np.concatenate([x_sample[0, i] for i in range(16)], axis=1)
            )
        output_grid.save(os.path.join(image_path, f'{name}_syncdreamer_grid.png'))
        images = select_syncdreamer_views(output_grid)
        print(f'[SyncDreamer] 6 vues sélectionnées parmi 16 ✓')

    outputs.append({'name': name, 'images': images})

if pipeline is not None:
    del pipeline
if syncdreamer_model is not None:
    del syncdreamer_model

###############################################################################
# Stage 2: Reconstruction
###############################################################################
input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
chunk_size = 20 if IS_FLEXICUBES else 1

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating mesh for {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_cameras_view = input_cameras[:, indices]
    else:
        input_cameras_view = input_cameras

    with torch.no_grad():
        planes = model.forward_planes(images, input_cameras_view)
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )

        if args.export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)

        print(f"Mesh saved to {mesh_path_idx}")

        if args.save_video:
            video_path_idx = os.path.join(video_path, f'{name}.mp4')
            render_size = infer_config.render_resolution
            render_cameras = get_render_cameras(
                batch_size=1, M=120, radius=args.distance,
                elevation=20.0, is_flexicubes=IS_FLEXICUBES,
            ).to(device)
            frames = render_frames(
                model, planes,
                render_cameras=render_cameras,
                render_size=render_size,
                chunk_size=chunk_size,
                is_flexicubes=IS_FLEXICUBES,
            )
            save_video(frames, video_path_idx, fps=30)
            print(f"Video saved to {video_path_idx}")
