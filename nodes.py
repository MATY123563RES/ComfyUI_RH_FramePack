import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
import hashlib
import random
import string
import torchvision
from torchvision.transforms.functional import to_pil_image
import comfy.utils

from PIL import Image
import folder_paths

class Kiki_FramePack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": True}),
                # "n_prompt": ("STRING", {"multiline": True}),
                "total_second_length": ("INT", {"default": 5, "min": 1, "max": 120, "step": 1}),
                "seed": ("INT", {"default": 3407}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "use_teacache": ("BOOLEAN", {"default": True}),
                "upscale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "description": "Resolution scaling factor. 1.0 = original size, >1.0 = upscale, <1.0 = downscale"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    CATEGORY = "Runninghub/FramePack"
    FUNCTION = "run"

    TITLE = 'RunningHub FramePack'
    OUTPUT_NODE = True

    def __init__(self):

        self.high_vram = False

        hunyuan_root = os.path.join(folder_paths.models_dir, 'HunyuanVideo')
        flux_redux_bfl_root = os.path.join(folder_paths.models_dir, 'flux_redux_bfl')
        framePackI2V_root = os.path.join(folder_paths.models_dir, 'FramePackI2V_HY')

        self.text_encoder = LlamaModel.from_pretrained(hunyuan_root, subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        self.text_encoder_2 = CLIPTextModel.from_pretrained(hunyuan_root, subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(hunyuan_root, subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(hunyuan_root, subfolder='tokenizer_2')
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(hunyuan_root, subfolder='vae', torch_dtype=torch.float16).cpu()

        self.feature_extractor = SiglipImageProcessor.from_pretrained(flux_redux_bfl_root, subfolder='feature_extractor')
        self.image_encoder = SiglipVisionModel.from_pretrained(flux_redux_bfl_root, subfolder='image_encoder', torch_dtype=torch.float16).cpu()

        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(framePackI2V_root, torch_dtype=torch.bfloat16).cpu()

        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.image_encoder.eval()
        self.transformer.eval()

        if not self.high_vram:
            self.vae.enable_slicing()
            self.vae.enable_tiling()

        self.transformer.high_quality_fp32_output_for_inference = True
        print('transformer.high_quality_fp32_output_for_inference = True')

        self.transformer.to(dtype=torch.bfloat16)
        self.vae.to(dtype=torch.float16)
        self.image_encoder.to(dtype=torch.float16)
        self.text_encoder.to(dtype=torch.float16)
        self.text_encoder_2.to(dtype=torch.float16)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        if not self.high_vram:
            # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
            DynamicSwapInstaller.install_model(self.transformer, device=gpu)
            DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)

    def preprocess_image(self, image):
        image_np = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8)).convert("RGB")
        input_image = np.array(image)
        return input_image

    def run(self, **kwargs):
        image = kwargs['ref_image']
        image_np = self.preprocess_image(image)
        prompt = kwargs['prompt']
        seed = kwargs['seed']
        total_second_length = kwargs['total_second_length']
        steps = kwargs['steps']
        use_teacache = kwargs['use_teacache']
        upscale = kwargs['upscale']
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        video_path = os.path.join(folder_paths.get_output_directory(), f'{random_str}.mp4')

        self.pbar = comfy.utils.ProgressBar(steps * total_second_length)

        self.exec(input_image=image_np, prompt=prompt, seed=seed, total_second_length=total_second_length, video_path=video_path, steps=steps, use_teacache=use_teacache, scale=upscale)
        if os.path.exists(video_path):
            fps = self.get_fps_with_torchvision(video_path)
            frames = self.extract_frames_as_pil(video_path)
            # os.remove(video_path)
            print(f'{video_path}:{fps} {len(frames)}')

        return (frames, fps)
        
    @torch.no_grad()
    def exec(self, input_image, video_path,
            prompt="The girl dances gracefully, with clear movements, full of charm.", 
            n_prompt="", 
            seed=31337, 
            total_second_length=5, 
            latent_window_size=9, 
            steps=25, 
            cfg=1, 
            gs=32, 
            rs=0, 
            gpu_memory_preservation=6, 
            use_teacache=True,
            scale=1.0):
        
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        try:
            # Clean GPU
            if not self.high_vram:
                unload_complete_models(
                    self.text_encoder, self.text_encoder_2, self.image_encoder, self.vae, self.transformer
                )

            # Text encoding

            print('Text encoding')

            if not self.high_vram:
                fake_diffusers_current_device(self.text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
                load_model_as_complete(self.text_encoder_2, target_device=gpu)

            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            # Processing input image

            print('Image processing ...')

            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            print(f"Resized height: {height}, Resized width: {width}")  # Print resized dimensions 
            def strict_align(h, w, scale):
                raw_h = h * scale
                raw_w = w * scale

                aligned_h = int(round(raw_h / 64)) * 64
                aligned_w = int(round(raw_w / 64)) * 64

                assert (aligned_h % 64 == 0) and (aligned_w % 64 == 0), "尺寸必须是64的倍数"
                assert (aligned_h//8) % 8 == 0 and (aligned_w//8) % 8 == 0, "潜在空间需要8的倍数"
                return aligned_h, aligned_w
            height, width = strict_align(height, width, scale)
            print(f"After Resized height: {height}, Resized width: {width}")  # Print resized dimensions
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # VAE encoding

            print('VAE encoding ...')

            if not self.high_vram:
                load_model_as_complete(self.vae, target_device=gpu)

            start_latent = vae_encode(input_image_pt, self.vae)

            # CLIP Vision

            print('CLIP Vision encoding ...')

            if not self.high_vram:
                load_model_as_complete(self.image_encoder, target_device=gpu)

            image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            # Dtype

            llama_vec = llama_vec.to(self.transformer.dtype)
            llama_vec_n = llama_vec_n.to(self.transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(self.transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(self.transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(self.transformer.dtype)

            print('Start Sample')

            rnd = torch.Generator("cpu").manual_seed(seed)
            num_frames = latent_window_size * 4 - 3

            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            latent_paddings = reversed(range(total_latent_sections))

            if total_latent_sections > 4:
                # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
                # items looks better than expanding it when total_latent_sections > 4
                # One can try to remove below trick and just
                # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            for latent_padding in latent_paddings:
                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * latent_window_size

                # if stream.input_queue.top() == 'end':
                #     stream.output_queue.push(('end', None))
                #     return

                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                if not self.high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(self.transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                if use_teacache:
                    self.transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    self.transformer.initialize_teacache(enable_teacache=False)

                def callback(d):
                    self.update(1)
                    return

                generated_latents = sample_hunyuan(
                    transformer=self.transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    # shift=3.0,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                if not self.high_vram:
                    offload_model_from_device_for_memory_preservation(self.transformer, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(self.vae, target_device=gpu)

                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, self.vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3

                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], self.vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                if not self.high_vram:
                    unload_complete_models()

                if is_last_section:
                    save_bcthw_as_mp4(history_pixels, video_path, fps=30)
                
                # print('video fps:', get_fps_with_torchvision('final.mp4'))
                # print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
        except Exception as e:
            print(e)
        finally:
            unload_complete_models()
        
    def update(self, in_progress):
        self.pbar.update(in_progress)

    def extract_frames_as_pil(self, video_path):
        video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')  # (T, H, W, C)
        frames = [to_pil_image(frame.permute(2, 0, 1)) for frame in video]
        frames = [torch.from_numpy(np.array(frame).astype(np.float32) / 255.0) for frame in frames]
        return frames               

    def get_fps_with_torchvision(self, video_path):
        _, _, info = torchvision.io.read_video(video_path, pts_unit='sec')
        return info['video_fps']

NODE_CLASS_MAPPINGS = {
    "RunningHub_FramePack": Kiki_FramePack,
}
