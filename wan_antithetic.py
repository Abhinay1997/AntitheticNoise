import torch
import argparse
import os
from test_prompts import short_prompts, long_prompts
from utils import pearson_correlation, plot_bchw_tensor
from diffusers import AutoencoderKLWan, WanPipeline

def run_analysis(pipe, prompts, output_dir, model_id):
    os.makedirs(output_dir, exist_ok=True)
    # Setup the params
    height = 480
    width = 832
    num_frames = 1
    lh = 2 * (int(height) // (pipe.vae_scale_factor_spatial * 2))
    lw = 2 * (int(width) // (pipe.vae_scale_factor_spatial * 2))
    num_latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    num_channels_latents = pipe.transformer.config.in_channels
    batch_size = 2
    guidance_scale = 5.0
    max_sequence_length = 512
    num_inference_steps = 30

    #antithetic
    anti_latents = torch.randn((batch_size//2, num_channels_latents, num_latent_frames, lh, lw), generator=torch.Generator('cpu').manual_seed(34239))
    anti_latents = torch.cat([anti_latents, -anti_latents], dim=0)

    #normal
    latents = torch.randn((batch_size//2, num_channels_latents, num_latent_frames, lh, lw), generator=torch.Generator('cpu').manual_seed(123287))
    rand_latents = torch.randn((batch_size//2, num_channels_latents, num_latent_frames, lh, lw), generator=torch.Generator('cpu').manual_seed(2))
    latents = torch.cat([latents, rand_latents], dim=0)

    storage = {}
    for prompt in prompts:
        storage[prompt] = {
            "latents": [],
            "noise_pred": [],
            "pixel_random": None,
            "pixel_anti": None
        }

    def antithetic_callback(pipe,i, t, callback_kwargs):
        noise_pred = callback_kwargs['noise_pred']
        latents = callback_kwargs['latents']
        prompt = callback_kwargs['prompt'][0]
        storage[prompt]["latents"].append((t, pearson_correlation(latents[0], latents[1])))
        storage[prompt]["noise_pred"].append((t, pearson_correlation(noise_pred[0], noise_pred[1])))
        return callback_kwargs
    
    for idx, prompt in enumerate(prompts):
        prompt_input = [prompt] * 2
        antithetic_images = pipe(
            prompt_input,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            latents=anti_latents,
            output_type='pt',
            callback_on_step_end_tensor_inputs = ['latents', 'noise_pred', 'prompt_embeds', "prompt"],
            callback_on_step_end=antithetic_callback
        ).frames
        print(antithetic_images.shape, antithetic_images.min(), antithetic_images.max())

        storage[prompt]['pixel_anti'] = pearson_correlation(antithetic_images[0], antithetic_images[1])

        random_images = pipe(
            prompt_input,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            latents=latents,
            output_type='pt',
        ).frames
        storage[prompt]['pixel_random'] = pearson_correlation(random_images[0], random_images[1])

        plot_bchw_tensor(antithetic_images[:, 0, :, :, :], title=f"Antithetic Images: {prompt}", 
                         save_path=os.path.join(output_dir, f"antithetic_images_{model_id.split('/')[-1]}_{idx}.jpg"))
        plot_bchw_tensor(random_images[:, 0, :, :, :], title=f"Random Images: {prompt}", 
                         save_path=os.path.join(output_dir, f"random_images_{model_id.split('/')[-1]}_{idx}.jpg"))
    print(storage)
    torch.save(storage, os.path.join(output_dir, f"correlation_results_{model_id.split('/')[-1]}.pt"))

def main(model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers"):
    # Load Pipeline
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe._callback_tensor_inputs = ["latents", "prompt_embeds", "noise_pred", "prompt"]

    # Run for short prompts
    run_analysis(pipe, short_prompts, f"wan_results/short_prompts", model_id)
    
    # Run for long prompts
    run_analysis(pipe, long_prompts, f"wan_results/long_prompts", model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flux correlation analysis with a specified model ID.")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                        help="Model ID for WanPipeline")
    args = parser.parse_args()
    main(model_id=args.model_id)