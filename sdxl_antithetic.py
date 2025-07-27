import torch
import argparse
from test_prompts import prompts
from utils import pearson_correlation, plot_bchw_tensor
from diffusers import StableDiffusionXLPipeline

def main(model_id="stabilityai/stable-diffusion-xl-base-1.0"):
    # Load Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe._callback_tensor_inputs = ["latents", "prompt_embeds", "noise_pred", "prompt"]
    pipe.to('cuda')

    # Setup the params
    height = 1024
    width = 1024
    lh = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    lw = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    num_channels_latents = pipe.unet.config.in_channels
    batch_size = 2
    guidance_scale = 5.0
    num_inference_steps = 50

    #antithetic
    anti_latents = torch.randn((batch_size//2, num_channels_latents, lh, lw), generator=torch.Generator('cpu').manual_seed(322836428))
    anti_latents = torch.cat([anti_latents, -anti_latents], dim=0).to(dtype=pipe.unet.dtype)

    #normal
    latents = torch.randn((batch_size//2, num_channels_latents, lh, lw), generator=torch.Generator('cpu').manual_seed(123287))
    rand_latents = torch.randn((batch_size//2, num_channels_latents, lh, lw), generator=torch.Generator('cpu').manual_seed(2))
    latents = torch.cat([latents, rand_latents], dim=0).to(dtype=pipe.unet.dtype)

    storage = {}
    for prompt in prompts:
        storage[prompt] = {
            "latents": [],
            "noise_pred": [],
            "pixel_random": None,
            "pixel_anti": None
        }
    """
    {
       "prompt": {
          "latents": (t, corr)...........
          "noise_pred": (t, corr)........
          "pixel_random": corr
          "pixel_anti": corr
        }
    }
    """

    def antithetic_callback(pipe,i, t, callback_kwargs):
        noise_pred = callback_kwargs['noise_pred']
        latents = callback_kwargs['latents']
        prompt = callback_kwargs['prompt'][0]
        storage[prompt]["latents"].append((t, pearson_correlation(latents[0], latents[1])))
        storage[prompt]["noise_pred"].append((t, pearson_correlation(noise_pred[0], noise_pred[1])))
        return callback_kwargs
    
    for prompt in prompts:
        prompt_input = [prompt] * 2
        antithetic_images = pipe(
            prompt_input,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            latents=anti_latents,
            output_type='pt',
            callback_on_step_end_tensor_inputs = ['latents', 'noise_pred', 'prompt_embeds', "prompt"],
            callback_on_step_end=antithetic_callback
        ).images
        print(antithetic_images.shape, antithetic_images.min(), antithetic_images.max())

        storage[prompt]['pixel_anti'] = pearson_correlation(antithetic_images[0], antithetic_images[1])

        random_images = pipe(
            prompt_input,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            latents=latents,
            output_type='pt',
        ).images
        storage[prompt]['pixel_random'] = pearson_correlation(random_images[0], random_images[1])

        plot_bchw_tensor(antithetic_images, title=f"Antithetic Images: {prompt}", 
                         save_path=f"antithetic_images_{model_id.split('/')[-1]}_{prompt.replace(' ', '_')}.jpg")
        plot_bchw_tensor(random_images, title=f"Random Images: {prompt}", 
                         save_path=f"random_images_{model_id.split('/')[-1]}_{prompt.replace(' ', '_')}.jpg")
    print(storage)
    torch.save(storage, f"correlation_results_{model_id.split('/')[-1]}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flux correlation analysis with a specified model ID.")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Model ID for SDXL")
    args = parser.parse_args()
    main(model_id=args.model_id)