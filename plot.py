import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse
import os

def main(file_path):
    try:
        # Load the data from the PyTorch file
        data = torch.load(file_path)

        # --- Aggregate Data Across All Prompts ---
        latents_correlations_by_timestep = defaultdict(list)
        noise_pred_correlations_by_timestep = defaultdict(list)
        pixel_anti_correlations = []
        pixel_random_correlations = []

        for prompt, values in data.items():
            if values.get('latents'):
                for t, corr in values['latents']:
                    latents_correlations_by_timestep[t.item()].append(corr.item())
            
            if values.get('noise_pred'):
                for t, corr in values['noise_pred']:
                    noise_pred_correlations_by_timestep[t.item()].append(corr.item())

            if values.get('pixel_anti') is not None:
                pixel_anti_correlations.append(values['pixel_anti'].item())
                
            if values.get('pixel_random') is not None:
                pixel_random_correlations.append(values['pixel_random'].item())

        # --- Calculate Statistics for Plotting ---
        
        # For latents
        latents_timesteps = sorted(latents_correlations_by_timestep.keys())
        latents_mean = [np.mean(latents_correlations_by_timestep[t]) for t in latents_timesteps]
        latents_min = [np.min(latents_correlations_by_timestep[t]) for t in latents_timesteps]
        latents_max = [np.max(latents_correlations_by_timestep[t]) for t in latents_timesteps]

        # For noise_pred
        noise_pred_timesteps = sorted(noise_pred_correlations_by_timestep.keys())
        noise_pred_mean = [np.mean(noise_pred_correlations_by_timestep[t]) for t in noise_pred_timesteps]
        noise_pred_min = [np.min(noise_pred_correlations_by_timestep[t]) for t in noise_pred_timesteps]
        noise_pred_max = [np.max(noise_pred_correlations_by_timestep[t]) for t in noise_pred_timesteps]

        # --- Create the Plot ---
        plt.figure(figsize=(14, 8))

        # Plot Latents Correlation Distribution
        plt.plot(latents_timesteps, latents_mean, color='blue', linestyle='-', label='Mean Latents Correlation')
        plt.fill_between(latents_timesteps, latents_min, latents_max, color='blue', alpha=0.2, label='Latents Correlation Range (Min-Max)')

        # Plot Noise Prediction Correlation Distribution
        plt.plot(noise_pred_timesteps, noise_pred_mean, color='orange', linestyle='--', label='Mean Noise Prediction Correlation')
        plt.fill_between(noise_pred_timesteps, noise_pred_min, noise_pred_max, color='orange', alpha=0.2, label='Noise Prediction Correlation Range (Min-Max)')

        # Plot Pixel-Level Correlation Distributions as horizontal bands
        if pixel_anti_correlations:
            mean_anti = np.mean(pixel_anti_correlations)
            min_anti = np.min(pixel_anti_correlations)
            max_anti = np.max(pixel_anti_correlations)
            plt.axhline(y=mean_anti, color='darkgreen', linestyle='--', label=f'Mean Antithetic Pixel Correlation ({mean_anti:.4f})')
            plt.fill_between(latents_timesteps, min_anti, max_anti, color='green', alpha=0.2, label='Antithetic Pixel Correlation Range')

        if pixel_random_correlations:
            mean_random = np.mean(pixel_random_correlations)
            min_random = np.min(pixel_random_correlations)
            max_random = np.max(pixel_random_correlations)
            plt.axhline(y=mean_random, color='darkred', linestyle=':', label=f'Mean Random Pixel Correlation ({mean_random:.4f})')
            plt.fill_between(latents_timesteps, min_random, max_random, color='red', alpha=0.2, label='Random Pixel Correlation Range')


        # --- Configure the Plot ---
        plt.title(f'Correlation Distribution vs. Timestep for {os.path.basename(file_path)}', fontsize=16)
        plt.xlabel('Diffusion Timestep', fontsize=12)
        plt.ylabel('Pearson Correlation', fontsize=12)
        plt.ylim(-1.05, 1.05)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Reverse the x-axis
        ax = plt.gca()
        ax.invert_xaxis()

        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save the plot
        plot_save_path = f"{os.path.splitext(file_path)[0]}_plot.png"
        plt.savefig(plot_save_path)
        print(f"Plot saved to {plot_save_path}")
        plt.show()


    except FileNotFoundError:
        print(f"Error: The data file was not found at '{file_path}'.")
        print("Please ensure the file exists in the correct directory before running this script.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot correlation results from a .pt file.")
    parser.add_argument("file_path", type=str, help="Path to the correlation results .pt file.")
    args = parser.parse_args()
    main(args.file_path)