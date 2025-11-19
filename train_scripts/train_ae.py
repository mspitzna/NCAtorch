import os
import sys
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Needed for VAE loss potentially
import shutil
from datetime import datetime
from tqdm import tqdm
import torch
import wandb

# Assume VAE class is defined in modules/models/VAE.py and AE in modules/models/AE.py
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nca.data.dataset_factory import create_dataset
from nca.utils.config import load_config
from nca.training.training_utils import export_model
from nca.utils.visualization import visualize_batch
from nca.utils.image_utils import create_warmup_cosine_scheduler, make_circle_damage_mask
from nca.core.models.model_factory import get_latent_encoder


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train AutoEncoder or VAE Model") # Updated description
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    return parser.parse_args()


def save_model(model, iteration, output_folder, model_type_str):
    """Save the model at the given iteration."""
    model.eval()
    # Use model_type_str in filename
    filename = f"{model_type_str.lower()}_{iteration}.pt"
    export_model(model, os.path.join(output_folder, filename))
    print(f"Model saved: {filename}")
    model.train()


def setup_output_folder(config_path, train_name):
    """Create output folder based on train name and timestamp. And copy the config file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include model type in folder name potentially? Or keep generic for now.
    output_folder = os.path.join("train_log", f"{train_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    # Copy config inside main log folder, not ae subfolder
    shutil.copy(config_path, os.path.join(output_folder, "config.yaml"))

    # Changed subfolder name to be more generic
    model_output_folder = os.path.join(output_folder, "ae_checkpoints")
    os.makedirs(model_output_folder, exist_ok=True)
    # Added visualization folder
    vis_output_folder = os.path.join(output_folder, "ae_imgs")
    os.makedirs(vis_output_folder, exist_ok=True)

    return model_output_folder, vis_output_folder # Return both


# --- VAE Loss Function ---
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0, reconstruction_criterion=nn.MSELoss(reduction='sum')):
    """
    Calculates VAE loss = Reconstruction Loss + Beta * KL Divergence
    Assumes reconstruction_criterion gives sum over batch *and* features.
    Returns average recon loss per sample, average KL loss per sample.
    """
    batch_size = x.size(0)

    # Reconstruction Loss
    # Adjust criterion based on output activation and target range
    # E.g., for Sigmoid output and [0,1] target, BCE might be better:
    # recon_loss = F.binary_cross_entropy(recon_x.view(batch_size, -1), x.view(batch_size, -1), reduction='sum')
    loss_dict = reconstruction_criterion(recon_x, x)
    recon_loss_sum = loss_dict["total_loss"]
    avg_recon_loss = recon_loss_sum / batch_size # Average total error per sample

    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
    avg_kld = torch.mean(kld_per_sample) # Average KL divergence per sample

    total_loss = avg_recon_loss + beta * avg_kld

    return total_loss, avg_recon_loss, avg_kld, loss_dict


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(config.DEVICE) # Ensure device is taken from config

    # update batch size
    config.TRAINING.BATCH_SIZE = config.LATENT_TRAINING.VAE_BATCH_SIZE

    # Get model type from config, default to AE
    model_type = config.LATENT_TRAINING.ENCODER_TYPE

    model_output_folder, vis_output_folder = setup_output_folder(args.config, config.TRAIN_NAME)
    print(f"Model Type: {model_type}")
    print(f"Model Checkpoints: {model_output_folder}")
    print(f"Visualizations: {vis_output_folder}")

    # Initialize wandb only if enabled
    use_wandb = getattr(config, 'WANDB', False)
    if use_wandb:
        wandb.init(
            project=f"{config.PROJECT_NAME}",
            name=f"{config.TRAIN_NAME}_{model_type.lower()}",
            config={
                "model_type": model_type,
                "latent_channels": config.LATENT_TRAINING.LATENT_AE_CHANNEL,
                "compression_level": config.LATENT_TRAINING.LATENT_AE_COMPRESSION,
                "learning_rate": config.LATENT_TRAINING.LATENT_AE_LR,
                "steps": config.LATENT_TRAINING.LATENT_AE_STEPS,
                "warmup_steps": config.LATENT_TRAINING.LATENT_AE_WARMUP_STEPS,
            }
        )
        print("WandB logging enabled")
    else:
        print("WandB logging disabled")

    # Prepare dataset
    dataloader, _, h, w = create_dataset(config)
    print(f"Input H: {h}, W: {w}")

    model, reconstruction_criterion, vae_kl_beta = get_latent_encoder(config, device)


    optimizer = optim.AdamW(model.parameters(), lr=config.LATENT_TRAINING.LATENT_AE_LR, weight_decay=1e-5)
    
    # Scheduler
    lr_scheduler = create_warmup_cosine_scheduler(
        optimizer, 
        warmup_steps=config.LATENT_TRAINING.LATENT_AE_WARMUP_STEPS,
        total_steps=config.LATENT_TRAINING.LATENT_AE_STEPS
    )

    # --- Add Beta Scheduling ---
    vae_kl_beta_target = config.LATENT_TRAINING.VAE_KL_BETA
    kl_warmup_steps = config.LATENT_TRAINING.VAE_KL_WARMUP_STEPS


    steps = config.LATENT_TRAINING.LATENT_AE_STEPS
    log_interval = config.LATENT_TRAINING.LATENT_AE_LOG_INTERVAL
    save_interval = config.LATENT_TRAINING.LATENT_AE_SAVE_INTERVAL

    # Loss accumulators
    train_loss_total_acc = 0.0
    train_loss_recon_acc = 0.0
    train_loss_kld_acc = 0.0

    data_iter = iter(dataloader)
    latent_shape_log = None # To store shape for final log

    # --- Training Loop ---
    for i in tqdm(range(steps), desc=f"Training {model_type}", mininterval=1):
        # Get data
        try:
            # Assuming dataloader yields (seed, cond, target)
            seed, _, target = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            seed, _, target = next(data_iter)

        seed, target = seed.to(device), target.to(device)
        num_samples = seed.size(0) // 2

        # apply damage if enabled in config
        if config.LATENT_TRAINING.APPLY_DAMAGE:
            indices = torch.randperm(target.size(0))
            target[indices[:num_samples]] *= make_circle_damage_mask(num_samples, target.size(-1)).to(target.device)

        # Data Preparation
        # Select half from seed, half from target to form batch
        indices = torch.randperm(seed.size(0))
        # Use all channels from seed (typically RGBA)
        seed_samples = seed[indices[:num_samples]]
        target_samples = target[indices[num_samples:num_samples*2]]
        batch = torch.cat((seed_samples, target_samples), dim=0)
        indices = torch.randperm(batch.size(0))
        batch = batch[indices] # Input to AE/VAE is shuffled mix of seeds/targets

        # --- Forward & Loss Calculation ---
        optimizer.zero_grad()

        if kl_warmup_steps > 0:
            current_beta = vae_kl_beta_target * min(1.0, i / kl_warmup_steps) # Linear warmup for beta
        else:
            current_beta = vae_kl_beta_target

        if model_type == "VAE":
            outputs, mu, logvar, latent_z = model(batch) # VAE returns more
            loss, recon_loss_val, kld_loss_val, loss_dict = vae_loss_function(
                outputs, batch, mu, logvar, current_beta, reconstruction_criterion
            )
            latent_shape_log = latent_z.shape # Log shape of sampled z
        else: # AE
            outputs, _, latent = model(batch) # AE returns fewer items
            loss = reconstruction_criterion(outputs, batch) # Use standard mean MSE for AE
            recon_loss_val = loss.item() # Store item for accumulation
            kld_loss_val = 0.0 # No KL loss for AE
            latent_shape_log = latent.shape # Log shape of deterministic latent

        # --- Backward & Optimize ---
        loss.backward()
        # Optional: Add gradient clipping if needed (especially for VAE/deeper models)
        # clip_norm = getattr(config.LATENT_TRAINING, 'GRADIENT_CLIPPING_NORM', None)
        # if clip_norm is not None:
        #    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        lr_scheduler.step()
        
        # Accumulate losses
        train_loss_total_acc += loss.item()
        train_loss_recon_acc += recon_loss_val if isinstance(recon_loss_val, float) else recon_loss_val.item()
        train_loss_kld_acc += kld_loss_val if isinstance(kld_loss_val, float) else kld_loss_val.item()

        # Log to wandb every step
        if use_wandb:
            wandb.log({
                "loss": loss_dict["total_loss"],
                "l1_loss": loss_dict["l1_loss"] if "l1_loss" in loss_dict else None,
                "vgg_loss": loss_dict["vgg_loss"] if "vgg_loss" in loss_dict else None,
                "kl_loss": kld_loss_val if isinstance(kld_loss_val, float) else kld_loss_val.item(),
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "step": i + 1
            })

        # --- Logging ---
        if (i + 1) % log_interval == 0:
            avg_loss = train_loss_total_acc / log_interval
            avg_recon_loss = train_loss_recon_acc / log_interval
            avg_kld_loss = train_loss_kld_acc / log_interval

            log_msg = f"Step {i+1}/{steps}: Total Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}"
            if model_type == "VAE":
                log_msg += f", KL Loss: {avg_kld_loss:.4f}"
            print(log_msg)

            # Generate visualization
            vis_result = visualize_batch(
                batch, # Show original mixed batch
                outputs, # Show reconstruction (from sampled z)
                model.decode(latent_z) if model_type == "VAE" else batch, # Show deterministic decode from z
                {}, # No extra info needed here
                i + 1,
                plot=False,
                save_path=vis_output_folder, # Save to dedicated folder
            )

            # Log images to wandb
            if use_wandb:
                vis_image_path = os.path.join(vis_output_folder, f"step_{i+1}.png")
                if vis_result is not None:
                    wandb.log({
                        "reconstructions": wandb.Image(vis_image_path),
                        "avg_loss": avg_loss,
                        "avg_recon_loss": avg_recon_loss,
                        "avg_kl_loss": avg_kld_loss
                    })
            
            # Reset accumulators
            train_loss_total_acc = 0.0
            train_loss_recon_acc = 0.0
            train_loss_kld_acc = 0.0

        # --- Saving ---
        if (i + 1) % save_interval == 0:
            save_model(model, i + 1, model_output_folder, model_type)
            # Log model to wandb
            if use_wandb:
                wandb.save(os.path.join(model_output_folder, f"{model_type.lower()}_{i+1}.pt"))

        if i == 0:  # Log initial model shape
            print("--- Initial Model Shape Summary ---")
            print(f"Final Input Batch Shape: {batch.shape}")
            print(f"Final Latent Shape: {latent_shape_log}")
            print(f"Final Output Shape: {outputs.shape}")
            print("--- End of Initial Model Shape Summary ---")
    # End of training loop



    # --- Final Save & Log ---
    final_model_name = f"{model_type.lower()}.pt"
    export_model(model, os.path.join(model_output_folder, final_model_name))

    print("\nTraining completed.")
    print(f"Model saved at {os.path.join(model_output_folder, final_model_name)}")
    if batch is not None: # Ensure batch exists for logging shapes
         print(f"Final Input Batch Shape: {batch.shape}")
    if latent_shape_log is not None:
        print(f"Final Latent Shape: {latent_shape_log}")
    if 'outputs' in locals() and outputs is not None: # Check if outputs exists
        print(f"Final Output Shape: {outputs.shape}")

    # Log final model to wandb
    #if use_wandb:
    #    wandb.save(os.path.join(model_output_folder, final_model_name))

    # Changed FOLDER_NAME to suggest model path for clarity
    print(f"\nConsider adding to your config: LATENT_MODEL_PATH: {os.path.join(model_output_folder, final_model_name)}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()