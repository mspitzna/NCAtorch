from abc import ABC, abstractmethod
import torch
import lpips
import torch.nn.functional as F
from torch import nn
from torchvision import models

class Loss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, predictions, targets) -> dict:
        pass


class MMDLoss(Loss):
    def __init__(self, kernel_type='rbf', sigma=1.0):
        super().__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma
    
    def rbf_kernel(self, x, y, sigma=1.0):
        """RBF (Gaussian) kernel"""
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)
        dist = x_norm + y_norm.t() - 2.0 * torch.mm(x, y.t())
        return torch.exp(-dist / (2 * sigma ** 2))
    
    def linear_kernel(self, x, y):
        """Linear kernel"""
        return torch.mm(x, y.t())
    
    def polynomial_kernel(self, x, y, degree=2, gamma=1.0, coef0=1.0):
        """Polynomial kernel"""
        return (gamma * torch.mm(x, y.t()) + coef0) ** degree
    
    def compute_kernel(self, x, y):
        """Compute kernel matrix based on kernel type"""
        if self.kernel_type == 'rbf':
            return self.rbf_kernel(x, y, self.sigma)
        elif self.kernel_type == 'linear':
            return self.linear_kernel(x, y)
        elif self.kernel_type == 'polynomial':
            return self.polynomial_kernel(x, y)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def forward(self, predictions, targets):
        """
        Compute MMD loss between predictions and targets
        
        Args:
            predictions: Tensor of shape [B, C, H, W] or [B, D]
            targets: Tensor of shape [B, C, H, W] or [B, D]
            
        Returns:
            dict: Dictionary containing the MMD loss
        """
        # Flatten tensors to [B, D] if they are images
        if predictions.dim() > 2:
            batch_size = predictions.shape[0]
            predictions_flat = predictions.view(batch_size, -1)
            targets_flat = targets.view(batch_size, -1)
        else:
            predictions_flat = predictions
            targets_flat = targets
        
        # Compute kernel matrices
        K_xx = self.compute_kernel(predictions_flat, predictions_flat)
        K_yy = self.compute_kernel(targets_flat, targets_flat)
        K_xy = self.compute_kernel(predictions_flat, targets_flat)
        
        # Compute MMD^2 estimate
        m = predictions_flat.size(0)
        n = targets_flat.size(0)
        
        mmd_loss = (K_xx.sum() - K_xx.trace()) / (m * (m - 1)) + \
                   (K_yy.sum() - K_yy.trace()) / (n * (n - 1)) - \
                   2 * K_xy.sum() / (m * n)
        
        # Ensure non-negative (due to numerical issues)
        mmd_loss = torch.clamp(mmd_loss, min=0.0)
        
        return {"total_loss": mmd_loss, "mmd_loss": mmd_loss}



class L1Loss(Loss):
    def __init__(self, overflow_loss=False, overflow_weight=1.0, config=None):
        super().__init__()
        self.loss_fn = nn.L1Loss()
        self.overflow_loss = overflow_loss
        self.overflow_weight = overflow_weight
        self.config = config

    def forward(self, predictions, targets):

        # Standard L1 loss for all channels
        l1_loss = self.loss_fn(predictions[:, :targets.shape[1]], targets)
        total_loss = l1_loss
        
        if self.overflow_loss:
            # Check for NaN values in predictions
            if torch.isnan(predictions).any():
                print("Warning: NaN values detected in predictions")
                
            # Safely compute overflow losses with checks
            img_part = predictions[:, :targets.shape[1]]
            hidden_part = predictions[:, targets.shape[1]:] if predictions.shape[1] > targets.shape[1] else None
            
            # Image overflow calculation with safety checks
            img_clamped = img_part.clamp(0, 1.0)
            overflow_loss_img = (img_part - img_clamped).abs().mean()
            
            # Hidden overflow calculation with safety checks
            if hidden_part is not None and hidden_part.numel() > 0:
                hidden_clamped = hidden_part.clamp(-1.0, 1.0)
                overflow_loss_hidden = (hidden_part - hidden_clamped).abs().mean()
            else:
                overflow_loss_hidden = torch.tensor(0.0, device=predictions.device)
            
            overflow_loss = overflow_loss_img + overflow_loss_hidden
            total_loss = total_loss + self.overflow_weight * overflow_loss
            
            return {"total_loss": total_loss, "l1_loss": l1_loss, "overflow_loss": overflow_loss}
        
        return {"total_loss": total_loss, "l1_loss": l1_loss}


class ReconstructionLoss(Loss):
    def __init__(self, loss_fn=None, overflow_loss=False, overflow_weight=1.0):
        super().__init__()
        self.overflow_loss = overflow_loss
        self.overflow_weight = overflow_weight
        if loss_fn is None:
            # If nothing is provided, you can set a default.
            self.loss_fn = nn.MSELoss()
        else:
            # IMPORTANT: Use the actual loss function instance that was passed in.
            self.loss_fn = loss_fn

    def forward(self, predictions, targets):
        mse_loss = self.loss_fn(predictions[:, :targets.shape[1]], targets)
        
        if self.overflow_loss:
            # Check for NaN values in predictions
            if torch.isnan(predictions).any():
                print("Warning: NaN values detected in predictions")
                
            # Safely compute overflow losses with checks
            img_part = predictions[:, :targets.shape[1]]
            hidden_part = predictions[:, targets.shape[1]:] if predictions.shape[1] > targets.shape[1] else None
            
            # Image overflow calculation with safety checks
            img_clamped = img_part.clamp(0, 1.0)
            overflow_loss_img = (img_part - img_clamped).abs().mean()
            
            # Hidden overflow calculation with safety checks
            if hidden_part is not None and hidden_part.numel() > 0:
                hidden_clamped = hidden_part.clamp(-1.0, 1.0)
                overflow_loss_hidden = (hidden_part - hidden_clamped).abs().mean()
            else:
                overflow_loss_hidden = torch.tensor(0.0, device=predictions.device)
            
            overflow_loss = overflow_loss_img + overflow_loss_hidden
            total_loss = mse_loss + self.overflow_weight * overflow_loss
            
            return {"total_loss": total_loss, "mse_loss": mse_loss, "overflow_loss": overflow_loss}
        
        return {"total_loss": mse_loss, "mse_loss": mse_loss}
    

class OverflowLoss(Loss):
    def __init__(self, overflow_weight=1.0):
        super().__init__()
        self.overflow_weight = overflow_weight

    def forward(self, predictions, targets, min_val=0.0, max_val=1.0):
        overflow_loss = (predictions - predictions.clamp(min_val, max_val)).abs().mean()
        return {"total_loss": overflow_loss, "overflow_loss": overflow_loss}


class GrayscaleMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        # Convert RGB to grayscale: simple mean of RGB channels
        predictions_gray = predictions.mean(dim=1, keepdim=True)  # Shape: [B, 1, H, W]
        targets_gray = targets.mean(dim=1, keepdim=True)  # Shape: [B, 1, H, W]

        # Compute MSE loss on grayscale images
        return F.mse_loss(predictions_gray, targets_gray)
    
class LPIPSLoss(Loss):
    def __init__(self, net="alex", device="cuda"):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net=net).to(device) 

    def forward(self, predictions, targets):
        # Handle grayscale targets by extending to RGB
        if targets.shape[1] == 1:
            targets_rgb = targets.repeat(1, 3, 1, 1)  # Extend grayscale to RGB
        else:
            targets_rgb = targets[:, :3]  # Take first 3 channels
        
        # Handle grayscale predictions by extending to RGB
        if predictions.shape[1] == 1:
            predictions_rgb = predictions.repeat(1, 3, 1, 1)  # Extend grayscale to RGB
        else:
            predictions_rgb = predictions[:, :3]  # Take first 3 channels
        
        # Compute LPIPS loss on RGB images
        lpips_loss = self.loss_fn(predictions_rgb, targets_rgb).mean()
        return {"total_loss": lpips_loss, "lpips_loss": lpips_loss}
    
class PixelCrossEntropyLoss(Loss):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, predictions, targets):
        """
        Compute pixel-wise cross-entropy loss.

        Args:
            predictions: [B, num_classes, H, W] - raw logits for each class per pixel
            targets: [B, num_classes, H, W] - one-hot encoded targets
                     OR [B, H, W] - class indices

        Returns:
            dict: Dictionary containing the total loss and cross-entropy loss
        """
        # Convert one-hot targets to class indices if needed
        if targets.dim() == 4 and targets.shape[1] > 1:
            # One-hot encoded: [B, C, H, W] -> [B, H, W]
            class_indices = torch.argmax(targets, dim=1)
        else:
            # Already class indices: [B, H, W] or [B, 1, H, W]
            class_indices = targets.squeeze(1) if targets.dim() == 4 else targets

        # predictions should have shape [B, num_classes, H, W]
        # CrossEntropyLoss expects: input=[B, C, H, W], target=[B, H, W]
        loss_value = self.loss_fn(predictions, class_indices)

        return {"total_loss": loss_value, "cross_entropy_loss": loss_value}

class ImageCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', overflow_loss=False, overflow_weight=1.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)
        self.overflow_loss = overflow_loss
        self.overflow_weight = overflow_weight

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 10, H, W] per-pixel logits
            targets:     [B]            single label per image
        Returns:
            dict:        {'total_loss': ..., 'cross_entropy_loss': ...}
        """
        
        pooled = predictions.amax(dim=(2, 3))   # [B, 10]
        pooled_target = targets.amax(dim=(2, 3))  # [B]

        # adjust pooled_target 0 is 0.1 and 1 is 0.9
        pooled_target = pooled_target * 0.8 + 0.1

        loss_value = self.loss_fn(pooled, pooled_target)

        if self.overflow_loss:
            overflow_loss = (predictions - predictions.clamp(-1.0, 1.0)).abs().mean()
            overflow_loss = overflow_loss.square().mean()

            return {"total_loss": loss_value + self.overflow_weight * overflow_loss, "cross_entropy_loss": loss_value, "overflow_loss": overflow_loss}
        return {"total_loss": loss_value, "cross_entropy_loss": loss_value}


class PixelAccuracyLoss(Loss):
    """
    Computes pixel-wise accuracy across all pixels.
    Returns a dictionary:
      {
        "total_loss": 1 - pixel_accuracy,
        "pixel_accuracy": pixel_accuracy
      }
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        # If one-hot, convert to integer class indices first.
        if targets.dim() == 4 and targets.shape[1] > 1:
            targets = targets.argmax(dim=1)  # -> [B, H, W]

        mask = (targets != 0)  # Ignore background class.
        correct_pixels = (predictions.argmax(dim=1) == targets).float() * mask.float()
        total_pixels = mask.sum().float()
        pixel_accuracy = correct_pixels.sum() / total_pixels

        return {
            "total_loss": pixel_accuracy,
            "pixel_accuracy": pixel_accuracy
        }


class TotalAgreementLoss(Loss):
    """
    For each sample, returns 1 if the mode (most frequent class) among valid pixels
    in the prediction matches the mode in the target, otherwise 0.
    The final metric is the average of these values over the batch.
    """
    def forward(self, predictions, targets):
        # Ensure batch dimensions.
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(0)
        if targets.dim() == 3:
            targets = targets.unsqueeze(0)
        
        # Compute valid mask. For one-hot targets of shape [B, 10, H, W],
        # valid if any channel > 0, and convert to class indices.
        if targets.dim() == 4 and targets.shape[1] == 10:
            valid_mask = (targets > 0.0).any(dim=1)  # [B, H, W]
            target_labels = targets.argmax(dim=1)     # [B, H, W]
        else:
            valid_mask = (targets > 0.0)
            target_labels = targets

        # Use magnitude (absolute value) to determine the class with highest confidence
        pred_magnitudes = predictions.abs()
        pred_classes = pred_magnitudes.argmax(dim=1)  # [B, H, W]
        
        batch_size = predictions.size(0)
        sample_scores = []

        for i in range(batch_size):
            sample_valid = valid_mask[i]
            if sample_valid.sum() == 0:
                # No valid pixels; count as correct.
                sample_scores.append(1.0)
            else:
                valid_preds = pred_classes[i][sample_valid]
                valid_targets = target_labels[i][sample_valid]
                # Compute the mode for predictions and targets.
                mode_pred = valid_preds.mode().values.item()
                mode_target = valid_targets.mode().values.item()
                sample_scores.append(1.0 if mode_pred == mode_target else 0.0)
                
        metric = sum(sample_scores) / batch_size
        return {"total_loss": torch.tensor(metric, device=predictions.device),
                "mode_agreement": sample_scores}


class WassersteinLoss(Loss):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, inputs, target, **kwargs):
        # Get critic output without any sigmoid activation
        output = self.critic(inputs)

        if target == "real":
            # Critic should output high values on real images
            loss = -torch.mean(output)
        elif target == "fake":
            # Critic should output low values on fake images
            loss = torch.mean(output)
        else:
            raise ValueError("Target must be either 'real' or 'fake'")
        return {"total_loss": loss}

class AdversarialLoss(Loss):
    def __init__(
        self,
        discriminator,
        adv_loss_fn=torch.nn.BCEWithLogitsLoss(),
        noise_value=0.1,
        noise_ratio=0.5,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.adv_loss_fn = adv_loss_fn
        self.noise_value = noise_value
        self.noise_ratio = noise_ratio

    def forward(self, predictions, targets, **kwargs):
        # Pass the predictions through the discriminator
        d_output = self.discriminator(predictions)

        # Apply Gaussian noise if noise_value is greater than 0
        if self.noise_value > 0 and torch.rand(1).item() < self.noise_ratio:
            noise = torch.randn_like(d_output) * self.noise_value
            d_output = d_output + noise

        # Set target labels with label smoothing
        if targets == "real":
            target_tensor = torch.full_like(d_output, 0.9, device=d_output.device)
        elif targets == "fake":
            target_tensor = torch.full_like(d_output, 0.1, device=d_output.device)
        else:
            raise ValueError("Targets must be either 'real' or 'fake'")

        # Compute the adversarial loss
        loss_value = self.adv_loss_fn(d_output, target_tensor)
        return {"total_loss": loss_value}


class AdversarialReconstructionLoss(Loss):
    def __init__(
        self,
        discriminator,
        recon_loss_fn=None,
        recon_loss_type="mse",
        adv_weight=1.0,
        recon_weight=1.0,
        noise_value=0.1,
        noise_ratio=0.5,
        device="cuda"
    ):
        super().__init__()
        
        # Initialize reconstruction loss based on type
        if recon_loss_fn is not None:
            self.recon_loss_fn = recon_loss_fn
        elif recon_loss_type.lower() == "lpips":
            self.recon_loss_fn = LPIPSLoss(device=device)
        elif recon_loss_type.lower() == "mse":
            self.recon_loss_fn = ReconstructionLoss()
        else:
            raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")
            
        self.adv_loss_fn = AdversarialLoss(
            discriminator,
            adv_loss_fn=torch.nn.BCEWithLogitsLoss(),
            noise_value=noise_value,
            noise_ratio=noise_ratio,
        )
        self.discriminator = discriminator
        self.adv_weight = adv_weight
        self.recon_weight = recon_weight

    def forward(self, predictions, targets, **kwargs):
        if self.recon_weight > 0:
            recon_loss = self.recon_loss_fn(predictions, targets)["total_loss"]
            adv_loss = self.adv_loss_fn(predictions, "real")["total_loss"]
            total_loss = self.recon_weight * recon_loss + self.adv_weight * adv_loss
            return {
                "total_loss": total_loss,
                "reconstruction_loss": recon_loss,
                "adversarial_loss": adv_loss,
            }
        else:
            adv_loss = self.adv_loss_fn(predictions, "real")["total_loss"]
            return {
                "total_loss": adv_loss,
                "adversarial_loss": adv_loss,
            }

class VGGStyleOTLoss(Loss):
    def __init__(self, proj_n=32, device=None, overflow_loss=False,):
        super(VGGStyleOTLoss, self).__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.vgg16 = models.vgg16(weights='IMAGENET1K_V1').features.to(self.device).eval()
        self.proj_n = proj_n
        self.style_layers = [1, 6, 11, 18, 25]
        # Remove target_img and yy from __init__
        self.overflow_loss = overflow_loss

    def calc_styles_vgg(self, imgs):
        # Ensure imgs are on the correct device
        imgs = imgs.to(self.device)
        # Define mean and std for normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        # Normalize the images
        x = (imgs[:, :3, :, :] - mean) / std
        # Initialize features list with the normalized images reshaped
        b, c, h, w = x.shape
        features = [x.view(b, c, h * w)]
        # Pass through VGG16 layers and extract features
        for i, layer in enumerate(self.vgg16[:max(self.style_layers) + 1]):
            x = layer(x)
            if i in self.style_layers:
                b, c, h, w = x.shape
                features.append(x.view(b, c, h * w))
        return features

    def project_sort(self, x, proj):
        # Project the features
        projected = torch.einsum('bcn,cp->bpn', x, proj)
        # Sort the projections
        sorted_proj = projected.sort(dim=-1)[0]
        return sorted_proj

    def ot_loss(self, source, target):
        ch, n = source.shape[-2:]  # Channels and number of elements
        # Generate random projections
        projs = F.normalize(torch.randn(ch, self.proj_n, device=self.device), dim=0)
        # Project and sort the source and target features
        source_proj = self.project_sort(source, projs)
        target_proj = self.project_sort(target, projs)
        # Interpolate target projections to match source size if needed
        if target_proj.size(-1) != n:
            target_proj = F.interpolate(target_proj, size=n, mode='nearest')
        # Compute the squared difference
        loss = (source_proj - target_proj).square().sum()
        return loss

    def forward(self, imgs, target_imgs):
        # Compute style features for input images and target images
        xx = self.calc_styles_vgg(imgs)
        yy = self.calc_styles_vgg(target_imgs)
        # Compute the total loss as the sum over all style layers
        total_loss = sum(self.ot_loss(x, y) for x, y in zip(xx, yy))

        if self.overflow_loss:
            overflow_loss_img = (imgs[:, :3, :, :] - imgs[:, :3, :, :].clamp(0, 1.0)).abs().mean()
            hidden = imgs[:, 3:, :, :]
            if hidden.numel() > 0:
                overflow_loss_hidden = (hidden - hidden.clamp(-1.0, 1.0)).abs().mean()
            else:
                overflow_loss_hidden = torch.tensor(0.0, device=imgs.device)
            overflow_loss = overflow_loss_img + overflow_loss_hidden
            total_loss = total_loss + overflow_loss
            return {"total_loss": total_loss, "overflow_loss": overflow_loss, "ot_loss": total_loss}
        

        return {"total_loss": total_loss, "ot_loss": total_loss}

class MajorityVotingClassificationLoss(Loss):
    """
    Classification loss based on majority voting from grayscale pixels.
    The grayscale channel in targets encodes digit values (0.0->0, 0.1->1, ..., 0.9->9).
    Performs majority voting on predicted grayscale values to determine the class.
    """
    def __init__(self, threshold=0.05):
        super().__init__()
        self.threshold = threshold
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, C, H, W] - predicted output (grayscale in channel 0)
            targets: [B, C, H, W] - target with grayscale encoding in channel 0
        """
        # Extract grayscale channels
        pred_grayscale = predictions[:, 0, :, :]  # [B, H, W]
        target_grayscale = targets[:, 0, :, :]    # [B, H, W]
        
        # Create mask for non-background pixels (using alpha channel if available)
        if targets.shape[1] > 1:
            alpha_mask = targets[:, 1, :, :] > self.threshold  # [B, H, W]
        else:
            alpha_mask = target_grayscale > self.threshold
        
        batch_size = predictions.shape[0]
        predicted_classes = []
        true_classes = []
        
        for b in range(batch_size):
            # Get valid pixels for this sample
            valid_mask = alpha_mask[b]
            
            if valid_mask.sum() == 0:
                # No valid pixels, default to class 0
                predicted_classes.append(0)
                true_classes.append(0)
                continue
            
            # Get valid grayscale values
            pred_valid = pred_grayscale[b][valid_mask]
            target_valid = target_grayscale[b][valid_mask]
            
            # Convert grayscale values to class predictions (0.0->0, 0.1->1, etc.)
            pred_classes = torch.round(pred_valid * 10).long().clamp(0, 9)
            target_classes = torch.round(target_valid * 10).long().clamp(0, 9)
            
            # Majority vote for predicted class
            pred_class_hist = torch.bincount(pred_classes, minlength=10)
            pred_majority = torch.argmax(pred_class_hist).item()
            
            # Majority vote for target class (should be consistent, but taking majority anyway)
            target_class_hist = torch.bincount(target_classes, minlength=10)
            target_majority = torch.argmax(target_class_hist).item()
            
            predicted_classes.append(pred_majority)
            true_classes.append(target_majority)
        
        # Convert to tensors
        pred_tensor = torch.tensor(predicted_classes, dtype=torch.long, device=predictions.device)
        true_tensor = torch.tensor(true_classes, dtype=torch.long, device=predictions.device)
        
        # Create logits for cross-entropy (one-hot style)
        pred_logits = torch.zeros(batch_size, 10, device=predictions.device)
        pred_logits[torch.arange(batch_size), pred_tensor] = 1.0
        
        # Compute cross-entropy loss
        loss = self.ce_loss(pred_logits, true_tensor)
        
        # Compute accuracy
        accuracy = (pred_tensor == true_tensor).float().mean()
        
        return {
            "total_loss": loss,
            "classification_loss": loss,
            "accuracy": accuracy,
            "predicted_classes": predicted_classes,
            "true_classes": true_classes
        }


class VGGLoss(nn.Module):
    """
    A perceptual loss function using a pre-trained VGG19 network.
    """
    def __init__(self, device, vgg_loss_weight, l1_loss_weight, loss_fn=nn.MSELoss(reduction='sum')):
        super().__init__()
        # Load pre-trained VGG19 from torchvision
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

        # Freeze the VGG network's parameters
        for param in vgg.parameters():
            param.requires_grad = False

        # Use features from an intermediate layer. The layer at index 35 (relu5_4)
        # is a common choice for capturing high-level features.
        self.features = nn.Sequential(*list(vgg.children())[:36])

        # We'll compare the features using L1 Loss
        self.criterion = loss_fn

        self.vgg_loss_weight = vgg_loss_weight
        self.l1_loss_weight = l1_loss_weight

    def forward(self, reconstructed_x, original_x):
        """
        Calculates the VGG perceptual loss.

        Args:
            reconstructed_x (torch.Tensor): The output from the VAE decoder.
            original_x (torch.Tensor): The original input image.
        """
        # Note: VGG expects 3-channel, normalized images.
        # If your VAE input/output is not normalized like ImageNet,
        # you might need to add a normalization step here.
        # Assuming your data is already in a suitable [0,1] or [-1,1] range.

        l1_loss = self.criterion(reconstructed_x, original_x) * self.l1_loss_weight

        # Convert 4-channel RGBA to 3-channel RGB for VGG features
        if reconstructed_x.shape[1] == 4:
            reconstructed_rgb = reconstructed_x[:, :3]  # Take only RGB channels
            original_rgb = original_x[:, :3]
        else:
            reconstructed_rgb = reconstructed_x
            original_rgb = original_x

        x_features = self.features(reconstructed_rgb)
        y_features = self.features(original_rgb)

        vgg_loss = self.criterion(x_features, y_features) * self.vgg_loss_weight

        total_loss = l1_loss + vgg_loss
        return {
            "total_loss": total_loss,
            "l1_loss": l1_loss,
            "vgg_loss": vgg_loss
        }
