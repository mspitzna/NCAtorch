import torch

import sys

sys.path.append("/home/mspitzna/NNCellA/growing_ca/")

from nca.utils.image_utils import make_circle_damage_mask

import torch
from typing import Optional, Tuple, List


class SamplePool:
    """
    Manages a pool of samples (either images x or latents z) for training.
    """
    def __init__(
        self,
        pool_size: int = 1024,
        # seed_ratio, damage_ratio etc. remain the same
        seed_ratio: float = 0.0,
        damage_ratio: float = 0.0,
        mutation_ratio: float = 0.0,
        delay: int = 0,
        replace_after_layer=None,
        device: torch.device = torch.device("cpu"),
    ):
        self.pool_size = pool_size
        self.seed_ratio = seed_ratio
        self.damage_ratio = damage_ratio
        self.current_damage_ratio = damage_ratio
        self.mutation_ratio = mutation_ratio
        self.delay = delay
        self.replace_after_layer = replace_after_layer
        self.device = device
        # self.max_pool_value_threshold = max_pool_value_threshold # Optional filtering

        # Generic pool to store either x (image) or z (latent)
        self.data_pool: List[torch.Tensor] = []
        # Cond and True likely remain images, adjust if needed
        self.cond_pool: List[Optional[torch.Tensor]] = []
        self.true_pool: List[torch.Tensor] = [] # Stores target images

        self.use_scheduling = False
        self.start_ratio = seed_ratio
        self.end_ratio = seed_ratio
        self.total_steps = 1
        self.current_step = 0
        self.dmg_delay = 0 # Added this based on your original code

    def commit(
        self, data: torch.Tensor, cond: Optional[torch.Tensor], true: torch.Tensor
    ) -> None:
        """ Adds new state data (x or z) to the pool. """
        batch_size = data.size(0)
        valid_indices = []

        for i in range(batch_size):
            data_i = data[i].detach().cpu() # Store data (x or z) on CPU

            # Basic validity checks (NaN, maybe all zero depending on data type)
            is_nan = torch.isnan(data_i).any()
            is_all_zero = torch.all(data_i == 0) # Check if relevant for latents too

            # Optional: Add back magnitude check if needed, applied to data_i
            # is_too_large = False
            # if self.max_pool_value_threshold is not None:
            #    max_abs_val = torch.inf if is_nan else torch.abs(data_i).max()
            #    is_too_large = max_abs_val > self.max_pool_value_threshold

            if not is_nan and not is_all_zero: # and not is_too_large:
                valid_indices.append(i)

        if not valid_indices: 
            return

        # Store the state data (x or z)
        data_list = [data[i].detach().cpu() for i in valid_indices]
        # Store corresponding target images and conditions
        true_list = [true[i].detach().cpu() for i in valid_indices]
        if cond is not None:
            cond_list = [cond[i].detach().cpu() for i in valid_indices]
        else:
            cond_list = [None] * len(valid_indices)

        self.data_pool.extend(data_list)
        self.cond_pool.extend(cond_list)
        self.true_pool.extend(true_list)

        # Limit pool size
        if len(self.data_pool) > self.pool_size:
            self.data_pool = self.data_pool[-self.pool_size:]
            self.cond_pool = self.cond_pool[-self.pool_size:]
            self.true_pool = self.true_pool[-self.pool_size:]

    def sample_and_replace(
        self, current_batch_data: torch.Tensor, # Now takes x OR z
        cond: Optional[torch.Tensor],
        true: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """ Replaces portion of current_batch_data (x or z) with data from pool. """

        # Ensure input data is on the correct device
        current_batch_data = current_batch_data.to(self.device)
        true = true.clone().to(self.device) # Keep true as image target
        if cond is not None:
            cond = cond.clone().to(self.device) # Keep cond as image?

        batch_size = current_batch_data.size(0)
        n_replace = int(batch_size * self.seed_ratio)

        if n_replace > 0 and len(self.data_pool) >= n_replace:
            # --- Sample from pool (contains x or z) ---
            pool_indices = torch.randperm(len(self.data_pool))[:n_replace]
            # Sampled data (x or z) - move to device
            data_pool_samples = torch.stack([self.data_pool[i].to(self.device) for i in pool_indices])
            # Sample corresponding target images and conditions from pool
            true_pool_samples = torch.stack([self.true_pool[i].to(self.device) for i in pool_indices])
            if cond is not None:
                cond_pool_samples = torch.stack(
                    [self.cond_pool[i].to(self.device) if self.cond_pool[i] is not None
                     else torch.zeros_like(cond[0]) # Placeholder if None was stored
                     for i in pool_indices]
                )

            # --- Select batch indices to replace ---
            replace_indices = torch.randperm(batch_size)[:n_replace]

            # --- Decide mutation vs full replace ---
            perm = torch.randperm(n_replace)
            n_full = int(n_replace * (1 - self.mutation_ratio))
            full_replace_indices = replace_indices[perm[:n_full]]
            mutation_replace_indices = replace_indices[perm[n_full:]]

            # Split pool samples
            full_data_pool_samples = data_pool_samples[perm[:n_full]]
            mutation_data_pool_samples = data_pool_samples[perm[n_full:]]
            full_true_pool_samples = true_pool_samples[perm[:n_full]]
            if cond is not None:
                full_cond_pool_samples = cond_pool_samples[perm[:n_full]]

            # --- Update the current_batch_data (x or z) ---
            # Apply replace_after_layer logic: only replace channels after a specific index
            if self.replace_after_layer is not None:
                # Only replace channels from replace_after_layer onwards
                print(current_batch_data.shape)
                if mutation_replace_indices.numel() > 0:
                    current_batch_data[mutation_replace_indices, self.replace_after_layer:] = mutation_data_pool_samples[:, self.replace_after_layer:]
                if full_replace_indices.numel() > 0:
                    current_batch_data[full_replace_indices, self.replace_after_layer:] = full_data_pool_samples[:, self.replace_after_layer:]
            else:
                # Full replacement of all channels
                if mutation_replace_indices.numel() > 0:
                    current_batch_data[mutation_replace_indices] = mutation_data_pool_samples
                if full_replace_indices.numel() > 0:
                    current_batch_data[full_replace_indices] = full_data_pool_samples


            # --- Update true and cond ONLY for full replacement indices ---
            # Note: We replace in the original true/cond tensors passed to the function
            if full_replace_indices.numel() > 0:
                true[full_replace_indices] = full_true_pool_samples
                if cond is not None:
                    cond[full_replace_indices] = full_cond_pool_samples

            # --- Optional damage (applied to x or z) ---
            # Note: apply_damage needs to work on both image and latent shapes/ranges
            if self.current_damage_ratio > 0.0 and replace_indices.numel() > 0:
                 # Be cautious if apply_damage assumes image structure/range
                 current_batch_data[replace_indices] = self.apply_damage(current_batch_data[replace_indices])

        # Return the modified state (x or z), and potentially modified cond and true
        return current_batch_data, cond, true


    def set_seed_ratio(self, new_ratio: float) -> None:
        """
        Sets a new seed ratio.

        Args:
            new_ratio (float): The new seed ratio.
        """
        self.seed_ratio = new_ratio

    def set_damage_ratio(self, new_ratio: float) -> None:
        """
        Sets a new damage ratio.

        Args:
            new_ratio (float): The new damage ratio.
        """
        self.damage_ratio = new_ratio

    def get_seed_ratio(self) -> float:
        """
        Gets the current seed ratio.

        Returns:
            float: The current seed ratio.
        """
        return self.seed_ratio

    def get_damage_ratio(self) -> float:
        """
        Gets the current damage ratio.

        Returns:
            float: The current damage ratio.
        """
        return self.current_damage_ratio

    def enable_seed_scheduling(
        self, total_steps: int, start_ratio: float = 0.0, end_ratio: float = 1.0
    ) -> None:
        """
        Enables linear scheduling for 'seed_ratio' over the specified number of steps.

        Args:
            total_steps (int): Total number of steps for scheduling.
            start_ratio (float): Starting seed ratio.
            end_ratio (float): Ending seed ratio.
        """
        self.use_scheduling = True
        self.total_steps = total_steps
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.current_step = 0

    def enable_damage_delay(self, delay: int) -> None:
        """
        Enables a delay before applying damage to the samples.

        Args:
            delay (int): Number of steps to delay damage application.
        """
        self.current_damage_ratio = 0.0
        self.dmg_delay = delay

    def step(self, current_step: Optional[int] = None) -> None:
        """
        Updates the seed ratio based on the current training step if scheduling is enabled.

        Args:
            current_step (int, optional): Current training step. If None, increments by 1.
        """
        if self.use_scheduling:
            self.current_step = (
                current_step if current_step is not None else self.current_step + 1
            )

            if self.current_step < self.delay:
                self.seed_ratio = 0
            elif self.current_step < self.total_steps:
                ratio = self.current_step / self.total_steps
                self.seed_ratio = (
                    self.start_ratio + (self.end_ratio - self.start_ratio) * ratio
                )
            else:
                self.seed_ratio = self.end_ratio

            if self.current_step == self.dmg_delay:
                self.current_damage_ratio = self.damage_ratio

    def apply_damage(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies damage to the samples by masking them.

        Args:
            tensor (Tensor): Samples to apply damage to.

        Returns:
            Tensor: Damaged samples.
        """
        mask = make_circle_damage_mask(tensor.shape[0], tensor.size(-1)).to(tensor.device)
        return tensor * mask


class TimeseriesSamplePool(SamplePool):
    def __init__(
        self,
        pool_size: int = 1024,
        seed_ratio: float = 0.0,
        damage_ratio: float = 0.0,
        delay: int = 0,
        class_transmute: bool = True,  # TODO: no functionality yet
        replace_after_layer=None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            pool_size,
            seed_ratio,
            damage_ratio,
            delay,
            class_transmute,
            replace_after_layer,
            device,
        )
        self.pool_map = {}

    def commit(
        self, data: torch.Tensor, cond: Optional[torch.Tensor], true: torch.Tensor
    ) -> None:
        """
        x: The newly predicted frames
        cond: The condition containing (video_idx, frame_idx)
        target: The ground-truth next frames (optional, depending on your usage)
        """
        # e.g., x shape [batch_size, ...], cond shape [batch_size, 2]
        # if cond holds (video_idx, frame_idx).

        for i in range(data.size(0)):
            # Convert cond[i] to integers
            vid_idx, time_idx = cond[i].tolist()

            # Append x[i], cond[i], target[i] to your pool
            self.data_pool.append(data[i].cpu())
            self.cond_pool.append(cond[i].cpu())
            self.true_pool.append(true[i].cpu())

            # Store the pool index in a dictionary for fast lookup
            pool_idx = len(self.data_pool) - 1
            self.pool_map[(vid_idx, time_idx)] = pool_idx

        # Evict old entries if we exceed max_pool_size
        while len(self.data_pool) > self.pool_size:
            # Remove from front (oldest)
            del self.data_pool[0]
            del self.cond_pool[0]
            del self.true_pool[0]
            # pool_map also needs updating if any item used an old index
            self._rebuild_pool_map()

    def _rebuild_pool_map(self):
        """
        Rebuild pool_map after removing items from x_pool front.
        """
        self.pool_map = {}
        for idx, (x_item, cond_item) in enumerate(zip(self.data_pool, self.cond_pool)):
            vid_idx, time_idx = cond_item.tolist()
            self.pool_map[(vid_idx, time_idx)] = idx

    def sample_and_replace(self, current_batch_data, cond, true):
        """
        For each item in the batch, we randomly pick a subset (n_replace) to attempt to replace.
        If (vid_idx, frame_idx-1) is in pool_map, replace it with the stored sample.
        """
        current_batch_data = current_batch_data.clone().to(self.device)
        true = true.clone().to(self.device)
        if cond is not None:
            cond = cond.clone().to(self.device)

        batch_size = current_batch_data.size(0)
        # Decide how many to replace
        n_replace = int(self.seed_ratio * batch_size)

        if n_replace > 0 and len(self.data_pool) > 0:
            replace_indices = torch.randperm(batch_size)[:n_replace]

            for i in replace_indices:
                vid_idx, frame_idx = cond[i].tolist()
                prev_key = (vid_idx, frame_idx - 1)
                if prev_key in self.pool_map:
                    pool_idx = self.pool_map[prev_key]
                    prev_prediction = self.data_pool[pool_idx].to(self.device)
                    current_batch_data[i] = prev_prediction

                    # Optionally replace the true label if you want
                    # true[i] = something if you store the next ground truth
                    # or keep it as is if your dataset is still correct.

        # (Optional) apply damage
        if self.current_damage_ratio > 0.0:
            current_batch_data = self.apply_damage(current_batch_data)

        return current_batch_data, cond, true


# Example usage
if __name__ == "__main__":
    total_steps = 10  # Total training steps
    pool = SamplePool(pool_size=1024, seed_ratio=0.0, delay=0, device="cpu")
    pool.enable_seed_scheduling(total_steps=total_steps, start_ratio=0.0, end_ratio=0.5)
    # Training loop example
    for step in range(total_steps):
        pool.step()

        # Assume we have tensors seed, cond, true of shape [b, x, x, x]
        b, x = 4, 3
        seed = torch.randn((b, x, x, x))
        cond = torch.randn((b, x, x, x))
        true = torch.randn((b, x, x, x))

        # Commit samples to the pool
        pool.commit(seed, cond, true)

        # Replace random positions in new samples with samples from the pool
        new_seed, new_cond, new_true = (
            torch.zeros((b, x, x, x)),
            torch.zeros((b, x, x, x)),
            torch.zeros((b, x, x, x)),
        )
        updated_seed, updated_cond, updated_true = pool.sample_and_replace(
            new_seed, new_cond, new_true
        )

        # verify the replacement of the seed, cond, true tensors
        print(updated_seed)

        # Here, the updated_seed, updated_cond, updated_true are used in training
        print(f"Step {step + 1}/{total_steps}, Seed Ratio: {pool.get_seed_ratio()}")
