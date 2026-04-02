import torch
from torch.utils.data import Dataset
from nca.data.datasets.base_dataset import NCADataset


class CFGDatasetWrapper(NCADataset):
    """
    Wrapper for conditional datasets to enable Classifier-Free Guidance (CFG) training.
    
    During training, this wrapper randomly nullifies condition vectors according to 
    cfg_dropout_prob to enable the model to learn both conditional and unconditional
    generation paths. Supports preserving specific channels from CFG dropout.
    """
    
    def __init__(self, base_dataset, cfg_dropout_prob=0.1, null_condition_type="zeros", preserve_channels=None):
        """
        Args:
            base_dataset (Dataset): The underlying conditional dataset to wrap
            cfg_dropout_prob (float): Probability of nullifying the condition vector
            null_condition_type (str): How to create null conditions ("zeros" or "learned")
            preserve_channels (List[int]): Channel indices to preserve during CFG (0-indexed)
        """
        self.base_dataset = base_dataset
        self.cfg_dropout_prob = cfg_dropout_prob
        self.null_condition_type = null_condition_type
        self.preserve_channels = preserve_channels or []
        
        # For learned null condition, we could add a learnable parameter
        # For now, we'll use zeros as it's simpler and commonly effective
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        seed, condition, target = self.base_dataset[idx]
        
        # Skip CFG if condition is scalar (no conditional info)
        if condition.dim() == 0:
            return seed, condition, target
        
        # Apply classifier-free guidance: randomly null the condition
        if torch.rand(1).item() < self.cfg_dropout_prob:
            condition = self._get_null_condition(condition)
        
        return seed, condition, target
    
    def _get_null_condition(self, original_condition):
        """
        Create a null condition vector for CFG training.
        Preserves specified channels and only zeros out others.
        """
        if self.null_condition_type == "zeros":
            null_condition = torch.zeros_like(original_condition)
        elif self.null_condition_type == "learned":
            # Could implement learnable null embedding here
            # For now, fallback to zeros
            null_condition = torch.zeros_like(original_condition)
        else:
            raise ValueError(f"Unknown null_condition_type: {self.null_condition_type}")
        
        # Preserve specific channels by copying from original condition
        if self.preserve_channels:
            for channel_idx in self.preserve_channels:
                if 0 <= channel_idx < original_condition.size(0):
                    null_condition[channel_idx] = original_condition[channel_idx]
                        
        return null_condition
    
    def __getattr__(self, name):
        if name == 'base_dataset':
            raise AttributeError(name)
        return getattr(self.base_dataset, name)