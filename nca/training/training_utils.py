import torch
def export_model(ca, base_fn):
    torch.save(ca.state_dict(), base_fn)
