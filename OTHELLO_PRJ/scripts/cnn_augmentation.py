"""D4 symmetry augmentation for Othello CNN training.

All transforms preserve player ownership and use the same spatial operation
for the (N,3,8,8) state, its action label, and 64-bit legal-action mask.
"""
import torch
from torch.nn import functional as F


def transform_d4_batch(states, labels, masks, rotations=0, flip=False):
    """Rotate 0/90/180/270 degrees CCW, then optionally mirror horizontally.

    Inputs: states [N,3,8,8], labels [N] (row*8+col), masks [N,64].
    Returns transformed tensors with unchanged dtype/device and shapes.
    """
    if states.ndim != 4 or tuple(states.shape[1:]) != (3, 8, 8):
        raise ValueError("states must have shape [N,3,8,8]")
    if labels.shape != (len(states),) or masks.shape != (len(states), 64):
        raise ValueError("labels/masks must have shapes [N] and [N,64]")
    if rotations not in (0, 1, 2, 3):
        raise ValueError("rotations must be 0, 1, 2 or 3")
    if bool(((labels < 0) | (labels >= 64)).any()):
        raise ValueError("labels outside 0..63")

    label_grid = F.one_hot(labels.long(), num_classes=64).reshape(-1, 8, 8)
    mask_grid = masks.reshape(-1, 8, 8)
    s = torch.rot90(states, rotations, dims=(-2, -1))
    l = torch.rot90(label_grid, rotations, dims=(-2, -1))
    m = torch.rot90(mask_grid, rotations, dims=(-2, -1))
    if flip:
        s = torch.flip(s, dims=(-1,))
        l = torch.flip(l, dims=(-1,))
        m = torch.flip(m, dims=(-1,))
    return s, l.reshape(-1, 64).argmax(dim=1).to(labels.dtype), m.reshape(-1, 64)


def augment_d4_batch(states, labels, masks):
    """Select one of eight D4 transforms independently for each train sample.

    Randomized *online* augmentation keeps the batch size and number of
    optimizer steps per epoch the same as the unaugmented baseline.
    """
    picks = torch.randint(0, 8, (len(states),), device=states.device)
    s_out, l_out, m_out = torch.empty_like(states), torch.empty_like(labels), torch.empty_like(masks)
    for choice in range(8):
        indices = torch.nonzero(picks == choice, as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        s, l, m = transform_d4_batch(states[indices], labels[indices], masks[indices],
                                   rotations=choice % 4, flip=(choice >= 4))
        s_out[indices] = s
        l_out[indices] = l
        m_out[indices] = m
    return s_out, l_out, m_out
