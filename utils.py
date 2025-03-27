import torch
import numpy as np


def impute_head_tail_zeros(traj_tensor):
    """    
    Impute zeros at the head and tail of each trajectory with their nearest non-zero neighbor.
    [MISSING VALUES IN AV2 DATASET (SEEMS TO) ONLY APPEAR AT HEAD/TAIL]
    
    Args:
        traj_tensor (torch.Tensor): Input tensor of shape [n, t, 2] where zeros appear only at head/tail.
    
    Returns:
        torch.Tensor: Tensor with head and tail zeros imputed, same shape [n, t, 2].
    """
    n, t, _ = traj_tensor.shape
    device = traj_tensor.device
    
    # Identify zero points (both x and y are 0)
    is_zero = (traj_tensor == 0).all(dim=2)  # [n, t], True where both coordinates are 0
    
    # Clone the input tensor to avoid modifying it in-place
    imputed_traj = traj_tensor.clone()
    
    # Process each trajectory
    for i in range(n):
        zero_mask = is_zero[i]  # [t]
        if not zero_mask.any():
            continue  # Skip if no zeros
        
        # Find the first and last non-zero indices
        valid_indices = torch.where(~zero_mask)[0]  # Indices of non-zero points
        if len(valid_indices) == 0:
            continue  # Skip if all zeros (though unlikely given your assumption)
        
        first_valid_idx = valid_indices[0]
        last_valid_idx = valid_indices[-1]
        
        # Impute head zeros (from start to first valid point)
        if first_valid_idx > 0:
            imputed_traj[i, :first_valid_idx] = imputed_traj[i, first_valid_idx]
        
        # Impute tail zeros (from last valid point to end)
        if last_valid_idx < t - 1:
            imputed_traj[i, last_valid_idx + 1:] = imputed_traj[i, last_valid_idx]
    
    return imputed_traj