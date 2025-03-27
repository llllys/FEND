import torch
import torch.nn as nn
import torch.nn.functional as F


class PCLLoss(torch.nn.Module):
    def __init__(self, tau=0.07, alpha=10.0):
        """
        Initialize the PCL loss module.
        
        Args:
            tau (float): Temperature for the instance-wise contrastive term.
            alpha (float): Smoothing factor for cluster density calculation.
        """
        super(PCLLoss, self).__init__()
        self.tau = tau
        self.alpha = alpha

    def instance_wise_loss(self, features, labels):
        """
        Compute the instance-wise contrastive loss using tensor operations.
        
        Args:
            features (torch.Tensor): Normalized feature embeddings, shape (batch_size, feature_dim)
            labels (torch.Tensor): Labels for each instance, shape (batch_size,)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        batch_size = features.shape[0]
        # Normalize features
        features = F.normalize(features, dim=1)
        # Compute similarity matrix: (batch_size, batch_size)
        similarity_matrix = torch.matmul(features, features.T) / self.tau
        
        # Create positive pair mask
        labels_eq = labels[:, None] == labels[None, :]  # (batch_size, batch_size)
        pos_mask = labels_eq
        # not_self = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
        # pos_mask = labels_eq & not_self  # True for positive pairs
        
        # Compute log-probabilities using log-softmax
        log_exp_sim = F.log_softmax(similarity_matrix, dim=1)  # (batch_size, batch_size)

        # Count positive samples per instance
        N_po = pos_mask.sum(dim=1).float()  # (batch_size,)
        valid_instances = N_po > 0
        
        # Extract log-probabilities for positive pairs, keeping [b, b] shape
        pos_log_probs = log_exp_sim * pos_mask.float()  # [b, b]
        pos_log_probs_sum = pos_log_probs.sum(dim=1)  # [b], total log-prob per agent
        
        # Compute loss per agent: -mean(log_prob) over positive pairs
        loss = torch.zeros(batch_size, device=features.device)  # [b]
        loss[valid_instances] = -pos_log_probs_sum[valid_instances] / N_po[valid_instances]
                
        return loss

    def compute_cluster_prototypes(self, features, labels):
        num_clusters = labels.max().item() + 1
        prototypes = torch.zeros(num_clusters, features.shape[1], device=features.device)
        cluster_sizes = torch.zeros(num_clusters, device=features.device)
        for cluster_id in range(num_clusters):
            mask = (labels == cluster_id)
            if mask.sum() > 0:
                prototypes[cluster_id] = features[mask].mean(dim=0)
                cluster_sizes[cluster_id] = mask.sum()
        return prototypes, cluster_sizes

    def compute_cluster_densities(self, features, labels, prototypes):
        num_clusters = prototypes.shape[0]
        densities = torch.zeros(num_clusters, device=features.device)
        for cluster_id in range(num_clusters):
            mask = (labels == cluster_id)
            if mask.sum() > 0:
                cluster_features = features[mask]
                Z = mask.sum().float()
                
                # No momentum update is implemented here [no detailed description of momentum update in FEND paper]
                # Assuming v_z' is the same as the original feature
                distances = torch.norm(cluster_features - prototypes[cluster_id], dim=1, p=2)
                mean_distance = distances.sum() / Z
                densities[cluster_id] = mean_distance / (Z * torch.log(Z + self.alpha))
        densities = densities.clamp(min=1e-6)  # Avoid division by zero
        return densities

    def instance_prototype_loss(self, features, hierarchical_labels):
        """
        Compute the instance-prototype loss using tensor operations.
        
        Args:
            features (torch.Tensor): Normalized feature embeddings, shape (batch_size, feature_dim)
            hierarchical_labels (torch.Tensor): cluster labels for each hierarchy, shape (batch_size, num_hierarchies)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        M = hierarchical_labels.shape[1]
        batch_size = features.shape[0]
        # Normalize features
        features = F.normalize(features, dim=1)
        loss = features.new_zeros(len(features))
        
        for m in range(M):
            labels = hierarchical_labels[:, m]
            
            # Compute prototypes and densities (assumed to be pre-vectorized)
            prototypes, cluster_sizes = self.compute_cluster_prototypes(features, labels)
            densities = self.compute_cluster_densities(features, labels, prototypes)
            prototypes = F.normalize(prototypes, dim=1)
            
            # Compute similarities: (batch_size, num_clusters)
            similarities = torch.matmul(features, prototypes.T)
            # Scale by densities
            scaled_similarities = similarities / densities.view(1, -1)
            
            # Get cluster indices for each instance
            cluster_ids = labels.long()  # Shape: (batch_size,)
            
            # Compute log-probabilities
            log_probs = F.log_softmax(scaled_similarities, dim=1)  # (batch_size, num_clusters)
            
            # Select log-prob for the correct cluster
            pos_log_probs = log_probs[torch.arange(batch_size), cluster_ids]
            
            # Loss for this hierarchy: negative mean log-prob
            # loss_m = -pos_log_probs.mean()
            
            loss_m = -pos_log_probs
            loss += loss_m
        
        # Average over hierarchies
        loss = loss / M if M > 0 else loss
        return loss

    def forward(self, features, hierarchical_labels):
        """
        Compute PCL loss with momentum updates.
        
        Args:
            features (torch.Tensor): Batch features, shape (batch_size, feature_dim)
            hierarchical_labels (torch.Tensor): cluster labels for each hierarchy, shape (batch_size, num_hierarchies)
        
        Returns:
            tuple: Total loss, instance-wise loss, instance-prototype loss
        """
        ins_loss = self.instance_wise_loss(features, hierarchical_labels[:, 0])
        proto_loss = self.instance_prototype_loss(features, hierarchical_labels)
        total_loss = ins_loss + proto_loss # Shape: [B]
        
        return total_loss