"""
model.py
PyTorch implementation of Embedded Multi-Layer Perceptron (Embedded MLP)
for tabular housing price regression.
"""

import torch
import torch.nn as nn
from typing import Dict, List

class EmbeddedMLP(nn.Module):
    """
    Embedded Multi-Layer Perceptron (MLP) for Tabular Regression.
    
    Categorical variables are transformed via learned Entity Embedding layers
    (mapping discrete levels into continuous latent representations),
    concatenated with continuous features, and passed through a deep MLP
    with Batch Normalization, LeakyReLU, and Dropout regularization.
    """
    def __init__(
        self,
        cat_dims: Dict[str, int],
        emb_dims: Dict[str, int],
        num_continuous: int,
        hidden_units: List[int] = [256, 128, 64, 32],
        emb_dropout_p: float = 0.08,
        mlp_dropout_p: float = 0.15
    ):
        super(EmbeddedMLP, self).__init__()
        
        self.cat_keys = list(cat_dims.keys())
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cat_dims[key], embedding_dim=emb_dims[key])
            for key in self.cat_keys
        ])
        
        self.emb_dropout = nn.Dropout(p=emb_dropout_p)
        self.bn_continuous = nn.BatchNorm1d(num_continuous)
        
        # Calculate total input dimension after embedding concatenation
        total_emb_dim = sum(emb_dims.values())
        in_features = total_emb_dim + num_continuous
        
        # Build MLP layers
        layers = []
        current_dim = in_features
        
        for h_dim in hidden_units:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(p=mlp_dropout_p))
            current_dim = h_dim
            
        # Final regression head
        layers.append(nn.Linear(current_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat: Tensor of shape (batch_size, num_categorical_columns) containing integer indices
            x_cont: Tensor of shape (batch_size, num_continuous_columns) containing float values
        Returns:
            Output price prediction tensor of shape (batch_size, 1)
        """
        # 1. Look up embeddings for each categorical feature
        emb_vectors = []
        for i, emb_layer in enumerate(self.embeddings):
            col_indices = x_cat[:, i]
            emb_vectors.append(emb_layer(col_indices))
            
        # 2. Concatenate embeddings
        x_emb = torch.cat(emb_vectors, dim=1)
        x_emb = self.emb_dropout(x_emb)
        
        # 3. Batch normalize continuous inputs
        x_cont_norm = self.bn_continuous(x_cont)
        
        # 4. Fusion of embeddings and continuous features
        x_fused = torch.cat([x_emb, x_cont_norm], dim=1)
        
        # 5. Deep MLP propagation
        output = self.mlp(x_fused)
        return output
        
    def get_embedding_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract learned embedding weights for visualization (e.g. t-SNE / PCA clustering).
        """
        weights = {}
        for i, key in enumerate(self.cat_keys):
            weights[key] = self.embeddings[i].weight.detach().cpu().numpy()
        return weights

if __name__ == '__main__':
    # Test initialization with dummy data
    sample_cat_dims = {'location': 10, 'property_type': 5, 'furnishing_status': 4}
    sample_emb_dims = {'location': 8, 'property_type': 4, 'furnishing_status': 3}
    num_cont = 19
    
    model = EmbeddedMLP(sample_cat_dims, sample_emb_dims, num_cont)
    print("EmbeddedMLP architecture initialized successfully:")
    print(model)
    
    dummy_cat = torch.randint(0, 4, (4, 3))
    dummy_cont = torch.randn(4, num_cont)
    out = model(dummy_cat, dummy_cont)
    print(f"\nForward pass verification output shape: {out.shape}")
