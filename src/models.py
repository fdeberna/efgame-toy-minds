from collections import defaultdict
from typing import Dict, Set, Tuple, Iterable, List, Optional
import numpy as np

from sklearn.manifold import TSNE
import networkx as nx
import numpy as np
import random
from typing import List, Dict, Set, Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

## Class for classifier learner
class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim=4, embed_dim=2,num_neurons=4,embed_hidden=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = nn.Linear(input_dim, embed_hidden) #8
        self.relu = nn.ReLU()
        self.project = nn.Linear(embed_hidden, embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2 , num_neurons),#16
            nn.ReLU(),
            nn.Linear(num_neurons, 1)
        )

    def embed_hidden(self, x):
        h = self.relu(self.hidden(x))        # 8D hidden activations
        z = self.project(h)                  # final embedding
        return h, z

    def forward(self, x1, x2):
        _, z1 = self.embed_hidden(x1)
        _, z2 = self.embed_hidden(x2)
        joint = torch.cat([z1, z2], dim=1)
        out = self.classifier(joint)
        return torch.sigmoid(out)


# Turn each 4-bit string into a 4D float tensor (0.0 or 1.0)
def string_to_tensor(s: str) -> torch.Tensor:
    return torch.tensor([float(ch) for ch in s], dtype=torch.float32)


# Convert training pairs to tensors
def batch(pairs, batch_size):
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        x1 = torch.stack([string_to_tensor(a) for a, b, _ in batch_pairs])
        x2 = torch.stack([string_to_tensor(b) for a, b, _ in batch_pairs])
        y = torch.tensor([float(label) for _, _, label in batch_pairs]).unsqueeze(1)
        yield x1, x2, y

# Contrastive Agent that walks within EF components
# #Occasionally teleports to explore disconnected components (like between even/odd parity clusters)
class Agent:
    def __init__(self, G):
        self.G = G
        self.components = list(nx.connected_components(G))
        self.node_to_component = {
            n: i for i, comp in enumerate(self.components) for n in comp
        }
        self.reset()

    def reset(self):
        # Pick a new component
        self.component_id = random.randint(0, len(self.components) - 1)
        self.nodes = list(self.components[self.component_id])
        self.current = random.choice(self.nodes)

    def sample_sequence(self, length=25):
        self.reset()  # Always start fresh in a new component
        seq = [self.current]
        for _ in range(length - 1):
            neighbors = list(self.G.neighbors(self.current))
            if neighbors:
                self.current = random.choice(neighbors)
            else:
                self.current = random.choice(self.nodes)
            seq.append(self.current)
        return seq


# simple embedder
class Embedder(nn.Module):
    def __init__(self, input_dim=4, embed_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


# Contrastive loss (positive = minimize distance)
def contrastive_loss(z1, z2):
    return ((z1 - z2)**2).sum(dim=1).mean()

def info_nce_loss(anchor, positive, negatives, temperature=0.5):
    """
    anchor: [B, D]
    positive: [B, D]
    negatives: [B, N, D]
    """
    B, D = anchor.shape
    N = negatives.shape[1]

    # Normalize
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negatives = F.normalize(negatives, dim=2)

    # Cosine similarity
    pos_sim = (anchor * positive).sum(dim=1, keepdim=True)  # [B, 1]
    neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2)  # [B, N]

    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature  # [B, 1+N]
    labels = torch.zeros(B, dtype=torch.long)  # target = index 0 (positive)

    return F.cross_entropy(logits, labels)

#main embedder with variable parameters
class EmbedderVariable(nn.Module):
    def __init__(self, input_dim=4, embed_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, x):
        return self.net(x)


