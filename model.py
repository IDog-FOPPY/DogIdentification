import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
class Resnet50(nn.Module):
    
    """
    Embedding extraction usimong Resnet-50 backbone

    Parameters
    ----------
    embedding_size:
        Size of embedding vector.

    pretrained:
        Whether to use pretrained weight on ImageNet.
    """
    def __init__(self, embedding_size: int, pretrained=False):
        super().__init__()

        model = models.resnet50(pretrained=pretrained)
        # Features extraction layers without the last fully-connected
        self.features = nn.Sequential(*list(model.children())[:-1])
        # Embeddding layer
        self.embedding = nn.Sequential(
            nn.Linear(in_features=2048, out_features=embedding_size)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding features from image.

        Parameters
        ----------
        image:
            RGB image [3 x H x W].

        Returns
        -------
        torch.Tensor
            Embedding vector
        """
        embedding: torch.Tensor = self.features(image)
        embedding = embedding.flatten(start_dim=1)

        embedding: torch.Tensor = self.embedding(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
