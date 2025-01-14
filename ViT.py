import torch
from torch import nn
from einops.layers.torch import Rearrange
import math

class GELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

# Patch Embedding with Conv2D
class PatchEmbeddingConv(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, embedding_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_size = embedding_size
        
        # A 2D convolution with a kernel size equal to the patch size, stride = patch_size
        self.conv = nn.Conv2d(in_channels, embedding_size, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # Shape of input: (batch_size, in_channels, height, width)
        x = self.conv(x)
        # Output shape: (batch_size, embedding_size, grid_height, grid_width)
        x = x.flatten(2)  # Flatten height and width dimensions into one
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embedding_size)
        return x

# Positional Embedding + CLS Token
class TransformerEmbedding(nn.Module):
    def __init__(self, embedding_size=128, patch_size=8, image_size=144):
        super().__init__()
        self.embedding_size = embedding_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Define CLS token: A learnable token of shape (1, 1, embedding_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_size))
        
        # Define Position Embedding: Learnable embedding for each patch
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embedding_size))

    def forward(self, x):
        # Add cls token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Shape: (batch_size, 1, embedding_size)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: (batch_size, num_patches + 1, embedding_size)

        # Add position embedding
        x += self.position_embedding
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim*expansion)
        self.activation = GELUActivation()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim*expansion, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, expansion, dropout):
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.feed_forward = FeedForward(embed_dim, expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        norm1 = self.pre_norm(x)
        #x += self.attention(norm1, norm1, norm1)[0]
        x = torch.add(x, self.attention(norm1, norm1, norm1)[0])
        x = self.dropout(x)

        norm2 = self.pre_norm(x)
        #x += self.feed_forward(norm2)
        x = torch.add(x, self.feed_forward(norm2))
        x = self.dropout(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, expansion, dropout, num_encoders):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, expansion, dropout) for _ in range(num_encoders)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = embed_dim)
        self.fc = nn.Linear(in_features = embed_dim, out_features = num_classes)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)
        return x

class ViT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_encoders, num_classes, patch_size, img_size, expansion, dropout=0.1):
        super(ViT, self).__init__()

        self.patch_embed = PatchEmbeddingConv(patch_size=patch_size, embedding_size=embed_dim)
        self.transformer_embed = TransformerEmbedding(embedding_size=embed_dim, patch_size=patch_size, image_size=img_size)

        self.encoder_block = TransformerEncoder(embed_dim, num_heads, expansion, dropout, num_encoders)

        self.mlp_classifier = MLPClassifier(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer_embed(x).permute(1, 0, 2)
        x = self.encoder_block(x)
        x = self.mlp_classifier(x[0])
        return x
