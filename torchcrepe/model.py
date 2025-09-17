import functools

import torch
import torch.nn.functional as F

import torchcrepe


###########################################################################
# Model definition
###########################################################################

class FCN993(torch.nn.Module):
    """FCN-993 model definition (Fully-Convolutional Network for Pitch Estimation of Speech Signals)"""

    def __init__(self, model='full'):
        super().__init__()

        # FCN-993 architecture parameters
        # Based on configuration JSON and build_model code:
        # layers = [1, 2, 3, 4, 5, 6]
        # filters = [256, 32, 32, 128, 256, 512]
        # widths = [32, 32, 32, 32, 32, 32] (kernel_size)
        # strides = [(1, 1), ...] (stride)
        # padding = 'valid' for all convolutions
        # MaxPool2D(pool_size=(2, 1)) only after conv1 (l<4), conv2 (l<4), conv3 (l<4)
        # classifier = Conv2D(486, (4, 1), padding='valid', activation='sigmoid')

        # BatchNorm definition
        # Typical values deduced from Keras (epsilon=0.001, momentum=0.99 in Keras -> pytorch momentum = 1 - 0.99 = 0.01)
        batch_norm_fn = functools.partial(
            torch.nn.BatchNorm2d,
            eps=0.001,      # Common value matching Keras
            momentum=0.01   # 1 - keras_momentum (0.99)
        )

        # Layer definitions
        # Expected input after preprocessing: [B, 993] -> transformed to [B, 1, 993, 1]
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,      # Mono input
            out_channels=256,   # Layer 1 filters
            kernel_size=(32, 1),# Filter size
            stride=(1, 1),      # Stride
            padding=0           # 'valid' padding
        )
        self.conv1_BN = batch_norm_fn(num_features=256)
        # MaxPool2d will be applied in forward after conv1

        self.conv2 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=32,
            kernel_size=(32, 1),
            stride=(1, 1),
            padding=0
        )
        self.conv2_BN = batch_norm_fn(num_features=32)
        # MaxPool2d will be applied in forward after conv2

        self.conv3 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(32, 1),
            stride=(1, 1),
            padding=0
        )
        self.conv3_BN = batch_norm_fn(num_features=32)
        # MaxPool2d will be applied in forward after conv3

        self.conv4 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=128,
            kernel_size=(32, 1),
            stride=(1, 1),
            padding=0
        )
        self.conv4_BN = batch_norm_fn(num_features=128)
        # No MaxPool2d after conv4

        self.conv5 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(32, 1),
            stride=(1, 1),
            padding=0
        )
        self.conv5_BN = batch_norm_fn(num_features=256)
        # No MaxPool2d after conv5

        self.conv6 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(32, 1),
            stride=(1, 1),
            padding=0
        )
        self.conv6_BN = batch_norm_fn(num_features=512)
        # No MaxPool2d after conv6

        # Final classifier layer (replaces Dense)
        # Based on spatial dimension calculations:
        # Input: [B, 1, 993, 1]
        # After conv1-BN-MaxPool: [B, 256, 481, 1]
        # After conv2-BN-MaxPool: [B, 32, 225, 1]
        # After conv3-BN-MaxPool: [B, 32, 97, 1]
        # After conv4-BN: [B, 128, 66, 1]
        # After conv5-BN: [B, 256, 35, 1]
        # After conv6-BN: [B, 512, 4, 1] <- Spatial dimension = 4
        # Classifier kernel is (4, 1) to cover this final dimension.
        self.classifier = torch.nn.Conv2d(
            in_channels=512,
            out_channels=PITCH_BINS_FCN993, # 486 pitch bins
            kernel_size=(4, 1),             # Kernel to cover remaining dimension
            stride=(1, 1),
            padding=0                       # 'valid' padding
        )
        # Sigmoid activation is applied in the forward method

    def forward(self, x, embed=False):
        """
        Forward pass
        Args:
            x (torch.Tensor): Input tensor of shape [B, 993] (as returned by preprocess for FCN-993)
            embed (bool): If True, return the embedding before the final classifier.
        Returns:
            torch.Tensor: If embed=False, probabilities of shape [B, 486].
                         If embed=True, embedding of shape [B, 512, 4].
        """
        # Ensure expected input shape for 2D convolutions
        # x shape: [B, 993]
        x = x.unsqueeze(1)      # [B, 1, 993]
        x = x.unsqueeze(-1)     # [B, 1, 993, 1] - Matches expected input

        # Pass through convolutional blocks
        # Conv1 block
        x = F.relu(self.conv1_BN(self.conv1(x))) # [B, 256, 962, 1]
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1)) # [B, 256, 481, 1]

        # Conv2 block
        x = F.relu(self.conv2_BN(self.conv2(x))) # [B, 32, 450, 1]
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1)) # [B, 32, 225, 1]

        # Conv3 block
        x = F.relu(self.conv3_BN(self.conv3(x))) # [B, 32, 194, 1]
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1)) # [B, 32, 97, 1]

        # Conv4 block
        x = F.relu(self.conv4_BN(self.conv4(x))) # [B, 128, 66, 1]
        # No pooling

        # Conv5 block
        x = F.relu(self.conv5_BN(self.conv5(x))) # [B, 256, 35, 1]
        # No pooling

        # Conv6 block
        x = F.relu(self.conv6_BN(self.conv6(x))) # [B, 512, 4, 1]
        # No pooling

        if embed:
            # Return output just before classifier layer
            # Shape: [B, 512, 4, 1]
            return x.squeeze(-1) # [B, 512, 4] - Remove singleton dimension

        # Classifier layer
        x = self.classifier(x)                  # [B, 486, 4-4+1=1, 1-1+1=1] -> [B, 486, 1, 1]
        logits = x.squeeze(-1).squeeze(-1)      # [B, 486] - Flatten spatial dimensions
        probabilities = torch.sigmoid(logits)   # [B, 486] - Apply sigmoid activation
        return probabilities

    def embed(self, x):
        """
        Convenience method to get the embedding.
        Args:
            x (torch.Tensor): Input tensor of shape [B, 993].
        Returns:
            torch.Tensor: Embedding of shape [B, 512, 4].
        """
        return self.forward(x, embed=True)