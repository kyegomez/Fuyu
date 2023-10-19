import torch
from torch import nn


class ImagePatchProjector(nn.Module):
    def __init__(self, patch_size, projected_dim):
        """ """
        super(ImagePatchProjector, self).__init__()
        self.patch_size = patch_size

        # Caluclate patch dimenions when flattened
        patch_dim = patch_size * patch_size * 3  # Assuming RGB
        self.projector = nn.Linear(patch_dim, projected_dim)

    def forward(self, img):
        """
        img: Input tensor of shape (batch_size, channels, height, width)
        """
        # Split image into patches
        patches = self.split_to_patches(img)

        # Flatten and project patches
        batch_size, num_patches_h, num_patches_w, channels, _, _ = patches.shape
        patches = patches.view(batch_size, num_patches_h * num_patches_w, -1)
        projected_patches = self.projector(patches)

        return projected_patches

    def split_to_patches(self, img):
        """
        Img: Input tensor of shape (batch_size, channels, height, width)
        Returns tensor of shape (batch_size, num_patches, channels, patch_size, patch_size)

        """
        b, c, h, w = img.shape
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size

        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

        return patches


# Example usage
img_tensor = torch.randn((8, 3, 224, 224))  # 8 images of size 224x224 with 3 channels
patch_projector = ImagePatchProjector(patch_size=16, projected_dim=64)
projected_patches = patch_projector(img_tensor)
print(
    projected_patches.shape
)  # Should be (batch_size, num_patches_h * num_patches_w, projected_dim)
