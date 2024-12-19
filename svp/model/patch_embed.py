from timm.models.layers import  to_2tuple
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
        2D Image to Patch Embedding

        Parameters:
            img_size (int or tuple): Size of the input image. If an integer is given, it assumes a square image. Default is 224.
            patch_size (int or tuple): Size of the patches to be extracted from the image. If an integer is provided, it assumes a square patch. Default is 16.
            stride (int or tuple): Stride for extracting patches. If an integer is given, it assumes a square stride. Default is 16.
            in_chans (int): Number of channels in the input image. Default is 3, corresponding to RGB images.
            embed_dim (int): Dimension of the embedding space for the patches. Default is 768.
            norm_layer (nn.Module): Layer for normalizing the embeddings. If None, no normalization is applied. Default is None.
            flatten (bool): Specifies whether to flatten the patch embeddings into a 1D vector. Default is True.
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):

        super().__init__()
        # Convert the input image size and patch size to tuples of length 2 for handling different shaped images
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # Calculate the grid size and total number of patches after patching
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Define the projection layer from image patches to embeddings and the normalization layer
        # 输入通道，输出通道，卷积核大小，步长
        # C*H*W->embed_dim*grid_size*grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        B, C, H, W = x.shape #[C:3, B:64, H:224, W:224]
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) # [B:64, embed_dim:192, grid_size:14, grid_size:14]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC [batch_size, num_patches:14*14, embed_dim:192]
        x = self.norm(x)
        return x
