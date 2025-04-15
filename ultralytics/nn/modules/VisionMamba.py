import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Định nghĩa PatchEmbedding - thêm padding
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim=3, patch_size=16):
        super().__init__()
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.embed_dim = in_dim * patch_size * patch_size
        
        self.proj = nn.Conv2d(
            in_channels=in_dim,
            out_channels=self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.pos_embed = None

    def forward(self, input_image):
        batch_size, _, height, width = input_image.shape
        
        # Tính padding để height, width chia hết cho patch_size
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            input_image = F.pad(input_image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        patches = self.proj(input_image)
        h_patches, w_patches = patches.shape[2], patches.shape[3]
        num_patches = h_patches * w_patches
        
        patches = rearrange(patches, 'b c h w -> b (h w) c')
        
        if self.pos_embed is None or self.pos_embed.shape[1] != num_patches:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        patches = patches + self.pos_embed
        return patches, height, width  # Trả thêm height, width gốc

# Định nghĩa SSMBlock
class SSMBlock(nn.Module):
    def __init__(self, hidden_dim, delta_init=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.ssm_A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.ssm_B = nn.Parameter(torch.randn(hidden_dim, 1))
        self.ssm_C = nn.Parameter(torch.randn(1, hidden_dim))
        self.ssm_delta = nn.Parameter(torch.ones(1) * delta_init)
        
        self.input_proj = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            groups=hidden_dim,  # Depthwise convolution
            bias=True
        )

    def generate_kernel(self, num_patches):
        discrete_A = torch.exp(self.ssm_delta * self.ssm_A)
        discrete_B = self.ssm_delta * self.ssm_B
        
        kernel = []
        A_power = torch.eye(self.hidden_dim, device=self.ssm_A.device)
        for k in range(num_patches):
            term = self.ssm_C @ A_power @ discrete_B
            kernel.append(term.squeeze())
            A_power = A_power @ discrete_A
        
        return torch.stack(kernel, dim=0)

    def apply_fft_convolution(self, patches, kernel):
        batch_size, num_patches, _ = patches.shape
        
        patches_flat = patches.mean(dim=-1)
        fft_length = 2 * num_patches
        kernel_padded = F.pad(kernel, (0, fft_length - kernel.size(0)))
        
        patches_fft = torch.fft.rfft(patches_flat, n=fft_length)
        kernel_fft = torch.fft.rfft(kernel_padded, n=fft_length)
        output_fft = patches_fft * kernel_fft
        output = torch.fft.irfft(output_fft, n=fft_length)[:, :num_patches]
        
        return output.unsqueeze(-1).repeat(1, 1, self.hidden_dim)

    def forward(self, input_patches):
        # input_patches có dạng (batch_size, num_patches, hidden_dim), ví dụ [1, 121, 576]
        batch_size, num_patches, hidden_dim = input_patches.shape
        # Giả sử num_patches = H * W, chọn H=W=sqrt(num_patches)
        H = W = int(num_patches ** 0.5)  # Với num_patches=121, H=W=11
        assert H * W == num_patches, "num_patches phải là số chính phương"
        
        # Định dạng thành (batch_size, hidden_dim, H, W)
        input_patches = input_patches.view(batch_size, H, W, hidden_dim).permute(0, 3, 1, 2).contiguous()  # [1, 576, 11, 11]
        
        # Áp dụng Conv2d
        input_patches = self.input_proj(input_patches)  # [1, 576, 11, 11]
        
        # Chuyển lại thành (batch_size, num_patches, hidden_dim)
        projected_patches = input_patches.permute(0, 2, 3, 1).view(batch_size, num_patches, hidden_dim)  # [1, 121, 576]

        forward_kernel = self.generate_kernel(projected_patches.shape[1])
        forward_result = self.apply_fft_convolution(projected_patches, forward_kernel)
        backward_kernel = forward_kernel.flip(0)
        backward_result = self.apply_fft_convolution(projected_patches, backward_kernel)
        return (forward_result + backward_result) / 2

# Định nghĩa VisionMambaBlock
class VisionMambaBlock(nn.Module):
    def __init__(self, hidden_dim, num_ssm_blocks=1, shortcut=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shortcut = shortcut
        
        self.ssm_blocks = nn.ModuleList([
            SSMBlock(hidden_dim=hidden_dim)
            for _ in range(num_ssm_blocks)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_patches):
        x = input_patches
        residual = x
        
        for ssm_block in self.ssm_blocks:
            x = self.layer_norm(x)
            x = ssm_block(x)
            x = x + residual if self.shortcut else x
            residual = x
        
        x = self.layer_norm(x)
        x = x + residual if self.shortcut else x
        return x

# Định nghĩa PatchMerging - không dùng interpolate
class PatchMerging(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, patch_size=3, embed_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            stride=1
        )

    def forward(self, input_patches, input_height, input_width):
        batch_size, num_patches, embed_dim = input_patches.shape
        assert embed_dim == self.embed_dim, \
            f"embed_dim ({self.embed_dim}) không khớp với input_patches ({embed_dim})"
        
        h_patches = w_patches = int(num_patches ** 0.5)
        if h_patches * w_patches != num_patches:
            w_patches = num_patches // h_patches
            while h_patches * w_patches != num_patches:
                h_patches -= 1
                w_patches = num_patches // h_patches
                if h_patches <= 0 or w_patches <= 0:
                    raise ValueError(f"Cannot factorize num_patches={num_patches}")
  
        output_image = rearrange(
            input_patches, 'b (h w) (p q c) -> b c (h p) (w q)',
            h=h_patches,
            w=w_patches,
            p=self.patch_size,
            q=self.patch_size,
            c=self.in_dim
        )
        
        # Cắt bỏ phần padding nếu cần để khớp với input_height, input_width
        if output_image.shape[2] > input_height or output_image.shape[3] > input_width:
            output_image = output_image[:, :, :input_height, :input_width]
        
        output_image = self.conv1(output_image)

        return output_image

# Định nghĩa VisionMamba
class VisionMamba(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, num_mamba_blocks=1, patch_size=16, shortcut=True):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            in_dim=in_dim,
            patch_size=patch_size
        )
        self.embed_dim = in_dim * patch_size * patch_size
        self.patch_size = patch_size
        
        self.mamba_block = VisionMambaBlock(
            hidden_dim=self.embed_dim,
            num_ssm_blocks=num_mamba_blocks,
            shortcut=shortcut
        )
        self.patch_merge = PatchMerging(
            in_dim=in_dim,
            out_dim=out_dim,
            patch_size=patch_size,
            embed_dim=self.embed_dim
        )

    def forward(self, input_image):
        patches, orig_height, orig_width = self.patch_embed(input_image)  # Nhận height, width gốc
        mamba_output = self.mamba_block(patches)
        output_image = self.patch_merge(mamba_output, orig_height, orig_width)
        return output_image
