import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL


class SDAutoencoderKL(nn.Module):
    def __init__(self, model_type: str):
        super().__init__()

        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{model_type}")

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.vae.encode(x).latent_dist.sample().mul_(0.18215)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / 0.18215).sample
