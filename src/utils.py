import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from torch import Tensor
from typeguard import typechecked as typechecker

from config import device


@jaxtyped(typechecker=typechecker)
def tensor_to_image(tensor: Float[Tensor, 'h w c'] | Int[Tensor, 'h w c']) -> Image.Image:
    tensor = tensor.sqrt()  # gamma correction
    tensor = tensor.multiply(255).clamp(0, 255)
    array = tensor.cpu().numpy().astype(np.uint8)
    image = Image.fromarray(array, mode='RGB')
    return image


@jaxtyped(typechecker=typechecker)
def random_unit_vector(shape: tuple[int, ...]) -> Float[Tensor, '... 3']:
    vec = torch.randn(*shape, device=device)
    vec = F.normalize(vec, dim=-1)
    return vec


@jaxtyped(typechecker=typechecker)
def random_on_hemisphere(normal: Float[Tensor, '... 3']) -> Float[Tensor, '... 3']:
    vec = random_unit_vector(normal.shape)
    dot_product = torch.sum(vec * normal, dim=-1, keepdim=True)
    return torch.where(dot_product > 0, vec, -vec)


@jaxtyped(typechecker=typechecker)
def background_color_gradient(sample: int, h: int, w: int) -> Float[Tensor, 'sample h w 3']:
    white: Float[Tensor, '3'] = torch.tensor([1.0, 1.0, 1.0], device=device)
    light_blue: Float[Tensor, '3'] = torch.tensor([0.5, 0.7, 1.0], device=device)
    a: Float[Tensor, 'h 1'] = torch.linspace(0, 1, h, device=device).unsqueeze(1)
    background_colors_single: Float[Tensor, 'h 3'] = a * light_blue + (1.0 - a) * white
    background_colors: Float[Tensor, 'sample h w 3'] = (
        background_colors_single.unsqueeze(0).unsqueeze(2).expand(sample, h, w, 3) * 255
    )
    return background_colors


@jaxtyped(typechecker=typechecker)
def random_in_unit_disk(shape: tuple[int, ...]) -> Float[Tensor, '... 2']:
    r: Float[Tensor, '...'] = torch.sqrt(torch.rand(*shape, device=device))
    theta: Float[Tensor, '...'] = torch.rand(*shape, device=device) * 2 * np.pi
    x: Float[Tensor, '...'] = r * torch.cos(theta)
    y: Float[Tensor, '...'] = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)
