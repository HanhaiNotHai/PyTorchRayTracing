from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hittable import HitRecord

from enum import IntEnum

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

from config import typed
from utils import random_unit_vector


class MaterialType(IntEnum):
    Lambertian = 0
    Metal = 1
    Dielectric = 2


@typed
def reflect(v: Float[Tensor, 'N 3'], n: Float[Tensor, 'N 3']) -> Float[Tensor, 'N 3']:
    # Reflects vector v around normal n
    return v - 2 * (v * n).sum(dim=1, keepdim=True) * n


@typed
def refract(
    uv: Float[Tensor, 'N 3'], n: Float[Tensor, 'N 3'], etai_over_etat: Float[Tensor, 'N 1']
) -> Float[Tensor, 'N 3']:
    one = torch.tensor(1.0)
    cos_theta = torch.minimum((-uv * n).sum(dim=1, keepdim=True), one)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -torch.sqrt(torch.abs(one - (r_out_perp**2).sum(dim=1, keepdim=True))) * n
    return r_out_perp + r_out_parallel


@typed
def reflectance(
    cosine: Float[Tensor, 'N 1'], ref_idx: Float[Tensor, 'N 1']
) -> Float[Tensor, 'N 1']:
    one = torch.tensor(1.0)
    r0 = ((one - ref_idx) / (one + ref_idx)) ** 2
    return r0 + (one - r0) * (one - cosine) ** 5


@typed
class Material(ABC):
    @typed
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    @typed
    def scatter_material(
        r_in: Float[Tensor, '* 3 2'],
        hit_record: 'HitRecord',
    ) -> tuple[
        Bool[Tensor, '*'],
        Float[Tensor, '* 3'],
        Float[Tensor, '* 3 2'],
    ]:
        pass


@typed
class Lambertian(Material):
    @typed
    def __init__(self, albedo: Float[Tensor, '3']):
        self.albedo = albedo

    @staticmethod
    @typed
    def scatter_material(
        r_in: Float[Tensor, 'N 3 2'],
        hit_record: 'HitRecord',
    ) -> tuple[
        Bool[Tensor, '*'],
        Float[Tensor, '* 3'],
        Float[Tensor, '* 3 2'],
    ]:
        N = r_in.shape[0]
        normals = hit_record.normal
        points = hit_record.point

        # Generate scatter direction
        scatter_direction = normals + random_unit_vector((N, 3))

        # Handle degenerate scatter direction
        zero_mask = scatter_direction.norm(dim=1) < 1e-8
        scatter_direction[zero_mask] = normals[zero_mask]

        # Normalize scatter direction
        scatter_direction = F.normalize(scatter_direction, dim=-1)

        # Create new rays for recursion
        new_origin = points
        new_direction = scatter_direction
        new_rays = torch.stack([new_origin, new_direction], dim=-1)

        # Attenuation is the albedo
        attenuation = hit_record.albedo
        scatter_mask = torch.ones(N, dtype=torch.bool)

        return scatter_mask, attenuation, new_rays


@typed
class Metal(Material):
    @typed
    def __init__(self, albedo: Float[Tensor, '3'], fuzz: float = 0.3):
        self.albedo = albedo
        self.fuzz = max(0.0, min(fuzz, 1.0))

    @staticmethod
    @typed
    def scatter_material(
        r_in: Float[Tensor, 'N 3 2'],
        hit_record: 'HitRecord',
    ) -> tuple[
        Bool[Tensor, '*'],
        Float[Tensor, 'N 3'],
        Float[Tensor, 'N 3 2'],
    ]:
        N = r_in.shape[0]
        normals = hit_record.normal  # Shape: [N, 3]
        points = hit_record.point  # Shape: [N, 3]

        # Incoming ray directions
        in_directions = r_in[:, :, 1]  # Shape: [N, 3]
        in_directions = F.normalize(in_directions, dim=-1)

        # Generate reflected directions
        fuzz = hit_record.fuzz.unsqueeze(1)  # Shape: [N, 1]

        reflected_direction = reflect(in_directions, normals)
        reflected_direction = reflected_direction + fuzz * random_unit_vector((N, 3))
        reflected_direction = F.normalize(reflected_direction, dim=-1)

        # Check if reflected ray is above the surface
        dot_product = torch.sum(reflected_direction * normals, dim=1)  # Shape: [N]
        scatter_mask = dot_product > 0  # Shape: [N], dtype: bool

        # Create new rays for recursion
        new_origin = points  # Shape: [N, 3]
        new_direction = reflected_direction  # Shape: [N, 3]
        new_rays = torch.stack([new_origin, new_direction], dim=-1)  # Shape: [N, 3, 2]

        # Attenuation is the albedo
        attenuation = hit_record.albedo

        return scatter_mask, attenuation, new_rays


@typed
class Dielectric(Material):
    def __init__(self, refraction_index: float):
        self.refraction_index = refraction_index

    @staticmethod
    @typed
    def scatter_material(
        r_in: Float[Tensor, 'N 3 2'],
        hit_record: 'HitRecord',
    ) -> tuple[
        Bool[Tensor, '*'],
        Float[Tensor, 'N 3'],
        Float[Tensor, 'N 3 2'],
    ]:
        N = r_in.shape[0]
        normals = hit_record.normal  # Shape: [N, 3]
        points = hit_record.point  # Shape: [N, 3]
        front_face = hit_record.front_face  # Shape: [N], dtype: bool
        unit_direction = F.normalize(r_in[:, :, 1], dim=1)  # Shape: [N, 3]

        # Attenuation is always (1, 1, 1) for dielectric materials
        attenuation = torch.ones(N, 3)  # Shape: [N, 3]

        one = torch.tensor(1.0)
        refractive_indices = hit_record.refractive_index.unsqueeze(1)  # Shape: [N, 1]
        refraction_ratio = torch.where(
            front_face.unsqueeze(1),
            one / refractive_indices,
            refractive_indices,
        )

        cos_theta = torch.minimum((-unit_direction * normals).sum(dim=1, keepdim=True), one)
        sin_theta = torch.sqrt(one - cos_theta**2)

        cannot_refract = (refraction_ratio * sin_theta) > one

        # Generate random numbers to decide between reflection and refraction
        reflect_prob = reflectance(cos_theta, refraction_ratio)
        random_numbers = torch.rand(N, 1)
        should_reflect = cannot_refract | (reflect_prob > random_numbers)

        # Compute reflected and refracted directions
        reflected_direction = reflect(unit_direction, normals)
        refracted_direction = refract(unit_direction, normals, refraction_ratio)
        direction = torch.where(
            should_reflect.expand(-1, 3), reflected_direction, refracted_direction
        )
        new_rays = torch.stack([points, direction], dim=-1)

        # Scatter mask is always True for dielectric materials
        scatter_mask = torch.ones(N, dtype=torch.bool)

        return scatter_mask, attenuation, new_rays
