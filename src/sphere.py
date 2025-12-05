from math import inf

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor
from typeguard import typechecked as typechecker

from hittable import HitRecord, Hittable
from materials import Material


@jaxtyped(typechecker=typechecker)
class Sphere(Hittable):
    def __init__(self, center: Float[Tensor, '3'], radius: float, material: Material):
        self.center: Float[Tensor, '3'] = center
        self.radius: float = max(radius, 0.0)
        self.material: Material = material

    def hit(
        self,
        pixel_rays: Float[Tensor, 'N 3 2'],
        t_min: float,
        t_max: float,
    ) -> HitRecord:
        N: int = pixel_rays.shape[0]
        record: HitRecord = HitRecord.empty((N,))

        origin: Float[Tensor, 'N 3'] = pixel_rays[:, :, 0]
        pixel_directions: Float[Tensor, 'N 3'] = pixel_rays[:, :, 1]

        oc: Float[Tensor, 'N 3'] = origin - self.center

        # Solve quadratic equation
        a: Float[Tensor, 'N'] = (pixel_directions**2).sum(dim=1)
        b: Float[Tensor, 'N'] = 2.0 * (pixel_directions * oc).sum(dim=1)
        c: Float[Tensor, 'N'] = (oc**2).sum(dim=1) - self.radius**2

        discriminant: Float[Tensor, 'N'] = b**2 - 4 * a * c
        sphere_hit: Bool[Tensor, 'N'] = discriminant >= 0

        t_hit: Float[Tensor, 'N'] = torch.full((N,), inf)
        sqrt_discriminant: Float[Tensor, 'N'] = torch.zeros(N)
        sqrt_discriminant[sphere_hit] = torch.sqrt(discriminant[sphere_hit])

        # Compute roots
        t0: Float[Tensor, 'N'] = torch.zeros(N)
        t1: Float[Tensor, 'N'] = torch.zeros(N)
        denom: Float[Tensor, 'N'] = 2.0 * a
        t0[sphere_hit] = (-b[sphere_hit] - sqrt_discriminant[sphere_hit]) / denom[sphere_hit]
        t1[sphere_hit] = (-b[sphere_hit] + sqrt_discriminant[sphere_hit]) / denom[sphere_hit]

        t0_valid: Bool[Tensor, 'N'] = (t0 > t_min) & (t0 < t_max)
        t1_valid: Bool[Tensor, 'N'] = (t1 > t_min) & (t1 < t_max)

        t_hit = torch.where((t0_valid) & (t0 < t_hit), t0, t_hit)
        t_hit = torch.where((t1_valid) & (t1 < t_hit), t1, t_hit)

        sphere_hit = sphere_hit & (t_hit < inf)

        # Compute hit points and normals where sphere_hit is True
        hit_points: Float[Tensor, 'N 3'] = origin + pixel_directions * t_hit.unsqueeze(-1)
        normal_vectors: Float[Tensor, 'N 3'] = F.normalize(hit_points - self.center, dim=1)

        # Update the record
        record.hit = sphere_hit
        record.t[sphere_hit] = t_hit[sphere_hit]
        record.point[sphere_hit] = hit_points[sphere_hit]
        record.normal[sphere_hit] = normal_vectors[sphere_hit]
        record.set_face_normal(pixel_directions, record.normal)

        # Set material for hits
        indices = sphere_hit.nonzero(as_tuple=False).squeeze(-1)
        for idx in indices:
            record.material[idx] = self.material
        return record


class SphereList(Hittable):
    def __init__(
        self,
        centers: Float[Tensor, '* 3'],
        radii: Float[Tensor, '*'],
        material_types: Int[Tensor, '*'],
        albedos: Float[Tensor, '* 3'],
        fuzzes: Float[Tensor, '*'],
        refractive_indices: Float[Tensor, '*'],
    ):
        self.centers: Float[Tensor, '* 3'] = centers
        self.radii: Float[Tensor, '*'] = radii
        self.material_types: Int[Tensor, '*'] = material_types
        self.albedos: Float[Tensor, '* 3'] = albedos
        self.fuzzes: Float[Tensor, '*'] = fuzzes
        self.refractive_indices: Float[Tensor, '*'] = refractive_indices

    def hit(self, pixel_rays: Float[Tensor, 'N 3 2'], t_min: float, t_max: float) -> HitRecord:
        N: int = pixel_rays.shape[0]
        M: int = self.centers.shape[0]
        rays_origin: Float[Tensor, 'N 3'] = pixel_rays[:, :, 0]
        rays_direction: Float[Tensor, 'N 3'] = pixel_rays[:, :, 1]

        rays_origin: Float[Tensor, 'N M 3'] = rays_origin.unsqueeze(1).expand(-1, M, -1)
        rays_direction: Float[Tensor, 'N M 3'] = rays_direction.unsqueeze(1).expand(-1, M, -1)
        centers: Float[Tensor, 'N M 3'] = self.centers.unsqueeze(0).expand(N, -1, -1)
        radii: Float[Tensor, 'N M'] = self.radii.unsqueeze(0).expand(N, -1)

        oc: Float[Tensor, 'N M 3'] = rays_origin - centers

        a: Float[Tensor, 'N M'] = (rays_direction**2).sum(dim=2)
        b: Float[Tensor, 'N M'] = 2.0 * (rays_direction * oc).sum(dim=2)
        c: Float[Tensor, 'N M'] = (oc**2).sum(dim=2) - radii**2

        discriminant: Float[Tensor, 'N M'] = b**2 - 4 * a * c
        valid_discriminant: Bool[Tensor, 'N M'] = discriminant >= 0

        sqrt_discriminant: Float[Tensor, 'N M'] = torch.zeros_like(discriminant)
        sqrt_discriminant[valid_discriminant] = torch.sqrt(discriminant[valid_discriminant])

        denom: Float[Tensor, 'N M'] = 2.0 * a
        t0: Float[Tensor, 'N M'] = torch.full_like(discriminant, inf)
        t1: Float[Tensor, 'N M'] = torch.full_like(discriminant, inf)

        t0[valid_discriminant] = (
            -b[valid_discriminant] - sqrt_discriminant[valid_discriminant]
        ) / denom[valid_discriminant]
        t1[valid_discriminant] = (
            -b[valid_discriminant] + sqrt_discriminant[valid_discriminant]
        ) / denom[valid_discriminant]

        t0_valid: Bool[Tensor, 'N M'] = (t0 > t_min) & (t0 < t_max)
        t1_valid: Bool[Tensor, 'N M'] = (t1 > t_min) & (t1 < t_max)

        t_hit: Float[Tensor, 'N M'] = torch.full_like(discriminant, inf)
        t_hit[t0_valid] = t0[t0_valid]
        t_hit[t1_valid & (t1 < t_hit)] = t1[t1_valid & (t1 < t_hit)]

        sphere_hit: Bool[Tensor, 'N M'] = valid_discriminant & (t_hit < inf)

        t_hit_min: Float[Tensor, 'N'] = torch.min(t_hit, dim=1)[0]
        sphere_indices: Int[Tensor, 'N'] = torch.min(t_hit, dim=1)[1]
        sphere_hit_any: Bool[Tensor, 'N'] = sphere_hit.any(dim=1)
        t_hit_min[~sphere_hit_any] = inf

        record: HitRecord = HitRecord.empty((N,))
        record.hit = sphere_hit_any
        record.t[sphere_hit_any] = t_hit_min[sphere_hit_any]
        rays_direction: Float[Tensor, 'N 3'] = rays_direction[:, 0, :]
        rays_origin: Float[Tensor, 'N 3'] = rays_origin[:, 0, :]
        hit_points: Float[Tensor, 'N 3'] = rays_origin + rays_direction * t_hit_min.unsqueeze(1)
        centers_hit: Float[Tensor, 'N 3'] = self.centers[sphere_indices]
        normal_vectors: Float[Tensor, 'N 3'] = F.normalize(hit_points - centers_hit, dim=1)
        record.point[sphere_hit_any] = hit_points[sphere_hit_any]
        record.normal[sphere_hit_any] = normal_vectors[sphere_hit_any]
        record.set_face_normal(rays_direction, record.normal)

        record.material_type[sphere_hit_any] = self.material_types[sphere_indices[sphere_hit_any]]
        record.albedo[sphere_hit_any] = self.albedos[sphere_indices[sphere_hit_any]]
        record.fuzz[sphere_hit_any] = self.fuzzes[sphere_indices[sphere_hit_any]]
        record.refractive_index[sphere_hit_any] = self.refractive_indices[
            sphere_indices[sphere_hit_any]
        ]

        return record


class SphereInfo:

    def __init__(self):
        self.sphere_centers: list[list[float, float, float]] = []
        self.sphere_radii: list[float] = []
        self.material_types: list[int] = []
        self.albedos: list[list[float, float, float]] = []
        self.fuzzes: list[float] = []
        self.refractive_indices: list[float] = []

    def add(
        self,
        sphere_center: list[float, float, float],
        sphere_radius: float,
        material_type: int,
        albedo: list[float, float, float],
        fuzz: float = 0.0,
        refractive_index: float = 0.0,
    ):
        self.sphere_centers.append(sphere_center)
        self.sphere_radii.append(sphere_radius)
        self.material_types.append(material_type)
        self.albedos.append(albedo)
        self.fuzzes.append(fuzz)
        self.refractive_indices.append(refractive_index)

    def pack(self):
        sphere_centers = torch.tensor(self.sphere_centers, dtype=torch.float32)
        sphere_radii = torch.tensor(self.sphere_radii, dtype=torch.float32)
        material_types = torch.tensor(self.material_types, dtype=torch.int64)
        albedos = torch.tensor(self.albedos, dtype=torch.float32)
        fuzzes = torch.tensor(self.fuzzes, dtype=torch.float32)
        refractive_indices = torch.tensor(self.refractive_indices, dtype=torch.float32)
        return sphere_centers, sphere_radii, material_types, albedos, fuzzes, refractive_indices
