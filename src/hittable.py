from abc import ABC, abstractmethod
from math import inf
from typing import List

import torch
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
class HitRecord:
    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        hit: Bool[Tensor, '...'],
        point: Float[Tensor, '... 3'],
        normal: Float[Tensor, '... 3'],
        t: Float[Tensor, '...'],
        front_face: Bool[Tensor, '...'],
        material_type: Int[Tensor, '...'],
        albedo: Float[Tensor, '... 3'],
        fuzz: Float[Tensor, '...'],
        refractive_index: Float[Tensor, '...'],
    ):
        self.hit = hit
        self.point = point
        self.normal = normal
        self.t = t
        self.front_face = front_face
        self.material_type = material_type
        self.albedo = albedo
        self.fuzz = fuzz
        self.refractive_index = refractive_index

    @jaxtyped(typechecker=typechecker)
    def set_face_normal(
        self,
        ray_direction: Float[Tensor, '... 3'],
        outward_normal: Float[Tensor, '... 3'],
    ) -> None:
        '''Determines whether the hit is from the outside or inside.'''
        self.front_face: Bool[Tensor, '...'] = (ray_direction * outward_normal).sum(dim=-1) < 0
        self.normal: Float[Tensor, '... 3'] = torch.where(
            self.front_face.unsqueeze(-1), outward_normal, -outward_normal
        )

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def empty(shape):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hit = torch.full(shape, False, dtype=torch.bool, device=device)
        point = torch.zeros((*shape, 3), dtype=torch.float32, device=device)
        normal = torch.zeros((*shape, 3), dtype=torch.float32, device=device)
        t_values = torch.full(shape, inf, dtype=torch.float32, device=device)
        front_face = torch.full(shape, False, dtype=torch.bool, device=device)
        material_type = torch.full(shape, -1, dtype=torch.long, device=device)
        albedo = torch.zeros((*shape, 3), dtype=torch.float32, device=device)
        fuzz = torch.zeros(shape, dtype=torch.float32, device=device)
        refractive_index = torch.zeros(shape, dtype=torch.float32, device=device)
        return HitRecord(
            hit, point, normal, t_values, front_face, material_type, albedo, fuzz, refractive_index
        )


@jaxtyped(typechecker=typechecker)
class Hittable(ABC):
    '''Abstract class for hittable objects.'''

    @abstractmethod
    @jaxtyped(typechecker=typechecker)
    def hit(
        self,
        pixel_rays: Float[Tensor, 'N 3 2'],
        t_min: float,
        t_max: float,
    ) -> HitRecord:
        pass


@jaxtyped(typechecker=typechecker)
class HittableList(Hittable):
    '''List of hittable objects.'''

    def __init__(self, objects: List[Hittable] = []):
        self.objects: List[Hittable] = objects

    def add(self, object: Hittable) -> None:
        self.objects.append(object)

    def hit(
        self,
        pixel_rays: Float[Tensor, 'N 3 2'],
        t_min: float,
        t_max: float,
    ) -> HitRecord:
        from config import device

        N: int = pixel_rays.shape[0]
        record: HitRecord = HitRecord.empty((N,))
        closest_so_far: Float[Tensor, 'N'] = torch.full((N,), t_max, device=device)

        for obj in self.objects:
            obj_record: HitRecord = obj.hit(pixel_rays, t_min, t_max)
            closer_mask: Bool[Tensor, 'N'] = obj_record.hit & (obj_record.t < closest_so_far)
            closest_so_far = torch.where(closer_mask, obj_record.t, closest_so_far)

            record.hit = record.hit | obj_record.hit
            record.point = torch.where(closer_mask.unsqueeze(-1), obj_record.point, record.point)
            record.normal = torch.where(
                closer_mask.unsqueeze(-1), obj_record.normal, record.normal
            )
            record.t = torch.where(closer_mask, obj_record.t, record.t)
            record.front_face = torch.where(closer_mask, obj_record.front_face, record.front_face)

            # Update materials
            indices = closer_mask.nonzero(as_tuple=False).squeeze(-1)
            for idx in indices.tolist():
                record.material[idx] = obj_record.material[idx]

        return record
