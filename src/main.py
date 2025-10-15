import random

import torch
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from camera import Camera
from config import device
from materials import MaterialType
from sphere import SphereList


@jaxtyped(typechecker=typechecker)
def random_color():
    return torch.tensor([random.random(), random.random(), random.random()], device=device)


def create_random_spheres_scene():
    sphere_centers = []
    sphere_radii = []
    material_types = []
    albedos = []
    fuzzes = []
    refractive_indices = []

    # Ground sphere
    sphere_centers.append(torch.tensor([0, -1000, 0], device=device))
    sphere_radii.append(1000.0)
    material_types.append(MaterialType.Lambertian)
    albedos.append(torch.tensor([0.5, 0.5, 0.5], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    # Random small spheres
    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = torch.tensor(
                [a + 0.9 * random.random(), 0.2, b + 0.9 * random.random()], device=device
            )
            if (center - torch.tensor([4, 0.2, 0], device=device)).norm() > 0.9:
                if choose_mat < 0.8:
                    # Diffuse
                    albedo = random_color() * random_color()
                    material_type = MaterialType.Lambertian
                    fuzz = 0.0
                    refractive_index = 0.0
                elif choose_mat < 0.95:
                    # Metal
                    albedo = random_color() * 0.5 + 0.5
                    fuzz = random.uniform(0, 0.5)
                    material_type = MaterialType.Metal
                    refractive_index = 0.0
                else:
                    # Glass
                    albedo = torch.tensor([0.0, 0.0, 0.0], device=device)
                    fuzz = 0.0
                    refractive_index = 1.5
                    material_type = MaterialType.Dielectric
                sphere_centers.append(center)
                sphere_radii.append(0.2)
                material_types.append(material_type)
                albedos.append(albedo)
                fuzzes.append(fuzz)
                refractive_indices.append(refractive_index)

    # Three larger spheres
    sphere_centers.append(torch.tensor([0, 1, 0], device=device))
    sphere_radii.append(1.0)
    material_types.append(MaterialType.Dielectric)
    albedos.append(torch.tensor([0.0, 0.0, 0.0], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(1.5)

    sphere_centers.append(torch.tensor([-4, 1, 0], device=device))
    sphere_radii.append(1.0)
    material_types.append(MaterialType.Lambertian)
    albedos.append(torch.tensor([0.4, 0.2, 0.1], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    sphere_centers.append(torch.tensor([4, 1, 0], device=device))
    sphere_radii.append(1.0)
    material_types.append(MaterialType.Metal)
    albedos.append(torch.tensor([0.7, 0.6, 0.5], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    # Convert lists to tensors
    sphere_centers = torch.stack(sphere_centers)
    sphere_radii = torch.tensor(sphere_radii, device=device)
    material_types = torch.tensor(material_types, device=device)
    albedos = torch.stack(albedos)
    fuzzes = torch.tensor(fuzzes, device=device)
    refractive_indices = torch.tensor(refractive_indices, device=device)

    world = SphereList(
        centers=sphere_centers,
        radii=sphere_radii,
        material_types=material_types,
        albedos=albedos,
        fuzzes=fuzzes,
        refractive_indices=refractive_indices,
    )

    camera = Camera(
        image_width=320,
        samples_per_pixel=50,
        aspect_ratio=16.0 / 9.0,
        max_depth=50,
        vfov=20,
        look_from=torch.tensor([13, 2, 3], dtype=torch.float32, device=device),
        look_at=torch.tensor([0, 0, 0], dtype=torch.float32, device=device),
        vup=torch.tensor([0, 1, 0], dtype=torch.float32, device=device),
        defocus_angle=0.6,
        focus_dist=10.0,
        batch_size=50_000,
    )

    return world, camera


def create_material_showcase_scene():
    sphere_centers = []
    sphere_radii = []
    material_types = []
    albedos = []
    fuzzes = []
    refractive_indices = []

    # Ground sphere
    sphere_centers.append(torch.tensor([0, -100.5, -1], device=device))
    sphere_radii.append(100)
    material_types.append(MaterialType.Lambertian)
    albedos.append(torch.tensor([0.5, 0.8, 0.3], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    # Center glass sphere
    sphere_centers.append(torch.tensor([0, 0, -1], device=device))
    sphere_radii.append(0.5)
    material_types.append(MaterialType.Dielectric)
    albedos.append(torch.tensor([1.0, 1.0, 1.0], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(1.5)

    # Left metallic sphere
    sphere_centers.append(torch.tensor([-1.0, 0, -0.8], device=device))
    sphere_radii.append(0.4)
    material_types.append(MaterialType.Metal)
    albedos.append(torch.tensor([0.8, 0.6, 0.2], device=device))
    fuzzes.append(0.2)
    refractive_indices.append(0.0)

    # Right matte sphere
    sphere_centers.append(torch.tensor([1.0, -0.1, -0.7], device=device))
    sphere_radii.append(0.3)
    material_types.append(MaterialType.Lambertian)
    albedos.append(torch.tensor([0.7, 0.3, 0.3], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    # Small floating glass sphere
    sphere_centers.append(torch.tensor([0.3, 0.3, -0.5], device=device))
    sphere_radii.append(0.15)
    material_types.append(MaterialType.Dielectric)
    albedos.append(torch.tensor([1.0, 1.0, 1.0], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(1.5)

    # Convert lists to tensors
    sphere_centers = torch.stack(sphere_centers)
    sphere_radii = torch.tensor(sphere_radii, device=device)
    material_types = torch.tensor(material_types, device=device)
    albedos = torch.stack(albedos)
    fuzzes = torch.tensor(fuzzes, device=device)
    refractive_indices = torch.tensor(refractive_indices, device=device)

    world = SphereList(
        centers=sphere_centers,
        radii=sphere_radii,
        material_types=material_types,
        albedos=albedos,
        fuzzes=fuzzes,
        refractive_indices=refractive_indices,
    )

    camera = Camera(
        image_width=1080,
        samples_per_pixel=50,
        aspect_ratio=16.0 / 9.0,
        max_depth=50,
        vfov=90,
        look_from=torch.tensor([0, 0, 0], dtype=torch.float32, device=device),
        look_at=torch.tensor([0, 0, -1], dtype=torch.float32, device=device),
        vup=torch.tensor([0, 1, 0], dtype=torch.float32, device=device),
        defocus_angle=0.0,
        focus_dist=1.0,
        batch_size=50_000,
    )

    return world, camera


def create_cornell_box_scene(max_depth):
    sphere_centers = []
    sphere_radii = []
    material_types = []
    albedos = []
    fuzzes = []
    refractive_indices = []

    # Red wall (left)
    sphere_centers.append(torch.tensor([-101, 0, 0], device=device))
    sphere_radii.append(100)
    material_types.append(MaterialType.Lambertian)
    albedos.append(torch.tensor([0.75, 0.25, 0.25], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    # Green wall (right)
    sphere_centers.append(torch.tensor([101, 0, 0], device=device))
    sphere_radii.append(100)
    material_types.append(MaterialType.Lambertian)
    albedos.append(torch.tensor([0.25, 0.75, 0.25], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    # White walls (top, bottom, back)
    for pos in [(0, 101, 0), (0, -101, 0), (0, 0, -101)]:
        sphere_centers.append(torch.tensor(pos, device=device))
        sphere_radii.append(100)
        material_types.append(MaterialType.Lambertian)
        albedos.append(torch.tensor([0.75, 0.75, 0.75], device=device))
        fuzzes.append(0.0)
        refractive_indices.append(0.0)

    # Add two spheres for more interesting scene
    # Glass sphere
    sphere_centers.append(torch.tensor([-0.5, -0.7, -0.5], device=device))
    sphere_radii.append(0.3)
    material_types.append(MaterialType.Dielectric)
    albedos.append(torch.tensor([1.0, 1.0, 1.0], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(1.5)

    # Metal sphere
    sphere_centers.append(torch.tensor([0.5, -0.7, 0.5], device=device))
    sphere_radii.append(0.3)
    material_types.append(MaterialType.Metal)
    albedos.append(torch.tensor([0.8, 0.8, 0.8], device=device))
    fuzzes.append(0.1)
    refractive_indices.append(0.0)

    # Convert lists to tensors
    sphere_centers = torch.stack(sphere_centers)
    sphere_radii = torch.tensor(sphere_radii, device=device)
    material_types = torch.tensor(material_types, device=device)
    albedos = torch.stack(albedos)
    fuzzes = torch.tensor(fuzzes, device=device)
    refractive_indices = torch.tensor(refractive_indices, device=device)

    world = SphereList(
        centers=sphere_centers,
        radii=sphere_radii,
        material_types=material_types,
        albedos=albedos,
        fuzzes=fuzzes,
        refractive_indices=refractive_indices,
    )

    camera = Camera(
        image_width=1080,
        samples_per_pixel=20,
        aspect_ratio=1.0,  # Square aspect ratio for traditional Cornell box
        max_depth=max_depth,
        vfov=40,  # Narrower field of view
        look_from=torch.tensor([0, 0, 4.5], dtype=torch.float32, device=device),
        look_at=torch.tensor([0, 0, 0], dtype=torch.float32, device=device),
        vup=torch.tensor([0, 1, 0], dtype=torch.float32, device=device),
        defocus_angle=0.0,
        focus_dist=4.5,
        batch_size=50_000,
    )

    return world, camera


def main():
    # Choose device
    print(f'Using device: {device}')

    # Define scenes to render
    scenes = {'random_spheres': create_random_spheres_scene}

    # Render all scenes
    for scene_name, scene_func in scenes.items():
        print(f'Rendering {scene_name}...')
        world, camera = scene_func()
        image = camera.render(world)
        image.save(f'image_{scene_name}.png')


if __name__ == '__main__':
    main()
