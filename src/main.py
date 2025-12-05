import random

import torch
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from camera import Camera
from config import device
from materials import MaterialType
from sphere import SphereInfo, SphereList


def create_random_spheres_scene():
    sphere_info = SphereInfo()

    # Ground sphere
    sphere_info.add(
        sphere_center=[0, -1000, 0],
        sphere_radius=1000,
        material_type=MaterialType.Lambertian,
        albedo=[0.5, 0.5, 0.5],
    )

    # Random small spheres
    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = [a + 0.9 * random.random(), 0.2, b + 0.9 * random.random()]
            if (torch.tensor(center) - torch.tensor([4, 0.2, 0])).norm() > 0.9:
                fuzz = 0
                refractive_index = 0
                if choose_mat < 0.8:
                    # Diffuse
                    material_type = MaterialType.Lambertian
                    albedo = [random.random() * random.random() for _ in range(3)]
                elif choose_mat < 0.95:
                    # Metal
                    material_type = MaterialType.Metal
                    albedo = [random.random() * 0.5 + 0.5 for _ in range(3)]
                    fuzz = random.uniform(0, 0.5)
                else:
                    # Glass
                    material_type = MaterialType.Dielectric
                    albedo = [0, 0, 0]
                    refractive_index = 1.5
                sphere_info.add(center, 0.2, material_type, albedo, fuzz, refractive_index)

    # Three larger spheres
    sphere_info.add(
        sphere_center=[0, 1, 0],
        sphere_radius=1,
        material_type=MaterialType.Dielectric,
        albedo=[0, 0, 0],
        refractive_index=1.5,
    )

    sphere_info.add(
        sphere_center=[-4, 1, 0],
        sphere_radius=1,
        material_type=MaterialType.Lambertian,
        albedo=[0.4, 0.2, 0.1],
    )

    sphere_info.add(
        sphere_center=[4, 1, 0],
        sphere_radius=1,
        material_type=MaterialType.Metal,
        albedo=[0.7, 0.6, 0.5],
    )

    # Convert lists to tensors
    world = SphereList(*sphere_info.pack())

    camera = Camera(
        image_width=320,
        samples_per_pixel=50,
        aspect_ratio=16.0 / 9.0,
        max_depth=50,
        vfov=20,
        look_from=torch.tensor([13, 2, 3], dtype=torch.float32),
        look_at=torch.tensor([0, 0, 0], dtype=torch.float32),
        vup=torch.tensor([0, 1, 0], dtype=torch.float32),
        defocus_angle=0.6,
        focus_dist=10.0,
        batch_size=50_000,
    )

    return world, camera


def create_material_showcase_scene():
    sphere_info = SphereInfo()

    # Ground sphere
    sphere_info.add(
        sphere_center=[0, -100.5, -1],
        sphere_radius=100,
        material_type=MaterialType.Lambertian,
        albedo=[0.5, 0.8, 0.3],
    )

    # Center glass sphere
    sphere_info.add(
        sphere_center=[0, 0, -1],
        sphere_radius=0.5,
        material_type=MaterialType.Dielectric,
        albedo=[1.0, 1.0, 1.0],
        refractive_index=1.5,
    )

    # Left metallic sphere
    sphere_info.add(
        sphere_center=[-1.0, 0, -0.8],
        sphere_radius=0.4,
        material_type=MaterialType.Metal,
        albedo=[0.8, 0.6, 0.2],
        fuzz=0.2,
    )

    # Right matte sphere
    sphere_info.add(
        sphere_center=[1.0, -0.1, -0.7],
        sphere_radius=0.3,
        material_type=MaterialType.Lambertian,
        albedo=[0.7, 0.3, 0.3],
    )

    # Small floating glass sphere
    sphere_info.add(
        sphere_center=[0.3, 0.3, -0.5],
        sphere_radius=0.15,
        material_type=MaterialType.Dielectric,
        albedo=[1.0, 1.0, 1.0],
        refractive_index=1.5,
    )

    # Convert lists to tensors
    world = SphereList(*sphere_info.pack())

    camera = Camera(
        image_width=1080,
        samples_per_pixel=50,
        aspect_ratio=16.0 / 9.0,
        max_depth=50,
        vfov=90,
        look_from=torch.tensor([0, 0, 0], dtype=torch.float32),
        look_at=torch.tensor([0, 0, -1], dtype=torch.float32),
        vup=torch.tensor([0, 1, 0], dtype=torch.float32),
        defocus_angle=0.0,
        focus_dist=1.0,
        batch_size=50_000,
    )

    return world, camera


def create_cornell_box_scene(max_depth):
    sphere_info = SphereInfo()

    # Red wall (left)
    sphere_info.add(
        sphere_center=[-101, 0, 0],
        sphere_radius=100,
        material_type=MaterialType.Lambertian,
        albedo=[0.75, 0.25, 0.25],
    )

    # Green wall (right)
    sphere_info.add(
        sphere_center=[101, 0, 0],
        sphere_radius=100,
        material_type=MaterialType.Lambertian,
        albedo=[0.25, 0.75, 0.25],
    )

    # White walls (top, bottom, back)
    for pos in [(0, 101, 0), (0, -101, 0), (0, 0, -101)]:
        sphere_info.add(
            sphere_center=pos,
            sphere_radius=100,
            material_type=MaterialType.Lambertian,
            albedo=[0.75, 0.75, 0.75],
        )

    # Add two spheres for more interesting scene
    # Glass sphere
    sphere_info.add(
        sphere_center=[-0.5, -0.7, -0.5],
        sphere_radius=0.3,
        material_type=MaterialType.Dielectric,
        albedo=[1.0, 1.0, 1.0],
        refractive_index=1.5,
    )

    # Metal sphere
    sphere_info.add(
        sphere_center=[0.5, -0.7, 0.5],
        sphere_radius=0.3,
        material_type=MaterialType.Metal,
        albedo=[0.8, 0.8, 0.8],
        fuzz=0.1,
    )

    # Convert lists to tensors
    world = SphereList(*sphere_info.pack())

    camera = Camera(
        image_width=1080,
        samples_per_pixel=20,
        aspect_ratio=1.0,  # Square aspect ratio for traditional Cornell box
        max_depth=max_depth,
        vfov=40,  # Narrower field of view
        look_from=torch.tensor([0, 0, 4.5], dtype=torch.float32),
        look_at=torch.tensor([0, 0, 0], dtype=torch.float32),
        vup=torch.tensor([0, 1, 0], dtype=torch.float32),
        defocus_angle=0.0,
        focus_dist=4.5,
        batch_size=50_000,
    )

    return world, camera


def main():
    # Choose device
    print(f'Using device: {device}')
    torch.set_default_device(device)

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
