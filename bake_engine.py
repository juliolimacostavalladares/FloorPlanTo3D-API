import os
import json
import math
import numpy as np
from PIL import Image, ImageDraw
import trimesh
import trimesh.visual.material
import xatlas


def generate_sobel_normal_map(img_gray, strength=2.5):
    """
    Gera um Normal Map no espaço tangente (RGB) a partir de uma imagem em escala de cinza (heightmap)
    utilizando convolução com filtro de Sobel 3x3.
    """
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    h, w = arr.shape

    padded = np.pad(arr, pad_width=1, mode='wrap')

    tl = padded[:-2, :-2]
    l  = padded[1:-1, :-2]
    bl = padded[2:, :-2]
    t  = padded[:-2, 1:-1]
    b  = padded[2:, 1:-1]
    tr = padded[:-2, 2:]
    r  = padded[1:-1, 2:]
    br = padded[2:, 2:]

    dX = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl)
    dY = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr)

    nx = -dX * strength
    ny = -dY * strength
    nz = np.ones_like(nx)

    length = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= length
    ny /= length
    nz /= length

    normal_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    normal_rgb[..., 0] = np.clip((nx * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    normal_rgb[..., 1] = np.clip((ny * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    normal_rgb[..., 2] = np.clip((nz * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(normal_rgb, mode='RGB')


def create_procedural_wood_textures(size=512):
    """Gera o conjunto completo de mapas PBR para assoalho de madeira."""
    albedo = Image.new('RGB', (size, size), '#b45309')
    draw = ImageDraw.Draw(albedo)

    num_planks = 8
    plank_h = size // num_planks
    for i in range(num_planks):
        y = i * plank_h
        color = '#92400e' if i % 2 == 0 else '#b45309'
        draw.rectangle([0, y, size, y + plank_h], fill=color)
        for j in range(3):
            vy = y + 8 + j * 16
            draw.line([0, vy, size, vy + 4], fill='#78350f', width=1)
        draw.line([0, y, size, y], fill='#381604', width=2)

    heightmap = albedo.convert('L')
    normal_map = generate_sobel_normal_map(heightmap, strength=3.0)

    r_arr = np.array(heightmap, dtype=np.float32) / 255.0
    r_arr = (0.22 + (1.0 - r_arr) * 0.45) * 255.0
    roughness_map = Image.fromarray(np.clip(r_arr, 0, 255).astype(np.uint8), mode='L')

    return albedo, normal_map, roughness_map


def create_procedural_marble_textures(size=512):
    """Gera o conjunto completo de mapas PBR para piso de porcelanato / mármore."""
    albedo = Image.new('RGB', (size, size), '#f8fafc')
    draw = ImageDraw.Draw(albedo)

    draw.line([30, 0, 180, 200, 380, 512], fill='#cbd5e1', width=3)
    draw.line([350, 0, 310, 280, 510, 512], fill='#cbd5e1', width=2)

    half = size // 2
    draw.rectangle([0, 0, half, half], outline='#64748b', width=3)
    draw.rectangle([half, 0, size, half], outline='#64748b', width=3)
    draw.rectangle([0, half, half, size], outline='#64748b', width=3)
    draw.rectangle([half, half, size, size], outline='#64748b', width=3)

    heightmap = albedo.convert('L')
    normal_map = generate_sobel_normal_map(heightmap, strength=1.8)

    r_arr = np.array(heightmap, dtype=np.float32) / 255.0
    r_arr = (0.12 + (1.0 - r_arr) * 0.50) * 255.0
    roughness_map = Image.fromarray(np.clip(r_arr, 0, 255).astype(np.uint8), mode='L')

    return albedo, normal_map, roughness_map


def create_procedural_wall_textures(size=256):
    """Gera o conjunto de mapas PBR para reboco fino de alvenaria."""
    np.random.seed(42)
    noise = np.random.randint(235, 255, (size, size), dtype=np.uint8)
    albedo = Image.fromarray(np.stack([noise, noise, noise], axis=-1), mode='RGB')

    heightmap = albedo.convert('L')
    normal_map = generate_sobel_normal_map(heightmap, strength=1.2)
    roughness_map = Image.new('L', (size, size), 215)

    return albedo, normal_map, roughness_map


def compute_ambient_occlusion_per_vertex(scene_mesh, num_samples=24, max_dist=3.0):
    """
    Calcula a Oclusão Ambiental Física (AO) para cada vértice da cena completa,
    projetando raios no hemisfério orientado pela normal da superfície.
    """
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(scene_mesh)
    verts = scene_mesh.vertices
    normals = scene_mesh.vertex_normals
    num_verts = len(verts)

    ao_values = np.ones(num_verts, dtype=np.float32)

    for i in range(num_verts):
        v = verts[i]
        n = normals[i]
        if np.linalg.norm(n) < 1e-4:
            continue

        rnd = np.random.randn(num_samples, 3)
        norm = np.linalg.norm(rnd, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        rnd /= norm

        dots = np.sum(rnd * n, axis=1)
        rnd[dots < 0] *= -1

        origins = v + n * 0.015 + np.zeros((num_samples, 3))
        hits = intersector.intersects_any(origins, rnd)

        occluded_ratio = np.sum(hits) / num_samples
        ao_values[i] = max(0.20, 1.0 - occluded_ratio * 0.85)

    return ao_values


def bake_house_to_glb(plan_data=None, output_glb_path="baked_house.glb", compute_ao=True, num_samples=24, plan_json_path=None):
    """
    Monta toda a casa (paredes, pisos, portas, janelas, móveis),
    calcula Ambient Occlusion física, aplica mapas PBR com Normal e Roughness,
    abre UVs com xatlas e exporta um arquivo .GLB consolidado.
    """
    if plan_data is None and plan_json_path:
        with open(plan_json_path, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
    elif isinstance(plan_data, str) and os.path.exists(plan_data):
        with open(plan_data, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
    elif plan_data is None:
        raise ValueError("plan_data ou plan_json_path deve ser fornecido")

    wood_alb, wood_nor, wood_rou = create_procedural_wood_textures()
    mrb_alb, mrb_nor, mrb_rou = create_procedural_marble_textures()
    wall_alb, wall_nor, wall_rou = create_procedural_wall_textures()

    wood_mat = trimesh.visual.material.PBRMaterial(
        name='WoodParquet_PBR',
        baseColorTexture=wood_alb,
        normalTexture=wood_nor,
        metallicFactor=0.02,
        roughnessFactor=0.35
    )

    marble_mat = trimesh.visual.material.PBRMaterial(
        name='MarbleFloor_PBR',
        baseColorTexture=mrb_alb,
        normalTexture=mrb_nor,
        metallicFactor=0.05,
        roughnessFactor=0.15
    )

    wall_mat = trimesh.visual.material.PBRMaterial(
        name='WallPlaster_PBR',
        baseColorTexture=wall_alb,
        normalTexture=wall_nor,
        metallicFactor=0.01,
        roughnessFactor=0.82
    )

    door_mat = trimesh.visual.material.PBRMaterial(
        name='DoorWood_PBR',
        baseColorFactor=[0.58, 0.25, 0.05, 1.0],
        roughnessFactor=0.45,
        metallicFactor=0.03
    )

    furniture_mat = trimesh.visual.material.PBRMaterial(
        name='Furniture_PBR',
        baseColorFactor=[0.25, 0.32, 0.42, 1.0],
        roughnessFactor=0.85,
        metallicFactor=0.05
    )

    all_meshes = []

    for f in plan_data.get('floors', []):
        sx, sy, sz = f['size']
        px, py, pz = f['position']
        tipo = f.get('tipo', 'porcelanato')
        box = trimesh.creation.box(extents=[sx, sy, sz])
        box.apply_translation([px, py, pz])
        mat = wood_mat if 'madeira' in tipo or 'laminado' in tipo else marble_mat
        box.visual.material = mat
        all_meshes.append(box)

    for elem in plan_data.get('elements_3d', []):
        sx, sy, sz = elem['size']
        px, py, pz = elem['position']
        etype = elem.get('type')
        box = trimesh.creation.box(extents=[sx, sy, sz])
        box.apply_translation([px, py, pz])

        if etype == 'wall':
            box.visual.material = wall_mat
        elif etype == 'door':
            box.visual.material = door_mat
        elif etype == 'window':
            box.visual.material = trimesh.visual.material.PBRMaterial(
                name='Glass_PBR',
                baseColorFactor=[0.85, 0.95, 1.0, 0.4],
                roughnessFactor=0.05,
                metallicFactor=0.1
            )
        else:
            box.visual.material = wall_mat
        all_meshes.append(box)

    for furn in plan_data.get('furniture', []):
        sx, sy, sz = furn['size']
        px, py, pz = furn['position']
        box = trimesh.creation.box(extents=[sx, sy, sz])
        box.apply_translation([px, py, pz])
        box.visual.material = furniture_mat
        all_meshes.append(box)

    unified = trimesh.util.concatenate(all_meshes)

    if compute_ao:
        ao_vals = compute_ambient_occlusion_per_vertex(unified, num_samples=num_samples)
        ao_rgb = np.zeros((len(ao_vals), 4), dtype=np.uint8)
        c_bytes = (ao_vals * 255).astype(np.uint8)
        ao_rgb[..., 0] = c_bytes
        ao_rgb[..., 1] = c_bytes
        ao_rgb[..., 2] = c_bytes
        ao_rgb[..., 3] = 255
        unified.visual.vertex_colors = ao_rgb

    vmapping, indices, uvs = xatlas.parametrize(unified.vertices, unified.faces)

    unwrapped_mesh = trimesh.Trimesh(
        vertices=unified.vertices[vmapping],
        faces=indices,
        vertex_normals=unified.vertex_normals[vmapping] if unified.vertex_normals is not None else None,
        process=False
    )
    if compute_ao and unified.visual.vertex_colors is not None and len(unified.visual.vertex_colors) == len(unified.vertices):
        unwrapped_mesh.visual.vertex_colors = unified.visual.vertex_colors[vmapping]
    unwrapped_mesh.visual.uv = uvs

    os.makedirs(os.path.dirname(os.path.abspath(output_glb_path)), exist_ok=True)
    glb_bytes = trimesh.exchange.gltf.export_glb(unwrapped_mesh)
    with open(output_glb_path, 'wb') as f:
        f.write(glb_bytes)

    return output_glb_path, len(glb_bytes), len(unwrapped_mesh.vertices), len(unwrapped_mesh.faces)
