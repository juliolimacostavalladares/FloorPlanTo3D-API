"""
Motor de Texture & Light Baking 3D Fotorealista com Mapas PBR (glTF 2.0 / .GLB).
Gera malhas com mapeamento UV dimensionalmente correto por metro,
associa texturas reais de alta resolução (Diffuse, Normal Map e Roughness),
e calcula Oclusão Ambiental Física (AO) por raytracing hemisférico
baked nas cores de vértice de toda a residência.
"""

import os
import math
import json
import numpy as np
import trimesh
from PIL import Image

TEXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")


def create_tiled_box(extents, translation, uv_scale=1.0):
    """
    Cria uma caixa com 24 vértices e 12 triângulos (4 vértices por face)
    com normais nítidas e coordenadas UV projetadas dimensionalmente por metro.
    """
    sx, sy, sz = extents
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    px, py, pz = translation

    verts = []
    uvs = []
    faces = []
    normals = []

    # 6 faces: (v0, v1, v2, v3, normal, uv_axes)
    face_defs = [
        # Top (+Y: Piso/Teto)
        ([-hx, hy, -hz], [hx, hy, -hz], [hx, hy, hz], [-hx, hy, hz], [0, 1, 0], (0, 2)),
        # Bottom (-Y)
        ([-hx, -hy, hz], [hx, -hy, hz], [hx, -hy, -hz], [-hx, -hy, -hz], [0, -1, 0], (0, 2)),
        # Front (+Z: Parede Frontal)
        ([-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz], [0, 0, 1], (0, 1)),
        # Back (-Z: Parede Traseira)
        ([hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz], [0, 0, -1], (0, 1)),
        # Right (+X: Parede Lateral)
        ([hx, -hy, hz], [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz], [1, 0, 0], (2, 1)),
        # Left (-X: Parede Lateral)
        ([-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz], [-1, 0, 0], (2, 1)),
    ]

    for v0, v1, v2, v3, norm, uv_axes in face_defs:
        idx = len(verts)
        v_list = [v0, v1, v2, v3]
        for v in v_list:
            verts.append([v[0] + px, v[1] + py, v[2] + pz])
            normals.append(norm)
            u = (v[uv_axes[0]] + (px if uv_axes[0] == 0 else (py if uv_axes[0] == 1 else pz))) * uv_scale
            w = (v[uv_axes[1]] + (px if uv_axes[1] == 0 else (py if uv_axes[1] == 1 else pz))) * uv_scale
            uvs.append([u, w])

        faces.append([idx, idx + 1, idx + 2])
        faces.append([idx, idx + 2, idx + 3])

    return (
        np.array(verts, dtype=np.float32),
        np.array(faces, dtype=np.int32),
        np.array(normals, dtype=np.float32),
        np.array(uvs, dtype=np.float32)
    )


def merge_boxes(box_list, uv_scale=1.0):
    """Funde uma lista de especificações de caixa (extents, pos) em uma única malha Trimesh."""
    if not box_list:
        return None

    all_verts = []
    all_faces = []
    all_normals = []
    all_uvs = []
    offset = 0

    for extents, pos in box_list:
        v, f, n, uv = create_tiled_box(extents, pos, uv_scale=uv_scale)
        all_verts.append(v)
        all_faces.append(f + offset)
        all_normals.append(n)
        all_uvs.append(uv)
        offset += len(v)

    combined_verts = np.vstack(all_verts)
    combined_faces = np.vstack(all_faces)
    combined_normals = np.vstack(all_normals)
    combined_uvs = np.vstack(all_uvs)

    return combined_verts, combined_faces, combined_normals, combined_uvs


def load_pbr_material(name, diff_name, norm_name=None, roughness=0.5, metalness=0.0):
    """Carrega mapas PNG da pasta textures e constrói um PBRMaterial glTF 2.0."""
    diff_path = os.path.join(TEXTURES_DIR, diff_name)
    if os.path.exists(diff_path):
        diff_img = Image.open(diff_path).convert('RGB')
    else:
        diff_img = Image.new('RGB', (256, 256), (220, 220, 220))

    norm_img = None
    if norm_name:
        norm_path = os.path.join(TEXTURES_DIR, norm_name)
        if os.path.exists(norm_path):
            norm_img = Image.open(norm_path).convert('RGB')

    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorTexture=diff_img,
        normalTexture=norm_img,
        roughnessFactor=roughness,
        metallicFactor=metalness
    )


def compute_ao_for_mesh(target_verts, target_normals, scene_mesh, num_samples=24):
    """
    Calcula oclusão de luz física hemisférica para os vértices da malha
    contra todos os elementos obstrutores da casa.
    """
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(scene_mesh)
    num_verts = len(target_verts)
    ao_values = np.ones(num_verts, dtype=np.float32)

    for i in range(num_verts):
        v = target_verts[i]
        n = target_normals[i]
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
        # Sombra de contato suave e realista (mínimo 25% de luz ambiente nos cantos)
        ao_values[i] = max(0.25, 1.0 - occluded_ratio * 0.78)

    ao_rgb = np.zeros((num_verts, 4), dtype=np.uint8)
    c_bytes = (ao_values * 255).astype(np.uint8)
    ao_rgb[..., 0] = c_bytes
    ao_rgb[..., 1] = c_bytes
    ao_rgb[..., 2] = c_bytes
    ao_rgb[..., 3] = 255
    return ao_rgb


def bake_house_to_glb(plan_data=None, output_glb_path="baked_house.glb", compute_ao=True, num_samples=24, plan_json_path=None):
    """
    Monta a casa completa agrupando elementos por categoria de material PBR:
    - Pisos em Madeira (Laminado)
    - Pisos em Porcelanato / Mármore
    - Paredes Internas (Gesso / Tinta Acetinada)
    - Paredes Externas / Fachada
    - Portas de Madeira
    - Vidros de Janela
    - Mobiliário (Camas, Sofás, Mesas, Sanitários)

    Aplica texturas reais 1024x1024 com Normal Maps, calcula Ambient Occlusion física
    por raytracing, e exporta cena glTF 2.0 / .GLB completa e realista.
    """
    if plan_data is None and plan_json_path:
        with open(plan_json_path, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
    elif isinstance(plan_data, str) and os.path.exists(plan_data):
        with open(plan_data, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
    elif plan_data is None:
        raise ValueError("plan_data ou plan_json_path deve ser fornecido")

    # Dicionário de caixas por categoria semântica
    categories = {
        'floor_wood': [],
        'floor_porcelain': [],
        'wall_interior': [],
        'wall_exterior': [],
        'door_wood': [],
        'window_glass': [],
        'furniture_bed': [],
        'furniture_sofa': [],
        'furniture_table': [],
        'furniture_toilet': [],
        'furniture_generic': []
    }

    # 1. Pisos
    for f in plan_data.get('floors', []):
        size = f['size']
        pos = f['position']
        tipo = f.get('tipo', '').lower()
        if 'madeira' in tipo or 'laminado' in tipo:
            categories['floor_wood'].append((size, pos))
        else:
            categories['floor_porcelain'].append((size, pos))

    # 2. Paredes, Portas e Janelas
    for elem in plan_data.get('elements_3d', []):
        size = elem['size']
        pos = elem['position']
        etype = elem.get('type')
        is_ext = elem.get('is_exterior', False)

        if etype == 'wall':
            if is_ext:
                categories['wall_exterior'].append((size, pos))
            else:
                categories['wall_interior'].append((size, pos))
        elif etype == 'door':
            categories['door_wood'].append((size, pos))
        elif etype == 'window':
            categories['window_glass'].append((size, pos))
        else:
            categories['wall_interior'].append((size, pos))

    # 3. Mobiliário
    for furn in plan_data.get('furniture', []):
        size = furn['size']
        pos = furn['position']
        tipo = furn.get('tipo', '').lower()
        if 'cama' in tipo:
            categories['furniture_bed'].append((size, pos))
        elif 'sofa' in tipo:
            categories['furniture_sofa'].append((size, pos))
        elif 'sanitario' in tipo or 'vaso' in tipo:
            categories['furniture_toilet'].append((size, pos))
        elif 'pia' in tipo or 'balcao' in tipo or 'mesa' in tipo:
            categories['furniture_table'].append((size, pos))
        else:
            categories['furniture_generic'].append((size, pos))

    # Construção das malhas simples para o intersector global de AO
    raw_meshes_for_ao = []
    for cat_name, boxes in categories.items():
        for size, pos in boxes:
            b = trimesh.creation.box(extents=size)
            b.apply_translation(pos)
            raw_meshes_for_ao.append(b)

    if not raw_meshes_for_ao:
        raise ValueError("Nenhuma geometria encontrada na planta.")

    unified_scene_mesh = trimesh.util.concatenate(raw_meshes_for_ao)

    # Materiais PBR com texturas reais
    materials = {
        'floor_wood': load_pbr_material('WoodParquet_PBR', 'floor_wood_diffuse.png', 'floor_wood_normal.png', roughness=0.35),
        'floor_porcelain': load_pbr_material('PorcelainMarble_PBR', 'floor_porcelain_diffuse.png', 'floor_porcelain_normal.png', roughness=0.15),
        'wall_interior': load_pbr_material('WallInterior_PBR', 'wall_interior_diffuse.png', 'wall_interior_normal.png', roughness=0.82),
        'wall_exterior': load_pbr_material('WallExterior_PBR', 'wall_exterior_diffuse.png', 'wall_exterior_normal.png', roughness=0.90),
        'door_wood': load_pbr_material('DoorWood_PBR', 'door_wood_diffuse.png', 'door_wood_normal.png', roughness=0.42),
        'window_glass': trimesh.visual.material.PBRMaterial(
            name='WindowGlass_PBR',
            baseColorFactor=[0.85, 0.95, 1.0, 0.35],
            roughnessFactor=0.05,
            metallicFactor=0.10
        ),
        'furniture_bed': load_pbr_material('FurnitureBed_PBR', 'furniture_bed_diffuse.png', 'furniture_bed_normal.png', roughness=0.75),
        'furniture_sofa': load_pbr_material('FurnitureSofa_PBR', 'fabric_sofa_diffuse.png', 'fabric_sofa_normal.png', roughness=0.85),
        'furniture_table': load_pbr_material('FurnitureTable_PBR', 'furniture_table_diffuse.png', 'furniture_table_normal.png', roughness=0.45),
        'furniture_toilet': load_pbr_material('FurnitureToilet_PBR', 'furniture_toilet_diffuse.png', 'furniture_toilet_normal.png', roughness=0.15),
        'furniture_generic': load_pbr_material('FurnitureGeneric_PBR', 'furniture_tv_shelf_diffuse.png', 'furniture_tv_shelf_normal.png', roughness=0.55),
    }

    # Escala de UV (repetição por metro) para cada categoria
    uv_scales = {
        'floor_wood': 1.0,        # 1 repetição por metro
        'floor_porcelain': 0.8,   # Placas grandes de porcelanato 60x60
        'wall_interior': 0.5,     # Tinta acetinada suave
        'wall_exterior': 0.6,     # Textura acrílica
        'door_wood': 1.2,
        'window_glass': 1.0,
        'furniture_bed': 0.8,
        'furniture_sofa': 1.5,
        'furniture_table': 1.0,
        'furniture_toilet': 1.0,
        'furniture_generic': 1.0
    }

    # Criação da cena com as sub-malhas texturizadas
    final_scene = trimesh.Scene()
    total_vertices = 0
    total_faces = 0

    for cat_name, boxes in categories.items():
        if not boxes:
            continue

        uv_scale = uv_scales.get(cat_name, 1.0)
        res = merge_boxes(boxes, uv_scale=uv_scale)
        if res is None:
            continue

        verts, faces, normals, uvs = res
        mat = materials[cat_name]

        vis = trimesh.visual.TextureVisuals(uv=uvs, material=mat)

        if compute_ao:
            ao_colors = compute_ao_for_mesh(verts, normals, unified_scene_mesh, num_samples=num_samples)
            vis.vertex_colors = ao_colors

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
            visual=vis,
            process=False
        )

        final_scene.add_geometry(mesh, node_name=cat_name)
        total_vertices += len(verts)
        total_faces += len(faces)

    os.makedirs(os.path.dirname(os.path.abspath(output_glb_path)), exist_ok=True)
    glb_bytes = trimesh.exchange.gltf.export_glb(final_scene)
    with open(output_glb_path, 'wb') as f:
        f.write(glb_bytes)

    return output_glb_path, len(glb_bytes), total_vertices, total_faces


if __name__ == '__main__':
    import sys
    test_json = "saved_plans/plan_20260912_045448.json"
    out_glb = "saved_plans/plan_20260912_045448_baked.glb"
    if len(sys.argv) > 1:
        test_json = sys.argv[1]
    if len(sys.argv) > 2:
        out_glb = sys.argv[2]

    out, size, v_count, f_count = bake_house_to_glb(
        plan_json_path=test_json,
        output_glb_path=out_glb,
        compute_ao=True,
        num_samples=16
    )
    print(f"Bake finalizado com sucesso! GLB: {out} ({size / 1024:.1f} KB, {v_count} verts, {f_count} faces)")
