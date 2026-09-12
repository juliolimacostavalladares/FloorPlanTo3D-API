#!/usr/bin/env python3
"""
Gera conjuntos de texturas PBR fotorrealistas em alta resolução (1024x1024):
- Albedo / Diffuse (PNG)
- Normal Map em espaço tangente com profundidade real via Sobel (PNG)
- Roughness Map (PNG)

Salva em FloorPlan3D-Viewer/textures e FloorPlanTo3D-API/textures
"""

import os
import math
import numpy as np
from PIL import Image, ImageFilter

SIZE = 1024

def height_to_normal_map(height, strength=2.5):
    """Calcula mapa de normal em espaço tangente usando filtro de convolução Sobel."""
    h = height.astype(np.float32)
    
    # Wrap padding for seamless tiling
    pad = np.pad(h, 1, mode='wrap')
    
    # Convolve
    dx = (
        pad[:-2, 2:] + 2 * pad[1:-1, 2:] + pad[2:, 2:] -
        pad[:-2, :-2] - 2 * pad[1:-1, :-2] - pad[2:, :-2]
    ) / 8.0
    dy = (
        pad[2:, :-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:] -
        pad[:-2, :-2] - 2 * pad[:-2, 1:-1] - pad[:-2, 2:]
    ) / 8.0
    
    dx = -dx * strength
    dy = -dy * strength
    dz = np.ones_like(dx)
    
    length = np.sqrt(dx * dx + dy * dy + dz * dz)
    nx = dx / length
    ny = dy / length
    nz = dz / length
    
    normal_map = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    normal_map[..., 0] = ((nx * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    normal_map[..., 1] = ((ny * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    normal_map[..., 2] = ((nz * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(normal_map)


def generate_wood_parquet():
    """Gera piso de madeira nobre / taco laminado com réguas, veios e chanfros."""
    rows = 8
    cols_per_row = 4
    row_height = SIZE // rows
    
    diff = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    height = np.zeros((SIZE, SIZE), dtype=np.float32)
    rough = np.zeros((SIZE, SIZE), dtype=np.float32)
    
    xx, yy = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    wood_grain = np.sin(xx * 0.12 + np.sin(yy * 0.05) * 6.0) * 0.5 + 0.5
    wood_grain += (np.sin(xx * 0.45) * 0.5 + 0.5) * 0.25
    
    np.random.seed(42)
    for r in range(rows):
        y_start = r * row_height
        y_end = (r + 1) * row_height
        shift = (r % 2) * (SIZE // (cols_per_row * 2))
        
        col_width = SIZE // cols_per_row
        for c in range(-1, cols_per_row + 2):
            x_start = c * col_width + shift
            x_end = (c + 1) * col_width + shift
            
            hue_var = np.random.uniform(0.85, 1.15)
            r_base = 195.0 * hue_var
            g_base = 142.0 * hue_var
            b_base = 92.0 * hue_var
            
            xs = max(0, x_start)
            xe = min(SIZE, x_end)
            if xs >= xe:
                continue
                
            plank_grain = wood_grain[y_start:y_end, xs:xe]
            
            diff[y_start:y_end, xs:xe, 0] = np.clip(r_base + (plank_grain - 0.5) * 35.0, 0, 255)
            diff[y_start:y_end, xs:xe, 1] = np.clip(g_base + (plank_grain - 0.5) * 28.0, 0, 255)
            diff[y_start:y_end, xs:xe, 2] = np.clip(b_base + (plank_grain - 0.5) * 20.0, 0, 255)
            
            height[y_start:y_end, xs:xe] = 0.5 + plank_grain * 0.15
            
            bevel = 4
            height[y_start:min(y_start + bevel, y_end), xs:xe] *= 0.2
            height[max(y_start, y_end - bevel):y_end, xs:xe] *= 0.2
            if xs == x_start:
                height[y_start:y_end, xs:min(xs + bevel, xe)] *= 0.2
            if xe == x_end:
                height[y_start:y_end, max(xs, xe - bevel):xe] *= 0.2
                
            diff[y_start:y_start + 2, xs:xe] *= 0.6
            diff[y_end - 2:y_end, xs:xe] *= 0.6
            if xs == x_start:
                diff[y_start:y_end, xs:xs + 2] *= 0.6
            if xe == x_end:
                diff[y_start:y_end, xe - 2:xe] *= 0.6
                
            rough[y_start:y_end, xs:xe] = 0.32 + plank_grain * 0.08
            rough[y_start:y_start + 3, xs:xe] = 0.75
            rough[y_end - 3:y_end, xs:xe] = 0.75

    diff_img = Image.fromarray(diff.astype(np.uint8))
    norm_img = height_to_normal_map(height, strength=4.5)
    rough_img = Image.fromarray((rough * 255).astype(np.uint8))
    return diff_img, norm_img, rough_img


def generate_porcelain_marble():
    """Gera porcelanato polido 60x60 tipo Calacatta com veios de mármore e rejunte."""
    tiles = 4
    tile_size = SIZE // tiles
    grout = 4
    
    diff = np.full((SIZE, SIZE, 3), 242.0, dtype=np.float32)
    height = np.full((SIZE, SIZE), 0.85, dtype=np.float32)
    rough = np.full((SIZE, SIZE), 0.12, dtype=np.float32)
    
    xx, yy = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    np.random.seed(101)
    
    v1 = np.sin(xx * 0.008 + yy * 0.012 + np.sin(xx * 0.03 + yy * 0.02) * 3.0)
    v2 = np.cos(xx * 0.015 - yy * 0.009 + np.cos(yy * 0.04) * 2.5)
    vein_mask = np.exp(-((v1 + v2 * 0.5) ** 2) / 0.08)
    
    v3 = np.sin(xx * 0.025 + yy * 0.02 + np.sin(xx * 0.08) * 1.5)
    fine_vein = np.exp(-(v3 ** 2) / 0.03) * 0.6
    
    tot_vein = np.clip(vein_mask + fine_vein, 0, 1)
    
    diff[..., 0] = diff[..., 0] * (1.0 - tot_vein * 0.42) + tot_vein * 110.0
    diff[..., 1] = diff[..., 1] * (1.0 - tot_vein * 0.40) + tot_vein * 115.0
    diff[..., 2] = diff[..., 2] * (1.0 - tot_vein * 0.38) + tot_vein * 125.0
    
    for t in range(tiles + 1):
        pos = t * tile_size
        p_start = max(0, pos - grout // 2)
        p_end = min(SIZE, pos + grout // 2)
        
        diff[p_start:p_end, :, 0] = 190.0
        diff[p_start:p_end, :, 1] = 188.0
        diff[p_start:p_end, :, 2] = 185.0
        height[p_start:p_end, :] = 0.15
        rough[p_start:p_end, :] = 0.88
        
        diff[:, p_start:p_end, 0] = 190.0
        diff[:, p_start:p_end, 1] = 188.0
        diff[:, p_start:p_end, 2] = 185.0
        height[:, p_start:p_end] = 0.15
        rough[:, p_start:p_end] = 0.88

    diff_img = Image.fromarray(diff.astype(np.uint8))
    norm_img = height_to_normal_map(height, strength=6.0)
    rough_img = Image.fromarray((rough * 255).astype(np.uint8))
    return diff_img, norm_img, rough_img


def generate_wall_interior():
    """Gera parede interna em tinta acetinada / drywall com micro-relevo de rolo."""
    diff = np.full((SIZE, SIZE, 3), 246.0, dtype=np.float32)
    diff[..., 0] = 248.0
    diff[..., 1] = 246.0
    diff[..., 2] = 242.0
    
    np.random.seed(77)
    noise = np.random.normal(0, 1.0, (SIZE, SIZE))
    im_noise = Image.fromarray(((noise + 3.0) / 6.0 * 255).clip(0, 255).astype(np.uint8))
    im_noise = im_noise.filter(ImageFilter.GaussianBlur(radius=1.2))
    stipple = np.array(im_noise).astype(np.float32) / 255.0
    
    diff = diff + (stipple[..., np.newaxis] - 0.5) * 8.0
    height = stipple
    rough = np.full((SIZE, SIZE), 0.78, dtype=np.float32) + (stipple - 0.5) * 0.05
    
    diff_img = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))
    norm_img = height_to_normal_map(height, strength=1.8)
    rough_img = Image.fromarray((np.clip(rough, 0, 1) * 255).astype(np.uint8))
    return diff_img, norm_img, rough_img


def generate_wall_exterior():
    """Gera fachada externa em reboco projetado rústico / textura acrílica."""
    diff = np.full((SIZE, SIZE, 3), 218.0, dtype=np.float32)
    diff[..., 0] = 222.0
    diff[..., 1] = 219.0
    diff[..., 2] = 212.0
    
    np.random.seed(99)
    granules = np.random.uniform(0, 1.0, (SIZE, SIZE))
    granules_img = Image.fromarray((granules * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=0.8))
    g_arr = np.array(granules_img).astype(np.float32) / 255.0
    
    coarse = Image.fromarray((np.random.uniform(0, 1.0, (SIZE, SIZE)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3.5))
    c_arr = np.array(coarse).astype(np.float32) / 255.0
    
    tot = g_arr * 0.65 + c_arr * 0.35
    diff = diff + (tot[..., np.newaxis] - 0.5) * 32.0
    height = tot
    rough = np.full((SIZE, SIZE), 0.90, dtype=np.float32)
    
    diff_img = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))
    norm_img = height_to_normal_map(height, strength=3.8)
    rough_img = Image.fromarray((rough * 255).astype(np.uint8))
    return diff_img, norm_img, rough_img


def generate_door_wood():
    """Gera porta moderna de madeira nobre (freijó / nogueira) frisada."""
    diff = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    height = np.full((SIZE, SIZE), 0.7, dtype=np.float32)
    rough = np.full((SIZE, SIZE), 0.40, dtype=np.float32)
    
    xx, yy = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    grain = np.sin(yy * 0.15 + np.sin(xx * 0.02) * 8.0) * 0.5 + 0.5
    grain += (np.sin(yy * 0.6) * 0.5 + 0.5) * 0.2
    
    r_base = 152.0
    g_base = 88.0
    b_base = 45.0
    
    diff[..., 0] = r_base + (grain - 0.5) * 38.0
    diff[..., 1] = g_base + (grain - 0.5) * 26.0
    diff[..., 2] = b_base + (grain - 0.5) * 16.0
    
    frisos_y = [200, 400, 600, 800]
    for fy in frisos_y:
        diff[fy-4:fy+4, :] *= 0.4
        height[fy-4:fy+4, :] = 0.1
        rough[fy-4:fy+4, :] = 0.8
        
    diff_img = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))
    norm_img = height_to_normal_map(height, strength=5.0)
    rough_img = Image.fromarray((rough * 255).astype(np.uint8))
    return diff_img, norm_img, rough_img


def generate_sofa_fabric():
    """Gera tecido de linho / boucle escandinavo para estofados e sofás."""
    diff = np.full((SIZE, SIZE, 3), 75.0, dtype=np.float32)
    diff[..., 0] = 82.0
    diff[..., 1] = 92.0
    diff[..., 2] = 105.0
    
    xx, yy = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    weave_x = np.sin(xx * 0.6) * 0.5 + 0.5
    weave_y = np.sin(yy * 0.6) * 0.5 + 0.5
    weave = (weave_x * weave_y)
    
    diff = diff + (weave[..., np.newaxis] - 0.5) * 24.0
    height = weave
    rough = np.full((SIZE, SIZE), 0.86, dtype=np.float32)
    
    diff_img = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))
    norm_img = height_to_normal_map(height, strength=3.0)
    rough_img = Image.fromarray((rough * 255).astype(np.uint8))
    return diff_img, norm_img, rough_img


def convert_unity_furniture_textures(out_dirs):
    """Converte e otimiza texturas 4K da pasta da Unity para 1024x1024 PNG."""
    unity_dir = "/Users/macbookpro/Desktop/home/FloorPlanTo3D-unityClient/Assets/AssetStoreAssets/Chalet style furniture/Textures"
    if not os.path.exists(unity_dir):
        return
        
    pairs = [
        ("bed_d.tga", "bed_n.tga", "furniture_bed"),
        ("table_1_d.tga", "table_1_n.tga", "furniture_table"),
        ("chair_d.tga", "chair_n.tga", "furniture_chair"),
        ("toilet_d.tga", "toilet_n.tga", "furniture_toilet"),
        ("tv_shelf_d.tga", "tv_shelf_n.tga", "furniture_tv_shelf"),
    ]
    
    for d_name, n_name, prefix in pairs:
        d_path = os.path.join(unity_dir, d_name)
        n_path = os.path.join(unity_dir, n_name)
        
        if os.path.exists(d_path):
            try:
                img_d = Image.open(d_path).convert('RGB')
                img_d = img_d.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
                for out_dir in out_dirs:
                    img_d.save(os.path.join(out_dir, f"{prefix}_diffuse.png"), optimize=True)
            except Exception as e:
                print(f"Erro ao converter {d_name}: {e}")
                
        if os.path.exists(n_path):
            try:
                img_n = Image.open(n_path).convert('RGB')
                img_n = img_n.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
                for out_dir in out_dirs:
                    img_n.save(os.path.join(out_dir, f"{prefix}_normal.png"), optimize=True)
            except Exception as e:
                print(f"Erro ao converter {n_name}: {e}")


def main():
    out_dirs = [
        "/Users/macbookpro/Desktop/home/FloorPlan3D-Viewer/textures",
        "/Users/macbookpro/Desktop/home/FloorPlanTo3D-API/textures"
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        
    generators = [
        ("floor_wood", generate_wood_parquet),
        ("floor_porcelain", generate_porcelain_marble),
        ("wall_interior", generate_wall_interior),
        ("wall_exterior", generate_wall_exterior),
        ("door_wood", generate_door_wood),
        ("fabric_sofa", generate_sofa_fabric)
    ]
    
    print("Gerando conjuntos PBR fotorrealistas em 1024x1024...")
    for name, gen in generators:
        print(f" -> Criando mapas PBR para: {name}")
        diff, norm, rough = gen()
        for d in out_dirs:
            diff.save(os.path.join(d, f"{name}_diffuse.png"), optimize=True)
            norm.save(os.path.join(d, f"{name}_normal.png"), optimize=True)
            rough.save(os.path.join(d, f"{name}_roughness.png"), optimize=True)
            
    print("Convertendo texturas de mobiliário da Unity...")
    convert_unity_furniture_textures(out_dirs)
    print("Todas as texturas PBR foram geradas com sucesso!")

if __name__ == "__main__":
    main()
