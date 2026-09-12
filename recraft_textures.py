"""
Módulo de Geração de Texturas PBR Arquitetônicas com Recraft AI.
Chama a API do Recraft para gerar mapas de Albedo/Difuso fotorrealistas em 1024x1024
e calcula automaticamente os mapas de Normal (espaço tangente via Sobel) e Rugosidade.
"""

import os
import io
import requests
import numpy as np
from PIL import Image

API_DIR = os.path.dirname(os.path.abspath(__file__))
API_TEXTURES_DIR = os.path.join(API_DIR, "textures")
VIEWER_TEXTURES_DIR = os.path.abspath(os.path.join(API_DIR, "..", "FloorPlan3D-Viewer", "textures"))

os.makedirs(API_TEXTURES_DIR, exist_ok=True)
os.makedirs(VIEWER_TEXTURES_DIR, exist_ok=True)

SIZE = 1024

DEFAULT_PROMPTS = {
    'floor_wood': (
        "Seamless texture of modern luxury oak wood parquet flooring planks, "
        "top-down flat perspective, architectural material, high resolution 8k, "
        "photorealistic, diffuse texture map, natural amber wood grain, subtle bevel joints"
    ),
    'floor_porcelain': (
        "Seamless texture of luxury Italian polished Calacatta white marble tiles 60x60cm "
        "with delicate grey and gold veins, clean grout joints, top-down flat orthographic view, "
        "architectural material, 8k resolution, photorealistic, glossy surface"
    ),
    'wall_interior': (
        "Seamless texture of modern interior drywall plaster wall painted in eggshell off-white, "
        "subtle fine roller texture, clean architectural surface, top-down flat lighting, "
        "8k resolution, photorealistic"
    ),
    'wall_exterior': (
        "Seamless texture of modern architectural facade stucco and light grey concrete plaster finish, "
        "rustic sand aggregate texture, exterior wall cladding, flat orthographic view, "
        "8k resolution, photorealistic"
    ),
    'door_wood': (
        "Seamless texture of luxury walnut wood door paneling with vertical grain "
        "and modern horizontal grooved slats, flat orthographic view, 8k resolution, photorealistic"
    ),
    'fabric_sofa': (
        "Seamless texture of Scandinavian woven linen upholstery fabric in slate blue-grey melange, "
        "close-up textile weave, flat orthographic lighting, 8k resolution, photorealistic"
    )
}

NORMAL_STRENGTHS = {
    'floor_wood': 4.0,
    'floor_porcelain': 4.5,
    'wall_interior': 1.6,
    'wall_exterior': 3.6,
    'door_wood': 3.8,
    'fabric_sofa': 2.8
}


def height_to_normal_map(height, strength=3.0):
    """Calcula mapa de normal em espaço tangente usando filtro de convolução Sobel."""
    h = height.astype(np.float32)
    pad = np.pad(h, 1, mode='wrap')

    # Sobel convolution
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

    normal_map = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.uint8)
    normal_map[..., 0] = ((nx * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    normal_map[..., 1] = ((ny * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    normal_map[..., 2] = ((nz * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(normal_map)


def generate_roughness_from_diffuse(diff_img, target):
    """Gera mapa de rugosidade calibrado para o material a partir da luminância do difuso."""
    gray = np.array(diff_img.convert('L')).astype(np.float32) / 255.0
    
    if target == 'floor_porcelain':
        # Porcelanato é altamente espelhado (~0.12), rejuntes mais foscos (~0.85)
        # Pontos escuros no mármore (rejuntes) ficam com maior rugosidade
        rough = 0.12 + (1.0 - gray) * 0.70
    elif target == 'floor_wood':
        # Madeira acetinada (~0.35), frisos mais foscos (~0.75)
        rough = 0.32 + (1.0 - gray) * 0.40
    elif target == 'wall_interior':
        # Tinta fosca acetinada (~0.82)
        rough = 0.80 + (gray - 0.5) * 0.08
    elif target == 'wall_exterior':
        # Fachada áspera (~0.90)
        rough = 0.88 + (gray - 0.5) * 0.10
    elif target == 'door_wood':
        # Porta semi-brilho (~0.42)
        rough = 0.38 + (1.0 - gray) * 0.35
    elif target == 'fabric_sofa':
        # Tecido fosco (~0.86)
        rough = 0.85 + (gray - 0.5) * 0.08
    else:
        rough = 0.50 + (gray - 0.5) * 0.20

    return Image.fromarray((np.clip(rough, 0, 1) * 255).astype(np.uint8))


def generate_pbr_texture_with_recraft(target, token, prompt=None, model="recraftv4_1"):
    """
    Gera textura fotográfica via Recraft AI, calcula Normal e Roughness maps,
    e salva nos diretórios do Viewer e da API.
    """
    if not token:
        raise ValueError("Token de autenticação da Recraft API é obrigatório.")

    if target not in DEFAULT_PROMPTS and not prompt:
        raise ValueError(f"Alvo '{target}' não suportado ou sem prompt customizado.")

    final_prompt = prompt or DEFAULT_PROMPTS[target]
    print(f"[Recraft Textures] Gerando textura para '{target}' com modelo {model}...")

    recraft_payload = {
        "prompt": final_prompt,
        "model": model,
        "size": "1024x1024"
    }

    url = "https://external.api.recraft.ai/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    resp = requests.post(url, headers=headers, json=recraft_payload, timeout=120)
    if resp.status_code != 200:
        err_msg = resp.text
        try:
            err_data = resp.json()
            err_msg = err_data.get('message') or err_data.get('error') or err_msg
        except:
            pass
        raise RuntimeError(f"Erro na API Recraft ({resp.status_code}): {err_msg}")

    res_json = resp.json()
    img_url = res_json.get('data', [{}])[0].get('url')
    if not img_url:
        raise RuntimeError("Nenhuma URL de imagem retornada pelo Recraft.")

    # Baixar a imagem gerada
    img_resp = requests.get(img_url, timeout=40)
    if img_resp.status_code != 200:
        raise RuntimeError("Falha ao baixar imagem gerada do Recraft.")

    diffuse_img = Image.open(io.BytesIO(img_resp.content)).convert('RGB')
    if diffuse_img.size != (SIZE, SIZE):
        diffuse_img = diffuse_img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)

    # Calcular mapa de normais por Sobel na luminância
    height_arr = np.array(diffuse_img.convert('L')).astype(np.float32) / 255.0
    strength = NORMAL_STRENGTHS.get(target, 3.0)
    normal_img = height_to_normal_map(height_arr, strength=strength)

    # Calcular mapa de rugosidade
    roughness_img = generate_roughness_from_diffuse(diffuse_img, target)

    # Salvar nos diretórios de texturas
    dest_dirs = [API_TEXTURES_DIR, VIEWER_TEXTURES_DIR]
    for d in dest_dirs:
        os.makedirs(d, exist_ok=True)
        diffuse_img.save(os.path.join(d, f"{target}_diffuse.png"), optimize=True)
        normal_img.save(os.path.join(d, f"{target}_normal.png"), optimize=True)
        roughness_img.save(os.path.join(d, f"{target}_roughness.png"), optimize=True)

    print(f"[Recraft Textures] Conjunto PBR completo salvo para '{target}'!")
    return {
        "status": "success",
        "target": target,
        "recraft_url": img_url,
        "prompt": final_prompt,
        "diffuse_url": f"/textures/{target}_diffuse.png",
        "normal_url": f"/textures/{target}_normal.png",
        "roughness_url": f"/textures/{target}_roughness.png"
    }
