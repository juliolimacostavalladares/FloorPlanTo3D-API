import os
import io
import tempfile
import math
import json
import base64
from datetime import datetime
import requests
import ezdxf
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify
from flask_cors import CORS

ROOT_DIR = os.path.abspath("./")
SAVED_PLANS_DIR = os.path.join(ROOT_DIR, "saved_plans")
os.makedirs(SAVED_PLANS_DIR, exist_ok=True)

application = Flask(__name__)
cors = CORS(application, resources={r"/*": {"origins": "*"}})

NINEROUTER_URL = os.environ.get("NINEROUTER_URL", "http://localhost:20128/v1/chat/completions")
NINEROUTER_KEY = os.environ.get("NINEROUTER_KEY", "sk-4d17a0a7e062b95e-dpfpwg-9d1ccc2f")
PREFERRED_MODEL = os.environ.get("FLOORPLAN_MODEL", "ag/gemini-3.7-flash-high")
FALLBACK_MODEL = "ag/claude-opus-4-6-thinking"


def myImageLoader(imageInput):
    w, h = imageInput.size
    return imageInput, w, h

def getClassNames(classIds):
	result=list()
	for classid in classIds:
		data={}
		if classid==1:
			data['name']='wall'
		if classid==2:
			data['name']='window'
		if classid==3:
			data['name']='door'
		result.append(data)

	return result
def normalizePoints(bbx,classNames):
	normalizingX=1
	normalizingY=1
	result=list()
	doorCount=0
	index=-1
	doorDifference=0
	for bb in bbx:
		index=index+1
		if(classNames[index]==3):
			doorCount=doorCount+1
			if(abs(bb[3]-bb[1])>abs(bb[2]-bb[0])):
				doorDifference=doorDifference+abs(bb[3]-bb[1])
			else:
				doorDifference=doorDifference+abs(bb[2]-bb[0])


		result.append([bb[0]*normalizingY,bb[1]*normalizingX,bb[2]*normalizingY,bb[3]*normalizingX])
	avg_door = (doorDifference / doorCount) if doorCount > 0 else 0.85
	return result, avg_door


def turnSubArraysToJson(objectsArr):
	result=list()
	for obj in objectsArr:
		data={}
		data['x1']=obj[1]
		data['y1']=obj[0]
		data['x2']=obj[3]
		data['y2']=obj[2]
		result.append(data)
	return result



def query_vision_ai(model_name, b64_img, w, h):
    prompt = f"""You are an expert architectural CAD & BIM engineer.
Examine this architectural floor plan image (resolution: {w}x{h} pixels).

CRITICAL ARCHITECTURAL RULES:
1. NEVER confuse dimension lines / cotas (thin lines with measurement numbers like 425, 300, 305, 340, 480, 290, 225, arrows, leader lines) with walls! Dimension lines are NOT walls.
2. NEVER confuse furniture (beds, sofas, tables, cars, counters, appliances) with walls.
3. Walls are the THICK solid double-line structural partitions enclosing the rooms and the perimeter of the house.
4. Extract EACH ROOM with its exact pixel bounding box [ymin, xmin, ymax, xmax]:
   - Identify all internal rooms: e.g. Quartos, Suíte, Banheiro / WC, Cozinha, Sala de Estar, Sala de Jantar, Área de Serviço, Garagem, Circulação.
   - For each room, provide:
     * 'name': room name in Portuguese
     * 'bbox': [ymin, xmin, ymax, xmax] tightly enclosing the interior space of the room in pixels
     * 'floor_type': 'porcelanato' | 'ceramica' | 'madeira' | 'concreto' | 'grama' | 'deck'
5. Extract DOORS:
   - 'bbox': [ymin, xmin, ymax, xmax] at the door opening
6. Extract WINDOWS:
   - 'bbox': [ymin, xmin, ymax, xmax] at the window opening in exterior walls

Return ONLY a raw valid JSON object:
{{
  "Width": {w},
  "Height": {h},
  "project_name": "Planta Baixa Residencial",
  "rooms": [
    {{"name": "Quarto 1", "bbox": [ymin, xmin, ymax, xmax], "floor_type": "porcelanato"}}
  ],
  "doors": [
    {{"bbox": [ymin, xmin, ymax, xmax]}}
  ],
  "windows": [
    {{"bbox": [ymin, xmin, ymax, xmax]}}
  ],
  "lot_width_m": 12.0,
  "lot_depth_m": 25.0
}}"""

    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
        ],
        "max_tokens": 8192
    }

    headers = {
        "Authorization": f"Bearer {NINEROUTER_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(NINEROUTER_URL, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"Vision AI status {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    if "don't see any image" in content.lower() or "not see any image" in content.lower():
        raise ValueError("Model does not support vision input on this endpoint")

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)


def extract_walls_from_rooms(rooms, w, h, wall_thickness=8):
    horizontal_segments = []
    vertical_segments = []

    for r in rooms:
        bbox = r.get("bbox", [])
        if len(bbox) == 4:
            y1, x1, y2, x2 = [float(v) for v in bbox]
            ymin, ymax = min(y1, y2), max(y1, y2)
            xmin, xmax = min(x1, x2), max(x1, x2)
            horizontal_segments.append((ymin, xmin, xmax))
            horizontal_segments.append((ymax, xmin, xmax))
            vertical_segments.append((xmin, ymin, ymax))
            vertical_segments.append((xmax, ymin, ymax))

    def merge_segments(segments):
        merged = []
        sorted_segs = sorted(segments, key=lambda s: (s[0], s[1]))
        tol = max(10, int(min(w, h) * 0.016))
        for coord, start, end in sorted_segs:
            placed = False
            for i, (m_coord, m_start, m_end) in enumerate(merged):
                if abs(coord - m_coord) <= tol:
                    if not (end < m_start - tol or start > m_end + tol):
                        merged[i] = (round((coord + m_coord) / 2.0), min(start, m_start), max(end, m_end))
                        placed = True
                        break
            if not placed:
                merged.append((coord, start, end))
        return merged

    merged_h = merge_segments(horizontal_segments)
    merged_v = merge_segments(vertical_segments)

    walls = []
    ht = wall_thickness // 2
    for y, x1, x2 in merged_h:
        if (x2 - x1) < w * 0.88 and (x2 - x1) >= 15:
            walls.append([max(0, y - ht), x1, min(h, y + ht), x2])

    for x, y1, y2 in merged_v:
        if (y2 - y1) < h * 0.88 and (y2 - y1) >= 15:
            walls.append([y1, max(0, x - ht), y2, min(w, x + ht)])

    return walls


def detect_with_cloud_ai(imagefile, w, h):
    buffered = io.BytesIO()
    imagefile.convert("RGB").save(buffered, format="JPEG", quality=95)
    b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

    parsed = None
    try:
        print(f"Detecting floor plan with {PREFERRED_MODEL}...")
        parsed = query_vision_ai(PREFERRED_MODEL, b64_img, w, h)
    except Exception as e:
        print(f"Preferred model {PREFERRED_MODEL} error: {e}. Falling back to {FALLBACK_MODEL}...")
        parsed = query_vision_ai(FALLBACK_MODEL, b64_img, w, h)

    rooms = parsed.get("rooms", [])
    doors = parsed.get("doors", [])
    windows = parsed.get("windows", [])
    rois = parsed.get("rois", [])

    # Extrai paredes arquitetônicas sólidas a partir dos limites dos cômodos
    clean_walls = []
    if rooms and len(rooms) >= 2:
        clean_walls = extract_walls_from_rooms(rooms, w, h)

    cleaned_rois = []
    for w_box in clean_walls:
        cleaned_rois.append({"type": "wall", "bbox": w_box})

    for d in doors:
        bbox = d.get("bbox", [])
        if len(bbox) == 4:
            cleaned_rois.append({"type": "door", "bbox": bbox})

    for wn in windows:
        bbox = wn.get("bbox", [])
        if len(bbox) == 4:
            cleaned_rois.append({"type": "window", "bbox": bbox})

    # Se não gerou pelas salas, usa rois filtrando linhas de cota que atravessam a tela
    if not clean_walls:
        for item in rois:
            t = item.get("type", "wall").lower()
            bbox = item.get("bbox", [])
            if len(bbox) == 4:
                y1, x1, y2, x2 = [float(v) for v in bbox]
                if t == "wall" and (abs(x2 - x1) > w * 0.85 or abs(y2 - y1) > h * 0.85):
                    continue
                cleaned_rois.append(item)

    parsed["rois"] = cleaned_rois

    bbx = []
    class_ids = []
    type_to_id = {"wall": 1, "window": 2, "door": 3}
    for item in cleaned_rois:
        t = item.get("type", "wall").lower()
        cid = type_to_id.get(t, 1)
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            bbx.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
            class_ids.append(cid)

    return bbx, class_ids, parsed, b64_img


def build_3d_viewer_data(bbx, class_ids, parsed, w, h, b64_img=""):
    lot_w_m = float(parsed.get("lot_width_m", 10.0))
    lot_d_m = float(parsed.get("lot_depth_m", 25.0))
    scale_x = lot_w_m / w
    scale_z = lot_d_m / h

    elements_3d = []
    has_boundary_muro = False

    for idx, (bb, cid) in enumerate(zip(bbx, class_ids)):
        y1, x1, y2, x2 = bb
        cx = (x1 + x2) / 2.0
        cz = (y1 + y2) / 2.0
        dx = abs(x2 - x1)
        dz = abs(y2 - y1)

        px = (cx - w / 2.0) * scale_x
        pz = (cz - h / 2.0) * scale_z

        if cid == 1: # Wall or perimeter muro
            is_muro = (x1 <= w * 0.06 or x2 >= w * 0.94 or y1 <= h * 0.05 or y2 >= h * 0.95)
            if is_muro:
                has_boundary_muro = True
            wall_id = f"muro_{idx}" if is_muro else f"wall_{idx}"
            sy = 2.2 if is_muro else 2.8
            py = sy / 2.0

            if dx >= dz:
                sx = max(0.2, dx * scale_x)
                sz = 0.16
            else:
                sz = max(0.2, dz * scale_z)
                sx = 0.16

            elements_3d.append({
                "id": wall_id,
                "type": "wall",
                "position": [round(px, 3), round(py, 3), round(pz, 3)],
                "size": [round(sx, 3), round(sy, 3), round(sz, 3)],
                "is_exterior": is_muro,
                "bbox_2d": [int(y1), int(x1), int(y2), int(x2)]
            })

        elif cid == 2: # Window
            is_horiz = (dx >= dz)
            win_width = max(0.70, max(dx * scale_x, dz * scale_z))
            wall_thick = 0.16
            if is_horiz:
                sx = win_width
                sz = wall_thick
            else:
                sx = wall_thick
                sz = win_width
            sy = 1.2
            py = 1.5
            elements_3d.append({
                "id": f"win_{idx}",
                "type": "window",
                "position": [round(px, 3), round(py, 3), round(pz, 3)],
                "size": [round(sx, 3), round(sy, 3), round(sz, 3)],
                "bbox_2d": [int(y1), int(x1), int(y2), int(x2)]
            })

        elif cid == 3: # Door
            # Door opening width (standard 0.80m - 0.90m) and wall thickness (0.16m)
            is_horiz = (dx >= dz)
            door_width = max(0.75, min(1.0, max(dx * scale_x, dz * scale_z)))
            wall_thick = 0.16
            if is_horiz:
                sx = door_width
                sz = wall_thick
            else:
                sx = wall_thick
                sz = door_width
            sy = 2.10
            py = sy / 2.0
            elements_3d.append({
                "id": f"door_{idx}",
                "type": "door",
                "position": [round(px, 3), round(py, 3), round(pz, 3)],
                "size": [round(sx, 3), round(sy, 3), round(sz, 3)],
                "bbox_2d": [int(y1), int(x1), int(y2), int(x2)]
            })

    # Add property boundary muros if not already present
    if not has_boundary_muro:
        muro_height = 2.2
        muro_py = muro_height / 2.0
        muro_thick = 0.18
        # Muro lateral esquerdo
        elements_3d.append({
            "id": "muro_perim_esq",
            "type": "wall",
            "position": [-round(lot_w_m / 2.0, 3), round(muro_py, 3), 0.0],
            "size": [muro_thick, muro_height, round(lot_d_m, 3)],
            "is_exterior": True
        })
        # Muro lateral direito
        elements_3d.append({
            "id": "muro_perim_dir",
            "type": "wall",
            "position": [round(lot_w_m / 2.0, 3), round(muro_py, 3), 0.0],
            "size": [muro_thick, muro_height, round(lot_d_m, 3)],
            "is_exterior": True
        })
        # Muro dos fundos
        elements_3d.append({
            "id": "muro_perim_fundos",
            "type": "wall",
            "position": [0.0, round(muro_py, 3), -round(lot_d_m / 2.0, 3)],
            "size": [round(lot_w_m, 3), muro_height, muro_thick],
            "is_exterior": True
        })
        # Muro frontal com vão de entrada
        elements_3d.append({
            "id": "muro_perim_frente",
            "type": "wall",
            "position": [round(lot_w_m * 0.3, 3), round(muro_py, 3), round(lot_d_m / 2.0, 3)],
            "size": [round(lot_w_m * 0.4, 3), muro_height, muro_thick],
            "is_exterior": True
        })

    # Floors and exterior ground zones
    floors = []
    # 1. Base lot ground slab (grama em todo o terreno)
    floors.append({
        "id": "floor_base_lote",
        "name": "Terreno / Gramado Base",
        "tipo": "grama",
        "position": [0.0, -0.01, 0.0],
        "size": [round(lot_w_m + 0.5, 3), 0.04, round(lot_d_m + 0.5, 3)]
    })

    rooms = parsed.get("rooms", [])
    has_pool = False
    has_deck = False

    for idx, room in enumerate(rooms):
        r_box = room.get("bbox", [])
        if len(r_box) == 4:
            ry1, rx1, ry2, rx2 = r_box
            rcx = (rx1 + rx2) / 2.0
            rcz = (ry1 + ry2) / 2.0
            rdx = abs(rx2 - rx1)
            rdz = abs(ry2 - ry1)
            r_px = (rcx - w / 2.0) * scale_x
            r_pz = (rcz - h / 2.0) * scale_z
            r_sx = max(0.6, rdx * scale_x)
            r_sz = max(0.6, rdz * scale_z)
            ftype = room.get("floor_type", "porcelanato")
            if ftype == "piscina":
                has_pool = True
            if ftype == "deck":
                has_deck = True

            floors.append({
                "id": f"floor_{idx}",
                "name": room.get("name", f"Ambiente {idx+1}"),
                "tipo": ftype,
                "position": [round(r_px, 3), 0.03, round(r_pz, 3)],
                "size": [round(r_sx, 3), 0.06, round(r_sz, 3)]
            })

    # Count statistics for UI
    wall_count = len([e for e in elements_3d if e["type"] == "wall"])
    door_count = len([e for e in elements_3d if e["type"] == "door"])
    window_count = len([e for e in elements_3d if e["type"] == "window"])

    return {
        "plan_dimensions_m": {"width": lot_w_m, "depth": lot_d_m, "height": 2.8},
        "counts": {
            "walls": wall_count,
            "doors": door_count,
            "windows": window_count,
            "furniture": 0
        },
        "elements_3d": elements_3d,
        "floors": floors,
        "furniture": [],
        "image_url": f"data:image/jpeg;base64,{b64_img}" if b64_img else None,
        "ai_analysis": {
            "projeto_nome": parsed.get("project_name", "Planta Baixa Residencial Completa"),
            "comodos": [{"nome": r.get("name"), "tipo": r.get("floor_type")} for r in rooms if r.get("name")],
            "ambientes_detectados": [r.get("name") for r in rooms if r.get("name")],
            "area_construida_m2": round(lot_w_m * lot_d_m * 0.55, 1)
        }
    }


@application.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "service": "FloorPlanTo3D-API",
        "preferred_model": PREFERRED_MODEL,
        "fallback_model": FALLBACK_MODEL
    })


@application.route('/api/plans', methods=['GET'])
def get_plans():
    plans = []
    if os.path.exists(SAVED_PLANS_DIR):
        files = [f for f in os.listdir(SAVED_PLANS_DIR) if f.endswith('.json')]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(SAVED_PLANS_DIR, x)), reverse=True)
        for fname in files:
            fpath = os.path.join(SAVED_PLANS_DIR, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    pid = content.get('plan_id') or os.path.splitext(fname)[0]
                    pname = (content.get('ai_analysis') or {}).get('projeto_nome') or content.get('filename') or pid
                    created_at = content.get('created_at') or datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M:%S')
                    plans.append({
                        "id": pid,
                        "title": pname,
                        "filename": fname,
                        "created_at": created_at,
                        "path": f"http://localhost:5001/api/plans/{pid}"
                    })
            except Exception as e:
                print(f"Erro ao ler plano {fname}: {e}")
    return jsonify(plans)


@application.route('/api/plans/<plan_id>', methods=['GET'])
def get_plan_by_id(plan_id):
    target = os.path.join(SAVED_PLANS_DIR, f"{plan_id}.json")
    if not os.path.exists(target):
        for f in os.listdir(SAVED_PLANS_DIR):
            if f == plan_id or f == f"{plan_id}.json":
                target = os.path.join(SAVED_PLANS_DIR, f)
                break
    if os.path.exists(target):
        with open(target, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Plano não encontrado"}), 404


@application.route('/api/plans/latest', methods=['GET'])
def get_latest_plan():
    if os.path.exists(SAVED_PLANS_DIR):
        files = [f for f in os.listdir(SAVED_PLANS_DIR) if f.endswith('.json')]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(SAVED_PLANS_DIR, x)), reverse=True)
            fpath = os.path.join(SAVED_PLANS_DIR, files[0])
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
    return jsonify({"error": "Nenhum plano salvo encontrado"}), 404


def process_cad_dxf(dxf_bytes, filename="arquivo.dxf"):
    raw_bytes = dxf_bytes if isinstance(dxf_bytes, bytes) else str(dxf_bytes).encode('latin1', errors='ignore')
    
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=True) as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        doc = ezdxf.readfile(tmp.name)
    
    msp = doc.modelspace()

    coords_x = []
    coords_y = []
    lines = []
    inserts = []
    texts = []

    for e in msp:
        t = e.dxftype()
        if t == 'LINE':
            p1 = (e.dxf.start.x, e.dxf.start.y)
            p2 = (e.dxf.end.x, e.dxf.end.y)
            lines.append((p1, p2))
            coords_x.extend([p1[0], p2[0]])
            coords_y.extend([p1[1], p2[1]])
        elif t == 'LWPOLYLINE':
            pts = [(p[0], p[1]) for p in e.get_points('xy')]
            for i in range(len(pts) - 1):
                lines.append((pts[i], pts[i+1]))
            if e.is_closed and len(pts) > 2:
                lines.append((pts[-1], pts[0]))
            for p in pts:
                coords_x.append(p[0])
                coords_y.append(p[1])
        elif t == 'INSERT':
            inserts.append({
                'name': e.dxf.name,
                'x': round(e.dxf.insert.x, 2),
                'y': round(e.dxf.insert.y, 2),
                'rot': round(getattr(e.dxf, 'rotation', 0), 1)
            })
            coords_x.append(e.dxf.insert.x)
            coords_y.append(e.dxf.insert.y)
        elif t in ('TEXT', 'MTEXT'):
            text_val = getattr(e.dxf, 'text', '') or getattr(e, 'text', '')
            texts.append({
                'text': str(text_val),
                'x': round(e.dxf.insert.x, 2),
                'y': round(e.dxf.insert.y, 2)
            })

    if not coords_x:
        coords_x = [0.0, 10.0]
    if not coords_y:
        coords_y = [0.0, 10.0]

    min_x, max_x = min(coords_x), max(coords_x)
    min_y, max_y = min(coords_y), max(coords_y)
    cad_w = max(1.0, max_x - min_x)
    cad_d = max(1.0, max_y - min_y)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    # 1. Gerar blueprint 2D vetorial de altíssima definição para o visualizador Split View
    IMG_SIZE = 1200
    PAD = 60
    draw_w = IMG_SIZE - 2 * PAD
    draw_h = IMG_SIZE - 2 * PAD
    scale = min(draw_w / cad_w, draw_h / cad_d)

    def to_px(x, y):
        px = PAD + (x - min_x) * scale
        py = IMG_SIZE - PAD - (y - min_y) * scale
        return int(px), int(py)

    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(10, 15, 26))
    draw = ImageDraw.Draw(img)

    for gx in range(0, IMG_SIZE, 40):
        draw.line([(gx, 0), (gx, IMG_SIZE)], fill=(18, 28, 45), width=1)
    for gy in range(0, IMG_SIZE, 40):
        draw.line([(0, gy), (IMG_SIZE, gy)], fill=(18, 28, 45), width=1)

    for p1, p2 in lines:
        px1 = to_px(p1[0], p1[1])
        px2 = to_px(p2[0], p2[1])
        draw.line([px1, px2], fill=(220, 235, 255), width=2)

    for b in inserts:
        pos = to_px(b['x'], b['y'])
        name = b['name']
        if name.startswith('P80'):
            draw.ellipse([pos[0]-6, pos[1]-6, pos[0]+6, pos[1]+6], fill=(76, 175, 80))
            draw.text((pos[0]+8, pos[1]-7), name, fill=(76, 175, 80))
        elif name.startswith('J1'):
            draw.ellipse([pos[0]-5, pos[1]-5, pos[0]+5, pos[1]+5], fill=(33, 150, 243))
            draw.text((pos[0]+8, pos[1]-7), name, fill=(33, 150, 243))
        else:
            draw.ellipse([pos[0]-5, pos[1]-5, pos[0]+5, pos[1]+5], fill=(0, 229, 255))
            draw.text((pos[0]+8, pos[1]-7), name, fill=(0, 229, 255))

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    b64_blueprint = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 2. IA cataloga entidades do CAD semanticamente
    layer_info = {}
    for e in msp:
        l = e.dxf.layer
        layer_info[l] = layer_info.get(l, 0) + 1

    cad_summary = {
        'blocks': inserts,
        'texts': texts,
        'layers_entity_count': layer_info,
        'cad_dimensions_meters': {'width': round(cad_w, 2), 'depth': round(cad_d, 2)}
    }

    system_prompt = """Você é um Engenheiro BIM e Especialista em IA para CAD/AutoCAD.
Analise a lista completa de blocos, anotações de texto e camadas (layers) extraídas de um arquivo DXF de arquitetura.
Catalogue e classifique inteligentemente TODAS as entidades arquitetônicas:

1. 'rooms': Ambientes/Cômodos (DORMI=Dormitório, ESTAR=Sala, COZINHA=Cozinha, WC=Banheiro, JANTAR=Jantar, AREA=Área de Serviço).
2. 'doors': Portas (P80... = porta 80cm) com posição e rotação.
3. 'windows': Janelas (J15... = 1.5m, J1... = 1.0m) com posição e rotação.
4. 'stairs': Escadas (identifique pela camada ARQ3 com degraus horizontais ou área de circulação).
5. 'furniture': Mobiliário e louças sanitárias identificadas a partir dos blocos (VASOSAN=sanitário, PIA=bancada_pia, GELAD=geladeira, FOGÃO4B=fogão, LAVAT=lavatório).
6. 'openings_specs': Especificações de corte para deixar a casa oca e transitável (alturas de vergas, peitoris e vão livre de 2.10m para portas).

Retorne EXCLUSIVAMENTE um objeto JSON válido (sem markdown, sem crases):
{
  "project_name": "Apartamento Residencial",
  "rooms": [
    {"name": "Dormitório 1", "center": [x, y], "width": 3.5, "depth": 3.2, "floor_type": "madeira"}
  ],
  "doors": [
    {"x": 3.9, "y": 4.7, "width": 0.82, "rotation": 0}
  ],
  "windows": [
    {"x": 1.9, "y": 5.4, "width": 1.50, "rotation": 90}
  ],
  "stairs": [
    {"name": "Escada de Acesso", "layer": "ARQ3", "step_count": 15, "step_height": 0.17}
  ],
  "furniture": [
    {"name": "Vaso Sanitário", "tipo": "sanitario", "x": 2.1, "y": 3.5, "size": [0.45, 0.75, 0.65]},
    {"name": "Geladeira", "tipo": "geladeira", "x": 8.4, "y": 2.2, "size": [0.75, 1.80, 0.75]}
  ],
  "openings_specs": {
    "door_clear_height": 2.10,
    "door_lintel_height": 0.70,
    "window_sill_height": 1.00,
    "window_lintel_height": 0.60,
    "wall_height": 2.80,
    "wall_thickness": 0.15
  }
}"""

    ai_data = None
    for model in [PREFERRED_MODEL, FALLBACK_MODEL]:
        try:
            resp = requests.post(
                NINEROUTER_URL,
                headers={'Authorization': f'Bearer {NINEROUTER_KEY}', 'Content-Type': 'application/json'},
                json={
                    'model': model,
                    'stream': False,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': f'Dados do DXF:\n{json.dumps(cad_summary, ensure_ascii=False)}'}
                    ],
                    'temperature': 0.1
                },
                timeout=60
            )
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content'].strip()
                if '```' in content:
                    content = content.split('```')[1]
                    if content.startswith('json'):
                        content = content[4:]
                    content = content.strip()
                ai_data = json.loads(content)
                break
        except Exception as err:
            print(f"Erro ao consultar modelo {model}: {err}")

    if not ai_data:
        ai_data = {
            "project_name": filename.replace(".dxf", ""),
            "rooms": [],
            "doors": [],
            "windows": [],
            "stairs": [],
            "furniture": [],
            "openings_specs": {
                "door_clear_height": 2.10,
                "door_lintel_height": 0.70,
                "window_sill_height": 1.00,
                "window_lintel_height": 0.60,
                "wall_height": 2.80,
                "wall_thickness": 0.15
            }
        }

    # 3. Montar elementos 3D com cortes arquitetônicos (BIM Openings Clipping)
    # usando as especificações volumétricas calculadas pela IA
    specs = ai_data.get('openings_specs', {})
    WALL_HEIGHT = float(specs.get('wall_height', 2.80))
    WALL_THICK = float(specs.get('wall_thickness', 0.15))
    DOOR_CLEAR_H = float(specs.get('door_clear_height', 2.10))
    DOOR_LINTEL_H = float(specs.get('door_lintel_height', round(WALL_HEIGHT - DOOR_CLEAR_H, 2)))
    WIN_SILL_H = float(specs.get('window_sill_height', 1.00))
    WIN_LINTEL_H = float(specs.get('window_lintel_height', 0.60))

    # Coletar aberturas precisas do CAD
    cad_doors = []
    cad_windows = []
    for b in inserts:
        b_name = b['name']
        b_rot = b.get('rot', 0)
        is_vert = (round(b_rot) in (90, 270))
        if b_name.startswith('P80'):
            cad_doors.append({'type': 'door', 'x': b['x'], 'y': b['y'], 'width': 0.85, 'is_vertical': is_vert, 'name': b_name, 'rot': b_rot})
        elif b_name.startswith('J15'):
            cad_windows.append({'type': 'window', 'x': b['x'], 'y': b['y'], 'width': 1.50, 'is_vertical': is_vert, 'name': b_name, 'rot': b_rot})
        elif b_name.startswith('J1'):
            cad_windows.append({'type': 'window', 'x': b['x'], 'y': b['y'], 'width': 1.00, 'is_vertical': is_vert, 'name': b_name, 'rot': b_rot})

    # Se a IA tiver achado portas/janelas adicionais
    for d in ai_data.get('doors', []):
        if not any(abs(d.get('x', 0) - cd['x']) < 0.5 and abs(d.get('y', 0) - cd['y']) < 0.5 for cd in cad_doors):
            rot = d.get('rotation', 0)
            cad_doors.append({'type': 'door', 'x': d.get('x', 0), 'y': d.get('y', 0), 'width': d.get('width', 0.85), 'is_vertical': round(rot) in (90, 270), 'name': 'P80', 'rot': rot})

    for w in ai_data.get('windows', []):
        if not any(abs(w.get('x', 0) - cw['x']) < 0.5 and abs(w.get('y', 0) - cw['y']) < 0.5 for cw in cad_windows):
            rot = w.get('rotation', 0)
            cad_windows.append({'type': 'window', 'x': w.get('x', 0), 'y': w.get('y', 0), 'width': w.get('width', 1.50), 'is_vertical': round(rot) in (90, 270), 'name': 'J15', 'rot': rot})

    all_openings = cad_doors + cad_windows

    # Classificar e agrupar linhas da camada 0 (paredes) e ARQ3 (degraus de escada)
    h_raw = []
    v_raw = []
    stairs_lines = []

    for e in msp:
        if e.dxftype() == 'LINE':
            if e.dxf.layer == 'ARQ3':
                stairs_lines.append(((round(e.dxf.start.x, 2), round(e.dxf.start.y, 2)),
                                     (round(e.dxf.end.x, 2), round(e.dxf.end.y, 2))))
            elif e.dxf.layer == '0':
                x1, y1 = round(e.dxf.start.x, 2), round(e.dxf.start.y, 2)
                x2, y2 = round(e.dxf.end.x, 2), round(e.dxf.end.y, 2)
                dx, dy = x2 - x1, y2 - y1
                length = (dx**2 + dy**2)**0.5
                if length < 0.20:
                    continue
                if abs(dy) <= 0.05:
                    h_raw.append((min(x1, x2), max(x1, x2), (y1 + y2) / 2.0))
                elif abs(dx) <= 0.05:
                    v_raw.append((min(y1, y2), max(y1, y2), (x1 + x2) / 2.0))

    def merge_1d_segments(intervals):
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for cur in intervals[1:]:
            prev = merged[-1]
            if cur[0] <= prev[1] + 0.12:
                merged[-1] = (prev[0], max(prev[1], cur[1]))
            else:
                merged.append(cur)
        return merged

    # Agrupar linhas horizontais paralelas (fusão de paredes duplas de 15cm)
    h_groups = []
    for xmin, xmax, y in h_raw:
        matched = False
        for grp in h_groups:
            if abs(grp['y'] - y) <= 0.20:
                grp['segments'].append((xmin, xmax))
                grp['y'] = (grp['y'] * grp['count'] + y) / (grp['count'] + 1)
                grp['count'] += 1
                matched = True
                break
        if not matched:
            h_groups.append({'y': y, 'segments': [(xmin, xmax)], 'count': 1})

    # Agrupar linhas verticais paralelas (fusão de paredes duplas de 15cm)
    v_groups = []
    for ymin, ymax, x in v_raw:
        matched = False
        for grp in v_groups:
            if abs(grp['x'] - x) <= 0.20:
                grp['segments'].append((ymin, ymax))
                grp['x'] = (grp['x'] * grp['count'] + x) / (grp['count'] + 1)
                grp['count'] += 1
                matched = True
                break
        if not matched:
            v_groups.append({'x': x, 'segments': [(ymin, ymax)], 'count': 1})

    def clip_wall_with_openings(w_start, w_end, w_fixed, is_horizontal, relevant_openings):
        cuts = []
        for op in relevant_openings:
            op_pos = op['x'] if is_horizontal else op['y']
            op_fixed = op['y'] if is_horizontal else op['x']
            if abs(op_fixed - w_fixed) <= 0.35 and (w_start - 0.20 <= op_pos <= w_end + 0.20):
                c_start = max(w_start, op_pos - op['width'] / 2.0)
                c_end = min(w_end, op_pos + op['width'] / 2.0)
                if c_end > c_start + 0.10:
                    cuts.append({'start': c_start, 'end': c_end, 'type': op['type'], 'op': op})

        if not cuts:
            return [{'start': w_start, 'end': w_end, 'type': 'solid'}]

        cuts.sort(key=lambda c: c['start'])
        result = []
        cur = w_start
        for c in cuts:
            if c['start'] > cur + 0.10:
                result.append({'start': cur, 'end': c['start'], 'type': 'solid'})
            result.append({'start': c['start'], 'end': c['end'], 'type': c['type'], 'op': c['op']})
            cur = max(cur, c['end'])
        if cur < w_end - 0.10:
            result.append({'start': cur, 'end': w_end, 'type': 'solid'})
        return result

    elements_3d = []
    wall_idx = 0

    # Construir paredes horizontais com vãos de corte
    for grp in h_groups:
        y = grp['y']
        for seg_start, seg_end in merge_1d_segments(grp['segments']):
            if seg_end - seg_start < 0.35:
                continue
            clipped = clip_wall_with_openings(seg_start, seg_end, y, True, all_openings)
            for part in clipped:
                p_len = part['end'] - part['start']
                p_mid = (part['start'] + part['end']) / 2.0
                px = p_mid - center_x
                pz = -(y - center_y)

                if part['type'] == 'solid':
                    # Parede sólida
                    elements_3d.append({
                        'id': f'wall_h_{wall_idx}',
                        'type': 'wall',
                        'position': [round(px, 3), round(WALL_HEIGHT / 2.0, 3), round(pz, 3)],
                        'size': [round(p_len, 3), WALL_HEIGHT, WALL_THICK],
                        'is_exterior': False
                    })
                    wall_idx += 1
                elif part['type'] == 'door':
                    # VÃO LIVRE DE PORTA: Chão aberto! Apenas verga superior (lintel)
                    lintel_h = round(DOOR_LINTEL_H, 3)
                    lintel_py = round(DOOR_CLEAR_H + lintel_h / 2.0, 3)
                    elements_3d.append({
                        'id': f'door_lintel_{wall_idx}',
                        'type': 'wall',
                        'position': [round(px, 3), lintel_py, round(pz, 3)],
                        'size': [round(p_len, 3), lintel_h, WALL_THICK],
                        'is_exterior': False
                    })
                    wall_idx += 1
                elif part['type'] == 'window':
                    # VÃO DE JANELA: Peitoril + Verga
                    elements_3d.append({
                        'id': f'win_sill_{wall_idx}',
                        'type': 'wall',
                        'position': [round(px, 3), round(WIN_SILL_H / 2.0, 3), round(pz, 3)],
                        'size': [round(p_len, 3), WIN_SILL_H, WALL_THICK],
                        'is_exterior': False
                    })
                    elements_3d.append({
                        'id': f'win_lintel_{wall_idx}',
                        'type': 'wall',
                        'position': [round(px, 3), round(WALL_HEIGHT - WIN_LINTEL_H / 2.0, 3), round(pz, 3)],
                        'size': [round(p_len, 3), WIN_LINTEL_H, WALL_THICK],
                        'is_exterior': False
                    })
                    wall_idx += 1

    # Construir paredes verticais com vãos de corte
    for grp in v_groups:
        x = grp['x']
        for seg_start, seg_end in merge_1d_segments(grp['segments']):
            if seg_end - seg_start < 0.35:
                continue
            clipped = clip_wall_with_openings(seg_start, seg_end, x, False, all_openings)
            for part in clipped:
                p_len = part['end'] - part['start']
                p_mid = (part['start'] + part['end']) / 2.0
                px = x - center_x
                pz = -(p_mid - center_y)

                if part['type'] == 'solid':
                    elements_3d.append({
                        'id': f'wall_v_{wall_idx}',
                        'type': 'wall',
                        'position': [round(px, 3), round(WALL_HEIGHT / 2.0, 3), round(pz, 3)],
                        'size': [WALL_THICK, WALL_HEIGHT, round(p_len, 3)],
                        'is_exterior': False
                    })
                    wall_idx += 1
                elif part['type'] == 'door':
                    lintel_h = round(DOOR_LINTEL_H, 3)
                    lintel_py = round(DOOR_CLEAR_H + lintel_h / 2.0, 3)
                    elements_3d.append({
                        'id': f'door_lintel_{wall_idx}',
                        'type': 'wall',
                        'position': [round(px, 3), lintel_py, round(pz, 3)],
                        'size': [WALL_THICK, lintel_h, round(p_len, 3)],
                        'is_exterior': False
                    })
                    wall_idx += 1
                elif part['type'] == 'window':
                    elements_3d.append({
                        'id': f'win_sill_{wall_idx}',
                        'type': 'wall',
                        'position': [round(px, 3), round(WIN_SILL_H / 2.0, 3), round(pz, 3)],
                        'size': [WALL_THICK, WIN_SILL_H, round(p_len, 3)],
                        'is_exterior': False
                    })
                    elements_3d.append({
                        'id': f'win_lintel_{wall_idx}',
                        'type': 'wall',
                        'position': [round(px, 3), round(WALL_HEIGHT - WIN_LINTEL_H / 2.0, 3), round(pz, 3)],
                        'size': [WALL_THICK, WIN_LINTEL_H, round(p_len, 3)],
                        'is_exterior': False
                    })
                    wall_idx += 1

    # Inserir Portas Arquitetônicas (dentro dos vãos livres cortados)
    for idx, door in enumerate(cad_doors):
        dx = door['x'] - center_x
        dz = -(door['y'] - center_y)
        w_d = door.get('width', 0.85)
        rot = door.get('rot', 0)
        is_horiz = not door.get('is_vertical', False)
        elements_3d.append({
            "id": f"door_{idx}",
            "type": "door",
            "position": [round(dx, 3), 1.05, round(dz, 3)],
            "size": [round(w_d, 3) if is_horiz else WALL_THICK, 2.10, WALL_THICK if is_horiz else round(w_d, 3)],
            "rotation": rot
        })

    # Inserir Janelas Arquitetônicas (vidro transparente e esquadria dentro dos vãos cortados)
    for idx, win in enumerate(cad_windows):
        wx = win['x'] - center_x
        wz = -(win['y'] - center_y)
        w_w = win.get('width', 1.50)
        rot = win.get('rot', 0)
        is_horiz = not win.get('is_vertical', False)
        elements_3d.append({
            "id": f"win_{idx}",
            "type": "window",
            "position": [round(wx, 3), 1.60, round(wz, 3)],
            "size": [round(w_w, 3) if is_horiz else WALL_THICK, 1.20, WALL_THICK if is_horiz else round(w_w, 3)],
            "rotation": rot
        })

    # Degraus de Escada (ARQ3) - Escada escalonada baixa, sem bloquear circulação!
    stairs_lines.sort(key=lambda seg: (seg[0][1] + seg[1][1]) / 2.0)
    for idx, (p1, p2) in enumerate(stairs_lines):
        dx = p2[0] - p1[0]
        step_w = abs(dx) or 2.20
        step_d = 0.25
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        px = mid_x - center_x
        pz = -(mid_y - center_y)
        step_h = round(0.12 + 0.12 * idx, 2) # Altura escalonada realista
        step_py = round(step_h / 2.0, 2)
        elements_3d.append({
            'id': f'stair_step_{idx}',
            'type': 'wall',
            'position': [round(px, 3), step_py, round(pz, 3)],
            'size': [round(step_w, 3), step_h, step_d],
            'is_exterior': False
        })

    # Pisos (Laje do terreno + acabamento de cada cômodo)
    floors = []
    floors.append({
        "id": "floor_base",
        "name": "Base Laje",
        "tipo": "porcelanato",
        "position": [0.0, -0.01, 0.0],
        "size": [round(cad_w + 1.0, 3), 0.04, round(cad_d + 1.0, 3)]
    })

    for idx, room in enumerate(ai_data.get('rooms', [])):
        c = room.get('center', [0, 0])
        rx = c[0] - center_x
        rz = -(c[1] - center_y)
        rw = room.get('width', 3.0)
        rd = room.get('depth', 3.0)
        floors.append({
            "id": f"floor_{idx}",
            "name": room.get('name', f'Ambiente {idx+1}'),
            "tipo": room.get('floor_type', 'porcelanato'),
            "position": [round(rx, 3), 0.03, round(rz, 3)],
            "size": [round(rw, 3), 0.06, round(rd, 3)]
        })

    # Mobiliário arquitetônico automático a partir dos blocos do CAD
    furniture = []
    furn_id = 0
    for b in inserts:
        b_name = b['name']
        bx = b['x'] - center_x
        bz = -(b['y'] - center_y)

        if b_name == 'VASOSAN':
            furniture.append({
                "id": f"furn_{furn_id}",
                "name": "Vaso Sanitário",
                "tipo": "sanitario",
                "position": [round(bx, 3), 0.40, round(bz, 3)],
                "size": [0.45, 0.75, 0.65],
                "cor": 0xffffff
            })
            furn_id += 1
        elif b_name == 'LAVAT':
            furniture.append({
                "id": f"furn_{furn_id}",
                "name": "Lavatório",
                "tipo": "bancada_pia",
                "position": [round(bx, 3), 0.45, round(bz, 3)],
                "size": [0.60, 0.85, 0.45],
                "cor": 0x1e293b
            })
            furn_id += 1
        elif b_name == 'PIA':
            furniture.append({
                "id": f"furn_{furn_id}",
                "name": "Bancada Cozinha",
                "tipo": "bancada_pia",
                "position": [round(bx, 3), 0.45, round(bz, 3)],
                "size": [1.40, 0.88, 0.60],
                "cor": 0x1e293b
            })
            furn_id += 1
        elif b_name == 'GELAD':
            furniture.append({
                "id": f"furn_{furn_id}",
                "name": "Geladeira",
                "tipo": "geladeira",
                "position": [round(bx, 3), 0.90, round(bz, 3)],
                "size": [0.75, 1.80, 0.75],
                "cor": 0x94a3b8
            })
            furn_id += 1
        elif b_name == 'FOGÃO4B':
            furniture.append({
                "id": f"furn_{furn_id}",
                "name": "Fogão 4 Bocas",
                "tipo": "balcao",
                "position": [round(bx, 3), 0.45, round(bz, 3)],
                "size": [0.60, 0.88, 0.60],
                "cor": 0x0f172a
            })
            furn_id += 1

    # Mobiliário nos cômodos identificados pela IA (Sofá na sala, Cama nos quartos)
    for r in ai_data.get('rooms', []):
        r_name = r.get('name', '').lower()
        c = r.get('center', [0, 0])
        rx = c[0] - center_x
        rz = -(c[1] - center_y)
        if 'dormitório' in r_name or 'quarto' in r_name:
            furniture.append({
                "id": f"furn_{furn_id}",
                "name": f"Cama Casal - {r.get('name')}",
                "tipo": "cama_casal",
                "position": [round(rx, 3), 0.30, round(rz, 3)],
                "size": [1.60, 0.60, 2.00],
                "cor": 0x475569
            })
            furn_id += 1
        elif 'estar' in r_name:
            furniture.append({
                "id": f"furn_{furn_id}",
                "name": "Sofá Retrátil",
                "tipo": "sofa",
                "position": [round(rx, 3), 0.45, round(rz, 3)],
                "size": [2.20, 0.85, 0.95],
                "cor": 0x1e293b
            })
            furn_id += 1
        elif 'jantar' in r_name:
            furniture.append({
                "id": f"furn_{furn_id}",
                "name": "Mesa de Jantar",
                "tipo": "mesa_jantar",
                "position": [round(rx, 3), 0.40, round(rz, 3)],
                "size": [1.60, 0.78, 0.90],
                "cor": 0x78350f
            })
            furn_id += 1

    wall_count = len([e for e in elements_3d if e["type"] == "wall"])
    door_count = len([e for e in elements_3d if e["type"] == "door"])
    window_count = len([e for e in elements_3d if e["type"] == "window"])

    return {
        "plan_dimensions_m": {"width": round(cad_w, 2), "depth": round(cad_d, 2), "height": 2.8},
        "counts": {
            "walls": wall_count,
            "doors": door_count,
            "windows": window_count,
            "furniture": len(furniture)
        },
        "elements_3d": elements_3d,
        "floors": floors,
        "furniture": furniture,
        "image_url": f"data:image/jpeg;base64,{b64_blueprint}",
        "ai_analysis": {
            "projeto_nome": ai_data.get("project_name", filename.replace(".dxf", "")),
            "comodos": [{"nome": r.get("name"), "tipo": r.get("floor_type")} for r in ai_data.get("rooms", [])],
            "ambientes_detectados": [r.get("name") for r in ai_data.get("rooms", [])],
            "area_construida_m2": round(cad_w * cad_d * 0.7, 1)
        },
        "points": [],
        "classes": [],
        "Width": IMG_SIZE,
        "Height": IMG_SIZE,
        "averageDoor": 0.8
    }


@application.route('/', methods=['POST'])
@application.route('/api/upload-and-generate-3d', methods=['POST'])
def prediction():
    file_obj = request.files.get('image') or request.files.get('file')
    if not file_obj:
        return jsonify({"error": "Nenhum arquivo enviado ('image' ou 'file')"}), 400

    filename = getattr(file_obj, 'filename', '') or 'arquivo_planta'
    is_dxf = filename.lower().endswith('.dxf')

    try:
        if is_dxf:
            print(f"==> Processando arquivo CAD DXF via IA Semântica: {filename}")
            data = process_cad_dxf(file_obj.read(), filename)
        else:
            imagefile = Image.open(file_obj.stream)
            image, w, h = myImageLoader(imagefile)
            print(f"==> Processando planta baixa (Imagem): {w}x{h}")

            bbx, class_ids, raw_ai, b64_img = detect_with_cloud_ai(imagefile, w, h)
            temp, averageDoor = normalizePoints(bbx, class_ids)
            temp = turnSubArraysToJson(temp)

            data = {}
            data['points'] = temp
            data['classes'] = getClassNames(class_ids)
            data['Width'] = w
            data['Height'] = h
            data['averageDoor'] = averageDoor

            viewer_3d = build_3d_viewer_data(bbx, class_ids, raw_ai, w, h, b64_img)
            data.update(viewer_3d)

        # Persistência do plano gerado
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data['plan_id'] = plan_id
        data['created_at'] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        data['filename'] = filename

        try:
            plan_path = os.path.join(SAVED_PLANS_DIR, f"{plan_id}.json")
            with open(plan_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"==> Planta salva com sucesso no disco: {plan_path}")
        except Exception as save_err:
            print(f"Aviso ao salvar plano: {save_err}")

        return jsonify({
            "status": "success",
            "data": data,
            **data
        })

    except Exception as e:
        print(f"Erro no processamento da predição: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"=========== Iniciando FloorPlanTo3D-API na porta {port} ===========")
    application.run(host="0.0.0.0", port=port, debug=False)
