import os
import io
import json
import base64
from datetime import datetime
import requests
from PIL import Image
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
	return result,(doorDifference/doorCount)


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


@application.route('/', methods=['POST'])
@application.route('/api/upload-and-generate-3d', methods=['POST'])
def prediction():
    file_obj = request.files.get('image') or request.files.get('file')
    if not file_obj:
        return jsonify({"error": "Nenhuma imagem enviada ('image' ou 'file')"}), 400

    try:
        imagefile = Image.open(file_obj.stream)
        image, w, h = myImageLoader(imagefile)
        print(f"==> Processando planta baixa: {w}x{h}")

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
        data['filename'] = getattr(file_obj, 'filename', f"{plan_id}.png")

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
