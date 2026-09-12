import os
import io
import json
import base64
import requests
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

ROOT_DIR = os.path.abspath("./")

application = Flask(__name__)
cors = CORS(application, resources={r"/*": {"origins": "*"}})

NINEROUTER_URL = os.environ.get("NINEROUTER_URL", "http://localhost:20128/v1/chat/completions")
NINEROUTER_KEY = os.environ.get("NINEROUTER_KEY", "sk-4d17a0a7e062b95e-dpfpwg-9d1ccc2f")
PREFERRED_MODEL = os.environ.get("FLOORPLAN_MODEL", "ag/claude-opus-4-6-thinking")
FALLBACK_MODEL = "ag/gemini-3.7-flash-high"


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
    prompt = f"""You are a professional architectural CAD & BIM engineer.
Analyze this floor plan image (resolution: {w}x{h} pixels).
Extract:
1. 'wall': wall segments as bounding boxes [y1, x1, y2, x2] in pixel coordinates.
   - Walls must be straight, aligned, and connected to form continuous closed boundaries.
2. 'window': window openings [y1, x1, y2, x2] in pixels.
3. 'door': door openings [y1, x1, y2, x2] in pixels.
4. 'rooms': interior rooms and outdoor spaces with:
   - 'name': room name in Portuguese (ex: 'Garagem', 'Sala de Estar', 'Cozinha', 'Suíte', 'Quarto 1', 'Banheiro', 'Deck Gourmet', 'Piscina', 'Jardim')
   - 'bbox': [y1, x1, y2, x2]
   - 'floor_type': one of ['porcelanato', 'madeira', 'ceramica', 'grama', 'deck', 'piscina']

Return ONLY a raw valid JSON object:
{{
  \"Width\": {w},
  \"Height\": {h},
  \"rois\": [
    {{\"type\": \"wall\", \"bbox\": [y1, x1, y2, x2]}},
    {{\"type\": \"door\", \"bbox\": [y1, x1, y2, x2]}},
    {{\"type\": \"window\", \"bbox\": [y1, x1, y2, x2]}}
  ],
  \"rooms\": [
    {{\"name\": \"...\", \"bbox\": [y1, x1, y2, x2], \"floor_type\": \"...\"}}
  ],
  \"lot_width_m\": 10.0,
  \"lot_depth_m\": 25.0
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

    rois = parsed.get("rois", [])
    bbx = []
    class_ids = []

    type_to_id = {"wall": 1, "window": 2, "door": 3}
    for item in rois:
        t = item.get("type", "wall").lower()
        cid = type_to_id.get(t, 1)
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            bbx.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
            class_ids.append(cid)

    return bbx, class_ids, parsed


def build_3d_viewer_data(bbx, class_ids, parsed, w, h):
    lot_w_m = float(parsed.get("lot_width_m", 10.0))
    lot_d_m = float(parsed.get("lot_depth_m", 25.0))
    scale_x = lot_w_m / w
    scale_z = lot_d_m / h

    elements_3d = []
    for idx, (bb, cid) in enumerate(zip(bbx, class_ids)):
        y1, x1, y2, x2 = bb
        cx = (x1 + x2) / 2.0
        cz = (y1 + y2) / 2.0
        dx = abs(x2 - x1)
        dz = abs(y2 - y1)

        px = (cx - w / 2.0) * scale_x
        pz = (cz - h / 2.0) * scale_z

        if cid == 1:
            sx = max(0.15, dx * scale_x)
            sz = max(0.15, dz * scale_z)
            sy = 2.8
            py = sy / 2.0
            is_ext = (x1 <= w * 0.1 or x2 >= w * 0.9 or y1 <= h * 0.1 or y2 >= h * 0.9)
            elements_3d.append({
                "id": f"wall_{idx}",
                "type": "wall",
                "position": [round(px, 3), round(py, 3), round(pz, 3)],
                "size": [round(sx, 3), round(sy, 3), round(sz, 3)],
                "is_exterior": is_ext
            })
        elif cid == 2:
            sx = max(0.12, dx * scale_x)
            sz = max(0.12, dz * scale_z)
            sy = 1.2
            py = 1.5
            elements_3d.append({
                "id": f"win_{idx}",
                "type": "window",
                "position": [round(px, 3), round(py, 3), round(pz, 3)],
                "size": [round(sx, 3), round(sy, 3), round(sz, 3)]
            })
        elif cid == 3:
            sx = max(0.12, dx * scale_x)
            sz = max(0.12, dz * scale_z)
            sy = 2.1
            py = sy / 2.0
            elements_3d.append({
                "id": f"door_{idx}",
                "type": "door",
                "position": [round(px, 3), round(py, 3), round(pz, 3)],
                "size": [round(sx, 3), round(sy, 3), round(sz, 3)]
            })

    floors = []
    rooms = parsed.get("rooms", [])
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
            r_sx = max(0.5, rdx * scale_x)
            r_sz = max(0.5, rdz * scale_z)
            floors.append({
                "id": f"floor_{idx}",
                "name": room.get("name", f"Ambiente {idx+1}"),
                "tipo": room.get("floor_type", "porcelanato"),
                "position": [round(r_px, 3), 0.025, round(r_pz, 3)],
                "size": [round(r_sx, 3), 0.05, round(r_sz, 3)]
            })

    return {
        "plan_dimensions_m": {"width": lot_w_m, "depth": lot_d_m, "height": 2.8},
        "elements_3d": elements_3d,
        "floors": floors,
        "furniture": [],
        "ai_analysis": {
            "projeto_nome": parsed.get("project_name", "Planta Arquitetônica"),
            "ambientes_detectados": [r.get("name") for r in rooms if r.get("name")]
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
    return jsonify([])


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

        bbx, class_ids, raw_ai = detect_with_cloud_ai(imagefile, w, h)
        temp, averageDoor = normalizePoints(bbx, class_ids)
        temp = turnSubArraysToJson(temp)

        data = {}
        data['points'] = temp
        data['classes'] = getClassNames(class_ids)
        data['Width'] = w
        data['Height'] = h
        data['averageDoor'] = averageDoor

        viewer_3d = build_3d_viewer_data(bbx, class_ids, raw_ai, w, h)
        data.update(viewer_3d)

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
