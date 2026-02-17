
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import io
import base64

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from water_quality_prediction import ensure_trained, predict_one
import timeseries_forecast as ts


app = FastAPI(title="LakeHouse ML Inference API")

DEVICE = torch.device("cpu")


# -------------------------
# WATER (UPDATED)
# -------------------------
class WaterFeatures(BaseModel):
    # allow using either python field name or the alias shown in /docs
    model_config = ConfigDict(populate_by_name=True)

    ph: Optional[float] = Field(None, alias="pH")
    nitrate: Optional[float] = Field(None, alias="Nitrate")
    chloride: Optional[float] = Field(None, alias="Chloride")
    lead: Optional[float] = Field(None, alias="Lead")
    turbidity: Optional[float] = Field(None, alias="Turbidity")
    sulfate: Optional[float] = Field(None, alias="Sulfate")
    conductivity: Optional[float] = Field(None, alias="Conductivity")
    total_dissolved_solids: Optional[float] = Field(None, alias="Total Dissolved Solids")


@app.get("/water/metrics")
def water_metrics():
    return ensure_trained()


@app.post("/water/predict")
def water_predict(features: WaterFeatures):
    ensure_trained()
    # IMPORTANT: send aliases so keys match dataset columns (e.g., "pH", "Total Dissolved Solids")
    result = predict_one(features.model_dump(by_alias=True))
    return {"task": "water", "result": result}


# -------------------------
# KPI Forecast
# -------------------------
KPI_FILES = {
    "boise": "data/boise_state_kpi.xlsx",
    "eastern": "data/eastern_oregon_kpi.xlsx",
    "uab": "data/uab_kpi.xlsx",

    # aliases
    "boise_state": "data/boise_state_kpi.xlsx",
    "eastern_oregon": "data/eastern_oregon_kpi.xlsx",
}


class KpiRequest(BaseModel):
    # style A
    school: Optional[str] = None
    target: Optional[str] = None

    # style B (n8n)
    university: Optional[str] = None
    metric: Optional[str] = None

    steps: int = 6
    rf_lags: int = 4
    rf_trees: int = 200
    lstm_units: int = 32


def _normalize_kpi_request(req: KpiRequest) -> Dict[str, Any]:
    school_raw = (req.school or req.university or "").strip().lower()
    target_raw = (req.target or req.metric or "").strip()

    if not school_raw:
        raise HTTPException(status_code=422, detail="Missing 'school' or 'university' in KPI request")
    if not target_raw:
        raise HTTPException(status_code=422, detail="Missing 'target' or 'metric' in KPI request")

    school_key = school_raw.replace("-", "_")

    if school_key not in KPI_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"school/university must be one of: {sorted(set(KPI_FILES.keys()))}"
        )

    return {
        "school_key": school_key,
        "target_name": target_raw,
        "steps": int(req.steps),
        "rf_lags": int(req.rf_lags),
        "rf_trees": int(req.rf_trees),
        "lstm_units": int(req.lstm_units),
    }


def _run_kpi_forecast(req: KpiRequest) -> Dict[str, Any]:
    norm = _normalize_kpi_request(req)

    df = pd.read_excel(KPI_FILES[norm["school_key"]])
    df_dt, used_date_col = ts.coerce_dt_index(df)

    if norm["target_name"] not in df_dt.columns:
        return {
            "task": "kpi",
            "error": f"target/metric '{norm['target_name']}' not found",
            "available_targets": list(df_dt.columns),
            "date_column_used": used_date_col,
            "school": norm["school_key"],
        }

    series = df_dt[norm["target_name"]]

    rf_fore, lstm_fore, fut_idx = ts.predict_series(
        series=series,
        label_for_rounding=norm["target_name"],
        df_all=df_dt,
        steps=norm["steps"],
        rf_lags=norm["rf_lags"],
        rf_trees=norm["rf_trees"],
        lstm_units=norm["lstm_units"],
        lstm_act_unused="tanh",
    )

    fut_idx_str = [str(x.date()) for x in fut_idx]

    return {
        "task": "kpi",
        "school": norm["school_key"],
        "target": norm["target_name"],
        "steps": norm["steps"],
        "future_index": fut_idx_str,
        "rf_forecast": [float(x) for x in rf_fore],
        "lstm_forecast": None if lstm_fore is None else [float(x) for x in lstm_fore],
        "date_column_used": used_date_col,
        "params": {
            "rf_lags": norm["rf_lags"],
            "rf_trees": norm["rf_trees"],
            "lstm_units": norm["lstm_units"],
        }
    }


@app.post("/kpi/forecast")
def kpi_forecast(req: KpiRequest):
    return _run_kpi_forecast(req)


@app.post("/kpi/predict")
def kpi_predict(req: KpiRequest):
    return _run_kpi_forecast(req)


# -------------------------
# Tumor (Classifier + Seg)
# -------------------------
CLASSIFIER_FILE = "models/brain_tumor_classifier_4cls.pt"
SEG_FILE = "models/seg_deeplabv3_2cls.pt"

clf_model = None
seg_model = None
class_names = None
clf_img_size = 224
seg_img_size = 512

clf_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

seg_tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    # handles "module." from DataParallel checkpoints
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    return {k[len(prefix):]: v for k, v in state.items()}


def load_tumor_models():
    global clf_model, seg_model, class_names, clf_img_size, seg_img_size

    # -------- classifier --------
    if clf_model is None:
        ckpt = torch.load(CLASSIFIER_FILE, map_location=DEVICE)

        arch = ckpt.get("arch", "resnet50")
        num_classes = int(ckpt.get("num_classes", 4))
        class_to_idx = ckpt.get("class_to_idx", None)
        clf_img_size = int(ckpt.get("img_size", 224))

        if arch != "resnet50":
            raise ValueError(f"Unsupported classifier arch: {arch}")

        state = ckpt.get("state_dict", ckpt)
        state = _strip_prefix(state, "module.")

        m = models.resnet50(weights=None)

        # some checkpoints were saved with fc as Sequential, so keys become fc.1.*
        has_fc1 = any(k.startswith("fc.1.") for k in state.keys())

        if has_fc1:
            m.fc = nn.Sequential(
                nn.Dropout(p=0.0),
                nn.Linear(m.fc.in_features, num_classes),
            )
        else:
            m.fc = nn.Linear(m.fc.in_features, num_classes)

        m.load_state_dict(state, strict=True)
        m.eval()
        clf_model = m

        if class_to_idx:
            inv = {v: k for k, v in class_to_idx.items()}
            class_names = [inv[i] for i in range(len(inv))]
        else:
            class_names = [f"class_{i}" for i in range(num_classes)]

    # -------- segmentation --------
    if seg_model is None:
        ckpt = torch.load(SEG_FILE, map_location=DEVICE)

        arch = ckpt.get("arch", "deeplabv3_resnet50")
        num_classes = int(ckpt.get("num_classes", 2))
        seg_img_size = int(ckpt.get("img_size", 512))

        if arch != "deeplabv3_resnet50":
            raise ValueError(f"Unsupported seg arch: {arch}")

        state = ckpt.get("state_dict", ckpt)
        state = _strip_prefix(state, "module.")

        m = models.segmentation.deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes
        )

        # your checkpoint has aux_classifier.* keys sometimes, but current model might not match
        filtered = {k: v for k, v in state.items() if not k.startswith("aux_classifier.")}

        missing, unexpected = m.load_state_dict(filtered, strict=False)

        # leave these prints so you can see once in docker logs if something is off
        if unexpected:
            print("Seg unexpected keys:", unexpected)
        if missing:
            print("Seg missing keys:", missing)

        m.eval()
        seg_model = m


def mask_to_base64_png(mask_01: np.ndarray) -> str:
    img = Image.fromarray((mask_01 * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/tumor/predict")
async def tumor_predict(file: UploadFile = File(...)):
    load_tumor_models()

    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    # classification
    x = clf_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = clf_model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    cls_idx = int(np.argmax(probs))
    cls_name = class_names[cls_idx]
    cls_conf = float(probs[cls_idx])

    # segmentation
    x2 = seg_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = seg_model(x2)
        if isinstance(out, dict) and "out" in out:
            out = out["out"]
        pred = torch.argmax(out, dim=1).cpu().numpy()[0]

    return {
        "task": "tumor",
        "class": cls_name,
        "confidence": cls_conf,
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "segmentation_mask_png_base64": mask_to_base64_png(pred),
        "note": "mask is PNG (base64). decode and display in UI if needed."
    }