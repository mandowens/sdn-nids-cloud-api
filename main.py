# =====================================================================
# FILE: main.py (FastAPI Backend Server - Cloud Deployment Engine)
# =====================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.stats import norm
import os
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(
    title="🛡️ SDN-NIDS Hybrid Cloud API Engine", 
    description="API Gateway Produksi untuk Inferensi Model Hibrida ML+DL Disertasi S3",
    version="1.0.0"
)

class OpenFlowTelemetry(BaseModel):
    duration_sec: float
    packet_count: int
    byte_count: int
    packet_rate: float
    port_no: int
    tx_bytes: int
    rx_bytes: int
    dt_ratio: float
    flow_active: int

# Mendefinisikan Relative Path Folder Model yang ikut di-upload
MODEL_DIR = "sdn_nids_output"
scaler, iso_features, xgb_base, fusion_layer_ml, model_dl_keras, dl_calibrator = [None]*6
AUC_ML_HISTORIS, AUC_DL_HISTORIS = 0.9850, 0.9910

@app.on_event("startup")
def load_production_models():
    """Memuat seluruh berkas biner matematika ke dalam kernel memori cloud saat startup."""
    global scaler, iso_features, xgb_base, fusion_layer_ml, model_dl_keras, dl_calibrator
    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_sdn.joblib'))
        iso_features = joblib.load(os.path.join(MODEL_DIR, 'iso_forest_features.joblib'))
        xgb_base = joblib.load(os.path.join(MODEL_DIR, 'xgb_base_model.joblib'))
        fusion_layer_ml = joblib.load(os.path.join(MODEL_DIR, 'logistic_fusion_ml.joblib'))
        model_dl_keras = load_model(os.path.join(MODEL_DIR, 'hybrid_dl_bilstm_attention.keras'), compile=False)
        dl_calibrator = joblib.load(os.path.join(MODEL_DIR, 'dl_platt_calibrator.joblib'))
        print("✅ SUCCESS: Seluruh komponen kecerdasan hibrida berhasil dimuat di Cloud!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Gagal memuat file model hibrida: {str(e)}")

def safe_clip(prob, eps=1e-6): 
    return np.clip(prob, eps, 1 - eps)

def execute_bayesian_copula(prob_ml, prob_dl):
    u1, u2 = safe_clip(prob_ml), safe_clip(prob_dl)
    z1, z2 = norm.ppf(u1), norm.ppf(u2)
    rho = 0.45; denom = 1 - rho**2
    copula_pos = np.exp(-0.5 * (rho**2 * (z1**2 + z2**2) - 2 * rho * z1 * z2) / denom) / np.sqrt(denom)
    copula_neg = np.exp(-0.5 * (rho**2 * (norm.ppf(1-u1)**2 + norm.ppf(1-u2)**2) - 2 * rho * norm.ppf(1-u1) * norm.ppf(1-u2)) / denom) / np.sqrt(denom)
    p_pos = u1 * u2 * copula_pos * (AUC_DL_HISTORIS / (AUC_ML_HISTORIS + AUC_DL_HISTORIS))
    p_neg = (1 - u1) * (1 - u2) * copula_neg * (AUC_ML_HISTORIS / (AUC_ML_HISTORIS + AUC_DL_HISTORIS))
    return float(safe_clip(np.nan_to_num(p_pos / (p_pos + p_neg), nan=0.5)))

@app.post("/predict")
async def evaluate_traffic(telemetry: OpenFlowTelemetry):
    try:
        kolom_asli = ['duration_sec', 'packet_count', 'byte_count', 'packet_rate', 'port_no', 'tx_bytes', 'rx_bytes', 'dt_ratio', 'flow_active']
        input_df = pd.DataFrame([telemetry.dict()], columns=kolom_asli)
        X_scaled = scaler.transform(input_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=kolom_asli)
        
        # 1. Jalur Evaluasi ML Stacking
        evidence = iso_features.decision_function(X_scaled_df).reshape(-1, 1)
        prob_base_xgb = xgb_base.predict_proba(X_scaled_df)[:, 1].reshape(-1, 1)
        prob_final_ml = float(fusion_layer_ml.predict_proba(np.hstack((prob_base_xgb, evidence)))[:, 1])
        
        # 2. Jalur Evaluasi DL BiLSTM
        raw_prob_dl = model_dl_keras.predict(np.expand_dims(X_scaled, axis=1), verbose=0)
        prob_final_dl = float(dl_calibrator.predict_proba(raw_prob_dl)[:, 1])
        
        # 3. Lapisan Keputusan Bayesian Copula
        composite_risk = execute_bayesian_copula(prob_final_ml, prob_final_dl)
        prediction = "ATTACK" if composite_risk > 0.5 else "NORMAL"
        action = "BLOCK_PORT_3" if prediction == "ATTACK" else "ALLOW_TRAFFIC"
        
        return {
            "verdict": prediction,
            "risk_index": round(composite_risk, 4),
            "ml_confidence": round(prob_final_ml, 4),
            "dl_confidence": round(prob_final_dl, 4),
            "mitigation_policy": {
                "action_required": action,
                "target_port": telemetry.port_no
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pengaturan Port Dinamis Mengikuti Regulasi Server Cloud
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
