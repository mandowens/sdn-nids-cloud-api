# =====================================================================
# MODULE: ONLINE INTRUSION SIMULATOR (test_online_simulator.py - LOCAL FILE SYNC)
# =====================================================================
print("--- Memulai Eksekusi Local Offline Traffic Simulator ---")
import time
import numpy as np
import joblib
import pandas as pd
import warnings
import os

# Disable screen warnings for clear output log visualization
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

CSV_FILE_PATH = "sdn_traffic_logs.csv"

# Clear old logs upon starting a new simulation run
if os.path.exists(CSV_FILE_PATH):
    os.remove(CSV_FILE_PATH)

# Creating sequence list for automated network attack pacing
jadwal_serangan = [3, 7, 11, 14]

# Load your custom mathematical hibrida serialized pipelines
try:
    scaler = joblib.load('sdn_nids_output/scaler_sdn.joblib')
    iso_features = joblib.load('sdn_nids_output/iso_forest_features.joblib')
    xgb_base = joblib.load('sdn_nids_output/xgb_base_model.joblib')
    fusion_ml = joblib.load('sdn_nids_output/logistic_fusion_ml.joblib')
    LIVE_MODEL = True
    print("✅ BERHASIL: Model hibrida asli matematika Anda sukses dimuat!")
except:
    print("⚠️ Berkas model lokal tidak ditemukan. Berjalan dalam mode simulasi cerdas.")
    LIVE_MODEL = False

def run_local_inference(p_rate, b_count):
    """Menghitung prediksi menggunakan model lokal Anda."""
    if not LIVE_MODEL:
        if p_rate > 750 and b_count > 300000:
            return "ATTACK", float(np.random.uniform(0.88, 0.98)), "BLOCK_PORT_3"
        return "NORMAL", float(np.random.uniform(0.01, 0.12)), "ALLOW_TRAFFIC"
        
    kolom_asli = ['duration_sec', 'packet_count', 'byte_count', 'packet_rate', 'port_no', 'tx_bytes', 'rx_bytes', 'dt_ratio', 'flow_active']
    raw_input_df = pd.DataFrame([[10.0, 2000, b_count, p_rate, 3, 15000, 35000, 0.4, 1]], columns=kolom_asli)
    
    X_scaled = scaler.transform(raw_input_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=kolom_asli)
    
    evidence = iso_features.decision_function(X_scaled_df).reshape(-1, 1)
    prob_base = xgb_base.predict_proba(X_scaled_df)[:, 1].reshape(-1, 1)
    
    # FIXED INDEX: Explicitly extracting item matrix coordinate to banish TypeError
    risk_proba_matrix = fusion_ml.predict_proba(np.hstack((prob_base, evidence)))
    risk = float(risk_proba_matrix[0, 1])
    
    pred = "ATTACK" if risk > 0.5 else "NORMAL"
    act = "BLOCK_PORT_3" if pred == "ATTACK" else "ALLOW_TRAFFIC"
    return pred, risk, act

print("🚀 Menyiapkan pipa data log streaming real-time...")
for i in range(1, 16): 
    is_attack_interval = (jadwal_serangan.count(i) > 0)
    
    if is_attack_interval:
        p_rate, b_count = float(np.random.uniform(850, 1100)), int(np.random.randint(400000, 700000))
        print(f"\n[{i}] 🚨 Mengirim Trafik Anomali (DDoS Target)...")
    else:
        p_rate, b_count = float(np.random.uniform(50, 200)), int(np.random.randint(10000, 50000))
        print(f"\n[{i}] ✅ Mengirim Trafik Produksi Normal Jaringan...")
        
    prediction, risk_score, action = run_local_inference(p_rate, b_count)
    
    # Pack parameters inside payload matching the exact database structure
    payload = {
        "created_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "packet_rate": p_rate,
        "byte_count": b_count,
        "prediction": prediction,
        "composite_risk_score": risk_score,
        "action_required": action
    }
    
    # Append row directly into local CSV tracking pipeline
    df_payload = pd.DataFrame([payload])
    if not os.path.exists(CSV_FILE_PATH):
        df_payload.to_csv(CSV_FILE_PATH, index=False)
    else:
        df_payload.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)
        
    print(f"    💾 Log appended to local cache! Risk: {risk_score*100:.2f}% | Action: {action}")
    time.sleep(4)

print("\n Pengujian simulasi selesai.")
