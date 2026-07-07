# =====================================================================
# MODULE: LIVE ICMP PING SNIFFER TO SDN-NIDS DASHBOARD PIPELINE
# =====================================================================
print("--- Memulai Pengendus Paket (Live Packet Sniffer) NIDS ---")
import time
import numpy as np
import joblib
import pandas as pd
import warnings
import os
from scapy.all import sniff, ICMP

warnings.filterwarnings('ignore')
CSV_FILE_PATH = "sdn_traffic_logs.csv"

# Global counters untuk menghitung trafik per jendela waktu (window)
packet_counter = 0
byte_counter = 0
start_time = time.time()

# Memuat model lokal hasil riset Anda dari PROSES 7
try:
    scaler = joblib.load('sdn_nids_output/scaler_sdn.joblib')
    iso_features = joblib.load('sdn_nids_output/iso_forest_features.joblib')
    xgb_base = joblib.load('sdn_nids_output/xgb_base_model.joblib')
    fusion_ml = joblib.load('sdn_nids_output/logistic_fusion_ml.joblib')
    LIVE_MODEL = True
    print("✅ BERHASIL: Model hibrida asli matematika Anda sukses dimuat!")
except:
    print("⚠️ Berkas model lokal tidak ditemukan. Menggunakan mode simulasi cerdas.")
    LIVE_MODEL = False

def run_local_inference(p_rate, b_count):
    if not LIVE_MODEL:
        if p_rate > 300 and b_count > 25000: # Threshold disesuaikan lebih sensitif untuk paket ICMP lokal
            return "ATTACK", float(np.random.uniform(0.88, 0.98)), "BLOCK_PORT_3"
        return "NORMAL", float(np.random.uniform(0.01, 0.12)), "ALLOW_TRAFFIC"
        
    kolom_asli = ['duration_sec', 'packet_count', 'byte_count', 'packet_rate', 'port_no', 'tx_bytes', 'rx_bytes', 'dt_ratio', 'flow_active']
    raw_input_df = pd.DataFrame([[10.0, 2000, b_count, p_rate, 3, 15000, 35000, 0.4, 1]], columns=kolom_asli)
    
    X_scaled = scaler.transform(raw_input_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=kolom_asli)
    evidence = iso_features.decision_function(X_scaled_df).reshape(-1, 1)
    prob_base = xgb_base.predict_proba(X_scaled_df)[:, 1].reshape(-1, 1)
    
    risk_proba_matrix = fusion_ml.predict_proba(np.hstack((prob_base, evidence)))
    risk = float(risk_proba_matrix[0, 1])
    
    pred = "ATTACK" if risk > 0.5 else "NORMAL"
    act = "BLOCK_PORT_3" if pred == "ATTACK" else "ALLOW_TRAFFIC"
    return pred, risk, act

def process_network_traffic():
    """Fungsi berkala untuk menghitung statistik jendela waktu dan push ke Dashboard."""
    global packet_counter, byte_counter, start_time
    end_time = time.time()
    duration = end_time - start_time
    
    if duration >= 3.0: # Evaluasi per jendela waktu 3 detik
        p_rate = float(packet_counter / duration)
        b_count = int(byte_counter)
        
        print(f"\n[📊 Window Stats] captured {packet_counter} ICMP packets | Rate: {p_rate:.2f} pkt/s | Bytes: {b_count}")
        prediction, risk_score, action = run_local_inference(p_rate, b_count)
        
        payload = {
            "created_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "packet_rate": p_rate,
            "byte_count": b_count,
            "prediction": prediction,
            "composite_risk_score": risk_score,
            "action_required": action
        }
        
        df_payload = pd.DataFrame([payload])
        if not os.path.exists(CSV_FILE_PATH):
            df_payload.to_csv(CSV_FILE_PATH, index=False)
        else:
            df_payload.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)
            
        print(f"    💾 Dashboard Updated! Risk Index: {risk_score*100:.2f}% -> Decision: {prediction}")
        
        # Reset ulang counter untuk jendela waktu berikutnya
        packet_counter = 0
        byte_counter = 0
        start_time = time.time()

def packet_callback(packet):
    """Callback otomatis yang dipicu setiap kali ada paket ping masuk."""
    global packet_counter, byte_counter
    if packet.haslayer(ICMP):
        packet_counter += 1
        byte_counter += len(packet)
        process_network_traffic()

print("🚀 Sniffer Aktif! Silakan lakukan pengujian PING dari terminal lain...")
# Memulai sniffing paket ICMP secara live (Mendengarkan jaringan lokal Mac Anda)
sniff(filter="icmp", prn=packet_callback, store=0)
