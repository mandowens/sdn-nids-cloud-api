# =====================================================================
# MODULE: LOCAL STREAMLIT DASHBOARD (app_online.py - SECURE LIVE SYNC)
# =====================================================================
import streamlit as st
import pandas as pd
import time
import os

st.set_page_config(page_title="SDN-NIDS Monitor Engine", layout="wide")

st.title("🛡️ SDN-NIDS Real-time Analytical Dashboard")
st.subheader("Visual Monitoring Layer via Local Pipeline Interprocess Cache")

CSV_FILE_PATH = "sdn_traffic_logs.csv"
placeholder = st.empty()

# Safely check for file availability and rows configuration
if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
    try:
        # Load up to latest 20 log records, sorting newest to oldest
        data_logs = pd.read_csv(CSV_FILE_PATH)
        
        if not data_logs.empty:
            # Re-sort to mirror descending timestamp sequence 
            data_logs = data_logs.iloc[::-1].reset_index(drop=True)
            latest_packet = data_logs.iloc[0]
            
            with placeholder.container():
                # Display High-level Dashboard KPI Grid
                col1, col2, col3 = st.columns(3)
                with col1:
                    status_text = "🚨 WARNING / ATTACK" if latest_packet['prediction'] == "ATTACK" else "✅ NORMAL"
                    st.metric(label="Status Keamanan Terbaru", value=status_text)
                with col2:
                    st.metric(label="Probabilitas Risiko (Copula)", value=f"{latest_packet['composite_risk_score']*100:.2f} %")
                with col3:
                    st.metric(label="Beban Trafik Jaringan", value=f"{latest_packet['packet_rate']:.2f} pkt/s")
                    
                st.write("---")
                
                # Visual Alert Logic Block
                if "NORMAL" in status_text:
                    st.success(f"### SYSTEM STATUS: {status_text} (Aktivitas Jaringan Bersih)")
                else:
                    st.error(f"### SYSTEM STATUS: {status_text}")
                    st.warning(f"🚨 **SDN Action Triggered:** Perintah `{latest_packet['action_required']}` otomatis dikirim ke switch OpenFlow untuk drop traffic.")

                # Interactive Analytical Forecasting Curves
                st.write("---")
                st.subheader("📈 Tren Risiko & Lonjakan Packet Rate Jaringan")
                
                graph_df = data_logs.iloc[:20][::-1].copy() # Extract chronological block for line chart
                graph_df['Time'] = pd.to_datetime(graph_df['created_at']).dt.strftime('%H:%M:%S')
                
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.write("**Grafik Lonjakan Fluktuasi Packet Rate (pkt/s)**")
                    st.line_chart(graph_df.set_index('Time')['packet_rate'])
                with col_g2:
                    st.write("**Kurva Pemetaan Indeks Risiko (Gaussian Copula)**")
                    st.area_chart(graph_df.set_index('Time')['composite_risk_score'])
                    
                st.write("---")
                st.subheader("📋 Log Forensik Aktivitas Trafik (Top 10 Terbaru)")
                st.dataframe(data_logs.head(10), use_container_width=True)
        else:
            with placeholder.container():
                st.info("🔌 Dashboard Siaga. Menunggu kiriman data log dari Traffic Simulator script...")
    except:
        # Fallback handle if thread collisions happen during simultaneous file read-write
        time.sleep(0.5)
else:
    with placeholder.container():
        st.info("🔌 Dashboard Siaga. Menunggu kiriman data log dari Traffic Simulator script...\n\n"
                "👉 Jalankan perintah `python3 test_online_simulator.py` di terminal untuk mulai mengalirkan data!")

# Trigger automatic UI runtime execution loop refresh every 3 seconds
time.sleep(3)
st.rerun()
