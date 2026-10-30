from datetime import datetime
from collections import Counter

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from streamlit_autorefresh import st_autorefresh

from components.live_feed import (
    start_websocket,
    get_connection_status,
    get_live_events
)


API_URL = "http://127.0.0.1:8000"


def render_live_monitor():

    st_autorefresh(
        interval=1000,
        limit=None,
        key="neuralshield_live_refresh"
    )

    start_websocket()

    st.markdown("## ◈ LIVE NETWORK MONITOR")
    st.caption("REAL-TIME PACKET TELEMETRY // PASSIVE MONITORING")

    # Fetch monitoring statistics
    try:
        response = requests.get(
            f"{API_URL}/monitor/status",
            timeout=3
        )
        stats = response.json()
    except requests.RequestException:
        stats = {}

    # Maintain event history across Streamlit reruns
    if "live_packet_history" not in st.session_state:
        st.session_state.live_packet_history = []

    new_events = get_live_events()

    for event in new_events:
        st.session_state.live_packet_history.append(event)

    # Keep only the latest 200 events
    st.session_state.live_packet_history = (
        st.session_state.live_packet_history[-200:]
    )

    history = st.session_state.live_packet_history

    connected = get_connection_status()
    monitoring = stats.get("running", False)

    # Connection indicators
    col1, col2 = st.columns(2)

    with col1:
        if monitoring:
            st.success("CAPTURE ENGINE: ACTIVE")
        else:
            st.warning("CAPTURE ENGINE: STOPPED")

    with col2:
        if connected:
            st.success("WEBSOCKET: CONNECTED")
        else:
            st.error("WEBSOCKET: DISCONNECTED")

    st.divider()

    # Metrics
    packets = stats.get("packets_captured", 0)
    bytes_captured = stats.get("bytes_captured", 0)
    source_ips = stats.get("source_ips", 0)
    destination_ips = stats.get("destination_ips", 0)

    mb_captured = bytes_captured / (1024 * 1024)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("PACKETS CAPTURED", f"{packets:,}")
    m2.metric("DATA TRANSFERRED", f"{mb_captured:.2f} MB")
    m3.metric("SOURCE IPs", source_ips)
    m4.metric("DESTINATION IPs", destination_ips)

    st.divider()

    # Start and stop controls
    st.markdown("### MONITOR CONTROL")

    c1, c2 = st.columns(2)

    with c1:
        if st.button(
            "▶ START CAPTURE",
            use_container_width=True
        ):
            try:
                result = requests.post(
                    f"{API_URL}/monitor/start",
                    json={"interface": None},
                    timeout=5
                )
                st.success(result.json().get("message"))
                st.rerun()
            except requests.RequestException as error:
                st.error(f"Could not start capture: {error}")

    with c2:
        if st.button(
            "■ STOP CAPTURE",
            use_container_width=True
        ):
            try:
                result = requests.post(
                    f"{API_URL}/monitor/stop",
                    timeout=5
                )
                st.info(result.json().get("message"))
                st.rerun()
            except requests.RequestException as error:
                st.error(f"Could not stop capture: {error}")

    if stats.get("capture_error"):
        st.error(f"Capture error: {stats['capture_error']}")

    st.divider()

    # Protocol distribution
    st.markdown("### PROTOCOL DISTRIBUTION")

    protocols = stats.get("protocols", {})

    if protocols:
        protocol_df = pd.DataFrame(
            list(protocols.items()),
            columns=["Protocol", "Packets"]
        )

        fig = px.pie(
            protocol_df,
            names="Protocol",
            values="Packets",
            hole=0.65,
            title="Captured Network Protocols"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Waiting for network traffic...")

    st.divider()

    # Live packet feed
    st.markdown("### LIVE PACKET FEED")

    if history:
        packet_df = pd.DataFrame(history)

        display_columns = [
            "timestamp",
            "source_ip",
            "destination_ip",
            "protocol",
            "source_port",
            "destination_port",
            "packet_size"
        ]

        packet_df = packet_df[
            [col for col in display_columns if col in packet_df.columns]
        ]

        st.dataframe(
            packet_df.tail(30).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        # Recent packet activity
        st.markdown("### RECENT TRAFFIC ACTIVITY")

        if "protocol" in packet_df.columns:
            recent_counts = (
                packet_df["protocol"]
                .value_counts()
                .rename_axis("Protocol")
                .reset_index(name="Packets")
            )

            activity_fig = px.bar(
                recent_counts,
                x="Protocol",
                y="Packets",
                title="Packets received in recent event history"
            )

            activity_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                activity_fig,
                use_container_width=True
            )

    else:
        st.info("No live packet events received yet.")

    st.caption(
        f"Last dashboard refresh: {datetime.now().strftime('%H:%M:%S')}"
    )