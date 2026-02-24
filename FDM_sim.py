import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


#  PAGE CONFIG

st.set_page_config(
    page_title="Aircraft FDM — Live Simulation",
    page_icon="✈",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #000d1a; }
    [data-testid="stSidebar"]          { background-color: #020c18; border-right: 1px solid #1a3045; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    h1, h2, h3, h4   { font-family: monospace !important; letter-spacing: 1px; }
    .section-banner {
        padding: 10px 18px;
        border-radius: 6px;
        margin: 22px 0 14px 0;
        font-family: monospace;
        font-size: 14px;
        font-weight: bold;
        letter-spacing: 2px;
        border-left: 5px solid;
    }
    /* Alert flash box */
    .alert-flash-crit {
        background: #2a0000;
        border: 2px solid #ff0033;
        border-radius: 8px;
        padding: 10px 16px;
        font-family: monospace;
        font-size: 13px;
        color: #ff0033;
        margin: 6px 0;
        animation: flash 1s infinite;
    }
    .alert-flash-warn {
        background: #1e1400;
        border: 2px solid #ffaa00;
        border-radius: 8px;
        padding: 10px 16px;
        font-family: monospace;
        font-size: 13px;
        color: #ffaa00;
        margin: 6px 0;
    }
    @keyframes flash { 0%,100%{opacity:1} 50%{opacity:0.5} }

    /* Playback status pill */
    .status-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-family: monospace;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)



#  SIMULATION DATA  

np.random.seed(42)
DURATION = 30
TIME     = np.arange(DURATION)
LABELS   = [f"T+{t:02d}m" for t in TIME]

PHASES = [
    "Takeoff"  if t < 3  else
    "Climb"    if t < 8  else
    "Cruise"   if t < 22 else
    "Descent"  if t < 27 else
    "Approach"
    for t in TIME
]

# Engine Temperature
eng = np.zeros(DURATION); eng[0] = 620
for i in range(1, DURATION):
    p = PHASES[i]
    if   p == "Takeoff": eng[i] = eng[i-1] + np.random.uniform(15, 30)
    elif p == "Climb":   eng[i] = eng[i-1] + np.random.uniform(20, 35)
    elif p == "Cruise":
        if   i == 10: eng[i] = 862
        elif i == 14: eng[i] = 972
        elif i == 15: eng[i] = 935
        elif i == 16: eng[i] = 855
        else:          eng[i] = 820 + np.random.uniform(-12, 12)
    elif p == "Descent": eng[i] = eng[i-1] - np.random.uniform(10, 20)
    else:                eng[i] = eng[i-1] - np.random.uniform(15, 25)
eng = np.clip(np.round(eng).astype(int), 400, 1100)

# Fuel Level
fuel = np.zeros(DURATION); fuel[0] = 98.0
for i in range(1, DURATION):
    rate = 2.6 + np.random.uniform(0.5, 1.0)
    if PHASES[i] in ("Takeoff", "Climb"): rate *= 1.7
    fuel[i] = fuel[i-1] - rate
fuel = np.clip(np.round(fuel, 1), 0, 100)

# Roll Angle
roll = np.zeros(DURATION)
for i in range(DURATION):
    p = PHASES[i]
    if   p == "Takeoff":  roll[i] = np.random.uniform(-8, 8)
    elif p == "Climb":    roll[i] = np.random.uniform(-12, 12)
    elif p == "Cruise":
        if   i == 11: roll[i] =  42.5
        elif i == 12: roll[i] =  48.0
        elif i == 17: roll[i] = -43.0
        elif i == 18: roll[i] = -41.5
        elif i == 19: roll[i] = -52.0
        elif i == 20: roll[i] = -47.5
        else:          roll[i] = np.random.uniform(-15, 15)
    elif p == "Descent":  roll[i] = np.random.uniform(-18, 18)
    else:                 roll[i] = np.random.uniform(-10, 10)
roll = np.round(roll, 1)

# Altitude
alt = np.zeros(DURATION); alt[0] = 800
for i in range(1, DURATION):
    p = PHASES[i]
    if   p == "Takeoff": alt[i] = alt[i-1] + np.random.uniform(1800, 2500)
    elif p == "Climb":   alt[i] = alt[i-1] + np.random.uniform(2800, 3500)
    elif p == "Cruise":  alt[i] = 36000 + np.random.uniform(-150, 150)
    elif p == "Descent":
        if   i == 23: alt[i] = alt[i-1] - 620
        elif i == 25: alt[i] = alt[i-1] - 1350
        elif i == 26: alt[i] = alt[i-1] - 1100
        else:          alt[i] = alt[i-1] - np.random.uniform(700, 950)
    else:                alt[i] = alt[i-1] - np.random.uniform(1200, 1800)
alt = np.clip(np.round(alt).astype(int), 0, 42000)
alt_drops = np.array([0] + [int(alt[i-1] - alt[i]) for i in range(1, DURATION)])



#  SIDEBAR — Thresholds + Simulation Speed

with st.sidebar:
    st.markdown("## ✈ Aircraft FDM")
    st.markdown("**Live Simulation**")
    st.divider()

    st.markdown("### ⚡ Simulation Speed")
    speed = st.select_slider(
        "Playback speed",
        options=["0.5×", "1×", "2×", "5×", "10×"],
        value="2×",
    )
    speed_map = {"0.5×": 2.0, "1×": 1.0, "2×": 0.5, "5×": 0.2, "10×": 0.1}
    step_delay = speed_map[speed]

    st.divider()
    st.markdown("### 🌡 Engine Temp")
    eng_warn  = st.slider("Warning (°C)",  800, 900,  850, step=10)
    eng_crit  = st.slider("Critical (°C)", 900, 1000, 950, step=10)

    st.divider()
    st.markdown("### ⛽ Fuel Level")
    fuel_warn = st.slider("Low Fuel (%)",     20, 40, 30, step=5)
    fuel_crit = st.slider("Critical Fuel (%)", 5, 25, 20, step=5)

    st.divider()
    st.markdown("### 🔄 Roll Angle")
    roll_alert  = st.slider("Alert (°)",       30, 44, 40, step=1)
    roll_instab = st.slider("Instability (°)", 41, 60, 45, step=1)

    st.divider()
    st.markdown("### 📉 Altitude Drop")
    alt_warn  = st.slider("Warning (ft/min)",   300, 800,  500, step=50)
    alt_emerg = st.slider("Emergency (ft/min)", 800, 1500, 1000, step=50)

    st.divider()
    st.markdown("### 📋 Flight Info")
    st.markdown("🛫 **FLT-BU01**")
    st.markdown("⏱ **Duration:** 30 minutes")
    st.markdown("📡 **Sampling:** 1 / minute")

#  STATUS HELPERS

def eng_status(t):
    if t >= eng_crit: return "CRITICAL"
    elif t >= eng_warn: return "WARNING"
    return "SAFE"

def fuel_status(f):
    if f <= fuel_crit: return "CRITICAL"
    elif f <= fuel_warn: return "WARNING"
    return "SAFE"

def roll_status(r):
    a = abs(r)
    if a > roll_instab: return "INSTABILITY"
    elif a > roll_alert: return "ALERT"
    return "SAFE"

def alt_drop_status(d):
    if d >= alt_emerg: return "EMERGENCY"
    elif d >= alt_warn: return "WARNING"
    return "SAFE"

SEV_COLOR = {
    "EMERGENCY":   "#ff0033",
    "CRITICAL":    "#ff2244",
    "INSTABILITY": "#ff00ff",
    "WARNING":     "#ffaa00",
    "ALERT":       "#ffaa00",
    "SAFE":        "#39ff14",
}

PHASE_CLR = {
    "Takeoff":  "rgba(255,215,0,0.12)",
    "Climb":    "rgba(0,191,255,0.10)",
    "Cruise":   "rgba(57,255,20,0.10)",
    "Descent":  "rgba(255,140,0,0.10)",
    "Approach": "rgba(255,107,53,0.10)",
}


 
# SESSION STATE — track playback position

if "sim_step"    not in st.session_state: st.session_state.sim_step    = 0
if "sim_running" not in st.session_state: st.session_state.sim_running = False
if "sim_done"    not in st.session_state: st.session_state.sim_done    = False



#  PAGE HEADER
st.markdown("# ✈ Aircraft FDM — Live Simulation")
st.markdown("**FLT-BU01** &nbsp;|&nbsp; Real-time flight data playback &nbsp;|&nbsp; All 4 conditions monitored")

# ── Playback controls ─────────────────────────────────────────
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1,1,1,5])
with ctrl1:
    if st.button("▶ START" if not st.session_state.sim_running else "⏸ PAUSE",
                 use_container_width=True):
        if st.session_state.sim_done:
            # restart
            st.session_state.sim_step    = 0
            st.session_state.sim_done    = False
            st.session_state.sim_running = True
        else:
            st.session_state.sim_running = not st.session_state.sim_running
        st.rerun()

with ctrl2:
    if st.button("⏹ RESET", use_container_width=True):
        st.session_state.sim_step    = 0
        st.session_state.sim_running = False
        st.session_state.sim_done    = False
        st.rerun()

with ctrl3:
    # Step forward one frame manually
    if st.button("⏭ STEP", use_container_width=True, disabled=st.session_state.sim_running):
        if st.session_state.sim_step < DURATION - 1:
            st.session_state.sim_step += 1
        st.rerun()

with ctrl4:
    # Timeline scrubber
    chosen = st.slider("⏱ Timeline", 0, DURATION-1,
                        st.session_state.sim_step,
                        disabled=st.session_state.sim_running,
                        label_visibility="collapsed")
    if not st.session_state.sim_running:
        st.session_state.sim_step = chosen

st.divider()

# Current step
n = st.session_state.sim_step  # how many minutes have elapsed (0..29)
visible = n + 1                 # number of data points to show

#  CURRENT-MINUTE STATUS BAR

phase_now    = PHASES[n]
eng_now      = eng[n]
fuel_now     = fuel[n]
roll_now     = roll[n]
alt_now      = alt[n]
drop_now     = alt_drops[n]

eng_s   = eng_status(eng_now)
fuel_s  = fuel_status(fuel_now)
roll_s  = roll_status(roll_now)
alt_s   = alt_drop_status(drop_now)

# Overall worst status
all_statuses = [eng_s, fuel_s, roll_s, alt_s]
if   "EMERGENCY"   in all_statuses or "CRITICAL"    in all_statuses: sys_status = "🔴 CRITICAL"
elif "WARNING"     in all_statuses or "ALERT"       in all_statuses: sys_status = "🟡 WARNING"
elif "INSTABILITY" in all_statuses:                                   sys_status = "🔴 INSTABILITY"
else:                                                                  sys_status = "🟢 NOMINAL"

# Progress
pct_done = int((n / (DURATION-1)) * 100)

hdr1, hdr2, hdr3 = st.columns([2,2,3])
with hdr1:
    st.markdown(f"**🕐 Elapsed:** `{LABELS[n]}`  &nbsp;&nbsp;  **Phase:** `{phase_now}`")
    st.progress(pct_done, text=f"Flight progress: {pct_done}%")
with hdr2:
    st.markdown(f"**System:** {sys_status}")
    # Show any live alerts this minute
    for lbl, s, val, unit in [
        ("🌡 ENG",  eng_s,  eng_now,  "°C"),
        ("⛽ FUEL", fuel_s, fuel_now, "%"),
        (f"🔄 ROLL", roll_s, roll_now, "°"),
        ("📉 DROP", alt_s,  drop_now, "ft/m"),
    ]:
        if s not in ("SAFE",):
            clr = SEV_COLOR.get(s, "#ffaa00")
            st.markdown(
                f'<div style="background:#0a0010;border-left:3px solid {clr};'
                f'padding:4px 10px;margin:2px 0;font-family:monospace;font-size:12px;color:{clr}">'
                f'⚡ {lbl}: <b>{val}{unit}</b> → {s}</div>',
                unsafe_allow_html=True)
with hdr3:
    # Live gauges row
    g1,g2,g3,g4 = st.columns(4)
    g1.metric("🌡 Eng Temp",  f"{eng_now}°C",
              delta=eng_s if eng_s != "SAFE" else "nominal",
              delta_color="inverse" if eng_s != "SAFE" else "off")
    g2.metric("⛽ Fuel",      f"{fuel_now}%",
              delta=fuel_s if fuel_s != "SAFE" else "nominal",
              delta_color="inverse" if fuel_s != "SAFE" else "off")
    g3.metric("🔄 Roll",      f"{roll_now}°",
              delta=roll_s if roll_s != "SAFE" else "nominal",
              delta_color="inverse" if roll_s != "SAFE" else "off")
    g4.metric("📉 Altitude",  f"{alt_now:,} ft",
              delta=f"-{drop_now} ft/m" if drop_now > 0 else "climbing",
              delta_color="inverse" if alt_s != "SAFE" else "off")

st.divider()



#  HELPER: draw line up to current step

def sim_line_chart(y_data, y_label, y_range, line_color,
                   warn_val, crit_val, warn_is_upper,
                   warn_color="#ffaa00", crit_color="#ff2244", height=280):

    x_vis = LABELS[:visible]
    y_vis = list(y_data[:visible])

    fig = go.Figure()

    # Phase bands (only visible range)
    changes = [0]
    for i in range(1, visible):
        if PHASES[i] != PHASES[i-1]: changes.append(i)
    changes.append(visible)
    for j in range(len(changes)-1):
        s, e = changes[j], changes[j+1]
        fig.add_vrect(x0=x_vis[s], x1=x_vis[e-1],
                      fillcolor=PHASE_CLR[PHASES[s]], layer="below", line_width=0)

    # Threshold zones
    if warn_is_upper:
        fig.add_hrect(y0=warn_val, y1=crit_val,   fillcolor=warn_color, opacity=0.06, line_width=0)
        fig.add_hrect(y0=crit_val, y1=y_range[1], fillcolor=crit_color, opacity=0.08, line_width=0)
    else:
        fig.add_hrect(y0=crit_val, y1=warn_val,   fillcolor=warn_color, opacity=0.06, line_width=0)
        fig.add_hrect(y0=y_range[0], y1=crit_val, fillcolor=crit_color, opacity=0.08, line_width=0)

    # Color-coded segments
    for i in range(visible - 1):
        mid = (y_vis[i] + y_vis[i+1]) / 2
        if warn_is_upper:
            sc = crit_color if mid >= crit_val else warn_color if mid >= warn_val else line_color
        else:
            sc = crit_color if mid <= crit_val else warn_color if mid <= warn_val else line_color
        fig.add_trace(go.Scatter(x=[x_vis[i], x_vis[i+1]], y=[y_vis[i], y_vis[i+1]],
            mode="lines", line=dict(color=sc, width=2.5), showlegend=False, hoverinfo="skip"))

    # Markers
    fig.add_trace(go.Scatter(x=x_vis, y=y_vis, mode="markers",
        marker=dict(color=[
            crit_color if (warn_is_upper and v >= crit_val) or (not warn_is_upper and v <= crit_val)
            else warn_color if (warn_is_upper and v >= warn_val) or (not warn_is_upper and v <= warn_val)
            else line_color for v in y_vis
        ], size=5, line=dict(color="#000d1a", width=1)),
        name=y_label, hovertemplate="<b>%{x}</b><br>" + y_label + ": %{y}<extra></extra>"))

    # Highlight current point with pulsing dot
    fig.add_trace(go.Scatter(x=[x_vis[-1]], y=[y_vis[-1]], mode="markers",
        marker=dict(color=line_color, size=12, symbol="circle",
                    line=dict(color="#ffffff", width=2)),
        showlegend=False,
        hovertemplate=f"<b>NOW</b><br>{y_label}: {y_vis[-1]}<extra></extra>"))

    # Fill
    fig.add_trace(go.Scatter(x=x_vis, y=y_vis, fill="tozeroy",
        fillcolor="rgba(255,107,53,0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))

    # Threshold lines
    fig.add_hline(y=warn_val, line_dash="dash", line_color=warn_color, line_width=1.5,
        annotation_text=f"⚠ {'≥' if warn_is_upper else '≤'}{warn_val}",
        annotation_font_color=warn_color, annotation_position="top right")
    fig.add_hline(y=crit_val, line_dash="dash", line_color=crit_color, line_width=1.5,
        annotation_text=f"🔴 {'≥' if warn_is_upper else '≤'}{crit_val}",
        annotation_font_color=crit_color, annotation_position="top right")

    # Alert markers
    ax, ay, atxt, aclr = [], [], [], []
    for i, v in enumerate(y_vis):
        if warn_is_upper:
            if v >= crit_val:   ax.append(x_vis[i]); ay.append(v); atxt.append("🔴"); aclr.append(crit_color)
            elif v >= warn_val: ax.append(x_vis[i]); ay.append(v); atxt.append("⚠");  aclr.append(warn_color)
        else:
            if v <= crit_val:   ax.append(x_vis[i]); ay.append(v); atxt.append("🔴"); aclr.append(crit_color)
            elif v <= warn_val: ax.append(x_vis[i]); ay.append(v); atxt.append("⚠");  aclr.append(warn_color)
    if ax:
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="markers+text",
            marker=dict(symbol="triangle-up", size=12, color=aclr, line=dict(color="#fff", width=1)),
            text=atxt, textposition="top center", textfont=dict(size=10), showlegend=False,
            hovertemplate="<b>%{x}</b><br>🔺 Alert: %{y}<extra></extra>"))

    fig.update_layout(plot_bgcolor="#050f1e", paper_bgcolor="#000d1a",
        font=dict(family="monospace", color="#c8d8e8"),
        xaxis=dict(title="Flight Time", gridcolor="#0d2035", tickfont=dict(size=9),
                   tickangle=-30, tickmode="array",
                   tickvals=LABELS[:visible], ticktext=LABELS[:visible],
                   range=[-0.5, DURATION - 0.5]),  # keep x-axis fixed width
        yaxis=dict(title=y_label, gridcolor="#0d2035", range=y_range),
        height=height, showlegend=False, margin=dict(l=60, r=40, t=20, b=45),
        hovermode="x unified")
    return fig


def sim_roll_chart(height=280):
    x_vis = LABELS[:visible]
    r_vis = list(roll[:visible])

    fig = go.Figure()

    # Phase bands
    changes = [0]
    for i in range(1, visible):
        if PHASES[i] != PHASES[i-1]: changes.append(i)
    changes.append(visible)
    for j in range(len(changes)-1):
        s, e = changes[j], changes[j+1]
        fig.add_vrect(x0=x_vis[s], x1=x_vis[e-1],
                      fillcolor=PHASE_CLR[PHASES[s]], layer="below", line_width=0)

    AC = "#ffaa00"; IC = "#ff00ff"
    fig.add_hrect(y0= roll_alert,  y1= roll_instab, fillcolor=AC, opacity=0.07, line_width=0)
    fig.add_hrect(y0= roll_instab, y1= 65,          fillcolor=IC, opacity=0.09, line_width=0)
    fig.add_hrect(y0=-roll_instab, y1=-roll_alert,  fillcolor=AC, opacity=0.07, line_width=0)
    fig.add_hrect(y0=-65,          y1=-roll_instab, fillcolor=IC, opacity=0.09, line_width=0)
    fig.add_hline(y=0, line_color="#2a4a6a", line_width=1)

    for yv, pos, lbl, clr in [
        ( roll_alert,  "top right",    f"⚠ +{roll_alert}°",   AC),
        ( roll_instab, "top right",    f"🚨 +{roll_instab}°", IC),
        (-roll_alert,  "bottom right", f"⚠ -{roll_alert}°",   AC),
        (-roll_instab, "bottom right", f"🚨 -{roll_instab}°", IC),
    ]:
        fig.add_hline(y=yv, line_dash="dash", line_color=clr, line_width=1.5,
                      annotation_text=lbl, annotation_font_color=clr, annotation_position=pos)

    for i in range(len(r_vis) - 1):
        am = (abs(r_vis[i]) + abs(r_vis[i+1])) / 2
        sc = IC if am > roll_instab else AC if am > roll_alert else "#ff3fa4"
        fig.add_trace(go.Scatter(x=[x_vis[i], x_vis[i+1]], y=[r_vis[i], r_vis[i+1]],
            mode="lines", line=dict(color=sc, width=2.5), showlegend=False, hoverinfo="skip"))

    mcolors = [IC if abs(v) > roll_instab else AC if abs(v) > roll_alert else "#ff3fa4" for v in r_vis]
    fig.add_trace(go.Scatter(x=x_vis, y=r_vis, mode="markers",
        marker=dict(color=mcolors, size=5, line=dict(color="#000d1a", width=1)),
        name="Roll", hovertemplate="<b>%{x}</b><br>Roll: %{y}°<extra></extra>"))

    # Current point highlight
    fig.add_trace(go.Scatter(x=[x_vis[-1]], y=[r_vis[-1]], mode="markers",
        marker=dict(color="#ff3fa4", size=12, line=dict(color="#ffffff", width=2)),
        showlegend=False, hovertemplate=f"<b>NOW</b><br>Roll: {r_vis[-1]}°<extra></extra>"))

    fig.add_trace(go.Scatter(x=x_vis, y=r_vis, fill="tozeroy",
        fillcolor="rgba(255,63,164,0.07)", line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip"))

    ax, ay, atxt, aclr = [], [], [], []
    for i, v in enumerate(r_vis):
        if abs(v) > roll_instab: ax.append(x_vis[i]); ay.append(v); atxt.append("🚨"); aclr.append(IC)
        elif abs(v) > roll_alert: ax.append(x_vis[i]); ay.append(v); atxt.append("⚠"); aclr.append(AC)
    if ax:
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="markers+text",
            marker=dict(symbol=["triangle-up" if v >= 0 else "triangle-down" for v in ay],
                        size=12, color=aclr, line=dict(color="#fff", width=1)),
            text=atxt,
            textposition=["top center" if v >= 0 else "bottom center" for v in ay],
            textfont=dict(size=10), showlegend=False,
            hovertemplate="<b>%{x}</b><br>🔺 Roll Alert: %{y}°<extra></extra>"))

    fig.update_layout(plot_bgcolor="#050f1e", paper_bgcolor="#000d1a",
        font=dict(family="monospace", color="#c8d8e8"),
        xaxis=dict(title="Flight Time", gridcolor="#0d2035", tickfont=dict(size=9),
                   tickangle=-30, tickmode="array",
                   tickvals=LABELS[:visible], ticktext=LABELS[:visible],
                   range=[-0.5, DURATION - 0.5]),
        yaxis=dict(title="Roll Angle (°)", gridcolor="#0d2035", range=[-65, 65], zeroline=False,
                   tickvals=[-60,-45,-40,-30,-20,-10,0,10,20,30,40,45,60],
                   ticktext=["-60°","-45°","-40°","-30°","-20°","-10°","0°",
                              "10°","20°","30°","40°","45°","60°"]),
        height=height, showlegend=False, margin=dict(l=70, r=40, t=20, b=45),
        hovermode="x unified")
    return fig


def sim_altitude_chart(height=280):
    x_vis = LABELS[:visible]
    a_vis = list(alt[:visible])
    d_vis = list(alt_drops[:visible])

    fig = go.Figure()

    # Phase bands + labels
    changes = [0]
    for i in range(1, visible):
        if PHASES[i] != PHASES[i-1]: changes.append(i)
    changes.append(visible)
    for j in range(len(changes)-1):
        s, e = changes[j], changes[j+1]
        fig.add_vrect(x0=x_vis[s], x1=x_vis[e-1],
                      fillcolor=PHASE_CLR[PHASES[s]], layer="below", line_width=0)
        mid = (s + e - 1) // 2
        fig.add_annotation(x=x_vis[mid], y=41500, text=PHASES[s], showarrow=False,
            font=dict(size=8, color="#c8d8e8"), yanchor="top")

    for i in range(len(a_vis) - 1):
        dn = d_vis[i+1]
        sc = "#ff0033" if dn >= alt_emerg else "#ffaa00" if dn >= alt_warn else "#00d4ff"
        fig.add_trace(go.Scatter(x=[x_vis[i], x_vis[i+1]], y=[a_vis[i], a_vis[i+1]],
            mode="lines", line=dict(color=sc, width=2.5), showlegend=False, hoverinfo="skip"))

    mcolors = ["#ff0033" if d_vis[i] >= alt_emerg else "#ffaa00" if d_vis[i] >= alt_warn else "#00d4ff"
               for i in range(len(a_vis))]
    fig.add_trace(go.Scatter(x=x_vis, y=a_vis, mode="markers",
        marker=dict(color=mcolors, size=5, line=dict(color="#000d1a", width=1)),
        name="Altitude",
        hovertemplate="<b>%{x}</b><br>Altitude: %{y:,} ft<br>Drop: %{customdata} ft/min<extra></extra>",
        customdata=d_vis))

    # Current point
    fig.add_trace(go.Scatter(x=[x_vis[-1]], y=[a_vis[-1]], mode="markers",
        marker=dict(color="#00d4ff", size=12, line=dict(color="#ffffff", width=2)),
        showlegend=False, hovertemplate=f"<b>NOW</b><br>{a_vis[-1]:,} ft<extra></extra>"))

    fig.add_trace(go.Scatter(x=x_vis, y=a_vis, fill="tozeroy",
        fillcolor="rgba(0,212,255,0.07)", line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip"))

    ex, ey, etxt, eclr = [], [], [], []
    for i in range(1, len(a_vis)):
        d = d_vis[i]
        if d >= alt_emerg: ex.append(x_vis[i]); ey.append(a_vis[i]); etxt.append(f"🚨-{d}"); eclr.append("#ff0033")
        elif d >= alt_warn: ex.append(x_vis[i]); ey.append(a_vis[i]); etxt.append(f"⚠-{d}"); eclr.append("#ffaa00")
    if ex:
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers+text",
            marker=dict(symbol="triangle-down", size=12, color=eclr, line=dict(color="#fff", width=1)),
            text=etxt, textposition="bottom center", textfont=dict(size=8), showlegend=False,
            hovertemplate="<b>%{x}</b><br>🔻 Drop: %{y:,} ft<extra></extra>"))

    fig.update_layout(plot_bgcolor="#050f1e", paper_bgcolor="#000d1a",
        font=dict(family="monospace", color="#c8d8e8"),
        xaxis=dict(title="Flight Time", gridcolor="#0d2035", tickfont=dict(size=9),
                   tickangle=-30, tickmode="array",
                   tickvals=LABELS[:visible], ticktext=LABELS[:visible],
                   range=[-0.5, DURATION - 0.5]),
        yaxis=dict(title="Altitude (ft)", gridcolor="#0d2035", range=[0, 43000], tickformat=","),
        height=height, showlegend=False, margin=dict(l=80, r=40, t=20, b=45),
        hovermode="x unified")
    return fig

#  RENDER CHARTS  

st.markdown(
    '<div class="section-banner" style="border-color:#00d4ff;background:#00d4ff08;color:#00d4ff">'
    f'📡 LIVE TELEMETRY — {LABELS[n]}  |  Phase: {phase_now}  |  {visible}/{DURATION} samples received'
    '</div>', unsafe_allow_html=True)

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

with row1_col1:
    st.markdown("**🌡 Engine Temperature**")
    st.plotly_chart(sim_line_chart(
        eng, "Engine Temp (°C)", [400, 1050], "#ff6b35",
        eng_warn, eng_crit, True), use_container_width=True)

with row1_col2:
    st.markdown("**⛽ Fuel Level**")
    st.plotly_chart(sim_line_chart(
        fuel, "Fuel Level (%)", [0, 100], "#39ff14",
        fuel_warn, fuel_crit, False,
        warn_color="#ffaa00", crit_color="#ff2244"), use_container_width=True)

with row2_col1:
    st.markdown("**🔄 Roll Angle**")
    st.plotly_chart(sim_roll_chart(), use_container_width=True)

with row2_col2:
    st.markdown("**📉 Altitude Profile**")
    st.plotly_chart(sim_altitude_chart(), use_container_width=True)

#  LIVE ALERT LOG  (cumulative alerts so far)

st.divider()
st.markdown("#### ⚠ Live Alert Log — Events Detected So Far")

all_alerts = []
for i in range(visible):
    es = eng_status(eng[i])
    fs = fuel_status(fuel[i])
    rs = roll_status(roll[i])
    ads = alt_drop_status(alt_drops[i])

    if es != "SAFE":
        all_alerts.append({"Time": LABELS[i], "Phase": PHASES[i], "Condition": "Engine Temp",
                            "Value": f"{eng[i]}°C", "Status": es,
                            "Message": f"Engine {eng[i]}°C — {es}"})
    if fs != "SAFE":
        all_alerts.append({"Time": LABELS[i], "Phase": PHASES[i], "Condition": "Fuel Level",
                            "Value": f"{fuel[i]}%", "Status": fs,
                            "Message": f"Fuel {fuel[i]}% — {fs}"})
    if rs != "SAFE":
        all_alerts.append({"Time": LABELS[i], "Phase": PHASES[i], "Condition": "Roll Angle",
                            "Value": f"{roll[i]}°", "Status": rs,
                            "Message": f"Roll {roll[i]}° — {rs}"})
    if ads != "SAFE" and i > 0:
        all_alerts.append({"Time": LABELS[i], "Phase": PHASES[i], "Condition": "Altitude Drop",
                            "Value": f"{alt_drops[i]} ft/m", "Status": ads,
                            "Message": f"Drop {alt_drops[i]} ft/min — {ads}"})

SEV_BG = {
    "EMERGENCY":   "background-color:#1a0000;color:#ff0033;font-weight:bold",
    "CRITICAL":    "background-color:#2a1000;color:#ff2244;font-weight:bold",
    "WARNING":     "background-color:#1e1600;color:#ffaa00;font-weight:bold",
    "INSTABILITY": "background-color:#1a0030;color:#ff00ff;font-weight:bold",
    "ALERT":       "background-color:#1e1600;color:#ffaa00;font-weight:bold",
    "SAFE":        "background-color:#001a0a;color:#39ff14",
}

if not all_alerts:
    st.success(f"✅ No anomalies detected in the first {visible} minute(s) of flight.")
else:
    alert_df = pd.DataFrame(all_alerts)
    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("Total Alerts", len(all_alerts))
    ac2.metric("🔴 Critical/Emergency", sum(1 for a in all_alerts if a["Status"] in ("CRITICAL","EMERGENCY")))
    ac3.metric("🟡 Warning/Alert",      sum(1 for a in all_alerts if a["Status"] in ("WARNING","ALERT")))
    ac4.metric("🚨 Instability",        sum(1 for a in all_alerts if a["Status"] == "INSTABILITY"))

    st.dataframe(
        alert_df.style
            .applymap(lambda v: SEV_BG.get(v,""), subset=["Status"])
            .hide(axis="index"),
        use_container_width=True,
        height=min(80 + len(all_alerts)*38, 400))
    
#  AUTO-ADVANCE when running

if st.session_state.sim_running:
    if st.session_state.sim_step < DURATION - 1:
        time.sleep(step_delay)
        st.session_state.sim_step += 1
        st.rerun()
    else:
        st.session_state.sim_running = False
        st.session_state.sim_done    = True
        st.success("✅ Simulation complete — FLT-BU01 has landed. Press RESET to replay.")

st.divider()
st.caption("✈ Aircraft FDM System | Live Simulation Mode | FLT-BU01 | All 4 Conditions Active")
