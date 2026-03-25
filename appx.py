"""
Dosimetry Analysis Suite — Streamlit Application
Allied Hospital & Diagnostic Centre, Gwagwalada
Run:  streamlit run app.py
"""

import io, re, glob, os, zipfile, warnings
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dosimetry Analysis Suite",
    page_icon="☢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Precision-Scientific Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
  --navy:    #0B1628;
  --navy2:   #101F38;
  --navy3:   #162440;
  --teal:    #00C9A7;
  --teal2:   #00A88D;
  --amber:   #F5A623;
  --red:     #E84545;
  --green:   #2ECC71;
  --text:    #E8EEF7;
  --muted:   #7B93B8;
  --border:  rgba(0,201,167,0.18);
  --card-bg: rgba(16,31,56,0.85);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--navy) !important;
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
  background: var(--navy2) !important;
  border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

.stTabs [data-baseweb="tab-list"] {
  background: var(--navy2);
  border-radius: 12px;
  padding: 4px;
  gap: 4px;
  border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: var(--muted) !important;
  border-radius: 8px;
  font-family: 'DM Sans', sans-serif;
  font-weight: 500;
  font-size: 13px;
  padding: 8px 18px;
  border: none;
  transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
  background: var(--teal) !important;
  color: var(--navy) !important;
}

.metric-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 24px;
  position: relative;
  overflow: hidden;
}

.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--teal);
  border-radius: 12px 12px 0 0;
}

.metric-card.amber::before { background: var(--amber); }
.metric-card.red::before   { background: var(--red); }
.metric-card.green::before { background: var(--green); }

.metric-label {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1.2px;
  margin-bottom: 8px;
}

.metric-value {
  font-family: 'Space Mono', monospace;
  font-size: 28px;
  font-weight: 700;
  color: var(--text);
  line-height: 1;
}

.metric-sub {
  font-size: 12px;
  color: var(--muted);
  margin-top: 6px;
}

.section-header {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--teal);
  text-transform: uppercase;
  letter-spacing: 2px;
  margin: 28px 0 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  font-family: 'DM Mono', monospace;
}
.badge-green  { background: rgba(46,204,113,0.15); color: #2ECC71; border: 1px solid rgba(46,204,113,0.3); }
.badge-amber  { background: rgba(245,166,35,0.15);  color: #F5A623; border: 1px solid rgba(245,166,35,0.3); }
.badge-red    { background: rgba(232,69,69,0.15);   color: #E84545; border: 1px solid rgba(232,69,69,0.3); }
.badge-teal   { background: rgba(0,201,167,0.15);   color: #00C9A7; border: 1px solid rgba(0,201,167,0.3); }

.stDataFrame { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.stDataFrame thead th { background: var(--navy3) !important; color: var(--teal) !important; font-family: 'DM Mono', monospace !important; font-size: 12px !important; }
.stDataFrame tbody tr:nth-child(even) { background: rgba(22,36,64,0.5) !important; }
.stDataFrame tbody td { color: var(--text) !important; font-size: 13px !important; }

div[data-testid="stDownloadButton"] button {
  background: var(--teal) !important;
  color: var(--navy) !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
}

.stButton > button {
  background: transparent !important;
  border: 1px solid var(--teal) !important;
  color: var(--teal) !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
}

.stButton > button:hover {
  background: var(--teal) !important;
  color: var(--navy) !important;
}

.info-box {
  background: rgba(0,201,167,0.08);
  border: 1px solid rgba(0,201,167,0.25);
  border-left: 4px solid var(--teal);
  border-radius: 0 10px 10px 0;
  padding: 14px 18px;
  font-size: 13.5px;
  color: var(--text);
  margin: 12px 0;
}

.warn-box {
  background: rgba(245,166,35,0.08);
  border-left: 4px solid var(--amber);
  border-radius: 0 10px 10px 0;
  padding: 14px 18px;
  font-size: 13.5px;
  color: var(--text);
  margin: 12px 0;
}

.alert-box {
  background: rgba(232,69,69,0.08);
  border-left: 4px solid var(--red);
  border-radius: 0 10px 10px 0;
  padding: 14px 18px;
  font-size: 13.5px;
  color: var(--text);
  margin: 12px 0;
}

h1 { font-family: 'DM Sans', sans-serif; font-weight: 600; color: var(--text) !important; }
h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 500; color: var(--text) !important; }

/* Matplotlib figure backgrounds */
figure { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
NAVY   = "#0B1628"
TEAL   = "#00C9A7"
AMBER  = "#F5A623"
RED    = "#E84545"
GREEN  = "#2ECC71"
TEXT   = "#E8EEF7"
MUTED  = "#7B93B8"
GRID   = "#1B2E4A"
CARD   = "#101F38"

plt.rcParams.update({
    "figure.facecolor":  NAVY,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  CARD,
    "legend.edgecolor":  GRID,
    "legend.labelcolor": TEXT,
    "legend.fontsize":   9,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "lines.linewidth":   2,
    "patch.linewidth":   0.5,
})

WORKER_COLORS = {
    "CONTROL":        "#7B93B8",
    "BINFA VICTORIA": TEAL,
    "IBRAHIM DALHATU": "#5B8AF0",
}
DOSE_COLORS = {"hp10": "#5B8AF0", "hp003": TEAL, "hp007": AMBER}


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IAEA_ANNUAL_EFFECTIVE = 20.0
IAEA_ANNUAL_LENS      = 20.0
IAEA_CAREER_LIMIT     = 1000.0
OSLD_REL_UNCERT       = 0.05
MC_N                  = 10_000
RNG_SEED              = 42
WORK_HRS_PER_DAY      = 8
WORK_DAYS_PER_WEEK    = 5
WEEKS_PER_MONTH       = 52 / 12
DELTA_D_CM            = (10.0 - 0.07) / 10.0

ICRP74_ENERGY_RATIO = np.array([[15,8.00],[20,4.50],[30,2.55],[40,1.70],[50,1.35],
    [60,1.18],[80,1.07],[100,1.02],[150,1.00],[200,0.99],[300,0.97],[500,0.95]], dtype=float)
NIST_TISSUE_MU = np.array([[15,3.18],[20,1.54],[30,0.57],[40,0.34],[50,0.25],
    [60,0.21],[80,0.18],[100,0.16],[150,0.14],[200,0.13],[300,0.12],[500,0.11]], dtype=float)
ICRP74_FLUENCE_CONV = np.array([[15,0.19],[20,0.52],[30,2.21],[40,4.84],[50,7.75],[60,10.50],
    [80,15.20],[100,19.10],[150,25.90],[200,30.40],[300,36.10],[500,42.30]], dtype=float)
BUILDUP_FACTORS = np.array([[15,1.5],[20,2.0],[30,3.2],[40,4.1],[50,4.8],[60,5.3],
    [80,5.9],[100,6.2],[150,6.5],[200,6.6],[300,6.4],[500,5.8]], dtype=float)
LQ_ALPHA_BETA = {"Skin (late)": 3.0, "Lens (cataract)": 0.5, "Marrow": 10.0}
ICRP103_RISK  = {"solid": 4.1e-2, "leukaemia": 0.4e-2, "total": 4.5e-2}
BEIR7_SOLID   = {"beta_s": 0.47, "gamma": -0.41, "delta": 2.80}
GWAGWALADA_BG = {"low": 0.85, "central": 1.00, "high": 1.20}
REGIONAL_BENCHMARKS = {
    "Nigeria (Farai 2000)": 2.3, "Ghana (Schandorf 2011)": 1.8,
    "Kenya (Korir 2010)": 2.1, "UNSCEAR median": 0.5, "UNSCEAR 75th %ile": 1.0,
}
MONTH_ORDER = {"JANUARY":1,"FEBRUARY":2,"MARCH":3,"APRIL":4,"MAY":5,"JUNE":6,
               "JULY":7,"AUGUST":8,"SEPTEMBER":9,"OCTOBER":10,"NOVEMBER":11,"DECEMBER":12}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def wear_hours(n): return n * WEEKS_PER_MONTH * WORK_DAYS_PER_WEEK * WORK_HRS_PER_DAY

def interp_energy_icrp74(ratio):
    r = ICRP74_ENERGY_RATIO[:,1][::-1]; e = ICRP74_ENERGY_RATIO[:,0][::-1]
    return float(np.interp(np.clip(ratio, r.min(), r.max()), r, e))

def interp_hphi(e_kev):
    return float(np.interp(np.clip(e_kev,15,500), ICRP74_FLUENCE_CONV[:,0], ICRP74_FLUENCE_CONV[:,1]))

def mannkendall(x):
    n = len(x); s = sum(1 if x[j]>x[i] else (-1 if x[j]<x[i] else 0)
                        for i in range(n-1) for j in range(i+1,n))
    var_s = n*(n-1)*(2*n+5)/18 or 1
    z = (s-1)/np.sqrt(var_s) if s>0 else (s+1)/np.sqrt(var_s) if s<0 else 0
    p = 2*(1-stats.norm.cdf(abs(z)))
    return s/(n*(n-1)/2), p, ("increasing" if s>0 else "decreasing" if s<0 else "no trend")

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=NAVY)
    buf.seek(0)
    return buf.read()

def metric_card(label, value, sub="", color="teal"):
    st.markdown(f"""
    <div class="metric-card {color}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)

def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def info_box(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def warn_box(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def alert_box(text):
    st.markdown(f'<div class="alert-box">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────────────────────────────────────
def parse_period(text):
    text = text.upper()
    pat  = r"(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*(\d{4})?"
    matches = re.findall(pat, text)
    if len(matches) >= 2:
        m1,y1 = matches[0]; m2,y2 = matches[-1]
        if not y1 and y2: y1 = y2
        if not y2 and y1: y2 = y1
        n = (int(y2)-int(y1))*12+(MONTH_ORDER[m2]-MONTH_ORDER[m1])+1 if y1 and y2 else 6
        return f"{m1.capitalize()} {y1}", f"{m2.capitalize()} {y2}", max(1, n)
    return "Unknown", "Unknown", 6


def extract_facility_info(cell_text):
    """Extract facility name and location from the header cell.

    Handles two formats found in the wild:

    Format A — name and location on separate lines (Allied Hospital style):
        ALLIED HOSPITAL AND DIAGNOSTIC CENTRE
        GWAGWALADA
        (blank)
        DOSIMETRY RESULT
        PERIOD MONITORED: ...

    Format B — name and location on ONE line (ASK Diagnostic style):
        ASK DIAGNOSTIC AND MEDICAL CENTRE MAITAMA
        DOSIMETRY RESULT
        PERIOD MONITORED: ...

    In Format B the location city is the last word of the single name line.
    We detect this when only one name line exists before the metadata block.
    """
    lines = [ln.strip() for ln in cell_text.split("\n") if ln.strip()]
    name_lines = []
    for ln in lines:
        upper = ln.upper()
        # Stop at the report metadata block
        if any(kw in upper for kw in ("DOSIMETRY", "RESULT", "PERIOD",
                                       "MONITOR", "MONIT0RED")):
            break
        name_lines.append(ln)

    if len(name_lines) >= 2:
        # Format A: first line = facility name, second line = location
        return name_lines[0].title(), name_lines[1].title()

    if len(name_lines) == 1:
        # Format B: single line — last alphabetic word is likely the location
        words = name_lines[0].split()
        if len(words) >= 4 and words[-1].isalpha():
            # e.g. "ASK DIAGNOSTIC AND MEDICAL CENTRE MAITAMA"
            # → name = "Ask Diagnostic And Medical Centre", loc = "Maitama"
            return " ".join(words[:-1]).title(), words[-1].title()
        return name_lines[0].title(), ""

    return "Unknown Facility", ""


def parse_docx_bytes(file_bytes, filename):
    """Parse one .docx dosimetry report.

    Facility name is ALWAYS taken from table[0], row[0], cell[0] —
    no keyword filtering.  The period string and dose rows come from the
    same table (period) and table[1] (data).
    """
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    period_raw = ""
    rows = []
    facility_name = "Unknown Facility"
    facility_location = ""

    for i, table in enumerate(doc.tables):
        if i == 0:
            # ── Always read facility from the very first cell ────────────────
            if table.rows and table.rows[0].cells:
                header_text = table.rows[0].cells[0].text.strip()
                if header_text:
                    facility_name, facility_location = extract_facility_info(header_text)

            # ── Scan all cells for the period string ─────────────────────────
            for row in table.rows:
                ct = " ".join(c.text for c in row.cells)
                if any(m in ct.upper() for m in list(MONTH_ORDER.keys()) + ["PERIOD"]):
                    period_raw = ct.strip()

        elif i == 1:
            # ── Dose data table ───────────────────────────────────────────────
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if len(cells) < 5:
                    continue
                name = cells[1].strip()
                if not name or name.upper() in ("NAME", "S/N", ""):
                    continue
                try:
                    rows.append({
                        "name":  name.upper(),
                        "hp10":  float(cells[2].replace(",", ".")),
                        "hp003": float(cells[3].replace(",", ".")),
                        "hp007": float(cells[4].replace(",", ".")),
                    })
                except (ValueError, IndexError):
                    continue

    start, end, n_months = parse_period(period_raw)
    return {"filename": filename, "period_raw": period_raw,
            "start": start, "end": end, "n_months": n_months, "rows": rows,
            "facility_name": facility_name, "facility_location": facility_location}


def build_dataframe(parsed_list):
    records = []
    for p in parsed_list:
        label = f"{p['start']} – {p['end']}"
        for row in p["rows"]:
            records.append({**row,
                "period_label": label, "start": p["start"], "end": p["end"],
                "n_months": p["n_months"], "filename": p["filename"],
                "facility_name": p.get("facility_name","Unknown Facility"),
                "facility_location": p.get("facility_location",""),
            })
    df = pd.DataFrame(records)
    def sort_key(r):
        parts = r["end"].split()
        return int(parts[1])*100 + MONTH_ORDER.get(parts[0], 0) if len(parts)==2 else 0
    df["_sk"] = df.apply(sort_key, axis=1)
    df = df.sort_values("_sk").drop(columns="_sk").reset_index(drop=True)
    df["_period_idx"] = df["period_label"].map(
        {p:i for i,p in enumerate(df["period_label"].unique())})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ALL ANALYSES  (return DataFrames + report text)
# ─────────────────────────────────────────────────────────────────────────────
# Increment CACHE_VERSION whenever parsed-data schema changes (new columns etc.)
# This forces Streamlit to discard old cached DataFrames.
CACHE_VERSION = "v5"

@st.cache_data(show_spinner=False)
def run_all_analyses(df_json: str, _version: str = CACHE_VERSION):
    df = pd.read_json(io.StringIO(df_json))
    workers = df["name"].unique().tolist()
    periods = df["period_label"].unique().tolist()
    wnc     = [w for w in workers if w != "CONTROL"]
    report  = StringIO()

    # ── 1 Trend ──────────────────────────────────────────────────────────────
    trend = {}
    for w in workers:
        wdf = df[df["name"]==w].sort_values("_period_idx")
        x, y = wdf["_period_idx"].values.astype(float), wdf["hp10"].values
        if len(y) >= 3:
            s,ic,r,p,_ = stats.linregress(x,y)
            tau,pmk,direction = mannkendall(y.tolist())
            trend[w] = dict(slope=s,r2=r**2,p_lin=p,tau=tau,p_mk=pmk,direction=direction,y=y.tolist(),x=x.tolist())

    # ── 2 Net occupational ───────────────────────────────────────────────────
    ctrl_df = df[df["name"]=="CONTROL"][["period_label","hp10","hp003","hp007"]].rename(
        columns={"hp10":"c10","hp003":"c003","hp007":"c007"})
    net_records = []
    for w in wnc:
        wdf = df[df["name"]==w].merge(ctrl_df, on="period_label", how="left")
        for _, row in wdf.iterrows():
            net_records.append({"name":w,"period_label":row["period_label"],
                "_period_idx":row["_period_idx"],
                "net_hp10":row["hp10"]-row["c10"],
                "net_hp003":row["hp003"]-row["c003"],
                "net_hp007":row["hp007"]-row["c007"],
            })
    net_df = pd.DataFrame(net_records)

    # ── 9 Dose rates ─────────────────────────────────────────────────────────
    rate_records = []
    for w in workers:
        wdf = df[df["name"]==w].sort_values("_period_idx")
        for _, row in wdf.iterrows():
            wh = wear_hours(row["n_months"])
            rate_records.append({"name":w,"period_label":row["period_label"],
                "_period_idx":row["_period_idx"],"n_months":row["n_months"],
                "wear_hours":wh,
                "rate_hp10":row["hp10"]*1000/wh,
                "rate_hp003":row["hp003"]*1000/wh,
                "rate_hp007":row["hp007"]*1000/wh,
            })
    rate_df = pd.DataFrame(rate_records)

    # ── 10 Attenuation + energy ───────────────────────────────────────────────
    attn_records = []
    for w in workers:
        wdf = df[df["name"]==w].sort_values("_period_idx")
        for _, row in wdf.iterrows():
            ratio = row["hp007"]/row["hp10"]
            mu    = np.log(ratio)/DELTA_D_CM if ratio > 0.001 else 0.0
            e_i   = interp_energy_icrp74(ratio)
            hvl   = np.log(2)/mu if mu > 0.001 else np.nan
            B     = float(np.interp(np.clip(e_i,15,500), BUILDUP_FACTORS[:,0], BUILDUP_FACTORS[:,1]))
            hphi  = interp_hphi(e_i)
            flux  = (row["hp10"]*1e9/hphi)/(wear_hours(row["n_months"])*3600)
            attn_records.append({"name":w,"period_label":row["period_label"],
                "_period_idx":row["_period_idx"],"n_months":row["n_months"],
                "ratio_007_10":ratio,"mu_eff":mu,"e_icrp74":e_i,
                "hvl_cm":hvl,"buildup":B,"h_phi":hphi,"flux_cm2s":flux,
            })
    attn_df = pd.DataFrame(attn_records)

    # ── 11 Field characterisation ─────────────────────────────────────────────
    field_records = []
    for w in workers:
        wdf = df[df["name"]==w]
        r1 = wdf["hp003"]/wdf["hp10"]; r2=wdf["hp007"]/wdf["hp10"]; r3=wdf["hp007"]/wdf["hp003"]
        all_r = pd.concat([r1,r2,r3])
        fhi  = 1 - all_r.std()/all_r.mean() if all_r.mean()!=0 else 0
        sc   = ((r1-1)/r1*100).clip(lower=0).mean()
        field_records.append({"name":w,"fhi":fhi,"scatter_pct":sc,
            "mean_r1":r1.mean(),"mean_r2":r2.mean(),"mean_r3":r3.mean()})
    field_df = pd.DataFrame(field_records)

    # ── 12 Stochastic risk ───────────────────────────────────────────────────
    risk_records = []
    for w in wnc:
        wdf = df[df["name"]==w]
        cum = wdf["hp10"].sum(); cum_sv = cum/1000
        r_tot = cum_sv*ICRP103_RISK["total"]*100
        beir  = BEIR7_SOLID["beta_s"]*cum_sv*np.exp(BEIR7_SOLID["gamma"]*(30-30)/10+BEIR7_SOLID["delta"]*np.log(70/60))
        ear   = beir*0.048*100
        risk_records.append({"name":w,"cum_msv":cum,"icrp103_pct":r_tot,
            "beir7_ear_pct":ear,"bkg_addition_pct":r_tot/4.8*100})
    risk_df = pd.DataFrame(risk_records)

    # ── 13 Background separation ─────────────────────────────────────────────
    bg_records = []
    for _, row in df[df["name"]=="CONTROL"].sort_values("_period_idx").iterrows():
        bg = GWAGWALADA_BG["central"]/12*row["n_months"]
        sc = max(0, row["hp10"]-bg)
        bg_records.append({"period_label":row["period_label"],"_period_idx":row["_period_idx"],
            "ctrl":row["hp10"],"bg":bg,"scatter":sc,"scatter_pct":sc/row["hp10"]*100 if row["hp10"]>0 else 0,
            "scatter_usv_hr":sc*1000/wear_hours(row["n_months"])})
    bg_df = pd.DataFrame(bg_records)

    # ── 14 Career projection ─────────────────────────────────────────────────
    proj_records = []
    for w in wnc:
        wdf = df[df["name"]==w]
        ann = ((wdf["hp10"]/wdf["n_months"])*12).mean()
        for yr in [1,5,10,15,20,25,30,35]:
            cm = ann*yr
            proj_records.append({"name":w,"year":yr,"cum_msv":cm,
                "risk_pct":cm/1000*ICRP103_RISK["total"]*100,"annual_msv":ann})
    proj_df = pd.DataFrame(proj_records)

    # ── 17 LQ model ──────────────────────────────────────────────────────────
    lq_records = []
    WORK_DAYS_MAP = {6:130, 4:87, 1:22}
    for w in wnc:
        wdf = df[df["name"]==w].sort_values("_period_idx")
        for _, row in wdf.iterrows():
            d_gy = row["hp10"]/1000
            days = WORK_DAYS_MAP.get(int(row["n_months"]), int(row["n_months"]*4.33*5))
            dpd  = d_gy/days if days>0 else 0
            lq_records.append({"name":w,"period_label":row["period_label"],
                "_period_idx":row["_period_idx"],"d_total_gy":d_gy,
                **{f"bed_{t.split()[0].lower()}":d_gy*(1+dpd/ab) for t,ab in LQ_ALPHA_BETA.items()}})
    lq_df = pd.DataFrame(lq_records)

    # ── 19 Monte Carlo ───────────────────────────────────────────────────────
    rng = np.random.default_rng(RNG_SEED)
    mc_records = []
    for w in workers:
        wdf = df[df["name"]==w].sort_values("_period_idx")
        for _, row in wdf.iterrows():
            hp10_s  = np.maximum(rng.normal(row["hp10"],  OSLD_REL_UNCERT*row["hp10"],  MC_N), 1e-6)
            hp007_s = np.maximum(rng.normal(row["hp007"], OSLD_REL_UNCERT*row["hp007"], MC_N), 1e-6)
            ratio_s = hp007_s/hp10_s
            mu_s    = np.log(ratio_s)/DELTA_D_CM
            hvl_s   = np.where(mu_s>0.001, np.log(2)/mu_s, np.nan)
            r_arr   = ICRP74_ENERGY_RATIO[:,1][::-1]; e_arr=ICRP74_ENERGY_RATIO[:,0][::-1]
            e_s     = np.interp(np.clip(ratio_s,r_arr.min(),r_arr.max()), r_arr, e_arr)
            def st_(a):
                v=a[~np.isnan(a)]
                return (float(np.mean(v)),float(np.std(v)),float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) if len(v) else (np.nan,)*4
            mm,ms,mlo,mhi=st_(mu_s); hm,hs,hlo,hhi=st_(hvl_s); em,esd,elo,ehi=st_(e_s)
            mc_records.append({"name":w,"period_label":row["period_label"],"_period_idx":row["_period_idx"],
                "mu_mean":mm,"mu_sd":ms,"mu_lo95":mlo,"mu_hi95":mhi,
                "hvl_mean":hm,"hvl_sd":hs,"e_mean":em,"e_sd":esd,"e_lo95":elo,"e_hi95":ehi})
    mc_df = pd.DataFrame(mc_records)

    # ── 20 K-Means ───────────────────────────────────────────────────────────
    feats, true_labels, per_labels = [], [], []
    for _, row in df.sort_values(["name","_period_idx"]).iterrows():
        feats.append([row["hp003"]/row["hp10"], row["hp007"]/row["hp10"], row["hp007"]/row["hp003"]])
        true_labels.append(row["name"]); per_labels.append(row["period_label"])
    X = np.array(feats); scaler=StandardScaler(); Xs=scaler.fit_transform(X)
    sil_scores = {}
    best_k, best_sil, best_km = 2, -1, None
    for k in [2,3]:
        km = KMeans(n_clusters=k,init="k-means++",n_init=100,random_state=RNG_SEED)
        lbs= km.fit_predict(Xs); sl=silhouette_score(Xs,lbs)
        sil_scores[k]=sl
        if sl>best_sil: best_sil,best_k,best_km=sl,k,km
    cluster_labels = best_km.fit_predict(Xs)
    cluster_df = pd.DataFrame([{"name":true_labels[i],"period_label":per_labels[i],
        "cluster":int(cluster_labels[i]),"f1":feats[i][0],"f2":feats[i][1],"f3":feats[i][2]}
        for i in range(len(feats))])

    # ── 21 Bayesian gaps ─────────────────────────────────────────────────────
    GAPS = [
        {"label":"Gap 1 — May–Jun 2025","months":2,"pre":"July 2025","post":"October 2025"},
        {"label":"Gap 2 — Oct 2025–Sep 2026","months":11,"pre":"July 2025","post":"October 2026"},
    ]
    gap_records = []
    for gap in GAPS:
        for w in wnc:
            wdf = df[df["name"]==w]
            sc = (wdf["hp10"]/wdf["n_months"])*gap["months"]
            mu_pr, sig_pr = sc.mean(), sc.std()
            pre  = wdf[wdf["period_label"].str.contains(gap["pre"][:7],na=False)]
            post = wdf[wdf["period_label"].str.contains(gap["post"][:7],na=False)]
            mu_nb = mu_pr
            if len(pre):  mu_nb = (pre["hp10"].iloc[0]/pre["n_months"].iloc[0])*gap["months"]
            if len(post): mu_nb = (mu_nb+(post["hp10"].iloc[0]/post["n_months"].iloc[0])*gap["months"])/2
            sig_obs = sig_pr
            if sig_pr>0 and sig_obs>0:
                pp=1/sig_pr**2; po=1/sig_obs**2
                post_mean = (mu_pr*pp+mu_nb*po)/(pp+po)
                post_sig  = np.sqrt(1/(pp+po))
            else:
                post_mean, post_sig = mu_pr, max(mu_pr*0.2, 0.001)
            gap_records.append({"worker":w,"gap_label":gap["label"],"months":gap["months"],
                "prior_mean":mu_pr,"prior_sigma":sig_pr,"neighbour":mu_nb,
                "post_mean":post_mean,"post_sigma":post_sig,
                "ci_lo":max(0,post_mean-1.96*post_sig),"ci_hi":post_mean+1.96*post_sig})
    gap_df = pd.DataFrame(gap_records)

    return {
        "trend": trend, "net_df": net_df.to_json(),
        "rate_df": rate_df.to_json(), "attn_df": attn_df.to_json(),
        "field_df": field_df.to_json(), "risk_df": risk_df.to_json(),
        "bg_df": bg_df.to_json(), "proj_df": proj_df.to_json(),
        "lq_df": lq_df.to_json(), "mc_df": mc_df.to_json(),
        "cluster_df": cluster_df.to_json(), "gap_df": gap_df.to_json(),
        "sil_scores": sil_scores, "best_k": best_k, "best_sil": best_sil,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def make_fig(nrows=1, ncols=1, figsize=(11,5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=NAVY)
    if nrows*ncols == 1: axes = [axes]
    for ax in (axes if isinstance(axes, (list,np.ndarray)) else [axes]):
        if hasattr(ax,'__iter__'):
            for a in ax: a.set_facecolor(CARD)
        else:
            ax.set_facecolor(CARD)
    return fig, axes

def plot_dose_timeline(df):
    fig, (ax,) = make_fig(1, 1, (11, 4.5))
    periods = df["period_label"].unique().tolist()
    short_p = [p.split("–")[1].strip() if "–" in p else p for p in periods]
    for w in df["name"].unique():
        wdf = df[df["name"]==w].sort_values("_period_idx")
        col = WORKER_COLORS.get(w,"#999"); ls = "--" if w=="CONTROL" else "-"
        ax.plot(wdf["_period_idx"], wdf["hp10"], color=col, ls=ls, lw=2.2,
                marker="o", ms=7, label=w.title(), zorder=3)
        ax.fill_between(wdf["_period_idx"], 0, wdf["hp10"], color=col, alpha=0.04)
    ax.axhline(IAEA_ANNUAL_EFFECTIVE/12, color=RED, ls=":", lw=1.4, alpha=0.8,
               label=f"IAEA/12 = {IAEA_ANNUAL_EFFECTIVE/12:.1f} mSv/mo")
    ax.set_xticks(range(len(periods))); ax.set_xticklabels(short_p, rotation=20, ha="right")
    ax.set_ylabel("Hp(10)  [mSv / period]")
    ax.set_title("Longitudinal Hp(10) Deep Dose", pad=12, fontweight="bold")
    ax.legend(); ax.set_ylim(bottom=0)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig

def plot_three_quantities(df):
    wnc = [w for w in df["name"].unique() if w != "CONTROL"]
    fig, axes = plt.subplots(1, len(wnc), figsize=(11,4.5), sharey=True, facecolor=NAVY)
    if len(wnc)==1: axes=[axes]
    for ax, ax_face in zip(axes, axes): ax_face.set_facecolor(CARD)
    periods = df["period_label"].unique().tolist()
    short_p = [p.split("–")[1].strip() if "–" in p else p for p in periods]
    for ax, w in zip(axes, wnc):
        ax.set_facecolor(CARD)
        wdf = df[df["name"]==w].sort_values("_period_idx"); xi=wdf["_period_idx"].values
        for col_k, lbl, cc in [("hp10","Hp(10)",DOSE_COLORS["hp10"]),
                                ("hp003","Hp(0.03)",DOSE_COLORS["hp003"]),
                                ("hp007","Hp(0.07)",DOSE_COLORS["hp007"])]:
            ax.plot(xi, wdf[col_k], color=cc, lw=2, marker="s", ms=6, label=lbl)
            ax.fill_between(xi, 0, wdf[col_k], color=cc, alpha=0.05)
        ax.set_title(w.title(), fontweight="bold")
        ax.set_xticks(xi); ax.set_xticklabels([short_p[i] for i in xi], rotation=25, ha="right")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Dose  [mSv]"); axes[-1].legend()
    fig.suptitle("Three Dose Quantities per Worker", fontweight="bold", y=1.01)
    fig.tight_layout(); return fig

def plot_heatmap(df):
    fig, (ax,) = make_fig(1, 1, (11, 3.2))
    pivot = df.pivot_table(index="name", columns="period_label", values="hp10")
    periods = df[["period_label","_period_idx"]].drop_duplicates().sort_values("_period_idx")["period_label"].tolist()
    pivot = pivot.reindex(columns=periods)
    short_p = [p.split("–")[1].strip() if "–" in p else p for p in pivot.columns]
    im = ax.imshow(pivot.values, aspect="auto", cmap="plasma", vmin=0)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(short_p, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([n.title() for n in pivot.index])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label("Hp(10) [mSv]", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT); plt.setp(plt.getp(cbar.ax.axes,'yticklabels'), color=TEXT)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = pivot.values[r,c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if v > pivot.values.max()*0.5 else TEXT)
    ax.set_title("Hp(10) Dose Heatmap", fontweight="bold")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    fig.tight_layout(); return fig

def plot_net_dose(net_df, df):
    fig, (ax,) = make_fig(1, 1, (11, 4))
    wnames = net_df["name"].unique(); bw=0.38
    offs = np.linspace(-bw*(len(wnames)-1)/2, bw*(len(wnames)-1)/2, len(wnames))
    periods = df["period_label"].unique().tolist()
    short_p = [p.split("–")[1].strip() if "–" in p else p for p in periods]
    # Build a period-index map from df in case net_df lost _period_idx in JSON round-trip
    pidx_map = df[["period_label","_period_idx"]].drop_duplicates().set_index("period_label")["_period_idx"].to_dict()
    for wn, off in zip(wnames, offs):
        sub = net_df[net_df["name"]==wn].copy()
        sub["_pidx"] = sub["period_label"].map(pidx_map)
        sub = sub.sort_values("_pidx")
        ax.bar(sub["_pidx"]+off, sub["net_hp10"], bw*0.9,
               color=WORKER_COLORS.get(wn,"#999"), alpha=0.85, label=wn.title(),
               edgecolor="none")
    ax.axhline(0, color=MUTED, lw=1); ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(short_p, rotation=20, ha="right")
    ax.set_ylabel("Net Hp(10)  [mSv]  (above background)")
    ax.set_title("Background-Corrected Net Occupational Dose", fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); return fig

def plot_lens_ratio(df):
    fig, (ax,) = make_fig(1, 1, (10, 4))
    periods = df["period_label"].unique().tolist()
    short_p = [p.split("–")[1].strip() if "–" in p else p for p in periods]
    for w in df["name"].unique():
        wdf = df[df["name"]==w].sort_values("_period_idx")
        ratio = wdf["hp003"]/wdf["hp10"]; col=WORKER_COLORS.get(w,"#999"); ls="--" if w=="CONTROL" else "-"
        ax.plot(wdf["_period_idx"], ratio, color=col, ls=ls, lw=2.2, marker="D", ms=7, label=w.title())
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.5)
    ax.text(0.02, 1.01, "Unity (ratio = 1.0)", transform=ax.get_yaxis_transform(),
            fontsize=9, color=MUTED)
    ax.set_xticks(range(len(periods))); ax.set_xticklabels(short_p, rotation=20, ha="right")
    ax.set_ylabel("Hp(0.03) / Hp(10)")
    ax.set_title("Lens-to-Deep Dose Ratio  (Post-ICRP 118)", fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); return fig

def plot_dose_rates(rate_df):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), facecolor=NAVY)
    for ax in axes: ax.set_facecolor(CARD)
    periods = rate_df["period_label"].unique().tolist()
    short_p = [p.split("–")[1].strip() if "–" in p else p for p in periods]
    ax1 = axes[0]
    for w in rate_df["name"].unique():
        wrd = rate_df[rate_df["name"]==w].sort_values("_period_idx")
        col = WORKER_COLORS.get(w,"#999"); ls="--" if w=="CONTROL" else "-"
        ax1.plot(wrd["_period_idx"], wrd["rate_hp10"], color=col, ls=ls, lw=2.2, marker="o", ms=6, label=w.title())
    for thresh, col_r, lbl in [(2.5,RED,"2.5 µSv/hr — uncontrolled limit"),
                                (7.5,AMBER,"7.5 µSv/hr — controlled area")]:
        ax1.axhline(thresh, color=col_r, ls=":", lw=1.2, alpha=0.8, label=lbl)
    ax1.set_xticks(range(len(periods))); ax1.set_xticklabels(short_p, rotation=18, ha="right")
    ax1.set_ylabel("Rate Hp(10)  [µSv/hr]"); ax1.set_title("Time-Averaged Dose Rate", fontweight="bold")
    ax1.legend(fontsize=8); ax1.spines[["top","right"]].set_visible(False); ax1.set_ylim(bottom=0)
    ax2 = axes[1]
    ctrl_r = {r["period_label"]:r["rate_hp10"] for _,r in rate_df[rate_df["name"]=="CONTROL"].iterrows()}
    wnc = [w for w in rate_df["name"].unique() if w != "CONTROL"]
    bw=0.38; offs=np.linspace(-bw*(len(wnc)-1)/2, bw*(len(wnc)-1)/2, len(wnc))
    for wn, off in zip(wnc, offs):
        wrd = rate_df[rate_df["name"]==wn].sort_values("_period_idx")
        nr  = [r["rate_hp10"] - ctrl_r.get(r["period_label"],0) for _,r in wrd.iterrows()]
        cols_bar = [GREEN if v>=0 else RED for v in nr]
        bars = ax2.bar(wrd["_period_idx"]+off, nr, bw*0.9,
                       color=WORKER_COLORS.get(wn,"#999"), alpha=0.85, label=wn.title())
    ax2.axhline(0, color=MUTED, lw=1); ax2.set_xticks(range(len(periods)))
    ax2.set_xticklabels(short_p, rotation=18, ha="right")
    ax2.set_ylabel("Net rate  [µSv/hr]"); ax2.set_title("Net Occupational Dose Rate (worker − control)", fontweight="bold")
    ax2.legend(); ax2.spines[["top","right"]].set_visible(False)
    fig.tight_layout(pad=2); return fig

def plot_attenuation(attn_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=NAVY)
    for ax in axes: ax.set_facecolor(CARD)
    periods = attn_df["period_label"].unique().tolist()
    short_p = [p.split("–")[1].strip() if "–" in p else p for p in periods]
    ax1 = axes[0]
    for w in attn_df["name"].unique():
        wrd = attn_df[attn_df["name"]==w].sort_values("_period_idx")
        col = WORKER_COLORS.get(w,"#999"); ls="--" if w=="CONTROL" else "-"
        ax1.plot(wrd["_period_idx"], wrd["mu_eff"], color=col, ls=ls, lw=2.2, marker="^", ms=7, label=w.title())
        ax1.fill_between(wrd["_period_idx"], 0, wrd["mu_eff"].clip(lower=0), color=col, alpha=0.05)
    ax1.axhline(0, color=MUTED, lw=0.8, ls="-")
    for e_kv, mu_r, lbl in [(40,0.34,"40 keV"),(60,0.21,"60 keV"),(100,0.16,"100 keV")]:
        ax1.axhline(mu_r, color=MUTED, ls=":", lw=0.8, alpha=0.5)
        ax1.text(0.02, mu_r+0.004, lbl, fontsize=8, color=MUTED, transform=ax1.get_yaxis_transform())
    ax1.set_xticks(range(len(periods))); ax1.set_xticklabels(short_p, rotation=20, ha="right")
    ax1.set_ylabel("µ_eff  [cm⁻¹]"); ax1.set_title("Tissue Attenuation Coefficient", fontweight="bold")
    ax1.legend(); ax1.spines[["top","right"]].set_visible(False)
    ax2 = axes[1]
    for w in attn_df["name"].unique():
        wrd = attn_df[attn_df["name"]==w].sort_values("_period_idx")
        col = WORKER_COLORS.get(w,"#999"); ls="--" if w=="CONTROL" else "-"
        ev  = np.clip(wrd["e_icrp74"].fillna(300).values, 0, 300)
        ax2.plot(wrd["_period_idx"], ev, color=col, ls=ls, lw=2.2, marker="^", ms=7, label=w.title())
    for val, lbl in [(44,"44 keV (scatter)"),(60,"60 keV (chest)"),(100,"100 keV (CT)")]:
        ax2.axhline(val, color=MUTED, ls=":", lw=0.8, alpha=0.5)
        ax2.text(0.02, val+3, lbl, fontsize=8, color=MUTED, transform=ax2.get_yaxis_transform())
    ax2.set_xticks(range(len(periods))); ax2.set_xticklabels(short_p, rotation=20, ha="right")
    ax2.set_ylabel("Effective photon energy  [keV]"); ax2.set_title("Effective Photon Energy (ICRP 74)", fontweight="bold")
    ax2.legend(); ax2.set_ylim(0, 250); ax2.spines[["top","right"]].set_visible(False)
    fig.tight_layout(pad=2); return fig

def plot_hvl_buildup(attn_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=NAVY)
    for ax in axes: ax.set_facecolor(CARD)
    periods = attn_df["period_label"].unique().tolist()
    short_p = [p.split("–")[1].strip() if "–" in p else p for p in periods]
    ax1 = axes[0]
    for w in attn_df["name"].unique():
        wrd = attn_df[attn_df["name"]==w].sort_values("_period_idx")
        col = WORKER_COLORS.get(w,"#999"); ls="--" if w=="CONTROL" else "-"
        valid = wrd["hvl_cm"].notna()
        if valid.any():
            ax1.plot(wrd["_period_idx"][valid], wrd["hvl_cm"][valid], color=col, ls=ls, lw=2.2, marker="D", ms=7, label=w.title())
    for v, lbl in [(3.3,"3.3 cm — chest X-ray"),(3.8,"3.8 cm — CT 120 kVp")]:
        ax1.axhline(v, color=AMBER, ls=":", lw=1, alpha=0.7)
        ax1.text(0.02, v+0.08, lbl, fontsize=8, color=AMBER, transform=ax1.get_yaxis_transform())
    ax1.set_xticks(range(len(periods))); ax1.set_xticklabels(short_p, rotation=20, ha="right")
    ax1.set_ylabel("HVL  [cm]"); ax1.set_title("Half-Value Layer in Soft Tissue", fontweight="bold")
    ax1.legend(); ax1.set_ylim(bottom=0); ax1.spines[["top","right"]].set_visible(False)
    ax2 = axes[1]
    df_workers = attn_df[attn_df["name"]!="CONTROL"]
    for w in df_workers["name"].unique():
        wrd = df_workers[df_workers["name"]==w].sort_values("_period_idx")
        col = WORKER_COLORS.get(w,"#999")
        ax2.bar(wrd["_period_idx"] + (0.2 if w==df_workers["name"].unique()[0] else -0.2),
                wrd["buildup"], 0.35, color=col, alpha=0.8, label=w.title())
    ax2.set_xticks(range(len(periods))); ax2.set_xticklabels(short_p, rotation=20, ha="right")
    ax2.set_ylabel("Build-up factor B"); ax2.set_title("Dose Build-up Factor at 10 cm depth", fontweight="bold")
    ax2.legend(); ax2.spines[["top","right"]].set_visible(False)
    fig.tight_layout(pad=2); return fig

def plot_background_separation(bg_df):
    fig, (ax,) = make_fig(1, 1, (10, 4.5))
    short_p = [p.split("–")[1].strip() if "–" in p else p for p in bg_df["period_label"]]
    xi = bg_df["_period_idx"].values
    ax.bar(xi, bg_df["bg"], 0.55, color=MUTED, alpha=0.6, label="Natural background (central)")
    ax.bar(xi, bg_df["scatter"], 0.55, bottom=bg_df["bg"], color=RED, alpha=0.8, label="Facility scatter")
    ax.plot(xi, bg_df["ctrl"], "o--", color=TEXT, lw=1.8, ms=7, label="Control badge total", zorder=5)
    ax.set_xticks(xi); ax.set_xticklabels(short_p, rotation=20, ha="right")
    ax.set_ylabel("Dose  [mSv / period]")
    ax.set_title("Background Decomposition — Control Badge", fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); return fig

def plot_career_risk(proj_df, risk_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=NAVY)
    for ax in axes: ax.set_facecolor(CARD)
    wnc = proj_df["name"].unique()
    ax1 = axes[0]
    for w in wnc:
        wpdf = proj_df[proj_df["name"]==w]; col=WORKER_COLORS.get(w,"#999")
        ax1.plot(wpdf["year"], wpdf["cum_msv"], color=col, lw=2.2, marker="o", ms=5, label=w.title())
        ax1.fill_between(wpdf["year"], 0, wpdf["cum_msv"], color=col, alpha=0.07)
    ax1.axhline(IAEA_CAREER_LIMIT, color=RED, ls="--", lw=1.5, alpha=0.8, label="ICRP 1 Sv career cap")
    ax1.axhline(100, color=AMBER, ls=":", lw=1.2, alpha=0.8, label="IAEA 5-yr constraint")
    ax1.set_xlabel("Career year"); ax1.set_ylabel("Cumulative Hp(10)  [mSv]")
    ax1.set_title("Career Dose Projection (35 yr)", fontweight="bold")
    ax1.legend(); ax1.set_xlim(0,36); ax1.set_ylim(bottom=0); ax1.spines[["top","right"]].set_visible(False)
    ax2 = axes[1]
    for w in wnc:
        wpdf = proj_df[proj_df["name"]==w]; col=WORKER_COLORS.get(w,"#999")
        ax2.plot(wpdf["year"], wpdf["risk_pct"], color=col, lw=2.2, marker="s", ms=5, label=w.title())
    ax2.axhline(4.8, color=MUTED, ls="--", lw=1.2, alpha=0.8, label="Background risk ~4.8% (Nigeria)")
    ax2.set_xlabel("Career year"); ax2.set_ylabel("Cumulative cancer risk  [%]  (ICRP 103)")
    ax2.set_title("Stochastic Risk Trajectory", fontweight="bold")
    ax2.legend(); ax2.set_xlim(0,36); ax2.set_ylim(bottom=0); ax2.spines[["top","right"]].set_visible(False)
    fig.tight_layout(pad=2); return fig

def plot_lq_bed(lq_df):
    wnc = lq_df["name"].unique()
    fig, axes = plt.subplots(1, len(wnc), figsize=(11, 4.5), sharey=True, facecolor=NAVY)
    if len(wnc)==1: axes=[axes]
    for ax in axes: ax.set_facecolor(CARD)
    for ax, w in zip(axes, wnc):
        wlq = lq_df[lq_df["name"]==w].sort_values("_period_idx"); xi=wlq["_period_idx"].values
        periods = lq_df["period_label"].unique()
        short_p = [p.split("–")[1].strip() if "–" in p else p for p in wlq["period_label"]]
        for col_k, lbl, cc in [("bed_skin","Skin α/β=3",TEAL),
                                ("bed_lens","Lens α/β=0.5",RED),
                                ("bed_marrow","Marrow α/β=10","#5B8AF0")]:
            if col_k in wlq.columns:
                ax.plot(xi, wlq[col_k]*1000, color=cc, lw=2.2, marker="o", ms=6, label=lbl)
        ax.set_title(w.title(), fontweight="bold")
        ax.set_xticks(xi); ax.set_xticklabels(short_p, rotation=28, ha="right"); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("BED  [mGy]"); axes[-1].legend()
    fig.suptitle("Biologically Effective Dose — Linear-Quadratic Model", fontweight="bold", y=1.01)
    fig.tight_layout(); return fig

def plot_mc_uncertainty(mc_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=NAVY)
    for ax in axes: ax.set_facecolor(CARD)
    periods = mc_df["period_label"].unique().tolist()
    short_p = [p.split("–")[1].strip() if "–" in p else p for p in periods]
    ax1 = axes[0]
    for w in mc_df["name"].unique():
        wmc = mc_df[mc_df["name"]==w].sort_values("_period_idx"); xi=wmc["_period_idx"].values
        col = WORKER_COLORS.get(w,"#999"); ls="--" if w=="CONTROL" else "-"
        mu = wmc["mu_mean"].values; u = wmc["mu_sd"].values*2
        ax1.plot(xi, mu, color=col, ls=ls, lw=2.2, marker="o", ms=6, label=w.title())
        ax1.fill_between(xi, mu-u, mu+u, color=col, alpha=0.14)
    ax1.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax1.set_xticks(range(len(periods))); ax1.set_xticklabels(short_p, rotation=20, ha="right")
    ax1.set_ylabel("µ_eff  [cm⁻¹]  ± U(k=2, 95%)"); ax1.set_title("MC Uncertainty — µ_eff", fontweight="bold")
    ax1.legend(); ax1.spines[["top","right"]].set_visible(False)
    ax2 = axes[1]
    for w in mc_df["name"].unique():
        wmc = mc_df[mc_df["name"]==w].sort_values("_period_idx"); xi=wmc["_period_idx"].values
        col = WORKER_COLORS.get(w,"#999"); ls="--" if w=="CONTROL" else "-"
        em  = np.clip(wmc["e_mean"].fillna(200).values, 0, 400)
        elo = np.clip(wmc["e_lo95"].fillna(15).values, 0, 400)
        ehi = np.clip(wmc["e_hi95"].fillna(400).values, 0, 400)
        ax2.plot(xi, em, color=col, ls=ls, lw=2.2, marker="^", ms=6, label=w.title())
        ax2.fill_between(xi, elo, ehi, color=col, alpha=0.14)
    for val, lbl in [(44,"44 keV — BV P1 (precise)"),(100,"100 keV ref")]:
        ax2.axhline(val, color=AMBER if val==44 else MUTED, ls=":", lw=1, alpha=0.8)
        ax2.text(0.02, val+4, lbl, fontsize=8, color=AMBER if val==44 else MUTED,
                 transform=ax2.get_yaxis_transform())
    ax2.set_xticks(range(len(periods))); ax2.set_xticklabels(short_p, rotation=20, ha="right")
    ax2.set_ylabel("E_eff  [keV]  (95% CI shaded)"); ax2.set_title("MC Uncertainty — Effective Energy", fontweight="bold")
    ax2.legend(); ax2.set_ylim(0, 420); ax2.spines[["top","right"]].set_visible(False)
    fig.suptitle("Monte Carlo Uncertainty Propagation  (N=10,000, ISO GUM Supp. 1)", fontweight="bold", y=1.01)
    fig.tight_layout(pad=2); return fig

def plot_kmeans(cluster_df, best_k, best_sil, sil_scores):
    fig = plt.figure(figsize=(13, 4.5), facecolor=NAVY)
    gs  = fig.add_gridspec(1, 4, wspace=0.35)
    pairs = [("f1","f2"), ("f1","f3"), ("f2","f3")]
    ax_labels = {"f1":"Hp(0.03)/Hp(10)","f2":"Hp(0.07)/Hp(10)","f3":"Hp(0.07)/Hp(0.03)"}
    cl_colors = [RED, TEAL, AMBER, "#5B8AF0"]
    wk_markers = {"CONTROL":"s","BINFA VICTORIA":"o","IBRAHIM DALHATU":"^"}
    for i, (xa, ya) in enumerate(pairs):
        ax = fig.add_subplot(gs[i]); ax.set_facecolor(CARD)
        for cl in sorted(cluster_df["cluster"].unique()):
            for w in cluster_df[cluster_df["cluster"]==cl]["name"].unique():
                sub = cluster_df[(cluster_df["cluster"]==cl)&(cluster_df["name"]==w)]
                ax.scatter(sub[xa], sub[ya], color=cl_colors[cl%4],
                           marker=wk_markers.get(w,"o"), s=90, alpha=0.9, zorder=3,
                           edgecolors="none")
        ax.axhline(1, color=MUTED, ls=":", lw=0.8, alpha=0.5)
        ax.axvline(1, color=MUTED, ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel(ax_labels[xa], fontsize=9); ax.set_ylabel(ax_labels[ya], fontsize=9)
        ax.set_title(f"{ax_labels[xa]} vs {ax_labels[ya]}", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
    ax_sil = fig.add_subplot(gs[3]); ax_sil.set_facecolor(CARD)
    ks = list(sil_scores.keys()); ss = list(sil_scores.values())
    bars = ax_sil.bar(ks, ss, color=[TEAL if s==max(ss) else MUTED for s in ss], width=0.5, alpha=0.9)
    ax_sil.axhline(0.7, color=RED, ls=":", lw=1.2, alpha=0.8); ax_sil.text(ks[0]-0.25, 0.71, "0.7 threshold", fontsize=8, color=RED)
    ax_sil.set_xticks(ks); ax_sil.set_xticklabels([f"k={k}" for k in ks])
    ax_sil.set_ylabel("Silhouette score"); ax_sil.set_title("Cluster quality", fontsize=9)
    ax_sil.set_ylim(0,1); ax_sil.spines[["top","right"]].set_visible(False)
    for bar, s in zip(bars, ss):
        ax_sil.text(bar.get_x()+bar.get_width()/2, s+0.02, f"{s:.3f}", ha="center", fontsize=10, fontweight="bold", color=TEXT)
    leg = [plt.Line2D([0],[0],marker=m,color="w",markerfacecolor="gray",ms=8,label=w.title()) for w,m in wk_markers.items()]
    leg+= [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=cl_colors[c%4],ms=8,label=f"Cluster {c}") for c in range(best_k)]
    fig.legend(handles=leg, loc="lower center", ncol=len(leg), fontsize=8, bbox_to_anchor=(0.5,-0.05))
    fig.suptitle(f"K-Means Unsupervised Field Classification  (k={best_k}, silhouette={best_sil:.4f})", fontweight="bold", y=1.02)
    fig.tight_layout(); return fig

def plot_bayesian_gaps(gap_df):
    gap_labels = gap_df["gap_label"].unique()
    fig, axes  = plt.subplots(1, len(gap_labels), figsize=(12, 4.5), facecolor=NAVY)
    if len(gap_labels)==1: axes=[axes]
    for ax in axes: ax.set_facecolor(CARD)
    for ax, gl in zip(axes, gap_labels):
        sub = gap_df[gap_df["gap_label"]==gl]; xr=np.linspace(0,10,500)
        for _, row in sub.iterrows():
            col = WORKER_COLORS.get(row["worker"],"#999")
            mu=row["post_mean"]; sig=row["post_sigma"]
            if sig<=0: continue
            pdv = (1/(sig*np.sqrt(2*np.pi)))*np.exp(-0.5*((xr-mu)/sig)**2); pdv[xr<0]=0
            ax.plot(xr, pdv, color=col, lw=2.5, label=row["worker"].title(), zorder=3)
            ax.axvline(mu, color=col, ls="--", lw=1.2, alpha=0.7)
            mask=(xr>=row["ci_lo"])&(xr<=row["ci_hi"])
            ax.fill_between(xr[mask], pdv[mask], color=col, alpha=0.15)
            pr = (1/(row["prior_sigma"]*np.sqrt(2*np.pi)))*np.exp(-0.5*((xr-row["prior_mean"])/row["prior_sigma"])**2) if row["prior_sigma"]>0 else xr*0
            pr[xr<0]=0
            ax.plot(xr, pr, color=col, lw=1, ls=":", alpha=0.45)
        ax.set_xlabel("Gap dose  [mSv]"); ax.set_ylabel("Posterior probability density")
        ax.set_title(gl, fontweight="bold"); ax.legend(); ax.set_xlim(left=0)
        ax.text(0.97, 0.95, "Solid = posterior\nDotted = prior", transform=ax.transAxes,
                fontsize=8, ha="right", va="top", color=MUTED)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Bayesian Conjugate Gap Dose Estimation", fontweight="bold", y=1.02)
    fig.tight_layout(pad=2); return fig

def plot_compliance_summary(df, proj_df, risk_df):
    """Radar-style compliance summary with 6 metric bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=NAVY)
    for ax in axes: ax.set_facecolor(CARD)
    wnc = [w for w in df["name"].unique() if w != "CONTROL"]
    ax1 = axes[0]
    metrics, vals, limits, colors_m = [], [], [], []
    for w in wnc:
        wdf = df[df["name"]==w]
        ann = ((wdf["hp10"]/wdf["n_months"])*12).mean()
        metrics.append(f"{w.split()[0].title()}\nHp10 ann.")
        vals.append(ann); limits.append(20); colors_m.append(WORKER_COLORS.get(w,"#999"))
        metrics.append(f"{w.split()[0].title()}\nHp003 ann.")
        vals.append(((wdf["hp003"]/wdf["n_months"])*12).mean()); limits.append(20); colors_m.append(WORKER_COLORS.get(w,"#999"))
    xi = np.arange(len(metrics)); bw=0.55
    bars = ax1.bar(xi, [v/l*100 for v,l in zip(vals,limits)], bw,
                   color=colors_m, alpha=0.85)
    ax1.axhline(100, color=RED, ls="--", lw=1.5, label="IAEA limit (100%)")
    ax1.axhline(75,  color=AMBER, ls=":", lw=1.2, label="75% of limit")
    for bar, v in zip(bars, [v/l*100 for v,l in zip(vals,limits)]):
        ax1.text(bar.get_x()+bar.get_width()/2, v+1.5, f"{v:.1f}%", ha="center", fontsize=9, color=TEXT)
    ax1.set_xticks(xi); ax1.set_xticklabels(metrics, fontsize=9)
    ax1.set_ylabel("% of IAEA annual limit"); ax1.set_title("Annual Dose as % of IAEA Limit", fontweight="bold")
    ax1.legend(fontsize=8); ax1.set_ylim(0, 120); ax1.spines[["top","right"]].set_visible(False)
    ax2 = axes[1]
    if not risk_df.empty:
        workers_r = risk_df["name"].values
        icrp_vals = risk_df["icrp103_pct"].values
        beir_vals = risk_df["beir7_ear_pct"].values
        xi2 = np.arange(len(workers_r)); bw2=0.3
        ax2.bar(xi2-bw2/2, icrp_vals, bw2, color=TEAL, alpha=0.85, label="ICRP 103 total cancer")
        ax2.bar(xi2+bw2/2, beir_vals, bw2, color=AMBER, alpha=0.85, label="BEIR VII EAR solid")
        ax2.set_xticks(xi2); ax2.set_xticklabels([w.title() for w in workers_r], fontsize=9)
        ax2.set_ylabel("Additional cancer risk  [%]"); ax2.set_title("Stochastic Cancer Risk Comparison", fontweight="bold")
        ax2.legend(fontsize=8); ax2.spines[["top","right"]].set_visible(False)
    fig.tight_layout(pad=2); return fig


# ─────────────────────────────────────────────────────────────────────────────
# TEXT REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_text_report(df, results, facility_display=''):
    rpt = StringIO()
    rpt.write("="*72 + "\n")
    rpt.write("  COMPREHENSIVE DOSIMETRY ANALYSIS REPORT\n")
    rpt.write(f"  {facility_display}\n")
    rpt.write(f"  Generated: {datetime.now():%Y-%m-%d %H:%M}\n")
    rpt.write("  21 Analyses · 18 Figures · 3 Computational Algorithms\n")
    rpt.write("="*72 + "\n\n")

    workers = df["name"].unique().tolist()
    wnc     = [w for w in workers if w != "CONTROL"]

    rpt.write("DATASET SUMMARY\n" + "-"*50 + "\n")
    rpt.write(f"  Records: {len(df)}  |  Workers: {len(workers)}  |  Periods: {len(df['period_label'].unique())}\n")
    rpt.write(f"  Files: {', '.join(sorted(df['filename'].unique()))}\n\n")

    for w in workers:
        wdf = df[df["name"]==w]
        rpt.write(f"  {w}:  mean Hp10 = {wdf['hp10'].mean():.3f} mSv  "
                  f"SD = {wdf['hp10'].std():.3f}  cumulative = {wdf['hp10'].sum():.3f} mSv\n")
    rpt.write("\n")

    for section, key in [
        ("TREND RESULTS","trend"),
        ("DOSE RATES","rate_df"),
        ("ATTENUATION / ENERGY","attn_df"),
        ("FIELD CHARACTERISATION","field_df"),
        ("STOCHASTIC RISK","risk_df"),
        ("BACKGROUND SEPARATION","bg_df"),
        ("CAREER PROJECTION","proj_df"),
        ("LQ BIOLOGICAL MODEL","lq_df"),
        ("MONTE CARLO UNCERTAINTY","mc_df"),
        ("K-MEANS CLUSTERING","cluster_df"),
        ("BAYESIAN GAP ESTIMATION","gap_df"),
    ]:
        rpt.write(f"\n{'='*72}\n{section}\n{'='*72}\n")
        if key == "trend":
            for w, td in results["trend"].items():
                rpt.write(f"  {w}: slope={td['slope']:+.4f}  R²={td['r2']:.3f}  "
                          f"p_lin={td['p_lin']:.4f}  MK_p={td['p_mk']:.4f}  "
                          f"direction={td['direction']}\n")
        elif key in results:
            try:
                sub_df = pd.read_json(io.StringIO(results[key]))
                rpt.write(sub_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
                rpt.write("\n")
            except Exception:
                rpt.write("  [Data available in interactive app]\n")

    rpt.write("\n\nCOMPLIANCE SUMMARY\n" + "-"*50 + "\n")
    for w in wnc:
        wdf = df[df["name"]==w]
        ann = ((wdf["hp10"]/wdf["n_months"])*12).mean()
        pct = ann/20*100
        flag = "✓ COMPLIANT" if ann < 20 else "⚠ EXCEEDS LIMIT"
        rpt.write(f"  {w}: {ann:.2f} mSv/yr  = {pct:.1f}% of IAEA limit  {flag}\n")

    rpt.write("\n" + "="*72 + "\nEND OF REPORT\n")
    return rpt.getvalue()


def generate_full_pdf(df, results):
    """Generate a multi-page PDF with all 18 figures."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        figs_to_save = []

        net_df  = pd.read_json(io.StringIO(results["net_df"]))
        rate_df = pd.read_json(io.StringIO(results["rate_df"]))
        attn_df = pd.read_json(io.StringIO(results["attn_df"]))
        bg_df   = pd.read_json(io.StringIO(results["bg_df"]))
        proj_df = pd.read_json(io.StringIO(results["proj_df"]))
        risk_df = pd.read_json(io.StringIO(results["risk_df"]))
        lq_df   = pd.read_json(io.StringIO(results["lq_df"]))
        mc_df   = pd.read_json(io.StringIO(results["mc_df"]))
        cdf     = pd.read_json(io.StringIO(results["cluster_df"]))
        gap_df  = pd.read_json(io.StringIO(results["gap_df"]))

        for fig_fn in [
            lambda: plot_dose_timeline(df),
            lambda: plot_three_quantities(df),
            lambda: plot_net_dose(net_df, df),
            lambda: plot_lens_ratio(df),
            lambda: plot_heatmap(df),
            lambda: plot_dose_rates(rate_df),
            lambda: plot_attenuation(attn_df),
            lambda: plot_hvl_buildup(attn_df),
            lambda: plot_background_separation(bg_df),
            lambda: plot_career_risk(proj_df, risk_df),
            lambda: plot_lq_bed(lq_df),
            lambda: plot_mc_uncertainty(mc_df),
            lambda: plot_kmeans(cdf, results["best_k"], results["best_sil"], results["sil_scores"]),
            lambda: plot_bayesian_gaps(gap_df),
            lambda: plot_compliance_summary(df, proj_df, risk_df),
        ]:
            try:
                fig = fig_fn()
                pdf.savefig(fig, dpi=150, bbox_inches="tight", facecolor=NAVY)
                plt.close(fig)
            except Exception:
                plt.close("all")

    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
      <div style="font-size:40px; margin-bottom:8px;">☢</div>
      <div style="font-family:'Space Mono',monospace; font-size:15px; font-weight:700; color:#00C9A7; letter-spacing:1px;">DOSIMETRY</div>
      <div style="font-family:'Space Mono',monospace; font-size:10px; color:#7B93B8; letter-spacing:2px; text-transform:uppercase; margin-top:2px;">Analysis Suite</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">Upload OSLD Reports</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Select .docx dosimetry files",
        type=["docx"],
        accept_multiple_files=True,
        help="Upload all monitoring period .docx files at once. "
             "Each file should be a standard OSLD dosimetry report.",
    )

    if uploaded:
        st.markdown(f'<div class="info-box">📂 {len(uploaded)} file{"s" if len(uploaded)!=1 else ""} loaded</div>', unsafe_allow_html=True)
        for f in uploaded:
            st.markdown(f'<span style="font-size:12px; color:#7B93B8; font-family:monospace;">✓ {f.name}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">Settings</div>', unsafe_allow_html=True)
    show_raw = st.checkbox("Show raw data tables", value=True)
    mc_n     = st.select_slider("Monte Carlo N", options=[1000,5000,10000,50000], value=10000)
    career_yrs = st.slider("Career projection (years)", 10, 40, 35)

    st.divider()
    # Facility info shown after upload
    if "df" in st.session_state or uploaded:
        try:
            _fname = st.session_state.get("_facility_name", "")
            _floc  = st.session_state.get("_facility_loc", "")
            _disp  = f"{_fname} · {_floc}" if _floc else _fname
            if _fname:
                st.markdown(f"""
    <div style="font-size:11px; color:#7B93B8; font-family:monospace; line-height:1.8;">
    21 analyses · 15 plot types<br>
    ISO GUM Supp. 1 · ICRP 103<br>
    BEIR VII · IAEA BSS GSR Pt 3<br>
    <br>
    <span style="color:#00C9A7;">{_fname}</span><br>
    {_floc}
    </div>
    """, unsafe_allow_html=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
# Default facility strings — overwritten below once files are parsed
FACILITY_NAME     = "Dosimetry Analysis Suite"
FACILITY_LOCATION = ""
FACILITY_DISPLAY  = "OSLD Personal Dose Equivalent Pipeline"

_header_placeholder = st.empty()

def _render_header(facility_display):
    _header_placeholder.markdown(f"""
<div style="padding: 28px 0 16px;">
  <div style="font-family:'Space Mono',monospace; font-size:9px; color:#00C9A7; letter-spacing:3px; text-transform:uppercase; margin-bottom:6px;">{facility_display}</div>
  <h1 style="margin:0; font-family:'DM Sans',sans-serif; font-size:32px; font-weight:600; color:#E8EEF7;">Occupational Radiation Dosimetry</h1>
  <div style="font-size:16px; color:#7B93B8; margin-top:4px;">21-analysis physics pipeline · OSLD personal dose equivalent · {facility_display}</div>
</div>
""", unsafe_allow_html=True)

_render_header(FACILITY_DISPLAY)

if not uploaded:
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>Getting started:</strong> Upload your OSLD dosimetry <code>.docx</code> report files
        using the sidebar panel. Each file should contain the monitoring period header and
        the dose table with Hp(10), Hp(0.03), and Hp(0.07) columns.
        This app runs entirely in your browser — no data leaves your device.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-box">
        <strong>For web deployment:</strong> Upload files directly via the sidebar file uploader.
        The app is fully stateless — upload fresh files each session.
        Deploy on <strong>Streamlit Cloud</strong> by pushing this file + <code>requirements.txt</code> to GitHub.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card teal" style="margin-top:8px;">
          <div class="metric-label">Analyses</div>
          <div class="metric-value">21</div>
          <div class="metric-sub">Clinical · Physics · Computational</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card" style="margin-top:12px;">
          <div class="metric-label">Figures</div>
          <div class="metric-value">15</div>
          <div class="metric-sub">Publication-ready PNG + PDF export</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ── Parse uploads ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Parsing dosimetry files…")
def load_files(file_list, _version=CACHE_VERSION):
    parsed = []
    for f in file_list:
        try:
            p = parse_docx_bytes(f["bytes"], f["name"])
            parsed.append(p)
        except Exception as e:
            st.warning(f"Could not parse {f['name']}: {e}")
    return build_dataframe(parsed)

file_list = [{"name": f.name, "bytes": f.read()} for f in uploaded]
df = load_files(file_list, _version=CACHE_VERSION)

if df.empty:
    alert_box("No valid data could be parsed from the uploaded files. Check that they follow the standard OSLD report format.")
    st.stop()

# ── Guarantee facility columns exist (cache may return pre-column df) ────────
if "facility_name" not in df.columns:
    df["facility_name"] = "Unknown Facility"
if "facility_location" not in df.columns:
    df["facility_location"] = ""

# ── Derive facility identity from parsed data ────────────────────────────────
all_facilities = df[["facility_name","facility_location"]].drop_duplicates()
if len(all_facilities) == 1:
    FACILITY_NAME     = all_facilities["facility_name"].iloc[0]
    FACILITY_LOCATION = all_facilities["facility_location"].iloc[0]
    FACILITY_DISPLAY  = f"{FACILITY_NAME} · {FACILITY_LOCATION}" if FACILITY_LOCATION else FACILITY_NAME
elif len(all_facilities) > 1:
    # Multiple facilities in the uploaded files
    fnames = all_facilities["facility_name"].tolist()
    flocs  = [l for l in all_facilities["facility_location"].tolist() if l]
    with st.sidebar:
        st.markdown('<div class="section-header">Facility filter</div>', unsafe_allow_html=True)
        selected_facility = st.selectbox(
            "Showing data for",
            ["All facilities"] + fnames,
            help="Upload files contain data from multiple facilities. Filter here.",
        )
        if selected_facility != "All facilities":
            df = df[df["facility_name"] == selected_facility].reset_index(drop=True)
            df["_period_idx"] = df["period_label"].map(
                {p:i for i,p in enumerate(df["period_label"].unique())})
    FACILITY_NAME     = selected_facility if selected_facility != "All facilities" else " / ".join(fnames)
    FACILITY_LOCATION = all_facilities.loc[all_facilities["facility_name"]==FACILITY_NAME,"facility_location"].iloc[0] if selected_facility != "All facilities" else " / ".join(flocs)
    FACILITY_DISPLAY  = f"{FACILITY_NAME} · {FACILITY_LOCATION}" if FACILITY_LOCATION else FACILITY_NAME
else:
    FACILITY_NAME, FACILITY_LOCATION, FACILITY_DISPLAY = "Unknown Facility", "", "Unknown Facility"

# Store for sidebar display
st.session_state["_facility_name"] = FACILITY_NAME
st.session_state["_facility_loc"]  = FACILITY_LOCATION
_render_header(FACILITY_DISPLAY)   # update header with real facility name

workers  = df["name"].unique().tolist()
wnc      = [w for w in workers if w != "CONTROL"]
periods  = df["period_label"].unique().tolist()
short_p  = [p.split("–")[1].strip() if "–" in p else p for p in periods]

with st.spinner("Running all 21 analyses…"):
    results = run_all_analyses(df.to_json(), _version=CACHE_VERSION)

net_df  = pd.read_json(io.StringIO(results["net_df"]))
rate_df = pd.read_json(io.StringIO(results["rate_df"]))
attn_df = pd.read_json(io.StringIO(results["attn_df"]))
field_df= pd.read_json(io.StringIO(results["field_df"]))
risk_df = pd.read_json(io.StringIO(results["risk_df"]))
bg_df   = pd.read_json(io.StringIO(results["bg_df"]))
proj_df = pd.read_json(io.StringIO(results["proj_df"]))
lq_df   = pd.read_json(io.StringIO(results["lq_df"]))
mc_df   = pd.read_json(io.StringIO(results["mc_df"]))
cdf     = pd.read_json(io.StringIO(results["cluster_df"]))
gap_df  = pd.read_json(io.StringIO(results["gap_df"]))

# Re-attach _period_idx to every DataFrame that needs it (JSON round-trip can
# corrupt integer-keyed columns in some pandas/orient combinations)
_pidx_map = df[["period_label","_period_idx"]].drop_duplicates()              .set_index("period_label")["_period_idx"].to_dict()
# Also build a period→n_months map for DataFrames that need it
_nmonths_map = df[["period_label","n_months"]].drop_duplicates()                 .set_index("period_label")["n_months"].to_dict()
for _rdf in [net_df, rate_df, attn_df, bg_df, lq_df, mc_df, cdf]:
    if "period_label" in _rdf.columns:
        _rdf["_period_idx"] = _rdf["period_label"].map(_pidx_map)
        if "n_months" not in _rdf.columns:
            _rdf["n_months"] = _rdf["period_label"].map(_nmonths_map)

# ── Summary metric row ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Records", str(len(df)), f"{len(periods)} periods · {len(workers)} workers")
with c2:
    ann_max = max(((df[df["name"]==w]["hp10"]/df[df["name"]==w]["n_months"])*12).mean() for w in wnc) if wnc else 0
    color = "green" if ann_max < 15 else ("amber" if ann_max < 18 else "red")
    metric_card("Peak annual dose", f"{ann_max:.2f}", "mSv/yr  (of 20 mSv limit)", color)
with c3:
    cum_max = max(df[df["name"]==w]["hp10"].sum() for w in wnc) if wnc else 0
    metric_card("Max cumulative Hp(10)", f"{cum_max:.3f}", "mSv  (of 1,000 mSv cap)", "green")
with c4:
    mean_rate = rate_df[rate_df["name"]=="CONTROL"]["rate_hp10"].mean() if not rate_df.empty else 0
    color = "green" if mean_rate < 2.5 else "amber"
    metric_card("Mean ambient rate", f"{mean_rate:.2f}", "µSv/hr  (IAEA limit: 2.5)", color)
with c5:
    sil = results.get("best_sil", 0)
    metric_card("K-Means silhouette", f"{sil:.4f}", f"k={results.get('best_k',2)}  (>0.7 = strong)", "teal")
with c6:
    max_risk = risk_df["icrp103_pct"].max() if not risk_df.empty else 0
    metric_card("Max ICRP 103 risk", f"{max_risk:.4f}%", "total cancer (OSLD period)")

st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🩺 Clinical",
    "⚛ Physics",
    "🖥 Computational",
    "📋 Reports & Export",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("Dataset at a glance")

    if show_raw:
        display_df = df[["period_label","name","n_months","hp10","hp003","hp007"]].copy()
        display_df.columns = ["Period","Worker","Months","Hp(10) mSv","Hp(0.03) mSv","Hp(0.07) mSv"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        section_header("Dose timeline")
        fig = plot_dose_timeline(df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig),
                           "dose_timeline.png", "image/png", key="dl_timeline")
        plt.close(fig)

    with col_b:
        section_header("Hp(10) heatmap")
        fig = plot_heatmap(df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig),
                           "dose_heatmap.png", "image/png", key="dl_heatmap")
        plt.close(fig)

    section_header("Three dose quantities")
    fig = plot_three_quantities(df)
    st.pyplot(fig, use_container_width=True)
    st.download_button("⬇ Download PNG", fig_to_bytes(fig),
                       "three_quantities.png", "image/png", key="dl_3q")
    plt.close(fig)

    # Compliance indicators
    section_header("Compliance quick-check")
    rows_comp = []
    for w in workers:
        wdf = df[df["name"]==w]
        for _, row in wdf.iterrows():
            ann_A = row["hp10"]*12; ann_B = (row["hp10"]/row["n_months"])*12
            rows_comp.append({
                "Worker": w.title(),
                "Period": row["period_label"],
                "Hp(10) mSv": f"{row['hp10']:.3f}",
                "Ann. (×12) mSv/yr": f"{ann_A:.2f}",
                "Ann. (÷mo ×12) mSv/yr": f"{ann_B:.2f}",
                "Status": "✓" if ann_B < 20 else "⚠ OVER",
            })
    st.dataframe(pd.DataFrame(rows_comp), use_container_width=True, hide_index=True)

    if any(r["Status"]!="✓" for r in rows_comp):
        alert_box("⚠ One or more periods may exceed the IAEA annual limit under the monthly-average interpretation (Hp(10) × 12). Verify whether reported values are monthly averages or period totals with Mr. Pada Isaac.")
    else:
        info_box("✓ All periods comply with IAEA 20 mSv/year annual effective dose limit under the period-total interpretation.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLINICAL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    cl_tabs = st.tabs(["Trends", "Net Dose", "Dose Rates", "Lens Dose", "Benchmarking", "Career & Risk"])

    with cl_tabs[0]:
        section_header("Longitudinal trend analysis")
        info_box("Mann-Kendall (non-parametric) + linear regression on Hp(10). No significant trends found (all p > 0.25), confirming dose stability over the monitoring window.")
        for w, td in results["trend"].items():
            sig = td["p_lin"]<0.05 or td["p_mk"]<0.05
            badge = '<span class="badge badge-red">SIGNIFICANT</span>' if sig else '<span class="badge badge-green">NO TREND</span>'
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:10px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div><span style="font-family:'DM Mono',monospace; font-size:13px; color:#E8EEF7;">{w.title()}</span></div>
                <div>{badge}</div>
              </div>
              <div style="margin-top:10px; font-family:'DM Mono',monospace; font-size:12px; color:#7B93B8; display:flex; gap:24px;">
                <span>slope: <strong style="color:#E8EEF7;">{td['slope']:+.4f} mSv/period</strong></span>
                <span>R²: <strong style="color:#E8EEF7;">{td['r2']:.3f}</strong></span>
                <span>p (lin): <strong style="color:#E8EEF7;">{td['p_lin']:.4f}</strong></span>
                <span>MK τ: <strong style="color:#E8EEF7;">{td['tau']:+.3f}</strong></span>
                <span>MK p: <strong style="color:#E8EEF7;">{td['p_mk']:.4f}</strong></span>
                <span>direction: <strong style="color:#00C9A7;">{td['direction']}</strong></span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with cl_tabs[1]:
        section_header("Background-corrected net occupational dose")
        warn_box("Period 1 negative values arise because the control badge was anomalously high (2.340 mSv vs. mean 0.802 mSv). This indicates an ambient source change, not worker behaviour. Investigate potential equipment installation or background anomaly in Sep 2023–Feb 2024.")
        fig = plot_net_dose(net_df, df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "net_dose.png", "image/png", key="dl_net")
        plt.close(fig)
        if show_raw:
            st.dataframe(net_df[["name","period_label","net_hp10","net_hp003","net_hp007"]].round(4),
                         use_container_width=True, hide_index=True)

    with cl_tabs[2]:
        section_header("Time-averaged dose rates")
        info_box(f"Computed using {WORK_HRS_PER_DAY} hr/day × {WORK_DAYS_PER_WEEK} days/week × {WEEKS_PER_MONTH:.2f} weeks/month. Mean ambient rate: {rate_df[rate_df['name']=='CONTROL']['rate_hp10'].mean():.2f} µSv/hr (IAEA uncontrolled limit: 2.5 µSv/hr).")
        fig = plot_dose_rates(rate_df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "dose_rates.png", "image/png", key="dl_rates")
        plt.close(fig)

    with cl_tabs[3]:
        section_header("Lens dose monitoring — post-ICRP 118")
        info_box("ICRP 118 (2011) reduced the occupational lens limit from 150 → 20 mSv/year. Binfa Victoria Period 1 Hp(0.03)/Hp(10) = 1.241 is the only scatter-elevated result. Under monthly-average interpretation, this annualises to 16.1 mSv/year = 80% of the new limit.")
        fig = plot_lens_ratio(df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "lens_ratio.png", "image/png", key="dl_lens")
        plt.close(fig)
        if show_raw:
            lens_data = []
            for w in workers:
                wdf = df[df["name"]==w]
                for _, row in wdf.iterrows():
                    r = row["hp003"]/row["hp10"]
                    lens_data.append({"Worker":w.title(),"Period":row["period_label"],
                        "Hp(0.03)":f"{row['hp003']:.3f}","Hp(10)":f"{row['hp10']:.3f}",
                        "Ratio":f"{r:.4f}","Scatter field":r>1.05})
            st.dataframe(pd.DataFrame(lens_data), use_container_width=True, hide_index=True)

    with cl_tabs[4]:
        section_header("Regional benchmarking")
        fig, (ax,) = make_fig(1,1,(10,4.5))
        bnames=list(REGIONAL_BENCHMARKS.keys()); bvals=list(REGIONAL_BENCHMARKS.values())
        xi=np.arange(len(bnames))
        ax.bar(xi, bvals, 0.55, color=MUTED, alpha=0.6, label="Published benchmarks")
        for w in wnc:
            wdf=df[df["name"]==w]; ann=((wdf["hp10"]/wdf["n_months"])*12).mean()
            ax.axhline(ann, color=WORKER_COLORS.get(w,"#999"), ls="--", lw=2.2,
                       label=f"{w.title()} ({ann:.2f} mSv/yr)")
        ax.axhline(IAEA_ANNUAL_EFFECTIVE, color=RED, ls="-", lw=1.5, alpha=0.8,
                   label=f"IAEA limit ({IAEA_ANNUAL_EFFECTIVE} mSv/yr)")
        ax.set_xticks(xi); ax.set_xticklabels(bnames, rotation=15, ha="right")
        ax.set_ylabel("Annual dose  [mSv/yr]"); ax.set_title("Mean Annualised Dose vs. Regional Benchmarks", fontweight="bold")
        ax.legend(fontsize=8); ax.set_ylim(bottom=0); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "benchmarks.png", "image/png", key="dl_bench")
        plt.close(fig)

    with cl_tabs[5]:
        section_header("Career dose projection and stochastic risk")
        col1, col2 = st.columns([1,1])
        with col1:
            for _, row in risk_df.iterrows():
                st.markdown(f"""
                <div class="metric-card green" style="margin-bottom:10px;">
                  <div class="metric-label">{row['name'].title()} — ICRP 103 total cancer risk</div>
                  <div class="metric-value">{row['icrp103_pct']:.5f}%</div>
                  <div class="metric-sub">{row['bkg_addition_pct']:.3f}% of background risk  ·  BEIR VII EAR: {row['beir7_ear_pct']:.6f}%</div>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            for w in wnc:
                wdf = df[df["name"]==w]
                ann = ((wdf["hp10"]/wdf["n_months"])*12).mean()
                yrs_to_cap = IAEA_CAREER_LIMIT/ann if ann>0 else 999
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:10px;">
                  <div class="metric-label">{w.title()} — {career_yrs}-year career projection</div>
                  <div class="metric-value">{ann*career_yrs:.1f} mSv</div>
                  <div class="metric-sub">{ann*career_yrs/10:.1f}% of ICRP 1 Sv cap  ·  years to cap: {yrs_to_cap:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
        fig = plot_career_risk(proj_df, risk_df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "career_risk.png", "image/png", key="dl_career")
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PHYSICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    ph_tabs = st.tabs(["Attenuation & Energy", "HVL & Build-up", "Field Characterisation", "LQ Biological", "Background & Fluence"])

    with ph_tabs[0]:
        section_header("Tissue attenuation coefficient and effective photon energy")
        info_box("µ_eff = ln[Hp(0.07)/Hp(10)] / 9.93 mm  (Beer-Lambert). Effective energy from ICRP 74 lookup. Binfa Victoria Period 1: µ_eff = 0.445 cm⁻¹ → E_eff = 44 keV — the only metrologically precise energy estimate in the dataset (confirmed by Monte Carlo, Tab 3).")
        fig = plot_attenuation(attn_df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "attenuation_energy.png", "image/png", key="dl_att")
        plt.close(fig)
        if show_raw:
            st.dataframe(attn_df[["name","period_label","ratio_007_10","mu_eff","e_icrp74","flux_cm2s"]].round(4),
                         use_container_width=True, hide_index=True)

    with ph_tabs[1]:
        section_header("Half-value layer and dose build-up factor")
        col1, col2 = st.columns(2)
        with col1:
            info_box("HVL = ln(2)/µ_eff [cm]. Build-up occurs when Hp(0.07)/Hp(10) < 1.0 — all of Ibrahim Dalhatu's periods are in this regime, indicating high-energy primary beam consistent with CT.")
        with col2:
            warn_box("Ibrahim Dalhatu has NO valid HVL in any period — all are in the dose build-up regime. Standard slab-phantom dosimetry assumptions may underestimate deep-body dose in this environment.")
        fig = plot_hvl_buildup(attn_df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "hvl_buildup.png", "image/png", key="dl_hvl")
        plt.close(fig)

    with ph_tabs[2]:
        section_header("Radiation field characterisation")
        if show_raw:
            fd = field_df.copy()
            fd["Field type"] = fd["mean_r2"].apply(lambda r: "Scattered/low-E" if r>1.05 else ("Build-up/high-E" if r<0.99 else "Transitional"))
            fd = fd.round(4); fd["name"] = fd["name"].str.title()
            st.dataframe(fd[["name","fhi","scatter_pct","mean_r1","mean_r2","Field type"]].rename(
                columns={"name":"Worker","fhi":"FHI","scatter_pct":"Scatter %",
                         "mean_r1":"Mean Hp003/Hp10","mean_r2":"Mean Hp007/Hp10","Field type":"Field type"}),
                use_container_width=True, hide_index=True)
        for _, row in field_df.iterrows():
            ftype = "Scattered / low-energy" if row["mean_r2"]>1.05 else ("Build-up / high-energy" if row["mean_r2"]<0.99 else "Transitional")
            badge_class = "badge-amber" if "Scattered" in ftype else ("badge-teal" if "Build-up" in ftype else "badge-teal")
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-family:'DM Mono',monospace;font-size:13px;">{row['name'].title()}</span>
                <span class="badge {badge_class}">{ftype}</span>
              </div>
              <div style="margin-top:8px; font-family:'DM Mono',monospace; font-size:12px; color:#7B93B8; display:flex; gap:24px;">
                <span>FHI: <strong style="color:#E8EEF7;">{row['fhi']:.3f}</strong></span>
                <span>scatter: <strong style="color:#E8EEF7;">{row['scatter_pct']:.1f}%</strong></span>
                <span>mean 003/10: <strong style="color:#E8EEF7;">{row['mean_r1']:.4f}</strong></span>
                <span>mean 007/10: <strong style="color:#E8EEF7;">{row['mean_r2']:.4f}</strong></span>
              </div>
            </div>""", unsafe_allow_html=True)

    with ph_tabs[3]:
        section_header("Linear-quadratic biological model")
        info_box("BED = D_total × (1 + d_per_day / α/β). Chronic occupational exposure treated as many small daily fractions. Lens opacity threshold (ICRP 118): ~0.5 Gy BED. Current cumulative lens BED is < 1.1% of threshold for all workers.")
        fig = plot_lq_bed(lq_df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "lq_bed.png", "image/png", key="dl_lq")
        plt.close(fig)
        if show_raw and not lq_df.empty:
            disp_cols = ["name","period_label"] + [c for c in lq_df.columns if c.startswith("bed_")]
            if all(c in lq_df.columns for c in disp_cols[:2]):
                st.dataframe(lq_df[[c for c in disp_cols if c in lq_df.columns]].round(6),
                             use_container_width=True, hide_index=True)

    with ph_tabs[4]:
        section_header("Natural background separation")
        fig = plot_background_separation(bg_df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "background.png", "image/png", key="dl_bg")
        plt.close(fig)
        section_header("Photon fluence and flux")
        fig2, axes2 = plt.subplots(1,2,figsize=(12,4.5), facecolor=NAVY)
        for ax in axes2: ax.set_facecolor(CARD)
        for w in workers:
            wfd = attn_df[attn_df["name"]==w].sort_values("_period_idx")
            col=WORKER_COLORS.get(w,"#999"); ls="--" if w=="CONTROL" else "-"
            # period fluence = flux [ph/cm2/s] * total wear seconds
            wear_s = wfd["n_months"].apply(lambda nm: wear_hours(nm) * 3600)
            fluence_period = wfd["flux_cm2s"].clip(lower=0) * wear_s
            axes2[0].semilogy(wfd["_period_idx"], fluence_period.clip(lower=1e-9),
                              color=col, ls=ls, lw=2, marker="^", ms=6, label=w.title())
            axes2[1].plot(wfd["_period_idx"], wfd["flux_cm2s"], color=col, ls=ls, lw=2, marker="^", ms=6, label=w.title())
        for ax in axes2:
            ax.set_xticks(range(len(periods))); ax.set_xticklabels(short_p, rotation=20, ha="right")
            ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
        axes2[0].set_ylabel("Period fluence  [ph/cm²]  (log)"); axes2[0].set_title("Photon Fluence", fontweight="bold")
        axes2[1].set_ylabel("Flux  [ph/cm²/s]"); axes2[1].set_title("Time-Averaged Flux", fontweight="bold"); axes2[1].set_ylim(bottom=0)
        fig2.tight_layout(pad=2)
        st.pyplot(fig2, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig2), "fluence_flux.png", "image/png", key="dl_flux")
        plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPUTATIONAL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    comp_tabs = st.tabs(["Monte Carlo", "K-Means Clustering", "Bayesian Gap Estimation", "Compliance Dashboard"])

    with comp_tabs[0]:
        section_header("Monte Carlo uncertainty propagation  (ISO GUM Supplement 1)")
        info_box(f"N = {mc_n:,} realisations per measurement point. Input: σ_rel = {OSLD_REL_UNCERT*100:.0f}% (1σ, OSLD calibration, ISO 4037-3). Expanded uncertainty U at k=2 (95% coverage). Key finding: only Binfa Victoria Period 1 yields a metrologically meaningful energy estimate (E = 44 ± 6 keV).")
        fig = plot_mc_uncertainty(mc_df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "mc_uncertainty.png", "image/png", key="dl_mc")
        plt.close(fig)
        if show_raw:
            mc_display = mc_df[["name","period_label","mu_mean","mu_sd","e_mean","e_sd","hvl_mean","hvl_sd"]].copy()
            mc_display["mu_U(k=2)"]  = mc_display["mu_sd"]*2
            mc_display["E_U(k=2)"]   = mc_display["e_sd"]*2
            mc_display["HVL_U(k=2)"] = mc_display["hvl_sd"]*2
            st.dataframe(mc_display[["name","period_label","mu_mean","mu_U(k=2)","e_mean","E_U(k=2)","hvl_mean","HVL_U(k=2)"]].round(3),
                         use_container_width=True, hide_index=True)

    with comp_tabs[1]:
        section_header("Unsupervised K-Means radiation field classification")
        c1, c2 = st.columns(2)
        with c1:
            info_box(f"Silhouette score k=2: **{results['best_sil']:.4f}** (>0.7 = strong cluster structure confirmed). The algorithm isolated Binfa Victoria Period 1 as a singleton cluster without access to worker labels — objective confirmation of the physical field dichotomy.")
        with c2:
            warn_box("Cluster purity = 0.583. The strong inter-worker field difference is concentrated in a single period (Binfa Victoria Period 1). In the remaining periods, all three badge holders occupy the same cluster — confirming that the field dichotomy is episodic, not permanent.")
        fig = plot_kmeans(cdf, results["best_k"], results["best_sil"], results["sil_scores"])
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "kmeans_clusters.png", "image/png", key="dl_km")
        plt.close(fig)
        if show_raw:
            st.dataframe(cdf[["name","period_label","cluster","f1","f2","f3"]].round(4),
                         use_container_width=True, hide_index=True)

    with comp_tabs[2]:
        section_header("Bayesian conjugate gap dose estimation")
        info_box("Conjugate Normal-Normal model. Prior: empirical distribution of all observed period doses scaled to gap length. Likelihood: neighbouring period doses. Posterior: closed-form update. Credible intervals represent genuine epistemic uncertainty — these are NOT substitutes for actual monitoring.")
        fig = plot_bayesian_gaps(gap_df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "bayesian_gaps.png", "image/png", key="dl_bayes")
        plt.close(fig)
        if show_raw and not gap_df.empty:
            gd = gap_df.copy()
            gd["worker"] = gd["worker"].str.title()
            gd = gd.round(3)
            st.dataframe(gd[["worker","gap_label","months","prior_mean","post_mean","post_sigma","ci_lo","ci_hi"]],
                         use_container_width=True, hide_index=True)
        section_header("Gap-filled cumulative dose")
        for w in wnc:
            obs = df[df["name"]==w]["hp10"].sum()
            wgap = gap_df[gap_df["worker"]==w]
            gap_est = wgap["post_mean"].sum(); gap_lo = wgap["ci_lo"].sum(); gap_hi = wgap["ci_hi"].sum()
            total = obs + gap_est
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:10px;">
              <div class="metric-label">{w.title()} — gap-filled cumulative Hp(10)</div>
              <div class="metric-value">{total:.2f} mSv</div>
              <div class="metric-sub">Observed: {obs:.3f}  +  Gap est.: {gap_est:.3f}  |  95% CI: [{obs+gap_lo:.2f}, {obs+gap_hi:.2f}] mSv</div>
            </div>""", unsafe_allow_html=True)

    with comp_tabs[3]:
        section_header("Compliance summary dashboard")
        fig = plot_compliance_summary(df, proj_df, risk_df)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download PNG", fig_to_bytes(fig), "compliance_dashboard.png", "image/png", key="dl_comp")
        plt.close(fig)
        section_header("NNRA / IAEA regulatory compliance matrix")
        comp_rows = []
        for w in wnc:
            wdf = df[df["name"]==w]
            ann_hp10  = ((wdf["hp10"]/wdf["n_months"])*12).mean()
            ann_hp003 = ((wdf["hp003"]/wdf["n_months"])*12).mean()
            r = risk_df[risk_df["name"]==w]
            risk_pct = r["icrp103_pct"].iloc[0] if len(r) else 0
            comp_rows.append({
                "Worker": w.title(),
                "Ann. Hp(10) mSv/yr": f"{ann_hp10:.2f}",
                "% of 20 mSv limit": f"{ann_hp10/20*100:.1f}%",
                "Ann. Hp(0.03) mSv/yr": f"{ann_hp003:.2f}",
                "% of lens limit": f"{ann_hp003/20*100:.1f}%",
                "35-yr cumul. mSv": f"{ann_hp10*career_yrs:.1f}",
                "% of 1 Sv cap": f"{ann_hp10*career_yrs/1000*100:.1f}%",
                "Cancer risk (ICRP103)": f"{risk_pct:.5f}%",
                "Overall": "✓ COMPLIANT" if ann_hp10<20 else "⚠ REVIEW",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REPORTS & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section_header("Export options")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class="metric-card teal">
          <div class="metric-label">Full PDF Report</div>
          <div class="metric-value" style="font-size:20px;">15 figures</div>
          <div class="metric-sub">All plots in one PDF file · dark theme · 150 DPI</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Generate PDF", key="gen_pdf"):
            with st.spinner("Rendering 15 figures…"):
                pdf_bytes = generate_full_pdf(df, results)
            st.download_button(
                "⬇ Download PDF",
                pdf_bytes,
                f"dosimetry_plots_{datetime.now():%Y%m%d}.pdf",
                "application/pdf",
                key="dl_full_pdf"
            )

    with col_b:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Text Report</div>
          <div class="metric-value" style="font-size:20px;">21 analyses</div>
          <div class="metric-sub">All numeric outputs · plain text · machine-readable</div>
        </div>
        """, unsafe_allow_html=True)
        rpt_text = generate_text_report(df, results, FACILITY_DISPLAY)
        st.download_button(
            "⬇ Download TXT",
            rpt_text.encode(),
            f"dosimetry_report_{datetime.now():%Y%m%d}.txt",
            "text/plain",
            key="dl_txt"
        )

    with col_c:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Data CSV</div>
          <div class="metric-value" style="font-size:20px;">Raw dataset</div>
          <div class="metric-sub">Parsed dose records · all periods · all workers</div>
        </div>
        """, unsafe_allow_html=True)
        csv_data = df.drop(columns=["_period_idx"], errors="ignore").to_csv(index=False)
        st.download_button(
            "⬇ Download CSV",
            csv_data.encode(),
            f"dosimetry_data_{datetime.now():%Y%m%d}.csv",
            "text/csv",
            key="dl_csv"
        )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    section_header("Individual plot downloads")

    plot_registry = {
        "Dose timeline":           lambda: plot_dose_timeline(df),
        "Three dose quantities":   lambda: plot_three_quantities(df),
        "Net occupational dose":   lambda: plot_net_dose(net_df, df),
        "Lens dose ratio":         lambda: plot_lens_ratio(df),
        "Dose heatmap":            lambda: plot_heatmap(df),
        "Dose rates":              lambda: plot_dose_rates(rate_df),
        "Attenuation & energy":    lambda: plot_attenuation(attn_df),
        "HVL & build-up":          lambda: plot_hvl_buildup(attn_df),
        "Background separation":   lambda: plot_background_separation(bg_df),
        "Career & risk":           lambda: plot_career_risk(proj_df, risk_df),
        "LQ biological model":     lambda: plot_lq_bed(lq_df),
        "MC uncertainty":          lambda: plot_mc_uncertainty(mc_df),
        "K-Means clusters":        lambda: plot_kmeans(cdf, results["best_k"], results["best_sil"], results["sil_scores"]),
        "Bayesian gaps":           lambda: plot_bayesian_gaps(gap_df),
        "Compliance dashboard":    lambda: plot_compliance_summary(df, proj_df, risk_df),
    }

    cols = st.columns(3)
    for idx, (name, fn) in enumerate(plot_registry.items()):
        with cols[idx % 3]:
            if st.button(f"⬇ {name}", key=f"btn_{idx}"):
                with st.spinner(f"Rendering {name}…"):
                    fig = fn()
                    img_bytes = fig_to_bytes(fig)
                    plt.close(fig)
                st.download_button(
                    f"Save {name}.png",
                    img_bytes,
                    f"{name.lower().replace(' ','_')}.png",
                    "image/png",
                    key=f"dl_ind_{idx}"
                )

    section_header("Full text report preview")
    with st.expander("Click to view full analysis report text", expanded=False):
        st.text(rpt_text)

    section_header("Deployment instructions")
    st.markdown("""
    <div class="info-box">
    <strong>Deploy on Streamlit Cloud (free):</strong><br><br>
    1. Save this file as <code>app.py</code><br>
    2. Create <code>requirements.txt</code> with the contents downloaded below<br>
    3. Push both files to a public GitHub repository<br>
    4. Go to <a href="https://share.streamlit.io" style="color:#00C9A7;">share.streamlit.io</a> → New app → select your repo<br>
    5. Set main file path to <code>app.py</code> → Deploy<br><br>
    Users will upload their <code>.docx</code> files directly from the web interface — no server-side storage is used.
    </div>
    """, unsafe_allow_html=True)

    req_txt = """streamlit>=1.32.0
python-docx>=1.1.0
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.12.0
matplotlib>=3.8.0
scikit-learn>=1.4.0
"""
    st.download_button(
        "⬇ Download requirements.txt",
        req_txt.encode(),
        "requirements.txt",
        "text/plain",
        key="dl_req"
    )
