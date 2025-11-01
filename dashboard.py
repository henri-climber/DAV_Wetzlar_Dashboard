# app.py — DAV Wetzlar Dashboard (multi-year, multi-file)

import io, os, re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from glob import glob

st.set_page_config(page_title="DAV Wetzlar – Results", layout="wide")

# ---------- utilities ----------

FILE_PAT = r"dav_wetzlar_results_(\d{4})_final\.(csv|xlsx)$"

def _excel_serial_to_dt(s):
    # Convert Excel serial day to Timestamp (Excel epoch 1899-12-30)
    return pd.to_datetime("1899-12-30") + pd.to_timedelta(s, unit="D")

def _coerce_dates(df, cols=("date_start","date_end")):
    for c in cols:
        if c not in df:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            df[c] = _excel_serial_to_dt(df[c])
        else:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    return df

def _extract_year_from_name(name: str):
    m = re.search(FILE_PAT, os.path.basename(name))
    return int(m.group(1)) if m else None

@st.cache_data(show_spinner=False)
def load_many(files_from_disk: list[tuple[str, bytes]], files_uploaded: list[tuple[str, bytes]]) -> pd.DataFrame:
    """files_*: list of (name, bytes). Merge all, normalize columns."""
    parts = []

    for name, data in (files_from_disk + files_uploaded):
        if name.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data))
        else:
            df = pd.read_excel(io.BytesIO(data))

        # normalize
        if "rank" in df: df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = _coerce_dates(df)

        # common text cleanup
        for c in ("name","team","category","discipline","event_name","location","url"):
            if c in df:
                df[c] = df[c].astype(str).str.strip()

        # derive year (from date and from filename as fallback)
        file_year = _extract_year_from_name(name)
        if "date_start" in df:
            df["year"] = df["date_start"].dt.year
            df["year"] = df["year"].fillna(file_year)
        else:
            df["year"] = file_year

        # keep a source column (debugging/auditing)
        df["source_file"] = os.path.basename(name)
        df["source_year"] = file_year

        parts.append(df)

    if not parts:
        return pd.DataFrame()

    df_all = pd.concat(parts, ignore_index=True)

    # engineering columns
    if "rank" in df_all:
        df_all["podium"] = df_all["rank"].between(1,3)
        df_all["top10"]  = df_all["rank"].le(10)

    # drop exact duplicates
    keep_keys = [c for c in ["rank","name","team","category","discipline","event_name","url","source_file"] if c in df_all.columns]
    if keep_keys:
        df_all = df_all.drop_duplicates(subset=keep_keys)

    return df_all

# ---------- load files ----------

# auto-discover yearly files on disk
disk_files = []
for path in glob("dav_wetzlar_results_*_final.csv") + glob("dav_wetzlar_results_*_final.xlsx"):
    with open(path, "rb") as f:
        disk_files.append((os.path.basename(path), f.read()))

st.sidebar.header("Data")
uploads = st.sidebar.file_uploader("Upload one or more files", type=["csv","xlsx"], accept_multiple_files=True)
uploaded_files = [(u.name, u.getvalue()) for u in uploads] if uploads else []

df = load_many(disk_files, uploaded_files)

if df.empty:
    st.info("No data found. Put files like `dav_wetzlar_results_2022_final.xlsx` next to the app or upload them in the sidebar.")
    st.stop()

st.title("DAV Wetzlar – Competition Evolution Dashboard")

# ---------- filters ----------

with st.sidebar:
    st.subheader("Filters")

    years = sorted([int(y) for y in df["year"].dropna().unique()]) if "year" in df else []
    if years:
        yr_min, yr_max = min(years), max(years)
        year_range = st.slider("Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max), step=1)
    else:
        year_range = (None, None)

    dis_choices = sorted(df["discipline"].dropna().unique()) if "discipline" in df else []
    sel_dis = st.multiselect("Discipline", dis_choices, default=dis_choices)

    cat_choices = sorted(df["category"].dropna().unique()) if "category" in df else []
    sel_cat = st.multiselect("Category", cat_choices, default=cat_choices)

    athlete_choices = sorted(df["name"].dropna().unique())
    sel_athletes = st.multiselect("Athletes (optional)", athlete_choices, default=[])

mask = pd.Series(True, index=df.index)
if "year" in df and all(v is not None for v in year_range):
    mask &= df["year"].between(year_range[0], year_range[1])
if sel_dis:
    mask &= df["discipline"].isin(sel_dis)
if sel_cat:
    mask &= df["category"].isin(sel_cat)
if sel_athletes:
    mask &= df["name"].isin(sel_athletes)

dfv = df.loc[mask].copy()

# ---------- KPIs ----------

def kpi(label, value):
    st.metric(label, value if pd.notna(value) else "—")

c1,c2,c3,c4 = st.columns(4)
with c1: kpi("Active athletes", dfv["name"].nunique())
with c2: kpi("Competitions attended",  len(dfv))
with c3: kpi("Podiums",        int(dfv["podium"].sum()) if "podium" in dfv else 0)
with c4: kpi("Avg rank",       round(dfv["rank"].mean(), 2) if "rank" in dfv and len(dfv) else "—")

st.markdown("---")

# ---------- charts ----------

# Athletes per year
if "year" in dfv:
    left, right = st.columns(2)

    with left:
        part = (dfv.groupby("year", as_index=False)["name"].nunique()
                  .rename(columns={"name":"athletes"}))
        st.plotly_chart(px.bar(part, x="year", y="athletes",
                               title="Athletes per Year",
                               labels={"year":"Year","athletes":"# Athletes"}),
                        use_container_width=True)

    with right:
        comps = (dfv.groupby("year", as_index=False).count()
                   .rename(columns={"event_name":"competitions"}))
        st.plotly_chart(px.bar(comps, x="year", y="competitions",
                               title="Competitions attended per Year",
                               labels={"year":"Year","competitions":"# Competitions"}),
                        use_container_width=True)

# Average rank trend
if "rank" in dfv and "year" in dfv:
    avg_rank = dfv.groupby("year", as_index=False)["rank"].mean().sort_values("year")
    fig = px.line(avg_rank, x="year", y="rank", markers=True,
                  title="Average Rank (lower is better)",
                  labels={"year":"Year","rank":"Average Rank"})
    st.plotly_chart(fig, use_container_width=True)

# Podiums per year
if "podium" in dfv and "year" in dfv:
    podiums = dfv.groupby("year", as_index=False)["podium"].sum().rename(columns={"podium":"podiums"})
    st.plotly_chart(px.bar(podiums, x="year", y="podiums",
                           title="Podiums per Year",
                           labels={"year":"Year","podiums":"# Podiums"}),
                    use_container_width=True)

# Rank distribution per year
if "rank" in dfv and "year" in dfv:
    st.plotly_chart(px.box(dfv.dropna(subset=["rank"]), x="year", y="rank", points="all",
                           title="Rank Distribution by Year",
                           labels={"year":"Year","rank":"Rank"}),
                    use_container_width=True)

st.markdown("---")

# ---------- leaderboard ----------

st.subheader("Athlete Leaderboard (within current filter)")

agg = (
    dfv.groupby("name", as_index=False)
       .agg(
           competitions=("event_name","nunique"),
           starts=("event_name","size"),
           average_rank=("rank","mean"),
           best_rank=("rank","min"),
           podiums=("podium","sum"),
           top10=("top10","sum")
       )
)
if not agg.empty:
    agg["average_rank"] = agg["average_rank"].round(2)
    agg = agg.sort_values(by=["average_rank","best_rank","starts"], ascending=[True, True, False])
st.dataframe(agg, use_container_width=True)

st.markdown("---")

# ---------- athlete detail ----------

st.subheader("Athlete Detail")
detail_name = st.selectbox("Choose athlete", sorted(dfv["name"].unique()))
detail = dfv[dfv["name"] == detail_name].copy().sort_values(["date_start","event_name","category"])

if "date_start" in detail:
    detail["date"] = detail["date_start"].dt.date
cols = ["date","rank","event_name","category","discipline","location","url","source_year"]
st.dataframe(detail[[c for c in cols if c in detail.columns]], use_container_width=True)

if "rank" in detail and "date_start" in detail:
    fig = px.line(detail.dropna(subset=["rank"]), x="date_start", y="rank", markers=True,
                  title=f"Rank timeline – {detail_name}",
                  labels={"date_start":"Date","rank":"Rank"})

    st.plotly_chart(fig, use_container_width=True)

st.caption("Files auto-loaded: " + ", ".join(sorted({os.path.basename(n) for n,_ in disk_files})) if disk_files else "None")