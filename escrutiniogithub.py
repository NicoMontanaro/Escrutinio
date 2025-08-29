import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- CONFIG ---
SHEET_ID = "1vYiQvkDqdx-zgtRbNPN5_0l2lXAceTF2py4mlM1pK_U"
SHEET_NAMES = {
    "raw": "Respuestas_raw",
    "escuelas": "Mapeo_Escuelas_raw",
    "alianzas": "Mapeo_Alianzas_raw",
}
AUTOREFRESH_SEC = 60  # refresco automático

# --- CARGA DE GOOGLE SHEETS ---
# Usamos gspread a través de st.secrets (service account).
# 1) En Streamlit Cloud, agrega st.secrets con la clave:
#    [gcp_service_account]
#    type="service_account"
#    project_id="..."
#    private_key="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
#    client_email="xxx@yyy.iam.gserviceaccount.com"
# 2) Comparte el Sheet con ese client_email como Viewer/Editor.

@st.cache_resource
def _gspread_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as e:
        st.stop()

    if "gcp_service_account" not in st.secrets:
        st.error("Faltan credenciales en st.secrets['gcp_service_account'].")
        st.stop()

    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    scoped = creds.with_scopes([
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ])
    return gspread.authorize(scoped)

def _sheet_to_df(gc, sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)
    rows = ws.get_all_values()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df

@st.cache_data(ttl=AUTOREFRESH_SEC)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gc = _gspread_client()
    df_raw = _sheet_to_df(gc, SHEET_ID, SHEET_NAMES["raw"])
    df_esc = _sheet_to_df(gc, SHEET_ID, SHEET_NAMES["escuelas"])
    df_ali = _sheet_to_df(gc, SHEET_ID, SHEET_NAMES["alianzas"])
    return df_raw, df_esc, df_ali

# --- NORMALIZACIONES ---
def normalize_mesa(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"[^0-9]", "", regex=True)
         .replace({"": np.nan})
         .astype("Int64")
    )

def bool_from_any(s: pd.Series) -> pd.Series:
    up = s.astype(str).str.upper().str.strip()
    return up.isin(["TRUE", "1", "SI", "SÍ", "YES", "VERDADERO"])

def detect_party_columns(df_raw: pd.DataFrame) -> List[str]:
    cols = []
    for c in df_raw.columns:
        if re.match(r"^\s*\d+", str(c)):  # empieza con número -> partido
            cols.append(c)
    return cols

def tidy_votes(df_raw: pd.DataFrame, party_cols: List[str]) -> pd.DataFrame:
    # Mesa normalizada
    mesa_key = normalize_mesa(df_raw.get("Mesa", pd.Series(index=df_raw.index)))
    df = df_raw.copy()
    df.insert(0, "MESA_KEY", mesa_key)
    # Melt de votos
    long = df.melt(
        id_vars=["MESA_KEY"],
        value_vars=party_cols,
        var_name="partido_header",
        value_name="votos"
    )
    long["votos"] = pd.to_numeric(long["votos"], errors="coerce").fillna(0).astype(int)
    # Número de partido desde el header
    long["numero_partido"] = (
        long["partido_header"].astype(str).str.extract(r"^\s*(\d+)", expand=False)
    )
    long["numero_partido"] = pd.to_numeric(long["numero_partido"], errors="coerce")
    return long.dropna(subset=["MESA_KEY", "numero_partido"]).astype({"numero_partido": int})

def prep_data():
    df_raw, df_esc, df_ali = load_data()

    if df_raw.empty:
        st.warning("Respuestas_raw está vacío.")
    if df_esc.empty:
        st.warning("Mapeo_Escuelas_raw está vacío.")
    if df_ali.empty:
        st.warning("Mapeo_Alianzas_raw está vacío.")

    # Normalizaciones en mapeos
    if "MESA" in df_esc.columns:
        df_esc["MESA_KEY"] = normalize_mesa(df_esc["MESA"])
    else:
        # fallback por nombre variante
        mesa_col = next((c for c in df_esc.columns if re.search(r"\bmesa\b", c, re.I)), None)
        if mesa_col:
            df_esc["MESA_KEY"] = normalize_mesa(df_esc[mesa_col])

    if "TESTIGO" in df_esc.columns:
        df_esc["TESTIGO_BOOL"] = bool_from_any(df_esc["TESTIGO"])
    else:
        df_esc["TESTIGO_BOOL"] = False

    # Columnas de partidos (headers que inician con número)
    party_cols = detect_party_columns(df_raw)

    # Datos tidy de votos
    long = tidy_votes(df_raw, party_cols)

    # Join con mapeo alianzas
    # Espera: Mapeo_Alianzas_raw: numero | Partidos políticos | orden | Alianza
    ali_num_col = next((c for c in df_ali.columns if re.match(r"^\s*numero\s*$", c, re.I)), None)
    ali_name_col = next((c for c in df_ali.columns if re.match(r"^\s*alianza\s*$", c, re.I)), None)
    party_name_col = next((c for c in df_ali.columns if re.search(r"partidos?\s+pol", c, re.I)), None)

    if not ali_num_col or not ali_name_col:
        st.error("En Mapeo_Alianzas_raw deben existir columnas: 'numero' y 'Alianza'.")
        st.stop()

    df_ali["_numero"] = pd.to_numeric(df_ali[ali_num_col], errors="coerce")
    ali_map = df_ali[["_numero", ali_name_col]].rename(columns={"_numero": "numero_partido", ali_name_col: "ALIANZA"})
    long = long.merge(ali_map, on="numero_partido", how="left")

    # Limpiar "No Usar"
    if not ali_map.empty:
        long["ALIANZA"] = long["ALIANZA"].fillna("")
        mask_no_usar = long["ALIANZA"].str.strip().str.match(r"(?i)^\s*no\s*usar\s*$")
        long.loc[mask_no_usar, "ALIANZA"] = np.nan

    # Join con escuelas (Depto, Testigo)
    if "MESA_KEY" in df_esc.columns:
        long = long.merge(
            df_esc[["MESA_KEY", "DEPARTAMENTO", "ESTABLECIMIENTO", "TESTIGO_BOOL"]],
            on="MESA_KEY",
            how="left"
        )
    else:
        long["DEPARTAMENTO"] = np.nan
        long["ESTABLECIMIENTO"] = np.nan
        long["TESTIGO_BOOL"] = False

    # Partidos: nombre legible
    if party_name_col:
        name_map = df_ali[["_numero", party_name_col]].rename(columns={"_numero": "numero_partido", party_name_col: "PARTIDO"})
        long = long.merge(name_map, on="numero_partido", how="left")
    else:
        # usar el texto del header sin el número
        long["PARTIDO"] = (
            long["partido_header"].astype(str)
                .str.replace(r"^\s*\d+\s*-\s*", "", regex=True)
                .str.strip()
        )

    return df_raw, df_esc, df_ali, long

# ---------------- UI ----------------
st.set_page_config(page_title="Escrutinio – Dashboard", layout="wide")
st.title("📊 Escrutinio – Resultados por Partido y Alianza")

# Auto-refresh
st.caption(f"Actualiza cada {AUTOREFRESH_SEC}s")
st.experimental_rerun  # just reference; (we rely on cache TTL)

df_raw, df_esc, df_ali, long = prep_data()

# Filtros
with st.sidebar:
    st.header("Filtros")
    # Departamentos disponibles
    depts = ["(Todos)"] + sorted([d for d in long["DEPARTAMENTO"].dropna().unique().tolist()])
    dept_sel = st.selectbox("Departamento", depts, index=0)
    only_testigo = st.toggle("Solo mesas testigo", value=False)
    # Rango de mesa
    mesas_disponibles = long["MESA_KEY"].dropna().astype(int).sort_values().unique().tolist()
    if mesas_disponibles:
        min_mesa, max_mesa = int(mesas_disponibles[0]), int(mesas_disponibles[-1])
        rango = st.slider("Rango de mesa", min_value=min_mesa, max_value=max_mesa, value=(min_mesa, max_mesa))
    else:
        rango = (0, 10**9)

# Aplicar filtros
mask = pd.Series(True, index=long.index)
if dept_sel != "(Todos)":
    mask &= (long["DEPARTAMENTO"] == dept_sel)
if only_testigo:
    mask &= (long["TESTIGO_BOOL"] == True)
mask &= long["MESA_KEY"].between(rango[0], rango[1])

flt = long.loc[mask].copy()

# KPIs
total_votos_validos = int(flt["votos"].sum())
total_mesas = flt["MESA_KEY"].nunique()
total_partidos = flt["PARTIDO"].nunique()
col1, col2, col3 = st.columns(3)
col1.metric("Votos (filtro aplicado)", f"{total_votos_validos:,}".replace(",", "."))
col2.metric("Mesas consideradas", total_mesas)
col3.metric("Partidos distintos", total_partidos)

# ---------- Resultados por ALIANZA ----------
st.subheader("Resultados por Alianza")
ali_df = (
    flt.dropna(subset=["ALIANZA"])
       .groupby("ALIANZA", as_index=False)["votos"].sum()
       .sort_values("votos", ascending=False)
)
if not ali_df.empty:
    ali_df["% sobre válidos"] = (ali_df["votos"] / ali_df["votos"].sum() * 100).round(2)
    c1, c2 = st.columns([2, 1.2])
    with c1:
        st.bar_chart(ali_df.set_index("ALIANZA")["votos"])
    with c2:
        st.dataframe(ali_df, use_container_width=True)
else:
    st.info("No hay votos asociados a alianzas con el filtro actual.")

# ---------- Resultados por PARTIDO ----------
st.subheader("Resultados por Partido")
part_df = (
    flt.groupby(["numero_partido", "PARTIDO"], as_index=False)["votos"].sum()
       .sort_values("votos", ascending=False)
)
if not part_df.empty:
    part_df["Etiqueta"] = part_df["numero_partido"].astype(int).astype(str) + " - " + part_df["PARTIDO"].fillna("")
    part_df["% sobre válidos"] = (part_df["votos"] / part_df["votos"].sum() * 100).round(2)
    c1, c2 = st.columns([2, 1.2])
    with c1:
        st.bar_chart(part_df.set_index("Etiqueta")["votos"])
    with c2:
        st.dataframe(part_df[["numero_partido", "PARTIDO", "votos", "% sobre válidos"]], use_container_width=True)
else:
    st.info("No hay votos de partidos con el filtro actual.")

# ---------- Solo mesas testigo (detalle por mesa) ----------
st.subheader("Detalle por Mesas Testigo (por Alianza)")
testigo_flt = flt.loc[flt["TESTIGO_BOOL"] == True].copy()
if not testigo_flt.empty:
    pivot_testigo = (
        testigo_flt.dropna(subset=["ALIANZA"])
                   .pivot_table(index=["MESA_KEY", "DEPARTAMENTO"],
                                columns="ALIANZA",
                                values="votos",
                                aggfunc="sum",
                                fill_value=0)
                   .reset_index()
                   .sort_values("MESA_KEY")
    )
    st.dataframe(pivot_testigo, use_container_width=True)
else:
    st.info("No hay mesas testigo con datos bajo los filtros actuales.")

# ---------- Tabla por mesa (todas) ----------
with st.expander("Ver tabla por mesa (todas las alianzas y filtros aplicados)"):
    pivot_all = (
        flt.dropna(subset=["ALIANZA"])
           .pivot_table(index=["MESA_KEY", "DEPARTAMENTO", "ESTABLECIMIENTO", "TESTIGO_BOOL"],
                        columns="ALIANZA",
                        values="votos",
                        aggfunc="sum",
                        fill_value=0)
           .reset_index()
           .sort_values(["DEPARTAMENTO", "MESA_KEY"])
    )
    st.dataframe(pivot_all, use_container_width=True)

st.caption("Tip: si algún valor se ve raro, revisá que los encabezados de partidos en Respuestas_raw empiecen con número y que 'numero'/'Alianza' existan en Mapeo_Alianzas_raw. ‘No Usar’ se excluye automáticamente.")
