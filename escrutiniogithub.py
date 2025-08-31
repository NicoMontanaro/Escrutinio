# -*- coding: utf-8 -*-
"""
Escrutinio - Dashboard Streamlit

Hojas esperadas (mismo Sheet ID):
  - Respuestas_raw
  - Mapeo_Escuelas_raw                 -> DEPARTAMENTO | ESTABLECIMIENTO | MESA | TESTIGO
  - Mapeo_Alianzas_raw                 -> numero | Partidos políticos | orden | Alianza
  - Mapeo_Mesa_Municipio_raw           -> MESA | MUNICIPIO | (DEPARTAMENTO opcional)
  - Padron_Departamento_raw            -> DEPARTAMENTO | PADRON
  - Padron_Municipio_raw               -> MUNICIPIO | PADRON (o VOTANTES)

Secrets (.streamlit/secrets.toml):
[gcp_service_account]
type="service_account"
project_id="..."
private_key_id="..."
private_key="-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
client_email="...@...iam.gserviceaccount.com"
client_id="..."
token_uri="https://oauth2.googleapis.com/token"
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ================== CONFIG ==================
SHEET_ID = "1vYiQvkDqdx-zgtRbNPN5_0l2lXAceTF2py4mlM1pK_U"
SHEET_NAMES = {
    "raw": "Respuestas_raw",
    "escuelas": "Mapeo_Escuelas_raw",
    "alianzas": "Mapeo_Alianzas_raw",
    "mesa_muni": "Mapeo_Mesa_Municipio_raw",
}
AUTOREFRESH_SEC = 180      # 3 minutos
TOTAL_MESAS_PROV = 2808    # para footer

st.set_page_config(page_title="Escrutinio - Dashboard", layout="wide")

# ================== PADRON FALLBACK (si no hay hoja depto) ==================
PADRON_FALLBACK = [
    {"DEPARTAMENTO": "Capital", "PADRON": 313265},
    {"DEPARTAMENTO": "Goya", "PADRON": 81590},
    {"DEPARTAMENTO": "Santo Tome", "PADRON": 55536},
    {"DEPARTAMENTO": "Paso de los Libres", "PADRON": 48417},
    {"DEPARTAMENTO": "Ituzaingo", "PADRON": 44736},
    {"DEPARTAMENTO": "Curuzu Cuatia", "PADRON": 40378},
    {"DEPARTAMENTO": "Mercedes", "PADRON": 38898},
    {"DEPARTAMENTO": "Monte Caseros", "PADRON": 34282},
    {"DEPARTAMENTO": "Esquina", "PADRON": 34192},
    {"DEPARTAMENTO": "Bella Vista", "PADRON": 33941},
    {"DEPARTAMENTO": "Lavalle", "PADRON": 29206},
    {"DEPARTAMENTO": "San Cosme", "PADRON": 26010},
    {"DEPARTAMENTO": "Saladas", "PADRON": 20974},
    {"DEPARTAMENTO": "Concepcion", "PADRON": 19913},
    {"DEPARTAMENTO": "San Roque", "PADRON": 18767},
    {"DEPARTAMENTO": "San Luis del Palmar", "PADRON": 18572},
    {"DEPARTAMENTO": "Empedrado", "PADRON": 16253},
    {"DEPARTAMENTO": "Gral Paz", "PADRON": 15077},
    {"DEPARTAMENTO": "Gral San Martin", "PADRON": 12512},
    {"DEPARTAMENTO": "Itati", "PADRON": 10942},
    {"DEPARTAMENTO": "San Miguel", "PADRON": 10296},
    {"DEPARTAMENTO": "Mburucuya", "PADRON": 9560},
    {"DEPARTAMENTO": "Sauce", "PADRON": 8208},
    {"DEPARTAMENTO": "Gral Alvear", "PADRON": 7304},
    {"DEPARTAMENTO": "Beron de Astrada", "PADRON": 2787},
]

# ================== GOOGLE SHEETS CLIENT ==================
def _normalize_private_key(pk: str) -> str:
    """Normaliza la clave PEM para evitar errores base64 por saltos de línea mal puestos."""
    if not isinstance(pk, str):
        return pk
    s = pk.strip().replace("\\r\\n", "\n").replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    s = s.replace("—", "-").replace("–", "-")
    start, end = "-----BEGIN PRIVATE KEY-----", "-----END PRIVATE KEY-----"
    if start not in s or end not in s:
        return s
    _, rest = s.split(start, 1)
    body, _ = rest.split(end, 1)
    lines = [ln.strip() for ln in body.strip().split("\n") if ln.strip()]
    b64 = re.sub(r"[^A-Za-z0-9+/=]", "", "".join(lines))
    wrapped = "\n".join([b64[i:i+64] for i in range(0, len(b64), 64)])
    return f"{start}\n{wrapped}\n{end}\n"

@st.cache_resource
def _gspread_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as e:
        st.error("Faltan dependencias (gspread/google-auth). Instalar desde requirements.txt\n\n" + str(e))
        st.stop()

    if "gcp_service_account" not in st.secrets:
        st.error("Falta el bloque [gcp_service_account] en secrets.")
        st.stop()

    info = dict(st.secrets["gcp_service_account"])
    info["private_key"] = _normalize_private_key(info.get("private_key", ""))

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    try:
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error("No se pudo crear el cliente de Google Sheets.\n\n" + str(e))
        st.stop()

def _sheet_to_df(gc, sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)
    rows = ws.get_all_values()
    return pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame()

@st.cache_data(ttl=AUTOREFRESH_SEC)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gc = _gspread_client()
    try:
        df_raw  = _sheet_to_df(gc, SHEET_ID, SHEET_NAMES["raw"])
        df_esc  = _sheet_to_df(gc, SHEET_ID, SHEET_NAMES["escuelas"])
        df_ali  = _sheet_to_df(gc, SHEET_ID, SHEET_NAMES["alianzas"])
        try:
            df_muni = _sheet_to_df(gc, SHEET_ID, SHEET_NAMES["mesa_muni"])
        except Exception:
            df_muni = pd.DataFrame(columns=["MESA", "MUNICIPIO"])
    except Exception as e:
        st.error("No se pudo abrir el Google Sheet. Revisá permisos, nombres de hoja y APIs habilitadas.\n\n" + str(e))
        st.stop()
    return df_raw, df_esc, df_ali, df_muni

@st.cache_data(ttl=AUTOREFRESH_SEC)
def load_padron_depto() -> pd.DataFrame:
    """Lee Padron_Departamento_raw. Si no existe, usa fallback."""
    gc = _gspread_client()
    try:
        sh = gc.open_by_key(SHEET_ID)
        titles = [w.title for w in sh.worksheets()]
        if "Padron_Departamento_raw" in titles:
            ws = sh.worksheet("Padron_Departamento_raw")
            rows = ws.get_all_values()
            dfp = pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame()
            if dfp.empty:
                return pd.DataFrame(PADRON_FALLBACK)
            if "DEPARTAMENTO" not in dfp.columns:
                return pd.DataFrame(PADRON_FALLBACK)
            pcol = "PADRON" if "PADRON" in dfp.columns else next((c for c in dfp.columns if re.search("padron", c, re.I)), None)
            if not pcol:
                return pd.DataFrame(PADRON_FALLBACK)
            dfp["DEPARTAMENTO"] = dfp["DEPARTAMENTO"].astype(str).str.strip()
            dfp["PADRON"] = pd.to_numeric(
                dfp[pcol].astype(str).str.replace(".", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce"
            ).fillna(0).astype(int)
            return dfp[["DEPARTAMENTO", "PADRON"]]
    except Exception:
        pass
    return pd.DataFrame(PADRON_FALLBACK)

@st.cache_data(ttl=AUTOREFRESH_SEC)
def load_padron_municipio() -> pd.DataFrame:
    """Lee Padron_Municipio_raw (MUNICIPIO y PADRON/VOTANTES). Devuelve: MUNICIPIO, PADRON, (opcional DEPARTAMENTO)."""
    gc = _gspread_client()
    try:
        sh = gc.open_by_key(SHEET_ID)
        titles = [w.title for w in sh.worksheets()]
        if "Padron_Municipio_raw" in titles:
            ws = sh.worksheet("Padron_Municipio_raw")
            rows = ws.get_all_values()
            df = pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame()
            if df.empty:
                return df
            if "MUNICIPIO" not in df.columns:
                mcol = next((c for c in df.columns if re.search("municip", c, re.I)), None)
                if mcol: df = df.rename(columns={mcol:"MUNICIPIO"})
            if "DEPARTAMENTO" not in df.columns:
                dcol = next((c for c in df.columns if re.search("depart", c, re.I)), None)
                if dcol: df = df.rename(columns={dcol:"DEPARTAMENTO"})
            if "PADRON" in df.columns:
                pcol = "PADRON"
            elif "VOTANTES" in df.columns:
                pcol = "VOTANTES"
            else:
                pcol = next((c for c in df.columns if re.search("padron|votantes", c, re.I)), None)
            if not pcol:
                return pd.DataFrame()
            df["MUNICIPIO"] = df["MUNICIPIO"].astype(str).str.strip()
            if "DEPARTAMENTO" in df.columns:
                df["DEPARTAMENTO"] = df["DEPARTAMENTO"].astype(str).str.strip()
            df["PADRON"] = pd.to_numeric(
                df[pcol].astype(str).str.replace(".", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce"
            ).fillna(0).astype(int)
            df["_mkey"] = normalize_name(df["MUNICIPIO"])
            if "DEPARTAMENTO" in df.columns:
                df["_dkey"] = normalize_name(df["DEPARTAMENTO"])
            return df
    except Exception:
        pass
    return pd.DataFrame()

# ================== HELPERS ==================
def normalize_mesa(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace({"": np.nan}).astype("Int64")

def bool_from_any(s: pd.Series) -> pd.Series:
    up = s.astype(str).str.upper().str.strip()
    return up.isin(["TRUE", "1", "SI", "SÍ", "YES", "VERDADERO"])

def find_col(df: pd.DataFrame, regex: str) -> str | None:
    for c in df.columns:
        if re.search(regex, str(c), re.IGNORECASE):
            return c
    return None

def detect_party_columns(df_raw: pd.DataFrame) -> List[str]:
    return [c for c in df_raw.columns if re.match(r"^\s*\d+", str(c))]

def tidy_votes(df_raw: pd.DataFrame, party_cols: List[str]) -> pd.DataFrame:
    mesa_col = "Mesa" if "Mesa" in df_raw.columns else (find_col(df_raw, r"^\s*mesa\s*$") or "Mesa")
    df = df_raw.copy()
    df["MESA_KEY"] = normalize_mesa(df.get(mesa_col, pd.Series(index=df.index)))
    long = df.melt(
        id_vars=["MESA_KEY"],
        value_vars=party_cols,
        var_name="partido_header",
        value_name="votos",
    )
    long["votos"] = pd.to_numeric(long["votos"], errors="coerce").fillna(0).astype(int)
    long["numero_partido"] = long["partido_header"].astype(str).str.extract(r"^\s*(\d+)", expand=False)
    long["numero_partido"] = pd.to_numeric(long["numero_partido"], errors="coerce")
    long = long.dropna(subset=["MESA_KEY", "numero_partido"]).astype({"numero_partido": int})
    long["PARTIDO_NOMBRE_HEADER"] = long["partido_header"].astype(str).str.replace(r"^\s*\d+\s*-\s*", "", regex=True).str.strip()
    return long

def normalize_name(s: pd.Series | str) -> pd.Series | str:
    def _n(x: str) -> str:
        x = str(x).strip()
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", x).strip().lower()
    if isinstance(s, pd.Series):
        return s.astype(str).map(_n)
    return _n(s)

def mesa_totales(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Devuelve MESA_KEY | TOTAL_MESA (si no existe el total, suma partidos + blancos/nulos/etc.)."""
    mesa_col = "Mesa" if "Mesa" in df_raw.columns else (find_col(df_raw, r"^\s*mesa\s*$") or "Mesa")
    tot_col = find_col(df_raw, r"total\s*de\s*votos\s*en\s*la\s*mesa")
    if not tot_col:
        party_cols = detect_party_columns(df_raw)
        df_tmp = df_raw.copy()
        df_tmp["MESA_KEY"] = normalize_mesa(df_tmp.get(mesa_col, pd.Series(index=df_tmp.index)))
        extra = [c for c in df_raw.columns if re.search(r"votos?\s+(en\s+)?(blancos|nulos|recurridos|impugnados)", c, re.I)]
        cols = party_cols + extra
        for c in cols:
            df_tmp[c] = pd.to_numeric(df_tmp[c], errors="coerce").fillna(0).astype(int)
        out = df_tmp[["MESA_KEY"] + cols].copy()
        out["TOTAL_MESA"] = out[cols].sum(axis=1)
        return out[["MESA_KEY", "TOTAL_MESA"]]
    else:
        df_tot = df_raw[[mesa_col, tot_col]].copy()
        df_tot["MESA_KEY"] = normalize_mesa(df_tot[mesa_col])
        df_tot["TOTAL_MESA"] = pd.to_numeric(
            df_tot[tot_col].astype(str).str.replace(".", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce"
        ).fillna(0).astype(int)
        return df_tot[["MESA_KEY", "TOTAL_MESA"]]

def pivot_pct_valid(df_long: pd.DataFrame, region_cols: list[str]) -> pd.DataFrame:
    """% de votos por alianza dentro de cada región (suma 100% por región)."""
    g = (df_long.dropna(subset=["ALIANZA"])
               .groupby(region_cols + ["ALIANZA"], as_index=False)["votos"].sum())
    tot = g.groupby(region_cols, as_index=False)["votos"].sum().rename(columns={"votos": "TOTAL_REGION"})
    g = g.merge(tot, on=region_cols, how="left")
    g["%"] = np.where(g["TOTAL_REGION"] > 0, g["votos"] / g["TOTAL_REGION"] * 100, 0)
    pvt = (g.pivot_table(index=region_cols, columns="ALIANZA", values="%", aggfunc="first")
             .fillna(0).round(2).reset_index())
    return pvt

# ================== Balotaje ==================
def ballotage_status(votes_df: pd.DataFrame, rule: str = "nacional") -> tuple[dict | None, pd.DataFrame]:
    """
    votes_df: DataFrame con columnas ['ALIANZA','votos'] (votos válidos por alianza).
    rule:
      - 'nacional' => gana si pct>=45% o (pct>=40% y diferencia >=10 puntos sobre 2.º)
      - 'mayoria_absoluta' => gana si pct>50%
    Devuelve:
      - resumen (dict) o None si no hay datos
      - ranking (ALIANZA | votos | pct) ordenado desc.
    """
    df = (votes_df.dropna(subset=["ALIANZA"])
                  .loc[votes_df["ALIANZA"] != "(Sin alianza)"]
                  .groupby("ALIANZA", as_index=False)["votos"].sum())
    total = df["votos"].sum()
    if total <= 0 or df.empty:
        return None, df

    df["pct"] = df["votos"] / total * 100
    df = df.sort_values("votos", ascending=False).reset_index(drop=True)

    top1 = df.iloc[0]
    top2 = df.iloc[1] if len(df) > 1 else pd.Series({"ALIANZA": None, "votos": 0, "pct": 0.0})

    if rule == "nacional":
        gana = (top1["pct"] >= 45) or (top1["pct"] >= 40 and (top1["pct"] - top2["pct"]) >= 10)
    elif rule == "mayoria_absoluta":
        gana = (top1["pct"] > 50)
    else:
        gana = False

    resumen = {
        "top1": str(top1["ALIANZA"]), "pct1": float(top1["pct"]),
        "top2": str(top2["ALIANZA"]), "pct2": float(top2["pct"]),
        "gana_primera_vuelta": bool(gana),
        "regla": rule,
    }
    return resumen, df[["ALIANZA", "votos", "pct"]]

# ================== PREP PIPELINE ==================
def prep_data():
    df_raw, df_esc, df_ali, df_muni = load_data()
    if df_raw.empty: st.warning("Respuestas_raw está vacío.")
    if df_esc.empty: st.warning("Mapeo_Escuelas_raw está vacío.")
    if df_ali.empty: st.warning("Mapeo_Alianzas_raw está vacío.")
    if df_muni.empty: st.info("Mapeo_Mesa_Municipio_raw está vacío o no existe.")

    # Escuelas -> DEPARTAMENTO / TESTIGO / ESTABLECIMIENTO
    mesa_esc_col = "MESA" if "MESA" in df_esc.columns else find_col(df_esc, r"\bmesa\b")
    df_esc["MESA_KEY"] = normalize_mesa(df_esc[mesa_esc_col]) if mesa_esc_col else pd.Series(dtype="Int64")
    test_col = "TESTIGO" if "TESTIGO" in df_esc.columns else find_col(df_esc, r"testig")
    df_esc["TESTIGO_BOOL"] = bool_from_any(df_esc[test_col]) if test_col else False

    # Municipios (DEPARTAMENTO opcional)
    df_muni_norm = pd.DataFrame(columns=["MESA_KEY", "MUNICIPIO"])
    if not df_muni.empty:
        dm = df_muni.copy()
        dm.columns = [str(c).strip() for c in dm.columns]
        upper_map = {c: c.strip().upper() for c in dm.columns}
        dm.rename(columns=upper_map, inplace=True)

        def _findcol(cols, pattern):
            for c in cols:
                if re.search(pattern, c, re.I):
                    return c
            return None

        mesa_c = "MESA" if "MESA" in dm.columns else _findcol(dm.columns, r"\bmesa\b")
        muni_c = "MUNICIPIO" if "MUNICIPIO" in dm.columns else _findcol(dm.columns, r"munic")
        dep_c  = "DEPARTAMENTO" if "DEPARTAMENTO" in dm.columns else _findcol(dm.columns, r"depart")

        if not mesa_c or not muni_c:
            st.error("Mapeo_Mesa_Municipio_raw debe contener columnas 'MESA' y 'MUNICIPIO'.")
            st.stop()

        dm["MESA_KEY"] = normalize_mesa(dm[mesa_c])
        dm["MUNICIPIO"] = dm[muni_c].astype(str).str.strip()
        if dep_c:
            dm["DEPARTAMENTO"] = dm[dep_c].astype(str).str.strip()  # opcional p/diagnóstico
        dm = dm.dropna(subset=["MESA_KEY"]).drop_duplicates(subset=["MESA_KEY"])

        keep_cols = ["MESA_KEY", "MUNICIPIO"]
        if "DEPARTAMENTO" in dm.columns:
            keep_cols.append("DEPARTAMENTO")
        df_muni_norm = dm[keep_cols].copy()

    df_muni_only = (
        df_muni_norm.loc[:, ["MESA_KEY", "MUNICIPIO"]].drop_duplicates("MESA_KEY")
        if not df_muni_norm.empty else pd.DataFrame(columns=["MESA_KEY", "MUNICIPIO"])
    )

    # Votos long
    parties = detect_party_columns(df_raw)
    long = tidy_votes(df_raw, parties)

    # Alianzas
    num_col = "numero" if "numero" in df_ali.columns else find_col(df_ali, r"^\s*numero\s*$")
    ali_col = "Alianza" if "Alianza" in df_ali.columns else find_col(df_ali, r"^\s*alianza\s*$")
    party_name_col = find_col(df_ali, r"Partidos?\s+pol")
    if not (num_col and ali_col):
        st.error("En Mapeo_Alianzas_raw deben existir columnas 'numero' y 'Alianza'.")
        st.stop()

    df_ali["_numero"] = pd.to_numeric(df_ali[num_col], errors="coerce")
    ali_map = df_ali[["_numero", ali_col]].rename(columns={"_numero": "numero_partido", ali_col: "ALIANZA"})
    long = long.merge(ali_map, on="numero_partido", how="left")
    long["ALIANZA"] = long["ALIANZA"].fillna("")
    no_usar = long["ALIANZA"].str.strip().str.match(r"(?i)^\s*no\s*usar\s*$")
    long.loc[no_usar, "ALIANZA"] = np.nan
    long["ALIANZA"] = long["ALIANZA"].where(long["ALIANZA"].notna(), "(Sin alianza)")

    if party_name_col:
        name_map = df_ali[["_numero", party_name_col]].rename(columns={"_numero": "numero_partido", party_name_col: "PARTIDO"})
        long = long.merge(name_map, on="numero_partido", how="left")
    else:
        long["PARTIDO"] = long["PARTIDO_NOMBRE_HEADER"]

    # Merge final (DEPARTAMENTO desde Escuelas, MUNICIPIO desde Mapeo_Mesa_Municipio)
    keep_esc = [c for c in ["DEPARTAMENTO", "ESTABLECIMIENTO", "TESTIGO_BOOL"] if c in df_esc.columns]
    long = long.merge(
        df_esc[["MESA_KEY"] + keep_esc].drop_duplicates("MESA_KEY"),
        on="MESA_KEY", how="left"
    )
    long["DEPARTAMENTO"] = long["DEPARTAMENTO"].fillna("(Sin depto)")

    if not df_muni_only.empty:
        long = long.merge(df_muni_only, on="MESA_KEY", how="left")
    if "MUNICIPIO" not in long.columns:
        long["MUNICIPIO"] = np.nan
    long["MUNICIPIO"] = long["MUNICIPIO"].fillna("(Sin municipio)")

    # Totales por mesa
    df_tot = mesa_totales(df_raw)

    # df_mesas_all = TOTAL_MESA + DEPARTAMENTO + MUNICIPIO + TESTIGO
    df_mesas_all = df_tot.merge(
        df_esc[["MESA_KEY", "DEPARTAMENTO", "TESTIGO_BOOL"]].drop_duplicates("MESA_KEY"),
        on="MESA_KEY", how="left"
    )
    df_mesas_all["DEPARTAMENTO"] = df_mesas_all["DEPARTAMENTO"].fillna("(Sin depto)")
    if not df_muni_only.empty:
        df_mesas_all = df_mesas_all.merge(df_muni_only, on="MESA_KEY", how="left")
    if "MUNICIPIO" not in df_mesas_all.columns:
        df_mesas_all["MUNICIPIO"] = np.nan
    df_mesas_all["MUNICIPIO"] = df_mesas_all["MUNICIPIO"].fillna("(Sin municipio)")

    # Diagnóstico de discrepancias (solo si mapeo muni trae DEPARTAMENTO)
    dept_mismatch = pd.DataFrame()
    if (not df_muni_norm.empty) and ("DEPARTAMENTO" in df_muni_norm.columns) and ("DEPARTAMENTO" in df_esc.columns):
        d1 = df_esc[["MESA_KEY", "DEPARTAMENTO"]].drop_duplicates("MESA_KEY").rename(columns={"DEPARTAMENTO": "DEP_ESC"})
        d2 = df_muni_norm[["MESA_KEY", "DEPARTAMENTO"]].drop_duplicates("MESA_KEY").rename(columns={"DEPARTAMENTO": "DEP_MAP"})
        mm = d1.merge(d2, on="MESA_KEY", how="inner")
        dept_mismatch = mm[(mm["DEP_ESC"].notna()) & (mm["DEP_MAP"].notna()) & (mm["DEP_ESC"] != mm["DEP_MAP"])]

    return df_raw, df_esc, df_ali, long, df_mesas_all, dept_mismatch

# ================== UI ==================
st.title("📊 Escrutinio - Alianzas, Departamentos y Municipios")
st.caption(f"Actualiza cada {AUTOREFRESH_SEC}s (cache TTL)")

# Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=AUTOREFRESH_SEC * 1000, key="refresh")
except Exception:
    with st.sidebar:
        if st.button("Actualizar ahora"):
            st.cache_data.clear()
            st.rerun()

# Data
df_raw, df_esc, df_ali, long, df_mesas_all, dept_mismatch = prep_data()

# Filtros
with st.sidebar:
    st.header("Filtros")
    depts = ["(Todos)"] + sorted([d for d in long["DEPARTAMENTO"].dropna().unique().tolist()])
    dept_sel = st.selectbox("Departamento", depts, index=0)

    if dept_sel != "(Todos)":
        muni_pool = long.loc[long["DEPARTAMENTO"] == dept_sel, "MUNICIPIO"].dropna().unique().tolist()
    else:
        muni_pool = long["MUNICIPIO"].dropna().unique().tolist()
    munis = ["(Todos)"] + sorted(muni_pool)
    muni_sel = st.selectbox("Municipio", munis, index=0)

    only_testigo = st.toggle("Solo mesas testigo", value=False)

    mesas_disponibles = long["MESA_KEY"].dropna().astype(int).sort_values().unique().tolist()
    if mesas_disponibles:
        min_mesa, max_mesa = int(mesas_disponibles[0]), int(mesas_disponibles[-1])
        rango = st.slider("Rango de mesa", min_value=min_mesa, max_value=max_mesa, value=(min_mesa, max_mesa))
    else:
        rango = (0, 10**9)

mask = pd.Series(True, index=long.index)
if dept_sel != "(Todos)":
    mask &= (long["DEPARTAMENTO"] == dept_sel)
if muni_sel != "(Todos)":
    mask &= (long["MUNICIPIO"] == muni_sel)
if only_testigo:
    mask &= (long["TESTIGO_BOOL"] == True)
mask &= long["MESA_KEY"].between(rango[0], rango[1])
flt = long.loc[mask].copy()

# % escrutado provincial
df_pad_depto = load_padron_depto()
total_emitidos_prov = int(df_mesas_all["TOTAL_MESA"].sum())
total_padron_prov   = int(df_pad_depto["PADRON"].sum()) if not df_pad_depto.empty else 0
pct_escrutado_prov  = (total_emitidos_prov / total_padron_prov * 100) if total_padron_prov > 0 else 0.0

st.subheader("Progreso provincial")
st.metric("Escrutado", f"{pct_escrutado_prov:.2f}%", help="(Votos emitidos de todas las mesas) / (Padrón total)")
st.divider()

# -------- Datos base generales --------
# Para % por alianza sobre válidos
ali_flt = (
    flt.dropna(subset=["ALIANZA"])
       .groupby(["DEPARTAMENTO", "MUNICIPIO", "ALIANZA"], as_index=False)["votos"].sum()
)

# Para tablas de totales y padrón por departamento
df_mesas_f = df_mesas_all.copy()
if dept_sel != "(Todos)":
    df_mesas_f = df_mesas_f[df_mesas_f["DEPARTAMENTO"] == dept_sel]
if muni_sel != "(Todos)":
    df_mesas_f = df_mesas_f[df_mesas_f["MUNICIPIO"] == muni_sel]
if only_testigo:
    df_mesas_f = df_mesas_f[df_mesas_f["TESTIGO_BOOL"] == True]
df_mesas_f = df_mesas_f[df_mesas_f["MESA_KEY"].between(rango[0], rango[1])]

dept_tot = (
    df_mesas_f.groupby("DEPARTAMENTO", as_index=False)["TOTAL_MESA"]
              .sum()
              .rename(columns={"TOTAL_MESA": "VOTOS_EMITIDOS"})
)
tmp = dept_tot.copy()
tmp["_key"] = normalize_name(tmp["DEPARTAMENTO"])
pad = df_pad_depto.copy()
pad["_key"] = normalize_name(pad["DEPARTAMENTO"])
dept_res = tmp.merge(pad[["_key", "PADRON"]], on="_key", how="left").drop(columns=["_key"])
dept_res["PADRON"] = dept_res["PADRON"].fillna(0).astype(int)
dept_res["% SOBRE PADRON"] = np.where(
    dept_res["PADRON"] > 0,
    (dept_res["VOTOS_EMITIDOS"] / dept_res["PADRON"] * 100).round(2),
    0.0,
)
dept_res = dept_res.sort_values("VOTOS_EMITIDOS", ascending=False)

# Municipios - totales y padrón
df_pad_muni = load_padron_municipio()
df_mesas_fm = df_mesas_all.copy()
if dept_sel != "(Todos)":
    df_mesas_fm = df_mesas_fm[df_mesas_fm["DEPARTAMENTO"] == dept_sel]
if muni_sel != "(Todos)":
    df_mesas_fm = df_mesas_fm[df_mesas_fm["MUNICIPIO"] == muni_sel]
if only_testigo:
    df_mesas_fm = df_mesas_fm[df_mesas_fm["TESTIGO_BOOL"] == True]
df_mesas_fm = df_mesas_fm[df_mesas_fm["MESA_KEY"].between(rango[0], rango[1])]

muni_tot = (
    df_mesas_fm.groupby(["DEPARTAMENTO", "MUNICIPIO"], as_index=False)["TOTAL_MESA"]
               .sum()
               .rename(columns={"TOTAL_MESA": "VOTOS_EMITIDOS"})
)
if not df_pad_muni.empty:
    muni_tmp = muni_tot.copy()
    muni_tmp["_mkey"] = normalize_name(muni_tmp["MUNICIPIO"])
    df_pad_muni["_mkey"] = normalize_name(df_pad_muni["MUNICIPIO"])
    muni_res = muni_tmp.merge(df_pad_muni[["_mkey", "PADRON"]], on="_mkey", how="left").drop(columns=["_mkey"])
    muni_res["PADRON"] = muni_res["PADRON"].fillna(0).astype(int)
    muni_res["% SOBRE PADRON"] = np.where(
        muni_res["PADRON"] > 0,
        (muni_res["VOTOS_EMITIDOS"] / muni_res["PADRON"] * 100).round(2),
        0.0,
    )
else:
    muni_res = muni_tot.copy()
muni_res = muni_res.sort_values(["DEPARTAMENTO", "VOTOS_EMITIDOS"], ascending=[True, False])

# ====== PESTAÑAS ======
tab_ali, tab_dept, tab_muni, tab_testigo, tab_mesas, tab_diag, tab_bal = st.tabs([
    "🧩 Alianzas",
    "🏙️ Departamentos",
    "🏘️ Municipios",
    "⭐ Testigo",
    "🗃️ Mesas (detalle)",
    "🔧 Diagnóstico",
    "🗳️ Balotaje",
])

# ---------------- Alianzas (global filtrado) ----------------
with tab_ali:
    ali_df = (
        flt.dropna(subset=["ALIANZA"])
           .groupby("ALIANZA", as_index=False)["votos"].sum()
           .sort_values("votos", ascending=False)
    )
    if not ali_df.empty:
        ali_df["% sobre válidos"] = (ali_df["votos"] / ali_df["votos"].sum() * 100).round(2)
    c1, c2 = st.columns([2, 1.2])
    with c1:
        st.markdown("#### Votos por Alianza (filtros aplicados)")
        if not ali_df.empty:
            st.bar_chart(ali_df.set_index("ALIANZA")["% sobre válidos"], use_container_width=True)
        else:
            st.info("Sin datos de alianzas para el filtro.")
    with c2:
        st.markdown("#### Tabla")
        st.dataframe(ali_df, use_container_width=True)

# ---------------- Departamentos ----------------
with tab_dept:
    st.markdown("### % de votos por Alianza - Departamento seleccionado")

    # Elegir departamento para mostrar % (si está '(Todos)')
    if dept_sel == "(Todos)":
        dept_opts = sorted(long["DEPARTAMENTO"].dropna().unique().tolist())
        dept_for_chart = st.selectbox("Elegí un departamento", dept_opts, key="dept_for_pct") if dept_opts else None
    else:
        dept_for_chart = dept_sel

    # Calcular % por alianza del departamento elegido
    if dept_for_chart:
        dsel = flt[flt["DEPARTAMENTO"] == dept_for_chart]
        ali_dep = (
            dsel.dropna(subset=["ALIANZA"])
                .groupby("ALIANZA", as_index=False)["votos"].sum()
                .sort_values("votos", ascending=False)
        )
        if not ali_dep.empty:
            ali_dep["% sobre válidos"] = (ali_dep["votos"] / ali_dep["votos"].sum() * 100).round(2)
            c1, c2 = st.columns([2, 1.2])
            with c1:
                st.bar_chart(ali_dep.set_index("ALIANZA")["% sobre válidos"], use_container_width=True)
            with c2:
                st.dataframe(ali_dep, use_container_width=True)
        else:
            st.info("Sin datos de alianzas para el departamento elegido con los filtros actuales.")

    st.markdown("---")
    st.markdown("#### Totales por Departamento y % sobre padrón")
    c1, c2 = st.columns([2, 1.2])
    with c1:
        if not dept_res.empty:
            st.bar_chart(dept_res.set_index("DEPARTAMENTO")["VOTOS_EMITIDOS"], use_container_width=True)
        else:
            st.info("Sin datos por departamento.")
    with c2:
        st.dataframe(dept_res[["DEPARTAMENTO", "VOTOS_EMITIDOS", "PADRON", "% SOBRE PADRON"]], use_container_width=True)

    st.markdown("#### Matriz - % por Alianza dentro de cada Departamento (sobre válidos)")
    dept_pct_valid = pivot_pct_valid(flt, ["DEPARTAMENTO"])
    st.dataframe(dept_pct_valid, use_container_width=True)

# ---------------- Municipios ----------------
with tab_muni:
    st.markdown("### % de votos por Alianza — Municipio seleccionado")

    # Opciones de municipio según filtros actuales del sidebar
    if dept_sel != "(Todos)":
        muni_opts = sorted(long.loc[long["DEPARTAMENTO"] == dept_sel, "MUNICIPIO"].dropna().unique().tolist())
    else:
        muni_opts = sorted(long["MUNICIPIO"].dropna().unique().tolist())

    # Elegir municipio (si el sidebar no tiene uno específico)
    if muni_sel == "(Todos)":
        muni_for_chart = st.selectbox("Elegí un municipio", muni_opts, key="muni_for_pct") if muni_opts else None
    else:
        muni_for_chart = muni_sel

    # % por alianza sobre válidos del municipio elegido
    if muni_for_chart:
        msel = flt[flt["MUNICIPIO"] == muni_for_chart]
        if dept_sel != "(Todos)":
            msel = msel[msel["DEPARTAMENTO"] == dept_sel]

        ali_muni = (
            msel.dropna(subset=["ALIANZA"])
                .groupby("ALIANZA", as_index=False)["votos"].sum()
                .sort_values("votos", ascending=False)
        )

        if not ali_muni.empty:
            ali_muni["% sobre válidos"] = (ali_muni["votos"] / ali_muni["votos"].sum() * 100).round(2)
            c1, c2 = st.columns([2, 1.2])
            with c1:
                st.bar_chart(ali_muni.set_index("ALIANZA")["% sobre válidos"], use_container_width=True)
            with c2:
                st.dataframe(ali_muni, use_container_width=True)
        else:
            st.info("Sin datos de alianzas para el municipio elegido con los filtros actuales.")
    else:
        st.info("Seleccioná un municipio para ver la distribución por alianza.")

    st.markdown("---")
    st.markdown("#### Totales por Municipio y % sobre padrón (referencia)")
    show = muni_res if dept_sel == "(Todos)" else muni_res[muni_res["DEPARTAMENTO"] == dept_sel]
    cols = ["DEPARTAMENTO", "MUNICIPIO", "VOTOS_EMITIDOS"]
    if "PADRON" in show.columns:
        cols += ["PADRON", "% SOBRE PADRON"]
    st.dataframe(show[cols], use_container_width=True)

    st.markdown("#### Matriz — % por Alianza dentro de cada Municipio (sobre válidos)")
    muni_pct_valid = pivot_pct_valid(flt, ["DEPARTAMENTO", "MUNICIPIO"])
    st.dataframe(muni_pct_valid, use_container_width=True)

# ---------------- Testigo ----------------
testigo_flt = flt.loc[flt["TESTIGO_BOOL"] == True].copy()
ali_testigo = pd.DataFrame()
if not testigo_flt.empty:
    ali_testigo = (
        testigo_flt.dropna(subset=["ALIANZA"])
                   .groupby("ALIANZA", as_index=False)["votos"].sum()
                   .sort_values("votos", ascending=False)
    )
    ali_testigo["% sobre válidos"] = (ali_testigo["votos"] / ali_testigo["votos"].sum() * 100).round(2)

with tab_testigo:
    st.markdown("#### Solo Mesas Testigo - % por Alianza")
    c1, c2 = st.columns([2, 1.2])
    with c1:
        if not ali_testigo.empty:
            st.bar_chart(ali_testigo.set_index("ALIANZA")["% sobre válidos"], use_container_width=True)
        else:
            st.info("No hay mesas testigo con datos bajo los filtros seleccionados.")
    with c2:
        st.dataframe(ali_testigo, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Detalle por Mesa Testigo (votos por alianza + blancos, nulos, recurridos, impugnados)")

    # Tabla por mesa: alianzas como columnas
    if not testigo_flt.empty:
        wide_ali = (
            testigo_flt.pivot_table(
                index=["MESA_KEY", "DEPARTAMENTO", "MUNICIPIO"],
                columns="ALIANZA",
                values="votos",
                aggfunc="sum",
                fill_value=0
            )
            .reset_index()
        )

        # Extras desde Respuestas_raw
        mesa_col = "Mesa" if "Mesa" in df_raw.columns else (find_col(df_raw, r"^\s*mesa\s*$") or "Mesa")
        df_ex = df_raw.copy()
        df_ex["MESA_KEY"] = normalize_mesa(df_ex.get(mesa_col, pd.Series(index=df_ex.index)))

        extra_cols = [c for c in df_raw.columns if re.search(r"votos?\s+(en\s+)?(blancos|nulos|recurridos|impugnados)", c, re.I)]
        extras_clean = {}
        for c in extra_cols:
            cname = c.upper()
            if re.search(r"BLANCO", cname):
                extras_clean[c] = "BLANCOS"
            elif re.search(r"NULO", cname):
                extras_clean[c] = "NULOS"
            elif re.search(r"RECURR", cname):
                extras_clean[c] = "RECURRIDOS"
            elif re.search(r"IMPUGN", cname):
                extras_clean[c] = "IMPUGNADOS"
            else:
                extras_clean[c] = c  # fallback

        for c in extra_cols:
            df_ex[c] = pd.to_numeric(df_ex[c], errors="coerce").fillna(0).astype(int)

        if extra_cols:
            grp = df_ex.groupby("MESA_KEY", as_index=False)[extra_cols].sum()
            grp = grp.rename(columns=extras_clean)
            wide_ali = wide_ali.merge(grp, on="MESA_KEY", how="left")

        wide_ali = wide_ali.fillna(0).sort_values(["DEPARTAMENTO", "MUNICIPIO", "MESA_KEY"])
        st.dataframe(wide_ali, use_container_width=True, height=500)
    else:
        st.info("No hay filas testigo para mostrar detalle por mesa.")

# ---------------- Mesas (detalle) ----------------
pivot_all = (
    flt.dropna(subset=["ALIANZA"])
       .pivot_table(index=["MESA_KEY", "DEPARTAMENTO", "MUNICIPIO", "ESTABLECIMIENTO", "TESTIGO_BOOL"],
                    columns="ALIANZA", values="votos", aggfunc="sum", fill_value=0)
       .reset_index()
       .sort_values(["DEPARTAMENTO", "MUNICIPIO", "MESA_KEY"])
)
total_mesas_plan = int(df_esc["MESA_KEY"].nunique())
mesas_escrutadas = int(df_mesas_all.loc[df_mesas_all["TOTAL_MESA"] > 0, "MESA_KEY"].nunique())
pct_mesas_escrutadas = (mesas_escrutadas / total_mesas_plan * 100) if total_mesas_plan > 0 else 0.0

with tab_mesas:
    st.markdown("#### Detalle por Mesa (todas las alianzas)")
    st.dataframe(pivot_all, height=520, use_container_width=True)
    st.caption(f"% de mesas escrutadas (provincial): {mesas_escrutadas} / {total_mesas_plan} = {pct_mesas_escrutadas:.2f}%")

# ---------------- Diagnóstico ----------------
raw_mesa_col = "Mesa" if "Mesa" in df_raw.columns else (find_col(df_raw, r"^\s*mesa\s*$") or "Mesa")
df_raw["_MESA_KEY"] = normalize_mesa(df_raw.get(raw_mesa_col, pd.Series(index=df_raw.index)))
ser_mesas = df_raw["_MESA_KEY"].dropna().astype("Int64")
dupes_count = (
    ser_mesas.value_counts(dropna=True)
            .rename_axis("MESA_KEY")
            .reset_index(name="CANTIDAD_CARGAS")
)
dupes_count = dupes_count.loc[dupes_count["CANTIDAD_CARGAS"] > 1].sort_values(
    by=["CANTIDAD_CARGAS", "MESA_KEY"], ascending=[False, True]
).reset_index(drop=True)

with tab_diag:
    st.markdown("#### Mesas duplicadas en Respuestas_raw")
    if not dupes_count.empty:
        st.dataframe(dupes_count, use_container_width=True)
        st.info("Revisar estas mesas en Respuestas_raw para resolver duplicidad.")
    else:
        st.success("No se detectaron mesas duplicadas.")

# ---------------- Balotaje ----------------
with tab_bal:
    st.markdown("### Chequeo de Balotaje (sobre votos válidos por alianza)")
    st.caption("Se calcula con los **filtros del sidebar** aplicados (testigo, rango de mesa, etc.).")

    ambito = st.radio("Ámbito", ["Provincia", "Departamento", "Municipio"], horizontal=True)

    # Usamos el dataset filtrado actual
    base = flt.copy()

    if ambito == "Provincia":
        votes = base[["ALIANZA", "votos"]].copy()
        titulo = "Provincia (según filtros)"
    elif ambito == "Departamento":
        dept_opts = sorted(long["DEPARTAMENTO"].dropna().unique().tolist())
        # por defecto, usar dept_sel si está definido
        idx = dept_opts.index(dept_sel) if (dept_sel != "(Todos)" and dept_sel in dept_opts) else 0
        dept_pick = st.selectbox("Departamento", dept_opts, index=idx)
        votes = base.loc[base["DEPARTAMENTO"] == dept_pick, ["ALIANZA", "votos"]].copy()
        titulo = f"Departamento: {dept_pick}"
    else:  # Municipio
        dep_opts = sorted(long["DEPARTAMENTO"].dropna().unique().tolist())
        idx_dep = dep_opts.index(dept_sel) if (dept_sel != "(Todos)" and dept_sel in dep_opts) else 0
        dep_pick = st.selectbox("Departamento", dep_opts, index=idx_dep, key="bal_dept")

        muni_opts = sorted(long.loc[long["DEPARTAMENTO"] == dep_pick, "MUNICIPIO"].dropna().unique().tolist())
        idx_m = muni_opts.index(muni_sel) if (muni_sel != "(Todos)" and muni_sel in muni_opts) else 0
        muni_pick = st.selectbox("Municipio", muni_opts, index=idx_m, key="bal_muni")

        votes = base.loc[
            (base["DEPARTAMENTO"] == dep_pick) & (base["MUNICIPIO"] == muni_pick),
            ["ALIANZA", "votos"]
        ].copy()
        titulo = f"Municipio: {muni_pick} ({dep_pick})"

    regla_lbl = st.selectbox("Regla de balotaje", ["nacional (45% o 40%+10)", "mayoría absoluta (50%+1)"])
    regla = "nacional" if regla_lbl.startswith("nacional") else "mayoria_absoluta"

    resumen, ranking = ballotage_status(votes, rule=regla)

    if resumen is None or ranking.empty:
        st.info("No hay votos válidos suficientes para evaluar balotaje en este ámbito.")
    else:
        gano = "Sí" if resumen["gana_primera_vuelta"] else "No"
        st.metric("¿Ganador en 1ª vuelta?", gano, help=f"Ámbito: {titulo} · Regla: {regla_lbl}")

        # Gráfico y tabla
        ranking = ranking.sort_values("pct", ascending=False).rename(columns={"pct": "% sobre válidos"})
        c1, c2 = st.columns([2, 1.2])
        with c1:
            st.bar_chart(ranking.set_index("ALIANZA")["% sobre válidos"], use_container_width=True)
        with c2:
            st.dataframe(ranking, use_container_width=True)

        st.caption(
            "Se consideran **votos válidos** (solo alianzas). "
            "Quedan excluidos blancos, nulos, recurridos e impugnados."
        )

# ================== FOOTER ==================
st.markdown("---")
st.caption(f"Mesas cargadas (TOTAL_MESA > 0): {mesas_escrutadas} / {TOTAL_MESAS_PROV}")











