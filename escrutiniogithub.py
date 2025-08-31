# -*- coding: utf-8 -*-
"""
Escrutinio - Dashboard Streamlit (sin mapas)

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
import re, unicodedata
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ================== CONFIG ==================
SHEET_ID = "1vYiQvkDqdx-zgtRbNPN5_0l2lXAceTF2py4mlM1pK_U"
SHEET_NAMES = {
    "raw": "Respuestas_raw",
    "escuelas": "Mapeo_Escuelas_raw",
    "alianzas": "Mapeo_Alianzas_raw",
    "mesa_muni": "Mapeo_Mesa_Municipio_raw",
}
AUTOREFRESH_SEC = 180      # 3 minutos
TOTAL_MESAS_PROV = 2808    # para footer/encabezados

st.set_page_config(page_title="Escrutinio - Dashboard", page_icon="🗳️", layout="wide")

# ================== COLORES POR ALIANZA ==================
ALLIANCE_COLORS = {
    "vamos corrientes": "#FFD500",   # amarillo
    "eco": "#2E7D32",                # verde
    "la libertad avanza": "#7E57C2", # violeta
    "libertad avanza": "#7E57C2",    # alias
    "limpia corrientes": "#1976D2",  # azul
}
FALLBACK_PALETTE = [
    "#6D4C41","#00897B","#8D6E63","#5E35B1","#00796B","#C0CA33",
    "#8E24AA","#039BE5","#43A047","#FB8C00","#26A69A","#7CB342",
    "#8E44AD","#546E7A"
]

def norm_txt(x: str) -> str:
    x = str(x).strip()
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", x).strip().lower()

def color_for_alliance(name: str) -> str | None:
    n = norm_txt(name)
    if "vamos corrientes" in n: return ALLIANCE_COLORS["vamos corrientes"]
    if "libertad avanza" in n:  return ALLIANCE_COLORS["libertad avanza"]
    if re.search(r"\beco\b", n): return ALLIANCE_COLORS["eco"]
    if "limpia corrientes" in n: return ALLIANCE_COLORS["limpia corrientes"]
    return None

def build_color_scale(names: List[str]) -> alt.Scale:
    domain, range_ = [], []
    fb_idx = 0
    for name in names:
        c = color_for_alliance(name)
        if c is None:
            c = FALLBACK_PALETTE[fb_idx % len(FALLBACK_PALETTE)]
            fb_idx += 1
        domain.append(name); range_.append(c)
    return alt.Scale(domain=domain, range=range_)

# ================== PADRON FALLBACK ==================
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
    if not isinstance(pk, str): return pk
    s = pk.strip().replace("\\r\\n","\n").replace("\r\n","\n").replace("\r","\n").replace("\\n","\n")
    s = s.replace("—","-").replace("–","-")
    start, end = "-----BEGIN PRIVATE KEY-----", "-----END PRIVATE KEY-----"
    if start not in s or end not in s: return s
    _, rest = s.split(start, 1); body, _ = rest.split(end, 1)
    lines = [ln.strip() for ln in body.strip().split("\n") if ln.strip()]
    import re as _re
    b64 = _re.sub(r"[^A-Za-z0-9+/=]", "", "".join(lines))
    wrapped = "\n".join([b64[i:i+64] for i in range(0,len(b64),64)])
    return f"{start}\n{wrapped}\n{end}\n"

@st.cache_resource
def _gspread_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as e:
        st.error("Faltan dependencias (gspread/google-auth). " + str(e)); st.stop()

    if "gcp_service_account" not in st.secrets:
        st.error("Falta el bloque [gcp_service_account] en secrets."); st.stop()

    info = dict(st.secrets["gcp_service_account"])
    info["private_key"] = _normalize_private_key(info.get("private_key",""))
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    from google.oauth2.service_account import Credentials
    try:
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        import gspread
        return gspread.authorize(creds)
    except Exception as e:
        st.error("No se pudo crear el cliente de Google Sheets.\n\n" + str(e)); st.stop()

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
            df_muni = pd.DataFrame(columns=["MESA","MUNICIPIO"])
    except Exception as e:
        st.error("No se pudo abrir el Google Sheet. Revisá permisos, nombres de hoja y APIs habilitadas.\n\n" + str(e))
        st.stop()
    return df_raw, df_esc, df_ali, df_muni

@st.cache_data(ttl=AUTOREFRESH_SEC)
def load_padron_depto() -> pd.DataFrame:
    gc = _gspread_client()
    try:
        sh = gc.open_by_key(SHEET_ID)
        titles = [w.title for w in sh.worksheets()]
        if "Padron_Departamento_raw" in titles:
            ws = sh.worksheet("Padron_Departamento_raw")
            rows = ws.get_all_values()
            dfp = pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame()
            if dfp.empty: return pd.DataFrame(PADRON_FALLBACK)
            if "DEPARTAMENTO" not in dfp.columns: return pd.DataFrame(PADRON_FALLBACK)
            pcol = "PADRON" if "PADRON" in dfp.columns else next((c for c in dfp.columns if re.search("padron", c, re.I)), None)
            if not pcol: return pd.DataFrame(PADRON_FALLBACK)
            dfp["DEPARTAMENTO"] = dfp["DEPARTAMENTO"].astype(str).str.strip()
            dfp["PADRON"] = pd.to_numeric(
                dfp[pcol].astype(str).str.replace(".","",regex=False).str.replace(",","",regex=False),
                errors="coerce"
            ).fillna(0).astype(int)
            return dfp[["DEPARTAMENTO","PADRON"]]
    except Exception:
        pass
    return pd.DataFrame(PADRON_FALLBACK)

@st.cache_data(ttl=AUTOREFRESH_SEC)
def load_padron_municipio() -> pd.DataFrame:
    gc = _gspread_client()
    try:
        sh = gc.open_by_key(SHEET_ID)
        titles = [w.title for w in sh.worksheets()]
        if "Padron_Municipio_raw" in titles:
            ws = sh.worksheet("Padron_Municipio_raw")
            rows = ws.get_all_values()
            df = pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame()
            if df.empty: return df
            if "MUNICIPIO" not in df.columns:
                mcol = next((c for c in df.columns if re.search("municip", c, re.I)), None)
                if mcol: df = df.rename(columns={mcol:"MUNICIPIO"})
            if "DEPARTAMENTO" not in df.columns:
                dcol = next((c for c in df.columns if re.search("depart", c, re.I)), None)
                if dcol: df = df.rename(columns={dcol:"DEPARTAMENTO"})
            if "PADRON" in df.columns: pcol = "PADRON"
            elif "VOTANTES" in df.columns: pcol = "VOTANTES"
            else: pcol = next((c for c in df.columns if re.search("padron|votantes", c, re.I)), None)
            if not pcol: return pd.DataFrame()
            df["MUNICIPIO"] = df["MUNICIPIO"].astype(str).str.strip()
            if "DEPARTAMENTO" in df.columns:
                df["DEPARTAMENTO"] = df["DEPARTAMENTO"].astype(str).str.strip()
            df["PADRON"] = pd.to_numeric(
                df[pcol].astype(str).str.replace(".","",regex=False).str.replace(",","",regex=False),
                errors="coerce"
            ).fillna(0).astype(int)
            df["_mkey"] = df["MUNICIPIO"].map(norm_txt)
            if "DEPARTAMENTO" in df.columns:
                df["_dkey"] = df["DEPARTAMENTO"].map(norm_txt)
            return df
    except Exception:
        pass
    return pd.DataFrame()

# ================== HELPERS ==================
def normalize_mesa(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]","",regex=True).replace({"":np.nan}).astype("Int64")

def bool_from_any(s: pd.Series) -> pd.Series:
    up = s.astype(str).str.upper().str.strip()
    return up.isin(["TRUE","1","SI","SÍ","YES","VERDADERO"])

def find_col(df: pd.DataFrame, regex: str) -> str | None:
    for c in df.columns:
        if re.search(regex, str(c), re.IGNORECASE): return c
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
    long = long.dropna(subset=["MESA_KEY","numero_partido"]).astype({"numero_partido":int})
    long["PARTIDO_NOMBRE_HEADER"] = long["partido_header"].astype(str).str.replace(r"^\s*\d+\s*-\s*","",regex=True).str.strip()
    return long

def mesa_totales(df_raw: pd.DataFrame) -> pd.DataFrame:
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
        return out[["MESA_KEY","TOTAL_MESA"]]
    else:
        df_tot = df_raw[[mesa_col, tot_col]].copy()
        df_tot["MESA_KEY"] = normalize_mesa(df_tot[mesa_col])
        df_tot["TOTAL_MESA"] = pd.to_numeric(
            df_tot[tot_col].astype(str).str.replace(".","",regex=False).str.replace(",","",regex=False),
            errors="coerce"
        ).fillna(0).astype(int)
        return df_tot[["MESA_KEY","TOTAL_MESA"]]

def pivot_pct_valid(df_long: pd.DataFrame, region_cols: list[str]) -> pd.DataFrame:
    g = (df_long.dropna(subset=["ALIANZA"])
               .groupby(region_cols + ["ALIANZA"], as_index=False)["votos"].sum())
    tot = g.groupby(region_cols, as_index=False)["votos"].sum().rename(columns={"votos":"TOTAL_REGION"})
    g = g.merge(tot, on=region_cols, how="left")
    g["%"] = np.where(g["TOTAL_REGION"]>0, g["votos"]/g["TOTAL_REGION"]*100, 0)
    pvt = (g.pivot_table(index=region_cols, columns="ALIANZA", values="%", aggfunc="first")
             .fillna(0).round(2).reset_index())
    return pvt

def bar_pct_chart(df: pd.DataFrame, cat_col: str, val_col: str, title: str = "") -> alt.Chart:
    if df.empty: return alt.Chart(pd.DataFrame({"x":[],"y":[]})).mark_bar()
    df = df.sort_values(val_col, ascending=True)
    names = df[cat_col].tolist()
    scale = build_color_scale(names)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{val_col}:Q", title=val_col),
            y=alt.Y(f"{cat_col}:N", sort=None, title=""),
            color=alt.Color(f"{cat_col}:N", scale=scale, legend=None),
            tooltip=[cat_col, alt.Tooltip(f"{val_col}:Q", format=".2f")]
        )
        .properties(height=max(240, 24 * len(df)), title=title)
    )

def bar_value_chart(df: pd.DataFrame, cat_col: str, val_col: str, title: str = "") -> alt.Chart:
    if df.empty: return alt.Chart(pd.DataFrame({"x":[],"y":[]})).mark_bar()
    df = df.sort_values(val_col, ascending=True)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{val_col}:Q", title=val_col),
            y=alt.Y(f"{cat_col}:N", sort=None, title=""),
            tooltip=[cat_col, alt.Tooltip(f"{val_col}:Q", format=",.0f")]
        )
        .properties(height=max(240, 24 * len(df)), title=title)
    )

# ================== Balotaje ==================
def ballotage_status(votes_df: pd.DataFrame, rule: str = "nacional") -> tuple[dict | None, pd.DataFrame]:
    df = (votes_df.dropna(subset=["ALIANZA"])
                  .loc[votes_df["ALIANZA"] != "(Sin alianza)"]
                  .groupby("ALIANZA", as_index=False)["votos"].sum())
    total = df["votos"].sum()
    if total <= 0 or df.empty: return None, df
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
    return resumen, df[["ALIANZA","votos","pct"]]

# ================== PREP PIPELINE ==================
def prep_data():
    df_raw, df_esc, df_ali, df_muni = load_data()
    if df_raw.empty: st.warning("Respuestas_raw está vacío.")
    if df_esc.empty: st.warning("Mapeo_Escuelas_raw está vacío.")
    if df_ali.empty: st.warning("Mapeo_Alianzas_raw está vacío.")
    if df_muni.empty: st.info("Mapeo_Mesa_Municipio_raw está vacío o no existe.")

    # Escuelas -> DEPARTAMENTO / TESTIGO
    mesa_esc_col = "MESA" if "MESA" in df_esc.columns else find_col(df_esc, r"\bmesa\b")
    df_esc["MESA_KEY"] = normalize_mesa(df_esc[mesa_esc_col]) if mesa_esc_col else pd.Series(dtype="Int64")
    test_col = "TESTIGO" if "TESTIGO" in df_esc.columns else find_col(df_esc, r"testig")
    df_esc["TESTIGO_BOOL"] = bool_from_any(df_esc[test_col]) if test_col else False

    # Municipios (DEPARTAMENTO opcional)
    df_muni_norm = pd.DataFrame(columns=["MESA_KEY","MUNICIPIO"])
    if not df_muni.empty:
        dm = df_muni.copy(); dm.columns = [str(c).strip() for c in dm.columns]
        upper_map = {c: c.strip().upper() for c in dm.columns}; dm.rename(columns=upper_map, inplace=True)
        def _findcol(cols, pattern):
            for c in cols:
                if re.search(pattern, c, re.I): return c
            return None
        mesa_c = "MESA" if "MESA" in dm.columns else _findcol(dm.columns, r"\bmesa\b")
        muni_c = "MUNICIPIO" if "MUNICIPIO" in dm.columns else _findcol(dm.columns, r"munic")
        dep_c  = "DEPARTAMENTO" if "DEPARTAMENTO" in dm.columns else _findcol(dm.columns, r"depart")

        if mesa_c and muni_c:
            dm["MESA_KEY"] = normalize_mesa(dm[mesa_c])
            dm["MUNICIPIO"] = dm[muni_c].astype(str).str.strip()
            if dep_c: dm["DEPARTAMENTO"] = dm[dep_c].astype(str).str.strip()
            dm = dm.dropna(subset=["MESA_KEY"]).drop_duplicates(subset=["MESA_KEY"])
            keep_cols = ["MESA_KEY","MUNICIPIO"]
            if "DEPARTAMENTO" in dm.columns: keep_cols.append("DEPARTAMENTO")
            df_muni_norm = dm[keep_cols].copy()

    df_muni_only = (
        df_muni_norm.loc[:,["MESA_KEY","MUNICIPIO"]].drop_duplicates("MESA_KEY")
        if not df_muni_norm.empty else pd.DataFrame(columns=["MESA_KEY","MUNICIPIO"])
    )

    # Votos long
    parties = detect_party_columns(df_raw)
    long = tidy_votes(df_raw, parties)

    # Alianzas
    num_col = "numero" if "numero" in df_ali.columns else find_col(df_ali, r"^\s*numero\s*$")
    ali_col = "Alianza" if "Alianza" in df_ali.columns else find_col(df_ali, r"^\s*alianza\s*$")
    party_name_col = find_col(df_ali, r"Partidos?\s+pol")
    if not (num_col and ali_col):
        st.error("En Mapeo_Alianzas_raw deben existir columnas 'numero' y 'Alianza'."); st.stop()

    df_ali["_numero"] = pd.to_numeric(df_ali[num_col], errors="coerce")
    ali_map = df_ali[["_numero", ali_col]].rename(columns={"_numero":"numero_partido", ali_col:"ALIANZA"})
    long = long.merge(ali_map, on="numero_partido", how="left")
    long["ALIANZA"] = long["ALIANZA"].fillna("")
    no_usar = long["ALIANZA"].str.strip().str.match(r"(?i)^\s*no\s*usar\s*$")
    long.loc[no_usar,"ALIANZA"] = np.nan
    long["ALIANZA"] = long["ALIANZA"].where(long["ALIANZA"].notna(), "(Sin alianza)")

    if party_name_col:
        name_map = df_ali[["_numero", party_name_col]].rename(columns={"_numero":"numero_partido", party_name_col:"PARTIDO"})
        long = long.merge(name_map, on="numero_partido", how="left")
    else:
        long["PARTIDO"] = long["PARTIDO_NOMBRE_HEADER"]

    # Merge final (DEPARTAMENTO desde Escuelas, MUNICIPIO desde Mapeo_Mesa_Municipio)
    keep_esc = [c for c in ["DEPARTAMENTO","ESTABLECIMIENTO","TESTIGO_BOOL"] if c in df_esc.columns]
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

    # Totales por mesa (y tabla "mesas all")
    df_tot = mesa_totales(df_raw)
    df_mesas_all = df_tot.merge(
        df_esc[["MESA_KEY","DEPARTAMENTO","TESTIGO_BOOL"]].drop_duplicates("MESA_KEY"),
        on="MESA_KEY", how="left"
    )
    df_mesas_all["DEPARTAMENTO"] = df_mesas_all["DEPARTAMENTO"].fillna("(Sin depto)")
    if not df_muni_only.empty:
        df_mesas_all = df_mesas_all.merge(df_muni_only, on="MESA_KEY", how="left")
    if "MUNICIPIO" not in df_mesas_all.columns:
        df_mesas_all["MUNICIPIO"] = np.nan
    df_mesas_all["MUNICIPIO"] = df_mesas_all["MUNICIPIO"].fillna("(Sin municipio)")

    # Diagnóstico discrepancias depto entre mapeos (si ambos tienen depto)
    dept_mismatch = pd.DataFrame()
    if (not df_muni_norm.empty) and ("DEPARTAMENTO" in df_muni_norm.columns) and ("DEPARTAMENTO" in df_esc.columns):
        d1 = df_esc[["MESA_KEY","DEPARTAMENTO"]].drop_duplicates("MESA_KEY").rename(columns={"DEPARTAMENTO":"DEP_ESC"})
        d2 = df_muni_norm[["MESA_KEY","DEPARTAMENTO"]].drop_duplicates("MESA_KEY").rename(columns={"DEPARTAMENTO":"DEP_MAP"})
        mm = d1.merge(d2, on="MESA_KEY", how="inner")
        dept_mismatch = mm[(mm["DEP_ESC"].notna()) & (mm["DEP_MAP"].notna()) & (mm["DEP_ESC"] != mm["DEP_MAP"])]

    # Para progreso por municipio: planificadas
    df_muni_plan = pd.DataFrame(columns=["MESA_KEY","MUNICIPIO","DEPARTAMENTO"])
    if not df_muni_only.empty:
        df_muni_plan = df_muni_only.merge(
            df_esc[["MESA_KEY","DEPARTAMENTO"]].drop_duplicates("MESA_KEY"),
            on="MESA_KEY", how="left"
        )
        df_muni_plan["DEPARTAMENTO"] = df_muni_plan["DEPARTAMENTO"].fillna("(Sin depto)")

    return df_raw, df_esc, df_ali, long, df_mesas_all, dept_mismatch, df_muni_plan

# ================== UI ==================
st.title("📊 Escrutinio - CENTRO DE COMPUTOS ECO")
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
df_raw, df_esc, df_ali, long, df_mesas_all, dept_mismatch, df_muni_plan = prep_data()

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

# ====== PROGRESO PROVINCIAL ======
df_pad_depto = load_padron_depto()
total_padron_prov   = int(df_pad_depto["PADRON"].sum()) if not df_pad_depto.empty else 0
total_emitidos_prov = int(df_mesas_all["TOTAL_MESA"].sum())
pct_participacion   = (total_emitidos_prov / total_padron_prov * 100) if total_padron_prov > 0 else 0.0

total_mesas_plan    = int(df_esc["MESA_KEY"].nunique()) if "MESA_KEY" in df_esc.columns else TOTAL_MESAS_PROV
mesas_escrutadas    = int(df_mesas_all.loc[df_mesas_all["TOTAL_MESA"] > 0, "MESA_KEY"].nunique())
pct_mesas_escrutadas = (mesas_escrutadas / total_mesas_plan * 100) if total_mesas_plan > 0 else 0.0

st.subheader("Progreso provincial")
c_prog1, c_prog2 = st.columns(2)
with c_prog1:
    st.metric("Mesas escrutadas", f"{mesas_escrutadas} / {total_mesas_plan}", f"{pct_mesas_escrutadas:.2f}%")
with c_prog2:
    st.metric("% de participación", f"{pct_participacion:.2f}%", help="Votos emitidos / Padrón total")
st.divider()

# -------- Datos base para depto/muni --------
ali_flt = (
    flt.dropna(subset=["ALIANZA"])
       .groupby(["DEPARTAMENTO","MUNICIPIO","ALIANZA"], as_index=False)["votos"].sum()
)

# Totales por departamento y padrón (con filtros)
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
              .sum().rename(columns={"TOTAL_MESA":"VOTOS_EMITIDOS"})
)
tmp = dept_tot.copy(); tmp["_key"] = tmp["DEPARTAMENTO"].map(norm_txt)
pad = df_pad_depto.copy(); pad["_key"] = pad["DEPARTAMENTO"].map(norm_txt)
dept_res = tmp.merge(pad[["_key","PADRON"]], on="_key", how="left").drop(columns=["_key"])
dept_res["PADRON"] = dept_res["PADRON"].fillna(0).astype(int)
dept_res["% SOBRE PADRON"] = np.where(
    dept_res["PADRON"]>0, (dept_res["VOTOS_EMITIDOS"]/dept_res["PADRON"]*100).round(2), 0.0
)
dept_res = dept_res.sort_values("VOTOS_EMITIDOS", ascending=False)

# Municipios - totales y padrón (con filtros)
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
    df_mesas_fm.groupby(["DEPARTAMENTO","MUNICIPIO"], as_index=False)["TOTAL_MESA"]
               .sum().rename(columns={"TOTAL_MESA":"VOTOS_EMITIDOS"})
)
if not df_pad_muni.empty:
    muni_tmp = muni_tot.copy()
    muni_tmp["_mkey"] = muni_tmp["MUNICIPIO"].map(norm_txt)
    df_pad_muni["_mkey"] = df_pad_muni["MUNICIPIO"].map(norm_txt)
    muni_res = muni_tmp.merge(df_pad_muni[["_mkey","PADRON"]], on="_mkey", how="left").drop(columns=["_mkey"])
    muni_res["PADRON"] = muni_res["PADRON"].fillna(0).astype(int)
    muni_res["% SOBRE PADRON"] = np.where(
        muni_res["PADRON"]>0, (muni_res["VOTOS_EMITIDOS"]/muni_res["PADRON"]*100).round(2), 0.0
    )
else:
    muni_res = muni_tot.copy()
muni_res = muni_res.sort_values(["DEPARTAMENTO","VOTOS_EMITIDOS"], ascending=[True, False])

# ====== PESTAÑAS ======
tab_ali, tab_dept, tab_muni, tab_testigo, tab_mesas, tab_bal, tab_diag = st.tabs([
    "🧩 Alianzas", "🏙️ Departamentos", "🏘️ Municipios",
    "⭐ Testigo", "🗃️ Mesas (detalle)", "🗳️ Balotaje", "🔧 Diagnóstico",
])

# ---------------- Alianzas ----------------
with tab_ali:
    ali_df = (
        flt.dropna(subset=["ALIANZA"])
           .groupby("ALIANZA", as_index=False)["votos"].sum()
           .sort_values("votos", ascending=False)
    )
    if not ali_df.empty:
        ali_df["% sobre válidos"] = (ali_df["votos"] / ali_df["votos"].sum() * 100).round(2)
        chart = bar_pct_chart(ali_df.rename(columns={"ALIANZA":"Alianza"}), "Alianza", "% sobre válidos", "Votos por Alianza (filtros aplicados)")
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(ali_df, use_container_width=True)
    else:
        st.info("Sin datos de alianzas para el filtro.")

# ---------------- Departamentos ----------------
with tab_dept:
    st.markdown("### % de votos por Alianza - Departamento seleccionado")
    if dept_sel == "(Todos)":
        dept_opts = sorted(long["DEPARTAMENTO"].dropna().unique().tolist())
        dept_for_chart = st.selectbox("Elegí un departamento", dept_opts, key="dept_for_pct") if dept_opts else None
    else:
        dept_for_chart = dept_sel
    if dept_for_chart:
        dsel = flt[flt["DEPARTAMENTO"] == dept_for_chart]
        ali_dep = (
            dsel.dropna(subset=["ALIANZA"])
                .groupby("ALIANZA", as_index=False)["votos"].sum()
                .sort_values("votos", ascending=False)
        )
        if not ali_dep.empty:
            ali_dep["% sobre válidos"] = (ali_dep["votos"] / ali_dep["votos"].sum() * 100).round(2)
            chart = bar_pct_chart(ali_dep.rename(columns={"ALIANZA":"Alianza"}), "Alianza", "% sobre válidos", f"Distribución en {dept_for_chart}")
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(ali_dep, use_container_width=True)
        else:
            st.info("Sin datos de alianzas para el departamento elegido con los filtros actuales.")

    st.markdown("---")
    st.markdown("#### Totales por Departamento y % sobre padrón")
    c1, c2 = st.columns([2, 1.2])
    with c1:
        if not dept_res.empty:
            chart = bar_value_chart(dept_res.rename(columns={"DEPARTAMENTO":"Departamento"}), "Departamento", "VOTOS_EMITIDOS", "Votos emitidos por departamento")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Sin datos por departamento.")
    with c2:
        st.dataframe(dept_res[["DEPARTAMENTO","VOTOS_EMITIDOS","PADRON","% SOBRE PADRON"]], use_container_width=True)

    st.markdown("#### Matriz - % por Alianza dentro de cada Departamento (sobre válidos)")
    dept_pct_valid = pivot_pct_valid(flt, ["DEPARTAMENTO"])
    st.dataframe(dept_pct_valid, use_container_width=True)

# ---------------- Municipios ----------------
with tab_muni:
    st.markdown("### % de votos por Alianza — Municipio seleccionado")
    if dept_sel != "(Todos)":
        muni_opts = sorted(long.loc[long["DEPARTAMENTO"] == dept_sel, "MUNICIPIO"].dropna().unique().tolist())
    else:
        muni_opts = sorted(long["MUNICIPIO"].dropna().unique().tolist())
    if muni_sel == "(Todos)":
        muni_for_chart = st.selectbox("Elegí un municipio", muni_opts, key="muni_for_pct") if muni_opts else None
    else:
        muni_for_chart = muni_sel

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
            ali_muni["% sobre válidos"] = (ali_muni["votos"]/ali_muni["votos"].sum()*100).round(2)
            chart = bar_pct_chart(ali_muni.rename(columns={"ALIANZA":"Alianza"}), "Alianza", "% sobre válidos", f"Distribución en {muni_for_chart}")
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(ali_muni, use_container_width=True)
        else:
            st.info("Sin datos de alianzas para el municipio elegido con los filtros actuales.")
    else:
        st.info("Seleccioná un municipio para ver la distribución por alianza.")

    st.markdown("---")
    st.markdown("#### Totales por Municipio y % sobre padrón")
    show = muni_res if dept_sel == "(Todos)" else muni_res[muni_res["DEPARTAMENTO"] == dept_sel]
    cols = ["DEPARTAMENTO","MUNICIPIO","VOTOS_EMITIDOS"]
    if "PADRON" in show.columns: cols += ["PADRON","% SOBRE PADRON"]
    st.dataframe(show[cols], use_container_width=True)

    st.markdown("#### Matriz — % por Alianza dentro de cada Municipio (sobre válidos)")
    muni_pct_valid = pivot_pct_valid(flt, ["DEPARTAMENTO","MUNICIPIO"])
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
    ali_testigo["% sobre válidos"] = (ali_testigo["votos"]/ali_testigo["votos"].sum()*100).round(2)

with tab_testigo:
    st.markdown("#### Solo Mesas Testigo - % por Alianza")
    if not ali_testigo.empty:
        chart = bar_pct_chart(ali_testigo.rename(columns={"ALIANZA":"Alianza"}), "Alianza", "% sobre válidos", "Distribución en mesas testigo")
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(ali_testigo, use_container_width=True)
    else:
        st.info("No hay mesas testigo con datos bajo los filtros seleccionados.")

    st.markdown("---")
    st.markdown("#### Detalle por Mesa Testigo (votos por alianza + blancos, nulos, recurridos, impugnados)")
    if not testigo_flt.empty:
        wide_ali = (
            testigo_flt.pivot_table(
                index=["MESA_KEY","DEPARTAMENTO","MUNICIPIO"],
                columns="ALIANZA", values="votos", aggfunc="sum", fill_value=0
            ).reset_index()
        )
        mesa_col = "Mesa" if "Mesa" in df_raw.columns else (find_col(df_raw, r"^\s*mesa\s*$") or "Mesa")
        df_ex = df_raw.copy()
        df_ex["MESA_KEY"] = normalize_mesa(df_ex.get(mesa_col, pd.Series(index=df_ex.index)))
        extra_cols = [c for c in df_raw.columns if re.search(r"votos?\s+(en\s+)?(blancos|nulos|recurridos|impugnados)", c, re.I)]
        extras_clean = {}
        for c in extra_cols:
            cname = c.upper()
            if re.search(r"BLANCO", cname): extras_clean[c]="BLANCOS"
            elif re.search(r"NULO", cname): extras_clean[c]="NULOS"
            elif re.search(r"RECURR", cname): extras_clean[c]="RECURRIDOS"
            elif re.search(r"IMPUGN", cname): extras_clean[c]="IMPUGNADOS"
            else: extras_clean[c]=c
        for c in extra_cols:
            df_ex[c] = pd.to_numeric(df_ex[c], errors="coerce").fillna(0).astype(int)
        if extra_cols:
            grp = df_ex.groupby("MESA_KEY", as_index=False)[extra_cols].sum().rename(columns=extras_clean)
            wide_ali = wide_ali.merge(grp, on="MESA_KEY", how="left")
        wide_ali = wide_ali.fillna(0).sort_values(["DEPARTAMENTO","MUNICIPIO","MESA_KEY"])
        st.dataframe(wide_ali, use_container_width=True, height=500)
    else:
        st.info("No hay filas testigo para mostrar detalle por mesa.")

# ---------------- Mesas (detalle) ----------------
pivot_all = (
    flt.dropna(subset=["ALIANZA"])
       .pivot_table(index=["MESA_KEY","DEPARTAMENTO","MUNICIPIO","ESTABLECIMIENTO","TESTIGO_BOOL"],
                    columns="ALIANZA", values="votos", aggfunc="sum", fill_value=0)
       .reset_index()
       .sort_values(["DEPARTAMENTO","MUNICIPIO","MESA_KEY"])
)

with tab_mesas:
    st.markdown("#### Detalle por Mesa (todas las alianzas)")
    st.dataframe(pivot_all, height=520, use_container_width=True)
    st.caption(f"Mesas cargadas: {mesas_escrutadas} / {TOTAL_MESAS_PROV}")

    # ====== Progreso por territorio ======
    st.markdown("---")
    st.subheader("Progreso por territorio")

    # Departamentos: planificadas vs escrutadas
    esc_plan = (
        df_esc[["DEPARTAMENTO","MESA_KEY"]].dropna().drop_duplicates()
             .groupby("DEPARTAMENTO", as_index=False)["MESA_KEY"].nunique()
             .rename(columns={"MESA_KEY":"PLANIFICADAS"})
    )
    esc_scru = (
        df_mesas_all.loc[df_mesas_all["TOTAL_MESA"]>0, ["DEPARTAMENTO","MESA_KEY"]]
                    .drop_duplicates()
                    .groupby("DEPARTAMENTO", as_index=False)["MESA_KEY"].nunique()
                    .rename(columns={"MESA_KEY":"ESCRUTADAS"})
    )
    prog_dept = esc_plan.merge(esc_scru, on="DEPARTAMENTO", how="left").fillna(0)
    prog_dept["RESTANTES"] = (prog_dept["PLANIFICADAS"] - prog_dept["ESCRUTADAS"]).clip(lower=0).astype(int)
    prog_dept["% ESCRUTADAS"] = np.where(prog_dept["PLANIFICADAS"]>0, (prog_dept["ESCRUTADAS"]/prog_dept["PLANIFICADAS"]*100).round(2), 0.0)
    prog_dept = prog_dept.sort_values("% ESCRUTADAS", ascending=False)

    # Escala normalizada fija 0–100%
    dept_long = prog_dept.melt(
        id_vars=["DEPARTAMENTO","PLANIFICADAS","% ESCRUTADAS"],
        value_vars=["ESCRUTADAS","RESTANTES"],
        var_name="Estado", value_name="Mesas"
    )
    x_norm = alt.X(
        "Mesas:Q", stack="normalize", title="Proporción de mesas",
        scale=alt.Scale(domain=[0,1]),
        axis=alt.Axis(format="%", tickCount=6)
    )
    chart_dept = (
        alt.Chart(dept_long)
        .mark_bar()
        .encode(
            y=alt.Y("DEPARTAMENTO:N", sort="-x", title="", axis=alt.Axis(labelLimit=220)),
            x=x_norm,
            color=alt.Color("Estado:N",
                scale=alt.Scale(domain=["ESCRUTADAS","RESTANTES"], range=["#43A047","#BDBDBD"]),
                legend=alt.Legend(title="Estado")),
            tooltip=["DEPARTAMENTO","PLANIFICADAS",alt.Tooltip("Mesas:Q", title="Cantidad"),
                     "Estado", alt.Tooltip("% ESCRUTADAS:Q", format=".2f")]
        )
        .properties(height=max(260, 22*len(prog_dept)), title="Departamentos — Mesas escrutadas / planificadas")
    )
    st.altair_chart(chart_dept, use_container_width=True)

    # Municipios: planificadas vs escrutadas (scrollable con barras de progreso)
    if not df_muni_plan.empty:
        st.markdown("##### Municipios — Mesas escrutadas / planificadas")

        muni_plan = (
            df_muni_plan[["DEPARTAMENTO","MUNICIPIO","MESA_KEY"]]
            .dropna()
            .drop_duplicates()
            .groupby(["DEPARTAMENTO","MUNICIPIO"], as_index=False)["MESA_KEY"]
            .nunique()
            .rename(columns={"MESA_KEY": "PLANIFICADAS"})
        )
        muni_scru = (
            df_mesas_all.loc[df_mesas_all["TOTAL_MESA"] > 0, ["DEPARTAMENTO","MUNICIPIO","MESA_KEY"]]
            .dropna(subset=["MUNICIPIO"])
            .drop_duplicates()
            .groupby(["DEPARTAMENTO","MUNICIPIO"], as_index=False)["MESA_KEY"]
            .nunique()
            .rename(columns={"MESA_KEY": "ESCRUTADAS"})
        )

        prog_muni = muni_plan.merge(muni_scru, on=["DEPARTAMENTO","MUNICIPIO"], how="left").fillna(0)
        prog_muni["RESTANTES"] = (prog_muni["PLANIFICADAS"] - prog_muni["ESCRUTADAS"]).clip(lower=0).astype(int)
        prog_muni["% ESCRUTADAS"] = np.where(
            prog_muni["PLANIFICADAS"] > 0,
            (prog_muni["ESCRUTADAS"] / prog_muni["PLANIFICADAS"] * 100).round(2),
            0.0,
        )

        dept_opts_muni = ["(Todos)"] + sorted(prog_muni["DEPARTAMENTO"].unique().tolist())
        dept_choice = st.selectbox(
            "Filtrar por departamento", dept_opts_muni, index=0, key="muni_prog_dept"
        )
        view = prog_muni if dept_choice == "(Todos)" else prog_muni[prog_muni["DEPARTAMENTO"] == dept_choice]
        view = view.sort_values(["DEPARTAMENTO", "% ESCRUTADAS", "MUNICIPIO"], ascending=[True, False, True])

        st.dataframe(
            view[["DEPARTAMENTO", "MUNICIPIO", "ESCRUTADAS", "PLANIFICADAS", "% ESCRUTADAS"]],
            use_container_width=True,
            height=520,  # scroll
            column_config={
                "% ESCRUTADAS": st.column_config.ProgressColumn(
                    "% escrutadas", min_value=0, max_value=100, format="%.2f%%"
                )
            },
        )
    else:
        st.info("Para progreso por municipio se requiere 'Mapeo_Mesa_Municipio_raw'.")

# ---------------- Balotaje (universo completo) ----------------
with tab_bal:
    st.markdown("### Chequeo de Balotaje — universo completo provincial (sin filtros)")
    base_total = long.copy()
    votes_total = base_total[["ALIANZA","votos"]].copy()
    resumen, ranking = ballotage_status(votes_total, rule="nacional")
    if resumen is None or ranking.empty:
        st.info("No hay votos válidos suficientes para evaluar balotaje.")
    else:
        st.metric("¿Ganador en 1ra vuelta?", "Sí" if resumen["gana_primera_vuelta"] else "No",
                  help="Regla fija: 45% o 40% + 10 puntos sobre el segundo (votos válidos).")
        df_pad_depto2 = load_padron_depto()
        total_pad = int(df_pad_depto2["PADRON"].sum()) if not df_pad_depto2.empty else 0
        tot_emit = int(df_mesas_all["TOTAL_MESA"].sum())
        st.metric("Participación provincial", f"{(tot_emit/total_pad*100) if total_pad>0 else 0.0:.2f}%")

        ranking = ranking.sort_values("pct", ascending=False).rename(columns={"pct":"% sobre válidos"})
        chart = bar_pct_chart(ranking.rename(columns={"ALIANZA":"Alianza"}), "Alianza", "% sobre válidos", "Participación por alianza (universo completo)")
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(ranking, use_container_width=True)
        st.caption("Se consideran votos válidos por alianza. Excluye blancos, nulos, recurridos e impugnados.")

# ---------------- Diagnóstico ----------------
with tab_diag:
    st.markdown("#### Mesas duplicadas en Respuestas_raw")
    raw_mesa_col = "Mesa" if "Mesa" in df_raw.columns else (find_col(df_raw, r"^\s*mesa\s*$") or "Mesa")
    df_raw["_MESA_KEY"] = normalize_mesa(df_raw.get(raw_mesa_col, pd.Series(index=df_raw.index)))
    ser_mesas = df_raw["_MESA_KEY"].dropna().astype("Int64")
    dupes_count = (
        ser_mesas.value_counts(dropna=True)
                .rename_axis("MESA_KEY")
                .reset_index(name="CANTIDAD_CARGAS")
    )
    dupes_count = dupes_count.loc[dupes_count["CANTIDAD_CARGAS"]>1].sort_values(
        by=["CANTIDAD_CARGAS","MESA_KEY"], ascending=[False, True]
    ).reset_index(drop=True)
    if not dupes_count.empty:
        st.dataframe(dupes_count, use_container_width=True)
        st.info("Revisar estas mesas en Respuestas_raw para resolver duplicidad.")
    else:
        st.success("No se detectaron mesas duplicadas.")

    st.markdown("---")
    st.markdown("#### Discrepancias entre 'Total de votos en la mesa' y suma de categorías")
    party_cols = detect_party_columns(df_raw)
    extra_cols = [c for c in df_raw.columns if re.search(r"votos?\s+(en\s+)?(blancos|nulos|recurridos|impugnados)", c, re.I)]
    tot_col = find_col(df_raw, r"total\s*de\s*votos\s*en\s*la\s*mesa")

    tmp = df_raw.copy()
    tmp["MESA_KEY"] = normalize_mesa(tmp.get(raw_mesa_col, pd.Series(index=tmp.index)))
    for c in party_cols + extra_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0).astype(int)

    agg = tmp.groupby("MESA_KEY", as_index=False)[party_cols + extra_cols].sum()
    agg["TOTAL_COMPUTADO"] = agg[party_cols + extra_cols].sum(axis=1)

    if tot_col:
        tmp[tot_col] = pd.to_numeric(tmp[tot_col].astype(str).str.replace(".","",regex=False).str.replace(",","",regex=False),
                                     errors="coerce").fillna(0).astype(int)
        declared = tmp.groupby("MESA_KEY", as_index=False)[tot_col].sum().rename(columns={tot_col:"TOTAL_DECLARADO"})
        discrep = agg.merge(declared, on="MESA_KEY", how="left")
    else:
        discrep = agg.copy(); discrep["TOTAL_DECLARADO"] = np.nan

    discrep["DIFERENCIA"] = discrep["TOTAL_DECLARADO"] - discrep["TOTAL_COMPUTADO"]
    out = discrep[(discrep["TOTAL_DECLARADO"].isna()) | (discrep["DIFERENCIA"] != 0)].copy()
    out = out.sort_values("MESA_KEY")
    if not out.empty:
        st.dataframe(out[["MESA_KEY","TOTAL_DECLARADO","TOTAL_COMPUTADO","DIFERENCIA"]], use_container_width=True)
        st.info("Si la columna 'Total de votos en la mesa' no existe, 'TOTAL_DECLARADO' aparece vacío.")
    else:
        st.success("Todos los totales declarados coinciden con la suma de categorías.")

# ================== FOOTER GLOBAL ==================
st.markdown("---")
st.caption(f"Mesas cargadas: {mesas_escrutadas} / {TOTAL_MESAS_PROV}")


