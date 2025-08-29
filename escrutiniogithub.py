# -*- coding: utf-8 -*-
"""
Escrutinio – Dashboard Streamlit

Lee un Google Sheet con hojas:
  - Respuestas_raw
  - Mapeo_Escuelas_raw
  - Mapeo_Alianzas_raw
  - (opcional) Padron_Departamento_raw  → DEPARTAMENTO | PADRON

Muestra resultados por Partido, por Alianza, detalle por Mesa,
y Participación (% sobre padrón), con filtros por Departamento,
rango de Mesa y Solo Testigo.
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
}
AUTOREFRESH_SEC = 60

st.set_page_config(page_title="Escrutinio – Dashboard", layout="wide")


# ================== PADRÓN FALLBACK (por si falta la hoja) ==================
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


# ================== GOOGLE SHEETS (credenciales robustas) ==================
def _normalize_private_key(pk: str) -> str:
    """
    Normaliza la clave PEM para evitar errores binascii/base64:
    - Convierte '\\n' -> '\n'
    - Normaliza CRLF/CR -> LF
    - Reemplaza guiones unicode por '-'
    - Remueve espacios al inicio/fin de cada línea
    - Extrae el bloque entre BEGIN/END y reenvuelve base64 a 64 chars/linea
    """
    if not isinstance(pk, str):
        return pk

    s = pk.strip()
    s = s.replace("\\r\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\n", "\n")
    s = s.replace("—", "-").replace("–", "-")

    start = "-----BEGIN PRIVATE KEY-----"
    end = "-----END PRIVATE KEY-----"

    if start not in s or end not in s:
        return s

    _, rest = s.split(start, 1)
    body, _ = rest.split(end, 1)
    lines = [ln.strip() for ln in body.strip().split("\n") if ln.strip()]
    b64 = re.sub(r"[^A-Za-z0-9+/=]", "", "".join(lines))
    wrapped = "\n".join([b64[i:i + 64] for i in range(0, len(b64), 64)])
    return f"{start}\n{wrapped}\n{end}\n"


@st.cache_resource
def _gspread_client():
    """
    Crea cliente gspread usando Service Account desde st.secrets.
    Tolera private_key con '\n' mal escapados y formatos PEM irregulares.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as e:
        st.error(
            "Faltan dependencias para Google Sheets.\n\n"
            "Instalá con:  python -m pip install -r requirements.txt\n\n"
            f"Detalle: {e}"
        )
        st.stop()

    if "gcp_service_account" not in st.secrets:
        st.error(
            "Falta el bloque [gcp_service_account] en Secrets.\n"
            "Pegá el JSON del Service Account en Settings → Secrets."
        )
        st.stop()

    info = dict(st.secrets["gcp_service_account"])
    pk = info.get("private_key", "")
    info["private_key"] = _normalize_private_key(pk)

    if not (
        isinstance(info.get("private_key"), str)
        and "BEGIN PRIVATE KEY" in info["private_key"]
        and "END PRIVATE KEY" in info["private_key"]
    ):
        st.error(
            "El `private_key` del Service Account no tiene el formato esperado.\n\n"
            "Opciones válidas en Secrets (TOML):\n"
            "  • Multilínea con triple comillas (recomendado):\n"
            '    private_key = """-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"""\n'
            "  • Una sola línea con \\n escapados:\n"
            '    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"\n'
        )
        st.stop()

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    try:
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(
            "No se pudo inicializar las credenciales del Service Account.\n"
            "Revisá que el `private_key` esté completo y sin caracteres extraños.\n\n"
            f"Detalle: {e}"
        )
        st.stop()


def _sheet_to_df(gc, sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    """Lee una worksheet por nombre y devuelve DataFrame (primera fila como encabezados)."""
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)
    rows = ws.get_all_values()
    return pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame()


@st.cache_data(ttl=AUTOREFRESH_SEC)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gc = _gspread_client()
    try:
        df_raw = _sheet_to_df(gc, SHEET_ID, SHEET_NAMES["raw"])
        df_esc = _sheet_to_df(gc, SHEET_ID, SHEET_NAMES["escuelas"])
        df_ali = _sheet_to_df(gc, SHEET_ID, SHEET_NAMES["alianzas"])
    except Exception as e:
        st.error(
            "No se pudo abrir el Google Sheet. Verificá:\n"
            "  • Que compartiste el archivo con el client_email del Service Account (Viewer/Editor)\n"
            "  • Que los nombres de hojas coinciden exactamente\n"
            "  • Que las APIs de Sheets y Drive están habilitadas en tu proyecto\n\n"
            f"Detalle: {e}"
        )
        st.stop()
    return df_raw, df_esc, df_ali


@st.cache_data(ttl=AUTOREFRESH_SEC)
def load_padron() -> pd.DataFrame:
    """Devuelve DEPARTAMENTO | PADRON desde 'Padron_Departamento_raw' si existe, o fallback."""
    gc = _gspread_client()
    try:
        sh = gc.open_by_key(SHEET_ID)
        titles = [ws.title for ws in sh.worksheets()]
        if "Padron_Departamento_raw" in titles:
            ws = sh.worksheet("Padron_Departamento_raw")
            rows = ws.get_all_values()
            dfp = pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame()
            if not dfp.empty:
                dfp["DEPARTAMENTO"] = dfp["DEPARTAMENTO"].astype(str).str.strip()
                dfp["PADRON"] = pd.to_numeric(
                    dfp["PADRON"].astype(str).str.replace(".", "", regex=False).str.replace(",", "", regex=False),
                    errors="coerce"
                ).fillna(0).astype(int)
                return dfp
    except Exception:
        pass
    return pd.DataFrame(PADRON_FALLBACK)


# ================== HELPERS ==================
def normalize_mesa(s: pd.Series) -> pd.Series:
    """Convierte 'Mesa 0123' → 123 (Int64)."""
    return (
        s.astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .replace({"": np.nan})
        .astype("Int64")
    )


def bool_from_any(s: pd.Series) -> pd.Series:
    """TRUE/1/SI/SÍ/YES/VERDADERO → True."""
    up = s.astype(str).str.upper().str.strip()
    return up.isin(["TRUE", "1", "SI", "SÍ", "YES", "VERDADERO"])


def find_col(df: pd.DataFrame, regex: str) -> str | None:
    for c in df.columns:
        if re.search(regex, str(c), re.IGNORECASE):
            return c
    return None


def detect_party_columns(df_raw: pd.DataFrame) -> List[str]:
    """Devuelve encabezados que empiezan con número (columnas de partidos)."""
    return [c for c in df_raw.columns if re.match(r"^\s*\d+", str(c))]


def tidy_votes(df_raw: pd.DataFrame, party_cols: List[str]) -> pd.DataFrame:
    """Wide → long: una fila por (MESA_KEY, partido) con 'votos'."""
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

    long["PARTIDO_NOMBRE_HEADER"] = (
        long["partido_header"].astype(str).str.replace(r"^\s*\d+\s*-\s*", "", regex=True).str.strip()
    )
    return long


def normalize_name(s: pd.Series | str) -> pd.Series | str:
    """Normaliza nombres: sin acentos, minúsculas, 1 espacio."""
    def _norm_one(x: str) -> str:
        x = str(x).strip()
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
        x = re.sub(r"\s+", " ", x).strip().lower()
        return x
    if isinstance(s, pd.Series):
        return s.astype(str).map(_norm_one)
    return _norm_one(s)


def mesa_totales_por_depto(df_raw: pd.DataFrame, df_esc: pd.DataFrame) -> pd.DataFrame:
    """Devuelve MESA_KEY | DEPARTAMENTO | TESTIGO_BOOL | TOTAL_MESA (int)"""
    mesa_col = "Mesa" if "Mesa" in df_raw.columns else (find_col(df_raw, r"^\s*mesa\s*$") or "Mesa")
    tot_col = find_col(df_raw, r"total\s*de\s*votos\s*en\s*la\s*mesa")

    if not tot_col:
        # Fallback: sumar partidos (y blancos/nulos si estuvieran)
        party_cols = detect_party_columns(df_raw)
        df_tmp = df_raw.copy()
        df_tmp["MESA_KEY"] = normalize_mesa(df_tmp.get(mesa_col, pd.Series(index=df_tmp.index)))
        party_like = party_cols + [
            c for c in df_raw.columns
            if re.search(r"votos?\s+(en\s+)?(blancos|nulos|recurridos|impugnados)", c, re.I)
        ]
        for c in party_like:
            df_tmp[c] = pd.to_numeric(df_tmp[c], errors="coerce").fillna(0).astype(int)
        df_tot = df_tmp[["MESA_KEY"] + party_like].copy()
        df_tot["TOTAL_MESA"] = df_tot[party_like].sum(axis=1)
    else:
        df_tot = df_raw[[mesa_col, tot_col]].copy()
        df_tot["MESA_KEY"] = normalize_mesa(df_tot[mesa_col])
        df_tot["TOTAL_MESA"] = pd.to_numeric(
            df_tot[tot_col].astype(str).str.replace(".", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce"
        ).fillna(0).astype(int)

    # Join a depto y testigo
    mesa_esc_col = "MESA" if "MESA" in df_esc.columns else (find_col(df_esc, r"\bmesa\b") or "MESA")
    test_col = "TESTIGO" if "TESTIGO" in df_esc.columns else (find_col(df_esc, r"testig") or "TESTIGO")

    df_e = df_esc.copy()
    df_e["MESA_KEY"] = normalize_mesa(df_e[mesa_esc_col])
    df_e["TESTIGO_BOOL"] = bool_from_any(df_e.get(test_col, False))

    out = df_tot.merge(
        df_e[["MESA_KEY", "DEPARTAMENTO", "TESTIGO_BOOL"]].drop_duplicates("MESA_KEY"),
        on="MESA_KEY",
        how="left",
    )
    out["DEPARTAMENTO"] = out["DEPARTAMENTO"].fillna("(Sin depto)")
    return out[["MESA_KEY", "DEPARTAMENTO", "TESTIGO_BOOL", "TOTAL_MESA"]]


def prep_data():
    df_raw, df_esc, df_ali = load_data()

    if df_raw.empty:
        st.warning("Respuestas_raw está vacío.")
    if df_esc.empty:
        st.warning("Mapeo_Escuelas_raw está vacío.")
    if df_ali.empty:
        st.warning("Mapeo_Alianzas_raw está vacío.")

    # ---- Escuelas: MESA_KEY, TESTIGO_BOOL, DEPARTAMENTO, ESTABLECIMIENTO
    mesa_esc_col = "MESA" if "MESA" in df_esc.columns else find_col(df_esc, r"\bmesa\b")
    if mesa_esc_col:
        df_esc["MESA_KEY"] = normalize_mesa(df_esc[mesa_esc_col])
    else:
        df_esc["MESA_KEY"] = pd.Series(dtype="Int64")

    test_col = "TESTIGO" if "TESTIGO" in df_esc.columns else find_col(df_esc, r"testig")
    if test_col:
        df_esc["TESTIGO_BOOL"] = bool_from_any(df_esc[test_col])
    else:
        df_esc["TESTIGO_BOOL"] = False

    # ---- Partidos desde headers
    parties = detect_party_columns(df_raw)
    long = tidy_votes(df_raw, parties)

    # ---- Alianzas: numero → Alianza
    num_col = "numero" if "numero" in df_ali.columns else find_col(df_ali, r"^\s*numero\s*$")
    ali_col = "Alianza" if "Alianza" in df_ali.columns else find_col(df_ali, r"^\s*alianza\s*$")
    party_name_col = find_col(df_ali, r"Partidos?\s+pol")

    if not (num_col and ali_col):
        st.error("En Mapeo_Alianzas_raw deben existir columnas: 'numero' y 'Alianza'.")
        st.stop()

    df_ali["_numero"] = pd.to_numeric(df_ali[num_col], errors="coerce")
    ali_map = df_ali[["_numero", ali_col]].rename(columns={"_numero": "numero_partido", ali_col: "ALIANZA"})
    long = long.merge(ali_map, on="numero_partido", how="left")

    # Excluir "No Usar"
    long["ALIANZA"] = long["ALIANZA"].fillna("")
    no_usar = long["ALIANZA"].str.strip().str.match(r"(?i)^\s*no\s*usar\s*$")
    long.loc[no_usar, "ALIANZA"] = np.nan
    long["ALIANZA"] = long["ALIANZA"].where(long["ALIANZA"].notna(), "(Sin alianza)")

    # Nombre de partido (desde mapeo si está, o desde header)
    if party_name_col:
        name_map = df_ali[["_numero", party_name_col]].rename(
            columns={"_numero": "numero_partido", party_name_col: "PARTIDO"}
        )
        long = long.merge(name_map, on="numero_partido", how="left")
    else:
        long["PARTIDO"] = long["PARTIDO_NOMBRE_HEADER"]

    # ---- Join con Escuelas
    keep_esc = [c for c in ["DEPARTAMENTO", "ESTABLECIMIENTO", "TESTIGO_BOOL"] if c in df_esc.columns]
    long = long.merge(df_esc[["MESA_KEY"] + keep_esc].drop_duplicates("MESA_KEY"), on="MESA_KEY", how="left")
    long["DEPARTAMENTO"] = long["DEPARTAMENTO"].where(long["DEPARTAMENTO"].notna(), "(Sin depto)")

    return df_raw, df_esc, df_ali, long


# ================== UI ==================
st.title("📊 Escrutinio – Resultados por Partido y Alianza")
st.caption(f"Actualiza cada {AUTOREFRESH_SEC}s (cache TTL)")

# Auto-refresh (si está instalado); si no, botón manual
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=AUTOREFRESH_SEC * 1000, key="data_refresh")
except Exception:
    with st.sidebar:
        if st.button("Actualizar datos ahora"):
            st.cache_data.clear()
            st.rerun()

df_raw, df_esc, df_ali, long = prep_data()

# -------- Filtros --------
with st.sidebar:
    st.header("Filtros")
    depts = ["(Todos)"] + sorted([d for d in long["DEPARTAMENTO"].dropna().unique().tolist()])
    dept_sel = st.selectbox("Departamento", depts, index=0)

    only_testigo = st.toggle("Solo mesas testigo", value=False)

    mesas_disponibles = long["MESA_KEY"].dropna().astype(int).sort_values().unique().tolist()
    if mesas_disponibles:
        min_mesa, max_mesa = int(mesas_disponibles[0]), int(mesas_disponibles[-1])
        rango = st.slider("Rango de mesa", min_value=min_mesa, max_value=max_mesa, value=(min_mesa, max_mesa))
    else:
        rango = (0, 10**9)

    ali_list = sorted([a for a in long["ALIANZA"].dropna().unique().tolist()])
    ali_sel = st.multiselect("Alianzas (opcional)", ali_list, default=ali_list)

mask = pd.Series(True, index=long.index)
if dept_sel != "(Todos)":
    mask &= (long["DEPARTAMENTO"] == dept_sel)
if only_testigo:
    mask &= (long["TESTIGO_BOOL"] == True)
mask &= long["MESA_KEY"].between(rango[0], rango[1])
if ali_sel:
    mask &= long["ALIANZA"].isin(ali_sel)

flt = long.loc[mask].copy()

# -------- Preparación de dataframes para tabs --------
# KPIs
total_votos = int(flt['votos'].sum())
total_mesas = flt['MESA_KEY'].nunique()
total_alis  = flt['ALIANZA'].nunique()
total_parts = flt['PARTIDO'].nunique()

# Alianzas
ali_df = (
    flt.dropna(subset=["ALIANZA"])
       .groupby("ALIANZA", as_index=False)["votos"].sum()
       .sort_values("votos", ascending=False)
)
if not ali_df.empty:
    ali_df["% sobre válidos"] = (ali_df["votos"] / ali_df["votos"].sum() * 100).round(2)

# Partidos
part_df = (
    flt.groupby(["numero_partido", "PARTIDO"], as_index=False)["votos"].sum()
       .sort_values("votos", ascending=False)
)
if not part_df.empty:
    part_df["Etiqueta"] = part_df["numero_partido"].astype(int).astype(str) + " - " + part_df["PARTIDO"].fillna("")
    part_df["% sobre válidos"] = (part_df["votos"] / part_df["votos"].sum() * 100).round(2)

# Mesas (todas)
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

# Solo Testigo
testigo_flt = flt.loc[flt["TESTIGO_BOOL"] == True].copy()
pivot_testigo = pd.DataFrame()
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

# Participación sobre padrón
df_pad = load_padron()
df_mesas = mesa_totales_por_depto(df_raw, df_esc)

mask_turnout = pd.Series(True, index=df_mesas.index)
if dept_sel != "(Todos)":
    mask_turnout &= df_mesas["DEPARTAMENTO"].eq(dept_sel)
if only_testigo:
    mask_turnout &= df_mesas["TESTIGO_BOOL"].fillna(False)
mask_turnout &= df_mesas["MESA_KEY"].between(rango[0], rango[1])
df_mesas_f = df_mesas.loc[mask_turnout].copy()

turnout = pd.DataFrame(columns=["DEPARTAMENTO","VOTOS_EMITIDOS","PADRON","% SOBRE PADRON"])
turnout_total = turnout.copy()
if not df_mesas_f.empty:
    g = (df_mesas_f.groupby("DEPARTAMENTO", as_index=False)["TOTAL_MESA"]
                  .sum()
                  .rename(columns={"TOTAL_MESA": "VOTOS_EMITIDOS"}))
    g["_key"] = normalize_name(g["DEPARTAMENTO"])
    df_pad = df_pad.copy()
    df_pad["_key"] = normalize_name(df_pad["DEPARTAMENTO"])
    turnout = g.merge(df_pad[["_key", "PADRON"]], on="_key", how="left").drop(columns=["_key"])
    turnout["PADRON"] = turnout["PADRON"].fillna(0).astype(int)
    turnout["% SOBRE PADRON"] = np.where(
        turnout["PADRON"] > 0,
        (turnout["VOTOS_EMITIDOS"] / turnout["PADRON"] * 100).round(2),
        0.0,
    )
    turnout = turnout.sort_values("% SOBRE PADRON", ascending=False)

    total_row = pd.DataFrame({
        "DEPARTAMENTO": ["TOTAL PROVINCIAL"],
        "VOTOS_EMITIDOS": [int(df_mesas_f["TOTAL_MESA"].sum())],
        "PADRON": [int(df_pad["PADRON"].sum())],
        "% SOBRE PADRON": [round(df_mesas_f["TOTAL_MESA"].sum() / max(1, df_pad["PADRON"].sum()) * 100, 2)]
    })
    turnout_total = pd.concat([turnout, total_row], ignore_index=True)

# ================== TABS ==================
(
    tab_resumen,
    tab_alianzas,
    tab_partidos,
    tab_mesas,
    tab_testigo,
    tab_participacion,
    tab_descargas,
    tab_diag,
) = st.tabs([
    "🏁 Resumen",
    "🧩 Alianzas",
    "🎟️ Partidos",
    "🗃️ Mesas",
    "⭐ Testigo",
    "📈 Participación",
    "⬇️ Descargas",
    "🔧 Diagnóstico",
])

# ---- Tab Resumen ----
with tab_resumen:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Votos (filtro)", f"{total_votos:,}".replace(",", "."))
    c2.metric("Mesas", total_mesas)
    c3.metric("Alianzas", total_alis)
    c4.metric("Partidos", total_parts)

    st.divider()
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Top Alianzas**")
        if not ali_df.empty:
            st.bar_chart(ali_df.set_index("ALIANZA")["votos"].head(10))
        else:
            st.info("Sin datos de alianzas para el filtro.")
    with cc2:
        st.markdown("**Top Partidos**")
        if not part_df.empty:
            st.bar_chart(part_df.set_index("Etiqueta")["votos"].head(10))
        else:
            st.info("Sin datos de partidos para el filtro.")

# ---- Tab Alianzas ----
with tab_alianzas:
    st.markdown("### Votos por Alianza")
    if not ali_df.empty:
        c1, c2 = st.columns([2, 1.2])
        with c1:
            st.bar_chart(ali_df.set_index("ALIANZA")["votos"])
        with c2:
            st.dataframe(ali_df, width='stretch')
    else:
        st.info("No hay votos asociados a alianzas con el filtro actual.")

# ---- Tab Partidos ----
with tab_partidos:
    st.markdown("### Votos por Partido")
    if not part_df.empty:
        c1, c2 = st.columns([2, 1.2])
        with c1:
            st.bar_chart(part_df.set_index("Etiqueta")["votos"])
        with c2:
            st.dataframe(part_df[["numero_partido", "PARTIDO", "votos", "% sobre válidos"]], width='stretch')
    else:
        st.info("No hay votos de partidos con el filtro actual.")

# ---- Tab Mesas ----
with tab_mesas:
    st.markdown("### Detalle por Mesa (todas las alianzas)")
    st.dataframe(pivot_all, width='stretch')

# ---- Tab Testigo ----
with tab_testigo:
    st.markdown("### Solo Mesas Testigo – Detalle por Alianza")
    if not pivot_testigo.empty:
        st.dataframe(pivot_testigo, width='stretch')
    else:
        st.info("No hay mesas testigo con datos bajo los filtros actuales.")

# ---- Tab Participación ----
with tab_participacion:
    st.markdown("### Participación – % sobre padrón")
    if not turnout.empty:
        c1, c2 = st.columns([2, 1.2])
        with c1:
            st.bar_chart(turnout.set_index("DEPARTAMENTO")["% SOBRE PADRON"])
        with c2:
            st.dataframe(turnout_total, width='stretch')
    else:
        st.info("Aún no hay votos emitidos para los filtros aplicados.")

# ---- Tab Descargas ----
with tab_descargas:
    st.markdown("### Exportar CSVs (según filtros)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Por Alianza",
            data=(ali_df.to_csv(index=False).encode("utf-8") if not ali_df.empty else b""),
            file_name="resultado_por_alianza.csv",
            mime="text/csv",
            disabled=ali_df.empty,
        )
    with c2:
        st.download_button(
            "Por Partido",
            data=(part_df.to_csv(index=False).encode("utf-8") if not part_df.empty else b""),
            file_name="resultado_por_partido.csv",
            mime="text/csv",
            disabled=part_df.empty,
        )
    with c3:
        st.download_button(
            "Detalle por Mesa",
            data=(pivot_all.to_csv(index=False).encode("utf-8") if not pivot_all.empty else b""),
            file_name="detalle_por_mesa.csv",
            mime="text/csv",
            disabled=pivot_all.empty,
        )
    with c4:
        st.download_button(
            "Participación",
            data=(turnout_total.to_csv(index=False).encode("utf-8") if not turnout_total.empty else b""),
            file_name="participacion_por_padron.csv",
            mime="text/csv",
            disabled=turnout_total.empty,
        )

# ---- Tab Diagnóstico ----
with tab_diag:
    st.write("Respuestas_raw columnas:", list(df_raw.columns)[:10], "… total:", len(df_raw.columns))
    st.write("Mapeo_Escuelas_raw columnas:", list(df_esc.columns))
    st.write("Mapeo_Alianzas_raw columnas:", list(df_ali.columns))
    st.write("Registros tidy:", long.shape)
    st.write("Mesas distintas:", int(long["MESA_KEY"].nunique()))
    st.write("Alianzas:", sorted([a for a in long["ALIANZA"].dropna().unique()]))
    st.write("Padrón total:", int(df_pad["PADRON"].sum()))
