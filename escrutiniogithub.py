# -*- coding: utf-8 -*-
"""
Escrutinio – Dashboard Streamlit
Lee un Google Sheet con hojas:
  - Respuestas_raw
  - Mapeo_Escuelas_raw
  - Mapeo_Alianzas_raw

Muestra resultados por Partido y por Alianza,
con filtros por Departamento, rango de Mesa y Solo Testigo.
"""

from __future__ import annotations

import re
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


# ================== GOOGLE SHEETS ==================
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

    # Normalizaciones de saltos/guiones
    s = s.replace("\\r\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\n", "\n")
    s = s.replace("—", "-").replace("–", "-")  # guiones unicode a ASCII

    start = "-----BEGIN PRIVATE KEY-----"
    end = "-----END PRIVATE KEY-----"

    if start not in s or end not in s:
        # A veces pegan el JSON entero o falta un salto; intentamos igual limpiar
        # Si no están los marcadores, devolvemos tal cual (google-auth fallará con mensaje claro)
        return s

    # Extraer solo el cuerpo base64 entre marcadores
    before, rest = s.split(start, 1)
    body, after = rest.split(end, 1)

    # Limpiar cada línea (strip) y descartar líneas vacías
    lines = [ln.strip() for ln in body.strip().split("\n") if ln.strip()]

    # Dejar solo caracteres base64 válidos
    b64 = re.sub(r"[^A-Za-z0-9+/=]", "", "".join(lines))

    # Re-envolver a 64 caracteres por línea (estándar PEM)
    wrapped = "\n".join([b64[i:i + 64] for i in range(0, len(b64), 64)])

    # Armar PEM canónico con \n finales
    normalized = f"{start}\n{wrapped}\n{end}\n"
    return normalized


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

    # Normalizar la clave privada (corrige \\n, CRLF, espacios, etc.)
    pk = info.get("private_key", "")
    info["private_key"] = _normalize_private_key(pk)

    # Validación rápida
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

# -------- KPIs --------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Votos (filtro)", f"{int(flt['votos'].sum()):,}".replace(",", "."))
col2.metric("Mesas", flt["MESA_KEY"].nunique())
col3.metric("Alianzas", flt["ALIANZA"].nunique())
col4.metric("Partidos", flt["PARTIDO"].nunique())

st.divider()

# -------- Resultados por Alianza --------
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

# -------- Resultados por Partido --------
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

# -------- Detalle por Mesa (todas) --------
st.subheader("Detalle por Mesa (todas las alianzas)")
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

# -------- Solo Mesas Testigo --------
st.subheader("Solo Mesas Testigo – Detalle por Alianza")
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

# -------- Descargas --------
st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "⬇️ CSV – por Alianza (filtro)",
        data=ali_df.to_csv(index=False).encode("utf-8"),
        file_name="resultado_por_alianza.csv",
        mime="text/csv",
        disabled=ali_df.empty,
    )
with c2:
    st.download_button(
        "⬇️ CSV – por Partido (filtro)",
        data=part_df.to_csv(index=False).encode("utf-8"),
        file_name="resultado_por_partido.csv",
        mime="text/csv",
        disabled=part_df.empty,
    )
with c3:
    st.download_button(
        "⬇️ CSV – detalle por Mesa (filtro)",
        data=pivot_all.to_csv(index=False).encode("utf-8"),
        file_name="detalle_por_mesa.csv",
        mime="text/csv",
        disabled=pivot_all.empty,
    )

# -------- Diagnóstico (opcional) --------
with st.expander("🔎 Diagnóstico"):
    st.write("Respuestas_raw columnas:", list(df_raw.columns)[:10], "… total:", len(df_raw.columns))
    st.write("Mapeo_Escuelas_raw columnas:", list(df_esc.columns))
    st.write("Mapeo_Alianzas_raw columnas:", list(df_ali.columns))
    st.write("Registros tidy:", long.shape)
    st.write("Mesas distintas:", int(long["MESA_KEY"].nunique()))
    st.write("Alianzas:", sorted([a for a in long["ALIANZA"].dropna().unique()]))

