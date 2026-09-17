import streamlit as st
import pandas as pd
import re
from collections import Counter

st.set_page_config(
    page_title="Mega Granjita - Verificación",
    page_icon="🐾",
    layout="centered"
)

GOOGLE_SHEET_ID = "1aP-qP6YXz7HcXuy77GXX4xqMKE3-noLP_jvQflqvE-I"
GOOGLE_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv"

ANIMALITOS_DICT = {
    0: "Delfín", 1: "Carnero", 2: "Toro", 3: "Ciempiés", 4: "Alacrán",
    5: "León", 6: "Rana", 7: "Perico", 8: "Ratón", 9: "Águila",
    10: "Tigre", 11: "Gato", 12: "Caballo", 13: "Mono", 14: "Paloma",
    15: "Zorro", 16: "Oso", 17: "Pavo", 18: "Burro", 19: "Chivo",
    20: "Cochino", 21: "Gallo", 22: "Camello", 23: "Cebra", 24: "Iguana",
    25: "Gallina", 26: "Vaca", 27: "Perro", 28: "Zamuro", 29: "Elefante",
    30: "Caimán", 31: "Lapa", 32: "Ardilla", 33: "Pescado", 34: "Venado",
    35: "Jirafa", 36: "Culebra", 100: "Ballena"
}


def fmt_num(n):
    if n == 100:
        return "00"
    if n == 0:
        return "0"
    return f"{n:02d}"


@st.cache_data(ttl=120)
def cargar_historial_google_sheets():
    try:
        df_raw = pd.read_csv(GOOGLE_SHEET_URL, header=None)

        # Buscar todas las filas que tienen "Hora" en la columna A
        filas_encabezado = []
        for fila in range(len(df_raw)):
            val = str(df_raw.iloc[fila, 0]).strip().lower()
            if val == "hora":
                filas_encabezado.append(fila)

        registros = []

        for idx, fila_enc in enumerate(filas_encabezado):
            if idx + 1 < len(filas_encabezado):
                fila_fin = filas_encabezado[idx + 1]
            else:
                fila_fin = len(df_raw)

            fechas_col = {}
            for col in range(1, len(df_raw.columns)):
                val = str(df_raw.iloc[fila_enc, col]).strip()
                if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', val):
                    try:
                        fecha_dt = pd.to_datetime(val, format="%d/%m/%Y", errors="coerce")
                        if pd.notna(fecha_dt):
                            fechas_col[col] = fecha_dt.strftime("%d/%m/%Y")
                    except:
                        pass

            fila_fin_datos = min(fila_fin, fila_enc + 13)

            for col, fecha in fechas_col.items():
                for fila_dato in range(fila_enc + 1, fila_fin_datos):
                    val = str(df_raw.iloc[fila_dato, col]).strip()
                    if not val or val.lower() == "nan" or val.lower() == "hora":
                        continue
                    match = re.search(r'\((\d+)\)', val)
                    if match:
                        num_str = match.group(1)
                        if num_str == "00":
                            num = 100
                        else:
                            num = int(num_str)
                        nombre = ANIMALITOS_DICT.get(num, re.sub(r'\s*\(\d+\)', '', val).strip())
                        registros.append({
                            "fecha": fecha,
                            "numero": num,
                            "nombre": nombre
                        })

        df = pd.DataFrame(registros)
        if not df.empty:
            df["fecha_dt"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
            df = df.sort_values(["fecha_dt"], kind="stable").reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Error al leer Google Sheet: {e}")
        return pd.DataFrame(columns=["fecha", "numero", "nombre"])


def main():
    st.title("🐾 Mega Granjita - Verificación")
    st.caption("Solo para verificar que lee bien los 6 meses de datos")

    if st.button("🔄 Recargar"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Leyendo hoja de cálculo..."):
        df = cargar_historial_google_sheets()

    if df.empty:
        st.error("No se pudieron cargar datos. Verifica que la hoja sea pública.")
        return

    # Estadísticas generales
    st.markdown("### 📊 Resumen de datos cargados")

    total = len(df)
    fechas_unicas = df["fecha"].nunique()
    primera_fecha = df["fecha"].iloc[0]
    ultima_fecha = df["fecha"].iloc[-1]

    st.write(f"**Total de sorteos:** {total}")
    st.write(f"**Total de días:** {fechas_unicas}")
    st.write(f"**Primera fecha:** {primera_fecha}")
    st.write(f"**Última fecha:** {ultima_fecha}")

    st.markdown("---")

    st.markdown("### 🎯 Último resultado")
    ultimo = df.iloc[-1]
    st.markdown(f"## {fmt_num(int(ultimo['numero']))} - {ultimo['nombre']}")
    st.caption(f"Fecha: {ultimo['fecha']}")

    st.markdown("---")

    st.markdown("### 🔁 Animales del último día completo")
    fecha_ultima = df["fecha"].iloc[-1]
    df_ultimo_dia = df[df["fecha"] == fecha_ultima].reset_index(drop=True)
    for i, row in df_ultimo_dia.iterrows():
        num = int(row["numero"])
        st.write(f"{i+1}. **{fmt_num(num)} - {row['nombre']}**")

    st.markdown("---")

    with st.expander("📋 Ver últimos 40 sorteos"):
        st.dataframe(df.tail(40)[["fecha", "numero", "nombre"]], use_container_width=True)

    with st.expander("🔍 Verificar días únicos (detectar duplicados)"):
        conteo_fechas = df.groupby("fecha").size().reset_index(name="sorteos")
        st.dataframe(conteo_fechas, use_container_width=True)


if __name__ == "__main__":
    main()
