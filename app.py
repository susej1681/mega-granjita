import streamlit as st
import pandas as pd
import re
from collections import Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Mega Granjita - IA",
    page_icon="🐾",
    layout="centered"
)

GOOGLE_SHEET_ID = "1aP-qP6YXz7HcXuy77GXX4xqMKE3-noLP_jvQflqvE-I"
GOOGLE_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv"

SORTEOS_POR_DIA = 12
DIAS_VENTANA = 5
VENTANA_SORTEOS = SORTEOS_POR_DIA * DIAS_VENTANA

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
    if n == 100: return "00"
    if n == 0: return "0"
    return f"{n:02d}"


@st.cache_data(ttl=120)
def cargar_historial_google_sheets():
    try:
        df_raw = pd.read_csv(GOOGLE_SHEET_URL, header=None)
        filas_encabezado = []
        for fila in range(len(df_raw)):
            val = str(df_raw.iloc[fila, 0]).strip().lower()
            if val == "hora":
                filas_encabezado.append(fila)
        registros = []
        for idx, fila_enc in enumerate(filas_encabezado):
            fila_fin = filas_encabezado[idx + 1] if idx + 1 < len(filas_encabezado) else len(df_raw)
            fechas_col = {}
            for col in range(1, len(df_raw.columns)):
                val = str(df_raw.iloc[fila_enc, col]).strip()
                if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', val):
                    try:
                        fecha_dt = pd.to_datetime(val, format="%d/%m/%Y", errors="coerce")
                        if pd.notna(fecha_dt):
                            fechas_col[col] = fecha_dt.strftime("%d/%m/%Y")
                    except: pass
            fila_fin_datos = min(fila_fin, fila_enc + 13)
            for col, fecha in fechas_col.items():
                for fila_dato in range(fila_enc + 1, fila_fin_datos):
                    val = str(df_raw.iloc[fila_dato, col]).strip()
                    if not val or val.lower() == "nan" or val.lower() == "hora":
                        continue
                    match = re.search(r'\((\d+)\)', val)
                    if match:
                        num_str = match.group(1)
                        num = 100 if num_str == "00" else int(num_str)
                        nombre = ANIMALITOS_DICT.get(num, re.sub(r'\s*\(\d+\)', '', val).strip())
                        registros.append({"fecha": fecha, "numero": num, "nombre": nombre})
        df = pd.DataFrame(registros)
        if not df.empty:
            df["fecha_dt"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
            df = df.sort_values(["fecha_dt"], kind="stable").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(columns=["fecha", "numero", "nombre"])


def aprender_jales(df, max_atraso=3):
    jales = {n: Counter() for n in ANIMALITOS_DICT.keys()}
    nums = df["numero"].tolist()
    for i in range(len(nums) - 1):
        origen = nums[i]
        for j in range(i + 1, min(i + 1 + max_atraso, len(nums))):
            jales[origen][nums[j]] += 1
    return jales


def detectar_alineaciones(df, salieron_hoy, min_repeticiones=2):
    nums = df["numero"].tolist()
    total = len(nums)
    alineaciones = []
    for i in range(total - 10):
        ventana = nums[i:i + 5]
        unicos = list(dict.fromkeys(ventana))
        for a in range(len(unicos)):
            for b in range(a + 1, len(unicos)):
                alineaciones.append(tuple(sorted([unicos[a], unicos[b]])))
    conteo = Counter(alineaciones)
    parejas_top = [p for p, c in conteo.most_common(10) if c >= min_repeticiones]
    resultado = []
    for par in parejas_top:
        if par[0] in salieron_hoy and par[1] in salieron_hoy:
            continue
        posiciones = []
        for i in range(total - 5):
            ventana = set(nums[i:i + 5])
            if par[0] in ventana and par[1] in ventana:
                posiciones.append(i)
        if len(posiciones) >= 2:
            ultima = posiciones[-1]
            atraso = total - 1 - ultima
            diffs = [posiciones[k + 1] - posiciones[k] for k in range(len(posiciones) - 1)]
            promedio = sum(diffs) / len(diffs) if diffs else 0
            if promedio > 0 and atraso >= promedio * 0.6:
                resultado.append({"par": par, "veces": len(posiciones), "atraso": atraso, "promedio": round(promedio, 1)})
    return resultado


def calcular_ritmo_historico(df):
    nums = df["numero"].tolist()
    ritmos = {}
    for num in ANIMALITOS_DICT.keys():
        posiciones = [i for i, n in enumerate(nums) if n == num]
        if len(posiciones) >= 2:
            diffs = [posiciones[k + 1] - posiciones[k] for k in range(len(posiciones) - 1)]
            ritmos[num] = {"promedio": round(sum(diffs) / len(diffs), 1), "veces_total": len(posiciones), "apariciones": posiciones}
        else:
            ritmos[num] = {"promedio": 999, "veces_total": len(posiciones), "apariciones": posiciones}
    return ritmos


def calcular_alerta_reventon(df, detalles, ritmos, salieron_hoy):
    """Reventón con 4 FILTROS ESTRICTOS:
    1. Ratio entre 0.9 y 1.1
    2. Freq(20) > 0
    3. Al menos 1 jal fuerte
    4. NO enjaulado (<60 sorteos)
    """
    total = len(df)
    alertas = []
    for num in ANIMALITOS_DICT.keys():
        # FILTRO 4: NO salió hoy
        if num in salieron_hoy:
            continue
        if num not in ritmos:
            continue
        ritmo = ritmos[num]["promedio"]
        if ritmo <= 0 or ritmo >= 500:
            continue
        posiciones = ritmos[num]["apariciones"]
        if not posiciones:
            continue
        atraso = total - 1 - posiciones[-1]
        ratio = atraso / ritmo

        # FILTRO 1: Ratio cerrado (0.9 a 1.1)
        if not (0.9 <= ratio <= 1.1):
            continue

        # FILTRO 2: Freq(20) > 0 (salió reciente)
        if detalles[num]["freq_20"] == 0:
            continue

        # FILTRO 3: Al menos 1 jal fuerte
        if detalles[num]["jales_in"] < 1:
            continue

        # FILTRO 4: NO enjaulado
        if detalles[num]["atraso"] >= 60:
            continue

        # Confianza según cercanía al 1.0
        distancia = abs(ratio - 1.0)
        if distancia <= 0.05:
            confianza = "ALTA"
        elif distancia <= 0.10:
            confianza = "MEDIA"
        else:
            confianza = "BAJA"

        sorteos_restantes = max(1, int(ritmo - atraso)) if atraso < ritmo else 1

        alertas.append({
            "num": num, "ratio": round(ratio, 2), "atraso": atraso,
            "ritmo": ritmo, "ventana": sorteos_restantes,
            "confianza": confianza, "jales": detalles[num]["jales_in"],
            "freq_20": detalles[num]["freq_20"]
        })
    alertas.sort(key=lambda x: (abs(x["ratio"] - 1.0), -x["jales"]))
    return alertas[:3]


def calcular_fijo_del_dia(df, scores, detalles, ritmos, salieron_hoy):
    total = len(df)
    candidatos = []
    for num in ANIMALITOS_DICT.keys():
        if num in salieron_hoy: continue
        if num not in ritmos or ritmos[num]["promedio"] >= 500: continue
        posiciones = ritmos[num]["apariciones"]
        if not posiciones: continue
        atraso_actual = total - 1 - posiciones[-1]
        ritmo = ritmos[num]["promedio"]
        if ritmo <= 0: continue
        ratio = atraso_actual / ritmo
        if ratio >= 1.5: prob = 0.85
        elif ratio >= 1.0: prob = 0.70
        elif ratio >= 0.7: prob = 0.55
        elif ratio >= 0.5: prob = 0.40
        else: prob = 0.20
        if ratio > 4: prob *= 0.5
        if detalles[num]["atraso_hoy"] <= 1: prob *= 0.3
        candidatos.append({"num": num, "prob": prob, "atraso": atraso_actual, "ritmo": ritmo, "ratio": round(ratio, 2), "score": scores.get(num, 0)})
    candidatos.sort(key=lambda x: (x["prob"], x["score"]), reverse=True)
    return candidatos[0] if candidatos else None


def entrenar_modelo_ml(df):
    try:
        nums = df["numero"].tolist()
        total = len(nums)
        if total < 500:
            return None, "Datos insuficientes"
        X, y = [], []
        lista_numeros = list(ANIMALITOS_DICT.keys())
        for i in range(100, total - 5, 6):
            v60 = nums[max(0, i - 60):i]
            v20 = nums[max(0, i - 20):i]
            v10 = nums[max(0, i - 10):i]
            for num in lista_numeros:
                f60 = v60.count(num); f20 = v20.count(num); f10 = v10.count(num)
                atraso = 999
                for j in range(i - 1, -1, -1):
                    if nums[j] == num:
                        atraso = i - 1 - j; break
                X.append([f60, f20, f10, min(atraso, 100)])
                y.append(1 if num in nums[i:i + 5] else 0)
        if len(X) < 500:
            return None, f"Muestras insuficientes ({len(X)})"
        modelo = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
        modelo.fit(np.array(X), np.array(y))
        return modelo, f"Entrenado con {len(X)} muestras"
    except Exception as e:
        return None, f"Error: {str(e)}"


def predecir_ml(modelo, df):
    try:
        nums = df["numero"].tolist()
        total = len(nums)
        v60 = nums[-60:]; v20 = nums[-20:]; v10 = nums[-10:]
        resultados = []
        for num in ANIMALITOS_DICT.keys():
            f60 = v60.count(num); f20 = v20.count(num); f10 = v10.count(num)
            atraso = 999
            for j in range(total - 1, -1, -1):
                if nums[j] == num:
                    atraso = total - 1 - j; break
            pred = modelo.predict_proba([[f60, f20, f10, min(atraso, 100)]])[0]
            prob = pred[1] if len(pred) > 1 else pred[0]
            resultados.append({"num": num, "prob_ml": round(float(prob) * 100, 2)})
        resultados.sort(key=lambda x: x["prob_ml"], reverse=True)
        return resultados
    except:
        return []


def calcular_ensemble(df, fijo_candidatos, predicciones_ml, jales_aprendidos, detalles, salieron_hoy):
    ultimos_10 = df.tail(10)["numero"].tolist()
    jales_entrantes = Counter()
    for nr in ultimos_10:
        for siguiente, c in jales_aprendidos.get(nr, Counter()).most_common(3):
            jales_entrantes[siguiente] += c
    max_jal = max(jales_entrantes.values()) if jales_entrantes else 1
    fijo_scores = {}
    for c in fijo_candidatos[:20]:
        fijo_scores[c["num"]] = round(c["prob"] * 100, 2)
    ml_scores = {p["num"]: p["prob_ml"] for p in predicciones_ml}
    jal_scores = {n: round(jales_entrantes.get(n, 0) / max_jal * 100, 2) for n in ANIMALITOS_DICT.keys()}
    ensemble = []
    for num in ANIMALITOS_DICT.keys():
        s_fijo = fijo_scores.get(num, 0)
        s_ml = ml_scores.get(num, 0)
        s_jal = jal_scores.get(num, 0)
        score = s_fijo * 0.40 + s_ml * 0.35 + s_jal * 0.25
        if num in salieron_hoy:
            score *= 0.05
        if detalles[num]["atraso_hoy"] <= 1:
            score *= 0.3
        ensemble.append({"num": num, "score": round(score, 2), "s_fijo": s_fijo, "s_ml": s_ml, "s_jal": s_jal})
    ensemble.sort(key=lambda x: x["score"], reverse=True)
    top_ens = ensemble[0]["num"] if ensemble else None
    if top_ens is None:
        return ensemble, 0, "Sin datos"
    fuentes_apoyo = 0
    if top_ens in [c["num"] for c in fijo_candidatos[:3]]:
        fuentes_apoyo += 1
    if top_ens in [p["num"] for p in predicciones_ml[:3]]:
        fuentes_apoyo += 1
    if top_ens in [n for n, _ in jales_entrantes.most_common(3)]:
        fuentes_apoyo += 1
    if fuentes_apoyo >= 3: consenso = "ALTO"
    elif fuentes_apoyo == 2: consenso = "MEDIO"
    elif fuentes_apoyo == 1: consenso = "BAJO"
    else: consenso = "MUY BAJO"
    return ensemble, fuentes_apoyo, consenso


def backtesting_simple(df):
    if len(df) < 500: return None
    nums = df["numero"].tolist()
    total = len(nums)
    aciertos = 0; pruebas = 0
    for i in range(total - 500, total - 12, 12):
        if i < 100: continue
        nums_hasta = nums[:i]
        candidatos = []
        for num in ANIMALITOS_DICT.keys():
            posiciones = [k for k, n in enumerate(nums_hasta) if n == num]
            if not posiciones: continue
            atraso = len(nums_hasta) - 1 - posiciones[-1]
            if len(posiciones) >= 2:
                diffs = [posiciones[k + 1] - posiciones[k] for k in range(len(posiciones) - 1)]
                ritmo = sum(diffs) / len(diffs)
            else: ritmo = 999
            if ritmo <= 0 or ritmo > 200: continue
            ratio = atraso / ritmo
            if 0.8 <= ratio <= 3:
                candidatos.append((num, ratio))
        if not candidatos: continue
        candidatos.sort(key=lambda x: x[1], reverse=True)
        if candidatos[0][0] in nums[i:i + 12]: aciertos += 1
        pruebas += 1
    if pruebas == 0: return None
    return {"total_pruebas": pruebas, "aciertos": aciertos, "porcentaje": round(aciertos / pruebas * 100, 1)}


def motor_casi_adivino(df):
    if df.empty or len(df) < 60:
        return {}, {}, None, [], [], None, {}, {}, set()
    df_ventana = df.tail(VENTANA_SORTEOS).copy()
    freq_ventana = Counter(df_ventana["numero"].tolist())
    freq_rec20 = Counter(df.tail(20)["numero"].tolist())
    freq_rec30 = Counter(df.tail(30)["numero"].tolist())
    atrasos = {}
    total = len(df)
    for num in ANIMALITOS_DICT.keys():
        idxs = df[df["numero"] == num].index.tolist()
        atrasos[num] = total - 1 - idxs[-1] if idxs else total
    fecha_hoy = df["fecha"].iloc[-1]
    df_hoy = df[df["fecha"] == fecha_hoy]
    total_hoy = len(df_hoy)
    salieron_hoy = set(df_hoy["numero"].tolist())
    atraso_hoy = {}
    for num in ANIMALITOS_DICT.keys():
        idxs_hoy = df_hoy[df_hoy["numero"] == num].index.tolist()
        if idxs_hoy:
            pos = df_hoy.index.get_loc(idxs_hoy[-1])
            atraso_hoy[num] = total_hoy - 1 - pos
        else: atraso_hoy[num] = 999
    jales_aprendidos = aprender_jales(df, max_atraso=3)
    ultimos_10 = df.tail(10)["numero"].tolist()
    jales_entrantes = Counter()
    for nr in ultimos_10:
        for siguiente, c in jales_aprendidos.get(nr, Counter()).most_common(3):
            jales_entrantes[siguiente] += c
    max_fv = max(freq_ventana.values()) if freq_ventana else 1
    max_f20 = max(freq_rec20.values()) if freq_rec20 else 1
    max_atr = max(atrasos.values()) if atrasos else 1
    max_jal = max(jales_entrantes.values()) if jales_entrantes else 1
    fechas_unicas = df["fecha"].unique().tolist()
    ultima_fecha_str = fechas_unicas[-1]
    ultima_fecha_dt = pd.to_datetime(ultima_fecha_str, format="%d/%m/%Y", errors="coerce")
    hoy_real_dt = pd.Timestamp.now().normalize()
    fecha_dia_anterior = None
    if pd.notna(ultima_fecha_dt):
        if ultima_fecha_dt.normalize() == hoy_real_dt:
            if len(fechas_unicas) >= 2:
                fecha_dia_anterior = fechas_unicas[-2]
        else:
            fecha_dia_anterior = ultima_fecha_str
    scores = {}; detalles = {}
    for num in ANIMALITOS_DICT.keys():
        fv = freq_ventana.get(num, 0); f20 = freq_rec20.get(num, 0); f30 = freq_rec30.get(num, 0)
        atr = atrasos.get(num, 0); atr_hoy = atraso_hoy.get(num, 999); jal = jales_entrantes.get(num, 0)
        n_fv = fv / max_fv if max_fv else 0
        n_f20 = f20 / max_f20 if max_f20 else 0
        n_atr = atr / max_atr if max_atr else 0
        n_jal = jal / max_jal if max_jal else 0
        bonus_caliente = 0.08 if f30 >= 3 else (0.04 if f30 == 2 else 0)
        penal_frio = 0
        if atr > 60: penal_frio = -0.35
        elif atr > 45: penal_frio = -0.20
        elif atr > 30: penal_frio = -0.10
        penal_reciente = 0
        if atr_hoy == 0: penal_reciente = -0.60
        elif atr_hoy == 1: penal_reciente = -0.45
        elif atr_hoy == 2: penal_reciente = -0.30
        elif atr_hoy == 3: penal_reciente = -0.20
        elif atr_hoy == 4: penal_reciente = -0.10
        score = n_fv*0.25 + n_f20*0.25 + n_atr*0.20 + n_jal*0.15 + bonus_caliente + penal_frio + penal_reciente
        if fv == 0: score *= 0.4
        scores[num] = round(max(score, 0) * 100, 2)
        detalles[num] = {"freq_ventana": fv, "freq_20": f20, "atraso": atr, "atraso_hoy": atr_hoy, "jales_in": jal, "caliente": bonus_caliente > 0, "penal": penal_frio < 0, "reciente": penal_reciente < 0}
    top_ordenado = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    alineaciones = detectar_alineaciones(df, salieron_hoy, min_repeticiones=2)
    ritmos = calcular_ritmo_historico(df)
    return scores, detalles, top_ordenado, jales_aprendidos, alineaciones, fecha_dia_anterior, atrasos, ritmos, salieron_hoy


def armar_resultados(scores, detalles, top_ordenado, atrasos, salieron_hoy):
    top_validos = [(n, s) for n, s in top_ordenado if n not in salieron_hoy]
    if len(top_validos) < 3:
        extra = [(n, s) for n, s in top_ordenado if n not in [x[0] for x in top_validos]]
        top_validos.extend(extra)
    top3 = []
    for num, sc in top_validos[:3]:
        top3.append({"numero": fmt_num(num), "int_num": num, "nombre": ANIMALITOS_DICT[num], "score": sc, "detalle": detalles[num]})
    individual = top3[0] if top3 else None
    nums_oficiales = set([t["int_num"] for t in top3])
    candidatos = [(n, s) for n, s in top_validos if n not in nums_oficiales and s > 0][:20]
    caliente = None
    for n, s in candidatos:
        if detalles[n]["freq_20"] >= 2: caliente = n; break
    if caliente is None and candidatos: caliente = candidatos[0][0]
    maduro = None
    for n, s in candidatos:
        if n == caliente: continue
        atr = atrasos.get(n, 0)
        if 10 <= atr <= 55: maduro = n; break
    if maduro is None:
        for n, s in candidatos:
            if n != caliente: maduro = n; break
    jale = None
    for n, s in candidatos:
        if n in (caliente, maduro): continue
        if detalles[n]["jales_in"] >= 2: jale = n; break
    if jale is None:
        for n, s in candidatos:
            if n not in (caliente, maduro): jale = n; break
    t_alt = [n for n in [caliente, maduro, jale] if n is not None]
    for n, s in candidatos:
        if len(t_alt) >= 3: break
        if n not in t_alt: t_alt.append(n)
    tripleta_alt = []
    for num in t_alt[:3]:
        tripleta_alt.append({"numero": fmt_num(num), "int_num": num, "nombre": ANIMALITOS_DICT[num], "score": scores[num], "detalle": detalles[num]})
    return individual, top3, tripleta_alt


def main():
    st.title("🐾 Mega Granjita IA")
    st.caption("Ensemble · ML · Fijo · Reventón 4-filtros · Backtesting")

    if st.button("🔄 Recargar datos"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Leyendo hoja de cálculo..."):
        df = cargar_historial_google_sheets()

    if df.empty:
        st.error("No se pudieron cargar datos.")
        return

    scores, detalles, top_ordenado, jales_aprendidos, alineaciones, fecha_dia_anterior, atrasos, ritmos, salieron_hoy = motor_casi_adivino(df)
    if not top_ordenado:
        st.warning("Datos insuficientes.")
        return

    individual, top3, tripleta_alt = armar_resultados(scores, detalles, top_ordenado, atrasos, salieron_hoy)
    ultimo = df.iloc[-1]

    st.caption(f"📅 Día actual: {ultimo['fecha']} · Ya salieron hoy: {len(salieron_hoy)} animalitos")

    # ALERTA DE REVENTÓN (CON 4 FILTROS)
    alertas = calcular_alerta_reventon(df, detalles, ritmos, salieron_hoy)
    if alertas:
        st.markdown("### 🚨 ALERTA DE REVENTÓN (4 filtros)")
        st.caption("Ratio 0.9-1.1 · Freq(20)>0 · Jales≥1 · NO enjaulado")
        for i, al in enumerate(alertas, 1):
            emoji_conf = "🔥" if al["confianza"] == "ALTA" else ("🟡" if al["confianza"] == "MEDIA" else "🟢")
            urgent = " ← MÁS URGENTE" if i == 1 else ""
            st.markdown(f"**{emoji_conf} #{i} - {fmt_num(al['num'])} {ANIMALITOS_DICT[al['num']]}{urgent}**")
            horas = al['ventana']
            ventana_txt = "próxima 1 hora" if horas == 1 else f"próximas {horas} horas"
            st.caption(f"Ratio: {al['ratio']} · Atraso: {al['atraso']} · Ritmo: {al['ritmo']} · Freq(20): {al['freq_20']} · Jales: {al['jales']} · Confianza: {al['confianza']}")
        st.markdown("---")
    else:
        st.info("🚨 Sin alertas de reventón con los 4 filtros. Ningún animal cumple los criterios.")
        st.markdown("---")

    # ENSEMBLE
    fijo_candidatos = []
    for num in ANIMALITOS_DICT.keys():
        if num in salieron_hoy: continue
        if num not in ritmos or ritmos[num]["promedio"] >= 500: continue
        pos = ritmos[num]["apariciones"]
        if not pos: continue
        atraso_actual = len(df) - 1 - pos[-1]
        ritmo = ritmos[num]["promedio"]
        if ritmo <= 0: continue
        ratio = atraso_actual / ritmo
        if ratio >= 1.5: prob = 0.85
        elif ratio >= 1.0: prob = 0.70
        elif ratio >= 0.7: prob = 0.55
        elif ratio >= 0.5: prob = 0.40
        else: prob = 0.20
        if ratio > 4: prob *= 0.5
        fijo_candidatos.append({"num": num, "prob": prob, "atraso": atraso_actual, "ritmo": ritmo, "ratio": round(ratio, 2), "score": scores.get(num, 0)})
    fijo_candidatos.sort(key=lambda x: (x["prob"], x["score"]), reverse=True)
    fijo = fijo_candidatos[0] if fijo_candidatos else None

    with st.spinner("Entrenando IA..."):
        modelo, mensaje = entrenar_modelo_ml(df)
    predicciones_ml = predecir_ml(modelo, df) if modelo else []

    ensemble, fuentes_apoyo, consenso = calcular_ensemble(df, fijo_candidatos, predicciones_ml, jales_aprendidos, detalles, salieron_hoy)

    if ensemble:
        top_ens = ensemble[0]
        st.markdown("### 🏆 RECOMENDACIÓN FINAL (Ensemble)")
        st.markdown(f"# {fmt_num(top_ens['num'])} - {ANIMALITOS_DICT[top_ens['num']]}")
        st.markdown(f"**Score combinado: {top_ens['score']}%**")
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Fijo", f"{top_ens['s_fijo']}%")
        col2.metric("🤖 ML", f"{top_ens['s_ml']}%")
        col3.metric("🔗 Jales", f"{top_ens['s_jal']}%")
        if consenso == "ALTO": st.success(f"✅ CONSENSO ALTO · Las 3 fuentes apoyan")
        elif consenso == "MEDIO": st.info(f"🟡 CONSENSO MEDIO · 2 de 3 fuentes apoyan")
        elif consenso == "BAJO": st.warning(f"⚠️ CONSENSO BAJO · Solo 1 fuente apoya")
        else: st.error(f"❌ SIN CONSENSO · Ninguna fuente respalda.")
        st.markdown("**Top 3 del Ensemble:**")
        for i, item in enumerate(ensemble[:3], 1):
            st.write(f"**#{i} - {fmt_num(item['num'])} {ANIMALITOS_DICT[item['num']]}** — {item['score']}%")
    st.markdown("---")

    # FIJO
    if fijo:
        st.markdown("### 🎯 FIJO DEL DÍA")
        st.markdown(f"## {fmt_num(fijo['num'])} - {ANIMALITOS_DICT[fijo['num']]}")
        st.markdown(f"**Probabilidad: {round(fijo['prob']*100, 1)}%**")
        st.caption(f"Atraso: {fijo['atraso']} · Ritmo: cada {fijo['ritmo']} sorteos · Ratio: {fijo['ratio']}")
    else:
        st.warning("Sin candidatos válidos para el Fijo.")
    st.markdown("---")

    # ML
    st.markdown("### 🤖 Predicción Machine Learning")
    if modelo:
        st.success(f"✅ {mensaje}")
        for i, p in enumerate(predicciones_ml[:5], 1):
            st.write(f"**#{i} - {fmt_num(p['num'])} {ANIMALITOS_DICT[p['num']]}** — {p['prob_ml']}%")
    else:
        st.warning(f"⚠️ {mensaje}")
    st.markdown("---")

    # BACKTESTING
    with st.spinner("Ejecutando backtesting..."):
        bt = backtesting_simple(df)
    if bt:
        st.markdown("### 📊 Backtesting del Fijo")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pruebas", bt["total_pruebas"])
        col2.metric("Aciertos", bt["aciertos"])
        col3.metric("% Acierto", f"{bt['porcentaje']}%")
        st.markdown("---")

    if alineaciones:
        st.markdown("### 🔗 Alineación Caliente")
        for al in alineaciones[:3]:
            a, b = al["par"]
            st.markdown(f"**{fmt_num(a)} {ANIMALITOS_DICT[a]} + {fmt_num(b)} {ANIMALITOS_DICT[b]}**")
            st.caption(f"{al['veces']} veces · Promedio cada {al['promedio']} · Atraso: {al['atraso']}")
        st.markdown("---")

    st.markdown("### 🎯 Último resultado")
    st.markdown(f"## {fmt_num(int(ultimo['numero']))} - {ultimo['nombre']}")
    st.caption(f"Fecha: {ultimo['fecha']}")
    st.markdown("---")

    if fecha_dia_anterior:
        st.markdown(f"### 🔁 Animales del {fecha_dia_anterior}")
        df_dia = df[df["fecha"] == fecha_dia_anterior].reset_index(drop=True)
        for i, row in df_dia.iterrows():
            num = int(row["numero"])
            d = detalles.get(num, {})
            st.write(f"{i+1}. **{fmt_num(num)} - {row['nombre']}** (atraso: {d.get('atraso', '?')})")
        st.markdown("---")

    if individual:
        st.markdown("### 🎯 Animal Individual")
        d = individual["detalle"]
        st.markdown(f"## {individual['numero']} - {individual['nombre']}")
        st.markdown(f"**Score: {individual['score']}%**")
        st.caption(f"Freq(60): {d['freq_ventana']} | Freq(20): {d['freq_20']} | Atraso: {d['atraso']} | Jales: {d['jales_in']}")
        st.markdown("---")

    st.markdown("### 🏆 Top 3 (sin repetidos de hoy)")
    for i, item in enumerate(top3, 1):
        d = item["detalle"]
        st.markdown(f"**#{i} - {item['numero']} {item['nombre']}** — {item['score']}%")
        st.caption(f"Freq(60): {d['freq_ventana']} | Freq(20): {d['freq_20']} | Atraso: {d['atraso']} | Jales: {d['jales_in']}")
    st.markdown("---")

    st.markdown("### 🎯 Tripleta OFICIAL (11 sorteos)")
    if len(top3) >= 3:
        st.success(" - ".join([f"{t['numero']} {t['nombre']}" for t in top3]))
    st.markdown("---")

    st.markdown("### ⚡ Tripleta ALTERNATIVA (11 sorteos)")
    if len(tripleta_alt) >= 3:
        st.info(" - ".join([f"{t['numero']} {t['nombre']}" for t in tripleta_alt]))
    st.markdown("---")

    st.markdown("### 🔗 Jales Aprendidos")
    ultimo_num = int(ultimo["numero"])
    jales_ult = jales_aprendidos.get(ultimo_num, Counter())
    if jales_ult:
        for jale, c in jales_ult.most_common(3):
            st.write(f"- Después de **{fmt_num(ultimo_num)} {ultimo['nombre']}** → **{fmt_num(jale)} {ANIMALITOS_DICT[jale]}** ({c} veces)")

    with st.expander("📋 Ver últimos 30 sorteos"):
        st.dataframe(df.tail(30)[["fecha", "numero", "nombre"]], use_container_width=True)


if __name__ == "__main__":
    main()
