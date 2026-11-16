"""CoopScore: scoring de morosidad para cooperativas de ahorro y crédito."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import modelo as m

st.set_page_config(page_title="CoopScore · Riesgo de mora", page_icon=":material/account_balance:",
                   layout="wide")

# ---------------------------------------------------------------- Identidad visual
NAVY, NAVY_2, ORANGE, SAND, INK, MUTED, LINE = (
    "#12304A", "#3B7EA1", "#C2561A", "#F6F4EE", "#1F2A37", "#5A6878", "#E2DED3"
)
RIESGO_ALTO, RIESGO_MEDIO, RIESGO_BAJO = "#C2561A", "#D99A2B", "#2F6F8F"

pio.templates["coop"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, Arial, sans-serif", size=14, color=INK),
        title=dict(font=dict(family="Space Grotesk, Arial, sans-serif", size=18, color=NAVY), x=0, xanchor="left"),
        colorway=[NAVY, ORANGE, NAVY_2, "#8A94A0", "#D99A2B"],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=LINE, zerolinecolor="#CFC9BA", linecolor="#CFC9BA"),
        yaxis=dict(gridcolor=LINE, zerolinecolor="#CFC9BA", linecolor="#CFC9BA"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=dict(text="")),
        separators=",.",
        margin=dict(l=10, r=10, t=56, b=10),
        hoverlabel=dict(font=dict(family="Inter, Arial, sans-serif")),
    )
)
pio.templates.default = "simple_white+coop"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input, textarea {{ font-family: 'Inter', Arial, sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', Arial, sans-serif !important; color: {NAVY} !important; letter-spacing: -0.3px; }}
#MainMenu, footer, [data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}

.hero {{ background: {NAVY}; border-radius: 18px; padding: 30px 36px; margin-bottom: 18px;
        display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; flex-wrap: wrap; }}
.hero .brand {{ font-family: 'Space Grotesk', Arial, sans-serif; font-size: 42px; font-weight: 700; color: {SAND}; line-height: 1; }}
.hero .brand span {{ color: #F2A65A; }}
.hero .tag {{ color: #C9D6E1; font-size: 17px; margin-top: 10px; max-width: 640px; line-height: 1.45; }}
.chips {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.chip {{ border: 1px solid #3A5874; color: #DCE6EE; border-radius: 999px; padding: 6px 14px; font-size: 13px; white-space: nowrap; }}

.eyebrow {{ font-size: 12px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: {ORANGE}; margin: 6px 0 2px; }}
.section-title {{ font-family: 'Space Grotesk', Arial, sans-serif; font-size: 26px; font-weight: 700; color: {NAVY}; margin: 0 0 4px; }}
.section-sub {{ color: {MUTED}; font-size: 15px; margin-bottom: 14px; max-width: 900px; line-height: 1.5; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; padding: 10px 16px; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {NAVY} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ORANGE}; }}

[data-testid="stForm"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 16px; padding: 22px 24px; }}
.group {{ font-size: 12px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: {MUTED};
          border-bottom: 1px solid {LINE}; padding-bottom: 6px; margin-bottom: 8px; }}

[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px; padding: 16px 18px; }}
[data-testid="stMetricLabel"] p {{ font-size: 13px !important; color: {MUTED} !important; }}
[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', Arial, sans-serif; color: {NAVY}; font-size: 28px !important; }}
[data-testid="stMetricLabel"] {{ white-space: normal !important; }}
[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] div {{ white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}

.card {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 16px; padding: 22px 24px; }}
.score-num {{ font-family: 'Space Grotesk', Arial, sans-serif; font-size: 64px; font-weight: 700; line-height: 1; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 5px 14px; font-size: 13px; font-weight: 700;
          letter-spacing: 1px; color: #FFFFFF; }}
.track {{ position: relative; height: 12px; border-radius: 999px; margin: 22px 0 8px;
          background: linear-gradient(90deg, {RIESGO_BAJO} 0%, {RIESGO_MEDIO} 55%, {RIESGO_ALTO} 100%); opacity: .9; }}
.marker {{ position: absolute; top: -7px; width: 4px; height: 26px; background: {INK}; border-radius: 2px; }}
.cut {{ position: absolute; top: -4px; width: 2px; height: 20px; background: #FFFFFF; border: 1px dashed {INK}; }}
.scale {{ display: flex; justify-content: space-between; font-size: 12px; color: {MUTED}; }}
.decision {{ font-family: 'Space Grotesk', Arial, sans-serif; font-size: 22px; font-weight: 700; margin: 18px 0 4px; }}
.muted {{ color: {MUTED}; font-size: 13px; line-height: 1.5; }}
.factors {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 6px; }}
.factor {{ border-radius: 12px; padding: 14px 16px; font-size: 14px; line-height: 1.5; }}
.factor b {{ display: block; font-size: 12px; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 4px; }}
.f-up {{ background: #FBEBDD; color: #6E2F0C; }}
.f-down {{ background: #E3EEF4; color: #163C52; }}

.side-title {{ font-family: 'Space Grotesk', Arial, sans-serif; font-weight: 700; color: {NAVY}; font-size: 18px; margin: 4px 0 10px; }}
.stat {{ border-left: 3px solid {ORANGE}; padding: 2px 0 2px 12px; margin-bottom: 14px; }}
.stat .v {{ font-family: 'Space Grotesk', Arial, sans-serif; font-size: 26px; font-weight: 700; color: {NAVY}; line-height: 1.1; }}
.stat .l {{ font-size: 13px; color: {MUTED}; line-height: 1.4; }}
.stat.alt {{ border-left-color: {NAVY_2}; }}

.foot {{ margin-top: 40px; padding-top: 14px; border-top: 1px solid {LINE}; color: {MUTED}; font-size: 13px;
         display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }}
</style>
""",
    unsafe_allow_html=True,
)


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def encabezado(eyebrow: str, titulo: str, sub: str = ""):
    html(f'<div class="eyebrow">{eyebrow}</div><div class="section-title">{titulo}</div>'
         + (f'<div class="section-sub">{sub}</div>' if sub else ""))


def usd(x: float) -> str:
    if abs(x) >= 1e6:
        return f"USD {x / 1e6:.2f} M".replace(".", ",")
    if abs(x) >= 1e3:
        return f"USD {x / 1e3:.0f} mil"
    return f"USD {x:.0f}"


def grafico(fig, height=400):
    fig.update_layout(height=height, legend_title_text="")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------- Modelo
@st.cache_resource(show_spinner="Entrenando y validando los modelos…")
def preparar():
    df = m.cargar_datos()
    tabla, ajustados, probs, split = m.evaluar_todos(df)
    final = m.entrenar_modelo_final(df)
    X_tr, _, y_tr, _ = split
    oof = m.prob_oof(m.modelos_candidatos()["Regresión logística"], X_tr, y_tr)
    return df, tabla, ajustados, probs, split, final, oof


df, tabla, ajustados, probs, (X_tr, X_te, y_tr, y_te), final, oof = preparar()
prob_test = final.predict_proba(X_te)[:, 1]
auc_final = float(tabla.loc[tabla["Modelo"] == "Regresión logística", "AUC test"].iloc[0])

# ---------------------------------------------------------------- Encabezado
html(f"""
<div class="hero">
  <div>
    <div class="brand">Coop<span>Score</span></div>
    <div class="tag">Evaluación del riesgo de mora para cooperativas de ahorro y crédito.
    Estima el riesgo de cada solicitud, explica sus causas y lo traduce a costo para la cooperativa.</div>
  </div>
  <div class="chips">
    <span class="chip">Regresión logística · AUC {auc_final:.2f}</span>
    <span class="chip">{len(df):,} solicitudes históricas</span>
    <span class="chip">Sin variables de sexo ni nacionalidad</span>
  </div>
</div>
""")

with st.sidebar:
    html('<div class="side-title">Contexto en Ecuador</div>')
    html("""
<div class="stat"><div class="v">8,05 %</div><div class="l">Morosidad de las cooperativas del segmento 1, dic-2025</div></div>
<div class="stat alt"><div class="v">2,99 %</div><div class="l">Morosidad de la banca privada en el mismo mes</div></div>
<div class="stat alt"><div class="v">6,3 millones</div><div class="l">Socios del sector financiero popular y solidario</div></div>
<div class="muted">Fuentes: SEPS; Primicias (2026).</div>
""")
    st.divider()
    html('<div class="side-title">Política de crédito</div>')
    umbral = st.slider(
        "Umbral de rechazo", 0.05, 0.95, 0.50, 0.01,
        help="Las solicitudes con score de riesgo mayor o igual al umbral se rechazan o pasan a comité.",
    )
    html(f'<div class="muted">Score menor a {umbral * 0.6:.2f}: aprobar · entre {umbral * 0.6:.2f} y '
         f'{umbral:.2f}: aprobar con condiciones · desde {umbral:.2f}: comité o rechazo.</div>')

tab_socio, tab_impacto, tab_modelos, tab_datos, tab_info = st.tabs(
    ["Evaluar socio", "Impacto financiero", "Modelos", "Datos de cartera", "Metodología"]
)

# ---------------------------------------------------------------- Evaluar socio
with tab_socio:
    encabezado("Solicitud", "Evaluación de un socio",
               "Ingrese los datos de la solicitud. El modelo devuelve un score de riesgo, una recomendación "
               "y los factores que más influyeron en el resultado.")
    opts = {c: sorted(df[c].unique()) for c in m.CATEGORICAS}
    with st.form("solicitud", border=False):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            html('<div class="group">Crédito</div>')
            monto = st.number_input("Monto solicitado", 250, 20000, 3000, 250)
            plazo = st.slider("Plazo (meses)", 4, 72, 18)
            destino = st.selectbox("Destino del crédito", opts["destino_credito"],
                                   index=opts["destino_credito"].index("Negocio / emprendimiento"))
            carga = st.select_slider("Carga de la cuota sobre el ingreso", [1, 2, 3, 4], 3,
                                     help="1 = cuota baja respecto al ingreso disponible, 4 = cuota alta")
            garantes = st.selectbox("Garantes", opts["garantes"], index=opts["garantes"].index("Ninguno"))
        with c2:
            html('<div class="group">Relación con la cooperativa</div>')
            saldo = st.selectbox("Saldo en cuenta a la vista", opts["saldo_cuenta"],
                                 index=opts["saldo_cuenta"].index("Alto (200 o más)"))
            ahorros = st.selectbox("Ahorros", opts["ahorros"], index=opts["ahorros"].index("100 a 500"))
            historial = st.selectbox("Historial crediticio", opts["historial_crediticio"],
                                     index=opts["historial_crediticio"].index("Créditos vigentes al día"))
            creditos = st.number_input("Créditos vigentes", 1, 4, 1)
            otras = st.selectbox("Otras deudas", opts["otras_deudas"], index=opts["otras_deudas"].index("Ninguna"))
        with c3:
            html('<div class="group">Perfil del socio</div>')
            edad = st.slider("Edad", 18, 75, 35)
            antig = st.selectbox("Antigüedad laboral", opts["antiguedad_laboral"],
                                 index=opts["antiguedad_laboral"].index("4 a 7 años"))
            ocup = st.selectbox("Ocupación", opts["ocupacion"], index=opts["ocupacion"].index("Empleado calificado"))
            vivienda = st.selectbox("Vivienda", opts["vivienda"], index=opts["vivienda"].index("Propia"))
            patrimonio = st.selectbox("Patrimonio", opts["patrimonio"],
                                      index=opts["patrimonio"].index("Vehículo u otro bien"))
            resid = st.slider("Años en la residencia actual", 1, 4, 2)
            cc1, cc2 = st.columns(2)
            cargas = cc1.radio("Cargas familiares", [1, 2], horizontal=True)
            tel = cc2.radio("Teléfono registrado", opts["telefono_registrado"], horizontal=True)
        st.form_submit_button("Calcular riesgo", type="primary", width="stretch")

    socio = pd.DataFrame([{
        "saldo_cuenta": saldo, "plazo_meses": plazo, "historial_crediticio": historial,
        "destino_credito": destino, "monto": monto, "ahorros": ahorros,
        "antiguedad_laboral": antig, "carga_cuota": carga, "garantes": garantes,
        "anios_residencia": resid, "patrimonio": patrimonio, "edad": edad,
        "otras_deudas": otras, "vivienda": vivienda, "creditos_vigentes": creditos,
        "ocupacion": ocup, "cargas_familiares": cargas, "telefono_registrado": tel,
    }])
    socio = m.limpiar(socio)[m.NUMERICAS + m.CATEGORICAS]
    p = float(final.predict_proba(socio)[:, 1][0])

    if p < umbral * 0.6:
        nivel, color, decision = "RIESGO BAJO", RIESGO_BAJO, "Aprobar"
    elif p < umbral:
        nivel, color, decision = "RIESGO MEDIO", RIESGO_MEDIO, "Aprobar con condiciones: garante, menor monto o plazo"
    else:
        nivel, color, decision = "RIESGO ALTO", RIESGO_ALTO, "Rechazar o enviar a comité de crédito"

    contrib = m.contribuciones_logistica(final, socio).head(10)
    sube = [m.ETIQUETAS.get(v, v) for v, a in contrib.items() if a > 0][:3]
    baja = [m.ETIQUETAS.get(v, v) for v, a in contrib.items() if a < 0][:3]

    st.write("")
    r1, r2 = st.columns([5, 7], gap="large")
    with r1:
        html(f"""
<div class="card">
  <div class="eyebrow" style="margin-top:0">Resultado</div>
  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
    <div class="score-num" style="color:{color}">{p * 100:.0f}<span style="font-size:30px"> / 100</span></div>
    <span class="badge" style="background:{color}">{nivel}</span>
  </div>
  <div class="track">
    <div class="cut" style="left:calc({umbral * 100:.1f}% - 1px)"></div>
    <div class="marker" style="left:calc({p * 100:.1f}% - 2px)"></div>
  </div>
  <div class="scale"><span>0</span><span>Umbral {umbral * 100:.0f}</span><span>100</span></div>
  <div class="decision" style="color:{color}">{decision}</div>
  <div class="muted">Cuota mensual aproximada sin intereses: <b>{monto / plazo:,.0f}</b></div>
  <div class="muted" style="margin-top:10px">El score ordena a los socios por riesgo. Se entrenó con pesos balanceados
  para priorizar la detección de morosos, por lo que no equivale a la probabilidad real de mora de la cartera.</div>
</div>
""")
    with r2:
        exp = pd.DataFrame({
            "Variable": [m.ETIQUETAS.get(v, v) for v in contrib.index],
            "Aporte": contrib.values,
        }).iloc[::-1]
        exp["Efecto"] = np.where(exp["Aporte"] > 0, "Aumenta el riesgo", "Reduce el riesgo")
        fig = px.bar(exp, x="Aporte", y="Variable", orientation="h", color="Efecto",
                     color_discrete_map={"Aumenta el riesgo": RIESGO_ALTO, "Reduce el riesgo": NAVY_2},
                     title="Qué explica el resultado (frente al socio promedio)")
        fig.update_layout(xaxis_title="Aporte al riesgo", yaxis_title=None, bargap=0.35)
        grafico(fig, 360)
        html(f"""
<div class="factors">
  <div class="factor f-up"><b>Elevan el riesgo</b>{', '.join(sube) or 'Ninguno relevante'}</div>
  <div class="factor f-down"><b>Reducen el riesgo</b>{', '.join(baja) or 'Ninguno relevante'}</div>
</div>
""")

# ---------------------------------------------------------------- Impacto financiero
with tab_impacto:
    encabezado("Finanzas", "Cuánto le ahorra el modelo a la cooperativa",
               "Aprobar a un socio que cae en mora hace perder parte del capital prestado; rechazar a un buen "
               "socio solo hace perder el margen de interés. Ajuste los supuestos de su cooperativa.")
    a1, a2, a3, a4 = st.columns(4, gap="medium")
    monto_prom = a1.number_input("Monto promedio por crédito (USD)", 500, 50000, 5000, 500)
    lgd = a2.slider("Pérdida si hay mora (% del monto)", 10, 100, 45,
                    help="Loss Given Default: porción del capital que no se recupera tras la cobranza.") / 100
    margen = a3.slider("Margen de un buen crédito (%)", 1, 40, 12,
                       help="Ganancia neta que se deja de percibir al rechazar a un buen socio.") / 100
    n_anual = a4.number_input("Solicitudes por año", 100, 100000, 2000, 100)

    c_fn, c_fp = monto_prom * lgd, monto_prom * margen
    u_opt, umbrales, _ = m.umbral_optimo(y_tr, oof, c_fn, c_fp)
    costos_test = np.array([m.costo_esperado(y_te, prob_test, u, c_fn, c_fp) for u in umbrales])
    n_test = len(y_te)
    escala = n_anual / n_test

    costo_todos = m.costo_esperado(y_te, prob_test, 1.01, c_fn, c_fp) * escala
    costo_modelo = m.costo_esperado(y_te, prob_test, umbral, c_fn, c_fp) * escala
    ahorro = costo_todos - costo_modelo

    st.write("")
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric("Pérdida anual aprobando a todos", usd(costo_todos))
    k2.metric(f"Costo anual con el modelo (umbral {umbral:.2f})", usd(costo_modelo))
    k3.metric(f"Ahorro anual estimado ({ahorro / costo_todos:.0%} menos)" if costo_todos else "Ahorro anual estimado",
              usd(ahorro))
    k4.metric("Umbral recomendado", f"{u_opt:.2f}",
              help="Minimiza el costo con predicciones de validación cruzada, sin usar los datos de prueba.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=umbrales, y=costos_test * escala, mode="lines", name="Costo con el modelo",
                             line=dict(color=NAVY, width=3), fill="tozeroy", fillcolor="rgba(18,48,74,0.06)"))
    fig.add_hline(y=costo_todos, line_dash="dash", line_color=ORANGE,
                  annotation_text="Aprobar a todos", annotation_font_color=ORANGE)
    fig.add_vline(x=umbral, line_dash="dot", line_color=INK, line_width=2, layer="above",
                  annotation_text="Umbral actual")
    fig.add_vline(x=u_opt, line_color=NAVY_2, line_width=2, layer="above", annotation_text="Recomendado",
                  annotation_position="bottom right", annotation_font_color=NAVY_2)
    fig.update_layout(title="Costo anual esperado según el umbral de rechazo", showlegend=False,
                      xaxis_title="Umbral de rechazo", yaxis_title="USD por año", yaxis_tickformat=",.0f")
    grafico(fig, 420)

    pred = (prob_test >= umbral).astype(int)
    tn, fp, fn, tp = ((y_te == 0) & (pred == 0)).sum(), ((y_te == 0) & (pred == 1)).sum(), \
        ((y_te == 1) & (pred == 0)).sum(), ((y_te == 1) & (pred == 1)).sum()
    html(f"""
<div class="card"><div class="muted" style="font-size:15px;color:{INK}">Con el umbral <b>{umbral:.2f}</b>, en el
conjunto de prueba ({n_test} solicitudes) el modelo detecta <b>{tp} de {tp + fn} morosos ({tp / (tp + fn):.0%})</b>,
rechaza a <b>{fp} buenos socios</b> y aprueba <b>{tn + fn}</b> créditos con una mora de
<b>{fn / max(tn + fn, 1):.1%}</b>, frente al {y_te.mean():.0%} si se aprobara a todos.</div>
<div class="muted" style="margin-top:8px">Estimación escalada al número de solicitudes anuales. El dataset tiene 30 % de
mora; con la morosidad real de una cooperativa del segmento 1 (alrededor del 8 %) el ahorro absoluto sería menor y debe
recalcularse con la cartera propia.</div></div>
""")

# ---------------------------------------------------------------- Modelos
with tab_modelos:
    encabezado("Machine learning", "Comparación de algoritmos supervisados",
               "Cuatro clasificadores con el mismo preprocesamiento (escalado estándar y one-hot), entrenados con "
               "el 75 % de los datos y validación cruzada estratificada de 5 folds. El 25 % restante se reservó "
               "como prueba.")
    vista = tabla.rename(columns={
        "AUC CV (media)": "AUC CV", "AUC CV (desv.)": "Desv. CV", "AUC-PR test": "AUC-PR",
        "Recall@0.5": "Recall (0,5)", "Costo test (umbral 0.5)": "Costo (0,5)",
        "Umbral óptimo (CV)": "Umbral óptimo", "Costo test (umbral óptimo)": "Costo (óptimo)",
    })
    st.dataframe(
        vista.style.format({c: "{:.3f}" for c in vista.columns if c != "Modelo"}
                           | {"Costo (0,5)": "{:.0f}", "Costo (óptimo)": "{:.0f}", "Umbral óptimo": "{:.2f}",
                              "Recall (0,5)": "{:.0%}"}, decimal=",")
        .highlight_max(subset=["AUC CV", "AUC test", "Recall (0,5)"], color="#FBE3CC")
        .highlight_min(subset=["Costo (0,5)", "Costo (óptimo)"], color="#FBE3CC"),
        hide_index=True, width="stretch",
    )
    html('<div class="muted">Costo = 5 × morosos aprobados + 1 × buenos socios rechazados (matriz de costos del '
         'dataset Statlog). El umbral óptimo de cada modelo se calcula con predicciones de validación cruzada.</div>')

    g1, g2 = st.columns(2, gap="large")
    with g1:
        roc = go.Figure()
        for nombre, pr in probs.items():
            fpr, tpr = m.curva_roc(y_te, pr)
            ancho = 3.5 if nombre == "Regresión logística" else 1.8
            roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=nombre, line=dict(width=ancho)))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                                 line=dict(dash="dash", color="#B8B2A4", width=1)))
        roc.update_layout(title="Curvas ROC en el conjunto de prueba", xaxis_title="Tasa de falsos positivos",
                          yaxis_title="Morosos detectados (recall)",
                          legend=dict(orientation="v", x=0.98, y=0.04, xanchor="right", yanchor="bottom",
                                      bgcolor="rgba(255,255,255,0.85)", bordercolor=LINE, borderwidth=1))
        grafico(roc, 440)
    with g2:
        pred = (prob_test >= umbral).astype(int)
        cm = pd.crosstab(pd.Series(y_te.values, name="Real"), pd.Series(pred, name="Predicción")) \
            .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        fig = px.imshow(cm.values, text_auto=True, color_continuous_scale=[[0, "#F3EFE6"], [1, NAVY]],
                        x=["Aprobado", "Rechazado"], y=["Buen socio", "Moroso"],
                        title=f"Matriz de confusión con umbral {umbral:.2f}")
        fig.update_traces(textfont=dict(size=22, family="Space Grotesk, Arial"))
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Decisión del modelo", yaxis_title="Resultado real")
        grafico(fig, 440)

    html(f"""
<div class="card" style="border-left:4px solid {ORANGE}">
<div class="eyebrow" style="margin-top:0">Modelo en producción</div>
<p style="font-size:15px;line-height:1.6;color:{INK};margin:0">Regresión logística con pesos balanceados. Gradient
Boosting logra el mejor AUC (0,82) y el menor costo cuando cada modelo usa su umbral óptimo (121 frente a 141, unos
cuatro morosos aprobados de diferencia en 250 solicitudes). Se adopta la regresión logística porque es explicable
variable por variable: el oficial de crédito, el comité y el supervisor pueden justificar cada aprobación o rechazo.
Migrar a Gradient Boosting con explicaciones SHAP es la principal mejora futura.</p></div>
""")
    st.write("")
    coefs = pd.Series(final.named_steps["clf"].coef_[0],
                      index=final.named_steps["prep"].get_feature_names_out())
    def etiqueta(n: str) -> str:
        tipo, resto = n.split("__", 1)
        if tipo == "num":
            return m.ETIQUETAS.get(resto, resto)
        var = next(c for c in m.CATEGORICAS if resto.startswith(c + "_"))
        return f"{m.ETIQUETAS[var]}: {resto[len(var) + 1:]}"
    coefs.index = [etiqueta(n) for n in coefs.index]
    top = pd.concat([coefs.nlargest(8), coefs.nsmallest(8)]).sort_values()
    efecto = np.where(top > 0, "Más riesgo", "Menos riesgo")
    fig = px.bar(x=top.values, y=top.index, orientation="h", color=efecto,
                 color_discrete_map={"Más riesgo": RIESGO_ALTO, "Menos riesgo": NAVY_2},
                 title="Variables con mayor peso en el modelo")
    fig.update_layout(xaxis_title="Coeficiente", yaxis_title=None, bargap=0.3)
    grafico(fig, 520)

# ---------------------------------------------------------------- Datos
with tab_datos:
    encabezado("Cartera", "Análisis exploratorio",
               "Cómo se distribuye la mora histórica entre los distintos perfiles de socios.")
    d1, d2, d3 = st.columns(3, gap="medium")
    d1.metric("Solicitudes analizadas", f"{len(df):,}")
    d2.metric("Tasa de mora histórica", f"{df[m.OBJETIVO].mean():.0%}")
    d3.metric("Variables usadas por el modelo", len(m.NUMERICAS) + len(m.CATEGORICAS))

    st.write("")
    var = st.selectbox("Tasa de mora según", m.CATEGORICAS, format_func=lambda c: m.ETIQUETAS.get(c, c))
    agg = df.groupby(var)[m.OBJETIVO].agg(["mean", "size"]).reset_index().sort_values("mean")
    fig = px.bar(agg, x="mean", y=var, orientation="h", text=agg["mean"].map("{:.0%}".format),
                 hover_data={"size": True}, color="mean",
                 color_continuous_scale=[[0, "#9DBBD0"], [0.5, "#D9B48A"], [1, RIESGO_ALTO]],
                 labels={"mean": "Tasa de mora", var: "", "size": "Solicitudes"})
    fig.add_vline(x=df[m.OBJETIVO].mean(), line_dash="dash", line_color=MUTED,
                  annotation_text="Promedio", annotation_font_color=MUTED,
                  annotation_position="bottom right")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%", bargap=0.35,
                      title=f"Tasa de mora por {m.ETIQUETAS[var].lower()}")
    grafico(fig, 420)

    e1, e2 = st.columns(2, gap="large")
    dfp = df.assign(Estado=df[m.OBJETIVO].map({0: "Buen pagador", 1: "Moroso"}))
    colores = {"Buen pagador": NAVY_2, "Moroso": RIESGO_ALTO}
    with e1:
        fig = px.box(dfp, x="Estado", y="plazo_meses", color="Estado", color_discrete_map=colores,
                     title="Plazo del crédito según resultado", labels={"plazo_meses": "Plazo (meses)", "Estado": ""})
        fig.update_layout(showlegend=False)
        grafico(fig, 380)
    with e2:
        fig = px.histogram(dfp, x="monto", color="Estado", nbins=40, barmode="overlay", opacity=0.75,
                           color_discrete_map=colores, title="Distribución del monto solicitado",
                           labels={"monto": "Monto", "Estado": ""})
        fig.update_layout(yaxis_title="Solicitudes")
        grafico(fig, 380)

# ---------------------------------------------------------------- Metodología
with tab_info:
    encabezado("Proyecto", "Problema, solución y fuentes")
    st.markdown(
        """
**Problema.** Las cooperativas de ahorro y crédito del segmento 1 en Ecuador cerraron 2025 con una morosidad de
**8,05 %**, casi el triple que la banca privada (**2,99 %**), sobre una cartera de **USD 15.617 millones**. El
sector popular y solidario atiende a más de **6,3 millones de socios**, muchos sin historial en la banca
tradicional, y más de 30 cooperativas han sido liquidadas desde 2022. Evaluar mejor el riesgo antes de aprobar un
crédito protege los ahorros de los socios y la solvencia de la cooperativa.

**Solución**
1. **Datos:** Statlog German Credit (UCI), 1000 solicitudes reales con resultado de pago, adaptadas al lenguaje de
   una cooperativa.
2. **Preprocesamiento:** traducción y limpieza, eliminación de duplicados, nueva variable *cuota mensual*,
   exclusión de variables sensibles (sexo/estado civil y nacionalidad), escalado estándar y codificación one-hot.
3. **Modelado:** regresión logística, Random Forest, Gradient Boosting y red neuronal MLP, comparados con
   validación cruzada estratificada y un conjunto de prueba independiente.
4. **Decisión financiera:** el umbral de rechazo minimiza el costo esperado (pérdida por mora frente a margen
   perdido), no solo la exactitud.
5. **Explicabilidad:** cada evaluación muestra qué variables subieron o bajaron el riesgo del socio.

**Fuentes**
- Superintendencia de Economía Popular y Solidaria (SEPS), boletines *Actualidad & Cifras*.
- Primicias (2026), indicadores de morosidad de cooperativas del segmento 1 a diciembre de 2025.
- Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository.
"""
    )

html("""
<div class="foot">
  <span>CoopScore · Proyecto final de Fundamentos de Inteligencia Artificial · Universidad Yachay Tech</span>
  <span>Joel Alexander Inca Tapia</span>
</div>
""")
