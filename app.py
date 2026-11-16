"""CoopScore: scoring de morosidad para cooperativas de ahorro y crédito."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import modelo as m

st.set_page_config(page_title="CoopScore · Riesgo de mora", page_icon="🏦", layout="wide")

AZUL, ROJO, VERDE, AMBAR = "#1f4e79", "#c0392b", "#2e8b57", "#d4a017"


@st.cache_resource(show_spinner="Entrenando modelos…")
def preparar():
    df = m.cargar_datos()
    tabla, ajustados, probs, split = m.evaluar_todos(df)
    final = m.entrenar_modelo_final(df)
    X_tr, _, y_tr, _ = split
    oof = m.prob_oof(m.modelos_candidatos()["Regresión logística"], X_tr, y_tr)
    return df, tabla, ajustados, probs, split, final, oof


df, tabla, ajustados, probs, (X_tr, X_te, y_tr, y_te), final, oof = preparar()
prob_test = final.predict_proba(X_te)[:, 1]

# ---------------------------------------------------------------- Encabezado
st.title("🏦 CoopScore — Scoring de riesgo de mora para cooperativas")
st.caption(
    "Modelo de machine learning que estima la probabilidad de que un socio caiga en mora "
    "antes de aprobar su crédito, y traduce la decisión a costo financiero para la cooperativa."
)

with st.sidebar:
    st.header("Contexto: Ecuador")
    st.metric("Morosidad cooperativas segmento 1 (dic-2025)", "8,05 %", "+5,06 pp vs. banca privada", delta_color="inverse")
    st.metric("Morosidad banca privada (dic-2025)", "2,99 %")
    st.metric("Socios del sector popular y solidario", "6,3 millones")
    st.caption("Fuentes: SEPS; Primicias (2026). Ver pestaña «Acerca de».")
    st.divider()
    st.header("Política de crédito")
    umbral = st.slider(
        "Umbral de rechazo (probabilidad de mora)", 0.05, 0.95, 0.50, 0.01,
        help="Solicitudes con probabilidad de mora mayor o igual al umbral se rechazan o pasan a comité.",
    )

tab_socio, tab_impacto, tab_modelos, tab_datos, tab_info = st.tabs(
    ["🧑‍💼 Evaluar socio", "💵 Impacto financiero", "🤖 Modelos", "📊 Datos de cartera", "ℹ️ Acerca de"]
)

# ---------------------------------------------------------------- Evaluar socio
with tab_socio:
    st.subheader("Solicitud de crédito")
    opts = {c: sorted(df[c].unique()) for c in m.CATEGORICAS}
    with st.form("solicitud"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Crédito**")
            monto = st.number_input("Monto solicitado", 250, 20000, 3000, 250)
            plazo = st.slider("Plazo (meses)", 4, 72, 18)
            destino = st.selectbox("Destino del crédito", opts["destino_credito"],
                                   index=opts["destino_credito"].index("Negocio / emprendimiento"))
            carga = st.select_slider("Carga de la cuota sobre ingreso disponible", [1, 2, 3, 4], 3,
                                     help="1 = cuota baja respecto al ingreso, 4 = cuota alta")
            garantes = st.selectbox("Garantes", opts["garantes"], index=opts["garantes"].index("Ninguno"))
        with c2:
            st.markdown("**Relación con la cooperativa**")
            saldo = st.selectbox("Saldo en cuenta a la vista", opts["saldo_cuenta"],
                                 index=opts["saldo_cuenta"].index("Alto (200 o más)"))
            ahorros = st.selectbox("Ahorros", opts["ahorros"], index=opts["ahorros"].index("100 a 500"))
            historial = st.selectbox("Historial crediticio", opts["historial_crediticio"],
                                     index=opts["historial_crediticio"].index("Créditos vigentes al día"))
            creditos = st.number_input("Créditos vigentes", 1, 4, 1)
            otras = st.selectbox("Otras deudas", opts["otras_deudas"], index=opts["otras_deudas"].index("Ninguna"))
        with c3:
            st.markdown("**Perfil del socio**")
            edad = st.slider("Edad", 18, 75, 35)
            antig = st.selectbox("Antigüedad laboral", opts["antiguedad_laboral"],
                                 index=opts["antiguedad_laboral"].index("4 a 7 años"))
            ocup = st.selectbox("Ocupación", opts["ocupacion"], index=opts["ocupacion"].index("Empleado calificado"))
            vivienda = st.selectbox("Vivienda", opts["vivienda"], index=opts["vivienda"].index("Propia"))
            patrimonio = st.selectbox("Patrimonio", opts["patrimonio"],
                                      index=opts["patrimonio"].index("Vehículo u otro bien"))
            resid = st.slider("Años en la residencia actual", 1, 4, 2)
            cargas = st.radio("Cargas familiares", [1, 2], horizontal=True)
            tel = st.radio("Teléfono registrado", opts["telefono_registrado"], horizontal=True)
        enviado = st.form_submit_button("Calcular riesgo", type="primary", width="stretch")

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
        nivel, color, decision = "BAJO", VERDE, "✅ Aprobar"
    elif p < umbral:
        nivel, color, decision = "MEDIO", AMBAR, "🟡 Aprobar con condiciones (garante, menor monto o plazo)"
    else:
        nivel, color, decision = "ALTO", ROJO, "⛔ Rechazar o enviar a comité de crédito"

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": " %"},
            title={"text": f"Score de riesgo de mora · {nivel}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, umbral * 60], "color": "#e8f5ec"},
                    {"range": [umbral * 60, umbral * 100], "color": "#fdf5dc"},
                    {"range": [umbral * 100, 100], "color": "#fbe5e2"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": umbral * 100},
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(gauge, width="stretch")
        st.markdown(f"### {decision}")
        st.caption(f"Cuota mensual aproximada (sin intereses): {monto / plazo:,.0f}")
        st.caption("El modelo se entrenó con pesos balanceados para priorizar la detección de morosos: el score ordena a los socios por riesgo y alimenta la decisión; no es una probabilidad calibrada a la morosidad real de la cartera.")

    with r2:
        contrib = m.contribuciones_logistica(final, socio).head(10)
        exp = pd.DataFrame({
            "Variable": [m.ETIQUETAS.get(v, v) for v in contrib.index],
            "Aporte": contrib.values,
        }).iloc[::-1]
        exp["Efecto"] = np.where(exp["Aporte"] > 0, "Aumenta el riesgo", "Reduce el riesgo")
        fig = px.bar(exp, x="Aporte", y="Variable", orientation="h", color="Efecto",
                     color_discrete_map={"Aumenta el riesgo": ROJO, "Reduce el riesgo": VERDE},
                     title="¿Por qué este resultado? (aporte al riesgo vs. socio promedio)")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), legend_title=None,
                          xaxis_title="Aporte al log-odds de mora")
        st.plotly_chart(fig, width="stretch")

    sube = [m.ETIQUETAS.get(v, v) for v, a in contrib.items() if a > 0][:3]
    baja = [m.ETIQUETAS.get(v, v) for v, a in contrib.items() if a < 0][:3]
    st.info(
        f"**Factores que más elevan el riesgo:** {', '.join(sube) or 'ninguno relevante'}.  \n"
        f"**Factores que más lo reducen:** {', '.join(baja) or 'ninguno relevante'}."
    )

# ---------------------------------------------------------------- Impacto financiero
with tab_impacto:
    st.subheader("¿Cuánto le ahorra el modelo a la cooperativa?")
    st.markdown(
        "Cada error tiene un costo distinto: **aprobar a un socio que cae en mora** hace perder parte del "
        "capital prestado, mientras que **rechazar a un buen socio** solo hace perder el margen de interés. "
        "Ajusta los supuestos de tu cooperativa:"
    )
    a1, a2, a3, a4 = st.columns(4)
    monto_prom = a1.number_input("Monto promedio por crédito (USD)", 500, 50000, 5000, 500)
    lgd = a2.slider("Pérdida si hay mora (% del monto)", 10, 100, 45,
                    help="Loss Given Default: porción del capital que no se recupera tras la cobranza.") / 100
    margen = a3.slider("Margen financiero de un buen crédito (%)", 1, 40, 12,
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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pérdida anual sin modelo (aprobar a todos)", f"USD {costo_todos:,.0f}")
    k2.metric(f"Costo con el modelo (umbral {umbral:.2f})", f"USD {costo_modelo:,.0f}")
    k3.metric("Ahorro anual estimado", f"USD {ahorro:,.0f}", f"{ahorro / costo_todos:.0%}" if costo_todos else None)
    k4.metric("Umbral recomendado para estos supuestos", f"{u_opt:.2f}",
              help="Minimiza el costo con predicciones de validación cruzada (sin usar datos de prueba).")

    st.caption(
        "Estimación sobre el conjunto de prueba, escalada al número de solicitudes anuales. El dataset tiene "
        "30 % de mora; con la morosidad real de una cooperativa del segmento 1 (~8 %) el ahorro absoluto sería "
        "menor y debe recalcularse con la cartera propia."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=umbrales, y=costos_test * escala, mode="lines", name="Costo con el modelo",
                             line=dict(color=AZUL, width=3)))
    fig.add_hline(y=costo_todos, line_dash="dash", line_color=ROJO, annotation_text="Aprobar a todos")
    fig.add_vline(x=umbral, line_dash="dot", annotation_text="Umbral actual")
    fig.add_vline(x=u_opt, line_dash="dot", line_color=VERDE, annotation_text="Recomendado",
                  annotation_position="bottom right")
    fig.update_layout(title="Costo anual esperado según el umbral de rechazo",
                      xaxis_title="Umbral (probabilidad de mora)", yaxis_title="USD por año", height=420)
    st.plotly_chart(fig, width="stretch")

    pred = (prob_test >= umbral).astype(int)
    tn, fp, fn, tp = ((y_te == 0) & (pred == 0)).sum(), ((y_te == 0) & (pred == 1)).sum(), \
        ((y_te == 1) & (pred == 0)).sum(), ((y_te == 1) & (pred == 1)).sum()
    st.markdown(
        f"Con el umbral **{umbral:.2f}**, en el conjunto de prueba ({n_test} solicitudes) el modelo "
        f"detecta **{tp} de {tp + fn} morosos ({tp / (tp + fn):.0%})**, rechaza a **{fp} buenos socios** "
        f"y aprueba **{tn + fn}** créditos con una mora de **{fn / max(tn + fn, 1):.1%}** "
        f"(frente al {y_te.mean():.0%} si se aprueba a todos)."
    )

# ---------------------------------------------------------------- Modelos
with tab_modelos:
    st.subheader("Comparación de algoritmos supervisados")
    st.markdown(
        "Se entrenaron cuatro clasificadores con el mismo preprocesamiento (escalado estándar + one-hot) "
        "sobre el 75 % de los datos, con validación cruzada estratificada de 5 folds; el 25 % restante "
        "se reservó como prueba."
    )
    st.dataframe(
        tabla.style.format({c: "{:.3f}" for c in tabla.columns if c != "Modelo"} | {"Costo test (umbral 0.5)": "{:.0f}", "Costo test (umbral óptimo)": "{:.0f}", "Umbral óptimo (CV)": "{:.2f}"})
        .highlight_max(subset=["AUC CV (media)", "AUC test", "Recall@0.5"], color="#d6eadf")
        .highlight_min(subset=["Costo test (umbral 0.5)", "Costo test (umbral óptimo)"], color="#d6eadf"),
        hide_index=True, width="stretch",
    )
    st.caption("Costo = 5 × morosos aprobados + 1 × buenos socios rechazados (matriz de costos del dataset Statlog). El umbral óptimo de cada modelo se calcula con predicciones de validación cruzada sobre el entrenamiento.")

    g1, g2 = st.columns(2)
    with g1:
        roc = go.Figure()
        for nombre, pr in probs.items():
            fpr, tpr = m.curva_roc(y_te, pr)
            roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=nombre))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                                 line=dict(dash="dash", color="gray")))
        roc.update_layout(title="Curvas ROC (conjunto de prueba)", xaxis_title="Tasa de falsos positivos",
                          yaxis_title="Tasa de verdaderos positivos (recall)", height=420)
        st.plotly_chart(roc, width="stretch")
    with g2:
        pred = (prob_test >= umbral).astype(int)
        cm = pd.crosstab(pd.Series(y_te.values, name="Real"), pd.Series(pred, name="Predicción")) \
            .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        fig = px.imshow(cm.values, text_auto=True, color_continuous_scale="Blues",
                        x=["Paga (aprobado)", "Mora (rechazado)"], y=["Buen socio", "Moroso"],
                        title=f"Matriz de confusión · Regresión logística · umbral {umbral:.2f}")
        fig.update_layout(height=420, coloraxis_showscale=False, xaxis_title="Predicción", yaxis_title="Real")
        st.plotly_chart(fig, width="stretch")

    st.markdown(
        "**Modelo elegido: regresión logística con pesos balanceados.** Gradient Boosting logra el mejor AUC "
        "(0,82) y el menor costo cuando cada modelo usa su umbral óptimo (121 vs. 141, unos 4 morosos "
        "aprobados de diferencia en 250 solicitudes). Aun así, la regresión logística se adopta en producción "
        "porque es **explicable variable por variable**: el oficial de crédito, el comité y el supervisor "
        "(SEPS) pueden justificar cada aprobación o rechazo. Migrar a Gradient Boosting con explicaciones SHAP "
        "es la principal mejora futura."
    )
    coefs = pd.Series(final.named_steps["clf"].coef_[0],
                      index=final.named_steps["prep"].get_feature_names_out())
    coefs.index = coefs.index.str.replace("num__", "").str.replace("cat__", "")
    top = pd.concat([coefs.nlargest(8), coefs.nsmallest(8)]).sort_values()
    fig = px.bar(top, orientation="h", color=top > 0,
                 color_discrete_map={True: ROJO, False: VERDE},
                 title="Variables con mayor peso en el modelo (rojo = más riesgo, verde = menos riesgo)")
    fig.update_layout(showlegend=False, height=500, xaxis_title="Coeficiente", yaxis_title=None)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Datos
with tab_datos:
    st.subheader("Análisis exploratorio de la cartera")
    d1, d2, d3 = st.columns(3)
    d1.metric("Solicitudes analizadas", f"{len(df):,}")
    d2.metric("Tasa de mora histórica", f"{df[m.OBJETIVO].mean():.0%}")
    d3.metric("Variables usadas por el modelo", len(m.NUMERICAS) + len(m.CATEGORICAS))

    var = st.selectbox("Tasa de mora según…", m.CATEGORICAS,
                       format_func=lambda c: m.ETIQUETAS.get(c, c))
    agg = df.groupby(var)[m.OBJETIVO].agg(["mean", "size"]).reset_index().sort_values("mean")
    fig = px.bar(agg, x="mean", y=var, orientation="h", text=agg["mean"].map("{:.0%}".format),
                 hover_data={"size": True}, color="mean", color_continuous_scale="RdYlGn_r",
                 labels={"mean": "Tasa de mora", var: m.ETIQUETAS[var], "size": "Solicitudes"})
    fig.add_vline(x=df[m.OBJETIVO].mean(), line_dash="dash", annotation_text="Promedio")
    fig.update_layout(coloraxis_showscale=False, height=420, xaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    e1, e2 = st.columns(2)
    dfp = df.assign(Estado=df[m.OBJETIVO].map({0: "Buen pagador", 1: "Moroso"}))
    with e1:
        fig = px.box(dfp, x="Estado", y="plazo_meses", color="Estado",
                     color_discrete_map={"Buen pagador": VERDE, "Moroso": ROJO},
                     title="Plazo del crédito vs. mora", labels={"plazo_meses": "Plazo (meses)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with e2:
        fig = px.histogram(dfp, x="monto", color="Estado", nbins=40, barmode="overlay",
                           color_discrete_map={"Buen pagador": VERDE, "Moroso": ROJO},
                           title="Distribución del monto solicitado", labels={"monto": "Monto"})
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Acerca de
with tab_info:
    st.subheader("Problema")
    st.markdown(
        """
Las cooperativas de ahorro y crédito del segmento 1 en Ecuador cerraron 2025 con una **morosidad de 8,05 %**,
casi **el triple que la banca privada (2,99 %)**, sobre una cartera de **USD 15.617 millones**. El sector
popular y solidario atiende a más de **6,3 millones de socios**, muchos de ellos sin historial en la banca
tradicional, y más de 30 cooperativas han sido liquidadas desde 2022. Evaluar mejor el riesgo **antes** de
aprobar un crédito protege los ahorros de los socios y la solvencia de la cooperativa.

### Solución
1. **Datos:** Statlog German Credit (UCI), 1000 solicitudes reales con resultado de pago, adaptadas al
   lenguaje de una cooperativa.
2. **Preprocesamiento:** traducción y limpieza, eliminación de duplicados, nueva variable *cuota mensual*,
   exclusión de variables sensibles (sexo/estado civil y nacionalidad) por equidad, escalado estándar y
   codificación one-hot.
3. **Modelado:** regresión logística, Random Forest, Gradient Boosting y red neuronal MLP, comparados con
   validación cruzada estratificada (AUC) y un conjunto de prueba independiente.
4. **Decisión financiera:** el umbral de rechazo se elige minimizando el costo esperado (pérdida por mora vs.
   margen perdido), no solo la exactitud.
5. **Explicabilidad:** cada evaluación muestra qué variables subieron o bajaron el riesgo del socio.

### Fuentes
- SEPS — Superintendencia de Economía Popular y Solidaria, boletines *Actualidad & Cifras*.
- Primicias (2026), indicadores de morosidad de cooperativas del segmento 1 a diciembre de 2025.
- Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository.
"""
    )
