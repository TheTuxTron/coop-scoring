# CoopScore — Scoring de riesgo de mora para cooperativas de ahorro y crédito

Proyecto final del módulo **Fundamentos de Inteligencia Artificial** (Universidad Yachay Tech) · Joel Alexander Inca Tapia.

**Aplicación en línea:** https://coopscore-yachay.streamlit.app/

CoopScore es un sistema de machine learning que estima el riesgo de que un socio caiga en mora **antes** de aprobar su crédito, explica qué factores determinan ese riesgo y traduce la política de aprobación a costo financiero para la cooperativa. Se publicó como aplicación web en Streamlit Community Cloud.

## Problema

- Las cooperativas de ahorro y crédito del **segmento 1** de Ecuador cerraron diciembre de 2025 con una **morosidad de 8,05 %**, frente al **2,99 %** de la banca privada, sobre una cartera de **USD 15.617 millones** (SEPS / Primicias).
- El sector financiero popular y solidario atiende a más de **6,3 millones de socios**, muchos sin historial en la banca tradicional, y más de 30 cooperativas han sido liquidadas desde 2022.
- La evaluación de crédito suele ser manual y poco estandarizada. Un error de aprobación cuesta capital, que son los ahorros de los propios socios; un rechazo injustificado cuesta margen e inclusión financiera.

## Solución

| Etapa | Qué se hizo |
|---|---|
| Datos | *Statlog German Credit* (UCI): 1000 solicitudes reales con resultado de pago (30 % mora). Variables y categorías traducidas al lenguaje de una cooperativa (ahorros, saldo a la vista, garantes, destino del crédito, etc.). |
| Limpieza y preprocesamiento | Eliminación de duplicados, nueva variable *cuota mensual* (monto / plazo), **exclusión de variables sensibles** (sexo/estado civil y nacionalidad) por equidad, escalado estándar de numéricas y one-hot de categóricas en un `Pipeline` de scikit-learn. |
| Modelado (aprendizaje supervisado, clasificación) | Regresión logística, Random Forest, Gradient Boosting y red neuronal MLP. División 75/25 estratificada, validación cruzada estratificada de 5 folds y manejo del desbalance con pesos de clase. |
| Decisión financiera | El umbral de rechazo minimiza el **costo esperado**: aprobar a un moroso (pérdida de capital) cuesta más que rechazar a un buen socio (margen perdido). El umbral se calibra con predicciones *out-of-fold*, sin tocar el conjunto de prueba. |
| Explicabilidad | Para cada socio se muestra el aporte de cada variable al riesgo frente al socio promedio (coeficiente × valor). |
| Despliegue | App en Streamlit con cinco secciones: evaluación de socio, impacto financiero, comparación de modelos, análisis de cartera y documentación. |

## Resultados

Conjunto de prueba (250 solicitudes):

| Modelo | AUC CV | AUC test | Recall @0,5 | Costo @0,5 | Costo @umbral óptimo |
|---|---|---|---|---|---|
| **Regresión logística (producción)** | 0,761 | 0,804 | **0,77** | **133** | 141 |
| Random Forest | **0,789** | 0,802 | 0,60 | 176 | 130 |
| Gradient Boosting | 0,770 | **0,816** | 0,56 | 182 | **121** |
| Red neuronal (MLP) | 0,776 | 0,809 | 0,48 | 210 | 133 |

*Costo = 5 × morosos aprobados + 1 × buenos socios rechazados (matriz de costos de Statlog).*

- La regresión logística detecta **77 % de los morosos** y reduce el costo de errores de **375 (aprobar a todos) a 133 (−65 %)**. La mora de los créditos aprobados baja del 30 % al 12 %.
- Los cuatro modelos tienen un AUC similar (0,80–0,82). Con su umbral óptimo, Gradient Boosting logra el menor costo; la regresión logística se mantuvo en producción por ser explicable, requisito para justificar cada decisión ante el comité de crédito y el supervisor.
- Según la importancia por permutación (caída de AUC en prueba), las variables más influyentes son el saldo en la cuenta a la vista (0,093), el historial crediticio, el destino del crédito, el monto y el plazo.

## Conclusiones

- Un modelo sencillo y explicable ya reduce de forma sustancial las pérdidas esperadas y estandariza la evaluación de crédito.
- Optimizar el **costo financiero** en lugar de la exactitud cambia la decisión: el umbral óptimo depende del monto, la pérdida en mora y el margen de cada cooperativa.
- Excluir variables sensibles casi no afecta el desempeño y evita discriminar a los socios.

## Limitaciones y mejoras futuras

- Reentrenar con datos propios de una cooperativa ecuatoriana (USD, morosidad real ~8 %) y **calibrar** las probabilidades, pues los pesos balanceados elevan el score respecto a la tasa real.
- Gradient Boosting / XGBoost con explicaciones **SHAP** para ganar precisión sin perder explicabilidad.
- Incorporar variables locales: buró de crédito, ingresos verificados, actividad económica, remesas y región.
- Monitoreo del modelo (drift) y auditoría periódica de equidad.
- Asistente con LLM que redacte el informe de crédito para el comité a partir del score y sus factores.

## Estructura

```
coop-scoring/
├── app.py                  # Aplicación Streamlit
├── modelo.py               # Datos, preprocesamiento, modelos, costo y explicabilidad
├── train.py                # Comparación de modelos y generación de reportes
├── data/german_credit_raw.csv
├── reports/                # Métricas generadas por train.py
└── requirements.txt
```

Ejecución local: `pip install -r requirements.txt` y `streamlit run app.py`.

## Fuentes

- Superintendencia de Economía Popular y Solidaria (SEPS), boletines *Actualidad & Cifras*: https://www.seps.gob.ec
- Primicias (2026), morosidad e indicadores de cooperativas del segmento 1: https://www.primicias.ec/economia/
- Vistazo (2026), cooperativas liquidadas desde 2022: https://www.vistazo.com
- Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
