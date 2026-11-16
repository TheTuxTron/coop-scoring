"""Comparación de modelos y generación de reportes (reports/)."""
import json
from pathlib import Path

import pandas as pd

import modelo as m

SALIDA = Path(__file__).parent / "reports"


def main():
    SALIDA.mkdir(exist_ok=True)
    df = m.cargar_datos()
    print(f"Registros: {len(df)} | tasa de mora: {df[m.OBJETIVO].mean():.1%}")

    tabla, ajustados, probs, (X_tr, X_te, y_tr, y_te) = m.evaluar_todos(df)
    pd.set_option("display.width", 200)
    print(tabla.round(3).to_string(index=False))
    tabla.to_csv(SALIDA / "comparacion_modelos.csv", index=False)

    final = ajustados["Regresión logística"]
    u, _, _ = m.umbral_optimo(y_tr, m.prob_oof(m.modelos_candidatos()["Regresión logística"], X_tr, y_tr))
    prob = probs["Regresión logística"]
    res = {
        "umbral_optimo_costo_5a1": u,
        "test_umbral_optimo": m.metricas(y_te, prob, u),
        "test_umbral_0.5": m.metricas(y_te, prob, 0.5),
        "costo_test_umbral_optimo": m.costo_esperado(y_te, prob, u),
        "costo_test_umbral_0.5": m.costo_esperado(y_te, prob, 0.5),
        "costo_test_aprobar_todos": m.costo_esperado(y_te, prob, 1.01),
    }
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    (SALIDA / "metricas_modelo_final.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
