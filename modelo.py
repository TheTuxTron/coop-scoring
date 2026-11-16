"""Preparación de datos, entrenamiento y evaluación del modelo de scoring de morosidad.

Fuente de datos: Statlog (German Credit Data), UCI Machine Learning Repository,
1000 solicitudes de crédito etiquetadas como buen o mal pagador.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RUTA_DATOS = Path(__file__).parent / "data" / "german_credit_raw.csv"
SEMILLA = 42

# ---------------------------------------------------------------------------
# Traducción del dataset al contexto de una cooperativa de ahorro y crédito
# ---------------------------------------------------------------------------
COLUMNAS = {
    "checking_status": "saldo_cuenta",
    "duration": "plazo_meses",
    "credit_history": "historial_crediticio",
    "purpose": "destino_credito",
    "credit_amount": "monto",
    "savings_status": "ahorros",
    "employment": "antiguedad_laboral",
    "installment_commitment": "carga_cuota",
    "other_parties": "garantes",
    "residence_since": "anios_residencia",
    "property_magnitude": "patrimonio",
    "age": "edad",
    "other_payment_plans": "otras_deudas",
    "housing": "vivienda",
    "existing_credits": "creditos_vigentes",
    "job": "ocupacion",
    "num_dependents": "cargas_familiares",
    "own_telephone": "telefono_registrado",
    "personal_status": "sexo_estado_civil",
    "foreign_worker": "extranjero",
    "class": "clase",
}

CATEGORIAS = {
    "saldo_cuenta": {
        "<0": "Negativo (sobregiro)",
        "0<=X<200": "Bajo (0 a 200)",
        ">=200": "Alto (200 o más)",
        "no checking": "Sin cuenta a la vista",
    },
    "historial_crediticio": {
        "critical/other existing credit": "Crítico / deudas en otras entidades",
        "existing paid": "Créditos vigentes al día",
        "delayed previously": "Retrasos en el pasado",
        "no credits/all paid": "Sin créditos previos",
        "all paid": "Todo pagado en la cooperativa",
    },
    "destino_credito": {
        "radio/tv": "Electrónicos",
        "education": "Educación",
        "furniture/equipment": "Muebles y equipamiento",
        "new car": "Vehículo nuevo",
        "used car": "Vehículo usado",
        "business": "Negocio / emprendimiento",
        "domestic appliance": "Electrodomésticos",
        "repairs": "Reparaciones de vivienda",
        "other": "Otro",
        "retraining": "Capacitación",
    },
    "ahorros": {
        "no known savings": "Sin ahorros registrados",
        "<100": "Menos de 100",
        "100<=X<500": "100 a 500",
        "500<=X<1000": "500 a 1000",
        ">=1000": "1000 o más",
    },
    "antiguedad_laboral": {
        "unemployed": "Desempleado",
        "<1": "Menos de 1 año",
        "1<=X<4": "1 a 4 años",
        "4<=X<7": "4 a 7 años",
        ">=7": "7 años o más",
    },
    "garantes": {"none": "Ninguno", "guarantor": "Garante", "co applicant": "Codeudor"},
    "patrimonio": {
        "real estate": "Bien inmueble",
        "life insurance": "Póliza / ahorro programado",
        "car": "Vehículo u otro bien",
        "no known property": "Sin patrimonio conocido",
    },
    "otras_deudas": {
        "none": "Ninguna",
        "bank": "Bancos u otras cooperativas",
        "stores": "Casas comerciales",
    },
    "vivienda": {"own": "Propia", "rent": "Arriendo", "for free": "Familiar / cedida"},
    "ocupacion": {
        "unemp/unskilled non res": "Sin empleo / no calificado",
        "unskilled resident": "No calificado",
        "skilled": "Empleado calificado",
        "high qualif/self emp/mgmt": "Profesional / independiente",
    },
    "telefono_registrado": {"yes": "Sí", "none": "No"},
}

# Variables excluidas del modelo por equidad: sexo/estado civil y nacionalidad
# no deben decidir el acceso al crédito de un socio.
SENSIBLES = ["sexo_estado_civil", "extranjero"]

NUMERICAS = [
    "plazo_meses",
    "monto",
    "carga_cuota",
    "anios_residencia",
    "edad",
    "creditos_vigentes",
    "cargas_familiares",
    "cuota_mensual",
]
CATEGORICAS = [
    "saldo_cuenta",
    "historial_crediticio",
    "destino_credito",
    "ahorros",
    "antiguedad_laboral",
    "garantes",
    "patrimonio",
    "otras_deudas",
    "vivienda",
    "ocupacion",
    "telefono_registrado",
]
OBJETIVO = "mora"

ETIQUETAS = {
    "saldo_cuenta": "Saldo en cuenta a la vista",
    "plazo_meses": "Plazo (meses)",
    "historial_crediticio": "Historial crediticio",
    "destino_credito": "Destino del crédito",
    "monto": "Monto solicitado",
    "ahorros": "Ahorros",
    "antiguedad_laboral": "Antigüedad laboral",
    "carga_cuota": "Carga de la cuota (1 baja – 4 alta)",
    "garantes": "Garantes",
    "anios_residencia": "Años en la residencia actual",
    "patrimonio": "Patrimonio",
    "edad": "Edad",
    "otras_deudas": "Otras deudas",
    "vivienda": "Vivienda",
    "creditos_vigentes": "Créditos vigentes",
    "ocupacion": "Ocupación",
    "cargas_familiares": "Cargas familiares",
    "telefono_registrado": "Teléfono registrado",
    "cuota_mensual": "Cuota mensual aprox.",
}


def cargar_datos(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV original, traduce columnas/categorías y crea la variable objetivo."""
    df = pd.read_csv(ruta).rename(columns=COLUMNAS)
    for col, mapa in CATEGORIAS.items():
        df[col] = df[col].map(mapa)
    df[OBJETIVO] = (df.pop("clase") == "bad").astype(int)
    return limpiar(df)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza y ingeniería de variables (se aplica igual en entrenamiento y en la app)."""
    df = df.drop(columns=[c for c in SENSIBLES if c in df.columns]).copy()
    df = df.drop_duplicates()
    df["cuota_mensual"] = df["monto"] / df["plazo_meses"].clip(lower=1)
    return df


def preprocesador() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ]
    )


def modelos_candidatos() -> dict[str, Pipeline]:
    base = {
        "Regresión logística": LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=2000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=SEMILLA,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
            random_state=SEMILLA,
        ),
        "Red neuronal (MLP)": MLPClassifier(
            hidden_layer_sizes=(16,), alpha=5.0, max_iter=3000,
            random_state=SEMILLA,
        ),
    }
    return {n: Pipeline([("prep", preprocesador()), ("clf", m)]) for n, m in base.items()}


def dividir(df: pd.DataFrame):
    X = df[NUMERICAS + CATEGORICAS]
    y = df[OBJETIVO]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMILLA)


def costo_esperado(y_true, prob, umbral, costo_fn=5.0, costo_fp=1.0) -> float:
    """Costo total: aprobar a un moroso (FN) vs. rechazar a un buen socio (FP).

    La matriz 5:1 es la recomendada por los autores del dataset Statlog.
    """
    pred = (np.asarray(prob) >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return costo_fn * fn + costo_fp * fp


def umbral_optimo(y_true, prob, costo_fn=5.0, costo_fp=1.0):
    umbrales = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costos = [costo_esperado(y_true, prob, u, costo_fn, costo_fp) for u in umbrales]
    i = int(np.argmin(costos))
    return float(umbrales[i]), umbrales, np.array(costos)


def metricas(y_true, prob, umbral) -> dict:
    pred = (np.asarray(prob) >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y_true, prob),
        "AUC-PR": average_precision_score(y_true, prob),
        "Recall (morosos detectados)": recall_score(y_true, pred),
        "Precisión": precision_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def evaluar_todos(df: pd.DataFrame):
    """Validación cruzada (5 folds) en entrenamiento + evaluación en test."""
    X_tr, X_te, y_tr, y_te = dividir(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    filas, ajustados, probs = [], {}, {}
    for nombre, pipe in modelos_candidatos().items():
        auc_cv = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        m = metricas(y_te, prob, 0.5)
        u_cv, _, _ = umbral_optimo(y_tr, prob_oof(modelos_candidatos()[nombre], X_tr, y_tr))
        filas.append({
            "Modelo": nombre,
            "AUC CV (media)": auc_cv.mean(),
            "AUC CV (desv.)": auc_cv.std(),
            "AUC test": m["AUC"],
            "AUC-PR test": m["AUC-PR"],
            "Recall@0.5": m["Recall (morosos detectados)"],
            "Costo test (umbral 0.5)": costo_esperado(y_te, prob, 0.5),
            "Umbral óptimo (CV)": u_cv,
            "Costo test (umbral óptimo)": costo_esperado(y_te, prob, u_cv),
        })
        ajustados[nombre] = pipe
        probs[nombre] = prob
    return pd.DataFrame(filas), ajustados, probs, (X_tr, X_te, y_tr, y_te)


def curva_roc(y_true, prob):
    fpr, tpr, _ = roc_curve(y_true, prob)
    return fpr, tpr


def contribuciones_logistica(pipe: Pipeline, x: pd.DataFrame) -> pd.Series:
    """Explicación local: aporte de cada variable al log-odds de mora (coef × valor).

    Se agrupan las columnas one-hot de vuelta a la variable original y se
    compara contra el socio promedio (valor 0 en escala estandarizada / la
    frecuencia media de cada categoría).
    """
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    nombres = prep.get_feature_names_out()
    z = prep.transform(x)
    z = z.toarray() if hasattr(z, "toarray") else np.asarray(z)
    media = getattr(pipe, "media_transformada_", np.zeros(z.shape[1]))
    aporte = (z[0] - media) * clf.coef_[0]
    grupos = {}
    for n, a in zip(nombres, aporte):
        tipo, resto = n.split("__", 1)
        var = resto if tipo == "num" else next(c for c in CATEGORICAS if resto.startswith(c + "_"))
        grupos[var] = grupos.get(var, 0.0) + a
    return pd.Series(grupos).sort_values(key=np.abs, ascending=False)


def prob_oof(pipe: Pipeline, X, y):
    """Probabilidades fuera de muestra (CV 5 folds) para calibrar el umbral sin sobreajuste."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def entrenar_modelo_final(df: pd.DataFrame) -> Pipeline:
    """Regresión logística en producción: competitiva en AUC y explicable para el comité de crédito."""
    X_tr, X_te, y_tr, y_te = dividir(df)
    pipe = modelos_candidatos()["Regresión logística"]
    pipe.fit(X_tr, y_tr)
    z = pipe.named_steps["prep"].transform(X_tr)
    z = z.toarray() if hasattr(z, "toarray") else np.asarray(z)
    pipe.media_transformada_ = z.mean(axis=0)
    return pipe
