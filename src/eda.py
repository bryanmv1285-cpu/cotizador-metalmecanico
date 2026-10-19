"""Funciones de exploración y análisis (perfilado, calidad, tendencias)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_loader import COLUMNAS_METALES

METALES = list(COLUMNAS_METALES.values())


def perfil_dataset(df: pd.DataFrame) -> dict:
    """Dimensiones, rango de fechas y nulos totales."""
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "desde": df["fecha"].min().strftime("%Y-%m"),
        "hasta": df["fecha"].max().strftime("%Y-%m"),
        "nulos_totales": int(df[METALES].isna().sum().sum()),
    }


def tabla_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Tipo, nulos, % nulos, mínimo y máximo por variable."""
    filas = []
    for col in METALES:
        serie = df[col]
        filas.append({
            "variable": col,
            "tipo": str(serie.dtype),
            "nulos": int(serie.isna().sum()),
            "% nulos": round(100 * serie.isna().mean(), 2),
            "mínimo": round(serie.min(), 2),
            "máximo": round(serie.max(), 2),
        })
    return pd.DataFrame(filas)


def resumen_estadistico(df: pd.DataFrame) -> pd.DataFrame:
    return df[METALES].describe().T.round(2)


def promedio_por_periodo(df: pd.DataFrame, columnas: list[str], periodo: str = "anio") -> pd.DataFrame:
    """Segmentación: promedio por año o por década."""
    datos = df.copy()
    if periodo == "decada":
        datos["decada"] = (datos["anio"] // 10) * 10
    return datos.groupby(periodo)[columnas].mean().round(2).reset_index()


def variaciones_mensuales(df: pd.DataFrame) -> pd.DataFrame:
    """Cambio porcentual mes a mes de cada metal."""
    return df.set_index("fecha")[METALES].pct_change().dropna() * 100


def volatilidad(df: pd.DataFrame, meses: int | None = None) -> pd.DataFrame:
    """Desviación estándar de la variación mensual (%). Más alta = precio más inestable."""
    cambios = variaciones_mensuales(df)
    if meses:
        cambios = cambios.tail(meses)
    tabla = cambios.std().round(2).rename("volatilidad_%").reset_index()
    return tabla.rename(columns={"index": "material"}).sort_values("volatilidad_%", ascending=False)


def correlaciones(df: pd.DataFrame) -> pd.DataFrame:
    """Correlación de las variaciones mensuales (evita la correlación falsa por tendencia)."""
    return variaciones_mensuales(df).corr().round(2)


def clasificar_volatilidad(valor: float) -> str:
    if valor >= 7:
        return "alta"
    if valor >= 4:
        return "media"
    return "baja"


def tendencia(df: pd.DataFrame, columna: str, hasta=None, meses: int = 12, proyeccion: int = 3) -> dict:
    """Regresión lineal sobre los últimos `meses` y proyección a `proyeccion` meses."""
    datos = df if hasta is None else df[df["fecha"] <= pd.Timestamp(hasta)]
    ventana = datos[["fecha", columna]].dropna().tail(meses)
    if len(ventana) < 3:
        raise ValueError("No hay suficientes meses para calcular la tendencia.")

    x = np.arange(len(ventana)).reshape(-1, 1)
    y = ventana[columna].to_numpy()
    modelo = LinearRegression().fit(x, y)
    ultimo = float(y[-1])
    # Proyección desde el último precio real, al ritmo de la pendiente estimada.
    pronostico = [ultimo + float(modelo.coef_[0]) * paso for paso in range(1, proyeccion + 1)]

    pendiente_pct = float(modelo.coef_[0] / ultimo * 100)
    cambios = ventana[columna].pct_change().dropna() * 100
    vol = float(cambios.std())

    r2 = float(modelo.score(x, y))
    if r2 < 0.3:
        direccion = "sin tendencia clara"
    elif pendiente_pct > 0.5:
        direccion = "subiendo"
    elif pendiente_pct < -0.5:
        direccion = "bajando"
    else:
        direccion = "estable"

    fechas_futuras = pd.date_range(ventana["fecha"].iloc[-1], periods=proyeccion + 1, freq="MS")[1:]
    return {
        "material": columna,
        "fecha_ref": ventana["fecha"].iloc[-1].strftime("%Y-%m"),
        "precio_actual": round(ultimo, 2),
        "pendiente_%_mes": round(pendiente_pct, 2),
        "direccion": direccion,
        "r2": round(r2, 3),
        "volatilidad_%": round(vol, 2),
        "nivel_volatilidad": clasificar_volatilidad(vol),
        "pronostico": [
            {"fecha": f.strftime("%Y-%m"), "precio": round(float(p), 2)}
            for f, p in zip(fechas_futuras, pronostico)
        ],
        "historico": ventana.rename(columns={columna: "precio"}),
    }
