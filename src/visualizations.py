"""Gráficos con Plotly. Cada función recibe datos ya calculados y devuelve una figura."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import ETIQUETAS


def linea_precios(df: pd.DataFrame, columnas: list[str], base_100: bool = False) -> go.Figure:
    """Evolución histórica; con base_100 todos parten de 100 para compararlos."""
    datos = df.set_index("fecha")[columnas]
    if base_100:
        datos = datos / datos.iloc[0] * 100
    largo = datos.reset_index().melt("fecha", var_name="material", value_name="valor")
    titulo = "Precios mensuales (base 100 = primer mes)" if base_100 else "Precios mensuales"
    fig = px.line(largo, x="fecha", y="valor", color="material", title=titulo)
    fig.update_layout(legend_title_text="", yaxis_title="Base 100" if base_100 else "USD/t")
    return fig


def barras_periodo(tabla: pd.DataFrame, periodo: str, columna: str) -> go.Figure:
    nombre = "año" if periodo == "anio" else "década"
    fig = px.bar(tabla, x=periodo, y=columna, title=f"Promedio de {ETIQUETAS.get(columna, columna)} por {nombre}")
    fig.update_layout(xaxis_title="", yaxis_title="USD/t")
    return fig


def histograma_variaciones(cambios: pd.DataFrame, columna: str) -> go.Figure:
    fig = px.histogram(cambios, x=columna, nbins=50, title=f"Distribución de la variación mensual: {columna}")
    fig.update_layout(xaxis_title="Variación mensual (%)", yaxis_title="Meses")
    return fig


def barras_volatilidad(tabla: pd.DataFrame) -> go.Figure:
    fig = px.bar(tabla, x="material", y="volatilidad_%", title="Volatilidad: desviación de la variación mensual (%)")
    fig.update_layout(xaxis_title="", yaxis_title="%")
    return fig


def mapa_correlacion(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlación entre variaciones mensuales")
    return fig


def tendencia_con_pronostico(info: dict) -> go.Figure:
    hist = info["historico"]
    futuro = pd.DataFrame(info["pronostico"])
    futuro["fecha"] = pd.to_datetime(futuro["fecha"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["fecha"], y=hist["precio"], mode="lines+markers", name="Histórico"))
    fig.add_trace(go.Scatter(
        x=[hist["fecha"].iloc[-1], *futuro["fecha"]],
        y=[hist["precio"].iloc[-1], *futuro["precio"]],
        mode="lines+markers", name="Proyección lineal", line={"dash": "dash"},
    ))
    fig.update_layout(title=f"{ETIQUETAS.get(info['material'], info['material'])}: últimos meses y proyección",
                      yaxis_title="USD/t")
    return fig


def composicion_costos(desglose: dict) -> go.Figure:
    datos = pd.DataFrame({"concepto": list(desglose.keys()), "usd": list(desglose.values())})
    datos = datos[datos["usd"] > 0]
    return px.pie(datos, names="concepto", values="usd", title="Composición del costo", hole=0.4)
