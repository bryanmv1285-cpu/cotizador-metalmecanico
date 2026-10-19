"""Carga y validación del dataset de precios de metales (FMI)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RUTA_DATOS = Path(__file__).resolve().parent.parent / "data" / "imf_commodity_prices.csv"

# Columnas originales del FMI -> nombres cortos en español usados en la app.
COLUMNAS_METALES = {
    "Aluminum": "aluminio",
    "Copper": "cobre",
    "China import Iron Ore Fines 62% FE spot": "mineral_hierro",
    "Nickel": "niquel",
    "Zinc": "zinc",
    "Tin": "estano",
    "Lead": "plomo",
    "Metals Price Index": "indice_metales",
}

ETIQUETAS = {
    "aluminio": "Aluminio (USD/t)",
    "cobre": "Cobre (USD/t)",
    "mineral_hierro": "Mineral de hierro (USD/t)",
    "niquel": "Níquel (USD/t)",
    "zinc": "Zinc (USD/t)",
    "estano": "Estaño (USD/t)",
    "plomo": "Plomo (USD/t)",
    "indice_metales": "Índice de metales (2005=100)",
}


def cargar_csv(origen=None) -> pd.DataFrame:
    """Lee el CSV incluido o un archivo subido desde Streamlit."""
    fuente = RUTA_DATOS if origen is None else origen
    return pd.read_csv(fuente)


def validar_columnas(df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de columnas obligatorias que faltan."""
    requeridas = ["Date", *COLUMNAS_METALES.keys()]
    return [c for c in requeridas if c not in df.columns]


def preparar_metales(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona los metales, convierte tipos y ordena por fecha."""
    faltantes = validar_columnas(df)
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {', '.join(faltantes)}")

    datos = df[["Date", *COLUMNAS_METALES.keys()]].rename(columns=COLUMNAS_METALES)
    datos = datos.rename(columns={"Date": "fecha"})
    datos["fecha"] = pd.to_datetime(datos["fecha"], errors="coerce")
    for col in COLUMNAS_METALES.values():
        datos[col] = pd.to_numeric(datos[col], errors="coerce")

    datos = datos.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
    datos["anio"] = datos["fecha"].dt.year
    return datos


def cargar_metales(origen=None) -> pd.DataFrame:
    """Atajo: carga y prepara en un solo paso."""
    return preparar_metales(cargar_csv(origen))
