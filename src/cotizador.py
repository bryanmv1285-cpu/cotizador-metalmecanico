"""Lógica de cotización (cálculos 100 % en Python; el LLM solo interpreta)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.eda import tendencia

IVA = 0.15  # IVA vigente en Ecuador
ANIO_CALIBRACION = 2016  # año al que se calibran los precios base indexados


@dataclass
class Material:
    nombre: str
    columna: str           # columna del dataset que mueve su precio
    modo: str              # "directo": precio del metal x factor | "indexado": precio base x índice
    factor: float = 1.0    # de lingote/mineral a plancha o perfil comercial
    precio_base_kg: float = 0.0


MATERIALES = {
    "Acero al carbono A36": Material("Acero al carbono A36", "mineral_hierro", "indexado", precio_base_kg=1.10),
    "Acero inoxidable 304": Material("Acero inoxidable 304", "niquel", "indexado", precio_base_kg=3.80),
    "Aluminio 6061": Material("Aluminio 6061", "aluminio", "directo", factor=2.2),
    "Cobre electrolítico": Material("Cobre electrolítico", "cobre", "directo", factor=1.3),
}

ACABADOS = {"Sin acabado": 0.0, "Pintura electrostática": 0.45, "Galvanizado en caliente": None}


@dataclass
class Pieza:
    descripcion: str
    material: str
    peso_kg: float
    cantidad: int = 1
    horas_maquina: float = 0.0
    horas_mano_obra: float = 0.0
    acabado: str = "Sin acabado"


@dataclass
class Cotizacion:
    pieza: Pieza
    fecha_ref: str
    precio_kg: float
    desglose: dict
    subtotal: float
    margen_usd: float
    recargo_riesgo_usd: float
    total_sin_iva: float
    iva_usd: float
    total: float
    validez_dias: int
    mercado: dict = field(default_factory=dict)

    def como_tabla(self) -> pd.DataFrame:
        filas = [(k, v) for k, v in self.desglose.items()]
        filas += [
            ("Margen", self.margen_usd),
            ("Recargo por riesgo de precio", self.recargo_riesgo_usd),
            ("Total sin IVA", self.total_sin_iva),
            (f"IVA {int(IVA * 100)} %", self.iva_usd),
            ("TOTAL", self.total),
        ]
        return pd.DataFrame(filas, columns=["Concepto", "USD"]).round(2)

    def resumen(self) -> dict:
        """Versión compacta para enviarla como contexto al agente."""
        return {
            "pieza": self.pieza.descripcion,
            "material": self.pieza.material,
            "cantidad": self.pieza.cantidad,
            "peso_unitario_kg": self.pieza.peso_kg,
            "fecha_referencia_precios": self.fecha_ref,
            "precio_material_usd_kg": round(self.precio_kg, 3),
            "desglose_usd": {k: round(v, 2) for k, v in self.desglose.items()},
            "margen_usd": round(self.margen_usd, 2),
            "recargo_riesgo_usd": round(self.recargo_riesgo_usd, 2),
            "total_sin_iva_usd": round(self.total_sin_iva, 2),
            "total_con_iva_usd": round(self.total, 2),
            "precio_unitario_con_iva_usd": round(self.total / self.pieza.cantidad, 2),
            "validez_oferta_dias": self.validez_dias,
            "mercado_material": {k: v for k, v in self.mercado.items() if k != "historico"},
        }


class CotizadorMetalmecanico:
    """Calcula el precio de una pieza a partir del precio de mercado del material."""

    def __init__(self, df: pd.DataFrame, tarifa_maquina: float = 18.0, tarifa_mano_obra: float = 6.0,
                 merma: float = 0.08, margen: float = 0.30):
        self.df = df
        self.tarifa_maquina = tarifa_maquina
        self.tarifa_mano_obra = tarifa_mano_obra
        self.merma = merma
        self.margen = margen

    def _valor_en(self, columna: str, fecha) -> float:
        datos = self.df[self.df["fecha"] <= pd.Timestamp(fecha)]
        return float(datos[columna].dropna().iloc[-1])

    def precio_kg(self, material: Material, fecha) -> float:
        valor = self._valor_en(material.columna, fecha)
        if material.modo == "directo":
            return valor / 1000 * material.factor
        base = self.df.loc[self.df["anio"] == ANIO_CALIBRACION, material.columna].mean()
        return material.precio_base_kg * valor / base

    def costo_acabado_kg(self, acabado: str, fecha) -> float:
        costo = ACABADOS.get(acabado, 0.0)
        if costo is None:  # galvanizado: depende del precio del zinc (aprox. 7 % del peso en zinc)
            zinc_kg = self._valor_en("zinc", fecha) / 1000
            return 0.07 * zinc_kg * 1.5 + 0.10
        return costo

    @staticmethod
    def dias_validez(nivel_volatilidad: str) -> int:
        return {"alta": 7, "media": 15}.get(nivel_volatilidad, 30)

    @staticmethod
    def recargo_por_riesgo(mercado: dict) -> float:
        """Si el precio sube y es volátil, se protege la oferta con un recargo pequeño."""
        if mercado["direccion"] != "subiendo":
            return 0.0
        return {"alta": 0.05, "media": 0.03}.get(mercado["nivel_volatilidad"], 0.01)

    def calcular(self, pieza: Pieza, fecha) -> Cotizacion:
        if pieza.material not in MATERIALES:
            raise ValueError(f"Material no disponible: {pieza.material}")
        if pieza.peso_kg <= 0 or pieza.cantidad <= 0:
            raise ValueError("El peso y la cantidad deben ser mayores que cero.")

        material = MATERIALES[pieza.material]
        mercado = tendencia(self.df, material.columna, hasta=fecha)
        precio = self.precio_kg(material, fecha)
        kilos = pieza.peso_kg * pieza.cantidad

        desglose = {
            "Material (incluye merma)": kilos * (1 + self.merma) * precio,
            "Horas máquina": pieza.horas_maquina * pieza.cantidad * self.tarifa_maquina,
            "Mano de obra": pieza.horas_mano_obra * pieza.cantidad * self.tarifa_mano_obra,
            "Acabado": kilos * self.costo_acabado_kg(pieza.acabado, fecha),
        }
        subtotal = sum(desglose.values())
        margen_usd = subtotal * self.margen
        recargo = desglose["Material (incluye merma)"] * self.recargo_por_riesgo(mercado)
        sin_iva = subtotal + margen_usd + recargo
        iva = sin_iva * IVA

        return Cotizacion(
            pieza=pieza, fecha_ref=mercado["fecha_ref"], precio_kg=precio, desglose=desglose,
            subtotal=subtotal, margen_usd=margen_usd, recargo_riesgo_usd=recargo,
            total_sin_iva=sin_iva, iva_usd=iva, total=sin_iva + iva,
            validez_dias=self.dias_validez(mercado["nivel_volatilidad"]), mercado=mercado,
        )

    def comparar_materiales(self, pieza: Pieza, fecha) -> pd.DataFrame:
        """Misma pieza cotizada con cada material (función reutilizada por el agente)."""
        filas = []
        for nombre in MATERIALES:
            variante = Pieza(**{**pieza.__dict__, "material": nombre})
            cot = self.calcular(variante, fecha)
            filas.append({"material": nombre, "precio_kg": round(cot.precio_kg, 2),
                          "total_con_iva": round(cot.total, 2), "validez_dias": cot.validez_dias,
                          "tendencia": cot.mercado["direccion"]})
        return pd.DataFrame(filas).sort_values("total_con_iva")
