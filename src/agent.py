"""Agente cotizador: elige herramientas de Python, arma contexto y pide a Ollama que interprete."""
from __future__ import annotations

import json
import unicodedata

import pandas as pd

from src import eda
from src.cotizador import MATERIALES, CotizadorMetalmecanico, Cotizacion, Pieza
from src.llm_client import OllamaClient

SISTEMA = (
    "Eres el asistente comercial de un taller metalmecánico en Ecuador. "
    "Respondes en español, claro y breve. Usa SOLO los números del CONTEXTO, que fueron calculados con Python; "
    "no inventes precios, materiales ni fechas. Si algo no está en el contexto, dilo. "
    "Los datos de mercado son históricos del FMI y la proyección es una regresión lineal simple: "
    "preséntala como referencia, no como garantía."
)

ESTILOS = {
    "Directo y breve": "Mensaje corto, máximo 5 líneas, con el precio y la validez al inicio.",
    "Detallado y técnico": "Explica el desglose de costos y el comportamiento del material.",
    "Cercano y amigable": "Tono cálido y cordial, como un vendedor de confianza, sin perder el precio exacto.",
}

PALABRAS_MATERIAL = {
    "acero": "mineral_hierro", "hierro": "mineral_hierro", "inox": "niquel", "niquel": "niquel",
    "aluminio": "aluminio", "cobre": "cobre", "zinc": "zinc", "galvaniz": "zinc",
    "estano": "estano", "plomo": "plomo",
}


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sin_tildes if not unicodedata.combining(c))


class AgenteCotizador:
    """Agente sencillo con herramientas autorizadas (funciones de Python)."""

    def __init__(self, df: pd.DataFrame, cotizador: CotizadorMetalmecanico, cliente: OllamaClient):
        self.df = df
        self.cotizador = cotizador
        self.cliente = cliente
        self.herramientas = {
            "perfil_dataset": lambda **_: eda.perfil_dataset(df),
            "volatilidad": lambda **_: eda.volatilidad(df, meses=36).to_dict("records"),
            "correlaciones": lambda **_: eda.correlaciones(df).to_dict(),
            "tendencia_material": self._tendencia,
            "comparar_materiales": self._comparar,
        }

    # ---- herramientas -------------------------------------------------
    def _tendencia(self, columna: str, fecha=None, **_) -> dict:
        info = eda.tendencia(self.df, columna, hasta=fecha)
        info.pop("historico")
        return info

    def _comparar(self, pieza: Pieza | None = None, fecha=None, **_) -> list[dict]:
        if pieza is None:
            return []
        return self.cotizador.comparar_materiales(pieza, fecha).to_dict("records")

    # ---- planificación ------------------------------------------------
    def elegir_herramientas(self, pregunta: str, hay_pieza: bool) -> list[tuple[str, dict]]:
        """Reglas simples (if/elif) para decidir qué funciones ejecutar."""
        texto = _normalizar(pregunta)
        plan: list[tuple[str, dict]] = []
        materiales = {col for palabra, col in PALABRAS_MATERIAL.items() if palabra in texto}
        for col in sorted(materiales):
            plan.append(("tendencia_material", {"columna": col}))
        if any(p in texto for p in ("volatil", "riesgo", "inestable", "estable")):
            plan.append(("volatilidad", {}))
        if any(p in texto for p in ("correl", "relacion", "se mueven juntos")):
            plan.append(("correlaciones", {}))
        if hay_pieza and any(p in texto for p in ("compar", "conviene", "mas barato", "alternativa", "otro material")):
            plan.append(("comparar_materiales", {}))
        if not plan:
            plan.append(("perfil_dataset", {}))
            plan.append(("volatilidad", {}))
        return plan

    def ejecutar(self, plan: list[tuple[str, dict]], fecha=None, pieza: Pieza | None = None) -> dict:
        resultados = {}
        for nombre, argumentos in plan:
            funcion = self.herramientas[nombre]
            clave = f"{nombre}:{argumentos.get('columna', '')}".rstrip(":")
            resultados[clave] = funcion(fecha=fecha, pieza=pieza, **argumentos)
        return resultados

    # ---- respuestas ---------------------------------------------------
    def responder(self, pregunta: str, fecha=None, pieza: Pieza | None = None,
                  cotizacion: Cotizacion | None = None) -> dict:
        plan = self.elegir_herramientas(pregunta, hay_pieza=pieza is not None)
        contexto = self.ejecutar(plan, fecha=fecha, pieza=pieza)
        if cotizacion is not None:
            contexto["cotizacion_actual"] = cotizacion.resumen()
        mensaje = f"CONTEXTO:\n{json.dumps(contexto, ensure_ascii=False, default=str)}\n\nPREGUNTA:\n{pregunta}"
        texto, origen = self._llamar_llm(mensaje, respaldo=lambda: self._respaldo_pregunta(contexto))
        return {"respuesta": texto, "herramientas": [n for n, _ in plan], "contexto": contexto, "origen": origen}

    def mensaje_cliente(self, cotizacion: Cotizacion, cliente: str, estilo: str) -> dict:
        contexto = cotizacion.resumen()
        instruccion = (
            f"Redacta un mensaje de WhatsApp para el cliente {cliente} con esta cotización. "
            f"Estilo: {ESTILOS[estilo]} Incluye el total con IVA, el precio unitario y la validez en días. "
            "Después, en 2 líneas aparte tituladas 'Nota interna para el vendedor', explica si conviene "
            "cerrar rápido según la tendencia y volatilidad del material."
        )
        mensaje = f"CONTEXTO:\n{json.dumps(contexto, ensure_ascii=False, default=str)}\n\nTAREA:\n{instruccion}"
        texto, origen = self._llamar_llm(mensaje, respaldo=lambda: self._respaldo_cotizacion(cotizacion, cliente))
        return {"respuesta": texto, "contexto": contexto, "origen": origen}

    def _llamar_llm(self, mensaje: str, respaldo) -> tuple[str, str]:
        conectado, _, _ = self.cliente.estado()
        if not conectado:
            return respaldo(), "respaldo local (Ollama no conectado)"
        try:
            return self.cliente.chat(SISTEMA, mensaje), f"Ollama · {self.cliente.modelo}"
        except Exception as exc:  # noqa: BLE001
            return respaldo() + f"\n\n(Ollama devolvió un error: {exc})", "respaldo local"

    # ---- respaldo sin LLM ---------------------------------------------
    @staticmethod
    def _respaldo_cotizacion(cot: Cotizacion, cliente: str) -> str:
        r = cot.resumen()
        m = r["mercado_material"]
        return (
            f"Hola {cliente}, le envío la cotización de {r['cantidad']} u. de {r['pieza']} en {r['material']}: "
            f"total USD {r['total_con_iva_usd']:,.2f} con IVA (USD {r['precio_unitario_con_iva_usd']:,.2f} c/u). "
            f"Oferta válida por {r['validez_oferta_dias']} días.\n\n"
            f"Nota interna: el material está {m['direccion']} ({m['pendiente_%_mes']} % al mes) "
            f"con volatilidad {m['nivel_volatilidad']}."
        )

    @staticmethod
    def _respaldo_pregunta(contexto: dict) -> str:
        partes = []
        for clave, valor in contexto.items():
            if clave.startswith("tendencia_material"):
                partes.append(f"{valor['material']}: {valor['direccion']} ({valor['pendiente_%_mes']} %/mes), "
                              f"volatilidad {valor['nivel_volatilidad']}.")
            elif clave == "volatilidad":
                top = valor[0]
                partes.append(f"Mayor volatilidad (36 meses): {top['material']} con {top['volatilidad_%']} %.")
            elif clave == "perfil_dataset":
                partes.append(f"Dataset con {valor['filas']} meses ({valor['desde']} a {valor['hasta']}).")
        return " ".join(partes) or "Sin datos para responder."


def materiales_disponibles() -> list[str]:
    return list(MATERIALES.keys())
