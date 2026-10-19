# Agente Cotizador Metalmecánico · Streamlit + Ollama

Proyecto Final Integrador · Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos (UCG, C2026 P1) · Bryan Morales

Aplicación modular en Streamlit que cotiza piezas metalmecánicas usando el precio de mercado de los metales. Todos los cálculos (precio por kg, costos, margen, IVA, tendencia y volatilidad) los hace Python. Un agente de IA local con **Ollama** interpreta esos resultados, responde preguntas sobre el mercado y redacta el mensaje de WhatsApp para el cliente.

## Dataset

- **Nombre:** IMF Commodity Prices (precios mensuales de materias primas del Fondo Monetario Internacional), 1980-02 a 2017-06, 449 meses.
- **Fuente:** DataHub / GitHub: https://github.com/datasets/commodity-prices (archivo `data/commodity-prices.csv`). Datos originales: FMI.
- **Licencia:** Open Data Commons Public Domain Dedication and License (ODC-PDDL-1.0).
- **Variables usadas (USD por tonelada métrica):** aluminio, cobre, mineral de hierro (62 % Fe), níquel, zinc, estaño, plomo e índice de metales (2005 = 100).
- El archivo está incluido en `data/imf_commodity_prices.csv`. Desde la app se puede subir un CSV más reciente con las mismas columnas.

## Estructura

```text
cotizador_metalmecanico/
├── app.py                 # Interfaz Streamlit (solo presentación)
├── README.md
├── requirements.txt
├── .gitignore
├── run_app.bat            # Atajo para Windows
├── data/
│   └── imf_commodity_prices.csv
├── assets/
│   └── arquitectura.png
└── src/
    ├── __init__.py
    ├── data_loader.py     # Carga, validación y limpieza del CSV
    ├── eda.py             # Perfil, calidad, segmentación, volatilidad, correlación, tendencia
    ├── visualizations.py  # Gráficos Plotly
    ├── cotizador.py       # Clases Material, Pieza, Cotizacion y CotizadorMetalmecanico
    ├── llm_client.py      # Clase OllamaClient (API local de Ollama)
    └── agent.py           # Clase AgenteCotizador: elige herramientas, arma contexto, llama a Ollama
```

## Requisitos

- Python 3.10 o superior
- Ollama (https://ollama.com/download)

## 1. Crear y activar el entorno virtual

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Instalar dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Instalar Ollama y descargar el modelo

1. Instalar Ollama desde https://ollama.com/download y abrirlo (queda escuchando en `http://localhost:11434`).
2. Descargar el modelo recomendado:

```bash
ollama pull llama3.2:3b
```

En equipos con poca memoria: `ollama pull llama3.2:1b` y elegirlo en la barra lateral de la app.

## 4. Ejecutar

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. En Windows también puede usarse `run_app.bat`.

## Uso

1. **Datos:** perfil, calidad (nulos, tipos, rangos) y resumen estadístico.
2. **Exploración:** evolución de precios (en USD o base 100), promedio por año o década e histograma de variaciones mensuales.
3. **Volatilidad:** qué metal cambia más de precio y correlación entre metales.
4. **Cotizador:** ingresar pieza, material, peso, cantidad, horas y acabado. Muestra total con IVA (15 %), desglose, validez sugerida y tendencia del material. El botón *Generar mensaje con IA* redacta el WhatsApp para el cliente.
5. **Agente IA:** preguntas libres, por ejemplo "¿Conviene cotizar en aluminio o en acero inoxidable este mes?". La app muestra qué herramientas de Python usó el agente y el contexto que recibió Ollama.

## Supuestos del cotizador

- Aluminio y cobre: precio del metal (USD/t ÷ 1000) × factor de transformación (2,2 y 1,3) para pasar de lingote a plancha o perfil.
- Acero A36 e inoxidable 304: precio base (1,10 y 3,80 USD/kg, calibrado al promedio de 2016) ajustado por el índice del mineral de hierro o del níquel.
- Galvanizado: 7 % del peso en zinc. Pintura electrostática: 0,45 USD/kg.
- Validez: 7 días si la volatilidad es alta, 15 si es media y 30 si es baja. Si el material sube, se agrega un recargo del 1 al 5 % sobre el material.
- Tarifas, margen y merma se cambian en la barra lateral.

Son supuestos académicos, no precios reales de un proveedor.

## Limitaciones

- Los datos del FMI en este repositorio llegan a junio de 2017. Para cotizar hoy se debe subir un CSV actualizado.
- La proyección es una regresión lineal de 12 meses; si el R² es menor a 0,3 la app la marca como "sin tendencia clara".
- Si Ollama no está activo, la app sigue funcionando y el agente responde con un resumen local de respaldo.
- El modelo de Ollama no se guarda en el repositorio.
