"""Interfaz Streamlit del Agente Cotizador Metalmecánico (solo presentación)."""
import pandas as pd
import streamlit as st

from src import eda, visualizations as viz
from src.agent import ESTILOS, AgenteCotizador, materiales_disponibles
from src.cotizador import ACABADOS, CotizadorMetalmecanico, Pieza
from src.data_loader import ETIQUETAS, cargar_metales
from src.llm_client import MODELO_POR_DEFECTO, OllamaClient

st.set_page_config(page_title="Agente Cotizador Metalmecánico", page_icon="🔩", layout="wide")


@st.cache_data
def obtener_datos(archivo):
    return cargar_metales(archivo)


# ---------------- Barra lateral ----------------
st.sidebar.title("🔩 Configuración")
fuente = st.sidebar.radio("Datos", ["Dataset incluido (FMI)", "Subir CSV actualizado"])
archivo = st.sidebar.file_uploader("CSV con el mismo formato del FMI", type="csv") if fuente.startswith("Subir") else None

try:
    df = obtener_datos(archivo)
except ValueError as error:
    st.error(str(error))
    st.stop()

fechas = df["fecha"].dt.strftime("%Y-%m").tolist()
fecha_ref = st.sidebar.selectbox("Mes de referencia de precios", fechas[::-1], index=0)

st.sidebar.subheader("Ollama")
cliente = OllamaClient()
conectado, modelos, estado = cliente.estado()
if conectado:
    st.sidebar.success(estado)
    opciones = modelos or [MODELO_POR_DEFECTO]
    cliente.modelo = st.sidebar.selectbox("Modelo", opciones,
                                          index=opciones.index(MODELO_POR_DEFECTO) if MODELO_POR_DEFECTO in opciones else 0)
else:
    st.sidebar.warning("Ollama no está activo: el agente usará un resumen local de respaldo.")
    st.sidebar.caption("Inicie Ollama y ejecute: ollama pull llama3.2:3b")

st.sidebar.subheader("Parámetros del taller")
tarifa_maquina = st.sidebar.number_input("Tarifa hora máquina (USD)", 1.0, 200.0, 18.0)
tarifa_mo = st.sidebar.number_input("Tarifa hora mano de obra (USD)", 1.0, 100.0, 6.0)
margen = st.sidebar.slider("Margen de ganancia", 0.0, 0.8, 0.30, 0.05)
merma = st.sidebar.slider("Merma de material", 0.0, 0.3, 0.08, 0.01)

cotizador = CotizadorMetalmecanico(df, tarifa_maquina, tarifa_mo, merma, margen)
agente = AgenteCotizador(df, cotizador, cliente)

# ---------------- Contenido ----------------
st.title("🔩 Agente Cotizador Metalmecánico")
st.caption("Cotiza piezas con precios de mercado de metales (FMI) y un agente de IA local con Ollama.")

tab_datos, tab_eda, tab_riesgo, tab_cotizar, tab_agente = st.tabs(
    ["📋 Datos", "📈 Exploración", "⚠️ Volatilidad", "🧾 Cotizador", "🤖 Agente IA"]
)

with tab_datos:
    perfil = eda.perfil_dataset(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meses", perfil["filas"])
    c2.metric("Variables", perfil["columnas"])
    c3.metric("Periodo", f"{perfil['desde']} a {perfil['hasta']}")
    c4.metric("Valores nulos", perfil["nulos_totales"])
    st.subheader("Calidad de datos")
    st.dataframe(eda.tabla_calidad(df), use_container_width=True, hide_index=True)
    st.subheader("Resumen estadístico")
    st.dataframe(eda.resumen_estadistico(df), use_container_width=True)
    with st.expander("Ver datos"):
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab_eda:
    elegidos = st.multiselect("Materiales", eda.METALES[:-1], default=["aluminio", "cobre", "niquel", "zinc"],
                              format_func=lambda c: ETIQUETAS[c])
    if elegidos:
        base = st.toggle("Comparar en base 100", value=True)
        st.plotly_chart(viz.linea_precios(df, elegidos, base_100=base), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            periodo = st.radio("Segmentar por", ["anio", "decada"], horizontal=True,
                               format_func=lambda p: "Año" if p == "anio" else "Década")
            material = st.selectbox("Material a segmentar", elegidos, format_func=lambda c: ETIQUETAS[c])
            tabla = eda.promedio_por_periodo(df, [material], periodo)
            st.plotly_chart(viz.barras_periodo(tabla, periodo, material), use_container_width=True)
        with col_b:
            cambios = eda.variaciones_mensuales(df)
            st.plotly_chart(viz.histograma_variaciones(cambios, material), use_container_width=True)
    else:
        st.info("Seleccione al menos un material.")

with tab_riesgo:
    meses = st.slider("Ventana para medir volatilidad (meses)", 12, len(df) - 1, 36)
    tabla_vol = eda.volatilidad(df, meses=meses)
    tabla_vol["nivel"] = tabla_vol["volatilidad_%"].apply(eda.clasificar_volatilidad)
    col_a, col_b = st.columns(2)
    col_a.plotly_chart(viz.barras_volatilidad(tabla_vol), use_container_width=True)
    col_b.plotly_chart(viz.mapa_correlacion(eda.correlaciones(df)), use_container_width=True)
    st.dataframe(tabla_vol, use_container_width=True, hide_index=True)

with tab_cotizar:
    with st.form("form_cotizacion"):
        col_a, col_b, col_c = st.columns(3)
        descripcion = col_a.text_input("Pieza", "Soporte de bomba")
        material = col_a.selectbox("Material", materiales_disponibles())
        cantidad = col_b.number_input("Cantidad", 1, 10000, 20)
        peso = col_b.number_input("Peso por pieza (kg)", 0.1, 5000.0, 3.5)
        horas_maquina = col_c.number_input("Horas máquina por pieza", 0.0, 100.0, 0.5)
        horas_mo = col_c.number_input("Horas mano de obra por pieza", 0.0, 100.0, 0.8)
        acabado = col_a.selectbox("Acabado", list(ACABADOS.keys()))
        enviado = st.form_submit_button("Calcular cotización", type="primary")

    if enviado:
        pieza = Pieza(descripcion, material, peso, int(cantidad), horas_maquina, horas_mo, acabado)
        try:
            st.session_state["cotizacion"] = cotizador.calcular(pieza, fecha_ref)
            st.session_state["pieza"] = pieza
        except ValueError as error:
            st.error(str(error))

    cot = st.session_state.get("cotizacion")
    if cot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total con IVA", f"USD {cot.total:,.2f}")
        m2.metric("Precio unitario", f"USD {cot.total / cot.pieza.cantidad:,.2f}")
        m3.metric("Precio material", f"USD {cot.precio_kg:,.2f}/kg")
        m4.metric("Validez sugerida", f"{cot.validez_dias} días")
        col_a, col_b = st.columns(2)
        col_a.dataframe(cot.como_tabla(), use_container_width=True, hide_index=True)
        col_b.plotly_chart(viz.composicion_costos(cot.desglose), use_container_width=True)
        st.plotly_chart(viz.tendencia_con_pronostico(cot.mercado), use_container_width=True)
        st.caption(f"Material {cot.mercado['direccion']} ({cot.mercado['pendiente_%_mes']} % por mes, "
                   f"R² {cot.mercado['r2']}), volatilidad {cot.mercado['nivel_volatilidad']}.")

        st.subheader("✉️ Mensaje para el cliente (agente IA)")
        col_a, col_b = st.columns(2)
        nombre_cliente = col_a.text_input("Nombre del cliente", "Ing. Pérez")
        estilo = col_b.selectbox("Estilo de comunicación", list(ESTILOS.keys()))
        if st.button("Generar mensaje con IA"):
            with st.spinner("El agente está redactando..."):
                resultado = agente.mensaje_cliente(cot, nombre_cliente, estilo)
            st.success(f"Generado con: {resultado['origen']}")
            st.text_area("Mensaje listo para copiar", resultado["respuesta"], height=260)
            with st.expander("Contexto calculado que recibió el agente"):
                st.json(resultado["contexto"])

with tab_agente:
    st.write("Pregunte sobre el mercado de metales o la cotización actual. "
             "El agente elige funciones de Python, calcula y luego Ollama interpreta.")
    ejemplos = ["¿Conviene cotizar en aluminio o en acero inoxidable este mes?",
                "¿Qué metal es el más volátil y qué riesgo tiene para mis cotizaciones?",
                "¿El cobre y el aluminio se mueven juntos?"]
    pregunta = st.text_input("Pregunta", ejemplos[0])
    st.caption("Ejemplos: " + " · ".join(ejemplos))
    if st.button("Preguntar al agente", type="primary"):
        with st.spinner("Consultando..."):
            resultado = agente.responder(pregunta, fecha=fecha_ref, pieza=st.session_state.get("pieza"),
                                         cotizacion=st.session_state.get("cotizacion"))
        st.success(f"Respuesta generada con: {resultado['origen']}")
        st.markdown(resultado["respuesta"])
        st.caption("Herramientas usadas: " + ", ".join(resultado["herramientas"]))
        with st.expander("Contexto calculado por Python"):
            st.json(resultado["contexto"], expanded=False)
