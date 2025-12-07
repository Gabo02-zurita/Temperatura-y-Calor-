import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math # Necesario para seno, coseno y sqrt

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Física Universitaria | Energía y Calor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- TÍTULO PRINCIPAL ---
st.title("👨‍🔬 FÍSICA UNIVERSITARIA | Simulaciones Interactivas")
st.markdown("Herramienta para visualizar conceptos clave de Energía, Calor y Conducción.")

# --- BARRA LATERAL (Sidebar) ---
st.sidebar.header("Parámetros Globales y Documentación")
st.sidebar.info("Esta aplicación utiliza módulos que integran conceptos de Termodinámica (Calor, Temperatura) y Mecánica (Conservación de la Energía).")

# --- FUNCIONES DE LOS MÓDULOS ---

# 1. Función para la conversión de escalas
def modulo_conversion():
    st.header("1. Conversión de Escalas Termométricas")
    st.markdown("Convierte entre Celsius (°C), Fahrenheit (°F) y Kelvin (K).")

    col1, col2 = st.columns(2)
    with col1:
        unidad_entrada = st.selectbox("Selecciona la unidad de entrada", ["Celsius", "Fahrenheit", "Kelvin"])
        valor_entrada = st.number_input(f"Ingresa el valor en {unidad_entrada}", value=20.0, step=0.1)

    # Conversiones
    if unidad_entrada == "Celsius":
        C = valor_entrada
        F = (C * 9/5) + 32
        K = C + 273.15
    elif unidad_entrada == "Fahrenheit":
        F = valor_entrada
        C = (F - 32) * 5/9
        K = C + 273.15
    else: # Kelvin
        K = valor_entrada
        C = K - 273.15
        F = (C * 9/5) + 32

    st.subheader("Resultados:")
    st.metric("Celsius (°C)", f"{C:.2f}")
    st.metric("Fahrenheit (°F)", f"{F:.2f}")
    st.metric("Kelvin (K)", f"{K:.2f}")

# 2. Función para el equilibrio térmico
def modulo_equilibrio():
    st.header("2. Equilibrio Térmico (Calorimetría)")
    st.markdown("Calcula la temperatura de equilibrio ($T_f$) de dos cuerpos usando $Q_{ganado} + Q_{perdido} = 0$.")

    st.subheader("Cuerpo A (Ganador de calor)")
    mA = st.number_input("Masa A ($m_A$ en kg)", value=1.0, min_value=0.1, key="mA")
    cA = st.number_input("Calor Específico A ($c_A$ en J/kg·K)", value=4186.0, min_value=1.0, key="cA") # Agua
    TiA = st.number_input("Temperatura Inicial A ($T_{iA}$ en °C)", value=20.0, key="TiA")

    st.subheader("Cuerpo B (Perdedor de calor)")
    mB = st.number_input("Masa B ($m_B$ en kg)", value=0.5, min_value=0.1, key="mB")
    cB = st.number_input("Calor Específico B ($c_B$ en J/kg·K)", value=900.0, min_value=1.0, key="cB") # Aluminio
    TiB = st.number_input("Temperatura Inicial B ($T_{iB}$ en °C)", value=80.0, key="TiB")

    if TiA >= TiB:
        st.error("Para el cálculo, el Cuerpo B (Perdedor) debe tener una temperatura inicial mayor a la del Cuerpo A (Ganador).")
        return

    if st.button("Calcular Temperatura de Equilibrio"):
        # Fórmula de Equilibrio Térmico (Despejando Tf de m_A*c_A*(Tf-TiA) + m_B*c_B*(Tf-TiB) = 0)
        numerador = (mA * cA * TiA) + (mB * cB * TiB)
        denominador = (mA * cA) + (mB * cB)
        Tf = numerador / denominador

        st.success(f"La Temperatura Final de Equilibrio ($T_f$) es: **{Tf:.2f} °C**")

        st.subheader("Detalles de la Transferencia (Q)")
        QA = mA * cA * (Tf - TiA)
        QB = mB * cB * (Tf - TiB)

        st.metric("Calor ganado por A ($Q_A$)", f"{QA:.2f} J")
        st.metric("Calor perdido por B ($Q_B$)", f"{QB:.2f} J")
        st.markdown(f"Verificación: $Q_A + Q_B = {QA + QB:.2f} J$ (Debe ser cercano a cero)")

# 3. Función para la conducción de calor (Ley de Fourier)
def modulo_conduccion():
    st.header("3. Conducción de Calor (Ley de Fourier)")
    st.markdown("Calcula la tasa de flujo de calor ($H$) a través de una pared y genera una gráfica del perfil de temperatura lineal.")

    col1, col2 = st.columns(2)
    with col1:
        A = st.number_input("Área de la pared ($A$ en $m^2$)", value=1.0, min_value=0.1, key="A")
        L = st.number_input("Grosor de la pared ($L$ en m)", value=0.1, min_value=0.01, key="L")
        k = st.number_input("Conductividad Térmica ($k$ en $W/m·K$)", value=0.8, min_value=0.01, key="k") # Ladrillo/Concreto

    with col2:
        TH = st.number_input("Temperatura Lado Caliente ($T_H$ en °C)", value=30.0, key="TH")
        TC = st.number_input("Temperatura Lado Frío ($T_C$ en °C)", value=10.0, key="TC")

    if st.button("Calcular Tasa de Flujo y Graficar"):
        if TH <= TC:
            st.error("La temperatura del lado caliente ($T_H$) debe ser mayor a la del lado frío ($T_C$).")
            return

        # Ley de Fourier: H = k * A * (TH - TC) / L
        H = k * A * (TH - TC) / L

        st.success(f"La Tasa de Flujo de Calor ($H$) es: **{H:.2f} W** (Joules por segundo)")

        # Gráfica del perfil de temperatura
        x = np.linspace(0, L, 100)
        # La temperatura T(x) es: T(x) = TH - (TH - TC) * (x / L)
        T_x = TH - ((TH - TC) * (x / L))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=T_x, mode='lines', name='Perfil de Temperatura',
                                 line=dict(color='red', width=3)))
        fig.update_layout(
            title='Perfil de Temperatura a Través de la Pared',
            xaxis_title='Posición (x) en m',
            yaxis_title='Temperatura (T) en °C',
            yaxis_range=[TC - 5, TH + 5]
        )
        st.plotly_chart(fig)

# 4. Función para la conservación de energía (NUEVO MÓDULO CAPÍTULO 7)
def modulo_conservacion_energia():
    st.header("4. Conservación de Energía y Disipación Térmica (Capítulo 7)")
    st.markdown("Simula un bloque deslizándose por un plano inclinado con fricción. La energía perdida por fricción se convierte en Energía Térmica ($Q$).")
    
    # 
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros del Sistema Mecánico")
        m = st.number_input("Masa del bloque ($m$ en kg)", value=2.0, min_value=0.1, key="m_cons")
        h_inicial = st.number_input("Altura inicial ($h_{i}$ en m)", value=5.0, min_value=0.1, key="h_cons")
        theta = st.slider("Ángulo del plano ($\theta$ en grados)", value=30, min_value=1, max_value=89, key="theta_cons")
        mu_k = st.slider("Coeficiente de Fricción Cinética ($\mu_k$)", value=0.2, min_value=0.0, max_value=1.0, step=0.05, key="mu_k_cons")

    # Constantes
    g = 9.81  # Aceleración de la gravedad (m/s²)
    theta_rad = math.radians(theta)
    
    # 1. Distancia recorrida a lo largo del plano (d)
    # Suponemos que la altura h es la altura vertical total.
    d = h_inicial / math.sin(theta_rad)
    
    # 2. Fuerzas y Trabajo
    W_g = m * g # Peso
    N = W_g * math.cos(theta_rad) # Fuerza Normal
    f_k = mu_k * N # Fuerza de Fricción
    
    # 3. Energías Iniciales (El sistema parte del reposo: v_i = 0)
    K_i = 0.0
    U_i = m * g * h_inicial
    E_mecanica_i = K_i + U_i
    
    # 4. Trabajo de Fricción (Energía Disipada)
    W_nc = -f_k * d
    Q_termica = abs(W_nc)
    
    # 5. Energías Finales (Al llegar a la base: h_f = 0)
    U_f = 0.0
    
    # 6. Conservación de Energía: K_f = E_mecanica_i + W_nc (Trabajo total no conservativo)
    K_f = E_mecanica_i + W_nc
    E_mecanica_f = K_f + U_f
    
    # 7. Velocidad Final (Si K_f > 0)
    if K_f < 0:
        # Esto ocurre si la fricción es tan alta que el bloque se detiene antes de llegar a la base
        v_f = 0.0
        K_f = 0.0
        st.warning("⚠️ **ATENCIÓN:** La fuerza de fricción es demasiado alta. El bloque se detiene antes de llegar a la base del plano. La energía cinética final se considera 0.")
    else:
        v_f = math.sqrt((2 * K_f) / m)

    with col2:
        st.subheader("Resultados del Análisis de Energía")
        st.metric("Energía Mecánica Inicial ($E_i = U_i + K_i$)", f"{E_mecanica_i:.2f} J")
        st.metric("Trabajo realizado por Fricción ($W_{nc}$)", f"{W_nc:.2f} J")
        st.metric("Energía Térmica Disipada ($Q = |W_{nc}|$)", f"{Q_termica:.2f} J")
        st.metric("Energía Mecánica Final ($E_f = K_f$)", f"{E_mecanica_f:.2f} J")
        st.metric("Velocidad Final ($v_f$)", f"{v_f:.2f} m/s")

    st.subheader("Gráfica de la Transformación de Energía")
    
    etiquetas = ['Energía Potencial Inicial ($U_i$)', 'Energía Cinética Final ($K_f$)', 'Energía Disipada (Q)', 'Energía Mecánica Final ($E_f$)']
    valores = [U_i, K_f, Q_termica, E_mecanica_f]
    colores = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']

    fig = go.Figure(data=[
        go.Bar(
            x=etiquetas[:3], 
            y=[U_i, K_f, Q_termica], 
            marker_color=colores[:3],
            name='Transferencia de Energía'
        )
    ])
    
    fig.add_trace(go.Scatter(
        x=[etiquetas[0], etiquetas[3]], 
        y=[E_mecanica_i, E_mecanica_f], 
        mode='lines+markers', 
        name='Conservación Total',
        line=dict(color='#000000', dash='dot')
    ))

    fig.update_layout(
        title='Distribución de la Energía',
        yaxis_title='Energía (Joules, J)',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Explicación del Concepto (Capítulo 7):**
    
    El principio de conservación de la energía no mecánica establece que la energía total del sistema (Mecánica + Térmica) es constante.
    
    * **Energía Mecánica Inicial ($E_i$):** $E_i = U_i = mgh_{i}$ (El bloque inicia en reposo).
    * **Trabajo No Conservativo ($W_{nc}$):** El trabajo realizado por la fuerza de fricción ($W_{nc} = -\mu_k N d$) se "roba" energía del sistema mecánico.
    * **Energía Térmica ($Q$):** El valor absoluto de $W_{nc}$ es la energía que se convierte en calor.
    * **Energía Mecánica Final ($E_f$):** La energía mecánica final es $E_f = E_i + W_{nc}$ (que, al final del plano, es puramente $K_f$).
    """)

# --- ESTRUCTURA PRINCIPAL DE STREAMLIT (Tabs) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Conversión de Escalas", 
    "2. Equilibrio Térmico", 
    "3. Conducción de Calor", 
    "4. Conservación de Energía (Cap. 7)"
])

with tab1:
    modulo_conversion()

with tab2:
    modulo_equilibrio()

with tab3:
    modulo_conduccion()

with tab4:
    modulo_conservacion_energia()
