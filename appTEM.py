import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Termodinámica Interactiva",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DATOS FÍSICOS (Valores típicos para las simulaciones) ---
C_AGUA = 4186  # J/(kg*K)
L_FUSION_HIELO = 334000  # J/kg (Calor latente de fusión del agua)
L_VAPORIZACION_AGUA = 2260000  # J/kg
T_FUSION_AGUA = 0 + 273.15  # K
T_EBULLICION_AGUA = 100 + 273.15 # K

# Calores específicos (J/kg*K)
CALORES_ESPECIFICOS = {
    "Agua (Líquida)": C_AGUA,
    "Aluminio": 900,
    "Cobre": 385,
    "Hierro": 450,
    "Hielo": 2090
}

# --- FUNCIÓN 1: Conversión de Escalas ---

def modulo_conversion():
    """Módulo para la conversión dinámica de escalas de temperatura."""
    st.header("1️⃣ Conversión Dinámica de Escalas")
    st.markdown("Convierte la temperatura entre las escalas **Celsius (°C)**, **Kelvin (K)**, **Fahrenheit (°F)** y **Rankine (°R)**.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        unidad_entrada = st.selectbox(
            "Selecciona la Escala de Entrada",
            ("Celsius (°C)", "Kelvin (K)", "Fahrenheit (°F)", "Rankine (°R)")
        )
    
    with col2:
        temp_entrada = st.number_input(
            "Valor de Temperatura",
            min_value=-500.0,
            max_value=1000.0,
            value=25.0,
            step=1.0,
            format="%.2f",
            help=f"Ingresa el valor en {unidad_entrada}"
        )

    T_C, T_K, T_F, T_R = 0.0, 0.0, 0.0, 0.0

    # Primero, convertir a Kelvin (la unidad base)
    if unidad_entrada == "Celsius (°C)":
        T_C = temp_entrada
        T_K = temp_entrada + 273.15
    elif unidad_entrada == "Kelvin (K)":
        T_K = temp_entrada
        T_C = temp_entrada - 273.15
    elif unidad_entrada == "Fahrenheit (°F)":
        T_F = temp_entrada
        T_C = (temp_entrada - 32) * 5/9
        T_K = T_C + 273.15
    elif unidad_entrada == "Rankine (°R)":
        T_R = temp_entrada
        T_K = temp_entrada * 5/9
        T_C = T_K - 273.15
        
    # Luego, calcular las otras escalas
    if unidad_entrada != "Fahrenheit (°F)":
        T_F = (T_C * 9/5) + 32
    if unidad_entrada != "Rankine (°R)":
        T_R = T_K * 9/5

    st.subheader("Resultados de la Conversión")
    st.markdown("---")

    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    col_res1.metric("Celsius (°C)", f"{T_C:.2f}")
    col_res2.metric("Kelvin (K)", f"{T_K:.2f}")
    col_res3.metric("Fahrenheit (°F)", f"{T_F:.2f}")
    col_res4.metric("Rankine (°R)", f"{T_R:.2f}")
    
    st.info("""
    **Fundamento Teórico:** Las escalas de temperatura se basan en puntos de referencia (como el punto de congelación y ebullición del agua). La escala **Kelvin** es la escala absoluta, donde 0 K representa el cero absoluto, el punto de mínima energía.
    """)

# --- FUNCIÓN 2: Equilibrio Térmico (Mezcla Simple) ---

def calcular_equilibrio_simple(m1, c1, T1, m2, c2, T2):
    """Calcula la temperatura final de equilibrio térmico de dos cuerpos."""
    # Q_ganado + Q_perdido = 0
    # m1*c1*(Tf - T1) + m2*c2*(Tf - T2) = 0
    # Tf * (m1*c1 + m2*c2) = m1*c1*T1 + m2*c2*T2
    
    num = (m1 * c1 * T1) + (m2 * c2 * T2)
    den = (m1 * c1) + (m2 * c2)
    
    if den == 0:
        return T1 # Ocurre si ambas masas son cero
        
    Tf = num / den
    
    Q1 = m1 * c1 * (Tf - T1) # Calor ganado/perdido por el cuerpo 1
    Q2 = m2 * c2 * (Tf - T2) # Calor ganado/perdido por el cuerpo 2
    
    return Tf, Q1, Q2

def modulo_equilibrio_simple():
    """Módulo para la simulación de equilibrio térmico simple."""
    st.subheader("2.1: Equilibrio Térmico Simple (2 Cuerpos sin Cambio de Fase)")
    st.markdown("Simula la mezcla de dos cuerpos o sustancias diferentes para encontrar la **temperatura final de equilibrio**.")
    
    col1, col2 = st.columns(2)
    
    # --- Cuerpo 1 ---
    with col1:
        st.markdown("### Cuerpo 1")
        m1 = st.slider("Masa $m_1$ (kg)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key='m1')
        T1_C = st.slider("Temperatura Inicial $T_1$ (°C)", min_value=-20.0, max_value=120.0, value=80.0, step=1.0, key='T1')
        material1 = st.selectbox("Material 1", list(CALORES_ESPECIFICOS.keys()), index=0, key='mat1')
        c1 = CALORES_ESPECIFICOS[material1]
        st.info(f"Calor Específico $c_1$: **{c1}** J/(kg·K)")
    
    # --- Cuerpo 2 ---
    with col2:
        st.markdown("### Cuerpo 2")
        m2 = st.slider("Masa $m_2$ (kg)", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key='m2')
        T2_C = st.slider("Temperatura Inicial $T_2$ (°C)", min_value=-20.0, max_value=120.0, value=20.0, step=1.0, key='T2')
        material2 = st.selectbox("Material 2", list(CALORES_ESPECIFICOS.keys()), index=1, key='mat2')
        c2 = CALORES_ESPECIFICOS[material2]
        st.info(f"Calor Específico $c_2$: **{c2}** J/(kg·K)")

    T1_K = T1_C + 273.15
    T2_K = T2_C + 273.15
    
    Tf_K, Q1, Q2 = calcular_equilibrio_simple(m1, c1, T1_K, m2, c2, T2_K)
    Tf_C = Tf_K - 273.15
    
    st.markdown("---")
    st.subheader("Resultados del Equilibrio")
    
    col_res_eq, col_graph_eq = st.columns(2)
    
    with col_res_eq:
        st.metric("Temperatura Final de Equilibrio $T_f$:", f"{Tf_C:.2f} °C")
        st.markdown(f"**Cuerpo 1 ({material1}):** Calor intercambiado $Q_1$: **{Q1/1000:.2f} kJ**")
        st.markdown(f"**Cuerpo 2 ({material2}):** Calor intercambiado $Q_2$: **{Q2/1000:.2f} kJ**")
        st.info(f"**Verificación:** $Q_1 + Q_2 \approx {(Q1 + Q2)/1000:.4f}$ kJ (Debe ser cercano a cero)")

    with col_graph_eq:
        # Gráfico de barras para el calor
        fig = go.Figure()
        fig.add_trace(go.Bar(name=material1, x=['Calor Intercambiado'], y=[Q1/1000], text=f'{Q1/1000:.2f} kJ', marker_color='red' if Q1 > 0 else 'blue'))
        fig.add_trace(go.Bar(name=material2, x=['Calor Intercambiado'], y=[Q2/1000], text=f'{Q2/1000:.2f} kJ', marker_color='blue' if Q2 > 0 else 'red'))
        
        # Gráfico de puntos para las temperaturas
        temp_data = [
            (T1_C, material1, 'red', 'T1'),
            (T2_C, material2, 'blue', 'T2'),
            (Tf_C, 'Equilibrio', 'green', 'Tf')
        ]
        
        fig_temp = go.Figure()
        
        for temp, name, color, label in temp_data:
            fig_temp.add_trace(go.Scatter(
                x=[label],
                y=[temp],
                mode='markers+text',
                marker=dict(size=15, color=color),
                name=name,
                text=[f'{temp:.2f}°C'],
                textposition="top center"
            ))

        fig_temp.update_layout(
            title="Temperaturas Iniciales y Final de Equilibrio (°C)",
            yaxis_title="Temperatura (°C)",
            showlegend=True
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        
    st.markdown("""
    **Fórmula Clave:** La temperatura final ($T_f$) se obtiene de la conservación de la energía, donde la suma de los calores ($Q$) intercambiados es cero: 
    $$ \sum Q = 0 \implies m_1 c_1 (T_f - T_1) + m_2 c_2 (T_f - T_2) = 0 $$
    El cuerpo con mayor **capacidad calorífica** ($m \cdot c$) tiene una mayor influencia en la temperatura final.
    """)
    

# --- FUNCIÓN 3: Equilibrio Térmico con Cambio de Fase (Extendido) ---

def calcular_calor_total_etapas(m, T_inicial_C, T_final_C, material):
    """Calcula el calor total en procesos con cambio de fase para el agua."""
    
    # Convertir a Kelvin
    T_inicial_K = T_inicial_C + 273.15
    T_final_K = T_final_C + 273.15
    
    Q_total = 0.0
    
    # Asumimos que el material es AGUA, por simplicidad en el cambio de fase
    if 'Agua' not in material:
        st.warning(f"Simulación de cambio de fase solo implementada para **Agua**. Se usará el calor específico del {material} para un proceso simple sin fases.")
        c_material = CALORES_ESPECIFICOS[material]
        Q_total = m * c_material * (T_final_K - T_inicial_K)
        return Q_total, [(0, 0, 0)] # Retorna un valor dummy para las etapas
        
    c_liq = CALORES_ESPECIFICOS["Agua (Líquida)"]
    c_sol = CALORES_ESPECIFICOS["Hielo"]
    
    etapas = [] # (Q, T_inicio, T_fin) en Kelvin

    # Etapa 1: Calentamiento de Hielo (sólido)
    if T_inicial_K < T_FUSION_AGUA:
        T_etapa = min(T_final_K, T_FUSION_AGUA)
        Q_calentamiento_solido = m * c_sol * (T_etapa - T_inicial_K)
        Q_total += Q_calentamiento_solido
        etapas.append((Q_calentamiento_solido, T_inicial_K, T_etapa))
        T_inicial_K = T_etapa
    
    # Etapa 2: Fusión (cambio de fase)
    if T_FUSION_AGUA <= T_inicial_K < T_EBULLICION_AGUA and T_final_K > T_FUSION_AGUA:
        if T_inicial_K == T_FUSION_AGUA and T_final_K > T_FUSION_AGUA:
            Q_fusion = m * L_FUSION_HIELO
            Q_total += Q_fusion
            etapas.append((Q_fusion, T_FUSION_AGUA, T_FUSION_AGUA))
            
    # Etapa 3: Calentamiento de Agua (líquido)
    if T_FUSION_AGUA < T_inicial_K < T_EBULLICION_AGUA or (T_inicial_K == T_FUSION_AGUA and T_final_K > T_FUSION_AGUA):
        T_etapa = min(T_final_K, T_EBULLICION_AGUA)
        Q_calentamiento_liq = m * c_liq * (T_etapa - T_inicial_K)
        Q_total += Q_calentamiento_liq
        etapas.append((Q_calentamiento_liq, T_inicial_K, T_etapa))
        T_inicial_K = T_etapa

    # Etapa 4: Vaporización (cambio de fase)
    if T_EBULLICION_AGUA <= T_inicial_K and T_final_K > T_EBULLICION_AGUA:
        if T_inicial_K == T_EBULLICION_AGUA and T_final_K > T_EBULLICION_AGUA:
            Q_vaporizacion = m * L_VAPORIZACION_AGUA
            Q_total += Q_vaporizacion
            etapas.append((Q_vaporizacion, T_EBULLICION_AGUA, T_EBULLICION_AGUA))

    # Etapa 5: Calentamiento de Vapor (gas, no implementado C_gas, solo Q_liq)
    if T_EBULLICION_AGUA < T_final_K:
        # Simplificación: asumimos que el calentamiento continúa con C_liq
        Q_calentamiento_gas = m * c_liq * (T_final_K - T_EBULLICION_AGUA)
        Q_total += Q_calentamiento_gas
        etapas.append((Q_calentamiento_gas, T_EBULLICION_AGUA, T_final_K))
        
    return Q_total, etapas


def modulo_equilibrio_fase():
    """Módulo para el cálculo de calor total en procesos con cambio de fase."""
    st.subheader("2.2: Cálculo del Calor Total con Cambio de Fase (Solo Agua)")
    st.markdown("Calcula el **calor total** necesario para llevar una masa de agua desde una temperatura inicial a una final, considerando las etapas de **fusión** y **vaporización**.")

    col1, col2 = st.columns(2)
    
    with col1:
        m = st.slider("Masa $m$ (kg)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key='m_fase')
        material = st.selectbox("Material (Solo Agua para CF)", ["Agua (Líquida)"], index=0, key='mat_fase')
        
    with col2:
        T_inicial_C = st.slider("Temperatura Inicial $T_i$ (°C)", min_value=-20.0, max_value=120.0, value=0.0, step=1.0, key='Ti_fase')
        T_final_C = st.slider("Temperatura Final $T_f$ (°C)", min_value=-20.0, max_value=120.0, value=100.0, step=1.0, key='Tf_fase')

    if T_inicial_C >= T_final_C:
        st.warning("La Temperatura Final debe ser mayor que la Inicial para el cálculo del calor total (calentamiento).")
        return

    Q_total, etapas = calcular_calor_total_etapas(m, T_inicial_C, T_final_C, material)
    
    st.markdown("---")
    st.subheader("Resultados del Proceso por Etapas")
    
    st.metric("Calor Total Requerido $Q_{total}$", f"{Q_total/1000:.2f} kJ")
    
    st.info("""
    **Fundamento Teórico:** Durante un **cambio de fase** (como la fusión o vaporización), la temperatura permanece constante, ya que toda la energía suministrada (el **calor latente**) se utiliza para romper o formar los enlaces moleculares, no para aumentar la energía cinética de las moléculas.
    """)

    # Mostrar gráfico de calentamiento
    fig = go.Figure()
    
    # Prepara los datos para el gráfico de calentamiento (Temperatura vs. Calor)
    Q_acumulado = 0
    Q_plot = [0]
    T_plot = [T_inicial_C]
    
    for Q_etapa, T_inicio_K, T_fin_K in etapas:
        T_inicio_C = T_inicio_K - 273.15
        T_fin_C = T_fin_K - 273.15
        
        Q_acumulado += Q_etapa / 1000 # En kJ
        Q_plot.append(Q_acumulado)
        
        # Si hay cambio de fase (T_inicio == T_fin), la temperatura se mantiene constante
        if abs(T_inicio_C - T_fin_C) < 0.1:
            T_plot.append(T_inicio_C)
        else:
            T_plot.append(T_fin_C)
        
    fig.add_trace(go.Scatter(x=Q_plot, y=T_plot, mode='lines+markers', name='Proceso de Calentamiento'))
    
    fig.update_layout(
        title="Diagrama de Calentamiento (Temperatura vs. Calor Suministrado)",
        xaxis_title="Calor Suministrado (kJ)",
        yaxis_title="Temperatura (°C)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detalle de las etapas
    st.markdown("**Desglose de Calor por Etapa (kJ):**")
    
    # Determinar qué pasos están incluidos para mostrar la tabla con un orden lógico
    data_table = []
    
    if T_inicial_C < 0 and T_final_C > 0:
        data_table.append(
            ["Calentamiento Hielo", f"{m * c_sol * (T_FUSION_AGUA - T_inicial_K)/1000:.2f}", f"{T_inicial_C}°C a 0°C"]
        )
    if T_inicial_C <= 0 and T_final_C >= 0:
        data_table.append(
            ["Fusión (Cambio de Fase)", f"{m * L_FUSION_HIELO/1000:.2f}", "0°C (Calor Latente)"]
        )
    if T_inicial_C < 100 and T_final_C > 0:
        data_table.append(
            ["Calentamiento Líquido", f"{m * c_liq * (min(100.0, T_final_C) - max(0.0, T_inicial_C))/1000:.2f}", f"{max(0.0, T_inicial_C)}°C a {min(100.0, T_final_C)}°C"]
        )
    
    # Solo mostrar si hay más de una etapa o la etapa total
    if len(data_table) > 0:
        st.table(data_table)
    
    st.markdown(f"**Calor Total:** **{Q_total/1000:.2f} kJ**")


# --- FUNCIÓN 4: Conducción de Calor 1D ---

def calcular_conduccion_estado_estacionario(L, Ta, Tb, N):
    """Calcula el perfil de temperatura en estado estacionario (lineal)."""
    # En estado estacionario, el perfil de temperatura es lineal: T(x) = Ta + (Tb - Ta) * x/L
    x = np.linspace(0, L, N)
    T = Ta + (Tb - Ta) * (x / L)
    return x, T

def modulo_conduccion_1d():
    """Módulo para la simulación de conducción de calor en 1D."""
    st.header("3️⃣ Simulación de Conducción de Calor (1D)")
    st.markdown("Simula la transferencia de calor en una **barra** (o pared) de longitud $L$, con sus extremos mantenidos a temperaturas constantes ($T_A$ y $T_B$).")
    
    st.subheader("Estado Estacionario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        L = st.slider("Longitud de la Barra $L$ (m)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key='L')
        Ta = st.slider("Temperatura Extremo A $T_A$ (°C)", min_value=0.0, max_value=200.0, value=100.0, step=5.0, key='Ta')
        Tb = st.slider("Temperatura Extremo B $T_B$ (°C)", min_value=0.0, max_value=200.0, value=20.0, step=5.0, key='Tb')
    
    with col2:
        k_val = st.slider("Conductividad Térmica $k$ (W/m·K)", min_value=1.0, max_value=400.0, value=50.0, step=1.0, key='k_cond')
        A = st.slider("Área de Sección Transversal $A$ ($m^2$)", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key='A')
        
        # Cálculo de la tasa de transferencia de calor Q_dot
        Q_dot = -k_val * A * (Tb - Ta) / L
        st.metric("Tasa de Transferencia de Calor $\dot{Q}$ (W)", f"{Q_dot:.2f}")

    N = 100 # Número de puntos para la visualización
    x, T = calcular_conduccion_estado_estacionario(L, Ta, Tb, N)
    
    # Gráfico del Perfil de Temperatura (Matplotlib para mayor control sobre el eje X)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, T, 'r-', linewidth=3)
    ax.set_title("Perfil de Temperatura en Estado Estacionario (1D)")
    ax.set_xlabel("Posición a lo largo de la Barra $x$ (m)")
    ax.set_ylabel("Temperatura $T$ (°C)")
    ax.grid(True, linestyle='--')
    ax.set_ylim(min(Ta, Tb) - 10, max(Ta, Tb) + 10)
    st.pyplot(fig)
    
    st.markdown("""
    **Fundamento Teórico (Estado Estacionario):** En estado estacionario, la temperatura en cada punto de la barra ya no cambia con el tiempo. La distribución de temperatura es **lineal** y la tasa de transferencia de calor ($\dot{Q}$) es constante en cualquier punto, siguiendo la **Ley de Fourier**:
    $$ \dot{Q} = -k A \frac{dT}{dx} $$
    donde $k$ es la conductividad térmica, $A$ el área, y $\frac{dT}{dx}$ el gradiente de temperatura.
    """)
    

# --- FUNCIÓN 5: Conducción de Calor 2D (Extendido) ---

def modulo_conduccion_2d():
    """Módulo para la simulación de conducción de calor en 2D (simplificado)."""
    st.header("4️⃣ Conducción 2D Simplificada (Placa Cuadrada)")
    st.markdown("Simulación del perfil de temperatura en una placa delgada cuadrada en **estado estacionario**. Los bordes superior, inferior, izquierdo y derecho se mantienen a temperaturas constantes.")

    L_placa = st.slider("Tamaño de la Placa $L$ (unidades)", min_value=10, max_value=50, value=20, step=5, key='L_placa')
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Temperaturas de los bordes
    T_superior = col1.slider("Borde Superior (°C)", 0.0, 100.0, 80.0, key='Ts')
    T_inferior = col2.slider("Borde Inferior (°C)", 0.0, 100.0, 20.0, key='Ti')
    T_izquierdo = col3.slider("Borde Izquierdo (°C)", 0.0, 100.0, 50.0, key='Tiz')
    T_derecho = col4.slider("Borde Derecho (°C)", 0.0, 100.0, 50.0, key='Tde')
    
    # Solución numérica simple de la Ecuación de Laplace (para estado estacionario)
    N = L_placa
    T_2d = np.zeros((N, N))
    
    # Condiciones de contorno
    T_2d[0, :] = T_superior
    T_2d[N-1, :] = T_inferior
    T_2d[:, 0] = T_izquierdo
    T_2d[:, N-1] = T_derecho
    
    # Iteración de Jacobi (simplificada para una visualización rápida)
    max_iter = 50
    for _ in range(max_iter):
        T_new = T_2d.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # Ecuación de Laplace discretizada (Diferencias Finitas)
                T_new[i, j] = 0.25 * (T_2d[i+1, j] + T_2d[i-1, j] + T_2d[i, j+1] + T_2d[i, j-1])
        T_2d = T_new
        
    # Visualización (Mapa de calor)
    fig = go.Figure(data=go.Heatmap(
        z=T_2d,
        x=np.arange(N),
        y=np.arange(N),
        colorscale='Hot',
        zmin=min(T_inferior, T_superior, T_izquierdo, T_derecho),
        zmax=max(T_inferior, T_superior, T_izquierdo, T_derecho)
    ))

    fig.update_layout(
        title='Mapa de Calor de la Distribución de Temperatura (Estado Estacionario 2D)',
        xaxis_title='Posición X',
        yaxis_title='Posición Y',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Fundamento Teórico (Conducción 2D Estacionaria):** La distribución de temperatura en estado estacionario sin generación interna de calor se rige por la **Ecuación de Laplace** ($\nabla^2 T = 0$). La simulación utiliza un método de **diferencias finitas** para aproximar esta solución, donde la temperatura de cada punto interno es el promedio de sus vecinos inmediatos.
    """)
    


# --- LÓGICA PRINCIPAL DE STREAMLIT ---

def main():
    st.title("🔥 Asistente de Termodinámica y Transferencia de Calor")
    st.caption("Una aplicación interactiva de Streamlit por tu Asistente de Programación.")

    # Sidebar para la navegación
    st.sidebar.title("Menú de Simulaciones")
    modulo_seleccionado = st.sidebar.radio(
        "Elige un Módulo:",
        ("1. Conversión de Escalas", "2. Equilibrio Térmico", "3. Conducción 1D", "4. Conducción 2D (Extendido)")
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Fundamentos")
    st.sidebar.markdown("""
    Esta aplicación modela fenómenos térmicos:
    * **Temperatura:** Medida de la energía cinética promedio de las moléculas.
    * **Calor:** Energía transferida debido a una diferencia de temperatura.
    * **Transferencia Térmica:** Procesos de Conducción, Convección o Radiación.
    """)

    # Módulos
    if modulo_seleccionado == "1. Conversión de Escalas":
        modulo_conversion()
    
    elif modulo_seleccionado == "2. Equilibrio Térmico":
        st.header("2️⃣ Equilibrio Térmico")
        st.markdown("Explora cómo los cuerpos alcanzan una temperatura común al mezclarse.")
        st.markdown("---")
        
        opcion_equilibrio = st.selectbox(
            "Selecciona la Opción de Equilibrio:",
            ("Equilibrio Térmico Simple (2 Cuerpos)", "Cálculo de Calor con Cambio de Fase (Extendido)")
        )
        
        if opcion_equilibrio == "Equilibrio Térmico Simple (2 Cuerpos)":
            modulo_equilibrio_simple()
        else:
            modulo_equilibrio_fase()
            
    elif modulo_seleccionado == "3. Conducción 1D":
        modulo_conduccion_1d()
        
    elif modulo_seleccionado == "4. Conducción 2D (Extendido)":
        modulo_conduccion_2d()


if __name__ == "__main__":
    main()
