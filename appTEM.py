import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.constants import convert_temperature

# --- 1. Configuración y Constantes ---

st.set_page_config(layout="wide", page_title="Simulador de Termodinámica")

# Constantes de Termodinámica (Valores de referencia, J/kg·K o J/kg)
CONSTANTES = {
    "Agua": {
        "ce_liquido": 4186,  # Calor específico líquido (J/kg·K)
        "ce_solido": 2090,   # Calor específico sólido (J/kg·K)
        "ce_gas": 2010,      # Calor específico gas (J/kg·K)
        "Lf": 334000,        # Calor latente de fusión (J/kg)
        "Lv": 2260000,       # Calor latente de vaporización (J/kg)
        "Tf": 0.0,           # Temperatura de fusión (°C)
        "Tv": 100.0,         # Temperatura de ebullición (°C)
        "k": 0.60,           # Conductividad térmica (W/m·K)
    },
    "Cobre": {"ce": 385, "k": 401.0},
    "Aluminio": {"ce": 900, "k": 237.0},
    "Vidrio": {"ce": 840, "k": 1.1},
}

# --- 2. Funciones de Conversión de Escalas ---

def convert_temp_rankine(valor, from_scale, to_scale):
    """Maneja la conversión de Rankine, que no está en scipy.constants."""
    # Convertir a Kelvin primero
    if from_scale == 'R':
        kelvin = valor / 1.8
    elif from_scale == 'F':
        kelvin = convert_temperature(valor, 'Fahrenheit', 'Kelvin')
    elif from_scale == 'C':
        kelvin = convert_temperature(valor, 'Celsius', 'Kelvin')
    else: # K
        kelvin = valor

    # Convertir de Kelvin a la escala de salida
    if to_scale == 'R':
        return kelvin * 1.8
    elif to_scale == 'F':
        return convert_temperature(kelvin, 'Kelvin', 'Fahrenheit')
    elif to_scale == 'C':
        return convert_temperature(kelvin, 'Kelvin', 'Celsius')
    else: # K
        return kelvin

# --- 3. Secciones de la Aplicación ---

# 3.1. Conversor Dinámico de Temperaturas
def seccion_conversor():
    st.header("1️⃣ Conversor Dinámico de Escalas de Temperatura")
    st.markdown("Convierte valores de temperatura entre **Celsius (°C)**, **Kelvin (°K)**, **Fahrenheit (°F)** y **Rankine (°R)** usando controles interactivos.")
    
    # Rango dinámico de valores de entrada
    col_sel, col_val, col_minmax = st.columns([1, 1, 2])
    
    with col_sel:
        escala_entrada = st.selectbox("Escala de Entrada", ['C', 'K', 'F', 'R'], index=0)
    
    with col_minmax:
        min_val = st.number_input("Límite Mínimo", value=-100.0, step=10.0)
        max_val = st.number_input("Límite Máximo", value=500.0, step=10.0)

    with col_val:
        valor_entrada = st.slider(f"Valor en °{escala_entrada}", min_value=min_val, max_value=max_val, value=25.0, step=0.1)
    
    st.divider()

    st.subheader("Resultados de Conversión")
    
    conversiones = {}
    escalas = ['C', 'K', 'F', 'R']
    
    # Calcular conversiones
    for escala_salida in escalas:
        if escala_salida != escala_entrada:
            conversiones[escala_salida] = convert_temp_rankine(valor_entrada, escala_entrada, escala_salida)

    # Mostrar resultados
    col_C, col_K, col_F, col_R = st.columns(4)
    
    for col, escala in zip([col_C, col_K, col_F, col_R], escalas):
        with col:
            if escala == escala_entrada:
                st.metric(f"{escala} (°{escala})", f"{valor_entrada:.2f}")
            else:
                st.metric(f"{escala} (°{escala})", f"{conversiones[escala]:.2f}")

    st.info("""
    **Explicación Física:** Las escalas Kelvin y Rankine son **escalas absolutas** (0 K y 0 R representan el cero absoluto, donde no hay movimiento molecular), mientras que Celsius y Fahrenheit se basan en puntos de referencia del agua. El Cero Absoluto es $0 K \\approx -273.15 °C$.
    """)
    # 

[Image of Temperature scales comparison showing Celsius, Kelvin, Fahrenheit, and Rankine scales]


# 3.2. Simulación de Equilibrio Térmico
def seccion_equilibrio_termico():
    st.header("2️⃣ Equilibrio Térmico y Calores Específicos")
    st.markdown("Simulación de la mezcla de dos cuerpos (o más) hasta alcanzar la **temperatura final de equilibrio**.")

    st.subheader("Parámetros de los Cuerpos")
    
    materiales = list(CONSTANTES.keys())
    
    # Interfaz para 2 cuerpos
    st.markdown("##### Cuerpo 1")
    col1, col2, col3 = st.columns(3)
    with col1:
        m1 = st.slider("Masa $m_1$ (kg)", 0.1, 5.0, 1.0, 0.1)
    with col2:
        ce1_key = st.selectbox("Material 1", materiales, index=2, key='mat1') # Aluminio
    with col3:
        T1 = st.slider("Temperatura Inicial $T_1$ (°C)", 50.0, 200.0, 100.0, 1.0)
    
    st.markdown("##### Cuerpo 2")
    col4, col5, col6 = st.columns(3)
    with col4:
        m2 = st.slider("Masa $m_2$ (kg)", 0.1, 5.0, 2.0, 0.1)
    with col5:
        ce2_key = st.selectbox("Material 2", materiales, index=0, key='mat2') # Agua
    with col6:
        T2 = st.slider("Temperatura Inicial $T_2$ (°C)", 0.0, 40.0, 20.0, 1.0)

    # Obtener calores específicos
    ce1 = CONSTANTES[ce1_key].get('ce', CONSTANTES[ce1_key].get('ce_liquido'))
    ce2 = CONSTANTES[ce2_key].get('ce', CONSTANTES[ce2_key].get('ce_liquido'))
    
    # Cálculo de la Temperatura Final de Equilibrio (Tf)
    # Ecuación: Tf = (m1*ce1*T1 + m2*ce2*T2) / (m1*ce1 + m2*ce2)
    try:
        Tf = (m1 * ce1 * T1 + m2 * ce2 * T2) / (m1 * ce1 + m2 * ce2)
    except ZeroDivisionError:
        Tf = 0.0

    # Cálculo de los calores
    Q1 = m1 * ce1 * (Tf - T1) # Q_perdido (será negativo)
    Q2 = m2 * ce2 * (Tf - T2) # Q_ganado (será positivo)
    
    st.divider()

    st.subheader("Resultados y Gráfico de Equilibrio")

    col_Tf, col_Q_info = st.columns(2)
    with col_Tf:
        st.metric("Temperatura Final de Equilibrio $T_f$ (°C)", f"{Tf:.2f}")

    with col_Q_info:
        st.markdown(f"**Calor Específico de {ce1_key} ($c_{{e1}}$):** ${ce1} \\ J/kg\\cdot°C$")
        st.markdown(f"**Calor Específico de {ce2_key} ($c_{{e2}}$):** ${ce2} \\ J/kg\\cdot°C$")
        st.markdown(f"**Calor Perdido por {ce1_key}:** ${Q1:,.0f}\ J$")
        st.markdown(f"**Calor Ganado por {ce2_key}:** ${Q2:,.0f}\ J$")

    # Visualización con Plotly (Gráfico de barras de calor)
    etiquetas = [f"Cuerpo 1 ({ce1_key})", f"Cuerpo 2 ({ce2_key})"]
    
    fig = go.Figure(data=[
        go.Bar(name='Temperatura Inicial', x=etiquetas, y=[T1, T2], marker_color=['red', 'blue']),
        go.Scatter(name='Temperatura Final', x=etiquetas, y=[Tf, Tf], mode='lines+markers', line=dict(color='black', dash='dash', width=2))
    ])
    fig.update_layout(
        title='Evolución de Temperatura al Equilibrio',
        yaxis_title='Temperatura (°C)',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Explicación Física:** El principio se basa en la **Conservación de la Energía**: en un sistema aislado, el calor total perdido por los cuerpos calientes es igual al calor total ganado por los cuerpos fríos ($\Sigma Q = 0$). La temperatura de equilibrio es el promedio ponderado por la capacidad calorífica ($m \\cdot c_e$) de cada cuerpo.
    """)
    # 

# 3.3. Procesos Térmicos y Cambios de Fase
def seccion_cambio_fase():
    st.header("3️⃣ Procesos con Cambio de Fase (Agua)")
    st.markdown("Calcula el **calor total** ($Q$) necesario para llevar una masa de agua a través de etapas de calentamiento y cambio de fase (fusión, ebullición).")

    # Parámetros de entrada
    col1, col2 = st.columns(2)
    with col1:
        m_agua = st.slider("Masa de Agua $m$ (kg)", 0.1, 5.0, 1.0, 0.1)
    with col2:
        T_inicial = st.slider("Temperatura Inicial $T_i$ (°C)", -50.0, 150.0, 20.0, 1.0)
        T_final = st.slider("Temperatura Final $T_f$ (°C)", -50.0, 150.0, 100.0, 1.0)
    
    if T_inicial == T_final:
        st.warning("Las temperaturas inicial y final son las mismas. No hay transferencia neta de calor.")
        return

    # Constantes del agua
    C = CONSTANTES["Agua"]
    Tf_C = C["Tf"]     # 0 °C
    Tv_C = C["Tv"]     # 100 °C
    
    st.divider()
    st.subheader("Cálculo del Calor Total por Etapas")

    # Inicializar calor total y lista de etapas
    Q_total = 0.0
    etapas_q = []
    T_puntos = [T_inicial]
    Q_acumulado = [0.0]

    def add_calor(m, ce, delta_T, nombre_etapa, T_act):
        Q = m * ce * delta_T
        Q_total_new = Q_acumulado[-1] + Q
        etapas_q.append((nombre_etapa, Q))
        T_puntos.append(T_act + delta_T)
        Q_acumulado.append(Q_total_new)
        return Q

    def add_calor_fase(m, L, nombre_fase, T_fase):
        Q = m * L
        Q_total_new = Q_acumulado[-1] + Q
        etapas_q.append((nombre_fase, Q))
        T_puntos.append(T_fase)
        Q_acumulado.append(Q_total_new)
        return Q

    T_actual = T_inicial

    # Algoritmo de Calentamiento (T_inicial < T_final)
    if T_inicial < T_final: 
        
        # Etapa 1: Calentamiento de Sólido (Hielo)
        if T_actual < Tf_C:
            T_target = min(T_final, Tf_C)
            add_calor(m_agua, C["ce_solido"], T_target - T_actual, 
                      f"1. Calentamiento Sólido: $T={T_actual:.0f}\\to{T_target:.0f} \\ °C$", T_actual)
            T_actual = T_target
            
        # Etapa 2: Fusión (si se alcanza 0°C y se sigue calentando)
        if T_actual == Tf_C and T_final > Tf_C:
            add_calor_fase(m_agua, C["Lf"], f"2. Fusión (Cambio de Fase): $T={Tf_C:.0f} \\ °C$", Tf_C)
            
        T_actual = T_puntos[-1] # Actualizar T_actual después de posible fusión
        
        # Etapa 3: Calentamiento de Líquido (Agua)
        if T_actual < Tv_C:
            T_target = min(T_final, Tv_C)
            add_calor(m_agua, C["ce_liquido"], T_target - T_actual, 
                      f"3. Calentamiento Líquido: $T={T_actual:.0f}\\to{T_target:.0f} \\ °C$", T_actual)
            T_actual = T_target

        # Etapa 4: Vaporización (si se alcanza 100°C y se sigue calentando)
        if T_actual == Tv_C and T_final > Tv_C:
            add_calor_fase(m_agua, C["Lv"], f"4. Vaporización (Cambio de Fase): $T={Tv_C:.0f} \\ °C$", Tv_C)
            
        T_actual = T_puntos[-1] # Actualizar T_actual después de posible vaporización

        # Etapa 5: Calentamiento de Gas (Vapor)
        if T_actual < T_final:
            add_calor(m_agua, C["ce_gas"], T_final - T_actual, 
                      f"5. Calentamiento Vapor: $T={T_actual:.0f}\\to{T_final:.0f} \\ °C$", T_actual)

    # Nota: Si T_inicial > T_final, el proceso es inverso (enfriamiento).
    else:
        st.error("Para esta simulación, por favor configure $T_i < T_f$. La simulación de enfriamiento es similar pero con $Q < 0$ y usando los calores latentes negativos.")
        return


    # 6. Mostrar Resultados
    Q_total = Q_acumulado[-1]
    st.metric("Calor Total Requerido $Q_{total}$ (Joule)", f"{Q_total:,.0f}")

    st.subheader("Detalle del Proceso por Etapas:")
    etapas_display = [f"**{nombre}:** ${Q:,.0f}\ J$" for nombre, Q in etapas_q if abs(Q) > 1]
    
    if etapas_display:
        for item in etapas_display:
            st.markdown(item)
    else:
        st.markdown("No se requirió calor, o la diferencia de temperatura fue demasiado pequeña.")

    # Gráfico de la Curva de Calentamiento (Heat Curve)
    fig = go.Figure(data=[
        go.Scatter(x=Q_acumulado, y=T_puntos, mode='lines+markers', name='Curva de Calentamiento',
                   line=dict(color='orange', width=3))
    ])
    fig.update_layout(
        title='Curva de Calentamiento: Temperatura vs. Calor Añadido',
        xaxis_title='Calor Añadido $Q$ (Joule)',
        yaxis_title='Temperatura $T$ (°C)'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Explicación Física:** El calor se usa para dos fines: **aumentar la temperatura** ($Q=mc_e\Delta T$) o **cambiar la fase** ($Q=mL$). Los tramos horizontales en la curva representan los cambios de fase (fusión a $0 °C$ y ebullición a $100 °C$) donde el calor latente ($L$) se absorbe sin cambiar la temperatura.
    """)
    # 

# 3.4. Simulación de Conducción de Calor 1D (Barra)
def seccion_conduccion_1d():
    st.header("4️⃣ Conducción de Calor en Barra (1D)")
    st.markdown("Simulación de la conducción de calor en una barra (1D), mostrando el perfil de temperatura en **estado estacionario** y su evolución temporal (animación simplificada).")
    
    col1, col2 = st.columns(2)
    materiales_k = {k: v['k'] for k, v in CONSTANTES.items() if 'k' in v}

    with col1:
        L = st.slider("Longitud de la Barra $L$ (m)", 0.1, 2.0, 1.0, 0.1)
        material = st.selectbox("Material de la Barra", list(materiales_k.keys()), index=1)
        k = materiales_k[material]
        st.markdown(f"Conductividad Térmica $k$ (W/m·K): **{k}**")

    with col2:
        T_caliente = st.slider("Temperatura Extremo Caliente $T_H$ (°C)", 50.0, 500.0, 150.0, 1.0)
        T_frio = st.slider("Temperatura Extremo Frío $T_C$ (°C)", 0.0, 100.0, 20.0, 1.0)
        sim_tipo = st.radio("Tipo de Simulación", ["Estado Estacionario", "Evolución Temporal"])

    st.divider()

    x = np.linspace(0, L, 100)
    
    if sim_tipo == "Estado Estacionario":
        st.subheader("Gráfico del Perfil de Temperatura (Estado Estacionario)")
        
        # T(x) es lineal: T(x) = T_H - (T_H - T_C) * (x / L)
        T_x = T_caliente - (T_caliente - T_frio) * (x / L)

        # Cálculo de Flujo de Calor (Asumimos Área A = 1 m²)
        A = 1.0
        # Ley de Fourier: P = Q/t = k * A * (T_H - T_C) / L
        Flujo_Potencia = k * A * (T_caliente - T_frio) / L
        
        fig = go.Figure(
            data=[go.Scatter(x=x, y=T_x, mode='lines', line=dict(color='red', width=3))],
            layout=go.Layout(
                title=f'Perfil de Temperatura $T(x)$ de la Barra de {material}',
                xaxis_title='Posición $x$ (m)',
                yaxis_title='Temperatura $T$ (°C)',
                yaxis_range=[min(T_frio, T_caliente) - 10, max(T_frio, T_caliente) + 10]
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Potencia de Flujo de Calor (Asumiendo $A=1m^2$) $P$ (W)", f"{Flujo_Potencia:,.2f}")
    
    else: # Evolución Temporal (Animación simplificada)
        st.subheader("Animación de la Evolución Temporal")
        
        # Malla y parámetros para la simulación temporal (Método de Diferencias Finitas Simplificado)
        Nx = 50 
        dx = L / Nx
        # Usamos la difusividad térmica alpha (k / (rho * ce))
        # Para el propósito de visualización, usamos un valor de alpha representativo
        alpha_sim = 1e-4 
        dt = 0.5 * dx**2 / (2 * alpha_sim) # Criterio de estabilidad
        
        T_curr = np.full(Nx, T_frio)
        T_curr[0] = T_caliente # Condición de borde inicial

        # Inicialización de la animación de Plotly
        T_frames = [T_curr.copy()]
        num_pasos = 100 # Número de frames a mostrar

        for _ in range(num_pasos):
            T_next = T_curr.copy()
            for i in range(1, Nx - 1):
                # Ecuación de conducción de calor 1D discretizada
                T_next[i] = T_curr[i] + alpha_sim * dt / (dx**2) * (T_curr[i+1] - 2 * T_curr[i] + T_curr[i-1])
            
            # Reaplicar condiciones de borde
            T_next[0] = T_caliente
            T_next[-1] = T_frio # Asumimos que el extremo frío se mantiene constante
            
            T_curr = T_next
            T_frames.append(T_curr.copy())

        # Creación de la animación con Plotly
        fig = go.Figure(
            data=[go.Scatter(x=np.linspace(0, L, Nx), y=T_frames[0], mode='lines', line=dict(color='red', width=3))],
            layout=go.Layout(
                title='Evolución del Perfil de Temperatura (Conducción 1D)',
                xaxis_title='Posición $x$ (m)',
                yaxis_title='Temperatura $T$ (°C)',
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                                "label": "Play",
                                "method": "animate"
                            }
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 87},
                        "showactive": False,
                        "type": "buttons",
                        "x": 0.1,
                        "xanchor": "right",
                        "y": 0,
                        "yanchor": "top"
                    }
                ]
            ),
            frames=[go.Frame(data=[go.Scatter(y=T_frame)]) for T_frame in T_frames]
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # USANDO RAW F-STRING (fr"""...""") para evitar el SyntaxError con LaTeX
    st.info(fr"""
    **Explicación Física:** La **conducción** es la transferencia de energía por contacto directo y colisiones moleculares. Se rige por la **Ley de Fourier**: $P = -k A \frac{{dT}}{{dx}}$, donde $k$ es la conductividad térmica.
    * **Estado Estacionario:** El perfil de temperatura es **lineal** porque la transferencia de calor es constante en cada sección.
    * **Evolución Temporal:** La temperatura evoluciona según la **Ecuación de Difusión de Calor** $\frac{{\partial T}}{{\partial t}} = \alpha \frac{{\partial^2 T}}{{\partial x^2}}$ (donde $\alpha = k / \rho c_e$). Materiales con alta conductividad ($k$) como el **Cobre** ($k={CONSTANTES['Cobre']['k']} W/m·K$) alcanzan el estado estacionario mucho más rápido que los aislantes como el **Vidrio** ($k={CONSTANTES['Vidrio']['k']} W/m·K$).
    """)
    # 


# 3.5. Simulación de Conducción de Calor 2D
def seccion_conduccion_2d():
    st.header("5️⃣ Conducción de Calor en Placa (2D Simplificada)")
    st.markdown("Visualización de la distribución de temperatura en una placa cuadrada en **estado estacionario** usando el método de relajación (Diferencias Finitas).")
    
    st.warning("La simulación 2D utiliza un método iterativo para encontrar la solución de estado estacionario de la Ecuación de Laplace ($\nabla^2 T = 0$).")

    # Parámetros de la malla
    L_placa = st.slider("Tamaño de la Malla (Nodos)", 10, 50, 20, 5)
    T_max_iter = st.slider("Iteraciones (Precisión del Cálculo)", 100, 2000, 500, 100)

    # Condiciones de Borde Interactivas
    st.subheader("Condiciones de Borde (°C)")
    col_t, col_l, col_r, col_b = st.columns(4)
    with col_t:
        T_borde_top = st.slider("Borde Superior $T_{Top}$", 0, 300, 100)
    with col_l:
        T_borde_left = st.slider("Borde Izquierdo $T_{Left}$", 0, 300, 75)
    with col_r:
        T_borde_right = st.slider("Borde Derecho $T_{Right}$", 0, 300, 25)
    with col_b:
        T_borde_bottom = st.slider("Borde Inferior $T_{Bottom}$", 0, 300, 50)

    # 1. Inicialización de la malla
    T = np.zeros((L_placa, L_placa))
    
    # Aplicar condiciones de borde
    T[0, :] = T_borde_top
    T[-1, :] = T_borde_bottom
    T[:, 0] = T_borde_left
    T[:, -1] = T_borde_right
    
    # 2. Solución Iterativa (Método de Jacobi Simplificado)
    # La temperatura de un nodo interior es el promedio de sus cuatro vecinos.
    for _ in range(T_max_iter):
        T_old = T.copy()
        for i in range(1, L_placa - 1):
            for j in range(1, L_placa - 1):
                T[i, j] = 0.25 * (T_old[i+1, j] + T_old[i-1, j] + T_old[i, j+1] + T_old[i, j-1])
        # Reaplicar los bordes (necesario si se usa T_old en el interior)
        T[0, :] = T_borde_top
        T[-1, :] = T_borde_bottom
        T[:, 0] = T_borde_left
        T[:, -1] = T_borde_right

    # 3. Visualización con Plotly (Heatmap 2D)
    st.subheader("Distribución de Temperatura en Estado Estacionario (Heatmap)")
    
    # Invertir el eje Y para que la fila 0 (Top) esté arriba en el gráfico
    T_display = np.flipud(T) 

    fig = px.imshow(T_display, 
                    color_continuous_scale=px.colors.sequential.Inferno, 
                    aspect="equal",
                    labels=dict(color="Temperatura (°C)"),
                    zmin=0, zmax=300 # Rango fijo para consistencia visual
                    )
    
    fig.update_layout(
        title='Mapa de Calor 2D (Conducción)',
        xaxis=dict(title='Posición X'), 
        yaxis=dict(title='Posición Y', scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # USANDO RAW STRING (r"""...""") para evitar el SyntaxError con LaTeX
    st.info(r"""
    **Explicación Física:** En un problema de conducción 2D en estado estacionario (temperaturas que no cambian con el tiempo), la distribución de temperatura está gobernada por la **Ecuación de Laplace** ($\nabla^2 T = 0$). Esto significa que no hay generación ni acumulación de calor. Las iteraciones numéricas simulan el proceso natural de difusión hasta que cada punto interior se "relaja" a la temperatura promedio de sus vecinos, satisfaciendo la condición de equilibrio.
    """)
    # 

# --- 4. Función Principal de la Aplicación ---

def main():
    st.sidebar.title("Menú de Simulación")
    
    opciones = {
        "1. Conversor de Temperaturas": seccion_conversor,
        "2. Equilibrio Térmico (2 Cuerpos)": seccion_equilibrio_termico,
        "3. Procesos con Cambio de Fase": seccion_cambio_fase,
        "4. Conducción de Calor 1D": seccion_conduccion_1d,
        "5. Conducción de Calor 2D": seccion_conduccion_2d,
    }

    seleccion = st.sidebar.selectbox("Seleccione la Simulación:", list(opciones.keys()))
    
    st.title("🌡️ Simulador Interactivo de Termodinámica")
    
    # Ejecutar la función correspondiente a la selección
    opciones[seleccion]()

if __name__ == "__main__":
    main()
