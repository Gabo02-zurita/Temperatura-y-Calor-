import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.constants import convert_temperature

# --- 1. Configuración y Constantes ---

st.set_page_config(layout="wide", page_title="Simulador de Termodinámica")

# Constantes de Termodinámica (Valores de referencia)
# c_e: J/kg·K | L: J/kg | k: W/m·K
CONSTANTES = {
    "Agua": {
        "ce_liquido": 4186,  # Calor específico líquido
        "ce_solido": 2090,   # Calor específico sólido (Hielo)
        "ce_gas": 2010,      # Calor específico gas (Vapor)
        "Lf": 334000,        # Calor latente de fusión
        "Lv": 2260000,       # Calor latente de vaporización
        "Tf": 0.0,           # Temperatura de fusión (°C)
        "Tv": 100.0,         # Temperatura de ebullición (°C)
        "k": 0.60,           # Conductividad térmica
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
    # ... (código de Conversor)
    
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
    **Explicación Física:** Las escalas Kelvin y Rankine son **escalas absolutas** (0 K y 0 R representan el cero absoluto), mientras que Celsius y Fahrenheit se basan en puntos de referencia. El Cero Absoluto es $0 K \\approx -273.15 °C$.
    """)

# 3.2. Simulación de Equilibrio Térmico (Sin Cambio de Fase)
def seccion_equilibrio_termico_sin_fase():
    st.header("2️⃣ Equilibrio Térmico (Sistema sin Cambio de Fase)")
    st.markdown("Simulación de la mezcla de dos cuerpos **sin cambio de fase** hasta alcanzar la **temperatura final de equilibrio**.")

    st.subheader("Parámetros de los Cuerpos")
    
    materiales = list(CONSTANTES.keys())
    
    # Interfaz para 2 cuerpos
    st.markdown("##### Cuerpo 1")
    col1, col2, col3 = st.columns(3)
    with col1:
        m1 = st.number_input("Masa $m_1$ (kg)", 0.1, 100.0, 1.0, 0.1, key='m1_2')
    with col2:
        ce1_key = st.selectbox("Material 1", materiales, index=2, key='mat1_2') # Aluminio
    with col3:
        T1 = st.number_input("Temperatura Inicial $T_1$ (°C)", -50.0, 500.0, 100.0, 1.0, key='T1_2')
    
    st.markdown("##### Cuerpo 2")
    col4, col5, col6 = st.columns(3)
    with col4:
        m2 = st.number_input("Masa $m_2$ (kg)", 0.1, 100.0, 2.0, 0.1, key='m2_2')
    with col5:
        ce2_key = st.selectbox("Material 2", materiales, index=0, key='mat2_2') # Agua
    with col6:
        T2 = st.number_input("Temperatura Inicial $T_2$ (°C)", -50.0, 500.0, 20.0, 1.0, key='T2_2')

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
    Q1 = m1 * ce1 * (Tf - T1) 
    Q2 = m2 * ce2 * (Tf - T2) 
    
    st.divider()

    st.subheader("Resultados y Gráfico de Equilibrio")

    col_Tf, col_Q_info = st.columns(2)
    with col_Tf:
        st.metric("Temperatura Final de Equilibrio $T_f$ (°C)", f"{Tf:.2f}")

    with col_Q_info:
        st.markdown(f"**Calor Específico de {ce1_key} ($c_{{e1}}$):** ${ce1} \\ J/kg\\cdot°C$")
        st.markdown(f"**Calor Específico de {ce2_key} ($c_{{e2}}$):** ${ce2} \\ J/kg\\cdot°C$")
        st.markdown(f"**Calor Perdido/Ganado por {ce1_key}:** ${Q1:,.0f}\ J$")
        st.markdown(f"**Calor Perdido/Ganado por {ce2_key}:** ${Q2:,.0f}\ J$")

    # Visualización con Plotly
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

    st.info(r"""
    **Explicación Física:** Se aplica la **Conservación de la Energía** en un sistema aislado: el calor total perdido es igual al calor total ganado ($\Sigma Q = 0$). La temperatura de equilibrio es el promedio ponderado por la capacidad calorífica ($m \cdot c_e$) de cada cuerpo.
    """)

# 3.3. Procesos Térmicos por Etapas (Q Total) - CUMPLIENDO REQUISITO
def seccion_cambio_fase_q_total():
    st.header("3️⃣ Procesos Térmicos por Etapas (Calor Total $Q$)")
    st.markdown("Calcula el **calor total** ($Q$) necesario para llevar una masa de agua a través de etapas de calentamiento y cambio de fase (fusión, ebullición).")

    # Parámetros de entrada (Usando number_input sin límite superior)
    col1, col2 = st.columns(2)
    with col1:
        m_agua = st.number_input("Masa de Agua $m$ (kg)", 0.1, None, 1.0, 0.1, key='m_q_total')
    with col2:
        T_inicial = st.number_input("Temperatura Inicial $T_i$ (°C)", -273.15, None, 20.0, 1.0, key='Ti_q_total')
        T_final = st.number_input("Temperatura Final $T_f$ (°C)", -273.15, None, 110.0, 1.0, key='Tf_q_total')
    
    if T_inicial >= T_final:
        st.warning("Para esta simulación, por favor configure $T_i < T_f$ (Proceso de Calentamiento).")
        return

    # Constantes del agua
    C = CONSTANTES["Agua"]
    Tf_C = C["Tf"]     # 0 °C
    Tv_C = C["Tv"]     # 100 °C
    
    st.divider()
    st.subheader("Cálculo del Calor Total por Etapas:")

    # Inicializar calor total y lista de etapas
    etapas_q = []
    T_puntos = [T_inicial]
    Q_acumulado = [0.0]
    T_actual = T_inicial

    def add_calor(m, ce, delta_T, nombre_etapa):
        Q = m * ce * delta_T
        Q_total_new = Q_acumulado[-1] + Q
        etapas_q.append((nombre_etapa, Q))
        T_puntos.append(T_actual + delta_T)
        Q_acumulado.append(Q_total_new)
        return Q_total_new

    def add_calor_fase(m, L, nombre_fase, T_fase):
        Q = m * L
        Q_total_new = Q_acumulado[-1] + Q
        etapas_q.append((nombre_fase, Q))
        T_puntos.append(T_fase)
        Q_acumulado.append(Q_total_new)
        return Q_total_new

    
    # 1. Calentamiento Sólido (Hielo)
    if T_actual < Tf_C:
        T_target = min(T_final, Tf_C)
        add_calor(m_agua, C["ce_solido"], T_target - T_actual, 
                  f"1. Calentamiento Sólido ($T={T_actual:.0f}\\to{T_target:.0f} \\ °C$)")
        T_actual = T_target
        
    # 2. Fusión (Cambio de Fase Sólido a Líquido)
    if T_actual == Tf_C and T_final > Tf_C:
        add_calor_fase(m_agua, C["Lf"], f"2. Fusión (Cambio de Fase) $T={Tf_C:.0f} \\ °C$", Tf_C)
        
    T_actual = T_puntos[-1] 
    
    # 3. Calentamiento Líquido (Agua)
    if T_actual < Tv_C:
        T_target = min(T_final, Tv_C)
        add_calor(m_agua, C["ce_liquido"], T_target - T_actual, 
                  f"3. Calentamiento Líquido ($T={T_actual:.0f}\\to{T_target:.0f} \\ °C$)")
        T_actual = T_target

    # 4. Vaporización (Cambio de Fase Líquido a Gas)
    if T_actual == Tv_C and T_final > Tv_C:
        add_calor_fase(m_agua, C["Lv"], f"4. Vaporización (Cambio de Fase) $T={Tv_C:.0f} \\ °C$", Tv_C)
        
    T_actual = T_puntos[-1]

    # 5. Calentamiento Gas (Vapor)
    if T_actual < T_final:
        add_calor(m_agua, C["ce_gas"], T_final - T_actual, 
                  f"5. Calentamiento Vapor ($T={T_actual:.0f}\\to{T_final:.0f} \\ °C$)")
        
    # 6. Mostrar Resultados
    Q_total = Q_acumulado[-1]
    st.metric("Calor Total Requerido $Q_{total}$ (Joule)", f"{Q_total:,.0f}")

    # Mostrar detalle de etapas
    etapas_display = [f"**{nombre}:** ${Q:,.0f}\ J$" for nombre, Q in etapas_q if abs(Q) > 1]
    for item in etapas_display:
        st.markdown(item)

    # Gráfico de la Curva de Calentamiento (Plotly)
    fig = go.Figure(data=[
        go.Scatter(x=Q_acumulado, y=T_puntos, mode='lines+markers', name='Curva de Calentamiento',
                   line=dict(color='orange', width=3))
    ])
    fig.update_layout(
        title='Curva de Calentamiento: Temperatura vs. Calor Añadido',
        xaxis_title='Calor Añadido $Q$ (Joule)',
        yaxis_title='Temperatura $T$ (°C)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(r"""
    **Explicación Física:** El calor se usa para dos fines: **aumentar la temperatura** ($Q=mc_e\Delta T$) o **cambiar la fase** ($Q=mL$). Los tramos horizontales representan los cambios de fase (fusión y ebullición) donde se absorbe calor latente ($L$) sin cambiar la temperatura.
    """)


# 3.4. Mezcla de Sustancias con Cambio de Fase (NUEVO REQUISITO)
def seccion_mezcla_cambio_fase():
    st.header("4️⃣ Equilibrio Térmico con Cambio de Fase (Mezcla $T_f=0°C$)")
    st.markdown("Simulación de la mezcla de un **Metal** caliente con **Agua/Hielo** fría, considerando la **fusión** del hielo en el proceso de equilibrio.")
    
    st.subheader("Parámetros de la Mezcla")
    
    materiales_metal = {k: v['ce'] for k, v in CONSTANTES.items() if 'ce' in v and k != "Agua"}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Cuerpo 1: Agua / Hielo")
        m_agua = st.number_input("Masa de Agua/Hielo $m_1$ (kg)", 0.1, None, 1.0, 0.1, key='m1_fase')
        T_agua = st.number_input("Temperatura Inicial $T_1$ (°C)", -20.0, 50.0, 5.0, 1.0, key='T1_fase')
        ce_agua = CONSTANTES['Agua']['ce_liquido'] if T_agua >= 0 else CONSTANTES['Agua']['ce_solido']
    
    with col2:
        st.markdown("##### Cuerpo 2: Metal Caliente")
        m_metal = st.number_input("Masa de Metal $m_2$ (kg)", 0.1, None, 0.5, 0.1, key='m2_fase')
        material_metal = st.selectbox("Material del Metal", list(materiales_metal.keys()), index=0, key='mat2_fase') # Cobre
        T_metal = st.number_input("Temperatura Inicial $T_2$ (°C)", 50.0, 500.0, 150.0, 1.0, key='T2_fase')
        ce_metal = materiales_metal[material_metal]

    C_agua = CONSTANTES['Agua']
    T_ref = C_agua['Tf'] # 0 °C
    L_f = C_agua['Lf']
    c_hielo = C_agua['ce_solido']
    c_agua_liq = C_agua['ce_liquido']

    st.divider()

    # --- Lógica de Equilibrio con Cambio de Fase (Hielo/Agua) ---
    
    # 1. Calcular el Calor para llevar cada sustancia a T_ref = 0°C
    
    # Calor que el metal PIERDE/GANA para llegar a 0°C (Q_M < 0 si T_metal > 0)
    Q_metal_to_ref = m_metal * ce_metal * (T_ref - T_metal) # Siempre negativo si T_metal > 0

    # Calor que el agua GANA/PIERDE para llegar a 0°C (Q_W > 0 si T_agua < 0)
    if T_agua < T_ref: # Hielo
        Q_agua_to_ref = m_agua * c_hielo * (T_ref - T_agua) 
    else: # Agua Líquida
        Q_agua_to_ref = m_agua * c_agua_liq * (T_ref - T_agua) # Negativo si T_agua > 0
    
    # Calor NETO disponible para el proceso de Fusión/Congelación
    Q_neto_disponible = Q_metal_to_ref + Q_agua_to_ref
    
    # Calor Requerido para Fundir Toda la Masa de Hielo/Agua Líquida
    if T_agua < T_ref:
        # Si hay hielo, se necesita calor para fundirlo
        Q_fusion_req = m_agua * L_f 
        # El Q_neto_disponible es el calor que GANA el sistema
        
        # Scenario 1: No hay suficiente calor para fundir todo el hielo.
        if Q_neto_disponible < Q_fusion_req:
            Tf = T_ref
            m_fundida = Q_neto_disponible / L_f if Q_neto_disponible > 0 else 0
            resultado_msg = f"El hielo se calienta a **$0°C$**. Solo se funde una masa de **${m_fundida:.3f}\ kg$**."
            Q_metal_transferido = -Q_metal_to_ref
            
        # Scenario 2: Hay suficiente calor para fundir todo el hielo y calentar el agua.
        else:
            Q_remanente = Q_neto_disponible - Q_fusion_req
            Tf = Q_remanente / (m_agua * c_agua_liq + m_metal * ce_metal)
            resultado_msg = f"Todo el hielo se funde. La temperatura final es **$T_f = {Tf:.2f}°C$** (Agua Líquida + Metal)."
            Q_metal_transferido = Q_metal_to_ref + m_agua * L_f * -1 # Q ganado para fundir + Q ganado para calentar. O más simple: Q perdido por metal = Q ganado por agua.
    
    else:
        # Caso simple: No hay cambio de fase (todo líquido > 0°C) o potencial congelación.
        
        # 1. Calcular Tf asumiendo NO phase change (si no está cerca de 0)
        Tf_simple = (m_agua * c_agua_liq * T_agua + m_metal * ce_metal * T_metal) / (m_agua * c_agua_liq + m_metal * ce_metal)
        
        if Tf_simple > T_ref:
            Tf = Tf_simple
            resultado_msg = f"No hay fase de hielo. La temperatura final es **$T_f = {Tf:.2f}°C$** (Agua Líquida + Metal)."
        else:
            # Si el cálculo simple sugiere Tf < 0, significa que puede haber congelación
            Q_total_para_congelar = m_agua * L_f
            
            # El calor neto (que necesita ser absorbido para congelarse) es Q_neto_disponible
            if Q_neto_disponible < -Q_total_para_congelar:
                # Caso extremo: Todo el agua se congela y baja de 0°C
                Q_remanente = Q_neto_disponible - (-Q_total_para_congelar) # Q_remanente es negativo
                Tf = Q_remanente / (m_agua * c_hielo + m_metal * ce_metal)
                resultado_msg = f"Toda el agua se congela. La temperatura final es **$T_f = {Tf:.2f}°C$** (Hielo + Metal)."
                
            else:
                # Tf = 0, congelación parcial
                Tf = T_ref
                Q_disponible_abs = abs(Q_neto_disponible)
                m_congelada = Q_disponible_abs / L_f 
                resultado_msg = f"El agua se enfría a **$0°C$**. Solo se congela una masa de **${m_congelada:.3f}\ kg$**."
                
    # Recalcular el calor transferido para el caso final
    Q_metal_final = m_metal * ce_metal * (Tf - T_metal)
    Q_agua_final = -Q_metal_final # Conservación de la energía

    st.subheader("Resultados de la Simulación")
    st.metric("Temperatura Final de Equilibrio $T_f$ (°C)", f"{Tf:.2f}")
    st.info(f"**Conclusión:** {resultado_msg}")
    
    col_Q_final, col_ce_info = st.columns(2)
    with col_Q_final:
        st.markdown(f"**Calor Absorbido por Agua/Hielo:** ${-Q_metal_final:,.0f}\ J$")
        st.markdown(f"**Calor Cedido por {material_metal}:** ${Q_metal_final:,.0f}\ J$")
    with col_ce_info:
        st.markdown(f"**Calor Específico del Metal:** ${ce_metal} \\ J/kg\\cdot°C$")
        st.markdown(f"**Calor Latente de Fusión del Agua:** ${L_f:,.0f} \\ J/kg$")

    # Visualización con Matplotlib (Gráfico de Distribución de Calor)
    fig, ax = plt.subplots()
    q_data = [-Q_metal_final, -Q_agua_final]
    labels = [material_metal, "Agua/Hielo"]
    colors = ['orange', 'skyblue']

    # Normalizar los datos para mostrar el % de calor total transferido
    total_q_abs = abs(Q_metal_final)
    ax.pie([total_q_abs, total_q_abs], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal') # Asegura que el gráfico de pastel sea un círculo.

    ax.set_title(f'Distribución de Energía ($Q$) en el Equilibrio (Total: {total_q_abs:,.0f} J)')
    st.pyplot(fig)
    

    st.info(r"""
    **Explicación Física:** La complejidad de esta mezcla radica en el **calor latente de fusión** ($L_f$). Para que la temperatura final ($T_f$) sea **mayor a $0°C$**, el calor cedido por el metal ($Q_{cedido}$) debe ser suficiente no solo para calentar el hielo a $0°C$ sino también para suministrar el $Q_{fusión} = m_{hielo} L_f$ necesario para fundir todo el hielo. Si $Q_{cedido} < Q_{fusión}$, la temperatura de equilibrio queda anclada en $T_f = 0°C$, con una mezcla de agua y hielo.
    """)


# 3.5. Simulación de Conducción de Calor 1D (Barra)
def seccion_conduccion_1d():
    st.header("5️⃣ Conducción de Calor en Barra (1D)")
    # ... (código de Conducción 1D)
    
    col1, col2 = st.columns(2)
    materiales_k = {k: v['k'] for k, v in CONSTANTES.items() if 'k' in v}

    with col1:
        L = st.number_input("Longitud de la Barra $L$ (m)", 0.1, 5.0, 1.0, 0.1, key='L_1d')
        material = st.selectbox("Material de la Barra", list(materiales_k.keys()), index=1, key='mat_1d')
        k = materiales_k[material]
        st.markdown(f"Conductividad Térmica $k$ (W/m·K): **{k}**")

    with col2:
        T_caliente = st.number_input("Temperatura Extremo Caliente $T_H$ (°C)", 50.0, 1000.0, 150.0, 1.0, key='Th_1d')
        T_frio = st.number_input("Temperatura Extremo Frío $T_C$ (°C)", 0.0, 500.0, 20.0, 1.0, key='Tc_1d')
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
        
        # Malla y parámetros para la simulación temporal
        Nx = 50 
        dx = L / Nx
        alpha_sim = 1e-4 
        dt = 0.5 * dx**2 / (2 * alpha_sim) 
        
        T_curr = np.full(Nx, T_frio)
        T_curr[0] = T_caliente 

        T_frames = [T_curr.copy()]
        num_pasos = 100 

        for _ in range(num_pasos):
            T_next = T_curr.copy()
            for i in range(1, Nx - 1):
                # Ecuación de conducción de calor 1D discretizada
                T_next[i] = T_curr[i] + alpha_sim * dt / (dx**2) * (T_curr[i+1] - 2 * T_curr[i] + T_curr[i-1])
            
            T_next[0] = T_caliente
            T_next[-1] = T_frio 
            
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

    st.info(fr"""
    **Explicación Física:** La **Ley de Fourier** rige la conducción: $P = -k A \frac{{dT}}{{dx}}$. En el **Estado Estacionario**, el perfil $T(x)$ es lineal. En la **Evolución Temporal**, la $\frac{{\partial T}}{{\partial t}} = \alpha \frac{{\partial^2 T}}{{\partial x^2}}$ muestra que materiales con alta conductividad ($k$) como el **Cobre** ($k={CONSTANTES['Cobre']['k']} W/m·K$) difunden el calor más rápido que el **Vidrio** ($k={CONSTANTES['Vidrio']['k']} W/m·K$).
    """)


# 3.6. Simulación de Conducción de Calor 2D
def seccion_conduccion_2d():
    st.header("6️⃣ Conducción de Calor en Placa (2D Simplificada)")
    # ... (código de Conducción 2D)
    
    st.warning("La simulación 2D utiliza el método de relajación (Diferencias Finitas) para encontrar la solución de estado estacionario de la Ecuación de Laplace ($\nabla^2 T = 0$).")

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
    for _ in range(T_max_iter):
        T_old = T.copy()
        for i in range(1, L_placa - 1):
            for j in range(1, L_placa - 1):
                T[i, j] = 0.25 * (T_old[i+1, j] + T_old[i-1, j] + T_old[i, j+1] + T_old[i, j-1])
        T[0, :] = T_borde_top
        T[-1, :] = T_borde_bottom
        T[:, 0] = T_borde_left
        T[:, -1] = T_borde_right

    # 3. Visualización con Plotly (Heatmap 2D)
    st.subheader("Distribución de Temperatura en Estado Estacionario (Heatmap)")
    
    T_display = np.flipud(T) 

    fig = px.imshow(T_display, 
                    color_continuous_scale=px.colors.sequential.Inferno, 
                    aspect="equal",
                    labels=dict(color="Temperatura (°C)"),
                    zmin=0, zmax=300
                    )
    
    fig.update_layout(
        title='Mapa de Calor 2D (Conducción)',
        xaxis=dict(title='Posición X'), 
        yaxis=dict(title='Posición Y', scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(r"""
    **Explicación Física:** En estado estacionario, la distribución de temperatura está gobernada por la **Ecuación de Laplace** ($\nabla^2 T = 0$). El método iterativo (relajación) simula el equilibrio donde la temperatura de cada punto es el promedio de sus vecinos, satisfaciendo la condición de no acumulación de calor.
    """)

# --- 4. Función Principal de la Aplicación ---

def main():
    st.sidebar.title("Menú de Simulación")
    
    opciones = {
        "1. Conversor de Temperaturas": seccion_conversor,
        "2. Equilibrio Térmico (Sin Cambio de Fase)": seccion_equilibrio_termico_sin_fase,
        "3. Procesos Térmicos por Etapas (Q Total)": seccion_cambio_fase_q_total,
        "4. Equilibrio con Cambio de Fase": seccion_mezcla_cambio_fase,
        "5. Conducción de Calor 1D": seccion_conduccion_1d,
        "6. Conducción de Calor 2D": seccion_conduccion_2d,
    }

    seleccion = st.sidebar.selectbox("Seleccione la Simulación:", list(opciones.keys()))
    
    st.title("🌡️ Simulador Interactivo de Termodinámica")
    
    # Ejecutar la función correspondiente a la selección
    opciones[seleccion]()

if __name__ == "__main__":
    main()
