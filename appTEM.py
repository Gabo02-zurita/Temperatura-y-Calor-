import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

# --- DATOS FÍSICOS (Basados en Física Universitaria) ---
# Constantes físicas y propiedades de materiales (Valores típicos)
MATERIAL_PROPERTIES = {
    "Agua (Líquido)": {"c": 4186, "Lf": 334e3, "Lv": 2256e3, "Tf": 0, "Tv": 100, "rho": 1000},
    "Hielo (Sólido)": {"c": 2050, "Lf": 334e3, "Tv": 100, "Tf": 0, "rho": 920},
    "Vapor (Gas)": {"c": 2010, "Lv": 2256e3, "Tv": 100, "Tf": 0, "rho": 0.6},
    "Aluminio": {"c": 900, "k": 205.0, "rho": 2700},
    "Cobre": {"c": 390, "k": 385.0, "rho": 8960},
    "Concreto": {"c": 880, "k": 1.1, "rho": 2400},
    "Vidrio": {"c": 840, "k": 0.8, "rho": 2500},
}

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Simulador de Termodinámica y Calor",
    layout="wide",
)

st.title("🔥 Física del Calor y la Temperatura (Simulador Interactivo)")
st.caption("Módulos basados en los fundamentos de Termodinámica y Transferencia de Calor.")
st.markdown("---")

# --- FUNCIONES DE LOS MÓDULOS ---

# 1. CONVERSIÓN DE ESCALAS
def modulo_conversion():
    st.header("1. Conversión Dinámica de Escalas Termométricas")
    st.markdown("Conversión entre Celsius (**°C**), Kelvin (**K**), Fahrenheit (**°F**) y Rankine (**°R**).")

    # Controles Interactivos
    T_C = st.slider("Temperatura de entrada (°C)", min_value=-300.0, max_value=1000.0, value=20.0, step=0.1)

    # Fórmulas de Conversión
    T_K = T_C + 273.15
    T_F = (T_C * 9/5) + 32
    T_R = T_F + 459.67 # T_R = (9/5) * T_K

    st.subheader("Resultados")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Celsius (°C)", f"{T_C:.2f}")
    col2.metric("Kelvin (K)", f"{T_K:.2f}")
    col3.metric("Fahrenheit (°F)", f"{T_F:.2f}")
    col4.metric("Rankine (°R)", f"{T_R:.2f}")

    st.markdown("""
    **Explicación Física:** La escala **Kelvin** es la escala absoluta, donde $0 K$ es el cero absoluto. Las diferencias de temperatura en K y °C son idénticas ($\Delta T_K = \Delta T_C$).
    """)
    

# 2. EQUILIBRIO TÉRMICO (CALORIMETRÍA SIMPLE)
def modulo_equilibrio():
    st.header("2. Simulación de Equilibrio Térmico (2 o 3 Cuerpos)")
    st.markdown("Calcula la temperatura de equilibrio ($T_f$) de la mezcla, asumiendo $Q_{neto}=0$.")
    

    num_cuerpos = st.slider("Número de Cuerpos a Mezclar", min_value=2, max_value=3, value=2)
    st.markdown("---")

    sum_mc_Ti = 0
    sum_mc = 0
    datos = []
    
    # Lógica de entrada de datos para N cuerpos
    for i in range(num_cuerpos):
        st.subheader(f"Cuerpo {i+1}")
        
        # Selección de material
        opciones_material = [k for k in MATERIAL_PROPERTIES.keys() if "Vapor" not in k and "Hielo" not in k]
        material = st.selectbox(f"Material Cuerpo {i+1}", opciones_material, index=i % len(opciones_material), key=f"mat{i}")
        
        # Parámetros (Controles con límites)
        c_default = MATERIAL_PROPERTIES.get(material, {}).get("c", 4186.0)
        c = st.number_input(f"Calor Específico ($c_{i+1}$ en J/kg·K)", value=c_default, min_value=1.0, max_value=10000.0, step=1.0, key=f"c{i}")
        m = st.slider(f"Masa ($m_{i+1}$ en kg)", value=1.0, min_value=0.1, max_value=5.0, step=0.1, key=f"m{i}")
        Ti = st.slider(f"Temperatura Inicial ($T_{{i,{i+1}}}$ en °C)", value=(10.0 if i == 0 else 90.0), min_value=-50.0, max_value=120.0, step=1.0, key=f"Ti{i}")

        datos.append({"m": m, "c": c, "Ti": Ti, "material": material})
        
        sum_mc_Ti += m * c * Ti
        sum_mc += m * c

    if st.button("Calcular Temperatura de Equilibrio ($T_f$)", key="btn_eq"):
        if sum_mc > 0:
            # Fórmula de Equilibrio Térmico
            Tf = sum_mc_Ti / sum_mc
            st.success(f"La **Temperatura Final de Equilibrio ($T_f$)** es: **{Tf:.2f} °C**")

            # Visualización de las temperaturas
            fig_temp = go.Figure()
            temps_iniciales = [d['Ti'] for d in datos]
            nombres = [d['material'] for d in datos]
            
            fig_temp.add_trace(go.Bar(
                x=[f"Cuerpo {i+1} ({nombres[i]})" for i in range(num_cuerpos)], 
                y=temps_iniciales, 
                name='Temperatura Inicial',
                marker_color='lightblue'
            ))
            fig_temp.add_trace(go.Scatter(
                x=[f"Cuerpo {i+1} ({nombres[i]})" for i in range(num_cuerpos)], 
                y=[Tf] * num_cuerpos, 
                mode='lines', 
                name='Temperatura Final de Equilibrio',
                line=dict(color='red', dash='dash')
            ))
            
            fig_temp.update_layout(title="Temperaturas Iniciales vs. Equilibrio", yaxis_title="Temperatura (°C)")
            st.plotly_chart(fig_temp, use_container_width=True)
    
    st.markdown("""
    **Fundamento Teórico (Capítulo 17 - Calorimetría):**
    El principio es la **conservación de la energía**: el calor total neto transferido es cero. La temperatura final está ponderada por las capacidades caloríficas ($m \cdot c$) de cada cuerpo.
    """)


# 3. CAMBIO DE FASE Y PROCESOS POR ETAPAS
def modulo_cambio_fase():
    st.header("3. Calor Total en Procesos por Etapas (Cambio de Fase del Agua)")
    st.markdown("Calcula el calor total ($Q$) para el Agua ($H_2O$) desde $T_i$ a $T_f$, considerando fusión y vaporización (Calor Latente).")
    
    st.subheader("Parámetros del Proceso")
    
    col1, col2, col3 = st.columns(3)
    masa = col1.slider("Masa ($m$ en kg)", value=1.0, min_value=0.1, max_value=5.0, step=0.1, key="m_fase")
    T_inicial = col2.slider("Temperatura Inicial ($T_i$ en °C)", value=-10.0, min_value=-50.0, max_value=120.0, step=1.0)
    T_final = col3.slider("Temperatura Final ($T_f$ en °C)", value=110.0, min_value=-50.0, max_value=120.0, step=1.0)
    
    # Constantes del Agua
    c_hielo = MATERIAL_PROPERTIES["Hielo (Sólido)"]["c"]
    c_agua = MATERIAL_PROPERTIES["Agua (Líquido)"]["c"]
    c_vapor = MATERIAL_PROPERTIES["Vapor (Gas)"]["c"]
    L_f = MATERIAL_PROPERTIES["Agua (Líquido)"]["Lf"]
    L_v = MATERIAL_PROPERTIES["Agua (Líquida)"]["Lv"]
    T_f = 0.0         # °C
    T_v = 100.0       # °C
    
    Q_total = 0
    etapas = []
    
    if T_inicial >= T_final:
        st.error("La temperatura final debe ser mayor que la inicial para un proceso de calentamiento.")
        return

    T_actual = T_inicial
    
    # 1. Calentamiento como Hielo (hasta 0°C)
    if T_actual < T_f:
        T_limite = min(T_final, T_f)
        Q_hielo = masa * c_hielo * (T_limite - T_actual)
        Q_total += Q_hielo
        etapas.append({"Q": Q_hielo, "Desc": f"Calentamiento Hielo ({T_actual:.1f} a {T_limite:.1f}°C)", "Tipo": "Sensible"})
        T_actual = T_limite

    # 2. Fusión (a 0°C)
    if T_actual == T_f and T_final >= T_f:
        Q_fusion = masa * L_f
        Q_total += Q_fusion
        etapas.append({"Q": Q_fusion, "Desc": f"Fusión (Calor Latente) a {T_f:.1f}°C", "Tipo": "Latente"})
    
    # 3. Calentamiento como Agua Líquida (de 0°C hasta 100°C)
    if T_actual < T_final and T_actual < T_v:
        T_inicio_liq = max(T_actual, T_f) # Asegura que inicia en 0 si hubo fusión
        T_limite = min(T_final, T_v)
        Q_agua = masa * c_agua * (T_limite - T_inicio_liq)
        Q_total += Q_agua
        etapas.append({"Q": Q_agua, "Desc": f"Calentamiento Agua ({T_inicio_liq:.1f} a {T_limite:.1f}°C)", "Tipo": "Sensible"})
        T_actual = T_limite

    # 4. Vaporización (a 100°C)
    if T_actual == T_v and T_final >= T_v:
        Q_vaporizacion = masa * L_v
        Q_total += Q_vaporizacion
        etapas.append({"Q": Q_vaporizacion, "Desc": f"Vaporización (Calor Latente) a {T_v:.1f}°C", "Tipo": "Latente"})
    
    # 5. Calentamiento como Vapor (si T_f > 100°C)
    if T_actual == T_v and T_final > T_v:
        Q_vapor = masa * c_vapor * (T_final - T_v)
        Q_total += Q_vapor
        etapas.append({"Q": Q_vapor, "Desc": f"Calentamiento Vapor ({T_v:.1f} a {T_final:.1f}°C)", "Tipo": "Sensible"})

    st.markdown("---")
    st.metric("Calor Total Requerido ($Q_{total}$)", f"{Q_total/1000:.2f} kJ", f"({Q_total:.2f} J)")

    # Visualización (Gráfico de la Curva de Calentamiento)
    Q_plot = [0]
    T_plot = [T_inicial]
    Q_acumulado = 0
    
    # Se añade cada etapa para la gráfica
    for etapa in etapas:
        Q_acumulado += etapa["Q"] / 1000 # kJ
        T_anterior = T_plot[-1]
        
        if etapa["Tipo"] == "Sensible":
            # Calentamiento (diagonal)
            T_plot.append(T_anterior)
            Q_plot.append(Q_acumulado - (etapa["Q"] / 1000) * 0.01) # Pequeño paso inicial
            
            T_final_etapa = float(etapa["Desc"].split(' a ')[1].split('°C')[0])
            T_plot.append(T_final_etapa)
            Q_plot.append(Q_acumulado)
            
        elif etapa["Tipo"] == "Latente":
            # Cambio de fase (plano)
            T_plot.append(T_anterior)
            Q_plot.append(Q_acumulado)
            
    fig_curva = go.Figure()
    fig_curva.add_trace(go.Scatter(x=Q_plot, y=T_plot, mode='lines+markers', name='Curva de Calentamiento'))
    
    fig_curva.update_layout(
        title='Curva de Calentamiento (Temperatura vs. Calor Suministrado)',
        xaxis_title='Calor Acumulado (kJ)',
        yaxis_title='Temperatura (°C)',
        height=450
    )
    st.plotly_chart(fig_curva, use_container_width=True)


# 4. CONDUCCIÓN DE CALOR 1D
def modulo_conduccion_1d():
    st.header("4. Simulación de Conducción de Calor 1D (Barra)")
    st.markdown("Modela el flujo de calor ($H$) y el perfil de temperatura en estado estacionario (Ley de Fourier).")
    

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Parámetros del Objeto")
        material_k = st.selectbox("Material (Determina k)", ["Aluminio", "Cobre", "Concreto", "Vidrio"], key="mat_k")
        k_default = MATERIAL_PROPERTIES[material_k].get("k", 1.0)
        k = st.slider("Conductividad Térmica ($k$ en $W/m·K$)", value=k_default, min_value=0.01, max_value=400.0, step=0.1)
        A = st.slider("Área de la sección transversal ($A$ en $m^2$)", value=0.01, min_value=0.001, max_value=1.0, step=0.001)
        L = st.slider("Longitud de la barra ($L$ en m)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)
        
    with col2:
        st.subheader("Condiciones de Frontera")
        TH = st.slider("Temperatura Lado Caliente ($T_H$ en °C)", value=100.0, min_value=-50.0, max_value=500.0, step=1.0)
        TC = st.slider("Temperatura Lado Frío ($T_C$ en °C)", value=20.0, min_value=-50.0, max_value=500.0, step=1.0)

    if TH <= TC:
        st.warning("La temperatura caliente ($T_H$) debe ser mayor a la fría ($T_C$) para ver un flujo de calor de izquierda a derecha.")
        
    # Cálculo de la Tasa de Flujo de Calor (H)
    H = k * A * (TH - TC) / L

    st.markdown("---")
    st.success(f"Tasa de Flujo de Calor (Corriente de Calor) ($H$): **{H:.2f} W** (J/s)")

    # Gráfica del perfil de temperatura
    st.subheader("Perfil de Temperatura en Estado Estacionario")
    x_pos = np.linspace(0, L, 100)
    T_x = TH - (TH - TC) * (x_pos / L)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_pos, y=T_x, mode='lines', name='Perfil de Temperatura', line=dict(color='red', width=3)))
    fig.update_layout(
        title=f'Distribución Lineal de Temperatura T(x) - Material: {material_k}',
        xaxis_title='Posición (x) en la Barra (m)',
        yaxis_title='Temperatura (T) en °C',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Fundamento Teórico (Capítulo 17 - Conducción):**
    En **estado estacionario**, la distribución de temperatura es lineal. El flujo de calor es proporcional a la diferencia de temperatura, al área y a la conductividad térmica ($k$), e inversamente proporcional a la longitud ($L$): $H = k A (T_H - T_C) / L$.
    """)


# 5. CONDUCCIÓN DE CALOR 2D SIMPLIFICADA
def modulo_conduccion_2d():
    st.header("5. Conducción 2D Simplificada (Placa Cuadrada)")
    st.markdown("Simulación del perfil de temperatura en una placa en estado estacionario (Ecuación de Laplace).")
    

    col1, col2 = st.columns(2)
    with col1:
        N = st.slider("Resolución de la Placa (N x N)", min_value=30, max_value=100, value=50, key="N_2d")
        st.subheader("Condiciones de Borde Superior e Inferior")
        T_superior = st.number_input("Borde Superior (°C)", value=100.0, min_value=0.0, max_value=200.0, key='Ts')
        T_inferior = st.number_input("Borde Inferior (°C)", value=20.0, min_value=0.0, max_value=200.0, key='Ti')
    
    with col2:
        st.subheader("Condiciones de Borde Izquierdo y Derecho")
        T_izquierdo = st.number_input("Borde Izquierdo (°C)", value=50.0, min_value=0.0, max_value=200.0, key='Tiz')
        T_derecho = st.number_input("Borde Derecho (°C)", value=50.0, min_value=0.0, max_value=200.0, key='Tde')

    # Inicialización con temperatura promedio para la simulación
    T = np.full((N, N), (T_superior + T_inferior + T_izquierdo + T_derecho) / 4)
    
    # Condiciones de contorno
    T[0, :] = T_superior
    T[N-1, :] = T_inferior
    T[:, 0] = T_izquierdo
    T[:, N-1] = T_derecho
    
    # Iteración de Jacobi (simplificación para el perfil estacionario)
    max_iter = 100
    for _ in range(max_iter): 
        T_new = T.copy()
        T_new[1:N-1, 1:N-1] = 0.25 * (T[2:N, 1:N-1] + T[0:N-2, 1:N-1] + T[1:N-1, 2:N] + T[1:N-1, 0:N-2])
        # Reaplicar contorno (importante para la simulación)
        T_new[0, :] = T_superior
        T_new[N-1, :] = T_inferior
        T_new[:, 0] = T_izquierdo
        T_new[:, N-1] = T_derecho
        T = T_new

    # Visualización (Plotly Heatmap)
    fig = go.Figure(data=go.Heatmap(
        z=T,
        colorscale='Jet', # Jet ofrece un buen rango visual para temperatura
        zmin=min(T_inferior, T_superior, T_izquierdo, T_derecho),
        zmax=max(T_inferior, T_superior, T_izquierdo, T_derecho)
    ))
    
    fig.update_layout(
        title='Mapa de Calor 2D de la Distribución de Temperatura',
        xaxis_title='Posición X',
        yaxis_title='Posición Y',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Fundamento Teórico (Caso Extendido):**
    En estado estacionario (tiempo largo), la distribución de temperatura se rige por la **Ecuación de Laplace** ($\nabla^2 T = 0$). El calor fluye perpendicularmente a las líneas isotérmicas (líneas de igual temperatura) mostradas en el mapa de calor.
    """)


# --- ESTRUCTURA PRINCIPAL DE STREAMLIT (Tabs) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Conversión de Escalas", 
    "2. Equilibrio Térmico (N Cuerpos)", 
    "3. Calor + Cambio de Fase (Etapas)", 
    "4. Conducción 1D (Barra)",
    "5. Conducción 2D (Placa)"
])

with tab1:
    modulo_conversion()

with tab2:
    modulo_equilibrio()

with tab3:
    modulo_cambio_fase()

with tab4:
    modulo_conduccion_1d()

with tab5:
    modulo_conduccion_2d()
