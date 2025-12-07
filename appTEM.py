import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.constants import convert_temperature

# --- 1. Configuración y Constantes ---

st.set_page_config(layout="wide", page_title="Simulador temperatura,calor y transferencia térmica🌡️")

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
    st.header("1️⃣ Conversor Avanzado de Escalas de Temperatura")
    st.markdown("Convierte valores entre **Celsius (°C)**, **Kelvin (K)**, **Fahrenheit (°F)** y **Rankine (°R)**.")
    
    with st.form("form_conversor"):
        col_sel, col_val, col_minmax = st.columns([1, 2, 3])
        
        with col_sel:
            escala_entrada = st.selectbox("Escala de Entrada", ['C', 'K', 'F', 'R'], index=0, key='sel_escala_in')
        
        with col_val:
            # Usamos number_input sin límites superiores
            valor_entrada = st.number_input(f"Valor en °{escala_entrada}", value=25.0, step=0.1, key='val_in')
        
        with col_minmax:
            st.markdown("<p style='visibility: hidden;'>Ocultar controles de slider</p>", unsafe_allow_html=True)
            st.markdown("_Use el selector de arriba, no se requiere el rango._")

        st.form_submit_button("Calcular Conversiones")
    
    st.divider()

    st.subheader("Resultados de Conversión")
    
    conversiones = {}
    escalas = ['C', 'K', 'F', 'R']
    
    for escala_salida in escalas:
        conversiones[escala_salida] = convert_temp_rankine(valor_entrada, escala_entrada, escala_salida)

    col_C, col_K, col_F, col_R = st.columns(4)
    
    resultados = [col_C, col_K, col_F, col_R]
    nombres = ["Celsius", "Kelvin", "Fahrenheit", "Rankine"]
    
    for col, escala, nombre in zip(resultados, escalas, nombres):
        with col:
            st.metric(f"{nombre} (°{escala})", f"{conversiones[escala]:,.2f}")

    st.info(r"""
    **Explicación Física y Fórmulas:**
    * **Celsius a Kelvin:** $T(K) = T(°C) + 273.15$
    * **Celsius a Fahrenheit:** $T(°F) = T(°C) \times 1.8 + 32$
    * **Kelvin a Rankine:** $T(°R) = T(K) \times 1.8$

    Las escalas **Kelvin (K)** y **Rankine (°R)** son **escalas absolutas**, donde $0$ representa el cero absoluto ($0 K \approx -273.15 °C$), el punto donde cesa todo movimiento molecular. 

[Image of Temperature scales comparison showing Celsius, Kelvin, Fahrenheit, and Rankine scales]

    """)

# 3.2. Simulación de Equilibrio Térmico (Sin Cambio de Fase)
def seccion_equilibrio_termico_sin_fase():
    st.header("2️⃣ Equilibrio Térmico (Sistema sin Cambio de Fase)")
    st.markdown("Calcula la **temperatura final de equilibrio** ($T_f$) al mezclar dos cuerpos de diferentes materiales y temperaturas, asumiendo que no hay cambios de fase.")
    
    with st.form("form_equilibrio_simple"):
        
        st.subheader("Parámetros de los Cuerpos")
        materiales = [k for k in CONSTANTES.keys() if k != "Agua"] + ["Agua"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### Cuerpo 1")
            m1 = st.number_input("Masa $m_1$ (kg)", 0.1, None, 1.0, 0.1, key='m1_2')
            T1 = st.number_input("Temperatura Inicial $T_1$ (°C)", -50.0, None, 100.0, 1.0, key='T1_2')
        with col2:
            st.markdown("##### Cuerpo 2")
            m2 = st.number_input("Masa $m_2$ (kg)", 0.1, None, 2.0, 0.1, key='m2_2')
            T2 = st.number_input("Temperatura Inicial $T_2$ (°C)", -50.0, None, 20.0, 1.0, key='T2_2')
        with col3:
            st.markdown("##### Selección de Materiales")
            ce1_key = st.selectbox("Material 1", materiales, index=2, key='mat1_2') # Aluminio
            ce2_key = st.selectbox("Material 2", materiales, index=0, key='mat2_2') # Agua
            st.markdown("<p style='visibility: hidden;'>Relleno</p>", unsafe_allow_html=True) # Espaciador
        
        submit = st.form_submit_button("Calcular Equilibrio")

    if submit:
        # Obtener calores específicos
        ce1 = CONSTANTES[ce1_key].get('ce', CONSTANTES[ce1_key].get('ce_liquido'))
        ce2 = CONSTANTES[ce2_key].get('ce', CONSTANTES[ce2_key].get('ce_liquido'))
        
        # Cálculo de la Temperatura Final de Equilibrio (Tf)
        try:
            Tf = (m1 * ce1 * T1 + m2 * ce2 * T2) / (m1 * ce1 + m2 * ce2)
        except ZeroDivisionError:
            Tf = T1 # Caso trivial o error
        
        # Cálculo de los calores
        Q1 = m1 * ce1 * (Tf - T1) 
        Q2 = m2 * ce2 * (Tf - T2) 
        
        st.divider()

        st.subheader("Resultados y Análisis")

        col_Tf, col_Q_info = st.columns(2)
        with col_Tf:
            st.metric("Temperatura Final de Equilibrio $T_f$ (°C)", f"{Tf:.2f}")

        with col_Q_info:
            st.markdown(f"**Capacidad Calorífica $C_1$ ({ce1_key}):** $C_1 = m_1 c_{{e1}} = {(m1 * ce1):,.0f} \\ J/°C$")
            st.markdown(f"**Capacidad Calorífica $C_2$ ({ce2_key}):** $C_2 = m_2 c_{{e2}} = {(m2 * ce2):,.0f} \\ J/°C$")
            st.markdown(f"**Calor Ganado/Perdido por {ce1_key} ($Q_1$):** ${Q1:,.0f}\ J$")
            st.markdown(f"**Calor Ganado/Perdido por {ce2_key} ($Q_2$):** ${Q2:,.0f}\ J$")

        # Visualización con Plotly
        etiquetas = [f"Cuerpo 1 ({ce1_key})", f"Cuerpo 2 ({ce2_key})"]
        
        fig = go.Figure(data=[
            go.Bar(name='Temperatura Inicial', x=etiquetas, y=[T1, T2], marker_color=['#F08080', '#ADD8E6']),
            go.Scatter(name='Temperatura Final de Equilibrio', x=etiquetas, y=[Tf, Tf], mode='lines', line=dict(color='#000000', dash='dash', width=3))
        ])
        fig.update_layout(
            title='Evolución de Temperatura al Equilibrio',
            yaxis_title='Temperatura $T$ (°C)',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(r"""
        **Principio Físico:** Se basa en la **Conservación de la Energía** en un sistema aislado: el calor total perdido es igual al calor total ganado, $\Sigma Q = 0 \implies Q_1 + Q_2 = 0$.
        
        **Ecuación Fundamental:**
        $$T_f = \frac{m_1 c_{e1} T_1 + m_2 c_{e2} T_2}{m_1 c_{e1} + m_2 c_{e2}}$$
        """)

# 3.3. Procesos Térmicos por Etapas (Q Total)
def seccion_cambio_fase_q_total():
    st.header("3️⃣ Procesos Térmicos por Etapas (Cálculo del Calor Total $Q$)")
    st.markdown("Calcula el **calor total** ($Q$) requerido para llevar una masa de agua de $T_i$ a $T_f$ pasando por las transiciones de fase (fusión a $0^\circ C$ y vaporización a $100^\circ C$).")

    with st.form("form_q_total"):
        col1, col2 = st.columns(2)
        with col1:
            m_agua = st.number_input("Masa de Agua $m$ (kg)", 0.1, None, 1.0, 0.1, key='m_q_total')
        with col2:
            T_inicial = st.number_input("Temperatura Inicial $T_i$ (°C)", -273.15, 150.0, 20.0, 1.0, key='Ti_q_total')
            T_final = st.number_input("Temperatura Final $T_f$ (°C)", -273.15, 150.0, 110.0, 1.0, key='Tf_q_total')
        
        submit = st.form_submit_button("Calcular Calor Total")

    if submit:
        if T_inicial >= T_final:
            st.warning("Para un proceso de calentamiento (endotérmico), la temperatura inicial debe ser menor que la final ($T_i < T_f$).")
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
            Q_acumulado.append(Q_total_new)
            return Q_total_new

        def add_calor_fase(m, L, nombre_fase, T_fase):
            Q = m * L
            Q_total_new = Q_acumulado[-1] + Q
            etapas_q.append((nombre_fase, Q))
            Q_acumulado.append(Q_total_new)
            T_puntos.append(T_fase) # T se mantiene constante durante la fase
            return Q_total_new

        # --- Algoritmo de Calentamiento ---
        
        # Etapa 1: Calentamiento Sólido (Hielo)
        if T_actual < Tf_C:
            T_target = min(T_final, Tf_C)
            add_calor(m_agua, C["ce_solido"], T_target - T_actual, f"1. Calentamiento Sólido ($c_s$): $T_{{i}}={T_actual:.0f}\\to{T_target:.0f} \\ °C$")
            T_actual = T_target
            T_puntos.append(T_actual)
            
        # Etapa 2: Fusión (Cambio de Fase Sólido a Líquido)
        if T_actual == Tf_C and T_final > Tf_C:
            add_calor_fase(m_agua, C["Lf"], f"2. Fusión ($L_f$): $T={Tf_C:.0f} \\ °C$", Tf_C)
            T_puntos.append(Tf_C)
            
        T_actual = T_puntos[-1] 
        
        # Etapa 3: Calentamiento Líquido (Agua)
        if T_actual < Tv_C:
            T_target = min(T_final, Tv_C)
            add_calor(m_agua, C["ce_liquido"], T_target - T_actual, f"3. Calentamiento Líquido ($c_l$): $T_{{i}}={T_actual:.0f}\\to{T_target:.0f} \\ °C$")
            T_actual = T_target
            T_puntos.append(T_actual)

        # Etapa 4: Vaporización (Cambio de Fase Líquido a Gas)
        if T_actual == Tv_C and T_final > Tv_C:
            add_calor_fase(m_agua, C["Lv"], f"4. Vaporización ($L_v$): $T={Tv_C:.0f} \\ °C$", Tv_C)
            T_puntos.append(Tv_C)

        T_actual = T_puntos[-1]

        # Etapa 5: Calentamiento Gas (Vapor)
        if T_actual < T_final:
            add_calor(m_agua, C["ce_gas"], T_final - T_actual, f"5. Calentamiento Vapor ($c_g$): $T_{{i}}={T_actual:.0f}\\to{T_final:.0f} \\ °C$")
            T_puntos.append(T_final) # La temperatura final del cálculo
            
        # 6. Mostrar Resultados
        Q_total = Q_acumulado[-1]
        st.metric("Calor Total Requerido $Q_{total}$ (Joule)", f"{Q_total:,.0f}")

        # Mostrar detalle de etapas
        st.markdown("#### Detalle del Proceso:")
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
        **Explicación Física y Fórmulas:** El calor se añade para dos propósitos:
        1.  **Aumento de Temperatura (Calor Sensible):** $Q = m c_{e} \Delta T$
        2.  **Cambio de Fase (Calor Latente):** $Q = m L$
        
        Las mesetas horizontales en el gráfico son los cambios de fase, donde el calor latente ($L$) se absorbe sin cambiar la temperatura del sistema. 
        """)

# 3.4. Mezcla de Sustancias con Cambio de Fase (COMPLEJA)
def seccion_mezcla_cambio_fase():
    st.header("4️⃣ Equilibrio Térmico con Cambio de Fase ($T_f$ Variable)")
    st.markdown("Simula la mezcla de un **Metal** caliente con **Agua/Hielo** fría. Determina la **temperatura final de equilibrio** y si ocurre **fusión o congelación parcial**.")
    
    with st.form("form_equilibrio_fase"):
        st.subheader("Parámetros de la Mezcla")
        
        materiales_metal = {k: v['ce'] for k, v in CONSTANTES.items() if 'ce' in v and k != "Agua"}
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Cuerpo 1: Agua / Hielo")
            m_agua = st.number_input("Masa de Agua/Hielo $m_1$ (kg)", 0.1, None, 1.0, 0.1, key='m1_fase')
            T_agua = st.number_input("Temperatura Inicial $T_1$ (°C)", -20.0, 50.0, 5.0, 1.0, key='T1_fase')
        
        with col2:
            st.markdown("##### Cuerpo 2: Metal Caliente")
            m_metal = st.number_input("Masa de Metal $m_2$ (kg)", 0.1, None, 0.5, 0.1, key='m2_fase')
            material_metal = st.selectbox("Material del Metal", list(materiales_metal.keys()), index=0, key='mat2_fase')
            T_metal = st.number_input("Temperatura Inicial $T_2$ (°C)", 50.0, 500.0, 150.0, 1.0, key='T2_fase')
            
        submit = st.form_submit_button("Calcular Equilibrio con Fase")

    if submit:
        # Constantes del agua
        C_agua = CONSTANTES['Agua']
        T_ref = C_agua['Tf'] # 0 °C
        L_f = C_agua['Lf']
        c_hielo = C_agua['ce_solido']
        c_agua_liq = C_agua['ce_liquido']
        ce_metal = materiales_metal[material_metal]

        st.divider()

        # --- Lógica de Equilibrio con Cambio de Fase (Hielo/Agua) ---
        
        # 1. Calcular el calor transferido si ambos alcanzan T_ref = 0°C
        
        # Calor cedido por el metal para llegar a 0°C (Q_metal_to_ref > 0)
        Q_metal_to_ref = m_metal * ce_metal * (T_metal - T_ref) 
        
        # Calor absorbido/cedido por el agua para llegar a 0°C (Q_agua_to_ref)
        if T_agua < T_ref: # Hielo: debe absorber calor para llegar a 0°C
            Q_agua_to_ref = m_agua * c_hielo * (T_ref - T_agua) 
        else: # Agua Líquida: debe ceder calor para llegar a 0°C
            Q_agua_to_ref = m_agua * c_agua_liq * (T_ref - T_agua) 
        
        # Calor requerido para fundir toda la masa de agua (si está en hielo)
        Q_fusion_req = m_agua * L_f
        
        # Calor requerido para congelar toda la masa de agua (si está en líquido)
        Q_congelacion_req = m_agua * L_f
        
        # 2. Análisis del resultado (Tres escenarios principales en 0°C)
        
        # Caso A: El sistema termina a T_f > 0°C (No hay hielo o todo el hielo se funde)
        if T_agua >= T_ref:
            # Si el T_metal es muy alto, podría haber ebullición, pero simplificamos a Tf < 100°C.
            # Se calcula Tf asumiendo que el calor específico del agua es el de líquido.
            Tf_simple = (m_agua * c_agua_liq * T_agua + m_metal * ce_metal * T_metal) / (m_agua * c_agua_liq + m_metal * ce_metal)
            
            if Tf_simple > T_ref: # Tf > 0°C
                Tf = Tf_simple
                fraccion_fundida = 1.0 # Todo es líquido
                resultado_msg = f"No hay cambio de fase. La $T_f$ es **mayor a $0°C$** (Agua Líquida + {material_metal})."
            else: # Tf_simple <= 0, significa que puede haber congelación parcial/total.
                # Q_metal_to_ref es el calor que *cede* el metal. 
                # Q_agua_to_ref es el calor que *cede* el agua para llegar a 0°C
                
                Q_total_cedido = Q_metal_to_ref + Q_agua_to_ref # > 0 si Tf < 0
                
                if Q_total_cedido < Q_congelacion_req:
                    Tf = T_ref
                    m_congelada = Q_total_cedido / L_f
                    fraccion_fundida = (m_agua - m_congelada) / m_agua
                    resultado_msg = f"El agua se enfría a **$0°C$** y **se congela parcialmente**. Masa congelada: **${m_congelada:.3f}\ kg$**."
                else:
                    Q_remanente_negativo = -(Q_total_cedido - Q_congelacion_req)
                    Tf = Q_remanente_negativo / (m_agua * c_hielo + m_metal * ce_metal)
                    fraccion_fundida = 0.0 # Todo es hielo
                    resultado_msg = f"Toda el agua se congela. La $T_f$ es **menor a $0°C$** (Hielo + {material_metal})."

        # Caso B: El sistema termina a T_f <= 0°C (Hay hielo)
        else: # T_agua < T_ref (Hay hielo inicialmente)
            # El sistema gana calor: Q_metal_to_ref (cedido por metal) - Q_agua_to_ref (absorbido por hielo a 0°C)
            Q_neto_disponible = Q_metal_to_ref - Q_agua_to_ref # Calor neto que el metal *puede dar* para fundir.
            
            if Q_neto_disponible < 0: 
                # El metal no tiene suficiente calor ni para calentar el hielo a 0°C.
                # Se calcula Tf usando Q_total = (m1*ce1*T1 + m2*ce2*T2) / (m1*ce1 + m2*ce2), con c1=hielo.
                Tf = (m_agua * c_hielo * T_agua + m_metal * ce_metal * T_metal) / (m_agua * c_hielo + m_metal * ce_metal)
                fraccion_fundida = 0.0 # Todo sigue siendo hielo
                resultado_msg = f"El sistema alcanza $T_f = {Tf:.2f}°C$. **No se produce fusión**. El metal no pudo calentar el hielo hasta $0°C$."
            
            elif Q_neto_disponible < Q_fusion_req:
                # El metal calienta el hielo a 0°C, pero no lo funde completamente.
                Tf = T_ref
                m_fundida = Q_neto_disponible / L_f
                fraccion_fundida = m_fundida / m_agua
                resultado_msg = f"El hielo se calienta a **$0°C$**. Solo se funde una masa de **${m_fundida:.3f}\ kg$**."
                
            else:
                # El metal funde todo el hielo y el calor remanente eleva la Tf > 0°C.
                Q_remanente = Q_neto_disponible - Q_fusion_req
                Tf = Q_remanente / (m_agua * c_agua_liq + m_metal * ce_metal)
                fraccion_fundida = 1.0
                resultado_msg = f"Todo el hielo se funde. La temperatura final es **$T_f = {Tf:.2f}°C$** (Agua Líquida + {material_metal})."

        # --- Resultados Finales y Visualización ---

        # Recalcular el calor total transferido (Q cedido por el metal)
        Q_metal_final = m_metal * ce_metal * (T_metal - Tf)
        Q_agua_final = Q_metal_final # Conservación de la energía
        
        st.subheader("Resultados Detallados")
        
        col_Tf, col_fase = st.columns(2)
        with col_Tf:
            st.metric("Temperatura Final de Equilibrio $T_f$ (°C)", f"{Tf:.2f}")
        with col_fase:
            st.metric("Fracción de Agua Fundida", f"{fraccion_fundida * 100:.2f} %")
        
        st.info(f"**Conclusión del Proceso:** {resultado_msg}")
        
        st.markdown(f"**Calor Cedido por {material_metal} ($Q_{{cedido}}$):** ${Q_metal_final:,.0f}\ J$")
        st.markdown(f"**Calor Ganado por Agua/Hielo ($Q_{{ganado}}$):** ${Q_agua_final:,.0f}\ J$")

        # Visualización con Matplotlib (Gráfico de Distribución de Calor)
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Mostrar la distribución de la masa (Hielo vs Agua Líquida)
        if Tf == T_ref and T_agua < T_ref:
            sizes = [m_fundida, m_agua - m_fundida]
            labels = ['Agua Líquida (Fundida)', 'Hielo Sólido (Remanente)']
            colors = ['#ADD8E6', '#90EE90']
            title = f'Distribución de Masa en $T_f = 0°C$'
            
        elif Tf == T_ref and T_agua >= T_ref:
            m_liquida = m_agua - m_congelada
            sizes = [m_liquida, m_congelada]
            labels = ['Agua Líquida (Remanente)', 'Hielo Sólido (Congelada)']
            colors = ['#ADD8E6', '#90EE90']
            title = f'Distribución de Masa en $T_f = 0°C$'

        elif Tf > T_ref:
            sizes = [m_agua, m_metal]
            labels = ['Agua Líquida Total', material_metal]
            colors = ['#ADD8E6', '#F08080']
            title = f'Equilibrio sin Fase en $T_f = {Tf:.2f}°C$'

        else: # Tf < 0
            sizes = [m_agua, m_metal]
            labels = ['Hielo Sólido Total', material_metal]
            colors = ['#90EE90', '#F08080']
            title = f'Equilibrio sin Fase en $T_f = {Tf:.2f}°C$'

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
        ax.axis('equal') 
        ax.set_title(title)
        st.pyplot(fig)
        
        st.info(r"""
        **Análisis Avanzado:** La clave está en el **Punto Triple de Verificación** en $0^\circ C$.
        1.  Primero, se calcula el calor necesario para llevar todas las sustancias a $0^\circ C$.
        2.  Luego, el calor neto disponible se compara con el calor latente total ($Q_{fusión} = m_{agua} L_f$) requerido para el cambio de fase.
        3.  Si $|Q_{neto}| < Q_{fusión}$, la temperatura final debe ser $T_f = 0^\circ C$, y el exceso o defecto de calor se traduce en una **fusión o congelación parcial** de la masa. 
        """)


# 3.5. Simulación de Conducción de Calor 1D (Barra)
def seccion_conduccion_1d():
    st.header("5️⃣ Conducción de Calor en Barra (1D)")
    st.markdown("Simulación del **perfil de temperatura** en una barra, mostrando la solución de **estado estacionario** y la **evolución temporal** (animación).")
    
    with st.form("form_conduccion_1d"):
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

        submit = st.form_submit_button("Ejecutar Simulación 1D")

    if submit:
        st.divider()
        x = np.linspace(0, L, 100)
        
        if sim_tipo == "Estado Estacionario":
            st.subheader("Gráfico del Perfil de Temperatura (Estado Estacionario)")
            
            # T(x) es lineal
            T_x = T_caliente - (T_caliente - T_frio) * (x / L)

            # Cálculo de Flujo de Calor (Asumimos Área A = 1 m²)
            A = 1.0
            Flujo_Potencia = k * A * (T_caliente - T_frio) / L
            
            fig = go.Figure(
                data=[go.Scatter(x=x, y=T_x, mode='lines', line=dict(color='red', width=3))],
                layout=go.Layout(
                    title=f'Perfil de Temperatura $T(x)$ de la Barra de {material}',
                    xaxis_title='Posición $x$ (m)',
                    yaxis_title='Temperatura $T$ (°C)',
                    yaxis_range=[min(T_frio, T_caliente) - 10, max(T_frio, T_caliente) + 10],
                    height=400
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Potencia de Flujo de Calor (Asumiendo $A=1m^2$) $P$ (W)", f"{Flujo_Potencia:,.2f}")
        
        else: # Evolución Temporal (Animación simplificada)
            st.subheader("Animación de la Evolución Temporal")
            
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
                    height=400,
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
            st.plotly_chart(fig, use_container_width=True)

        st.info(fr"""
        **Ecuación de Conducción:** El flujo de calor ($P$) se rige por la **Ley de Fourier**:
        $$P = -k A \frac{{dT}}{{dx}}$$
        
        * **Estado Estacionario:** $\frac{{dT}}{{dx}}$ es constante, por lo que $T(x)$ es **lineal**. 
        * **Evolución Temporal:** La difusión se describe por la **Ecuación de Difusión de Calor** $\frac{{\partial T}}{{\partial t}} = \alpha \frac{{\partial^2 T}}{{\partial x^2}}$ (donde $\alpha = k / \rho c_e$).
        """)


# 3.6. Simulación de Conducción de Calor 2D
def seccion_conduccion_2d():
    st.header("6️⃣ Conducción de Calor en Placa (2D - Estado Estacionario)")
    st.markdown("Visualización de la distribución de temperatura en una placa cuadrada usando el método de relajación (Diferencias Finitas).")
    
    with st.form("form_conduccion_2d"):
        col_res, col_iter = st.columns(2)
        with col_res:
            L_placa = st.slider("Resolución de la Malla (Nodos N)", 10, 80, 40, 5)
        with col_iter:
            T_max_iter = st.slider("Iteraciones (Precisión del Cálculo)", 100, 5000, 1000, 100)

        st.subheader("Condiciones de Borde (°C)")
        col_t, col_l, col_r, col_b = st.columns(4)
        with col_t:
            T_borde_top = st.number_input("Borde Superior $T_{Top}$", 0, 500, 100)
        with col_l:
            T_borde_left = st.number_input("Borde Izquierdo $T_{Left}$", 0, 500, 75)
        with col_r:
            T_borde_right = st.number_input("Borde Derecho $T_{Right}$", 0, 500, 25)
        with col_b:
            T_borde_bottom = st.number_input("Borde Inferior $T_{Bottom}$", 0, 500, 50)
            
        submit = st.form_submit_button("Ejecutar Simulación 2D")

    if submit:
        st.divider()
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
            # Reaplicar los bordes (por si acaso el método de Jacobi puro causó problemas en el borde)
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
                        zmin=0, zmax=300 if T_borde_top < 300 else T_borde_top + 50 # Rango dinámico
                        )
        
        fig.update_layout(
            title='Mapa de Calor 2D (Conducción)',
            xaxis=dict(title='Posición X'), 
            yaxis=dict(title='Posición Y', scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(r"""
        **Ecuación de Laplace:** En estado estacionario (sin cambios temporales) y sin fuentes internas de calor, la distribución de temperatura ($T$) en 2D se rige por la **Ecuación de Laplace**:
        $$\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$$
        
        El método de diferencias finitas aproxima esta ecuación diciendo que la temperatura de un nodo interior es el promedio de sus cuatro vecinos.
        """)

# --- 4. Función Principal de la Aplicación ---

def main():
    st.sidebar.title("Menú de Simulación")
    
    opciones = {
        "1. Conversor Avanzado de Temperaturas": seccion_conversor,
        "2. Equilibrio Térmico (Sin Cambio de Fase)": seccion_equilibrio_termico_sin_fase,
        "3. Procesos Térmicos por Etapas (Q Total)": seccion_cambio_fase_q_total,
        "4. Equilibrio con Cambio de Fase (Avanzado)": seccion_mezcla_cambio_fase,
        "5. Conducción de Calor 1D": seccion_conduccion_1d,
        "6. Conducción de Calor 2D": seccion_conduccion_2d,
    }

    seleccion = st.sidebar.selectbox("Seleccione la Simulación:", list(opciones.keys()))
    
    st.title("🌡️ Simulador Interactivo y Avanzado de Termodinámica")
    
    st.markdown("""
    Este simulador cubre los principales temas de **Temperatura, Calor y Transferencia de Calor**. Utiliza constantes de referencia estándar, como las que se encuentran en textos académicos como *Física Universitaria* de Sears Zemansky.
    """)
    
    # Ejecutar la función correspondiente a la selección
    opciones[seleccion]()

if __name__ == "__main__":
    main()
