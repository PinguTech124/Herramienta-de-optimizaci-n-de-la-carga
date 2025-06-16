import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from modelos import DaySimulation

st.sidebar.title("Menú")
opcion = st.sidebar.radio("Seleccione una opción", [
    "Preparar Datos",
    "Optimizar",
    "Inspeccionar Resultados",
    "Resumen Estadístico"
])

if opcion == "Resumen Estadístico":
    st.title("📊 Resumen Estadístico de la Simulación")

    try:
        with open("data/simulacion.pkl", "rb") as f:
            data = pickle.load(f)
        dias = data["dias"]
        deltaT = data["deltaT"]
        energy_cost = data["energy_cost"]  # Load energy_cost

        # Acumuladores
        modelos = {"A": {"coches": 0, "horas": 0.0, "energia": 0.0},
                   "B": {"coches": 0, "horas": 0.0, "energia": 0.0},
                   "C": {"coches": 0, "horas": 0.0, "energia": 0.0},
                   "D": {"coches": 0, "horas": 0.0, "energia": 0.0}}

        valores_obj = []

        for dia in dias:
            sim = DaySimulation(dia["vehiculos"])
            res = dia["resultado"]
            if res:
                valores_obj.append(res.value_objetivo)
                for i, veh in enumerate(sim.vehicles):
                    nombre = veh.modelo.nombre
                    modelos[nombre]["coches"] += 1
                    modelos[nombre]["horas"] += (veh.slot_dep - veh.slot_arr) * deltaT
                    energia = sum(res.P_values[i][veh.slot_arr:veh.slot_dep]) * deltaT
                    modelos[nombre]["energia"] += energia

        # Crear DataFrame
        df = pd.DataFrame([
            {
                "Modelo": m,
                "Nº Coches": modelos[m]["coches"],
                "Horas Conectadas": round(modelos[m]["horas"], 2),
                "Energía Consumida (kWh)": round(modelos[m]["energia"], 2)
            }
            for m in modelos
        ])
        st.subheader("Tabla de Modelos y Consumo")
        st.dataframe(df)

        # Gráfico histograma
        st.subheader("Distribución de la Función Objetivo")
        fig, ax = plt.subplots()
        ax.hist(valores_obj, bins=10)
        ax.set_xlabel("Valor función objetivo")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Histograma de la función objetivo")
        st.pyplot(fig)

        # Tabla de potencia máxima diaria y promedio
        st.subheader("Potencia máxima diaria y promedio")
        max_powers = [max(d["resultado"].total_power) for d in dias if d["resultado"]]
        avg_max_power = sum(max_powers) / len(max_powers) if max_powers else 0
        max_power_table = pd.DataFrame({
            "Día": list(range(len(max_powers))),
            "Potencia Máxima (kW)": max_powers
        })
        max_power_table.loc[len(max_power_table)] = ["Promedio", avg_max_power]
        st.dataframe(max_power_table)

        # Tabla de kWh totales y número de slots con consumo
        st.subheader("Consumo total de energía y slots con consumo")
        energy_consumption = []
        for i, d in enumerate(dias):
            if d["resultado"]:
                total_energy = sum(d["resultado"].total_power) * deltaT
                # Count slots with non-zero power
                slots_with_consumption = sum(1 for p in d["resultado"].total_power if abs(p) > 1)
                energy_consumption.append({
                    "Día": i,
                    "Energía Total (kWh)": round(total_energy, 2),
                    "Slots con Consumo": slots_with_consumption
                })

        # Calculate averages
        avg_energy = round(sum(row["Energía Total (kWh)"] for row in energy_consumption) / len(energy_consumption), 2) if energy_consumption else 0
        avg_slots = round(sum(row["Slots con Consumo"] for row in energy_consumption) / len(energy_consumption), 2) if energy_consumption else 0

        # Add averages row
        energy_table = pd.DataFrame(energy_consumption)
        energy_table.loc[len(energy_table)] = {"Día": "Promedio", "Energía Total (kWh)": avg_energy, "Slots con Consumo": avg_slots}
        st.dataframe(energy_table)

        # Tabla de coste total de carga
        st.subheader("Coste total de carga por día")
        total_costs = []
        for i, dia in enumerate(dias):
            if dia["resultado"]:
                # Calculate total cost based on energy consumption and prices
                total_cost = sum(dia["resultado"].total_power[t] * deltaT * energy_cost[t] for t in range(len(dia["resultado"].total_power)))
                total_costs.append({"Día": i, "Coste Total (€)": round(total_cost, 2)})

        # Calcular el coste promedio
        avg_cost = round(sum(row["Coste Total (€)"] for row in total_costs) / len(total_costs), 2) if total_costs else 0

        # Crear DataFrame para la tabla de costes
        cost_table = pd.DataFrame(total_costs)
        cost_table.loc[len(cost_table)] = {"Día": "Promedio", "Coste Total (€)": avg_cost}
        st.dataframe(cost_table)

    except Exception as e:
        st.error(f"Error cargando datos: {e}")

elif opcion == "Preparar Datos":
    st.title("🚗 Preparación de Datos")
    st.subheader("Simulación de vehículos para múltiples días")

    # Parámetros ajustables
    N_dias = st.number_input("Número de días a simular", min_value=1, max_value=50, value=4)
    deltaT = st.selectbox("Duración de cada slot (horas)", [0.25, 0.5, 1.0], index=1)
    Pmax_parking_lot = st.number_input("Potencia máxima del parking (kW)", min_value=10.0, max_value=6000.0, value=2000.0)

    # Número de coches por modelo
    st.subheader("Número de coches por modelo")
    N_coches_por_modelo = {
        "A": st.number_input("Modelo A", min_value=0, max_value=100, value=18),
        "B": st.number_input("Modelo B", min_value=0, max_value=100, value=6),
        "C": st.number_input("Modelo C", min_value=0, max_value=100, value=7),
        "D": st.number_input("Modelo D", min_value=0, max_value=100, value=4)
    }

    # Coeficientes de ponderación para la función objetivo
    st.subheader("Coeficientes de ponderación para la función objetivo")
    variance_scaling_factor = st.number_input(
        "Coeficiente para los picos de potencia", 
        min_value=0.0, max_value=1.0, value=0.01, step=0.001
    )
    energy_cost_scaling_factor = st.number_input(
        "Coeficiente para el coste energético", 
        min_value=0.0, max_value=1000000.0, value=1.0, step=1000.0
    )

    # Modelos, distribuciones y rangos de energía predefinidos
    modelos_ev = {
        "A": {"Emax": 356, "Pmax": 150},
        "B": {"Emax": 710, "Pmax": 150},
        "C": {"Emax": 433, "Pmax": 150},
        "D": {"Emax": 600, "Pmax": 150}
    }

    distribuciones = {
        "A": {"entrada": (11.0, 0.5), "salida": (18.0, 0.5)},
        "B": {"entrada": (12.0, 0.5), "salida": (18.5, 0.5)},
        "C": {"entrada": (11.5, 0.5), "salida": (18.5, 0.5)},
        "D": {"entrada": (10.5, 0.5), "salida": (19.0, 0.5)}
    }

    rangos_energia = {
        "A": (0.2577, 0.3149),
        "B": (0.162, 0.198),
        "C": (0.2, 0.2457),
        "D": (0.0235, 0.0287)
    }

    if st.button("Simular y guardar"):
        from modelos import FleetSimulator
        import pickle

        sim = FleetSimulator(modelos_ev, deltaT=deltaT, distribuciones=distribuciones, rangos_energia=rangos_energia)
        dias = []
        for _ in range(N_dias):
            dia = []
            for modelo, n_coches in N_coches_por_modelo.items():
                dia.extend(sim.simular_dia(n_coches, modelo))
            dias.append({"vehiculos": dia, "resultado": None})

        # Load prices
        precios_df = pd.read_csv("src/precio_energia.txt")
        prices = precios_df["Precio (€/kWh)"].tolist()

        # Shift prices to start at 12:00
        start_hour = 12
        prices = prices[start_hour:] + prices[:start_hour]

        # Process prices for deltaT
        if deltaT == 1.0:
            energy_cost = prices
        elif deltaT == 0.5:
            energy_cost = []
            for precio in prices:
                energy_cost.extend([precio, precio])
        elif deltaT == 0.25:
            energy_cost = []
            for precio in prices:
                energy_cost.extend([precio, precio, precio, precio])
        else:
            st.error("DeltaT no soportado. Debe ser 0.25, 0.5 o 1.0")
            st.stop()

        datos = {
            "deltaT": deltaT,
            "Pmax_parking_lot": Pmax_parking_lot,
            "variance_scaling_factor": variance_scaling_factor,
            "energy_cost_scaling_factor": energy_cost_scaling_factor,
            "dias": dias,
            "energy_cost": energy_cost  # Save energy_cost
        }

        with open("data/simulacion.pkl", "wb") as f:
            pickle.dump(datos, f)

        st.success("✅ Simulación guardada correctamente")

elif opcion == "Optimizar":
    st.title("⚙️ Optimización de todos los días simulados")

    import pickle
    from modelos import DaySimulation, optimizar_dia

    try:
        with open("data/simulacion.pkl", "rb") as f:
            datos = pickle.load(f)

        dias = datos["dias"]
        deltaT = datos["deltaT"]
        Pmax_parking_lot = datos["Pmax_parking_lot"]
        variance_scaling_factor = datos["variance_scaling_factor"]
        energy_cost_scaling_factor = datos["energy_cost_scaling_factor"]

        # Cargar precios de energía desde el archivo
        precios_df = pd.read_csv("src/precio_energia.txt")
        prices = precios_df["Precio (€/kWh)"].tolist()

        # Shift prices to start at 12:00
        start_hour = 12
        prices = prices[start_hour:] + prices[:start_hour]

        # Procesar precios para que coincidan con deltaT
        if deltaT == 1.0:
            energy_cost = prices
        elif deltaT == 0.5:
            # Duplicar cada precio para slots de 0.5 horas
            energy_cost = []
            for precio in prices:
                energy_cost.extend([precio, precio])
        elif deltaT == 0.25:
            # Cuadruplicar cada precio para slots de 0.25 horas
            energy_cost = []
            for precio in prices:
                energy_cost.extend([precio, precio, precio, precio])
        else:
            st.error("DeltaT no soportado. Debe ser 0.25, 0.5 o 1.0")
            st.stop()
            
        # Error handling: Check the length of energy_cost
        expected_length = int(24 / deltaT)
        if len(energy_cost) != expected_length:
            st.error(f"Error: La longitud de energy_cost ({len(energy_cost)}) no coincide con la longitud esperada ({expected_length}) para deltaT = {deltaT}.")
            st.stop()

        resultados = []

        for i, dia in enumerate(dias):
            sim = DaySimulation(dia["vehiculos"])
            if dia["resultado"] is None:
                res = optimizar_dia(
                    sim, deltaT, Pmax_parking_lot, energy_cost, 
                    variance_scaling_factor=variance_scaling_factor, 
                    energy_cost_scaling_factor=energy_cost_scaling_factor
                )
                if res:
                    dia["resultado"] = res
            resultados.append({
                "día": i,
                "valor función objetivo": round(dia["resultado"].value_objetivo, 2) if dia["resultado"] else "error"
            })

        # Guardar resultados
        with open("data/simulacion.pkl", "wb") as f:
            pickle.dump(datos, f)

        st.success("✅ Optimización completada")
        st.dataframe(resultados)

    except Exception as e:
        st.error(f"Error durante la optimización: {e}")

elif opcion == "Inspeccionar Resultados":
    st.title("🔍 Inspección de Resultados por Día")

    import pickle
    import matplotlib.pyplot as plt
    from modelos import DaySimulation

    try:
        with open("data/simulacion.pkl", "rb") as f:
            datos = pickle.load(f)

        dias = datos["dias"]
        deltaT = datos["deltaT"]

        dia_idx = st.number_input("Seleccionar día", min_value=0, max_value=len(dias)-1, value=0)

        dia = dias[dia_idx]
        res = dia["resultado"]
        sim = DaySimulation(dia["vehiculos"])

        if res is None:
            st.warning("⚠️ Este día aún no ha sido optimizado.")
        else:
            st.subheader("Energía acumulada por coche")
            fig1, ax1 = plt.subplots()
            for i, v in enumerate(sim.vehicles):
                E = [v.E_arr]
                for t in range(v.slot_arr, v.slot_dep):
                    E.append(E[-1] + res.P_values[i][t] * deltaT)
                ax1.plot(range(v.slot_arr, v.slot_dep + 1), E, label=f"Coche {i+1}")
            ax1.set_xlabel("Slot", labelpad=10)
            ax1.set_ylabel("Energía (kWh)", labelpad=10)
            ax1.set_title("Energía acumulada", pad=20)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig1)

            st.subheader("Potencia total por slot")
            fig2, ax2 = plt.subplots()
            ax2.plot(res.total_power)
            ax2.set_xlabel("Slot", labelpad=10)
            ax2.set_ylabel("Potencia (kW)", labelpad=10)
            ax2.set_title("Potencia total", pad=20)
            st.pyplot(fig2)

            st.subheader("Potencia por coche")
            fig3, ax3 = plt.subplots()
            for i, v in enumerate(sim.vehicles):
                P = res.P_values[i][v.slot_arr:v.slot_dep]
                ax3.plot(range(v.slot_arr, v.slot_dep), P, label=f"Coche {i+1}")
            ax3.set_xlabel("Slot", labelpad=10)
            ax3.set_ylabel("Potencia (kW)", labelpad=10)
            ax3.set_title("Potencia por vehículo", pad=20)
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig3)

            st.subheader("Tabla resumen de Energía")
            resumen = []
            for i, (slot_arr, E_arr, slot_dep, E_dep) in enumerate(sim.get_params_cars()):
                E_calc = E_arr + sum(res.P_values[i][slot_arr:slot_dep]) * deltaT
                resumen.append({
                    "Coche": i + 1,
                    "slot_arr": slot_arr,
                    "slot_dep": slot_dep,
                    "E_arr": round(E_arr, 2),
                    "E_dep": round(E_dep, 2),
                    "E_calc": round(E_calc, 2)
                })
            st.dataframe(resumen)

    except Exception as e:
        st.error(f"Error inspeccionando resultados: {e}")