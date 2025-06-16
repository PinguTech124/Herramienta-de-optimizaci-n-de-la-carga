from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import cvxpy as cp

@dataclass
class EVModel:
    nombre: str
    Emax: float
    Pmax: float

@dataclass
class Vehicle:
    modelo: EVModel
    slot_arr: int
    slot_dep: int
    E_arr: float
    E_dep: float

@dataclass
class DaySimulation:
    vehicles: List[Vehicle]

    def get_params_cars(self) -> List[Tuple[int, float, int, float]]:
        return [(v.slot_arr, v.E_arr, v.slot_dep, v.E_dep) for v in self.vehicles]

    def get_Pmax_cars(self) -> List[float]:
        return [v.modelo.Pmax for v in self.vehicles]

class FleetSimulator:
    def __init__(self, modelos_ev: Dict[str, Dict[str, float]], deltaT: float, 
                 distribuciones: Dict[str, Dict[str, Tuple[float, float]]],  # Distribuciones por modelo
                 rangos_energia: Dict[str, Tuple[float, float]]):  # Rango de energía por modelo
        self.deltaT = deltaT
        self.modelos = {
            nombre: EVModel(nombre, datos["Emax"], datos["Pmax"]) 
            for nombre, datos in modelos_ev.items()
        }
        self.distribuciones = distribuciones  # Distribuciones de entrada y salida por modelo
        self.rangos_energia = rangos_energia  # Rango de energía por modelo

    def simular_dia(self, n_cars: int, modelo_nombre: str) -> List[Vehicle]:
        import random
        vehiculos = []
        modelo = self.modelos[modelo_nombre]
        
        # Obtener distribución específica para el modelo
        t_arr_mean, t_arr_std = self.distribuciones[modelo_nombre]["entrada"]
        t_dep_mean, t_dep_std = self.distribuciones[modelo_nombre]["salida"]
        
        for _ in range(n_cars):
            t_arr = np.random.normal(t_arr_mean, t_arr_std)
            t_dep = np.random.normal(t_dep_mean, t_dep_std)
            slot_arr = max(0, int(np.floor(t_arr / self.deltaT)))
            slot_dep = int(np.floor(t_dep / self.deltaT))
            if slot_dep <= slot_arr:
                slot_dep = slot_arr + 1
            
            # Obtener rango de energía específico para el modelo
            E_min, E_max = self.rangos_energia[modelo_nombre]
            E_arr = random.uniform(E_min, E_max) * modelo.Emax
            
            vehiculos.append(Vehicle(
                modelo=modelo,
                slot_arr=slot_arr,
                slot_dep=slot_dep,
                E_arr=E_arr,
                E_dep=modelo.Emax
            ))
        return vehiculos

@dataclass
class OptimResult:
    P_values: List[List[float]]
    total_power: List[float]
    value_objetivo: float

def optimizar_dia(dia: DaySimulation, deltaT: float, Pmax_parking_lot: float, energy_cost: List[float], 
                  variance_scaling_factor: float = 0.0001, energy_cost_scaling_factor: float = 1.0) -> OptimResult:
    """
    Optimiza la carga de vehículos eléctricos para un día dado.

    Args:
        dia (DaySimulation): Simulación del día con la información de los vehículos.
        deltaT (float): Duración de cada slot en horas.
        Pmax_parking_lot (float): Potencia máxima del parking en kW.
        energy_cost (list): Lista con el coste de la energía en cada slot.
        variance_scaling_factor (float): Coeficiente de ponderación para el término de varianza.
        energy_cost_scaling_factor (float): Coeficiente de ponderación para el término de coste energético.

    Returns:
        OptimResult: Objeto con los resultados de la optimización.
    """
    # Parámetros
    N = len(dia.vehicles)
    T = int(24 / deltaT)

    # Datos del problema
    params = dia.get_params_cars()
    Pmax_cars = dia.get_Pmax_cars()

    # Variables de decisión
    P = cp.Variable((N, T), nonneg=True)

    # Función objetivo
    total_power = cp.sum(P, axis=0)
    sum_of_squares = cp.sum(cp.square(total_power))  
    energy_cost_component = energy_cost @ total_power  

    objective = cp.Minimize(variance_scaling_factor * sum_of_squares + energy_cost_scaling_factor * energy_cost_component)

    # Restricciones
    constraints = []
    for i in range(N):
        constraints.append(cp.sum(P[i, :]) * deltaT >= params[i][3] - params[i][1])  
        for t in range(T):
            if params[i][0] <= t < params[i][2]:
                constraints.append(P[i, t] <= Pmax_cars[i])  
            else:
                constraints.append(P[i, t] == 0)  

    for t in range(T):
        constraints.append(cp.sum(P[:, t]) <= Pmax_parking_lot)  # parking_power_limit

    # Resolver el problema
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status == cp.OPTIMAL:
        # Extraer los valores de P de la solución
        P_values = P.value.tolist()
        total_power = np.sum(P.value, axis=0).tolist()
        return OptimResult(P_values=P_values, total_power=total_power, value_objetivo=problem.value)
    else:
        print("No se encontró la solución óptima")
        return None


