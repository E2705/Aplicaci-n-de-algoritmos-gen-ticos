"""
Requisito: Ejecutar el algoritmo del Ejercicio 1 tres veces variando
únicamente la probabilidad de mutación en 0.01, 0.1 y 0.5),
generando las tres gráficas de convergencia y comparándolas.
"""
import random
import matplotlib.pyplot as plt
import numpy as np
from .AlgoritmoGenetico import AlgoritmoGenetico
from .MaximizacionCubica import CHROM_LEN_EX1, decode_cubic, fitness_cubic


def ejecutar():
    """
    Esta función ejecuta el algoritmo genético para
    resolver el Ejercicio 3 - Impacto en la Tasa de Mutación.
    """
    print("\nEjercicio 3: Impacto en la Tasa de Mutación")
    pm_values = [0.01, 0.1, 0.5]
    histories = {}

    for pm in pm_values:
        random.seed(42)
        np.random.seed(42)
        """
        Se establecen los parametros del algoritmo genético
        para el ejercicio 3, que en realidad son los parametros
        del ejercicio 1.

        Y se itera sobre la lista pm_values reiniciando la semilla aleatoria
        para asegurar que los tres experimentos inicien con la misma población
        base y la comparación sea justa
        """

        ga = AlgoritmoGenetico(
            population_size=20,
            chromosome_length=CHROM_LEN_EX1,
            pc=0.8,
            pm=pm,
            fitness_func=fitness_cubic,
            decode_func=decode_cubic,
            selection_method="roulette",
            elitism=True,
            elite_count=1,
        )
        ga.run(num_generations=40)
        histories[pm] = (ga.max_fitness_history, ga.avg_fitness_history)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    """
    En este caso se usa plt.subplot(1, 2, 1) y plt.subplot(1, 2, 2) para trazar
    en una misma grafica la evolución del Fitness Máximo y Fitness Promedio a lo
    largo de las 40 generaciones para las tres variables de pm.
    """
    for pm, (max_h, _) in histories.items():
        plt.plot(range(1, 41), max_h, label=f"pm = {pm}")
    plt.title("Fitness Máximo por Tasa de Mutación")
    plt.xlabel("Generación")
    plt.ylabel("Fitness Máximo")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for pm, (_, avg_h) in histories.items():
        plt.plot(range(1, 41), avg_h, label=f"pm = {pm}")
    plt.title("Fitness Promedio por Tasa de Mutación")
    plt.xlabel("Generación")
    plt.ylabel("Fitness Promedio")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()