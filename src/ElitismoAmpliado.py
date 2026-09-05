"""
Requisito: Modificar la función interna de elitismo del algoritmo para preservar
de manera intacta a los 3 mejores individuos de la generación actual hacia la siguiente.
"""
import matplotlib.pyplot as plt
import numpy as np
from .AlgoritmoGenetico import AlgoritmoGenetico
from .MaximizacionCubica import CHROM_LEN_EX1, decode_cubic, fitness_cubic

def ejecutar():
    """
    Esta función ejecuta el algoritmo genético para
    resolver el Ejercicio 4 - Elitismo de 3 Individuos.
    """
    print("\nEjercicio 4: Elitismo de 3 Individuos")

    # Parámetros del algoritmo genético (con elite_count=3)
    ga = AlgoritmoGenetico(
        population_size=20,
        chromosome_length=CHROM_LEN_EX1,
        pc=0.8,
        pm=0.02,
        fitness_func=fitness_cubic,
        decode_func=decode_cubic,
        selection_method="roulette",
        elitism=True,
        elite_count=3,  # Preservar a los 3 mejores individuos de cada generación
    )

    # Ejecutar el algoritmo genético
    ga.run(num_generations=30)

    # Evaluar la población final para obtener aptitudes y fenotipos
    fitness_values, phenotypes = ga._calculate_all_fitness(ga.population)

    # Ordenar los índices de la población de mayor a menor fitness
    sorted_indices = np.argsort(fitness_values)[::-1]

    print("\n--- VConvergencia de la población final ---")
    print(
        f"Los 3 genotipos de élite preservados son el mismo genotipo debido a la convergencia."
    )
    print(
        f"Genotipo dominante en el Top 3: {''.join(map(str, ga.population[sorted_indices[0]]))}"
    )
    print(
        f"Fitness del individuo dominante: {fitness_values[sorted_indices[0]]:.4f}"
    )

    """
    Se filtra para ver cuales sson los 3 mejores genotipos distintos al dominante
    """
    unique_indices = []
    seen_genotypes = set()

    for idx in sorted_indices:
        gen_tuple = tuple(ga.population[idx])
        if gen_tuple not in seen_genotypes:
            seen_genotypes.add(gen_tuple)
            unique_indices.append(idx)
        if len(unique_indices) == 3:
            break

    print("\n--- 3 Genotipos Unicos distintos al dominante---")
    print(
        f"| {'Rango':<8} | {'Genotipo':<10} | {'Fenotipo (x)':<12} | {'Fitness f(x)':<12} |"
    )
    print(
        "|----------|------------|--------------|--------------|"
    )

    for rank, idx in enumerate(unique_indices, start=1):
        genotype_str = "".join(map(str, ga.population[idx]))
        x_val = phenotypes[idx]
        fit_val = fitness_values[idx]
        print(
            f"| #{rank:<7} | {genotype_str:<10} | {x_val:<12.4f} | {fit_val:<12.4f} |"
        )

    # Gráfica de convergencia del algoritmo
    plt.figure(figsize=(7, 4))
    plt.plot(
        range(1, 31),
        ga.max_fitness_history,
        label="Fitness Máximo (K=3)",
        color="purple",
    )
    plt.plot(
        range(1, 31),
        ga.avg_fitness_history,
        label="Fitness Promedio (K=3)",
        color="green",
    )
    plt.title("Ejercicio 4: Elitismo de 3 Individuos")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()