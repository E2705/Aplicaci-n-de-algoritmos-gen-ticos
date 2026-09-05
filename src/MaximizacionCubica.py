"""
Requisitos: Maximizar f(x) = x^3 - 4x^2 + 5x,
definir un espacio de búsqueda (rango continuo para x)
y ajustar la decodificación binaria a dicho rango.
"""
import matplotlib.pyplot as plt
from .AlgoritmoGenetico import AlgoritmoGenetico

"""
Ajustes del rango continuo [-1, 4]
para esto se definieron las constantes
CHROM_LEN_EX1 = 8
X_MIN_EX1 = -1.0
X_MAX_EX1 = 4.0
"""
CHROM_LEN_EX1 = 8
X_MIN_EX1 = -1.0
X_MAX_EX1 = 4.0


def decode_cubic(genotype: list[int]) -> float:
    """
    Convierte un genotipo binario a un fenotipo real en el rango [-1, 4].
    """
    integer_val = int("".join(str(bit) for bit in genotype), 2)
    max_integer = (2**CHROM_LEN_EX1) - 1
    return X_MIN_EX1 + (integer_val / max_integer) * (X_MAX_EX1 - X_MIN_EX1)


def fitness_cubic(x: float) -> float:
    """
    Calcula el fitness de un fenotipo real en el rango [-1, 4].
    """
    return float(x**3 - 4 * x**2 + 5 * x)


def ejecutar():
    """
    Esta función ejecuta el algoritmo genético para
    resolver el Ejercicio 1 - Maximizacion Cubica.
    """
    print("\nEjercicio 1: Maximizacion Cubica")
    ga = AlgoritmoGenetico(
        #Parametros del algoritmo genético
        population_size=20,
        chromosome_length=CHROM_LEN_EX1,
        pc=0.8,
        pm=0.02,
        fitness_func=fitness_cubic,
        decode_func=decode_cubic,
        selection_method="roulette",
        elitism=True,
        elite_count=1,
    )

    best_genotype, best_fitness = ga.run(num_generations=30)
    best_x = decode_cubic(best_genotype)

    print(f"Mejor Genotipo: {''.join(map(str, best_genotype))}")
    print(f"Mejor Fenotipo (x): {best_x:.4f}")
    print(f"Mejor Fitness f(x): {best_fitness:.4f}")

    # Gráfica de la evolución del algoritmo genético
    plt.figure(figsize=(7, 4))
    plt.plot(
        range(1, 31),
        ga.max_fitness_history,
        label="Fitness Máximo",
        color="b",
    )
    plt.plot(
        range(1, 31),
        ga.avg_fitness_history,
        label="Fitness Promedio",
        color="orange",
    )
    plt.title("Ejercicio 1: Maximización Cúbica")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()