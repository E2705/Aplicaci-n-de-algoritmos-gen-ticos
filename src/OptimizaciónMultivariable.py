"""
Requisito: Modificar el cromosoma para que soporte dos variables
x e y, dividiéndolo a la mitad y minimizar la función f(x,y) = x^2 + y^2
devolviendo el valor negativo en la aptitud.
"""
import matplotlib.pyplot as plt
from .AlgoritmoGenetico import AlgoritmoGenetico

"""
Se establece CHROM_LEN_EX2 = 10 bits (5 bits para x y 5 bits para y)
Para la función f(x,y) = x^2 + y^2
la cual tiene su valor mínimo en (0, 0), por lo tanto, el
individuo que se acerque más a este valor será el que tenga
la mayor aptitud.
"""
CHROM_LEN_PER_VAR = 5
CHROM_LEN_EX2 = CHROM_LEN_PER_VAR * 2
VAR_MIN_EX2 = -5.0
VAR_MAX_EX2 = 5.0


def decode_multivariable(genotype: list[int]) -> tuple[float, float]:
    """
    Decodifica un genotipo binario en dos fenotipos reales (x e y) para el
    ejercicio, para hacer esto la función divide el genotipo
    en dos partes, donde la primera mitad es para la variable x y la
    segunda mitad es para la variable y, usando el rango predefinido (VAR_MIN_EX2, VAR_MAX_EX2).
    """
    mid = len(genotype) // 2
    gen_x, gen_y = genotype[:mid], genotype[mid:]
    max_int = (2**mid) - 1

    int_x = int("".join(str(bit) for bit in gen_x), 2)
    int_y = int("".join(str(bit) for bit in gen_y), 2)

    x = VAR_MIN_EX2 + (int_x / max_int) * (VAR_MAX_EX2 - VAR_MIN_EX2)
    y = VAR_MIN_EX2 + (int_y / max_int) * (VAR_MAX_EX2 - VAR_MIN_EX2)
    return x, y


def fitness_multivariable(phenotype: tuple[float, float]) -> float:
    x, y = phenotype
    return -float(x**2 + y**2)  # Negativo para minimizar f(x,y)


def ejecutar():
    """
    Esta función ejecuta el algoritmo genético para
    resolver el Ejercicio 2 - Optimizacion Multivariable.
    """
    print("\nEjercicio 2: Optimizacion Multivariable")
    ga = AlgoritmoGenetico(
        #Parametros del algoritmo genético
        population_size=30,
        chromosome_length=CHROM_LEN_EX2,
        pc=0.8,
        pm=0.03,
        fitness_func=fitness_multivariable,
        decode_func=decode_multivariable,
        selection_method="tournament",
        tournament_size=3,
        elitism=True,
        elite_count=1,
    )

    best_genotype, best_fitness = ga.run(num_generations=40)
    best_x, best_y = decode_multivariable(best_genotype)

    print(f"Mejor Genotipo: {''.join(map(str, best_genotype))}")
    print(f"Fenotipo (x, y): ({best_x:.4f}, {best_y:.4f})")
    print(f"Valor Mínimo f(x,y): {-best_fitness:.4f}")

    # Gráfica de la evolución del algoritmo genético
    plt.figure(figsize=(7, 4))
    plt.plot(
        range(1, 41), ga.max_fitness_history, label="Fitness Máximo (-f(x,y))"
    )
    plt.plot(range(1, 41), ga.avg_fitness_history, label="Fitness Promedio")
    plt.title("Ejercicio 2: Minimización f(x,y) = x² + y²")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()