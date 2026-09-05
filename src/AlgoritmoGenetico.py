"""
Este proyecto se realizo en base a la POO, donde cada ejercicio
es una clase y se reutiliza la clase "AlgoritmoGenetico" que
en si es la clase base del proyecto, donde se encuentran las funciones
para la abstracción del bucle evolutivo, Mecanismos de selección,
operadores geneticos de reproducción y el elitismo

El por que se hizo asi es para tener un control sobre cada ejercicio, ya que no todos necesitan las mismas cosas.

La mayoria de la documentación de este archivo y su estructura fue tomada de:
Clase 1 AG https://colab.research.google.com/drive/1A5lqWRZ32LWPUJhDyth2k6MicwBcE4uN?usp=drive_link#scrollTo=eb2db27c
"""
import copy
import random
import numpy as np


class AlgoritmoGenetico:
    """
    Implementación genérica de un Algoritmo Genético Canónico.

    Atributos:
        population_size (int): Número de individuos en la población.
        chromosome_length (int): Longitud del cromosoma (número de bits/genes).
        pc (float): Probabilidad de cruzamiento (crossover).
        pm (float): Probabilidad de mutación.
        elitism (bool): Si se aplica elitismo (preservar al mejor individuo).
        fitness_func (callable): Función que calcula la aptitud de un fenotipo.
        decode_func (callable): Función que decodifica un genotipo a un fenotipo.
        selection_method (str): Método de selección ('roulette' o 'tournament').
        tournament_size (int, opcional): Tamaño del torneo para la selección por torneo.
    """
    
    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        pc: float,
        pm: float,
        fitness_func: callable,
        decode_func: callable,
        selection_method: str = "roulette",
        elitism: bool = True,
        elite_count: int = 1,
        tournament_size: int = 3,
    ):
        if not (0 <= pc <= 1 and 0 <= pm <= 1):
            raise ValueError("pc y pm deben estar entre 0 y 1.")
        if population_size <= 0 or chromosome_length <= 0:
            raise ValueError(
                "population_size y chromosome_length deben ser positivos."
            )
        if selection_method not in ["roulette", "tournament"]:
            raise ValueError(
                "selection_method debe ser 'roulette' o 'tournament'."
            )

        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.pc = pc
        self.pm = pm
        self.elitism = elitism
        self.elite_count = elite_count
        self.fitness_func = fitness_func
        self.decode_func = decode_func
        self.selection_method = selection_method
        self.tournament_size = tournament_size

        self.population: list[list[int]] = []
        self.max_fitness_history: list[float] = []
        self.avg_fitness_history: list[float] = []
        self.best_individual_genotype: list[int] = []
        self.best_individual_fitness: float = -float("inf")

    def _initialize_population(self) -> None:
        """
        Inicializa la población con cromosomas binarios generados aleatoriamente.
        Corresponde a la fase inicial de generación de individuos aleatorios en una población.
        Cada individuo es una lista de bits (0 o 1) de longitud `chromosome_length`.
        """
        self.population = [
            [random.randint(0, 1) for _ in range(self.chromosome_length)]
            for _ in range(self.population_size)
        ]

    def _calculate_all_fitness(self, population: list[list[int]]) -> tuple[list[float], list]:
        """
        Calcula la aptitud (fitness) para cada individuo en la población.
        Para cada cromosoma (genotipo), primero lo decodifica a su representación real (fenotipo),
        y luego evalúa la aptitud de ese fenotipo utilizando `fitness_func`.

        Args:
            population (list[list[int]]): La población actual de genotipos.

        Returns:
            tuple[list[float], list]: Una tupla que contiene la lista de valores de aptitud
            y la lista de fenotipos decodificados.
        """
        fitness_values = []
        phenotypes = []
        for chromosome in population:
            phenotype = self.decode_func(chromosome)
            fitness = self.fitness_func(phenotype)
            fitness_values.append(fitness)
            phenotypes.append(phenotype)
        return fitness_values, phenotypes

    def _select_proportional(self, population: list[list[int]], fitness_values: list[float]) -> list[list[int]]:
        """
        Realiza la selección de individuos utilizando el método de la Ruleta de Holland.
        Individuos con mayor aptitud tienen una mayor probabilidad de ser seleccionados para la siguiente generación.
        Maneja valores de aptitud no negativos mediante un desplazamiento si es necesario para evitar probabilidades negativas.

        Args:
            population (list[list[int]]): La población actual.
            fitness_values (list[float]): Los valores de aptitud correspondientes a cada individuo.

        Returns:
            list[list[int]]: La nueva población (pool de apareamiento) después de la selección, con el mismo tamaño que la población original.

        """
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_values]
        else:
            adjusted_fitness = fitness_values

        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            return random.choices(population, k=self.population_size)

        selection_probabilities = [f / total_fitness for f in adjusted_fitness]
        return random.choices(
            population, weights=selection_probabilities, k=self.population_size
        )

    def _select_tournament(self, population: list[list[int]], fitness_values: list[float]) -> list[list[int]]:
        """
        Realiza la selección de individuos utilizando el método por Torneo.
        En cada paso, se seleccionan aleatoriamente `tournament_size` individuos y el de mayor aptitud entre ellos es elegido.
        Este proceso se repite `population_size` veces para formar la nueva población.

        Args:
            population (list[list[int]]): La población actual.
            fitness_values (list[float]): Los valores de aptitud correspondientes.

        Returns:
            list[list[int]]: La nueva población (pool de apareamiento) después de la selección.
        """
        new_population = []
        for _ in range(self.population_size):
            contestants = random.sample(
                range(self.population_size), self.tournament_size
            )
            best_idx = max(contestants, key=lambda i: fitness_values[i])
            new_population.append(copy.deepcopy(population[best_idx]))
        return new_population

    def _crossover_one_point(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        """
        Realiza el cruzamiento de un punto entre dos padres para producir dos hijos.
        Con una probabilidad `pc`, se elige un punto de corte aleatorio y los segmentos de los padres se intercambian.
        Si no se produce cruzamiento, los hijos son copias de los padres.

        Args:
            parent1 (list[int]): Genotipo del primer padre.
            parent2 (list[int]): Genotipo del segundo padre.

        Returns:
            tuple[list[int], list[int]]: Una tupla con los genotipos de los dos hijos resultantes.
        """
        if random.random() < self.pc:
            point = random.randint(1, self.chromosome_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def _mutate_flip_bit(self, chromosome: list[int]) -> list[int]:
        """
        Realiza la mutación bit a bit en un cromosoma.
        Por cada gen en el cromosoma, con una probabilidad `pm`, su valor se invierte (0 a 1, o 1 a 0).

        Args:
            chromosome (list[int]): El genotipo a mutar.

        Returns:
            list[int]: El genotipo mutado.
        """
        mutated = copy.deepcopy(chromosome)
        for i in range(self.chromosome_length):
            if random.random() < self.pm:
                mutated[i] = 1 - mutated[i]
        return mutated

    def _apply_elitism(self, old_population: list[list[int]], old_fitness: list[float], new_population: list[list[int]]) -> list[list[int]]:
        """
        Aplica el mecanismo de elitismo, preservando al mejor individuo de la generación anterior
        (almacenado como `self.best_individual_genotype`) en la nueva población.
        El individuo con menor aptitud de la nueva población es reemplazado por el élite.

        Args:
            new_population (list[list[int]]): La población recién generada (después de crossover y mutación).

        Returns:
            list[list[int]]: La población con el individuo élite insertado.
        """
        elite_indices = np.argsort(old_fitness)[-self.elite_count :]
        elites = [copy.deepcopy(old_population[i]) for i in elite_indices]

        new_fitness_values, _ = self._calculate_all_fitness(new_population)
        worst_indices = np.argsort(new_fitness_values)[: self.elite_count]

        for idx, elite_individual in zip(worst_indices, elites):
            new_population[idx] = elite_individual

        return new_population

    def run(self, num_generations: int) -> tuple[list[int], float]:
        """
        Ejecuta el Algoritmo Genético para un número dado de generaciones.
        Este es el bucle principal del AG, que coordina la inicialización, evaluación, selección, cruzamiento y mutación
        a lo largo de múltiples generaciones.

        Args:
            num_generations (int): El número de generaciones a ejecutar.

        Returns:
            tuple[list[int], float]: El genotipo del mejor individuo encontrado en toda la ejecución y su aptitud.
        """
        self._initialize_population()

        for generation in range(num_generations):
            fitness_values, _ = self._calculate_all_fitness(self.population)

            current_best_index = np.argmax(fitness_values)
            if fitness_values[current_best_index] > self.best_individual_fitness:
                self.best_individual_fitness = fitness_values[
                    current_best_index
                ]
                self.best_individual_genotype = copy.deepcopy(
                    self.population[current_best_index]
                )

            self.max_fitness_history.append(self.best_individual_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))

            if self.selection_method == "roulette":
                mating_pool = self._select_proportional(
                    self.population, fitness_values
                )
            else:
                mating_pool = self._select_tournament(
                    self.population, fitness_values
                )

            if len(mating_pool) % 2 != 0:
                mating_pool.append(random.choice(mating_pool))
            random.shuffle(mating_pool)

            new_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = mating_pool[i], mating_pool[i + 1]
                child1, child2 = self._crossover_one_point(parent1, parent2)
                child1 = self._mutate_flip_bit(child1)
                child2 = self._mutate_flip_bit(child2)
                new_population.extend([child1, child2])

            new_population = new_population[: self.population_size]

            if self.elitism:
                new_population = self._apply_elitism(
                    self.population, fitness_values, new_population
                )

            self.population = new_population

        return self.best_individual_genotype, self.best_individual_fitness