from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class Population:
    """
    This class represents a population of individuals in a genetic algorithm.
    It initializes a population with a specified size and individual class.
    Each individual in the population is an instance of the provided individual class.
    It provides methods to get and set the individuals in the population.
    """

    def __init__(self,
                 population_size,
                 individual_class,
                 genes=None,
                 starting_gene = None,
                 individuals=None,
                 iterations=500,
                 generation=0,
                 mutation_rate=0.025,
                 cross_over_rate=0.25,
                 min_travel_distance=None,
                 best_individual=None, ):
        """
        Initialize the population with a specified size and individual class.
        Each individual in the population is an instance of the provided individual class.
        If genes are provided, each individual is initialized with those genes.
        If no genes are provided, individuals are initialized with default genes.
        The minimum travel distance is set to infinity by default, and the best individual is set to None.
        The mutation rate is set to 0.025 by default, but can be specified.

        :param population_size: The size of the population.
        :param individual_class: The class to use for creating individuals in the population.
        :param genes: A list of genes to initialize each individual with, if applicable.
        :param starting_gene: A gene to start the population with, if applicable.
        :param individuals: A list of individuals to initialize the population with, if applicable.
        :param iterations: Total number of iterations for the genetic algorithm.
        :param generation: Current generation number.
        :param mutation_rate: Probability of mutation for each gene in an individual.
        :param cross_over_rate: Probability of crossover between parents during breeding.
        :param min_travel_distance:
        :param best_individual:
        """
        self.population_size = population_size
        self.individual_class = individual_class
        self.genes = genes
        self.starting_gene = starting_gene
        if individuals:
            print("Initializing population with provided individuals.")
            if len(individuals) != population_size:
                raise ValueError("Number of individuals must match the population size.")
            self.individuals = individuals
        else:
            if genes and starting_gene:
                print("Initializing population with provided genes and starting gene.")
                self.individuals = [individual_class(original_genes=genes, starting_gene=starting_gene)
                                   for _ in range(population_size)]
            elif genes:
                print("Initializing population with provided genes.")
                self.individuals = [individual_class(original_genes=genes)
                                   for _ in range(population_size)]
            elif starting_gene:
                print("Initializing population with starting gene.")
                self.individuals = [individual_class(starting_gene=self.starting_gene)
                                   for _ in range(population_size)]
            else:
                # If no genes or starting gene is provided, initialize with default genes
                print("Initializing population with default genes.")
                self.individuals = [individual_class() for _ in range(population_size)]

        self.generation = generation if generation is not None else 0
        self.iterations = iterations if iterations is not None else 500
        self.mutation_rate = mutation_rate if mutation_rate is not None else 0.025
        self.cross_over_rate = cross_over_rate if cross_over_rate is not None else 0.25
        self.min_travel_distance = float('inf') if min_travel_distance is None else min_travel_distance
        self.best_individual = best_individual if best_individual else None

    def get_individuals(self):
        return self.individuals

    def set_individuals(self, individuals):
        if len(individuals) == self.population_size:
            self.individuals = individuals
        else:
            raise ValueError("Number of individuals must match the population size.")

    def calculate_fitness(self, fitness_function):
        """
        Calculate the fitness of each individual in the population using the provided fitness function.
        The fitness function should take an individual as input and return its fitness value.

        :param fitness_function: A function that calculates the fitness of an individual.
        """
        for individual in self.get_individuals():
            individual.fitness = fitness_function(individual)
            if individual.fitness < self.min_travel_distance:
                self.min_travel_distance = individual.fitness
                self.best_individual = individual

    def calculate_fitness_parallel(self, fitness_function=calculate_fitness):
        """
        Parallel version of fitness calculation. Uses all available CPU cores.
        
        :param fitness_function: A function that calculates the fitness of an individual.
        """
        # Use all available cores
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # Submit all individuals for parallel fitness evaluation
            fitness_results = list(executor.map(fitness_function, self.individuals))

        # Assign fitness and find best individual
        for individual, fitness in zip(self.individuals, fitness_results):
            individual.fitness = fitness
            if fitness < self.min_travel_distance:
                self.min_travel_distance = fitness
            self.best_individual = individual