class Population:
    """
    This class represents a population of individuals in a genetic algorithm.
    It initializes a population with a specified size and individual class.
    Each individual in the population is an instance of the provided individual class.
    It provides methods to get and set the individuals in the population.
    """
    def __init__(self, population_size, individual_class, genes = None):
        self.population_size = population_size
        self.individual_class = individual_class
        if genes:
            self.individuals = [individual_class(genes) for _ in range(population_size)]
        else:
            self.individuals = [individual_class() for _ in range(population_size)]
        self.min_travel_distance = float('inf')
        self.best_individual = None

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
        for individual in self.individuals:
            individual.fitness = fitness_function(individual)
            if individual.fitness < self.min_travel_distance:
                self.min_travel_distance = individual.fitness
                self.best_individual = individual