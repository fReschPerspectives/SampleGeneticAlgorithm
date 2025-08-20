from SampleGeneticAlgorithm.General_Utils.Loss_Functions import calculate_fitness
from SampleGeneticAlgorithm.Genetic.Population import Population
import itertools

class Breeding:
    """
    A class to handle the breeding process in a genetic algorithm.
    It manages the selection of parents, crossover, and mutation to create new generations.
    """

    def __init__(self, population: Population):
        self.population = population

    def breed(self):
        """
        Perform breeding on the current population to create a new generation.
        This involves selecting parents, performing crossover, and applying mutation.
        """

        init_population_size = self.population.population_size  # Get the current population size
        self.population = Population(self.population.population_size, self.population.individual_class, sorted(self.population.get_individuals(), key = lambda obj: obj.fitness) ) # Sort individuals by fitness (minimized fitness is desirable, strange as that sounds)
        individuals = self.population.get_individuals()  # Get the list of individuals in the population

        # Ensure the new parent population size is half of the current population size
        parents = individuals[:init_population_size // 2]  # Create pairs of parents
        children = []

        # Iterate through all combinations of parents to create children
        for p in list(itertools.combinations(parents,2)):
            parent1 = p[0]  # Select the first parent from the pair
            parent2 = p[1]  # Select the second parent from the pair

            # Perform crossover to create a child
            child = parent1.crossover(parent2) # Perform a crossover between the two parents to create a child

            # Mutate the child
            child.mutate(mutation_rate=0.1)

            # Ensure the child has a unique chromosome by repairing it
            child.repair_chromosome()

            # Add the child to the new population
            children.append(child)

        # Add the children to the new population
        child_population = Population(population_size=len(children), individual_class=self.population.individual_class, genes=children)  # Create a new population with the children

        # Assess the fitness of the new population
        child_population.calculate_fitness(calculate_fitness)  # Calculate the fitness of each child in the new population

        # Now return the new population as the next generation with the best performers from the current generation and best of the children
        total_pool = individuals + child_population.get_individuals()  # Combine the current population with the new children
        total_pool = sorted(total_pool, key = lambda obj: obj.fitness) # Sort the combined pool by fitness

        most_fit_individuals = total_pool[:init_population_size]  # Select the most fit individuals to form the new generation

        new_population = Population(population_size=len(most_fit_individuals), individual_class=self.population.individual_class, genes=most_fit_individuals)  # Create a new population with the most fit individuals
        new_population.min_travel_distance = most_fit_individuals[0].fitness  # Set the minimum travel distance to the fitness of the best individual
        new_population.best_individual = most_fit_individuals[0]  # Set the best individual to the first individual in the sorted list

        return new_population  # return the next generation population
