from SampleGeneticAlgorithm.General_Utils.Loss_Functions import calculate_fitness
from SampleGeneticAlgorithm.Genetic.Population import Population
import random
import itertools
import copy

class Breeding(Population):
    """
    A class to handle the breeding process in a genetic algorithm.
    It manages the selection of parents, crossover, and mutation to create new generations.
    """

    def __init__(self, population: Population):
        super().__init__(
            population_size=population.population_size,
            individual_class=population.individual_class,
            genes=population.get_individuals(),
            best_individual=population.best_individual,
            min_travel_distance=population.min_travel_distance,
            mutation_rate=population.mutation_rate
            )

    def breed(self):
        """
        Perform breeding on the current population to create a new generation.
        This involves selecting parents, performing crossover, and applying mutation.
        """
        elite = self.best_individual  # Get the best individual from the current population

        init_population_size = self.population_size  # Get the current population size

        individuals = sorted(self.get_individuals(), key = lambda obj: obj.fitness) # Get the list of individuals in the population

        # Select the top 50% of individuals as parents
        parents = individuals[:len(individuals) // 2]  # Select the top
        children = []

        parent_pairs = list(itertools.combinations(parents, 2))  # Create pairs of parents for crossover

        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(create_child, p1, p2, self.cross_over_rate, self.mutation_rate)
                for p1, p2 in parent_pairs
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                children.append(future.result())

        # Add the children to the new population
        child_population = Population(population_size=len(children), individual_class=self.individual_class, genes=children)  # Create a new population with the children

        # Assess the fitness of the new population
        child_population.calculate_fitness(calculate_fitness)  # Calculate the fitness of each child in the new population
        print(f"Child population size: {len(child_population.get_individuals())}")  # Print the size of the child population
        print(f"Children population minimum fitness: {child_population.min_travel_distance}")  # Print the minimum travel distance of the child population
        print(f"Total population size: {len(individuals) + len(children) + len(parents)}")  # Print the total population size after breeding

        # Add some completely new individuals to the population
        # This is optional and can be used to introduce new genetic material into the population
        new_individuals = [self.individual_class() for _ in range(init_population_size // 4)]
        new_individuals_population = Population(population_size=len(new_individuals), individual_class=self.individual_class, genes=new_individuals)  # Create a new population with the new individuals
        new_individuals_population.calculate_fitness(calculate_fitness)  # Calculate the fitness of the new individuals

        # Combine the parent population with the child population and mutated parents
        total_pool = [elite] + new_individuals_population.get_individuals() + child_population.get_individuals()  # Combine the current population with the new children
        total_pool = sorted(total_pool, key = lambda obj: obj.fitness) # Sort the combined pool by fitness

        most_fit_individuals = total_pool[:init_population_size]  # Select the most fit individuals to form the new generation

        new_population = Population(population_size=len(most_fit_individuals), individual_class=self.individual_class, genes=most_fit_individuals)  # Create a new population with the most fit individuals
        new_population.min_travel_distance = most_fit_individuals[0].fitness  # Set the minimum travel distance to the fitness of the best individual
        new_population.best_individual = most_fit_individuals[0]  # Set the best individual to the first individual in the sorted list

        self.population = None  # Clear the current population to free up memory
        return new_population  # return the next generation population

def create_child(p1, p2, crossover_rate, mutation_rate):
    child = p1.crossover(p2, crossover_rate)
    child.mutate(mutation_rate)
    child.repair_chromosome()
    return child