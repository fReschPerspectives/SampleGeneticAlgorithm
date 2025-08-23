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
            population_size = population.population_size,
            individual_class = population.individual_class,
            desired_chromosome_length = population.desired_chromosome_length,
            genes = population.genes,
            starting_gene = population.starting_gene,
            individuals = population.individuals,
            iterations = population.iterations,
            generation = population.generation,
            mutation_rate = population.mutation_rate,
            cross_over_rate = population.cross_over_rate,
            min_travel_distance = population.min_travel_distance,
            best_individual = population.best_individual,
            )

    def __deepcopy__(self, memo):
        # Only copy relevant attributes
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['population', 'parent']:  # skip recursive or unnecessary refs
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def breed(self):
        """
        Perform breeding on the current population to create a new generation.
        This involves selecting parents, performing crossover, and applying mutation.
        """
        # Constants
        if self.generations_without_improvement >= self.max_allowed_generations_without_improvement:
            print(f"Max generations without improvement reached ({self.generations_without_improvement}). Resetting counter.")
            self.generations_for_rate_reset = 0 # Reset the counter for generations without improvement
            self.generations_without_improvement = self.generations_without_improvement / 2 # Reset the counter for generations without improvement even though it may not result in improvement since we don't want to get in a loop where we keep rates resetting to initial values
            # Reset mutation and crossover rates to their maximum values
            self.mutation_rate = self.max_mutation_rate
            self.cross_over_rate = self.max_cross_over_rate
        else:
            gamma = 0.95  # decay factor between 0 and 1

            # New rates after decay
            self.mutation_rate = self.initial_mutation_rate * (gamma ** (self.generations_for_rate_reset + 1e-6)) # Decay mutation rate over generations while avoiding zero, resets if no improvement for a set number of generations
            self.cross_over_rate = self.initial_cross_over_rate * (gamma ** (self.generations_for_rate_reset + 1e-6)) # Decay crossover rate over generations while avoiding zero, resets if no improvement for a set number of generations

        # Debugging output
        print(f"Generation: {self.generation + 1}, Mutation Rate: {self.mutation_rate}, Crossover Rate: {self.cross_over_rate}")

        elite = self.best_individual  # Get the best individual from the current population

        # Create mutants of the elite individual
        elite_mutants = []
        for i in range(2):  # Mutate the elite individual a few times to create diversity
            mutant = copy.deepcopy(elite)
            mutant.mutate(self.mutation_rate)
            elite_mutants.append(mutant)
            del mutant

        elite_mutants_population = Population(population_size=len(elite_mutants), individual_class=self.individual_class, desired_chromosome_length=self.desired_chromosome_length, individuals=elite_mutants, mutation_rate=self.mutation_rate, cross_over_rate=self.cross_over_rate, generation=self.generation, iterations=self.iterations)  # Create a new population with the elite mutants
        elite_mutants_population.calculate_fitness(calculate_fitness)  # Calculate the fitness of the elite mutants

        # Create the children population
        init_population_size = self.population_size  # Get the current population size
        individuals = self.get_individuals()
        individuals = sorted(individuals, key = lambda obj: obj.fitness) # Get the list of individuals in the population

        # Select the top 50% of individuals as parents
        parents = individuals[:len(individuals) // 2]  # Select the top
        children = []

        # Create the pairs
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
        child_population = Population(population_size=len(children), individual_class=self.individual_class, desired_chromosome_length=self.desired_chromosome_length, individuals=children, mutation_rate=self.mutation_rate, cross_over_rate=self.cross_over_rate, generation=self.generation, iterations=self.iterations)  # Create a new population with the children

        # Assess the fitness of the new population
        child_population.calculate_fitness(calculate_fitness)  # Calculate the fitness of each child in the new population

        print(f"Elite population size: {len(elite_mutants_population.get_individuals())}")  # Print the size of the elite mutants population
        print(f"Elite population minimum fitness: {elite_mutants_population.min_travel_distance}")  # Print the minimum travel distance of the elite mutants population
        print(f"Child population size: {len(child_population.get_individuals())}")  # Print the size of the child population
        print(f"Children population minimum fitness: {child_population.min_travel_distance}")  # Print the minimum travel distance of the child population
        print(f"Total population size: {len(individuals) + len(children) + len(parents)}")  # Print the total population size after breeding

        # Add some completely new individuals to the population
        # This is optional and can be used to introduce new genetic material into the population
        #new_individuals = [self.individual_class(original_genes=self.genes, starting_gene=self.starting_gene) for _ in range(init_population_size // 4)]
        new_individuals_population = Population(population_size=(init_population_size // 4), individual_class=self.individual_class, desired_chromosome_length=self.desired_chromosome_length, genes=self.genes, starting_gene=self.starting_gene, mutation_rate=self.mutation_rate, cross_over_rate=self.cross_over_rate, generation=self.generation, iterations=self.iterations)  # Create a new population with the new individuals
        new_individuals_population.calculate_fitness(calculate_fitness)  # Calculate the fitness of the new individuals

        print(f"New individuals population size: {len(new_individuals_population.get_individuals())}")  # Print the size of the new individuals population
        print(f"New individuals population minimum fitness: {new_individuals_population.min_travel_distance}")  # Print the minimum travel distance of the new individuals population

        # Combine the parent population with the child population and mutated parents
        total_pool = [elite] + elite_mutants_population.get_individuals() + new_individuals_population.get_individuals() + child_population.get_individuals()  # Combine the current population with the new children
        total_pool = sorted(total_pool, key = lambda obj: obj.fitness) # Sort the combined pool by fitness

        most_fit_individuals = total_pool[:init_population_size]  # Select the most fit individuals to form the new generation

        new_population = Population(population_size=len(most_fit_individuals), individual_class=self.individual_class, desired_chromosome_length=self.desired_chromosome_length, genes=self.genes, starting_gene=self.starting_gene, individuals=most_fit_individuals, generation=self.generation, iterations=self.iterations, mutation_rate=self.mutation_rate, cross_over_rate=self.cross_over_rate)  # Create a new population with the most fit individuals
        new_population.min_travel_distance = most_fit_individuals[0].fitness  # Set the minimum travel distance to the fitness of the best individual
        new_population.best_individual = most_fit_individuals[0]  # Set the best individual to the first individual in the sorted list

        self.population = None  # Clear the current population to free up memory
        return new_population  # return the next generation population

def create_child(p1, p2, crossover_rate, mutation_rate):
    child = p1.crossover(p2, crossover_rate)
    child.mutate(mutation_rate)
    child.repair_chromosome()
    return child