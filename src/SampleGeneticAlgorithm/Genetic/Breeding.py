from SampleGeneticAlgorithm.General_Utils.Loss_Functions import calculate_fitness
from SampleGeneticAlgorithm.Genetic.Population import Population
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
        elite_mutated = copy.deepcopy(elite)  # Create a copy of the elite individual to mutate

        init_population_size = self.population_size  # Get the current population size

        # create elite mutants
        elite_mutants = []
        for i in range(init_population_size):
            elite_mutant = copy.deepcopy(elite_mutated)
            elite_mutant.mutate(mutation_rate=self.mutation_rate)  # Mutate the
            elite_mutants.append(elite_mutant)

        elite_mutants_population = Population(population_size=len(elite_mutants),
                                              individual_class=self.individual_class,
                                              genes=elite_mutants)
        elite_mutants_population.calculate_fitness(calculate_fitness)  # Calculate the fitness of the elite mutants

        individuals = copy.deepcopy(sorted(self.get_individuals(), key = lambda obj: obj.fitness)) # Get the list of individuals in the population

        # Ensure the new parent population size is half of the current population size
        parents = copy.deepcopy(individuals[:init_population_size // 2])  # Create pairs of parents
        children = []

        # Iterate through all combinations of parents to create children
        for p in list(itertools.combinations(parents,2)):
            parent1 = p[0]  # Select the first parent from the pair
            parent2 = p[1]  # Select the second parent from the pair

            # Perform crossover to create a child
            child = parent1.crossover(parent2) # Perform a crossover between the two parents to create a child

            # Mutate the child
            child.mutate(mutation_rate=self.mutation_rate)

            # Ensure the child has a unique chromosome by repairing it
            child.repair_chromosome()

            # Add the child to the new population
            children.append(child)

        # Add the children to the new population
        child_population = Population(population_size=len(children), individual_class=self.individual_class, genes=children)  # Create a new population with the children

        # Assess the fitness of the new population
        child_population.calculate_fitness(calculate_fitness)  # Calculate the fitness of each child in the new population

        # Iterate through the parents and mutate them
        for parent in parents:
            parent.mutate(mutation_rate=self.mutation_rate)  # Mutate each parent with the specified mutation rate

        # make a new parent population with the mutated parents
        parent_population = Population(population_size=len(parents)
                                       , individual_class=self.individual_class
                                       , genes=parents)  # Create a new population with the mutated parents
        parent_population.calculate_fitness(calculate_fitness)  # Calculate the fitness of each parent in the new population

        print(f"Parent population minimum fitness: {parent_population.min_travel_distance}")  # Print the minimum travel distance of the parent population
        print(f"Children population minimum fitness: {child_population.min_travel_distance}")  # Print the minimum travel distance of the child population
        print(f"Elite individual fitness: {elite.fitness}")  # Print the fitness of the elite individual
        print(f"Total population size: {len(individuals) + len(children) + len(parents)}")  # Print the total population size after breeding

        # Add some completely new individuals to the population
        # This is optional and can be used to introduce new genetic material into the population
        new_individuals = [self.individual_class() for _ in range(init_population_size // 4)]
        new_individuals_population = Population(population_size=len(new_individuals), individual_class=self.individual_class, genes=new_individuals)  # Create a new population with the new individuals
        new_individuals_population.calculate_fitness(calculate_fitness)  # Calculate the fitness of the new individuals

        # Combine the parent population with the child population and mutated parents
        total_pool = [elite] + elite_mutants_population.get_individuals() + individuals + new_individuals_population.get_individuals() + parent_population.get_individuals() + child_population.get_individuals()  # Combine the current population with the new children
        total_pool = sorted(total_pool, key = lambda obj: obj.fitness) # Sort the combined pool by fitness

        most_fit_individuals = total_pool[:init_population_size]  # Select the most fit individuals to form the new generation

        new_population = Population(population_size=len(most_fit_individuals), individual_class=self.individual_class, genes=most_fit_individuals)  # Create a new population with the most fit individuals
        new_population.min_travel_distance = most_fit_individuals[0].fitness  # Set the minimum travel distance to the fitness of the best individual
        new_population.best_individual = most_fit_individuals[0]  # Set the best individual to the first individual in the sorted list

        return new_population  # return the next generation population
