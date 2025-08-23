from SampleGeneticAlgorithm.Genetic.Chromosomes import Chromosome
from SampleGeneticAlgorithm.Genetic.Breeding import Breeding
from SampleGeneticAlgorithm.Genetic.Population import Population
from SampleGeneticAlgorithm.Capitals.Capital import get_capital_by_city_name
import os

if __name__ == "__main__":
    #os.add_dll_directory("C:\\gdal-3.11.3-proj-9.6.2-arm64\\bin") # annoying bit to get pyogrio to import without complaining
    from SampleGeneticAlgorithm.General_Utils.Plotting import plot_trail

    # Create a population of chromosomes
    initial_population_size = 250  # Define the initial population size

    # Initialize the population with a specified size and individual class
    # Each individual is a Chromosome object initialized with a set of genes
    population = Population(initial_population_size, Chromosome, starting_gene=get_capital_by_city_name(name = "Denver"), mutation_rate=0.06, cross_over_rate=0.6)
    print(f"Population initialized with {len(population.individuals)} individuals.")
    print(f"Population initialized with mutation rate: {population.mutation_rate}, crossover rate: {population.cross_over_rate}")

    # Calculate the fitness of each individual in the population
    from SampleGeneticAlgorithm.General_Utils.Loss_Functions import calculate_fitness
    population.calculate_fitness(calculate_fitness)

    # Print the initial best fitness
    print(f"Initial Best Fitness: {population.best_individual.fitness}")

    # Define the number of generations to iterate through:
    num_generations = 500
    for i in range(num_generations):
        print(f"Generation {i + 1}: Best Fitness = {population.best_individual.fitness}")

        previous_best_fitness = population.best_individual.fitness

        # Set up the breeding process with the current population
        breeding = Breeding(population)

        # Perform breeding to create a new generation
        new_generation = breeding.breed()

        # Update the population with the new generation
        population.individuals.clear()
        population = new_generation
        population.generation +=1
        population.iterations = 500

        best_genes = population.best_individual.get_genes()
        latitudes = [gene.Latitude for gene in best_genes]
        longitudes = [gene.Longitude for gene in best_genes]

        # Plot the trail of the best individual in the population if different from the previous generation
        new_best_fitness = population.best_individual.fitness
        if (new_best_fitness != previous_best_fitness) or (i == 0):
            plot_trail(latitudes,
                       longitudes,
                       title=f"Generation {i + 1} - Best Fitness - {population.best_individual.fitness}",
                      )

    print(f"Final Best Fitness: {population.best_individual.fitness}")