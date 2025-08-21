from SampleGeneticAlgorithm.Genetic.Chromosomes import Chromosome
from SampleGeneticAlgorithm.Genetic.Breeding import Breeding
from SampleGeneticAlgorithm.Genetic.Population import Population
import os

if __name__ == "__main__":
    os.add_dll_directory("C:\\gdal-3.11.3-proj-9.6.2-arm64\\bin") # annoying bit to get pyogrio to import without complaining
    from SampleGeneticAlgorithm.General_Utils.Plotting import plot_trail

    # Create a population of chromosomes
    initial_population_size = 50

    # Initialize the population with a specified size and individual class
    # Each individual is a Chromosome object initialized with a set of genes
    population = Population(initial_population_size, Chromosome, mutation_rate=0.15)

    # Calculate the fitness of each individual in the population
    from SampleGeneticAlgorithm.General_Utils.Loss_Functions import calculate_fitness
    population.calculate_fitness(calculate_fitness)

    # Print the initial best fitness
    print(f"Initial Best Fitness: {population.best_individual.fitness}")

    # Define the number of generations to iterate through:
    num_generations = 200
    for i in range(num_generations):
        print(f"Generation {i + 1}: Best Fitness = {population.best_individual.fitness}")

        # Set up the breeding process with the current population
        breeding = Breeding(population)

        # Perform breeding to create a new generation
        new_generation = breeding.breed()

        # Update the population with the new generation
        population = new_generation

        latitudes = [gene.Latitude for gene in population.best_individual.get_genes()]
        longitudes = [gene.Longitude for gene in population.best_individual.get_genes()]
        # Plot the trail of the best individual in the population
        plot_trail(latitudes,
                   longitudes,
                   title=f"Generation {i + 1} - Best Fitness: {population.best_individual.fitness}",
                   )

    print(f"Final Best Fitness: {population.best_individual.fitness}")