from SampleGeneticAlgorithm.Genetic.Chromosomes import Chromosome
from SampleGeneticAlgorithm.Genetic.Genes import Genes
from SampleGeneticAlgorithm.Genetic.Population import Population
import os

if __name__ == "__main__":
    os.add_dll_directory("C:\\gdal-3.11.3-proj-9.6.2-arm64\\bin") # annoying bit to get pyogrio to import without complaining
    from SampleGeneticAlgorithm.General_Utils.Plotting import plot_trail

    # Create a population of chromosomes
    initial_population_size = 2

    # Initialize the population with a specified size and individual class
    # Each individual is a Chromosome object initialized with a set of genes
    population = Population(initial_population_size, Chromosome)

    # # Perform crossover with another chromosome
    # new_chromosome = chromosome.crossover(other_chromosome)
    #
    # # Print the new chromosome after crossover
    # print(new_chromosome)
    #
    # # Mutate the new chromosome with a mutation rate of 0.1
    # new_chromosome.mutate(0.4)
    #
    # # Repair the chromosome by ensuring it contains unique genes
    # new_chromosome.repair_chromosome()
    #
    # # Print the mutated chromosome
    # print(new_chromosome)

    # Calculate the fitness of each individual in the population
    from SampleGeneticAlgorithm.General_Utils.Loss_Functions import calculate_fitness
    population.calculate_fitness(calculate_fitness)

    # testing a single choromosome pathing for plotting
    thing = population.best_individual

    thing1 = thing.genes # this is a list of the genes in the first chromosome
    latitudes = [] # list to hold latitudes of capitals
    longitudes = [] #

    for g in thing1:
        latitudes.append(g.Latitude)  # Assuming each gene has a Latitude attribute
        longitudes.append(g.Longitude)  # Assuming each gene has a Longitude attribute

    plot_trail(latitudes, longitudes, title="Random Path of Capitals") # Plot the trail traversed by the first chromosome's genes

