from SampleGeneticAlgorithm.Genetic.Chromosomes import Chromosome
from SampleGeneticAlgorithm.Genetic.Genes import Genes
from SampleGeneticAlgorithm.Genetic.Population import Population
import os

if __name__ == "__main__":
    os.add_dll_directory("C:\\gdal-3.11.3-proj-9.6.2-arm64\\bin") # annoying bit to get pyogrio to import without complaining

    # Create a population of chromosomes
    initial_population_size = 20

    # Initialize the population with a specified size and individual class
    # Each individual is a Chromosome object initialized with a set of genes
    population = Population(initial_population_size, Chromosome)

    # Create unique genes for each chromosome
    for individual in population.get_individuals():
        genes = Genes()
        genes.create_genes()
        individual.genes = genes.get_genes()
        print(individual)  # Print the second chromosome in the population

    # # Create a Chromosome object with the generated genes
    # chromosome = Chromosomes.Chromosome(first.get_genes())
    # print(chromosome)
    #
    # other_chromosome = Chromosomes.Chromosome(second.get_genes())
    # print(other_chromosome)
    #
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