from Genetic import Genes, Chromosomes

if __name__ == "__main__":
    # Create a Genes object and generate random genes
    first = Genes.Genes()
    second = Genes.Genes()

    first.create_genes()
    second.create_genes()

    # Create a Chromosome object with the generated genes
    chromosome = Chromosomes.Chromosome(first.get_genes())
    print(chromosome)

    other_chromosome = Chromosomes.Chromosome(second.get_genes())
    print(other_chromosome)

    # Perform crossover with another chromosome
    new_chromosome = chromosome.crossover(other_chromosome)

    # Print the new chromosome after crossover
    print(new_chromosome)

    # Mutate the new chromosome with a mutation rate of 0.1
    new_chromosome.mutate(0.4)

    # Repair the chromosome by ensuring it contains unique genes
    new_chromosome.repair_chromosome()

    # Print the mutated chromosome
    print(new_chromosome)