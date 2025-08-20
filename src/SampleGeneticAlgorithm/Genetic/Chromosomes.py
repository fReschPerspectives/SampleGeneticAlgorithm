from SampleGeneticAlgorithm.Capitals.Capital import Capital
from SampleGeneticAlgorithm.Genetic.Genes import Genes
import random

class Chromosome:
    """
    A class representing a set of chromosomes, each containing a list of genes.
    """
    def __init__(self, original_genes:
        Genes = None):
        self.genes = Capital.create_state_capitals()
        random.shuffle(self.genes)  # Shuffle the genes to create a random order
        self.original_genes = original_genes if original_genes else Capital.create_state_capitals()
        self.fitness = float('inf')  # Initialize fitness to infinity

    def __str__(self):
        return f"Chromosomes(genes={self.genes})"


    def __repr__(self):
        return self.__str__()

    def get_genes(self):
        """
        Get the genes of the chromosome.

        :return: List of genes in the chromosome.
        """
        return self.genes

    def crossover(self, other):
        """
        Perform crossover between this chromosome and another chromosome.
        The crossover point is randomly selected, and genes are exchanged
        between the two chromosomes at that point.

        :param other: Another Chromosomes object to perform crossover with.
        :return: A new Chromosomes object with the crossed-over genes.
        """

        if len(self.genes) != len(other.genes):
            raise ValueError("Chromosomes must have the same number of genes for crossover.")

        crossover_point = random.randint(0, len(self.genes) - 1)

        new_genes = self.genes[:crossover_point] + other.genes[crossover_point:]

        return Chromosome(new_genes)


    def mutate(self, mutation_rate):
        """
        Mutate the genes based on the mutation rate; i.e. for each gene,
        pop the element with a probability equal to the mutation rate.
        Then reorder the popped elements randomly. and place back into the
        genes list.

        :param mutation_rate: Probability of mutation for each gene.
        """

        # Store popped elements to reorder them later
        popped_elements = []
        popped_indices = []

        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                popped_elements.append(self.genes[i])
                popped_indices.append(i)

        # Shuffle the popped elements
        random.shuffle(popped_elements)

        # Replace the original genes with the shuffled popped elements
        for i in range(len(popped_indices)):
            self.genes[popped_indices[i]] = popped_elements[i]


    def repair_chromosome(self):
        """
        Repair the chromosome by ensuring that all genes are unique.
        If duplicates are found, they are replaced with random genes
        from the original set of genes not present in the chromosome.

        :return: None
        """
        original_genes = self.original_genes
        seen = set()
        duplicate_indices = []
        for i in range(len(self.genes)):
            if self.genes[i] in seen:
                seen.add(self.genes[i])
                duplicate_indices.append(i)
            else:
                seen.add(self.genes[i])

        missing_genes = [gene for gene in original_genes if gene not in seen]
        for i in range(len(self.genes)):
            if self.genes[i] in seen:
                # Replace with a random gene and remove it from missing_genes
                replacement = random.choice(missing_genes)
                self.genes[i] = replacement
                missing_genes.remove(replacement)
            else:
                # If the gene is unique, just move on
                pass