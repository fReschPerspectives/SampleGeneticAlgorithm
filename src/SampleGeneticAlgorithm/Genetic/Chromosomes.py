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

    def crossover(self, other, crossover_rate=0.05):
        """
        Perform crossover between this chromosome and another chromosome.

        Based on the crossover_rate, select a number of cities (genes) from `self`
        and force their order into `other` at the same indices. This ensures
        partial inheritance of gene order while preserving position.

        :param other: Another Chromosome object to crossover with.
        :param crossover_rate: Fraction of genes to use for crossover.
        :return: A new Chromosome object with crossover-applied genes.
        """

        if len(self.genes) != len(other.genes):
            raise ValueError("Chromosomes must have the same number of genes for crossover.")

        gene_count = len(self.genes)
        num_selected = max(1, int(crossover_rate * gene_count))

        # Randomly select genes from self to impose their order
        selected_genes = random.sample(self.genes, num_selected)

        # Get their order in self
        selected_genes_in_order = [gene for gene in self.genes if gene in selected_genes]

        # Find their indices in other
        selected_indices_in_other = [i for i, gene in enumerate(other.genes) if gene in selected_genes]

        # Sort the selected_genes_in_order based on self
        # Put those into the same positions in a copy of other.genes
        new_genes = other.genes[:]
        for idx, gene in zip(selected_indices_in_other, selected_genes_in_order):
            new_genes[idx] = gene

        return Chromosome(new_genes)


    def mutate(self, mutation_rate, mode="swap", max_chunk_size=3):
        """
        Mutate the genes based on the mutation rate using different strategies.

        :param mutation_rate: Float [0,1] indicating mutation intensity.
        :param mode: Mutation type: "swap", "chunk", or "shuffle".
        :param max_chunk_size: For "chunk" mode, maximum size of chunks to swap.
        """

        n = len(self.genes)
        num_mutations = max(1, int(mutation_rate * n))

        if mode == "swap":
            # Swap individual elements
            for _ in range(num_mutations):
                i, j = random.sample(range(n), 2)
                self.genes[i], self.genes[j] = self.genes[j], self.genes[i]

        elif mode == "chunk":
            # Swap entire chunks of the array
            for _ in range(num_mutations):
                chunk_size = random.randint(1, max_chunk_size)
                if chunk_size * 2 > n:
                    continue  # Not enough space for two chunks

                start1 = random.randint(0, n - chunk_size)
                start2 = random.randint(0, n - chunk_size)

                # Avoid overlapping chunks
                if abs(start1 - start2) < chunk_size:
                    continue

                # Swap the chunks
                for i in range(chunk_size):
                    self.genes[start1 + i], self.genes[start2 + i] = self.genes[start2 + i], self.genes[start1 + i]

        elif mode == "shuffle":
            # Randomly select a subset of indices and shuffle the elements
            num_indices = min(n, max(2, int(mutation_rate * n)))
            indices = random.sample(range(n), num_indices)
            values = [self.genes[i] for i in indices]
            random.shuffle(values)
            for idx, val in zip(indices, values):
                self.genes[idx] = val

        else:
            raise ValueError(f"Unsupported mutation mode: {mode}")


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
        if len(missing_genes) > 0:
            for i in range(len(self.genes)):
                if self.genes[i] in seen:
                    # Replace with a random gene and remove it from missing_genes
                    replacement = random.choice(missing_genes)
                    self.genes[i] = replacement
                    missing_genes.remove(replacement)
                else:
                    # If the gene is unique, just move on
                    pass
        else:
            pass