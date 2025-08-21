from SampleGeneticAlgorithm.Capitals.Capital import Capital
import random

class Genes:
    def __init__(self, genes=None, seed: int = None):
        if genes is None:
            self.genes = []
        else:
            self.genes = genes

    def __str__(self):
        return f"Genes: {self.genes}"

    def create_genes(self, starting_gene: Capital = None):
        all_genes = Capital.create_state_capitals()
        if starting_gene:
            # If a starting gene is provided, ensure it is included in the genes
            if starting_gene not in all_genes:
                raise ValueError("Starting gene must be one of the state capitals.")
                self.genes = [starting_gene] + random.shuffle([gene for gene in all_genes if gene != starting_gene])
            else:
                self.genes = Capital.create_state_capitals()
                random.shuffle(self.genes)  # Shuffle the genes to create a random order

    def get_genes(self):
        return self.genes

    def clear_genes(self):
        self.genes.clear()