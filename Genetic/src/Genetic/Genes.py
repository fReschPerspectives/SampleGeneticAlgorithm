import Capitals
import random

class Genes:
    def __init__(self, genes=None, seed: int = None):
        if genes is None:
            self.genes = []
        else:
            self.genes = genes

        if seed is None:
            random.seed(42)
        else:
            random.seed(seed)

    def __str__(self):
        return f"Genes: {self.genes}"

    def create_genes(self):
        capitals = Capitals.Capital.create_state_capitals()
        random.shuffle(capitals)
        self.genes = capitals

    def get_genes(self):
        return self.genes

    def clear_genes(self):
        self.genes.clear()