from SampleGeneticAlgorithm.Capitals.Capital import Capital
from SampleGeneticAlgorithm.Capitals.Capital import get_capital_by_city_name
from SampleGeneticAlgorithm.Genetic.Genes import Genes
import random

class Chromosome:
    """
    A class representing a set of chromosomes, each containing a list of genes.
    """
    def __init__(self,
                 original_genes:Genes = None,
                 starting_gene:Genes = None):
        all_genes = Capital.create_state_capitals()
        self.genes = self.genes = all_genes[:]
        if starting_gene is not None:
            other_genes = [gene for gene in all_genes if gene != starting_gene]
            random.shuffle(other_genes)  # Shuffle the other genes to create a random order
            print(f"Initializing chromosome with provided starting gene: {starting_gene}.")
            self.genes = [starting_gene] + other_genes
        else:
            random.shuffle(self.genes)  # Shuffle the genes to create a random order
        self.starting_gene = starting_gene if starting_gene else random.sample(Capital.create_state_capitals(), 1)
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

    import random

    def insert_mutation(self, mutation_rate: float):
        # Pair city names and their capital objects together
        # Repair this chromosome first to ensure no duplicates
        n = len(self.genes)

        if n != 50:
            capital_objects = repair_chromosome(set(self.genes), self.genes)
        else:
            capital_objects = self.genes[:]

        capital_pairs = [(capital.Name, capital) for capital in capital_objects]

        assert len(capital_pairs) == 50, "Expected 50 capital pairs initially"
        assert len(set(name for name, _ in capital_pairs)) == 50, "Duplicate cities found initially"

        # Compute how many mutations to perform
        num_mutations = max(int(len(capital_pairs) * mutation_rate), 1)

        # Compute the haversine distance between capitals
        # Calculate weights: prefer indices whose neighbors are far
        capital_distances = Capital.get_capital_distances()
        n = len(capital_pairs)
        weights = []
        for i, (name, _) in enumerate(capital_pairs):
            prev = capital_pairs[(i - 1) % n][0]
            nxt = capital_pairs[(i + 1) % n][0]
            distances = capital_distances.get(name, {})
            max_distance = max(distances.values(), default=1)
            dist_prev = distances.get(prev, 0)
            dist_next = distances.get(nxt, 0)
            weight = (dist_prev + dist_next + 1e-6) / (2 * max_distance)
            weights.append(weight)

        # Grab the indices to mutate based on weights
        indices = random.choices(range(len(capital_pairs)), weights=weights, k = num_mutations)

        # Select the base cities for mutation
        selected_pairs = [capital_pairs[i] for i in indices]

        for i, (base_city, _) in zip(indices, selected_pairs):
            neighbors = Capital.get_capital_distances().get(base_city, {})
            if not neighbors:
                continue

            # Weighted selection based on inverse haversine distances
            neighbor_candidates = list(neighbors.keys())
            weights = [1 / (neighbors[n] + 1e-6) for n in neighbor_candidates]

            selected_neighbor = random.choices(neighbor_candidates, weights=weights, k=1)[0]
            selected_capital = get_capital_by_city_name(selected_neighbor)

            # Remove selected_neighbor if it already exists
            existing_indices = [idx for idx, (name, _) in enumerate(capital_pairs) if name == selected_neighbor]
            if existing_indices:
                remove_idx = existing_indices[0]
                del capital_pairs[remove_idx]

                # Adjust insertion index if removal was before it
                if remove_idx < i:
                    i -= 1

            # Insert before or after base_city's index
            insert_idx = i if random.random() < 0.5 else i + 1
            capital_pairs.insert(insert_idx, (selected_neighbor, selected_capital))

        # Final validation
        assert len(capital_pairs) == 50, f"Expected 50 capital pairs after mutation, got {len(capital_pairs)}"
        assert len(set(name for name, _ in capital_pairs)) == 50, "City names must be unique after mutation"

        # Update genes
        self.genes = [capital for _, capital in capital_pairs]

    def swap_mutation(self, mutation_rate):
        n = len(self.genes)
        num_mutations = max(1, int(mutation_rate * n))
        for _ in range(num_mutations):
            i, j = random.sample(range(n), 2)
            self.genes[i], self.genes[j] = self.genes[j], self.genes[i]

    def chunk_mutation(self, mutation_rate, max_chunk_size=3):
        n = len(self.genes)
        num_mutations = max(1, int(mutation_rate * n))
        for _ in range(num_mutations):
            chunk_size = random.randint(1, max_chunk_size)
            if chunk_size * 2 > n:
                continue
            start1 = random.randint(0, n - chunk_size)
            start2 = random.randint(0, n - chunk_size)
            if abs(start1 - start2) < chunk_size:
                continue
            for i in range(chunk_size):
                self.genes[start1 + i], self.genes[start2 + i] = self.genes[start2 + i], self.genes[start1 + i]

    def shuffle_mutation(self, mutation_rate):
        n = len(self.genes)
        num_indices = min(n, max(2, int(mutation_rate * n)))
        indices = random.sample(range(n), num_indices)
        values = [self.genes[i] for i in indices]
        random.shuffle(values)
        for idx, val in zip(indices, values):
            self.genes[idx] = val

    def mutate(self, mutation_rate, max_chunk_size=3):
        # Randomly choose a mutation mode
        mode = random.choice(["swap", "chunk", "shuffle", "insert"])

        if mode == "swap":
            self.swap_mutation(mutation_rate)
        elif mode == "chunk":
            self.chunk_mutation(mutation_rate, max_chunk_size)
        elif mode == "shuffle":
            self.shuffle_mutation(mutation_rate)
        elif mode == "insert":
            self.insert_mutation(mutation_rate)
        else:
            raise ValueError(f"Unsupported mutation mode: {mode}")

    def repair_chromosome(self):
        """
        Repair the chromosome by ensuring that all genes are unique.
        If duplicates are found, they are replaced with random genes
        from the original set of genes not present in the chromosome.

        :return: None
        """
        from collections import Counter

        original_names = [gene.Name for gene in self.original_genes]
        current_names = [gene.Name for gene in self.genes]

        # Count duplicates
        name_counts = Counter(current_names)
        duplicates = [name for name, count in name_counts.items() if count > 1]
        missing = list(set(original_names) - set(current_names))

        # Remove duplicates (but keep one occurrence)
        seen = set()
        cleaned_genes = []
        for gene in self.genes:
            if gene.Name not in seen:
                cleaned_genes.append(gene)
                seen.add(gene.Name)

        # Add missing genes
        for name in missing:
            cleaned_genes.append(Capital.get_capital_by_name(name))

        assert len(cleaned_genes) == len(set(original_names)), f"Chromosome length mismatch: {len(cleaned_genes)} vs {len(original_names)}"

        self.genes = cleaned_genes

def repair_chromosome(original_genes, genes):
    """
    Repair the chromosome by ensuring that all genes are unique.
    If duplicates are found, they are replaced with missing genes
    from the original set.

    :param original_genes: Full set of valid Capital objects (expected 50)
    :param genes: Possibly broken chromosome (Capital objects with duplicates or missing ones)
    :return: Repaired list of 50 Capital objects with unique cities
    """
    from collections import Counter

    original_names = [gene.Name for gene in original_genes]
    current_names = [gene.Name for gene in genes]

    # Count duplicates
    name_counts = Counter(current_names)
    duplicates = [name for name, count in name_counts.items() if count > 1]
    missing = list(set(original_names) - set(current_names))

    # Remove duplicates (but keep one occurrence)
    seen = set()
    cleaned_genes = []
    for gene in genes:
        if gene.Name not in seen:
            cleaned_genes.append(gene)
            seen.add(gene.Name)

    # Add missing genes
    for name in missing:
        cleaned_genes.append(Capital.get_capital_by_name(name))

    assert len(cleaned_genes) == len(set(original_names)), f"Chromosome length mismatch: {len(cleaned_genes)} vs {len(original_names)}"

    return cleaned_genes
