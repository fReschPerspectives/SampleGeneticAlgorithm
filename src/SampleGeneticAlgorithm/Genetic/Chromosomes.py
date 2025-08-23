from SampleGeneticAlgorithm.Capitals.Capital import Capital
from SampleGeneticAlgorithm.Capitals.Capital import get_capital_by_city_name
from SampleGeneticAlgorithm.Genetic.Genes import Genes
import random

##TODO: Update all the sanity checks to use length if present or len(self.original_genes) instead of hardcoding 50
class Chromosome:
    """
    A class representing a set of chromosomes, each containing a list of genes.
    """
    def __init__(self,
                 original_genes:Genes = None,
                 starting_gene:Genes = None,
                 length: int = None):
        all_genes = Capital.create_state_capitals() if original_genes is None else original_genes
        self.original_genes = all_genes[:]

        self.starting_gene = starting_gene if starting_gene else random.choice(all_genes)

        if starting_gene is not None:
            other_genes = [gene for gene in all_genes if gene.Name != starting_gene.Name]
            random.shuffle(other_genes)  # Shuffle the other genes to create a random order
            print(f"Initializing chromosome with provided starting gene: {starting_gene}.")
            self.genes = [starting_gene] + other_genes
        else:
            self.genes = all_genes[:]
            random.shuffle(self.genes)  # Shuffle the genes to create a random order

        self.fitness = float('inf')  # Initialize fitness to infinity
        self.length = len(self.original_genes) if length is None else length

        print(f"Chromosome initialized with genes: {[gene.Name for gene in self.genes]}")
        print(f"Chromosome length set to: {self.length}")
        print(f"Observed gene count: {len(self.genes)}")

        assert len(self.genes) == self.length, "Incorrect gene count after init"
        assert len({g.Name for g in self.genes}) == self.length, "Extra genes in init!"


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
        print(f"Performing crossover with rate {crossover_rate}")
        print(f"Self genes: {[gene.Name for gene in self.genes]}")
        print(f"Length of self genes: {len(self.genes)}")
        print(f"Other genes: {[gene.Name for gene in other.genes]}")
        print(f"Length of other genes: {len(other.genes)}")

        if len(self.genes) != len(other.genes):
            print(f"Chromosomes must have the same number of genes for crossover; attempting to repair.")
            self.repair_chromosome()
            other.repair_chromosome()

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

        # Ensure the chromosome is valid (no duplicates)
        new_genes = repair_chromosome(self.original_genes, new_genes, self.starting_gene)

        # Explictly return a chromosome
        return Chromosome(original_genes=new_genes, starting_gene=self.starting_gene, length=self.length)

    import random

    ##TODO: Make calls to this randomize the use_weights parameter
    def insert_mutation(self, mutation_rate: float, use_weights: bool = True):
        # Pair city names and their capital objects together
        # Repair this chromosome first to ensure no duplicates
        n = len(self.genes)

        print(f"Applying insert mutation with rate {mutation_rate} on chromosome of length {n}")
        print(f"Current genes: {[gene.Name for gene in self.genes]}")

        # Fix if somehow got a bad length
        if n != 50:
            self.repair_chromosome()

        capital_objects = self.genes[:]
        capital_pairs = [(capital.Name, capital) for capital in capital_objects]

        assert len(capital_pairs) == 50, "Expected 50 capital pairs initially"
        assert len(set(name for name, _ in capital_pairs)) == 50, "Duplicate cities found initially"

        # If successfully repaired, continue and reset n
        n = len(capital_pairs)

        # Compute how many mutations to perform
        num_mutations = max(int(len(capital_pairs) * mutation_rate), 1)

        if use_weights:
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
        else:
            indices = random.sample(range(len(capital_pairs)), num_mutations)

        # Select the base cities for mutation
        selected_pairs = [capital_pairs[i] for i in indices]

        for i, (base_city, _) in zip(indices, selected_pairs):
            neighbors = Capital.get_capital_distances().get(base_city, {})
            if not neighbors:
                continue

            # Weighted selection based on inverse haversine distances
            current_city_names = [name for name, _ in capital_pairs]
            neighbor_candidates = [
                city for city in neighbors.keys()
                if city in current_city_names
                    and (self.starting_gene is None or city != self.starting_gene.Name)
            ]
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
        new_genes = [capital for _, capital in capital_pairs]

        # Explicit return of a chromosome
        self.genes=new_genes
        self.repair_chromosome()

    ##TODO: Make calls to this randomize the use_weights parameter
    def swap_mutation(self, mutation_rate, use_weights=False):
        n = len(self.genes)

        print(f"Applying swap mutation with rate {mutation_rate} on chromosome of length {n}")
        print(f"Current genes: {[gene.Name for gene in self.genes]}")

        # Repair this chromosome first to ensure no duplicates
        if n != 50:
            self.repair_chromosome()

        capital_objects = self.genes[:]
        capital_pairs = [(capital.Name, capital) for capital in capital_objects]

        # Test this is true or break
        assert len(capital_pairs) == 50, "Expected 50 capital pairs initially"
        assert len(set(name for name, _ in capital_pairs)) == 50, "Duplicate cities found initially"

        # If successfully repaired, continue and reset n
        n = len(capital_pairs)

        # Compute how many mutations to perform
        num_mutations = max(1, int(mutation_rate * n))

        # Determine indices to swap
        if use_weights:
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
        else:
            # Randomly select indices to swap
            indices = random.sample(range(len(capital_pairs)), 2*num_mutations)

        for _ in range(num_mutations):
            i, j = random.sample(indices, k= 2)
            self.genes[i], self.genes[j] = self.genes[j], self.genes[i]

        # Explicit return of a chromosome
        self.repair_chromosome()

    def chunk_mutation(self, mutation_rate, max_chunk_size=3):
        n = len(self.genes)

        print(f"Applying chunk mutation with rate {mutation_rate} on chromosome of length {n}")
        print(f"Current genes: {[gene.Name for gene in self.genes]}")

        # Repair this chromosome first to ensure no duplicates
        if n != 50:
            self.repair_chromosome()

        # Test this is true or break
        assert len(self.genes) == 50, "Expected 50 capital pairs initially"

        # If successfully repaired, continue and reset n
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

        # Explicit return of a chromosome
        self.repair_chromosome()

    def shuffle_mutation(self, mutation_rate):
        n = len(self.genes)
        print(f"Applying shuffle mutation with rate {mutation_rate} on chromosome of length {n}")
        print(f"Current genes: {[gene.Name for gene in self.genes]}")

        # Repair this chromosome first to ensure no duplicates
        if n != 50:
            self.repair_chromosome()

        # Test this is true or break
        assert len(self.genes) == 50, "Expected 50 capital pairs initially"

        # If successfully repaired, continue and reset n
        n = len(self.genes)

        num_indices = min(n, max(2, int(mutation_rate * n)))
        indices = random.sample(range(n), num_indices)
        values = [self.genes[i] for i in indices]
        random.shuffle(values)
        for idx, val in zip(indices, values):
            self.genes[idx] = val

        # Explicit return of a chromosome
        self.repair_chromosome()

    def mutate(self, mutation_rate, max_chunk_size=3):
        # Store original length
        original_length = len(self.genes)

        # Randomly choose a mutation mode
        mode = random.choice(["swap", "chunk", "shuffle", "insert"])

        print(f"Applying {mode} mutation with rate {mutation_rate}")

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

        # Validate length after mutation
        if len(self.genes) != original_length:
            print(f"Length mismatch after {mode} mutation: {len(self.genes)} vs {original_length}")
            self.repair_chromosome()

        # Final validation
        assert len(self.genes) == 50, f"Invalid chromosome length after {mode} mutation: {len(self.genes)}"

    # Ensure the length remains unchanged
    def repair_chromosome(self):
        """
        Remove duplicate genes, ensure starting_gene is first if provided,
        and restore any missing genes to reach original set.
        """
        original_map = {gene.Name: gene for gene in self.original_genes}
        seen = set()
        cleaned = []

        # 1. If a starting gene is specified, add it once at the front
        if self.starting_gene:
            cleaned.append(self.starting_gene)
            seen.add(self.starting_gene.Name)

        # 2. Add other genes, skipping duplicates
        for gene in self.genes:
            if gene.Name not in seen:
                cleaned.append(gene)
                seen.add(gene.Name)

        # 3. Append missing genes to restore full length
        missing = [name for name in original_map if name not in seen]
        for name in missing:
            cleaned.append(original_map[name])

        # Sanity checks
        assert len(cleaned) == len(self.original_genes) == 50, f"Chromosome length mismatch: {len(cleaned)}"
        assert len({g.Name for g in cleaned}) == 50, "Duplicates detected after repair"
        if self.starting_gene:
            assert cleaned[0] == self.starting_gene, "Starting gene not at front"

        print(f"Length after repair: {len(cleaned)}")

        self.genes = cleaned

def repair_chromosome(original_genes, genes, starting_gene=None):
    original_map = {g.Name: g for g in original_genes}
    seen = set()
    cleaned = []

    if starting_gene:
        cleaned.append(starting_gene)
        seen.add(starting_gene.Name)

    for gene in genes:
        if gene.Name not in seen:
            cleaned.append(gene)
            seen.add(gene.Name)

    for name in original_map:
        if name not in seen:
            cleaned.append(original_map[name])

    assert len(cleaned) == len(original_genes) == 50
    assert len({g.Name for g in cleaned}) == 50
    if starting_gene:
        assert cleaned[0] == starting_gene

    print(f"Length after repair: {len(cleaned)}")

    return cleaned
