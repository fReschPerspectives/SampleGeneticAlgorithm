"""
This module contains the loss functions used in the genetic algorithm.
It includes functions to calculate the fitness of a chromosome based on its genes.
For the capitals problem, the fitness is calculated as the minimum distance to travel between all capitals.
"""

from SampleGeneticAlgorithm.Capitals.Capital import Capital
from SampleGeneticAlgorithm.General_Utils.Haversine import haversine_distance

def calculate_fitness(chromosome):
    """
    Calculate the fitness of a chromosome based on the distance between its genes.
    The fitness is defined as the total distance traveled between all capitals in the chromosome.

    :param chromosome: A Chromosome object containing genes representing capitals.
    :return: The total distance traveled between all capitals in the chromosome.
    """
    from SampleGeneticAlgorithm.Capitals.Capital import Capital

    total_distance = 0.0
    genes = chromosome.get_genes()

    for i in range(len(genes) - 1):
        capital1 = genes[i].get_capital_by_name(genes[i].Name)
        capital2 = genes[i + 1].get_capital_by_name(genes[i + 1].Name)

        if capital1 and capital2:
            distance = haversine_distance(
                capital1.Latitude, capital1.Longitude,
                capital2.Latitude, capital2.Longitude
            )
            total_distance += distance

    # Add distance from last to first to complete the loop
    if len(genes) > 1:
        capital1 = genes[-1].get_capital_by_name(genes[-1].Name)
        capital2 = genes[0].get_capital_by_name(genes[0].Name)
        if capital1 and capital2:
            distance = haversine_distance(
                capital1.Latitude, capital1.Longitude,
                capital2.Latitude, capital2.Longitude
            )
            total_distance += distance

    return total_distance

