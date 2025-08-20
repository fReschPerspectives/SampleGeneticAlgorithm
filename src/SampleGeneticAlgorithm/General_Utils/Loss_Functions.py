"""
This module contains the loss functions used in the genetic algorithm.
It includes functions to calculate the fitness of a chromosome based on its genes.
For the capitals problem, the fitness is calculated as the minimum distance to travel between all capitals.
"""

import SampleGeneticAlgorithm.Capitals as Capitals

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth specified by latitude and longitude.

    :param lat1: Latitude of the first point in degrees.
    :param lon1: Longitude of the first point in degrees.
    :param lat2: Latitude of the second point in degrees.
    :param lon2: Longitude of the second point in degrees.
    :return: Distance in kilometers between the two points.
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def calculate_fitness(chromosome):
    """
    Calculate the fitness of a chromosome based on the distance between its genes.
    The fitness is defined as the total distance traveled between all capitals in the chromosome.

    :param chromosome: A Chromosome object containing genes representing capitals.
    :return: The total distance traveled between all capitals in the chromosome.
    """
    total_distance = 0.0
    genes = chromosome.get_genes()

    for i in range(len(genes) - 1):
        capital1 = Capitals.Capital.get_capital_by_name(genes[i])
        capital2 = Capitals.Capital.get_capital_by_name(genes[i + 1])

        if capital1 and capital2:
            distance = haversine_distance(
                capital1.latitude, capital1.longitude,
                capital2.latitude, capital2.longitude
            )
            total_distance += distance

    # Add distance from last to first to complete the loop
    if len(genes) > 1:
        capital1 = Capitals.Capital.get_capital_by_name(genes[-1])
        capital2 = Capitals.Capital.get_capital_by_name(genes[0])
        if capital1 and capital2:
            distance = haversine_distance(
                capital1.latitude, capital1.longitude,
                capital2.latitude, capital2.longitude
            )
            total_distance += distance

    return total_distance