import geopandas as gpd
import matplotlib.pyplot as plt
import os

def plot_trail(latitudes, longitudes, title="Trail Map"):
    """
    Plot a trail on a US map given ordered lists of latitudes and longitudes.

    Args:
        latitudes (list): Ordered list of latitude points
        longitudes (list): Ordered list of longitude points
        title (str): Title for the plot
    """

    # Use the Natural Earth dataset directly from the URL if not locally cached
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

    # Read directly from URL via geopandas/fiona
    world = gpd.read_file(url)
    usa = world[world['NAME'] == 'United States of America']

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot US base map
    usa.plot(ax=ax, alpha=0.5, color='lightgray', edgecolor='black')

    # Plot the trail path
    ax.plot(longitudes, latitudes, 'r-', linewidth=2, zorder=2)

    # Plot trail points
    ax.scatter(longitudes, latitudes, c='blue', s=50, zorder=3)

    # Set map bounds (Continental US)
    ax.set_xlim([-125, -65])
    ax.set_ylim([25, 50])

    # Add labels
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.show()
