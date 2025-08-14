import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd

def plot_trail(latitudes, longitudes, title="Trail Map"):
    """
    Plot a trail on a US map given ordered lists of latitudes and longitudes.

    Args:
        latitudes (list): Ordered list of latitude points
        longitudes (list): Ordered list of longitude points
        title (str): Title for the plot
    """
    # Load US states map data
    usa = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    usa = usa[usa.continent == 'North America']

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot US states
    usa.plot(ax=ax, alpha=0.5, color='lightgray')

    # Plot the trail
    ax.plot(longitudes, latitudes, 'r-', linewidth=2, zorder=2)

    # Plot points
    ax.scatter(longitudes, latitudes, c='blue', s=50, zorder=3)

    # Set plot bounds to continental US
    ax.set_xlim([-125, -65])
    ax.set_ylim([25, 50])

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.show()