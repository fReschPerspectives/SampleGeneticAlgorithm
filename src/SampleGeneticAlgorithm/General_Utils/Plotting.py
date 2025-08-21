import geopandas as gpd
import matplotlib.pyplot as plt
import os

# Module-level cache
_cached_usa = None

def get_usa_geometry():
    """Load and cache the USA geometry once."""
    global _cached_usa
    if _cached_usa is None:
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
        _cached_usa = world[world['NAME'] == 'United States of America']
    return _cached_usa


def plot_trail(latitudes, longitudes, title="Trail Map"):
    """
    Plot a trail on a US map given ordered lists of latitudes and longitudes.

    Args:
        latitudes (list): Ordered list of latitude points
        longitudes (list): Ordered list of longitude points
        title (str): Title for the plot
    """
    usa = get_usa_geometry()

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

    # Create 'plots' directory if it doesn't exist
    save_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Save the figure
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    plt.show(block=False)
    plt.pause(2)
    plt.close(fig)
