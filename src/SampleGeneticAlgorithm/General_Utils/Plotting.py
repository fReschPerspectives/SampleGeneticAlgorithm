import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import os
import platform

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
    Plot a looping trail on a US map given ordered lists of latitudes and longitudes.
    The trail loops back to the first point, first segment is green, last is purple,
    and all others are blue.
    """
    try:
        usa = get_usa_geometry()

        fig, ax = plt.subplots(figsize=(15, 10))
        usa.plot(ax=ax, alpha=0.5, color='lightgray', edgecolor='black')

        # Loop through points, including loop back to first
        total_points = len(latitudes)
        for idx in range(total_points):
            start_lon = longitudes[idx]
            start_lat = latitudes[idx]
            # wrap to first point
            end_lon = longitudes[(idx + 1) % total_points]
            end_lat = latitudes[(idx + 1) % total_points]

            # Color logic
            if idx == 0:
                segment_color = 'green'
            elif idx == total_points - 1:
                segment_color = 'purple'
            else:
                segment_color = 'blue'

            ax.plot([start_lon, end_lon], [start_lat, end_lat],
                    color=segment_color, linewidth=2, zorder=2)

        # Plot all points for clarity
        ax.scatter(longitudes, latitudes, c='black', s=50, zorder=3)

        ax.set_xlim([-125, -65])
        ax.set_ylim([25, 50])
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        save_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

        if platform.system() == "Darwin":
            plt.close(fig)
        else:
            plt.show(block=False)
            plt.pause(2)
            plt.close(fig)
    except Exception as e:
        print(f"Error plotting trail: {e}")