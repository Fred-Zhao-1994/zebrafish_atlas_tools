# Zebrafish Atlas Tools

A Python toolkit for anatomical analysis and visualization of neuron data registered to the MapZebrain standard atlas. This package provides a `BrainAtlas` class to streamline common tasks such as localizing neurons, quantifying regional distribution, and creating 2D/3D visualizations.

## Key Features

*   **Memory Efficient:** Implements on-demand ("lazy") loading for a large number of brain region masks, only holding required masks in memory.
*   **Robust Mask Generation:** Programmatically generates a master brain mask from the union of provided anatomical regions, with options for excluding specific regions (e.g., retina) and tuning parameters.
*   **Data Caching:** Automatically caches computationally expensive results to disk, including the master brain mask and 3D surface meshes, for near-instant loading in subsequent runs.
*   **Spatial Queries:**
    *   `get_neurons_in_region()`: Efficiently find which neurons from a set of coordinates fall within a given anatomical mask.
    *   `get_neurons_near_point()`: Find all neurons within a spherical radius of a specific 3D point.
*   **Quantitative Analysis:**
    *   `get_region_distribution()`: Generate a pandas DataFrame summarizing the count, percentage, and density of a neuron population across multiple brain regions.
*   **Rich Visualization:**
    *   `plot_orthogonal_views()`: Generate publication-quality 2D projections (horizontal, coronal, sagittal) with support for plotting multiple neuron clusters, shading anatomical regions, and highlighting search spheres.
    *   `plot_3d_plotly()`: Create fully interactive 3D renderings of the brain surface and neuron clusters directly within a Jupyter Notebook, powered by the stable `Plotly` backend.

## Prerequisites

This toolkit requires that you have the **MapZebrain atlas data** downloaded locally on your machine. The code expects the following structure in your data folder (e.g., `C:\Scientific_Data\atlas_and_masks\`):

atlas_and_masks/
├── T_AVG_H2BGCaMP.tif <-- The reference brain volume
└── mapZebrain__regions__v2.0.1/ <-- The folder containing all .tif region masks
├── tectum.tif
├── thalamus.tif
└── ...


## Installation

This package is designed to be installed directly from its GitHub repository into your project's conda environment.

1.  **Activate your conda environment:**
    ```bash
    conda activate my_analysis_env
    ```

2.  **Install using `pip` and `git`:**
    ```bash 
    pip install git+https://github.com/Fred-Zhao-1994/zebrafish_atlas_tools.git
    ```

## Quick Start & Usage

It is recommended to create a `config.py` file in your analysis project to manage the path to your large atlas dataset.

**1. Create `config.py` in your analysis project:**
   *(Make sure to add `config.py` to your `.gitignore` file!)*
   ```python
   # In config.py
   ATLAS_DATA_PATH = r"C:\Scientific_Data\atlas_and_masks"
---------------------------
In your Jupyter Notebook or analysis script:


# Import the class and your configuration
from atlas_utils import BrainAtlas
from config import ATLAS_DATA_PATH
import numpy as np

# --- Initialization ---
# The first time you run this, it will generate and cache the master brain mask.
# Subsequent runs will be much faster.
atlas = BrainAtlas(atlas_root_path=ATLAS_DATA_PATH)


# --- Example Usage ---

# Create some fake neuron coordinates
neuron_coords = np.random.normal(loc=[300, 400, 150], scale=50, size=(500, 3))

# 1. Find which neurons are in the Thalamus
is_in_thalamus = atlas.get_neurons_in_region(neuron_coords, 'dorsal_thalamus_proper')
thalamus_neurons = neuron_coords[is_in_thalamus]
print(f"Found {len(thalamus_neurons)} neurons in the Thalamus.")


# 2. Get a quantitative summary of the distribution
distribution = atlas.get_region_distribution(
    neuron_coords, 
    region_list=['tectum', 'dorsal_thalamus_proper', 'cerebellum']
)
print("\nDistribution:")
print(distribution)


# 3. Create a 2D plot showing the thalamic neurons with region shading
plot_data_2d = {
    'Thalamus Neurons': (thalamus_neurons, 'cyan')
}
atlas.plot_orthogonal_views(
    plot_data_2d,
    show_regions=['dorsal_thalamus_proper'],
    size=15
)


# 4. Create an interactive 3D plot
# This will be slow the first time as it generates and caches the 3D meshes.
# Subsequent runs will be fast.
plot_data_3d = {
    'My Neurons': (neuron_coords, 'magenta')
}
atlas.plot_3d_plotly(
    plot_data_3d,
    show_regions=['dorsal_thalamus_proper'],
    region_opacity=0.3
)

