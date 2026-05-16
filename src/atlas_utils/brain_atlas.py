# In file: src/atlas_utils/brain_atlas.py
# Add these imports at the very top of the file
import os
import glob
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from scipy.ndimage import binary_fill_holes, binary_closing
from skimage.measure import marching_cubes # For creating the mesh
from scipy.ndimage import center_of_mass
from scipy.ndimage import zoom
import plotly.graph_objects as go
from tqdm import tqdm

# ... (imports should be at the top of the file) ...

class BrainAtlas:
    # ...

    def __init__(self, atlas_root_path=None, aliases=None):
        """
        Initializes the atlas by locating data paths. Masks are loaded on-demand.
        """
        # --- 1. Path Setup ---
        if atlas_root_path is None:
            self.root_path = r"C:\Scientific_Data\atlas_and_masks"
        else:
            self.root_path = atlas_root_path

        if not os.path.isdir(self.root_path):
            raise FileNotFoundError(f"Atlas root data path not found: {self.root_path}")

        # --- THIS IS THE FIX ---
        # Define the full paths as attributes of the class BEFORE they are used.
        self.reference_atlas_path = os.path.join(self.root_path, 'T_AVG_H2BGCaMP.tif')
        self.region_masks_path = os.path.join(self.root_path, 'mapZebrain__regions__v2.0.1')
        # ----------------------

        # --- 2. Load ONLY the Reference Atlas ---
        print("Loading reference atlas...")
        # Now this line will work correctly
        self.atlas_volume = self._load_tif(self.reference_atlas_path)
        
        # --- 3. Discover Masks, Do NOT Load Them ---
        self.masks = {}  # This will store loaded masks as a cache
        self._mask_paths = {} # This stores the file paths
        
        mask_files = glob.glob(os.path.join(self.region_masks_path, '*.tif'))
        if aliases is None:
            aliases = {}
            
        print(f"Discovering {len(mask_files)} region masks...")
        for file_path in mask_files:
            original_name = os.path.basename(file_path).replace('.tif', '')
            # Skip the main atlas file if it's in the same directory
            if "T_AVG_H2BGCaMP" in original_name:
                continue
            mask_name = aliases.get(original_name, original_name)
            self._mask_paths[mask_name] = file_path
        
        print("Initialization complete. Masks will be loaded on demand.")

        # --- 4. Initialize placeholders for on-demand objects ---
        self.brain_mask = None 
        self.projection_xy = np.mean(self.atlas_volume[140:280, :, :], axis=0)
        self.projection_xz = np.mean(self.atlas_volume, axis=1)
        self.projection_yz = np.mean(self.atlas_volume, axis=2)


    def get_mask(self, region_name):
        """
        Returns the mask for a given region. Loads from disk if not already in cache.
        """
        if region_name not in self._mask_paths:
            raise ValueError(f"Region '{region_name}' not found.")
        
        # Check if the mask is already loaded in our cache
        if region_name not in self.masks:
            print(f"Loading mask for '{region_name}' from disk...")
            mask_data = self._load_tif(self._mask_paths[region_name])
            self.masks[region_name] = mask_data > 0 # Load and binarize
            
        return self.masks[region_name]

    def clear_mask_cache(self):
        """
        Releases the memory used by all currently loaded region masks.
        """
        num_cleared = len(self.masks)
        self.masks.clear()
        # The master brain mask is also a large object, clear it too.
        self.brain_mask = None 
        print(f"Cleared {num_cleared} masks from memory cache.")
    
    def get_master_brain_mask(self, regenerate=False, exclude=None, iterations=5):
        """
        Generates or loads the master brain mask. Now an on-demand method.
        """
        if exclude is None: exclude = []
        master_mask_path = os.path.join(self.root_path, 'master_brain_mask.tif')

        if self.brain_mask is not None and not regenerate:
            return self.brain_mask

        if regenerate or not os.path.exists(master_mask_path):
            print("Generating new master brain mask...")
            
            masks_for_union = []
            for name in self._mask_paths.keys():
                if name not in exclude:
                    masks_for_union.append(self.get_mask(name))
            
            # ... (rest of mask generation logic is the same) ...
            combined_mask = np.stack(masks_for_union).any(axis=0)
            # ... etc ...
            self.brain_mask = binary_closing(...) 
            tifffile.imwrite(master_mask_path, self.brain_mask.astype(np.uint8))
        else:
            print("Loading master brain mask from disk...")
            self.brain_mask = self._load_tif(master_mask_path) > 0
            
        return self.brain_mask

    def _load_tif(self, file_path):
        """Helper function to load a TIF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TIF file not found: {file_path}")
        return tifffile.imread(file_path)

    # --- Method Implementations ---
# --------------------------------------------------------------------------------------------- 
    # Visualization Methods
    def plot_orthogonal_views(self, 
                            cluster_data, 
                            show_regions=None,
                            region_colors=None,
                            search_sphere=None, # The parameter is here
                            size=5, 
                            alpha=0.8,
                            as_subplots=True,
                            base_size=8):
        """
        Plots 3 orthogonal views with clusters, shading, and an optional search sphere.
        """
        z_dim, y_dim, x_dim = self.atlas_volume.shape
        
        # Proportional size calculations (same as before)
        scale_factor = base_size / max(y_dim, x_dim)
        xy_size = (x_dim * scale_factor, y_dim * scale_factor)
        xz_size = (x_dim * scale_factor, z_dim * scale_factor)
        yz_size = (y_dim * scale_factor, z_dim * scale_factor)

        if as_subplots:
            from matplotlib.gridspec import GridSpec
            fig = plt.figure(figsize=(base_size * 2, base_size * 0.7), dpi=150)
            gs = GridSpec(1, 3, width_ratios=[xy_size[0], xz_size[0], yz_size[0]])
            
            ax_xy = fig.add_subplot(gs[0])
            ax_xz = fig.add_subplot(gs[1])
            ax_yz = fig.add_subplot(gs[2])

            # --- THE FIX IS HERE ---
            # We must pass 'search_sphere=search_sphere' to the helper function.
            self._plot_single_orthogonal_view(ax_xy, 'xy', cluster_data, show_regions, region_colors, size, alpha, search_sphere=search_sphere)
            self._plot_single_orthogonal_view(ax_xz, 'xz', cluster_data, show_regions, region_colors, size, alpha, search_sphere=search_sphere)
            self._plot_single_orthogonal_view(ax_yz, 'yz', cluster_data, show_regions, region_colors, size, alpha, search_sphere=search_sphere)
            
            plt.tight_layout()
            plt.show()
        else:
            # --- AND THE FIX IS HERE AS WELL ---
            fig_xy, ax_xy = plt.subplots(figsize=xy_size, dpi=200)
            self._plot_single_orthogonal_view(ax_xy, 'xy', cluster_data, show_regions, region_colors, size, alpha, search_sphere=search_sphere)
            plt.show()

            fig_xz, ax_xz = plt.subplots(figsize=xz_size, dpi=200)
            self._plot_single_orthogonal_view(ax_xz, 'xz', cluster_data, show_regions, region_colors, size, alpha, search_sphere=search_sphere)
            plt.show()

            fig_yz, ax_yz = plt.subplots(figsize=yz_size, dpi=200)
            self._plot_single_orthogonal_view(ax_yz, 'yz', cluster_data, show_regions, region_colors, size, alpha, search_sphere=search_sphere)
            plt.show()


    def _plot_single_orthogonal_view(self, ax, view, cluster_data, show_regions, 
                                    region_colors, size, alpha, search_sphere=None):
        """
        Helper function to draw a single view, now with search sphere plotting.
        """
        z_dim, y_dim, x_dim = self.atlas_volume.shape
        if show_regions is None: show_regions = []
        if region_colors is None: region_colors = {}
        
        # 1. Draw background projection
        if view == 'xy':
            ax.imshow(self.projection_xy, cmap='gray', aspect='equal', zorder=1)
            ax.set_title('Horizontal (Top-Down)')
        elif view == 'xz':
            ax.imshow(np.flipud(self.projection_xz), cmap='gray', aspect='equal', zorder=1)
            ax.set_title('Coronal (Front)')
        elif view == 'yz':
            ax.imshow(np.flipud(self.projection_yz), cmap='gray', aspect='equal', zorder=1)
            ax.set_title('Sagittal (Side)')

        # 2. Draw region shading by loading masks on-demand
        default_colors = plt.cm.get_cmap('Set2').colors
        for i, region_name in enumerate(show_regions):
            try:
                # --- THE KEY CHANGE ---
                # Get the mask using the on-demand loading function
                mask_volume = self.get_mask(region_name)
                # --------------------
                
                color = region_colors.get(region_name, default_colors[i % len(default_colors)])
                rgba_color = mcolors.to_rgba(color, alpha=0.3)
                color_map = mcolors.ListedColormap([[0, 0, 0, 0], rgba_color])

                if view == 'xy':
                    mask_proj = mask_volume.any(axis=0)
                    ax.imshow(mask_proj, cmap=color_map, aspect='equal', zorder=2)
                elif view == 'xz':
                    mask_proj = mask_volume.any(axis=1)
                    ax.imshow(np.flipud(mask_proj), cmap=color_map, aspect='equal', zorder=2)
                elif view == 'yz':
                    mask_proj = mask_volume.any(axis=2)
                    ax.imshow(np.flipud(mask_proj), cmap=color_map, aspect='equal', zorder=2)
            except ValueError as e:
                print(f"Warning: Could not plot region '{region_name}'. Reason: {e}")

        # 3. Draw neuron scatter points
        for label, (coords, color) in cluster_data.items():
            if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0: continue
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            
            if view == 'xy':
                ax.scatter(x, y, color=color, s=size, alpha=alpha, edgecolors='none', label=label, zorder=3)
            elif view == 'xz':
                ax.scatter(x, z_dim - z, color=color, s=size, alpha=alpha, edgecolors='none', zorder=3)
            elif view == 'yz':
                ax.scatter(y, z_dim - z, color=color, s=size, alpha=alpha, edgecolors='none', zorder=3)
# --- Step 4: Draw Search Sphere (if provided) ---
        if search_sphere is not None:
                point, radius = search_sphere
                px, py, pz = point
                
                # Define the center coordinates and the circle patch ONCE
                if view == 'xy':
                    center_coords = (px, py)
                    marker_coords = (px, py)
                elif view == 'xz':
                    center_coords = (px, z_dim - pz)
                    marker_coords = (px, z_dim - pz)
                elif view == 'yz':
                    center_coords = (py, z_dim - pz)
                    marker_coords = (py, z_dim - pz)
                else: # Should not happen
                    center_coords = None
                    marker_coords = None

                if center_coords and marker_coords:
                    # Draw the 'X' marker for the center point
                    ax.plot(marker_coords[0], marker_coords[1], 'X', color='yellow', markersize=12, markeredgewidth=2.5, zorder=5)
                    
                    # Draw the circle representing the radius
                    circle_patch = Circle(center_coords, radius, color='yellow', fill=False, linestyle='--', linewidth=1.5, zorder=4)
                    ax.add_patch(circle_patch)

        # --- Step 5: Final Touches ---
        if view == 'xy' and any(cluster_data): ax.legend()
        
        # Remove axis decorations
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

# -----------------------------------------------------------------------------------------------
    def plot_3d_plotly(self, 
                   cluster_data, 
                   show_regions=None,
                   region_colors=None,
                   mesh_opacity=0.1,
                   region_opacity=0.2,
                   regenerate_meshes=False,
                   downsample_factor=3):
        """
        Generates a stable, interactive 3D plot using only Plotly and NumPy.
        Displays the main brain outline, neuron clusters, and specified anatomical regions.
        """
        plot_data = [] # Initialize list to hold all plotly objects

        # --- Step 1: Main Brain Mesh (using your working .npy method) ---
        verts_path = os.path.join(self.root_path, f'mesh_verts_ds{downsample_factor}.npy')
        faces_path = os.path.join(self.root_path, f'mesh_faces_ds{downsample_factor}.npy')
        
        if regenerate_meshes or not os.path.exists(verts_path) or not os.path.exists(faces_path):
            print(f"Generating new main brain mesh components with ds-factor: {downsample_factor}...")
            master_mask = self.get_master_brain_mask(regenerate=regenerate_meshes)
            if master_mask is None:
                raise RuntimeError("Master brain mask is not available.")
            
            downsampled_mask = zoom(master_mask, 1/downsample_factor, order=0)
            del master_mask
            
            volume = downsampled_mask.astype(np.float32)
            verts, faces, _, _ = marching_cubes(volume, level=0.5)
            del volume, downsampled_mask
            
            verts = verts * downsample_factor
            
            print(f"Saving main mesh components to: {self.root_path}")
            np.save(verts_path, verts)
            np.save(faces_path, faces)
        else:
            print("Loading cached main mesh components from .npy files...")
            verts = np.load(verts_path)
            faces = np.load(faces_path)

        # Add main brain mesh to the plot
        plot_data.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='lightgrey', opacity=mesh_opacity, name='Brain Outline', showscale=False
        ))

        # --- Step 2: NEW - Generate/Load and Add Region Meshes ---
        if show_regions is None: show_regions = []
        if region_colors is None: region_colors = {}
        default_colors = plt.cm.get_cmap('Set2').colors
            
        for i, region_name in enumerate(show_regions):
            try:
                # Define unique cache file paths for each region mesh
                r_verts_path = os.path.join(self.root_path, f'region_{region_name}_verts_ds{downsample_factor}.npy')
                r_faces_path = os.path.join(self.root_path, f'region_{region_name}_faces_ds{downsample_factor}.npy')
                
                if regenerate_meshes or not os.path.exists(r_verts_path) or not os.path.exists(r_faces_path):
                    print(f"Generating mesh components for region: {region_name}...")
                    mask_volume = self.get_mask(region_name)
                    
                    downsampled_mask = zoom(mask_volume, 1/downsample_factor, order=0)
                    del mask_volume
                    
                    r_verts, r_faces, _, _ = marching_cubes(downsampled_mask.astype(np.float32), level=0.5)
                    del downsampled_mask

                    r_verts = r_verts * downsample_factor
                    
                    np.save(r_verts_path, r_verts)
                    np.save(r_faces_path, r_faces)
                else:
                    r_verts = np.load(r_verts_path)
                    r_faces = np.load(r_faces_path)

                color = region_colors.get(region_name, default_colors[i % len(default_colors)])
                
                plot_data.append(go.Mesh3d(
                    x=r_verts[:, 0], y=r_verts[:, 1], z=r_verts[:, 2],
                    i=r_faces[:, 0], j=r_faces[:, 1], k=r_faces[:, 2],
                    color=color, opacity=region_opacity, name=region_name, showscale=False
                ))
            except Exception as e:
                print(f"Warning: Could not process or plot region '{region_name}'. Reason: {e}")
                
        # --- Step 3: Add Neuron Clusters (same as before) ---
        for label, (coords, color) in cluster_data.items():
            if coords.ndim == 2 and coords.shape[1] == 3 and coords.shape[0] > 0:
                plot_data.append(go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                    mode='markers',
                    marker=dict(size=2.5, color=color, opacity=0.8),
                    name=label
                ))

        # --- Step 4: Define Layout and Display Figure (same as before) ---
        layout = go.Layout(
            title='3D Neuron Clusters',
            template='plotly_dark',
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False),
                yaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False),
                zaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(font=dict(color='white'))
        )
        
        fig = go.Figure(data=plot_data, layout=layout)
        fig.show()
#-----------------------------------------------------------------------------------------------
# Analysis Functions
    def get_neurons_in_region(self, neuron_coords, region_name):
        """
        Returns a boolean mask indicating which neurons fall within a specified brain region.
        This method is vectorized for performance.

        Args:
            neuron_coords (np.ndarray): 
                An (N, 3) numpy array of (x, y, z) coordinates for N neurons.
            region_name (str): 
                The name of the brain region mask to check against (use aliased name if available).
            
        Returns:
            np.ndarray: 
                A boolean array of shape (N,) where True indicates the corresponding
                neuron is inside the specified region.
        """
        # --- Input Validation ---
        if neuron_coords.ndim != 2 or neuron_coords.shape[1] != 3:
            raise ValueError("neuron_coords must be an (N, 3) numpy array.")
            
        # --- On-Demand Mask Loading ---
        # Use the get_mask() method to ensure the required mask is loaded into memory.
        try:
            mask_volume = self.get_mask(region_name)
        except ValueError as e:
            # Re-raise the error with a more helpful message
            raise ValueError(f"Could not check for region '{region_name}'. Reason: {e}") from e

        z_dim, y_dim, x_dim = mask_volume.shape

        # --- Vectorized Coordinate Checking ---
        
        # 1. Round coordinates and convert to integer type for indexing.
        # .clip() ensures that coordinates outside the volume don't cause an index error.
        z_coords = np.round(neuron_coords[:, 2]).astype(int).clip(0, z_dim - 1)
        y_coords = np.round(neuron_coords[:, 1]).astype(int).clip(0, y_dim - 1)
        x_coords = np.round(neuron_coords[:, 0]).astype(int).clip(0, x_dim - 1)

        # 2. Check boundary conditions.
        # Create a boolean mask for neurons that are physically within the volume's bounding box.
        in_bounds = (
            (neuron_coords[:, 0] >= 0) & (neuron_coords[:, 0] < x_dim) &
            (neuron_coords[:, 1] >= 0) & (neuron_coords[:, 1] < y_dim) &
            (neuron_coords[:, 2] >= 0) & (neuron_coords[:, 2] < z_dim)
        )
        
        # Initialize the result array to all False.
        is_in_region = np.zeros(len(neuron_coords), dtype=bool)
        
        # 3. For only the neurons that are in bounds, check their value in the mask volume.
        # This is the key vectorized step. It uses advanced numpy indexing to look up
        # the mask value for all in-bounds neurons at once.
        in_bounds_indices = np.where(in_bounds)[0]
        
        # If there are any neurons within the bounding box...
        if len(in_bounds_indices) > 0:
            # Get the coordinates of just those neurons
            z_lookup = z_coords[in_bounds_indices]
            y_lookup = y_coords[in_bounds_indices]
            x_lookup = x_coords[in_bounds_indices]
            
            # Perform the lookup and update our result array
            # mask_volume is boolean, so this returns True/False
            is_in_region[in_bounds_indices] = mask_volume[z_lookup, y_lookup, x_lookup]

        return is_in_region


    def get_region_distribution(self, neuron_coords, region_list):
        """
        Calculates and returns the distribution of a neuron population across regions.
        """
        if not isinstance(region_list, list):
            raise TypeError("region_list must be a list of region names.")
        if neuron_coords.ndim != 2 or neuron_coords.shape[1] != 3:
            raise ValueError("neuron_coords must be an (N, 3) numpy array.")

        total_neurons = len(neuron_coords)
        if total_neurons == 0:
            return pd.DataFrame(columns=['count', 'percentage', 'volume_voxels', 'density_mm3'])

        distribution_data = []
        # Use tqdm for a progress bar
        for region_name in tqdm(region_list, desc="Analyzing regions"):
            try:
                is_in_region = self.get_neurons_in_region(neuron_coords, region_name)
                count = np.sum(is_in_region)
                
                # Now this call will work
                props = self.get_region_properties(region_name)
                volume = props['volume_voxels']
                
                percentage = (count / total_neurons) * 100 if total_neurons > 0 else 0
                density = (count / volume) * 1e6 if volume > 0 else 0
                
                distribution_data.append({
                    'region': region_name,
                    'count': count,
                    'percentage': percentage,
                    'volume_voxels': volume,
                    'density_mm3': density
                })
            except Exception as e: # Catch any error during processing
                print(f"Warning: Could not process region '{region_name}'. Reason: {e}")

        # If no regions were successfully processed, return an empty frame
        if not distribution_data:
            return pd.DataFrame()

        # --- Handle Overlaps and Unassigned Neurons ---
        assignment_matrix = np.array([
            self.get_neurons_in_region(neuron_coords, r['region']) for r in distribution_data
        ]).T
        
        regions_per_neuron = np.sum(assignment_matrix, axis=1)
        unassigned_count = np.sum(regions_per_neuron == 0)
        multi_assigned_count = np.sum(regions_per_neuron > 1)

        # --- Final Summary ---
        summary_df = pd.DataFrame(distribution_data).set_index('region')
        
        unassigned_percentage = (unassigned_count / total_neurons) * 100
        summary_df.loc['_Unassigned'] = [unassigned_count, unassigned_percentage, np.nan, np.nan]
        summary_df.loc['_Total_Unique_Neurons'] = [total_neurons, 100.0, np.nan, np.nan]
        
        print(f"\nAnalysis complete. Found {multi_assigned_count} neurons assigned to >1 region.")

        return summary_df.round(2)
    
    def get_region_properties(self, region_name):
        """
        Calculates properties (volume, center of mass) for a specified region mask.

        Args:
            region_name (str): The name of the region mask.
            
        Returns:
            dict: A dictionary containing the properties.
        """
        # Use the on-demand loading method to get the mask
        mask_volume = self.get_mask(region_name)
        
        # Calculate volume by summing all non-zero pixels
        volume = np.sum(mask_volume)
        
        # Calculate the center of mass
        # Note: center_of_mass returns coordinates in (z, y, x) order
        com_zyx = center_of_mass(mask_volume)
        
        # Return as a dictionary
        return {
            'volume_voxels': volume,
            'center_of_mass_zyx': com_zyx
        }

    def get_neurons_near_point(self, neuron_coords, point, radius):
        """
        Finds all neurons within a given spherical radius of a specific 3D point.
        This method is vectorized for high performance.

        Args:
            neuron_coords (np.ndarray): 
                An (N, 3) numpy array of (x, y, z) coordinates for the full neuron population.
            point (tuple or np.ndarray): 
                The (x, y, z) coordinate of the center of the sphere.
            radius (float or int): 
                The radius of the sphere in the same units as the coordinates.
                
        Returns:
            np.ndarray: 
                A boolean array of shape (N,) where True indicates the corresponding
                neuron is inside the specified sphere.
        """
        # --- Input Validation ---
        if neuron_coords.ndim != 2 or neuron_coords.shape[1] != 3:
            raise ValueError("neuron_coords must be an (N, 3) numpy array.")
        
        point = np.asarray(point)
        if point.shape != (3,):
            raise ValueError("point must be a 3-element tuple or array (x, y, z).")
            
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("radius must be a positive number.")

        # --- Vectorized Distance Calculation ---
        
        # 1. Calculate the squared distance from the point to all neurons at once.
        #    We use squared distance to avoid a costly square root operation.
        #    This subtracts the 'point' from every row in 'neuron_coords',
        #    squares the result element-wise, and sums along the rows.
        squared_distances = np.sum((neuron_coords - point)**2, axis=1)
        
        # 2. Compare with the squared radius.
        squared_radius = radius**2
        
        # 3. The result is a boolean array of all neurons within the radius.
        is_near_point = squared_distances < squared_radius
        
        return is_near_point
