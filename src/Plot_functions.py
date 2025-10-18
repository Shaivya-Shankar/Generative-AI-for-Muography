import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_2d_correlations_by_transformation(
    all_datasets,
    X0_values,
    transformations=None,
    materials=None,
    include_target_pairs=True,
    include_condition_target_pairs=True,
    figsize_per_plot=(4, 3),
    cmap='viridis',
    save_plots=False,
    save_path=None
):
    """
    Generate 2D correlation plots for original data and all available transformations.
    """
    
    print("\n--- Generating 2D Correlation Plots from Loaded Scaled Data ---")
    
    # Get configuration from any dataset
    any_dataset = list(all_datasets.values())[0]
    
    # Set defaults if not provided
    if materials is None:
        materials = list(all_datasets.keys())
    if transformations is None:
        transformations = any_dataset.transformations
    
    first_material = list(materials)[0]
    first_dataset = all_datasets[first_material]
    X0_value = float(X0_values[first_material])
    
    # Get feature names from plotting config
    try:
        plotting_config = first_dataset.plotting_config
        cond_features = plotting_config['condition']['features']
        target_features = plotting_config['target']['features']
        cond_labels = plotting_config['condition']['labels'] 
        target_labels = plotting_config['target']['labels']
        
        print(f"Condition features: {cond_features}")
        print(f"Target features: {target_features}")
        
    except Exception as e:
        print(f"Error getting plotting config: {e}")
        return
    
    # Define plot pairs
    plot_definitions = []
    
    # Target vs Target pairs
    if include_target_pairs and len(target_features) >= 2:
        target_pairs = list(itertools.combinations(range(len(target_features)), 2))
        for var1_idx, var2_idx in target_pairs:
            plot_definitions.append({
                'type': 'target_vs_target',
                'xlabel': target_labels[var1_idx],
                'ylabel': target_labels[var2_idx],
                'x_feature': target_features[var1_idx],
                'y_feature': target_features[var2_idx]
            })
    
    # Condition vs Target pairs
    if include_condition_target_pairs:
        for target_idx in range(len(target_features)):
            plot_definitions.append({
                'type': 'cond_vs_target',
                'xlabel': cond_labels[0],  # P
                'ylabel': target_labels[target_idx],
                'x_feature': cond_features[0],  # P
                'y_feature': target_features[target_idx]
            })
    
    # Create list with original data first, then all transformations
    plot_order = ['Original'] + list(transformations)
    
    # Create separate figure for each data type
    for data_type in plot_order:
        print(f"\nGenerating 2D plots for: {data_type}")
        
        # Setup grid layout
        num_rows = len(materials)
        num_cols = len(plot_definitions)
        
        if num_cols == 0:
            continue
        
        fig, axs = plt.subplots(
            num_rows, num_cols,
            figsize=(figsize_per_plot[0] * num_cols, figsize_per_plot[1] * num_rows),
            constrained_layout=True
        )
        
        # Ensure axs is always 2D array
        if num_rows == 1 and num_cols == 1:
            axs = np.array([[axs]])
        elif num_rows == 1:
            axs = np.array([axs])
        elif num_cols == 1:
            axs = np.array([[ax] for ax in axs])
        
        # Set figure title
        if data_type == 'Original':
            fig.suptitle('2D Correlations - Original (Unscaled) Data', 
                        fontsize=16, weight='bold')
        else:
            fig.suptitle(f'2D Correlations - {data_type} Transformation', 
                        fontsize=16, weight='bold')
        
        h = None
        
        # Process each material and plot type
        for row_idx, material in enumerate(materials):
            for col_idx, plot_def in enumerate(plot_definitions):
                ax = axs[row_idx, col_idx]
                
                try:
                    dataset = all_datasets[material]
                    X0_value = float(X0_values[material])
                    
                    if data_type == 'Original':
                        # Access the original masked data from dataset.objects
                        common_mask = np.logical_and.reduce(list(dataset.masks.values()))
                        
                        if plot_def['type'] == 'target_vs_target':
                            # Get original target data
                            target_obj = dataset.objects['target'][common_mask]  # Shape: [N, 3]
                            
                            x_feature_idx = target_features.index(plot_def['x_feature'])
                            y_feature_idx = target_features.index(plot_def['y_feature'])
                            
                            x_data = target_obj[:, x_feature_idx].numpy()
                            y_data = target_obj[:, y_feature_idx].numpy()
                            
                        elif plot_def['type'] == 'cond_vs_target':
                            if plot_def['x_feature'] in cond_features:
                                # Get condition data
                                cond_obj = dataset.objects['condition'][common_mask]
                                x_feature_idx = cond_features.index(plot_def['x_feature'])
                                x_data = cond_obj[:, x_feature_idx].numpy()
                            
                            if plot_def['y_feature'] in target_features:
                                # Get target data
                                target_obj = dataset.objects['target'][common_mask]
                                y_feature_idx = target_features.index(plot_def['y_feature'])
                                y_data = target_obj[:, y_feature_idx].numpy()
                        
                        print(f"    Original data ranges: x=[{x_data.min():.3g}, {x_data.max():.3g}], y=[{y_data.min():.3g}, {y_data.max():.3g}]")
                        
                    else:
                        # Use transformed data
                        if plot_def['type'] == 'target_vs_target':
                            target_data = dataset.get_transformed_data('target', X0_value, data_type)
                            data_dict = target_data['target'][X0_value][data_type]
                            x_data = data_dict[plot_def['x_feature']]
                            y_data = data_dict[plot_def['y_feature']]
                            
                        elif plot_def['type'] == 'cond_vs_target':
                            if plot_def['x_feature'] in cond_features:
                                cond_data = dataset.get_transformed_data('condition', X0_value, data_type)
                                x_data = cond_data['condition'][X0_value][data_type][plot_def['x_feature']]
                            
                            if plot_def['y_feature'] in target_features:
                                target_data = dataset.get_transformed_data('target', X0_value, data_type)
                                y_data = target_data['target'][X0_value][data_type][plot_def['y_feature']]
                        
                        print(f"    Scaled data ranges: x=[{x_data.min():.3f}, {x_data.max():.3f}], y=[{y_data.min():.3f}, {y_data.max():.3f}]")
                    
                    # Remove invalid data
                    if data_type == 'Original':
                        # For original data, filter out non-positive values for log plots
                        valid_mask = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
                    else:
                        # For scaled data, just filter NaN/inf
                        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
                    
                    x_data = x_data[valid_mask]
                    y_data = y_data[valid_mask]
                    
                    if len(x_data) == 0:
                        raise ValueError("No valid data points")
                    
                    # Create appropriate bins
                    if data_type == 'Original':
                        # Logarithmic bins for original physics data
                        x_bins = np.logspace(np.log10(x_data.min()*0.9), np.log10(x_data.max()*1.1), 51)
                        y_bins = np.logspace(np.log10(y_data.min()*0.9), np.log10(y_data.max()*1.1), 51)
                    else:
                        # Linear bins for scaled data (should be in [-1, 1] range)
                        x_bins = np.linspace(min(-1.1, x_data.min()*1.1), max(1.1, x_data.max()*1.1), 51)
                        y_bins = np.linspace(min(-1.1, y_data.min()*1.1), max(1.1, y_data.max()*1.1), 51)
                    
                    # Create 2D histogram
                    h = ax.hist2d(x_data, y_data, bins=[x_bins, y_bins], 
                                 norm=LogNorm(vmin=1, vmax=None), cmap=cmap, alpha=0.8)
                    
                    # Set appropriate scales
                    if data_type == 'Original':
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                    # Scaled data uses linear scales (default)
                    
                    # Styling
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
                    ax.tick_params(axis='both', labelsize=8)
                    
                    # REMOVED: Info text/legend boxes that were showing inside plots
                    # The following lines have been removed to clean up the plots:
                    # ax.text(0.02, 0.98, info_text, transform=ax.transAxes, ...)
                    
                    print(f"✓ Plotted {material} - {plot_def['xlabel']} vs {plot_def['ylabel']}")
                    
                except Exception as e:
                    print(f"Warning: Could not plot {material} - {plot_def['type']}: {e}")
                    ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=8)
                
                # Set labels and titles
                if row_idx == 0:
                    ax.set_title(f"{plot_def['ylabel']} vs {plot_def['xlabel']}", 
                               fontsize=10, weight='bold')
                
                if col_idx == 0:
                    ax.set_ylabel(f"{material}\n(X0={X0_values[material]:.2e})", 
                                fontsize=10, weight='bold')
                
                if row_idx == num_rows - 1:
                    ax.set_xlabel(plot_def['xlabel'], fontsize=9)
        
        # Add colorbar
        if h:
            cbar = fig.colorbar(h[3], ax=axs, location='right', pad=0.02, shrink=0.8)
            cbar.set_label('Counts in Bin', fontsize=10)
        
        # Save or show
        if save_plots:
            if save_path is None:
                raise ValueError("save_path must be provided when save_plots=True")
            if data_type == 'Original':
                plt.savefig(f"{save_path}/2d_correlations_original.png", 
                           dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"{save_path}/2d_correlations_{data_type}.png", 
                           dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    print("\n--- All 2D correlation plots completed ---")




import matplotlib.pyplot as plt
import numpy as np

def plot_distributions(all_train_datasets, X0_VALUES, 
                       figsize_base_original=3.0,    # Smaller for original data
                       figsize_base_scaled=5.0,      # LARGER for scaled data
                       original_linewidth=2.0,
                       preprocessed_linewidth=2.5,
                       show_plots=True):
    """
    Generate distribution plots with larger size for scaled/preprocessed data.
    Grid layout: rows = variables, columns = materials
    """
    
    print("\n--- Preparing to Generate Distribution Plots with Larger Scaled Data Plot ---")
    
    # Get configuration from first dataset
    first_material = list(all_train_datasets.keys())[0]
    any_dataset = all_train_datasets[first_material]
    plot_config = any_dataset.plotting_config['target']
    plot_labels = plot_config['labels']
    plot_features = plot_config['features']
    num_vars = len(plot_features)
    materials = list(all_train_datasets.keys())
    num_materials = len(materials)
    
    # Get transformations used
    transformations_list = any_dataset.transformations
    scaling_methods = ', '.join(transformations_list)
    
    # Define colors for different transformations
    colors = plt.cm.get_cmap('tab10', len(transformations_list))
    
    # ==============================================================================
    # Figure 1: Original Data Plots - SMALLER SIZE
    # ==============================================================================
    print("--- Generating plots for ORIGINAL (physical scale) data ---")
    
    fig1, axs1 = plt.subplots(num_vars, num_materials, 
                              figsize=(figsize_base_original * num_materials, 
                                       figsize_base_original * num_vars),
                              facecolor='white')
    
    # Handle subplot indexing properly
    if num_vars == 1 and num_materials == 1: 
        axs1 = np.array([[axs1]])
    elif num_vars == 1: 
        axs1 = np.array([axs1])
    elif num_materials == 1: 
        axs1 = np.array([[ax] for ax in axs1])
    
    fig1.suptitle('Distributions of Target Variables (Original Physical Scale)', 
                  fontsize=16, weight='bold')
    
    for var_idx in range(num_vars):
        for material_idx, material in enumerate(materials):
            ax = axs1[var_idx, material_idx]
            dataset = all_train_datasets[material]
            
            try:
                scaled_data = dataset.tensors['target']
                original_scale_data = dataset.scaling.inverse(
                    name='target', x=scaled_data, features=plot_features
                ).numpy()
                data_to_plot = original_scale_data[:, var_idx]
                
                positive_data = data_to_plot[data_to_plot > 0]
                if len(positive_data) > 0:
                    bins = np.logspace(np.log10(np.min(positive_data)), 
                                      np.log10(np.max(positive_data)), 101)
                else:
                    bins = 50
                
                ax.hist(data_to_plot, bins=bins, histtype='step', 
                       linewidth=original_linewidth, color='black')
                
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                
            except (AttributeError, KeyError) as e:
                print(f"Warning: Cannot access original scale data for {material}: {e}")
                ax.text(0.5, 0.5, f'No original data\navailable for {material}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
            
            # Set font sizes for smaller plot
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontsize(10)
    
    # Set titles and labels for Figure 1
    for material_idx, material in enumerate(materials):
        axs1[0, material_idx].set_title(f"{material}\n(X0={X0_VALUES[material]:.2e})", 
                                       fontsize=12)
    for var_idx in range(num_vars):
        axs1[var_idx, 0].set_ylabel(f'Counts ({plot_labels[var_idx]})', fontsize=12)
    
    plt.tight_layout()
    if show_plots:
        plt.show()
    
    # ==============================================================================
    # Figure 2: Preprocessed Data Plots - LARGER SIZE
    # ==============================================================================
    print("\n--- Generating plots for PREPROCESSED data with LARGER size ---")
    
    fig2, axs2 = plt.subplots(num_vars, num_materials, 
                              figsize=(figsize_base_scaled * num_materials,  # LARGER
                                       figsize_base_scaled * num_vars),      # LARGER
                              facecolor='white')
    
    # Handle subplot indexing properly
    if num_vars == 1 and num_materials == 1: 
        axs2 = np.array([[axs2]])
    elif num_vars == 1: 
        axs2 = np.array([axs2])
    elif num_materials == 1: 
        axs2 = np.array([[ax] for ax in axs2])
    
    fig2.suptitle(f'Distributions of Target Variables (Preprocessed: {scaling_methods} Scaling)', 
                  fontsize=20, weight='bold')  # Larger title font
    
    # Create legend
    legend_handles = []
    legend_labels = []
    
    for var_idx in range(num_vars):
        for material_idx, material in enumerate(materials):
            ax = axs2[var_idx, material_idx]
            dataset = all_train_datasets[material]
            
            # Plot all transformations on the same subplot
            for trans_idx, transformation in enumerate(transformations_list):
                try:
                    X0_value = float(X0_VALUES[material])
                    transformed_target_data = dataset.get_transformed_data(
                        data_type='target', 
                        X0_value=X0_value, 
                        transformation=transformation
                    )
                    feature_name = plot_features[var_idx]
                    data_to_plot = transformed_target_data['target'][X0_value][transformation][feature_name]
                    
                    bins = np.linspace(np.min(data_to_plot), np.max(data_to_plot), 101)
                    
                    ax.hist(data_to_plot, bins=bins, histtype='step', 
                           linewidth=preprocessed_linewidth,
                           color=colors(trans_idx), alpha=0.8)
                    
                    # Store legend info only once (from first subplot)
                    if var_idx == 0 and material_idx == 0:
                        legend_handles.append(plt.Line2D([0], [0], color=colors(trans_idx), 
                                                       linewidth=preprocessed_linewidth))
                        legend_labels.append(transformation)
                       
                except (KeyError, AttributeError) as e:
                    print(f"Warning: Cannot access transformed data for {material}, {transformation}: {e}")
                    continue
            
            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            
            # Set font sizes for larger plot
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontsize(12)  # Larger tick labels
    
    # Set titles and labels for Figure 2 - LARGER FONTS
    for material_idx, material in enumerate(materials):
        axs2[0, material_idx].set_title(f"{material}\n(X0={X0_VALUES[material]:.2e})", 
                                       fontsize=16)  # Larger title font
    for var_idx in range(num_vars):
        axs2[var_idx, 0].set_ylabel(f'Counts ({plot_labels[var_idx]})', fontsize=16)  # Larger y-label
    for material_idx in range(num_materials):
        axs2[-1, material_idx].set_xlabel('Value (Normalized Units)', fontsize=16)  # Larger x-label
    
    # Add legend with larger font
    fig2.legend(legend_handles, legend_labels, 
               loc='center left',
               bbox_to_anchor=(1.01, 0.5),
               fontsize=16,  # Larger legend font
               frameon=True,
               fancybox=True,
               shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    if show_plots:
        plt.show()
    
    print("\n--- All plots displayed with larger scaled data plot. ---")
    
    return fig1, fig2






