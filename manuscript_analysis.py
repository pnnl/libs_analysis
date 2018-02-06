# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:36:39 2017

@author: nune558
"""
#%% Imports
import os

import numpy as np
from skimage.transform import resize

import libs_analysis_module as mod

#%% Set parameters and read in data

# Read in element vs wavelength dictionary
element_wavelengths, element_list = mod.get_element_wavelength_dict()

# Choose elements of interest to see in several plots
elements_to_plot = ['C', 'H', 'Ca', 'K', 'P', 'Fe']

# Names of folders with raw LIBS data
paths = ['InSitu',
         'ExSitu']
delimiter = ','

for j in [0]:#range(len(paths)):
    
    path = paths[j]
    files = [xx for xx in os.listdir(path) if '.txt' in xx]
    
    for i in [0]:#range(len(files)):
        filename = files[i]

#%% Generate necessary data

        # Read in raw data
        all_lambdas, x, y, step_size, data = mod.get_data(path, filename, delimiter)
        
        # Combine by element wavelengths of interest
        data_by_element = mod.gen_data_by_element(data, element_wavelengths, all_lambdas, element_list)
        
        # Run through PCA (it will normalize the data itself)
        pc, score, latent = mod.pca_analysis(data_by_element, i, j)
        
        # Normalize data
        data_by_element = mod.normalize(data_by_element.T).T

        # Generate masks
        true_root = mod.gen_skeleton_mask(x, y, score)
        false_root = mod.manual_pixels(path + str(i), true_root)
        true_root = true_root + false_root # for all others
#        true_root[false_root] = False # for resin root
#        false_root[false_root] = False # for resin root
        skele_root = np.array(mod.gen_skeleton_mask(x, y, score, 2),dtype=np.int32)
        thresh_root = np.array(mod.gen_skeleton_mask(x, y, score, 1),dtype=np.int32)
        combo_root = skele_root + thresh_root
        
        # Generate distance matrix
        distances = mod.gen_distance_matrix(true_root, x, y, score, step_size)

#%% Plotting
        fig_name=None
        # Plot elements
#        fig_name = '%s/%i_elementPlots.png' % (path, i)
        mod.plot_elements(elements_to_plot, element_list, data_by_element, x, y, name=fig_name)

        # Plot PCA image
#        fig_name = '%s/%i_pcaPlots.png' % (path, i)
        mod.plot_pca(x, y, score, name=fig_name)

        # Plot root mask
#        fig_name = '%s/%i_mask' % (path, i)
        mod.plot_root_mask(true_root, false_root, name=fig_name)
        
        # Plot distance matrix
#        fig_name = '%s/%i_distanceMap' % (path, i)
        mod.plot_distance_matrix(distances, name=fig_name)

        # PCA component graph
#        fig_name = '%s/%i_pcaGraph' % (path, i)
        mod.pca_graph(x, y, distances, score, latent, name=fig_name)
        
        # Hot spot comparison
        mod.compare_points(elements_to_plot, element_list, data_by_element, combo_root, x, y, 
                           distances)
        
        # Intensity vs distance plots
        mod.plot_elementint_vs_dist(elements_to_plot, element_list, distances, data_by_element, 
                                    x, y, combo_root)
        
#%% Contour/geodesics analysis

        # Upscale images to find fine grain paths
        temp_distances = resize(np.copy(distances)/np.max(distances), (40, 40)) * np.max(distances)
        temp_true_root = resize(np.copy(true_root), (40, 40))
        
        # Generate line paths
        line_paths = mod.generate_contour_paths(temp_distances, temp_true_root, min_path_len=18)
        
        # Plot images of the line paths
        mod.plot_contour_paths([line_paths[0], line_paths[5], line_paths[11]], distances)
        
        # Generate and plot path trends
        mod.path_trends(data_by_element, elements_to_plot, element_list, temp_distances, 
                           line_paths, x, y)
