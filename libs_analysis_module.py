# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:58:49 2017

@author: nune558
"""
#%% Imports
from math import ceil, floor

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import stats, interpolate
import seaborn as sns
from skimage import measure
from skimage.morphology import skeletonize
from skimage.transform import resize

import cmaputil as cmu

#%% Globals
FAX = 16
FLAB = 20
NSTD = 1.95996 # for 95% CIs: nstd = X standard deviations, 1.95996 = 95% CI
LIBS_TOL = 0.08
C1_1 = (220/255., 41/255., 12/255.)
C1_2 = (45/255., 145/255., 167/255.)
C2_1 = (99/255., 198/255., 10/255.)
C2_2 = (184/255., 156/255., 239/255.)
FIG_DPI = 500

# Register cividis with matplotlib
rgb_cividis = np.loadtxt('cividis.txt').T
cmap = colors.ListedColormap(rgb_cividis.T, name='cividis')
cm.register_cmap(name='cividis', cmap=cmap)

#%% Functions

def find_closest(all_lambdas, value, tol=LIBS_TOL, index=True):
    matches = []
    all_lambdas = list(np.copy(all_lambdas))
    l_temp = list(np.copy(all_lambdas))
    c = min(l_temp, key=lambda x:abs(x - value))
    while (c - value) < tol:
        matches.append(c)
        l_temp.remove(c)
        c = min(l_temp, key=lambda x:abs(x - value))
    if index:
        return [all_lambdas.index(x) for x in matches]
    return matches

def ind_array(all_lambdas):
    ind = np.around(np.linspace(np.min(all_lambdas), np.max(all_lambdas),
                    np.sqrt(len(all_lambdas))), decimals=3)
    return [float(x) for x in ind]

def array2img(x, y, a, pca=False):
    h = int(np.sqrt(len(y)))
    img = np.zeros((h, h))

    x_ind = ind_array(x)
    y_ind = ind_array(y)

    # Map intensity values to image
#    if not pca:
#        a = _normalize(np.sum(np.copy(a), axis=0))

    for i in range(len(a)):
        img[h - y_ind.index(y[i]) - 1, x_ind.index(x[i])] = a[i]

    return img

def img2array(x, y, img):
    h = int(np.sqrt(len(y)))
    a = np.zeros((h * h))

    x_ind = ind_array(x)
    y_ind = ind_array(y)
    
    for i in range(h * h):
        a[i] = img[h - y_ind.index(y[i]) - 1, x_ind.index(x[i])]

    return a

def get_row_col(x, y, i):
    h = int(np.sqrt(len(y)))
    x_ind = ind_array(x)
    y_ind = ind_array(y)
    row = h - y_ind.index(y[i]) - 1
    col = x_ind.index(x[i])
    return row, col
    
def _normalize(col):
    return (col - np.nanmean(col)) / np.nanstd(col)

def normalize(data):
    data = np.copy(data)
    for j in range(data.shape[1]):
        data[:, j] = _normalize(data[:, j])
    return data

def show_element(e, elements, data, x, y, single=True, vmin=-1, vmax=2, name=None, cmap='cividis'):

    img = array2img(x, y, data[elements.index(e), :])
    if single:
        plt.figure(figsize=(10,8))
    ax = plt.gca()
    img = resize(img, (200,200), preserve_range=True)
    im = plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    plt.title(e, fontsize=20)
    plt.axis('off')
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=range(vmin, vmax + 1))
    cbar.ax.tick_params(labelsize=16)
    
    
    if single:
        if name is not None:
            plt.savefig(name, bbox_inches='tight', dpi=FIG_DPI)
        plt.show()
    return img


def plot_elements(elements_to_plot, elements, data, x, y, name=None, cmap='cividis'):
    
    plt.figure(figsize=(24, 16))
    for i in range(len(elements_to_plot)):
        plt.subplot(2, 3, i + 1)
        
        show_element(elements_to_plot[i], elements, data, x, y, single=False, cmap=cmap)

    if name is not None:
        plt.savefig(name, bbox_inches='tight', dpi=FIG_DPI)
    plt.show()

def create_lambda_ind(elements, element_wavelengths, all_lambdas, single_wavelength=False):
    lambda_ind = []
    for e in elements:
        if single_wavelength:
            wls = [element_wavelengths[e][0]]
        else:
            wls = element_wavelengths[e][:]
        new = []
        for i in range(len(wls)):
            new.extend(find_closest(all_lambdas, float(wls[i])))
        new = list(set(new)) # list(set()) to remove repeats
        lambda_ind.append(new)
    return lambda_ind

# Create expected wavelengths vs. elements dictionary
def get_element_wavelength_dict(fname='Element_Intensity_Profiles_Shortlist.csv'):
    
    # Load list of elementy intensity values at each wavelength
    f = np.genfromtxt(fname, dtype=str, delimiter=',')
    
    # Create dictionary with list of wavelengths for each element
    element_wavelengths = {}
    for i in range(f.shape[0]):
        element = f[i, 0]
        wavelengths = [temp for temp in f[i, 1:] if len(temp) > 0]
        element_wavelengths[element] = wavelengths
    
    # Remove elements not expected to be in the soil
    remove_elements = ['Ac', 'Ag', 'At', 'Au', 'Be', 'Bi', 'Br', 'Ce', 'Cs', 'Dy',
                       'Er', 'Eu', 'F', 'Ga', 'Gd', 'Ge', 'He', 'Hf', 'Hg', 'Ho',
                       'I', 'In', 'Ir', 'Kr', 'La', 'Li', 'Lu', 'Nb', 'Nd', 'Ne',
                       'Np', 'Os', 'Pa', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Ra', 'Rb',
                       'Re', 'Rh', 'Rn', 'Ru', 'Sc', 'Se', 'Sm', 'Sn', 'Ta', 'Tb',
                       'Tc', 'Te', 'Th', 'Tl', 'Tm', 'U', 'W', 'Xe', 'Y', 'Yb',
                       'Zr']
    for e in remove_elements:
        for match in [temp for temp in element_wavelengths if e == temp.split(' ')[0]]:
            element_wavelengths.pop(match)
            
    # Only keep combined wavelength elements (ex: keep Fe, remove Fe I and Fe II)
    for match in [temp for temp in element_wavelengths if temp != temp.split(' ')[0]]:
        element_wavelengths.pop(match)
    
    element_list = list(element_wavelengths)
    element_list.sort()
    
    return element_wavelengths, element_list

def get_data(path, filename, delimiter):
    
    # Read in file with all data
    data = np.genfromtxt('%s/%s' % (path, filename), dtype=str, delimiter=delimiter)
    
    # Read in measured wavelengths
    all_lambdas = np.asarray(data[10:-1, 0], dtype=float)
    
    # Read in X and Y positions (Z assumed to stay constant for now)
    x = np.around(np.asarray(data[5, 1:], dtype=float), decimals=3)
    y = np.around(np.asarray(data[6, 1:], dtype=float), decimals=3)
    step_size = y[-2] - y[-1] 
    
    # Create matrix (and norm'd matrix) for intensity data
    data = np.asarray(data[10:-1, 1:], dtype=float)
    
    return all_lambdas, x, y, step_size, data

def gen_data_by_element(data, element_wavelengths, all_lambdas, element_list):
    # Combine by element. Remove all other data
    data_norm = normalize(data.T).T
    data_by_element = np.zeros((len(element_wavelengths), data_norm.shape[1]))

    # Sum up all associated element intensities
    k = 0
    for e in element_list:
    
        # Get the indices for the data matrix with the associated element data
        lambda_ind = create_lambda_ind([e], element_wavelengths, all_lambdas)[0]
        
        # Pull these rows and sum them up
        data_by_element[k, :] = np.sum(data_norm[lambda_ind, :], axis=0)
        
        k += 1
    
    return data_by_element
        
def princomp(data, ignore=[], norm=True):
    data = np.copy(data)
    if norm and len([x for x in np.mean(data.T, axis=1) if abs(x) > 1E-10]) > 0:
        print('Here')
        data = normalize(data)
    [latent, coeff] = np.linalg.eig(np.cov(data.T))
    score = np.dot(coeff.T, data.T).T # projection of the data in the new space
    latent = np.real(latent)
    latent /= sum(latent)
    return coeff, score, latent

def pca_analysis(data, i, j):

    pc, score, latent = princomp(data.T, norm=True)
    
    # Ensure the root is the lowest values
    flip = [[False, True, False],
            [True, True, True],
            [False, False, True, False]]
                
    if flip[j][i]:
        score = score * -1

    return pc, score, latent

def plot_pca(x, y, score, num=3, name=None):
    # num = num PCAs to plot
    # Plot top num prinicipal components
    plt.figure(figsize=(15, num / 2 * 4))
    sp = 1
    for i in range(num):
        plt.subplot(num / 2, 3, sp)
        plt.imshow(array2img(x, y, score[:, i].T, pca=True), cmap='gray', interpolation='none')
        plt.axis('off')
        sp += 1
        
    if name is not None:
            plt.savefig(name, bbox_inches='tight', dpi=FIG_DPI)
    plt.show()

def gen_skeleton_mask(x, y, score, cond=1):
    
    # Generate initial mask
    mask = array2img(x, y, score[:, 0].T, pca=True)
    
    # Normalize
    mask = (np.copy(mask) - np.nanmean(mask)) / np.std(mask)
    
    # Darkest pixels = root
    mask = mask <= -0.85

    # Remove pixels with no neighbors
    if cond > 0:
        for i in range(1, mask.shape[0] - 1):
            for j in range(1, mask.shape[1] - 1):
                if mask[i, j] \
                and not mask[i - 1, j] and not mask[i + 1, j] \
                and not mask[i, j - 1] and not mask[i, j + 1] \
                and not mask[i - 1, j - 1] and not mask[i - 1, j + 1] \
                and not mask[i + 1, j - 1] and not mask[i + 1, j + 1]:
                    mask[i, j] = False
    
    # Skeletonize the root
    if cond > 1:
        mask = skeletonize(mask)
    
    return mask

# Create image to show which pixels in mask belong in each category
def plot_root_mask(true_root, false_root, name=None, cmap='cividis'):

    # Generate RGB image
    img = np.zeros((true_root.shape[0], true_root.shape[1], 3))
    for i in range(true_root.shape[0]):
        for j in range(true_root.shape[1]):
            if true_root[i, j]:
                img[i, j, :] = rgb_cividis[:, 255]
            elif false_root[i,j]:
                img[i, j, :] = rgb_cividis[:, 127]
            else:
                img[i, j, :] = rgb_cividis[:, 0]
    
    # Plot       
    plt.figure(figsize=(8,8))
    plt.imshow(img, interpolation='none')
    plt.axis('off')
    if name is not None:
        plt.savefig(name, bbox_inches='tight', dpi=FIG_DPI)
    plt.show()

def gen_distance_matrix(mask, x, y, score, step_size):

    # Init
    distances = np.array(np.invert(np.copy(mask)), dtype=np.float32)
    ind_root = np.where(distances == 0) # returnind of all root pixels
    
    # Cycle through all pixels and assign distance
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if distances[i, j]: # True -> need to find distance to nearest root pixel
                # 1 == true
                # Init
                b = np.array([i, j])
                min_dist = 100
                min_dist_ind = []
                
                # Compare distance to all root pixels. Find smallest.
                for p in range(len(ind_root[0])):
                    a = np.array([ind_root[0][p],  ind_root[1][p]])
                    temp_dist = abs(np.linalg.norm(a-b))
                    if temp_dist < min_dist:
                        min_dist = temp_dist
                        min_dist_ind = a
                        
                # Only count if this distance is to a true root
                # "mask" needs to be changed to "true_root" if testing true vs false
                if len(min_dist_ind) > 0 and mask[min_dist_ind[0], min_dist_ind[1]]:
                    distances[i, j] = min_dist * float(abs(int(step_size * 1000)))
                else: # Otherwise, throw out
                    distances[i, j] = -1
    
    return distances

def _plot_distance_matrix(distances, cmap='cividis'):
    
    # Get contour lines
    contours = measure.find_contours(distances, 1)
    
    # Plot
    plt.imshow(distances, cmap=cmap, interpolation='none')
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], c='w', linewidth=2, zorder=32)
    plt.axis('off')
    
    return distances
        
def plot_distance_matrix(distances, name=None, cmap='cividis'):

    plt.figure(figsize=(8, 8))
    _plot_distance_matrix(distances, cmap=cmap)
    if name is not None:
        plt.savefig(name, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()
    
def manual_pixels(data_name, true_root, fake=True, rogue=True):
    false_root = np.zeros((true_root.shape), dtype=bool)
           
    if data_name == 'InSitu0':
        if rogue:
            true_root[5, 0] = False
    
    if data_name == 'InSitu1':
        if fake:
            false_root[19, 0] = True # Need to finish
            
    if data_name == 'ResinRoot0':
        if fake:
            false_root[2:8, 5:10] = True
            false_root[3:6, 10] = True
        if rogue:
            false_root[6:8, 14:19] = True

    return false_root

def plot_95ci(x, y):

    sns.set_style('white')
    sns.regplot(x=x, y=y, color='k', order=1, scatter_kws={"color": "white"})

    return

def plot_elementint_vs_dist(elements_to_plot, element_list, dist_matrix, data, 
                            x, y, combo_mask, cols=3, name=None, ignore_root=True):
    
    # Init return variable
    intensities = [[], [], []]

    # Plot each element
    for e in elements_to_plot:

        plt.figure(figsize=(6, 4))
        ind = element_list.index(e)
        
        # Reinit
        intensity = [];       distance = []
        intensity_rhizo = []; distance_rhizo = []
        intensity_root = [];  distance_root = []
        
        for i in range(data.shape[1]):
            row, col = get_row_col(x, y, i)
            
            dist = dist_matrix[row, col]
            
            if dist <= 1000:
                inte = data[ind, i]
                
                # Soil
                if combo_mask[row, col] == 0:
                    if dist < 250:
                        distance_rhizo.append(dist)
                        intensity_rhizo.append(inte)
                    else:
                        distance.append(dist)
                        intensity.append(inte)

                #  Root
                else:
                    distance_root.append(dist)
                    intensity_root.append(inte)
        
        distance = np.array(distance); intensity = np.array(intensity)
        distance_rhizo = np.array(distance_rhizo); intensity_rhizo = np.array(intensity_rhizo)
        distance_root = np.array(distance_root); intensity_root = np.array(intensity_root)

        # Plot trend and 95% CI for soil and rhizosphere
        xx = list(np.concatenate((distance, distance_rhizo)))
        yy = list(np.concatenate((intensity, intensity_rhizo)))
        xx, yy = (list(t) for t in zip(*sorted(zip(xx, yy))))
        plot_95ci(np.array(xx), np.array(yy))
        plot_95ci(np.array(xx), np.array(yy)) # Twice to make it darker
        
        # Scatter plot
        a = 0.4; s = 60; z=32
        plt.scatter(distance, intensity, c='gray', edgecolors='k', alpha=a, s=s, label='Soil', zorder=z)
        plt.scatter(distance_rhizo, intensity_rhizo, c=C1_2, edgecolors=C1_2, alpha=a, s=s, label='Rhizo', zorder=z)
        plt.scatter(distance_root, intensity_root, c=C1_1, edgecolors=C1_1, alpha=a, s=s, label='Root', zorder=z)

        # Append all normalized pixel intensities to the return variable
        intensities[0].append(intensity)
        intensities[1].append(intensity_rhizo)
        intensities[2].append(intensity_root)
        
        # Format plot
        plt.xlabel('Distance ($\mu$m)', fontsize=FLAB)
        plt.ylabel('Normalized Intensity', fontsize=FLAB)
        plt.xticks([0, 500, 1000], fontsize=FAX)
        plt.title(e, fontsize=FLAB)
        plt.axis([-25, 1025, -3, 3]) 
        plt.yticks([-2, 0, 2], fontsize=FAX)
        
        if name is not None:
            plt.savefig('%s_%s' % (name, e), dpi=FIG_DPI)
        plt.show()
    
    return intensities
    
def _plot_ci(x, y, a, c):
    
    # Calculate CI Parameters
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eig(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    theta = np.degrees(np.arctan2(*np.real(vecs[:,0][::-1])))
    w, h = 2 * NSTD * np.sqrt(vals)
    
    # Plot 95% CI
    ell = mpatches.Ellipse(xy=(np.mean(x),np.mean(y)),
          width=w, height=h,
          angle=theta, color=c, alpha=a)
    ell.set_facecolor('none')
    ell.set_edgecolor(c)
    ell.set_linewidth(4)
    
    return ell

def plot_ci(ax, x, y, a=1, c='k', single=False):
    if ax is None:
        plt.figure(figsize=(6,6))
        ax = plt.subplot(111)
    
    if len(x) > 0 and len(y) > 0:
        ell = _plot_ci(x, y, a, c)
        ax.add_artist(ell)

    if ax is None:
        plt.show()

def pca_graph(x, y, distances, score, latent, rhizo_cutoff=250, name=None):
    
    d_array = img2array(x, y, distances)
    root = np.where(d_array == 0)[0]
    rhizo = list(set(np.where(d_array <= rhizo_cutoff)[0]) - set(root))
    soil = np.where(d_array > rhizo_cutoff)[0]
    pc1 = score[:, 0]
    pc2 = score[:, 1]
    
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    xlab = 'PC1 (%i%s)' % (int(latent[0] * 100), '%')
    ylab = 'PC2 (%i%s)' % (int(latent[1] * 100), '%')
    plot_ci(ax, pc1[root], pc2[root], a=1, c=C1_1)
    ax.scatter(pc1[root], pc2[root], c=C1_1, edgecolors=C1_1, label='Root  (d = 0)', s=50)
    plot_ci(ax, pc1[rhizo], pc2[rhizo], a=1, c=C1_2)
    ax.scatter(pc1[rhizo], pc2[rhizo], c=C1_2, edgecolors=C1_2, label='Rhizo (0 < d < %i)' % rhizo_cutoff, s=50)
    plot_ci(ax, pc1[soil], pc2[soil], a=1, c='k')
    ax.scatter(pc1[soil], pc2[soil], c='k', edgecolors='k', label='Soil    (d > %i)' % rhizo_cutoff, s=50)

    plt.xticks([])
    plt.yticks([])
    plt.xlabel(xlab, fontsize=FLAB)
    plt.ylabel(ylab, fontsize=FLAB)
    
    if name is not None:
        plt.savefig(name, dpi=450, bbox_inches='tight')
    plt.show()

    return

def compare_points(elements_to_plot, element_list, data_by_element, combo_mask,
                   x, y, distances,
                   name1=None, name2=None):
    
    data = np.copy(data_by_element)
    
    # Find points based on highest and lowest carbon signature
    c_ind = element_list.index('C')
    xx = data[c_ind, :]
    xx = array2img(x, y, xx)
    xx[np.where(combo_mask != 0)] = np.nan
    xx[distances > 250] = np.nan
    p1 = np.where(xx == np.nanmax(xx))
    xx[np.where(distances != distances[p1])] = np.nan
    p2 = np.where(xx == np.nanmin(xx))
    
    # Overlay points on mask
    rgb, _ = cmu.get_rgb_jab('gray')
    rgb = np.fliplr(rgb)
    data[np.where(data <= 0.0)] = 0
    combo_mask[combo_mask > 0] = 1
    img = cmu.overlay_colormap(combo_mask, rgb)
    img[p1[0], p1[1]] = [255, 0, 0]
    img[p2[0], p2[1]] = [0, 0, 255]
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(img, interpolation='none')
    plt.axis('off')
    if name1 is not None:
        plt.savefig(name1, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

    # Gen bar plot to compare
    plt.figure(figsize=(8, 4))
    i = 1
    plt.plot([0, len(elements_to_plot) + 2], [0, 0], 'k', lw=2)
    for e in elements_to_plot:
        ind = element_list.index(e)
        intensities = data[ind, :]
        img = array2img(x, y, intensities)
        ratio = img[p1[0], p1[1]] - img[p2[0], p2[1]]
#        ratio = np.log2(img[p1[0], p1[1]] / img[p2[0], p2[1]])
        print(img[p1[0], p1[1]] - img[p2[0], p2[1]])
        plt.bar(i, ratio, align='center', color='k', edgecolor='k', bottom=0)
#        plt.text(i, 0.1 * (ratio / abs(ratio)), '%.2f' % ratio, color='w', fontsize=FLAB)
        i += 1
    
    # Formatting
    plt.yticks([], fontsize=FAX)
    plt.xticks(range(1, len(elements_to_plot) + 1), elements_to_plot, fontsize=FAX)
    plt.tick_params(axis='x', which='both', bottom='off', top='off')
    plt.xlabel('Element', fontsize=FLAB)
    plt.ylabel('log2(Raw Int. Ratio)', fontsize=FLAB)
    plt.axis([0.5, 6.5, -1.5, 3.15])    
    if name2 is not None:
        plt.savefig(name2, bbox_inches='tight', dpi=FIG_DPI)
    plt.show()

def contour_path_next(i, j, dx, dy):
    
    dxx = dx[i, j]
    dyy = dy[i, j]
    
    # Returns 0 - 90 degrees
    a = np.degrees(np.arctan(abs(dyy / dxx)))

    # Convert to 0 - 360
    if dyy > 0:
        if dxx > 0:
             # Quad 1
            a = a
        else:
            # Quad 2
            a = 180 - a
    else:
        if dxx < 0:
            # Quad 3
            a = 180 + a
        else:
            # Quad 4
            a = 360 - a
    
    # Up Left
    if a > 301.8198052 and a <= 328.1801949:
        return i - 1, j - 1
    
    # Up
    if (a > 328.1801949 and a <= 360) or (a >= 0 and a <= 31.81980516):
        return i - 1, j
    
    # Up Right
    if a > 31.81980515 and a <= 58.18019485:
        return i - 1, j + 1
    
    # Right
    if a > 58.18019485 and a <= 121.8198052:
        return i, j + 1
    
    # Down Right
    if a > 121.8198052 and a <= 148.1801949:
        return i + 1, j + 1
    
    # Down
    if a > 148.1801949 and a <= 211.8198052:
        return i + 1, j
    
    # Down Left
    if a > 211.8198052 and a <= 238.1801949:
        return i + 1, j - 1
    
    # Left
    if a > 238.1801949 and a <= 301.8198052:
        return i, j - 1

    # Print error and return None if path fell off the edge
    print('Error. dxx=%.2f, dyy=%.2f, a=%.2f' % (dxx, dyy, a))
    return None, None

def contour_path(i, j, dx, dy, root):
    path = [[i, j]]
    steps = 0

    while steps < root.shape[0] * 2 and \
    i >= 0 and i < root.shape[0] and \
    j >= 0 and j < root.shape[1] and \
    not root[i, j]:
        path.append([i, j])
        i, j = contour_path_next(i, j, dx, dy)
        steps += 1

    return path

def generate_contour_paths(distances, root, min_path_len=8):
    
    dx, dy = np.gradient(distances, 2)

    # Generate all paths
    line_paths_list = []
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            
            # Create shortest path to root from this point
            line_path = contour_path(i, j, dx, -dy, root)
            if len(line_path) >= min_path_len:
                line_paths_list.append(line_path)
    
    # Remove paths that are subsets of other paths
    line_paths_sets_list_final = []
    for line_path in line_paths_list:
        add = True
        this_set = set()
        
        # Turn this into a set
        for p in line_path:
            this_set.add(tuple(p))
        
        # Check this path against all others
        for other_path in line_paths_list:
            
            # Don't compare to self
            if line_path != other_path:
                
                # Turn this into a set
                other_set = set()
                for p in other_path:
                    other_set.add(tuple(p))
                
                # Check for subset relationship
                if this_set.issubset(other_set):
                    # This path is a subset of another
                    add = False
        
        # If path made it through cycle without being a subset of any other path, add it to the list.
        if add:
            line_paths_sets_list_final.append(line_path)
    
    return line_paths_sets_list_final

def plot_contour_paths(line_paths, og_distances, min_path_len=8, name=None, row=4,  col=3):
    
    val = np.max(og_distances) * 1.1
    
    # Plot all unique paths
    plt.figure(figsize=(col * 3, row * 3))
    subplot_i = 1
    for line_path in line_paths:
        plt.subplot(row, col, subplot_i)
        
        # Plot distance image with contours
        dist_temp = _plot_distance_matrix(np.copy(og_distances))
        
        # Overlay path on image directly to damp all other pixels
        for p in line_path:
            dist_temp[p[0] / 2, p[1] / 2] = val
        img = cmu.overlay_colormap(dist_temp, rgb_cividis)
        
        # Now overlay as a light gray
        for p in line_path:
            img[p[0] / 2, p[1] / 2] = [240] * 3
        img[line_path[0][0] / 2, line_path[0][1] / 2 - 1] = [240] * 3    

        # Plot (contours from last plot will remain)
        plt.imshow(img, interpolation='none')
        plt.axis('off')
        
        subplot_i += 1
            
    if name is not None:
        plt.savefig(name, bbox_inches='tight', dpi=FIG_DPI)

    plt.show()
    
def path_trends(data, elements_to_plot, element_list, distances, line_paths, 
                    x, y, bin_size=25, name=None):
    
    results_x = []
    results_y = []
    stdev = []
    
    for e in elements_to_plot:
        
        intensities = data[element_list.index(e), :]
        img = array2img(x, y, intensities)
        img = resize(img / np.max(img), (40, 40)) * np.max(img)
        
        temp_x = []
        temp_y = []
        
        for line_path in line_paths:
            
            xx = []
            yy = []

            for p in line_path:
                d = round(distances[p[0], p[1]])
                d = d - (d % bin_size)
                inte = img[p[0], p[1]]
                if d <= 1000:
                    xx.append(d)
                    yy.append(inte)
            
            temp_x.extend(xx)
            temp_y.extend(yy)
        
        # Average instensities across each distance
        dlist = np.sort(list(set(temp_x)))
        iavg = []
        istd = []
        temp_y = np.array(temp_y)
        for d in dlist:
            ind = [i for i in range(len(temp_x)) if temp_x[i] == d]
            iavg.append(np.average(temp_y[ind]))
            istd.append(np.std(temp_y[ind]) / np.sqrt(len(ind)))

        # Add to final results
        results_x.append(dlist)
        results_y.append(iavg)
        stdev.append(istd)

    # Generate plot to compare paths
    plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)

    rgb = cm.viridis
    colors = ['k', 'gray', rgb(50), rgb(125), rgb(200), rgb(255)]
    alpha = [1, 1, 0.75, 0.75, .75, .75]
    
    # Plot line along y=0
    plt.plot([0, 2000], [0, 0], 'k--', lw=3)
    
    # Plot averages for all elements
    for i in range(len(results_y)):
        xx = results_x[i]
        yy = results_y[i]
        std = stdev[i]
        
        plt.plot(xx, yy, c=colors[i], markeredgecolor=colors[i], markersize=5, label=elements_to_plot[i], lw=4, alpha=alpha[i])
        plt.errorbar(xx, yy, yerr=std, ecolor=colors[i], errorevery=2, fmt='none', lw=2, capthick=1)

    # Format
    plt.xlabel('Distance From Root ($\mu$m)', fontsize=12)
    plt.ylabel('Normalized Intensity', fontsize=12)
    plt.xticks([0, 1000], fontsize=12)
    plt.yticks([-0.5, 0, 1, 1.5], fontsize=12)
    plt.axis([90, 1010 , -0.75, 1.6])
    ax.legend(fontsize=12, bbox_to_anchor=(1.2, 0.7))
    plt.tick_params(axis='both', which='both', bottom='off', top='off', right='off')
    
    if name is not None:
        plt.savefig(name, bbox_inches='tight', dpi=600)
    plt.show()

def plot_pca_vs_distance(x, y, distances, score, latent, pca_num=0):
    pca = score[:, pca_num]
    plt.scatter(img2array(x, y, distances), pca)
    plt.xlabel('Distance From Root ($\mu$m)', fontsize=FLAB)
    plt.ylabel('PC%i (%i%s)' % (pca_num + 1, int(latent[pca_num] * 100), '%'), fontsize=FLAB)
    plt.xticks([0, 500, 1000], fontsize=FAX)
    plt.yticks([-100, 0, 100], fontsize=FAX)
    plt.axis([-25, 1025, -100, 140])
