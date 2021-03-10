# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:53:25 2021

@author: Milo
"""
import numpy as np

def _extract_patches(image, patch_size=96):
    '''
    Gets 16 patches from the center of each cell of a 4x4 grid
    '''
    cell_size = (int(image.shape[0]/4), int(image.shape[1]/4))

    patches = []
    labels  = []
    n = 0
    for i in range(0,4):
        start_x = (cell_size[1] / 2) - (patch_size / 2)
        end_x   = start_x + patch_size
        
        #Go to next cell on X axis
        start_x += (i * cell_size[0]) 
        end_x   += (i * cell_size[0]) 
        for j in range(0, 4):    
            start_y = (cell_size[0] / 2) - (patch_size / 2)
            end_y   = start_y + patch_size
   
            #Go to next cell on Y axis
            start_y += (j * cell_size[1])
            end_y   += (j * cell_size[1])
            
            patches.append(image[int(start_x):int(end_x), int(start_y):int(end_y)])
            labels.append(n)
            n+=1
        
    return patches, labels
    

def _extract_crop_edge(image, size_factor, imposed_crop_rectangle, multiple_8pxl):
    '''
    Crops an image with a size factor, it always sticks to an edge.
    '''
    width, height = image.shape[1], image.shape[0]

    if imposed_crop_rectangle == None:
        # print("Croppped")
        x1 = np.random.uniform(0, 1 - size_factor)
        y1 = np.random.uniform(0, 1 - size_factor)
        stick_edge = np.random.randint(0, 4)
        if stick_edge == 0:  # left
            x1 = 0.0
        elif stick_edge == 1:  # right
            x1 = 1.0 - size_factor
        elif stick_edge == 2:  # top
            y1 = 0.0
        else:  # bottom
            y1 = 1.0 - size_factor
        x2 = x1 + size_factor
        y2 = y1 + size_factor

    else:
        x1, x2, y1, y2 = imposed_crop_rectangle

    x1_pxl = max(0, int(x1 * width))
    y1_pxl = max(0, int(y1 * height))
    x2_pxl = min(width, int(x2 * width))
    y2_pxl = min(height, int(y2 * height))

    if multiple_8pxl:
        x1_pxl = max(0, int(round(x1_pxl / 8) * 8))
        y1_pxl = max(0, int(round(y1_pxl / 8) * 8))
        x2_pxl = min(width, int(round(x2_pxl / 8) * 8))
        y2_pxl = min(height, int(round(y2_pxl / 8) * 8))
        x1 = x1_pxl / width
        x2 = x2_pxl / width
        y1 = y1_pxl / height
        y2 = y2_pxl / height

    return image[y1_pxl:y2_pxl, x1_pxl:x2_pxl], [x1, x2, y1, y2], (x1_pxl, x2_pxl, y1_pxl, y2_pxl)


def _extract_random_crop_edge(image, min_factor, max_factor, imposed_crop_rectangle, multiple_8pxl):
    '''
    Returns a random crop that sticks to an edge or corner.
    The aspect ratio is preserved in order to make for a believable photograph.
    '''
    size_factor = np.random.uniform(min_factor, max_factor)
    crop, bounds, bounds_pxl = _extract_crop_edge(
        image, size_factor, imposed_crop_rectangle, multiple_8pxl)
    return crop, bounds, bounds_pxl, size_factor

def _apply_red_transverse_outward_abberation():
    pass

def _apply_blue_transverse_outward_abberation():
    pass
