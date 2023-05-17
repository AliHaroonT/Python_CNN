# -*- coding: utf-8 -*-
"""
Created on Thu May 11 21:37:33 2023

@author: aliha
"""

import numpy as np


class SlowPool:
  # A Max Pooling layer using stride 1.

  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 patch to slide over the image.
    - image is a 2d numpy array
    '''
    h, w, _ = image.shape
    new_h = h - 1
    new_w = w - 1
    
    for i in range(new_h):
      for j in range(new_w):
        im_region = image[ i : (i + 2) , j : (j  + 2) ]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h - 1, w - 2 , num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    self.last_input = input

    h, w, num_filters = input.shape
    output = np.zeros(((h-1) , (w-1) , num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.amax(im_region, axis=(0, 1))

    return output

# backprop is not working rightnow 
  def backprop(self, d_L_d_out):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''
    d_L_d_input = np.zeros(self.last_input.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      h, w, f = im_region.shape
      amax = np.amax(im_region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            # If this pixel was the max value, copy the gradient to it.
            if im_region[i2, j2, f2] == amax[f2]:
              d_L_d_input[i + i2, j + j2, f2] = d_L_d_out[i, j, f2]

    return d_L_d_input