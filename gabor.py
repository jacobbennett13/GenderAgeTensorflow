#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import cv2
import numpy as np
import scipy.io as sio
import scipy.signal as sgn
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm




def spatial_filter(num_scales, num_orientations, 
                 filter_size, gamma, eta, freq_max):
    
        # Create a placeholder array for output
        filters = np.zeros(shape=(filter_size, filter_size,
                                  num_scales, num_orientations), dtype=np.complex64)
        
        # In the frequency domain, we need to calculate the filter for the 
        # range -f/2 < x < f/2
        size1 = int((filter_size - 1) / 2)
        size2 = int((filter_size - 1) / 2)
        
        # Iterate through scales
        for k in range(0, num_scales):
            # Calculate scale frequency
            fk = freq_max/( np.sqrt(2) ** k)
            
            # Set up some intermediate variables to simplify later maths
            alpha = fk / gamma
            beta = fk / eta
            
            #Iterate through orientations
            for j in range(0, num_orientations):
                gabor = np.zeros((filter_size, filter_size), dtype=np.complex64)
                theta_j = float(j) / num_orientations * np.pi
                
                # Iterate over 2-D image dimensions
                for x in range(-size1, (size1+1)):
                    for y in range(-size2, (size2+1)):
                        xc = x * np.cos(theta_j) + y * np.sin(theta_j)
                        yc = -x * np.sin(theta_j) + y * np.cos(theta_j)
                        
                        gabor[size2+y, size1+x] = (fk**2 / (np.pi * gamma * eta)) * np.exp(- (alpha**2 * xc**2 + beta**2 * yc**2) ) * np.exp((2 * np.pi * fk * xc) * 1j)
            
                # Save 2D filter into index (j,k) i.e. (orientation, scale)      
                filters[:,:,k, j] = gabor
            
        return filters     
        
        
def circular_filter(num_scales, filter_size, gamma, eta, freq_max):
    
     # Create a placeholder array for output
     filters = np.zeros(shape=(filter_size, filter_size, num_scales), dtype=np.complex64)
        
     # In the frequency domain, we need to calculate the filter for the 
     # range -f/2 < f < f/2
     size1 = int((filter_size - 1) / 2)
     size2 = int((filter_size - 1) / 2)
     
     # Iterate through scales
     for k in range(0, num_scales):
         # Calculate scale frequency
         fk = freq_max/( np.sqrt(2) ** k)
            
         # Set up some intermediate variables to simplify later maths
         alpha = fk / gamma
         beta = fk / eta
         
         gabor = np.zeros((filter_size, filter_size), dtype=np.complex64)
         
         # Iterate over 2-D image dimensions
         for x in range(-size1, size1+1):
             for y in range(-size2, size2+1):
                 euclidean_dist = np.sqrt(x**2 + y**2)
                 gabor[size2+y, size1+x] = (fk**2 / (np.pi * gamma * eta)) * np.exp(- (alpha**2 * x**2 + beta**2 * y**2) ) * np.exp(2 * np.pi * fk * euclidean_dist * 1j)
                 
         # Save 2D filter into index (k) i.e. (scale)         
         filters[:,:,k] = gabor
         
     return filters
 
def extend_array(A):
    # Extends a 2D array with dimensions h x w into a 3D array with dimensions h x w x 1
    return A[:, :, np.newaxis]

def mean_pool_3d(A):
    #Performs mean pooling over a 3-dimensional matrix on each separate 2D image plane
    r, c, d = A.shape
    
    try:
        assert(d is not None)
    except:
        print('Dimension error')
        return -1
    
    
    for depth in range(d):
        for row in range(0, r, 2):
            for col in range(0, c, 2):
                avg = (A[row, col, depth] + A[row+1, col, depth] + A[row, col+1, depth] + A[row+1, col+1, depth])/4
                
                A[row, col, depth] = avg
                A[row+1, col, depth] = avg
                A[row, col+1, depth] = avg
                A[row+1, col+1, depth] = avg
        
    
    
    return A


    
 
def directional_features(net, data):
    # Applies directional filtering from net to data
    
    # Convert data to double precision floats
    data = data.astype(np.float64)
    
    # Get dimensions of spatial gabor filter network
    r, c, s, o = net['gabor'].shape 
    
    data_row, data_col, data_chan = data.shape
        
    Rrow = data_row - r + 1
    Ccol = data_col - c + 1
     
    rf = int( (r - 1) / 2 )
    a = net['La']
    sigma = 0.001
    Y = np.zeros(shape=(int(Rrow/2), int(Ccol/2), data_chan, s * o))
    q = 0
    
    cgabor = net['cgabor']
    gabor = net['gabor']
    
    # Iterate over frequency scales
    for n in range(0, s):

        Den = np.abs(sgn.convolve(data, extend_array(cgabor[:,:,n]), mode='same'))
        
        # Iterate over orientations
        for m in range(0, o):
            
            top = np.abs(sgn.convolve(data, extend_array(gabor[:,:,n,m]), mode='same'))
            bot = np.exp(np.divide(-np.power(Den,2), (2 * np.power(sigma,2))), dtype=np.float32) + Den
            
            out = np.divide(top, bot, dtype=np.float32)
            out = out[rf:-rf, rf:-rf, :]
        
            test = mean_pool_3d(out)

            
            # This does not work
            # out = sgn.convolve(out, ones, mode='same')
            

            out = test[0:-1:2, 0:-1:2, :]   # modified out -> test
            s1, s2 ,s3 = out.shape
            
            
            #print(np.mean( np.mean(np.abs(out), 0), 0))
        
            mu = np.tile( np.mean( np.mean(np.abs(out), 0), 0), (s1, s2, 1)) 
            
            Y[:,:,:,q] = np.divide( np.power(out,a), (np.power(out,a) + np.power(mu,a)), dtype=np.float64)
            q += 1
            
            
            
    return Y


def apply_filters(data, filter_size=5, num_orientations=9, num_scales=4,
                  gamma=1, eta=1, freq_max=0.4, La=3):
    
    # Applies gabor filtering with shunting inhibition to input data
    
    # Set up the network filters and variables
    net = {'La': La, 
           'gabor': spatial_filter(num_scales, num_orientations, filter_size,
                                   gamma, eta, freq_max),
           'cgabor': circular_filter(num_scales, filter_size,
                                     gamma, eta, freq_max)}
           
    return directional_features(net, data)
    
        

def demo():              
    filter_size = 7
    num_orientations = 12
    num_scales = 5
    gamma = 1
    eta = 1         
    freq_max = 0.4 
    La = 1          # Naka-Rushton Equation power constant
    
    np.set_printoptions(threshold='nan')
    
    # Dictionary to hold network filters
    net = {'La': La, 
           'gabor': spatial_filter(num_scales, num_orientations, filter_size,
                                   gamma, eta, freq_max),
           'cgabor': circular_filter(num_scales, filter_size,
                                     gamma, eta, freq_max)}
            
           
    data = sio.loadmat('DATA.mat')
    data = data['data']

    

    Y = directional_features(net, data)
  
    
    for k in range(0, num_scales * num_orientations):
        plt.subplot(num_scales, num_orientations, k+1)
               
        
        plt.imshow(Y[:,:,5,k], cmap='gray')
        
    
    plt.show()
    
    
    plt.imshow(data[:,:,5], cmap='gray')
    plt.show()
    
    
def plot_3d():
    
    
    np.set_printoptions(threshold='nan')
    
    # Dictionary to hold network filters
    net = {'La': 3, 
           'gabor': spatial_filter(4, 9, 20, 1, 1, 0.4),
           'cgabor': circular_filter(4, 20, 1, 1, 0.4)}
    
    
    
    
    
    gabor = net['gabor']
    
    x = np.arange(0, gabor.shape[0])
    y = np.arange(0, gabor.shape[1])
    
    X, Y = np.meshgrid(x, y)
    
     
    real_gabor = np.real(gabor)
    im_gabor = np.imag(gabor) 
    
    ax = plt.axes(projection='3d')
    ax.view_init(10,-120)
    ax.plot_surface(X, Y, im_gabor[:,:,0,0], rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')
    
    plt.show()
    
    
def demo_image():
    
    img = cv2.imread('lena.jpg')
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    
    
    filter_size = 3
    num_orientations = 9
    num_scales = 4
    gamma = 1
    eta = 1         
    freq_max = 0.6
    La = 4          # Naka-Rushton Equation power constant
    
    
    # Dictionary to hold network filters
    net = {'La': La, 
           'gabor': spatial_filter(num_scales, num_orientations, filter_size,
                                   gamma, eta, freq_max),
           'cgabor': circular_filter(num_scales, filter_size,
                                     gamma, eta, freq_max)}
            
    

    Y = directional_features(net, extend_array(img))
    
    print(Y.shape)
  
    
    for k in range(0, num_scales * num_orientations):
        plt.subplot(num_scales, num_orientations, k+1)
               
        
        plt.imshow(Y[:,:,0,k], cmap='gray')
        
    
    plt.show()
    
    
    #plt.imshow(img, cmap='gray')
    #plt.show()
    
    
    
    
if __name__ == '__main__': 
    demo()
    
    