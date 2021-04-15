import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import sqrt, inf
from numpy.linalg import norm

#################################################input and preliminaries

# image input and conversion into matrix
img = Image.open('batman.jpg')
og_size = float(os.path.getsize('batman.jpg'))/1024
image = np.matrix(img.getdata(band = 0), float)/255
image.shape = (img.size[1], img.size[0])

print("Image dimensions", image.shape, "Image size %.2f Kb"%og_size)
plt.imshow(image, cmap = 'gray')

# Preliminaries
def make_vector(V): 
    return [np.squeeze(np.asarray(v)) for v in V]

def outer_product(u, v):
    return u*(v.T)

# return e_j, the j-th vector in the standard basis
def e(j, n):
    v = [0]*n; v[j - 1] = 1
    return np.matrix(v).T

#############################################singular salue decomposition

# Power method to calculate maximum eigen value
def maximum_eigen(A, epsilon = 0.0001):
    n = np.shape(A)[0]; u = e(1, n)
    
    while True:
        z = A*u; eig = norm(z); v = z/eig
        
        if norm(v - u) < epsilon: 
            return (eig, v)
        if norm(v + u) < epsilon: 
            return (-eig, v)
        u = v

# The best k-rank approximator
def SVD(A, k = inf):
    n = A.shape[1]
    k = min(k, n)
    S = [0]; V = [e(1, n)]; U = []
    
    P = A.T*A
    for j in range(k):
        eig = S[-1]; v = V[-1]
        P = P - eig*outer_product(v, v)
        
        # The right singular vectors, values
        new_eig, new_v = maximum_eigen(P)
        S.append(new_eig); V.append(new_v)
        
        # The left singular vectors
        if new_eig != 0:
            U.append(A*new_v/sqrt(new_eig))
            
        else: break
            
    del S[0]; del V[0]
    S = [sqrt(x) for x in S]
    return (make_vector(U), S, make_vector(V))

U, S, V = SVD(image, 300)

###################################################constructing output

# Reconstructing the image matrix with k singular vectors
def image_constructor(k):
    m = len(V)
    k = min(m, k)
    U1 = np.matrix(U[:k]).T
    V1 = np.matrix(V[:k])
    S1 = np.diag(S[:k])
    im = Image.fromarray(U1 * S1 * V1 * 255)
    return im

sizes = []
for j in range(1, 11):
    im = image_constructor(j*30)
    print("Image with %d singular vectors."%(j*30))
    plt.figure()
    plt.imshow(im, cmap = 'gray')
    im_name = 'im' + str(j) + '.jpg'
    plt.imsave(im_name, im)
    
    size = float(os.path.getsize(im_name))/1024
    sizes.append((j*30, size))

print("Orgininal size %.2f Kb \n\n No. of singular vectors    Compressed Image Size (kb)"%og_size)
for size in sizes:
    print("%21d %30.2f"%(size[0], size[1]))