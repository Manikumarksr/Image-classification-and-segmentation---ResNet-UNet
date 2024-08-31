import numpy as np
import os
import matplotlib.pyplot as plt
import skimage as ski
import pandas as pd

class PCA:

    def __init__(self, n_pcs):
        self.n_pcs = n_pcs
        self.components = None
        self.eigenvalues=None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X -  self.mean #centering mean
        
        cov_matrix= sum([x.reshape(-1,1) @ x.reshape(1,-1) for x in X])/len(X)

        #         calculating eigenvectors and eigenvalues
        eigenvalues,eigenvectors = np.linalg.eig(cov_matrix)
        eigenvectors = eigenvectors.T

        # sorting eigenvectors
        index = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[index]
        self.components = eigenvectors[:self.n_pcs]
        self.eigenvalues=eigenvalues

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)
    
    def inverse_transform(self,X):
        X_rev_transformed = np.dot(X, self.components)
        X_rev_transformed =X_rev_transformed +self.mean
        return X_rev_transformed
    

image=ski.io.imread('C:/Users/ksrma/Documents/courses/mlds/mlds-assignment-2/train/train/Img_1.png',as_gray=True)[:,:256]
pca=PCA(40)
pca.fit(image)
out=pca.transform(image)
out=pca.inverse_transform(out)
# plt.imsave("path",out)
plt.imshow(out)