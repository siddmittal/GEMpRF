#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:01:02 2024

@author: mwoletz
"""

import numpy as np
from sklearn.neighbors import KDTree
import matplotlib .pyplot as plt

class NeighbourLookup:
    def __init__(self, points):
        self.__points = np.asfarray(points)
        self.__N      = self.points.shape[0]
        self.__D      = self.points.shape[1]
        
        # create lists for unique values and the mappings for each dimension
        self.__unique_values              = []
        self.__unique_mappings            = []
        
        # store also trees for the unique values for fast lookup
        self.__unique_trees               = []
        
        self.__less_than_trees            = []
        self.__greater_than_trees         = []        
        
        self.__less_than_trees_indices    = []
        self.__greater_than_trees_indices = []
        
        for d in range(self.D):
            # unique will already sort the values, the mapping will be a vector of length N and will contain the indices for each point to the unique array
            unique_values, unique_mapping = np.unique(self.points[:,d], return_inverse=True)
            self.__unique_values.append(unique_values)
            self.__unique_mappings.append(unique_mapping)
            
            self.__unique_trees.append(KDTree(unique_values[:,None]))
            
            less_than_trees    = []
            greater_than_trees = []
            
            less_than_trees_indices    = []
            greater_than_trees_indices = []
            
            # build the trees in each dimension for smaller and larger values
            for i in range(len(unique_values)):
                lt_tree = None
                lt_indices = np.array([])
                
                gt_tree = None
                gt_indices = np.array([])
                
                if i > 0:
                    mask = unique_mapping < i
                    lt_indices = np.nonzero(mask)[0]
                    
                    lt_tree = KDTree(self.points[mask])
                if i < len(unique_values) - 1:
                    mask = unique_mapping > i
                    gt_indices = np.nonzero(mask)[0]
                    
                    gt_tree = KDTree(self.points[mask])
                
                less_than_trees.append(lt_tree)
                less_than_trees_indices.append(lt_indices)
                
                greater_than_trees.append(gt_tree)
                greater_than_trees_indices.append(gt_indices)
            
            self.__less_than_trees.append(less_than_trees)
            self.__greater_than_trees.append(greater_than_trees)
            
            self.__less_than_trees_indices.append(less_than_trees_indices)
            self.__greater_than_trees_indices.append(greater_than_trees_indices)
            
        self.__neighbours = []
        
        for p in self.points:
            self.__neighbours.append(self.query(p))
            
        self.__neighbours = np.array(self.__neighbours, dtype=object)
    
    @property
    def points(self):
        return self.__points
    
    @property
    def N(self):
        return self.__N
    
    @property
    def D(self):
        return self.__D
    
    def query(self, p):
        neighbours = []
        
        for d in range(self.D):
            pd = p[d]
            
            unique_index = self.__unique_trees[d].query(np.atleast_2d(pd), return_distance=False)[0,0]
            
            lt_tree = self.__less_than_trees[d][unique_index]
            gt_tree = self.__greater_than_trees[d][unique_index]
            
            if lt_tree is not None:
                n_lt = lt_tree.query([p], return_distance=False)[0,0]
                neighbours.append(self.__less_than_trees_indices[d][unique_index][n_lt])
                
            if gt_tree is not None:
                n_gt = gt_tree.query([p], return_distance=False)[0,0]
                neighbours.append(self.__greater_than_trees_indices[d][unique_index][n_gt])
        
        return np.array(neighbours)
    
    @property
    def neighbours(self):
        return self.__neighbours
    
if __name__ == "__main__":
    # x = np.linspace(-10, 10, 50)
    # y = np.linspace(-10, 10, 100)
    # s = np.linspace(0.5, 5, 8)

    x = np.linspace(-10, 10, 5)
    y = np.linspace(-10, 10, 5)
    s = np.linspace(1, 5, 5)

    # X,Y,S = np.meshgrid(x,y,s, indexing='ij')
    X,Y,S = np.meshgrid(x,y,s)

    points = np.vstack((X.flatten(), Y.flatten(), S.flatten())).T
    
    lookup = NeighbourLookup(points)
    
    # print(lookup.neighbours)

    # Test
    query_point = np.array([0,0,2])
    print(points[lookup.query(query_point)])
            

    # Plot all points in 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='All Points')

    # Plot the query point in 3D
    ax.scatter(query_point[0], query_point[1], query_point[2], color='red', label='Query Point')

    # Plot the nearest neighbors in 3D
    nearest_neighbors = points[lookup.query(query_point)] # points[indices[0]]  # Get nearest neighbors using indices
    ax.scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1], nearest_neighbors[:, 2], color='green', label='Nearest Neighbors')

    # Connect the query point with its nearest neighbors in 3D
    for neighbor in nearest_neighbors:
        ax.plot([query_point[0], neighbor[0]], [query_point[1], neighbor[1]], [query_point[2], neighbor[2]], color='gray', linestyle='--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('KD-Tree Nearest Neighbors in 3D')
    ax.legend()
    plt.show()

    print