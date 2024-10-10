# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 15:34:48 2023

@author: mittal
"""

plt.imshow(stimImagesUnConv[...,10])


n = 10
I = np.zeros(window.shape)
I[window] = stimImagesUnConv[:,n]
plt.imshow(I)



n = 10
I = np.zeros(window.shape)
I[window] = stimImages[:,n]
plt.imshow(I)




plt.plot(stimImagesUnConv[4000,:])




plt.plot(stimImagesUnConv[4000,:])
plt.plot(stimImages[4000,:])