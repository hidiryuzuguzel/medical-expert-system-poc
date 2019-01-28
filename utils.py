# -*- coding: utf-8 -*-
"""
Created on Tue May 29 00:56:17 2018

@author: hidir
"""

import numpy as np

def nary(IDX, b, m):
    """
    Returns the representation of N in base B using at least M bits

    """
    n = IDX.copy()
    zz = np.zeros((len(n), m))
    
    for k in range(len(n)):
        z = np.zeros(m)
        i = 0
        while n[k]:
            z[i] = n[k] % b
            n[k] = np.fix(n[k]/b)
            i += 1
            
        z = z[::-1] # fliplr
    
        zz[k, :] = z
            
    return zz