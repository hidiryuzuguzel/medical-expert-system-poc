# -*- coding: utf-8 -*-
"""
Created on Sat May 26 01:40:25 2018

@author: hidir
"""

import numpy as np
import itertools
from utils import nary

def make_engine(D, theta0= 0.99, theta=0.02 * np.ones((20, 1)),
                pd = 0.01 * np.ones((10,1)), dnames=[], snames=[]):
    """ 
    Generates an inference engine structure
    Args:
        D: numberOfSymptoms * numberOfDiseases symptom/disease influence matrix
        theta0: Probability of not observing the symptom when no disease is present
        theta:  Probability of not observing the symptom j when a disease is present
        pd:     Prior probability of each diseases
    Returns:
        eng: Inference engine parameter structure    
    """
    
    S, N = D.shape

    assert S == len(theta), 'Mismatch in number of symptoms'
    assert N == len(pd), 'Mismatch in number of diseases'
    assert len(np.unique(D)) == 2, 'symptom/disease influence matrix should be binary'

    
    # Probability matrix not observing a symptom when disease i is present
    pSD = (D * theta) + (1 - D)

    if not dnames:
        for i in range(N):
            dnames.append('Disease' + str(i+1))
            
    if not snames:
        for j in range(S):
            snames.append('Symptom' + str(j+1))
            
    eng = {'S': S, 'N': N, 'pd': pd, 'D': D, 'pSD': pSD, 'th0': theta0, 'th': theta,
           'dn': dnames, 'sn': snames}
    
    return eng


def generate_data(eng, dlist=[]):
    """
    Generates a random disease array and corresponding symptoms
    Args:
        eng: Inference engine
        dlist: List of present diseases
    Returns:
        d_true: Disease array
        s_true: Observed Symptoms
    
    """
    
    # Generate data
    # Generate diseases
    if not dlist:
        d = np.random.rand(eng["N"], 1) < eng["pd"]
    else:
        d = np.zeros((eng["N"], 1), dtype=bool)
        d[dlist] = True
        
    log_ps0 = np.log(eng["th0"]) + np.sum(np.log(eng["pSD"][:, d.astype(int)]), axis=1)
    
    # Generate Symptoms
    s = np.random.rand(eng["S"], 1) < (1 - np.exp(log_ps0))
        
    return d, s
        

def infer_best_k(eng, sidx, so, best_k, MX = 3):
    """
    Infer the best K disease array configurations  
    Args:
            eng: Inference engine
            sidx: observed symptom indices
            so:   Observed symptom values (0/1)
            best_k: Retrieve only the best_k configurations
            MX: Number of maximum diseases to concurrently search for
            
    Returns:
            conf:  Best k configurations
            logP:  Corresponding unnormalized posterior value
    
    """

    N = eng["N"]
    
    # generate disease configurations to search for.
    if MX == 3:
        K = 1 + N + N*(N-1)/2 + N*(N-1)*(N-2)/6
    elif MX == 2:
        K = 1 + N + N*(N-1)/2


    IDX = np.array([0])
    for mx in range(MX):
        nchoosek = list(itertools.combinations(range(N), mx+1))
        IDX = np.concatenate((IDX, np.squeeze(np.sum(2 ** np.array(nchoosek), axis=1))))
        
    dd = nary(IDX, 2, N)

    logP = np.sum((dd @ np.log(eng["pd"])) + ((1-dd) @ np.log(1-eng["pd"])), axis=1)

    for f in range(int(K)):
        for j in range(len(so)):
            if so[j] == 1:
                temp = np.log(1 - eng["th0"] * np.prod(eng["th"][sidx[j]] ** (eng["D"][sidx[j], :] * dd[f, :])))
            else: #if so[j] == 0:
                temp = np.log(eng["th0"]) + np.sum(np.log(eng["th"][sidx[j]]) * (eng["D"][sidx[j], :] * dd[f, :]))
            logP[f] = logP[f] + temp

    mx = np.argsort(logP)[::-1]

    best_k = min(best_k, len(mx))
    conf = nary(IDX[mx[:best_k]], 2, N)
    logPconf = logP[mx[:best_k]]

    return conf, logPconf
