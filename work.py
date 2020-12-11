import os, shutil, operator
import sys
import numpy as np

import logging

import multiprocessing
from multiprocessing import Pool
import time
import traceback

import emcee

import astropy
from astropy.table import Table, join, hstack, Column, vstack



#https://arxiv.org/pdf/1510.07674.pdf, IAU 2015 Resolution B3 on Recommended Nominal Conversion Constants for Selected Solar and Planetary Properties
Rsun = 6.957*(10**8)
Lsun = 3.828*(10**26)
Tsun = 5772
GMsun = 1.3271244*(10**20)

G = 6.67408*(10**-11) # 2014 CODATA

# http://maia.usno.navy.mil/NSFA/NSFA_cbe.html
c = 299792.458

SIGMA = 5.670367*(10**-8) # 2014 CODATA

# Using the IAU 2012 Resolution B2 definition of the astronomical unit, the parsec corresponds to 3.085 677 581
KPCTOM = 3.085677581*(10**19)




def main():
    run_id = 'test'
    
    if not os.path.exists(run_id): 
        os.makedirs(run_id)

    mcmc_fit(run_id, 1000, 64)


def mcmc_fit(run_id, niter, nwalkers, state=False):

    global g_priors
    global data

    data = Table.read('GALAH_DR3_SpOMgSiCaYBa.fits')

    print(data.colnames)

    fdsfsd

    # SET PRIORS
    # ==========

    prirs = { 
        'par0': (3800,7300)
    
    }
    # Select the priors for model parameters, for now we take all
    ndim = len(prirs.keys())
    g_priors = [prirs[i] for i in prirs.keys()]



    # SET p0, INITIAL WALKER POSITIONS
    # ================================  
    if state:        
        p0 = np.genfromtxt(state, skip_header=1)[-nwalkers:, 1:ndim+1]      
        print("state file loaded p0")
    else:   
        p0 = np.array([[p[0] + (p[1]-p[0])*np.random.rand() for p in g_priors] for i in range(nwalkers)])




    pool = Pool(32) # dedicate 32 cpu threads to the pool, this should be roughly 2-4x what is available on a machine
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[], pool=pool)
  
    print("start mcmc for run %s" % run_id)
    
    start = time.time()  
  
    
    f = open('%s/chain.mcmc' % (run_id), "a")
    
    
  
    for result in sampler.sample(p0, iterations=niter, store=False):        
        
        position, log_prob, random_state = result
    
        for k in range(position.shape[0]):
      
      
            f.write("%d %s %f\n" % (k, # walker #
                                    " ".join(['%.5f' % i for i in position[k]]), # adjpars
                                    log_prob[k] # lnprob value
                                )
            )      
     
    f.close()
    
  
    print("time elapsed: %s" % str((time.time()-start)/3600.))

    print("end mcmc for run %s" % run_id)
    

    return True


def lnprob(x):  
    
    # effectively reject any walk into forbidden parameter space
    if not np.all([g_priors[i][0] < x[i] < g_priors[i][1] for i in range(len(g_priors))]):    
        return -np.inf
   
    # compute the loglikelihood
    chi2 = 0 # we have to write this up
    
    lnp = -0.5 * chi2
 
    if np.isnan(lnp):
        return -np.inf

    return lnp


def print_exception():
    e = sys.exc_info()
    exc_type, exc_value, exc_traceback = e
    a, j = (traceback.extract_tb(exc_traceback, 1))[0][0:2]
    k = (traceback.format_exception_only(exc_type, exc_value))[0]
    print(a, j, k)


if __name__ == '__main__':
    main()