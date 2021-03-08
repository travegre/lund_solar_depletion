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
from matplotlib import pyplot as plt


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
    run_id = 'test3'
    
    if not os.path.exists(run_id): 
        os.makedirs(run_id)

    mcmc_fit(run_id, 300, 64)


def mcmc_fit(run_id, niter, nwalkers, state=False, testing=False):

    global g_priors, g_prior_keys, g_selected_abundances
    global data, g_tc, p0

    # Load data and abundance list with condensation temperatures
    data_table = Table.read('test_data.fits')
    
    tc_values = Table.read('Tc_values.txt', format='ascii')
    # create dictionary abundance->Tc for likelihood function
    g_tc = dict(zip(tc_values['El'], tc_values['Tc']))


    '''
    #El     Tc
    Li_fe       1142    
    C_fe        40
    O_fe        180 
    Na_fe       958 
    Mg_fe       1336    
    Al_fe       1653    
    Si_fe       1310    
    K_fe        1006    
    Ca_fe       1517    
    Sc_fe       1659    
    Ti_fe       1582    
    V_fe        1429    
    Cr_fe       1296    
    Mn_fe       1158    
    Fe_fe       1334    
    Co_fe       1352    
    Ni_fe       1353    
    Cu_fe       1037    
    Zn_fe       726 
    Rb_fe       800 
    Sr_fe       1464    
    Y_fe        1659    
    Zr_fe       1741    
    Mo_fe       1590    
    Ru_fe       1551    
    Ba_fe       1455    
    La_fe       1578    
    Ce_fe       1478    
    Nd_fe       1602    
    Sm_fe       1590    
    Eu_fe       1356    
    '''

    

    # SET PRIORS
    # ==========

    prirs = { 
        'f': (0,1),
        'mu_m_D': (-0.01,0.01),
        'sig_m_D': (0,0.01),
        'mu_b_D': (-0.1,0.1),
        'sig_b_D': (0,0.1),
        'mu_m_ND': (-0.01,0.01),
        'sig_m_ND': (0,0.01),
        'mu_b_ND': (-0.1,0.1),
        'sig_b_ND': (0,0.1)    
    }
    # Select the first necessary priors for model parameters
    g_prior_keys = list(prirs.keys())


    # Add intrinsic scatter of abundances to priors
    for el in tc_values:
        prirs.update({'sig_%s_D'%el['El']: (0,0.1), 'sig_%s_ND'%el['El']: (0,0.1)})


    # select a subsample of abundances to fit
    g_selected_abundances = ['O_fe', 'Mg_fe', 'Si_fe', 'Ca_fe', 'Y_fe', 'Ba_fe']
    selected_abun = ['sig_'+x+'_D' for x in g_selected_abundances] + ['sig_'+x+'_ND' for x in g_selected_abundances]     
    g_prior_keys.extend(selected_abun) #prirs.keys()

    data = np.empty((len(data_table), 2*len(g_selected_abundances)))
    for i, abundance in enumerate(g_selected_abundances):
        data[:, 2*i] = data_table[abundance]
        data[:, 2*i+1] = data_table[f'e_{abundance}']**2

    ndim = len(g_prior_keys)
    print("Number of model parameters: ", ndim)

    # Initialise priors of selected model parameters
    g_priors = np.array([prirs[i] for i in g_prior_keys])


    # SET p0, INITIAL WALKER POSITIONS
    # ================================  
    if state:        
        p0 = np.genfromtxt(state, skip_header=1)[-nwalkers:, 1:ndim+1]      
        print("state file loaded p0")
    else:   
        p0 = np.array(g_priors[:,0] + (g_priors[:,1]-g_priors[:,0])*np.random.rand(nwalkers, g_priors.shape[0]))

    # This if-statement is an ugly hack, but it will do the job for now
    if not testing:

        pool = Pool(32) # dedicate 32 cpu threads to the pool, this should be roughly 2-4x what is available on a machine
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[], pool=pool)
      
        print("start mcmc for run %s" % run_id)
        
        start = time.time()  
      
        
        f = open('%s/chain.mcmc' % (run_id), "a")

        f.write('Nwalker %s lnprob\n' % ' '.join(g_prior_keys))
        
        
      
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
    if np.any(g_priors[:, 0] > x) or np.any(g_priors[:, 1] < x):
        return -np.inf

    # Connect parameter keys to walker position values
    pars = dict(zip(g_prior_keys, x))


    t0 = time.time()

    # number of samples drawn from a Gaussian
    Z = 500

    gauss = np.empty((len(g_selected_abundances), 2, 2, Z))
    exponents = np.empty((len(data), len(g_selected_abundances), 2, Z))
    err_term = np.empty(exponents.shape[:3])
    for i, depletion in enumerate(('D', 'ND')):
        for j, param in enumerate(('m', 'b')):
            gauss[:, j, i] = np.random.normal(loc=pars[f'mu_{param}_{depletion}'], scale=pars[f'sig_{param}_{depletion}'], size=Z)
        err_term[..., i] = data[:, 1::2]
        for j, abundance in enumerate(g_selected_abundances):
            err_term[:, j, i] += pars[f'sig_{abundance}_{depletion}']**2
            gauss[j, 0, i] *= g_tc[abundance]
    err_term *= 2
    gauss = np.sum(gauss, axis=1)
    exponents = -(data[:, ::2, np.newaxis, np.newaxis] - gauss[np.newaxis, :])**2/err_term[..., np.newaxis]
    L = np.prod(np.sum(np.exp(exponents), axis=3)/np.sqrt(np.pi*err_term), axis=1)
    L = np.sum(np.log(pars['f']*L[:, 0] + (1-pars['f'])*L[:, 1]), axis=0)


    print('likelihood took: ', time.time()-t0)
    
 
    if np.isnan(L):
        return -np.inf

    return L


def print_exception():
    e = sys.exc_info()
    exc_type, exc_value, exc_traceback = e
    a, j = (traceback.extract_tb(exc_traceback, 1))[0][0:2]
    k = (traceback.format_exception_only(exc_type, exc_value))[0]
    print(a, j, k)


if __name__ == '__main__':
    main()


