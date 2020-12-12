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
    run_id = 'test5'
    
    if not os.path.exists(run_id): 
        os.makedirs(run_id)

    mcmc_fit(run_id, 3000, 128)


def mcmc_fit(run_id, niter, nwalkers, state=False):

    global g_priors, g_prior_keys, g_selected_abundances
    global data, g_tc

    # Load data and abundance list with condensation temperatures
    data = Table.read('GALAH_DR3_SpOMgSiCaYBa_solar.fits')

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
        prirs.update({'sig_%s_D'%el['El']: (0,0.2), 'sig_%s_ND'%el['El']: (0,0.2)})


    # select a subsample of abundances to fit
    g_selected_abundances = ['Ni_fe', 'Mg_fe', 'Si_fe', 'Ca_fe', 'Al_fe']
    selected_abun = ['sig_'+x+'_D' for x in g_selected_abundances] + ['sig_'+x+'_ND' for x in g_selected_abundances]     
    g_prior_keys.extend(selected_abun) #prirs.keys()

    # Clean data of nan values for abundances
    mask = data[g_selected_abundances[0]] > -9999
    for i in g_selected_abundances:
        mask = mask & (~np.isnan(data[i]))
    data = data[mask][:3000]


    ndim = len(g_prior_keys)
    print("Number of model parameters: ", ndim)

    # Initialise priors of selected model parameters
    g_priors = [prirs[i] for i in g_prior_keys]



    # SET p0, INITIAL WALKER POSITIONS
    # ================================  
    if state:        
        p0 = np.genfromtxt(state, skip_header=1)[-nwalkers:, 1:ndim+1]      
        print("state file loaded p0")
    else:   
        p0 = np.array([[p[0] + (p[1]-p[0])*np.random.rand() for p in g_priors] for i in range(nwalkers)])




    pool = Pool(64) # dedicate 32 cpu threads to the pool, this should be roughly 2-4x what is available on a machine
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
    if not np.all([g_priors[i][0] < x[i] < g_priors[i][1] for i in range(len(g_priors))]):    
        return -np.inf

    # Connect parameter keys to walker position values
    pars = dict(zip(g_prior_keys, x))


    t0 = time.time()

    # number of samples drawn from a Gaussian
    Z = 500
    gauss_m_D = np.random.normal(loc=pars['mu_m_D'], scale=pars['sig_m_D'], size=Z)
    gauss_b_D = np.random.normal(loc=pars['mu_b_D'], scale=pars['sig_b_D'], size=Z)
    gauss_m_ND = np.random.normal(loc=pars['mu_m_ND'], scale=pars['sig_m_ND'], size=Z)
    gauss_b_ND = np.random.normal(loc=pars['mu_b_ND'], scale=pars['sig_b_ND'], size=Z)
    #plt.hist(gauss_m_D)
    #plt.show()

    L = 0
    for i in data:
        Li_D = 1
        Li_ND = 1

        for j in g_selected_abundances:
            err_term = i['e_'+j]**2 + pars[f'sig_{j}_D']**2            
            Li_D *= np.sum(   (1./np.sqrt(2*np.pi*err_term)) * np.exp( -((i[j] - (gauss_m_D*g_tc[j] + gauss_b_D))**2)/(2*err_term) )   )

            err_term = i['e_'+j]**2 + pars[f'sig_{j}_ND']**2
            Li_ND *= np.sum(   (1./np.sqrt(2*np.pi*err_term)) * np.exp( -((i[j] - (gauss_m_ND*g_tc[j] + gauss_b_ND))**2)/(2*err_term) )   )
        
        L += np.log(pars['f']*Li_D + (1-pars['f'])*Li_ND)

    print(L)
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


