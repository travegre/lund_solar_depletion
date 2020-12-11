import work
import pytest
from astropy.table import Table, join, hstack, Column, vstack
import numpy as np


def test():
    global g_priors, g_prior_keys, g_selected_abundances
    global data, g_tc

    work.data = Table.read('GALAH_DR3_SpOMgSiCaYBa.fits')[:100]
    work.data.write('test_data.fits')
    tc_values = Table.read('Tc_values.txt', format='ascii')
    # create dictionary abundance->Tc for likelihood function
    work.g_tc = dict(zip(tc_values['El'], tc_values['Tc']))

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
    work.g_prior_keys = list(prirs.keys())


    # Add intrinsic scatter of abundances to priors
    for el in tc_values:
        prirs.update({'sig_%s_D'%el['El']: (0,0.1), 'sig_%s_ND'%el['El']: (0,0.1)})


    # select a subsample of abundances to fit
    work.g_selected_abundances = ['O_fe', 'Mg_fe', 'Si_fe', 'Ca_fe', 'Y_fe', 'Ba_fe']
    selected_abun = ['sig_'+x+'_D' for x in work.g_selected_abundances] + ['sig_'+x+'_ND' for x in work.g_selected_abundances]     
    work.g_prior_keys.extend(selected_abun) #prirs.keys()


    ndim = len(work.g_prior_keys)
    print("Number of model parameters: ", ndim)

    # Initialise priors of selected model parameters
    work.g_priors = [prirs[i] for i in work.g_prior_keys]



    # SET p0, INITIAL WALKER POSITIONS
    # ================================  
    np.random.seed(100)
    p0 = np.array([p[0] + (p[1]-p[0])*np.random.rand() for p in work.g_priors])




    a = work.lnprob(p0)
    assert a == pytest.approx(1670.895460353493)
