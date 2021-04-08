import os, shutil, operator
import sys
import numpy as np

import logging

import time
import traceback

import pandas as pd
import corner

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
    run_id = 'test31'
    walkers_corner(run_id, 256)
    
    
def walkers_corner(run_id, nwalkers):

    
    if False:
        data = Table.read('APOGEE_Nibauer_selection.fits')
        g_selected_abundances = ['Si_fe', 'Mg_fe', 'Ni_fe', 'Ca_fe', 'Al_fe']
    
        # Clean data of nan values for abundances
        mask = data[g_selected_abundances[0]] > -9999
        for i in g_selected_abundances:
            mask = mask & (~np.isnan(data[i]))
        data = data[mask]

        fig, axes = plt.subplots(figsize=(20,4), dpi=100, nrows=1, ncols=len(g_selected_abundances))
        for i in range(len(g_selected_abundances)):
            ax = axes[i]
            ax.scatter(data['fe_h'], data['e_'+g_selected_abundances[i]], s=1, color='black', edgecolor='none', alpha=0.9)
            ax.set_xlabel('[Fe/H]')
            ax.set_ylabel('['+g_selected_abundances[i]+']')
            ax.set_ylim([-0.3, 0.3])
        plt.tight_layout()
        plt.show()

        
    samples_all = pd.read_csv('%s/chain.mcmc' % (run_id), '\s+', header=0)
    labels = list(samples_all.columns)[1:-1]
    print(labels)

    samples_all = np.array(samples_all).astype(float)
    samples = samples_all[:, 1:]
    

    fig = plt.figure()
    # show walkers convergence
    for i in range(nwalkers):        
      plt.plot(samples_all[i::nwalkers][:, -1])       

    fig.savefig(run_id+'/walkers_test.png')  
    plt.show()
    plt.close(fig)
    

    burn_cut_up = -1
    lnprob_lim_high = np.inf
    lnprob_lim_low = 250500

    burn_cut = -5000

   

    samples_by_walkers = []

    for i in range(nwalkers):
        wmin = min(samples_all[i::nwalkers][burn_cut:burn_cut_up, -1])
        wmax = max(samples_all[i::nwalkers][burn_cut:burn_cut_up, -1])

        if (wmin > lnprob_lim_low) & (wmax < lnprob_lim_high):
            samples_by_walkers.append(samples[i::nwalkers][burn_cut:burn_cut_up])

      
    samples = np.concatenate(np.transpose(np.array(samples_by_walkers), (1,0,2)))  
    samples = samples[:, 0:len(labels)]

    averages = list(zip(np.nanmean(samples, axis=0), np.nanstd(samples, axis=0)))
    averages = np.array(averages)
    
    percentiles_posterior = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
 
    
    # print results
    #print "\nMCMC RESULTS:\n"
    mcmc_results = {}
    mcmc_forfile = []

    for x in zip(labels, percentiles_posterior):
        mcmc_results[x[0]] = x[1]
        #print('%s & %.2f \pm^{%.2f}_{%.2f}' % (x[0], x[1][0], x[1][1], x[1][2]))
        print(x[0], x[1])
        mcmc_forfile.extend(x[1])     

    
    printed_labels = labels[:9]

    new_samples = []
    new_averages = []
    for lab in printed_labels:
        new_samples.append(samples[:, printed_labels.index(lab)])
        new_averages.append(averages[printed_labels.index(lab)])
    new_samples = np.array(new_samples).T
    new_averages = np.array(new_averages)
    
    fig = corner.corner(new_samples, quantiles=[0.16, 0.5, 0.84], labels=printed_labels, label_kwargs={"fontsize": 20}, show_titles=True, title_fmt='.2f', title_kwargs={"fontsize": 16}, truths=new_averages[:, 0])
    
    
    '''
    ndim = len(printed_labels)
    axes = np.array(fig.axes).reshape((ndim, ndim))
    
    gauss_forfile = []
    for m in range(ndim):          
      ax = axes[m, m]
      x, y = [], []
      xys = ax.patches[0].get_xy()[1:-1]
      for n, xy in enumerate(xys[::2]):             
        x.append((xys[2*n+1][0]+xy[0])/2.0)
        y.append(xy[1])
      
      # Fit the data using a Gaussian
      g_init.amplitude = max(y)
      g_init.mean = np.median(x)
      g_init.stddev = np.std(x)
      
      g = fit_g(g_init, x, y)

      #g_amp = g.amplitude[0]
      g_mean = g.mean[0]
      g_std = g.stddev[0]
      gauss_forfile.extend([g_mean, g_std])
      # Plot the data with the best-fit model
      ax.plot(x, g(x), label='Gaussian')
      ax.axvline(x=g_mean, linewidth=1, color='r')
      #ax.scatter(hist[:, 0], hist[:, 1])
    
    #all_res_gauss.append([spec_id, media, sigma, is_converged] + gauss_forfile)
    '''

    fig.savefig(run_id+'/triangle.png')
    plt.show()    
    plt.close(fig)
    
    # Plot intrinsic abundance scatter
    printed_labels = labels[9:]

    new_samples = []
    new_averages = []
    for lab in printed_labels:
        new_samples.append(samples[:, printed_labels.index(lab)+9])
        new_averages.append(averages[printed_labels.index(lab)+9])
    new_samples = np.array(new_samples).T
    new_averages = np.array(new_averages)
    
    fig = corner.corner(new_samples, quantiles=[0.16, 0.5, 0.84], labels=printed_labels, label_kwargs={"fontsize": 20}, show_titles=True, title_fmt='.2f', title_kwargs={"fontsize": 16}, truths=new_averages[:, 0])

    fig.savefig(run_id+'/triangle2.png')
    plt.show()    
    plt.close(fig)


def print_exception():
    e = sys.exc_info()
    exc_type, exc_value, exc_traceback = e
    a, j = (traceback.extract_tb(exc_traceback, 1))[0][0:2]
    k = (traceback.format_exception_only(exc_type, exc_value))[0]
    print(a, j, k)


if __name__ == '__main__':
    main()