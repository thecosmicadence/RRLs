import band_sep as bs
from astropy.io import fits
import numpy as np
import math
import pandas as pd

def vband(period, gaiaid):
    Period=10**period # Calculates the period
    
    # Opens fits file of the corresponding star
    
    filename='EPOCH_PHOTOMETRY-Gaia DR3 '+str(gaiaid)+'.fits' 
    img=fits.open(filename)
    spectra=img[1].data
    j=0
    
    # Checks for magnitude in 'RP' band and counts number of data points (The number of data points in each band is equal
    
    for k in range(len(spectra)):
        if(spectra[k][2]=='RP'):
            j+=1
            
   # To save the spectral data band-wise. Since there are equal number of datapoints for BP, RP and G, we take only the counts of RP

    spectra_RP=[[0 for l in range(2)] for y in range(j)] 
    spectra_BP=[[0 for l in range(2)] for y in range(j)] 
    spectra_G=[[0 for l in range(2)] for y in range(j)] 
    spectra_V=[[0 for l in range(2)] for y in range(j)] 
    
    # Calls bandsep() which separates the data according to the band.
    
    bs.bandsep(j,'G',spectra,spectra_G)
    bs.bandsep(j,'RP',spectra,spectra_RP)
    bs.bandsep(j,'BP',spectra,spectra_BP)
    spectraG=np.array(spectra_G)
    spectraRP=np.array(spectra_RP)
    spectraBP=np.array(spectra_BP)
    spectraV=np.array(spectra_V)
    spectraV=spectraG
    
    # Calculates the magnitude in V band using the relation between G, BP and RP

    for k in range(j):
        if(spectraV[k][1]):
            a=np.subtract(spectraBP[k][1],spectraRP[k][1])
            spectraV[k][1]=np.subtract(spectraV[k][1],-0.02704+0.01424*a-0.2156*a**2+0.01426*a**3)
    
    # Calculation of phase and magnitude

    mask = np.all(spectraV != 0, axis=1)
    SpectraV=spectraV[mask]
    max_index=np.argmin(SpectraV[:,1])
    T0=SpectraV[max_index][0] # Epoch reference time will be the one at which the star is the brightest (minimum mag)
    phase=[0 for l in range(len(SpectraV[:,1]))]
    mag=[0 for l in range(len(SpectraV[:,1]))]
    for k in range(len(SpectraV[:,1])):
        x=np.subtract(SpectraV[k][0],T0)
        phase[k]=(x/Period)-int(x/Period)
        mag[k]=SpectraV[k][1]
        if(phase[k]<0):
            phase[k]+=1
    return phase, mag, Period