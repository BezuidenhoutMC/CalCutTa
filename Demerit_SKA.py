#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Derive the 'demerit' score at a position or a list of positions.

import argparse
import re

import numpy as np

from astropy import coordinates as c
from astropy import units as u
from astropy.io import fits


# In[2]:


## COORDS
def make_coords(ras,decs):
    ra_deg = np.empty(len(ras), dtype=np.float32)
    dec_deg = np.empty(len(decs), dtype=np.float32)
    for i, ra in enumerate(ras):
        ra_deg[i] = c.Angle(ra, unit=u.hourangle).deg
    for i, dec in enumerate(decs):
        dec_deg[i] = c.Angle(dec, unit=u.deg).deg
    
    input_coords = c.SkyCoord( ra_deg * u.deg , dec_deg * u.deg )
    return input_coords

## FREQ, BEAMWIDTH
def get_freq(band):
    ## Uing formula FWHM = 66 * lambda/D
    BAND_FREQ = {'1': [750. * u.MHz, 1.75878242 * u.deg],
             '2': [1355. * u.MHz, 0.973495805 * u.deg],
             '3': [2350. * u.MHz, 0.561313538 * u.deg],
             '4': [3990. * u.MHz, 0.330598199 * u.deg],
             '5a': [6550. * u.MHz, 0.2013873 * u.deg],
             '5b': [11850. * u.MHz, 0.111315343 * u.deg]}

    freq, beamfwhm = BAND_FREQ[band]
    return freq,beamfwhm



# In[3]:


def read_cat(catfile,freq):
    ffcat = fits.open(catfile) ## Using AllSky catalog from MKT demerit script
    ffdata = ffcat[1].data
    cat_freq = ffcat[1].header['REF_FREQ'] * u.Hz
    if type(freq) != u.Quantity:
        log.warn('Unknown unit for given frequency, assuming Hz')
    freq = freq << u.Hz
    flux_scale = (freq / cat_freq)**-0.7
    peak_flux = flux_scale * ffdata['PEAK'] * (u.Jy / u.beam)
    catalogue = c.SkyCoord(ffdata['RA(2000)'] * u.deg, ffdata['DEC(2000)'] * u.deg)

    return peak_flux,catalogue
    


# In[4]:


def calc_demerit(ras,decs,band):
    catfile = 'AllSky.fits.gz'
    num_sources=5

    # Default MeerKAT rms dish pointing in degrees.
    RMS_POINTING = 30. * u.arcsec
    # Default MeerKAT rms gain fluctuation.
    RMS_GAIN = 0.01

    search_radius = 1.0  ## search radius in degrees

    # band = '2'
    # ras, decs = ['12:00:00.0','06:00:00.0'], ['-30:00:00.0','0:00:00.0']

    #--------------------------------------
    input_coords = make_coords(ras,decs)

    freq,beamfwhm = get_freq(band)
    limit = beamfwhm * search_radius
    
    peak_flux,catalogue = read_cat(catfile,freq)

    cum = np.load('cum_demerit_L.npy')

    sky_fracs = []
    # print(f"Demerit score results at Band {band}:")
    for i, source in enumerate(input_coords):
        # print(f"{'=' * 80}")
        # print(f"{'Input Position' : ^20} {f'Demerit Score' : ^13} {f'Num. sources from {catfile}' : ^45}")
        # print(f"{' ' *20} {'mJy / beam' : ^13} {f'within {limit : .1f} ({search_radius} x {(beamfwhm << u.arcmin) : <4.1f} FWHM)' : ^45}")
        # print(f"{'=' * 80}")
        seps = catalogue.separation(source)
        keep = np.where(seps < limit)[0]
        seps = seps[keep]

        fwhmratio = seps/beamfwhm
        allatten = np.exp(-4. * np.log(2) * fwhmratio * fwhmratio )

        allflux = peak_flux[keep]
        allattenflux = allflux * allatten

        beamfwhm = beamfwhm << u.deg
        pconst = 8. * np.log(2.) / (beamfwhm * beamfwhm)
        pointing_rms = RMS_POINTING << beamfwhm.unit
        # Pointing flux error
        dsp = pconst * allattenflux * seps * RMS_POINTING
        # Gain error
        dsg = allattenflux * RMS_GAIN
        ds_squared = (dsp * dsp) + (dsg * dsg)

        sort_args = np.argsort(ds_squared)[:-num_sources - 1:-1]
        this_d = np.sqrt(np.sum(ds_squared)) << u.mJy / u.beam
        this_sky_frac = np.interp(this_d.value, cum[0], cum[1])

        # print(f"{source.to_string('hmsdms', precision=0)} {this_d.value : 8.1f} {len(keep) : 27d}")
        # print(f"\n{this_sky_frac : .1f}% of sky has a lower demerit score.")
        # print("\n"
        #       f"    The {len(sort_args)} sources of greatest demerit:\n"
        #       f"    {'-' * 70}\n"
        #       f"    {'Position' : ^20} {'Separation' : ^14} {'Amplitude' : ^14} {'Demerit' : ^14}\n"
        #       f"    {' ' * 20} {'arcmin' : ^14} {'mJy / beam' : ^14} {'mJy / beam' : ^14}\n"
        #       f"    {'-' * 70}")

        for arg in sort_args:
            this_d = np.sqrt(ds_squared[arg]) << u.mJy / u.beam
            max_flux = allflux[arg] << u.mJy / u.beam
            max_pos = catalogue[keep][arg].to_string('hmsdms', precision=0)
            max_sep = seps[arg]
            # print(f"    {max_pos} {max_sep.arcmin : 10.1f} {max_flux.value : 14.1f} {this_d.value : 14.2f}")

        # print(f"{'=' * 80}")
        sky_fracs.append(this_sky_frac)
    return sky_fracs
    # print(sky_fracs)


# In[ ]:




