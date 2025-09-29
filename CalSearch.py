#!/usr/bin/env python
# coding: utf-8

from astroquery.vizier import Vizier
from astroquery.skyview import SkyView
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle, EarthLocation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.visualization import wcsaxes
import Demerit_SKA as demerit
from astropy.coordinates import match_coordinates_sky
import plotly.express as px
import plotly.graph_objects as go

def query_catalog(catalog="J/other/PASA/38.58/galcut", Fpk=">500", DEJ2000="<30"):
    viz = Vizier(columns=["**"])
    viz.ROW_LIMIT=-1

    result = viz.query_constraints(
        catalog=catalog,  #RACS catalog; galcut: |b| > 5 deg; galreg: |b| < 5 deg
        Fpk=Fpk,  # peak flux density in mJy
        DEJ2000=DEJ2000
    )[0].to_pandas()
    return result

def plot_skyview(c, result):
    fig = go.Figure()

    # --- Sources ---
    fig.add_trace(go.Scattergeo(
        lon = result["RAJ2000"],
        lat = result["DEJ2000"],
        text = result["RACS-DR1"],
        marker = dict(
            size = 5,
            color = np.log10(result["Ftot"]),
            colorscale = "Viridis",
            colorbar_title="log(Ftot"
        ),
        mode="markers",
        hovertemplate="<b>%{text}</b><br>log(Ftot): %{marker.color}<br>RA: %{lon}<br>Dec: %{lat}<extra></extra>"
    ))

    # --- RA grid lines ---
    ra_lines = np.arange(0, 360, 30)
    for ra in ra_lines:
        fig.add_trace(go.Scattergeo(
            lon = np.full(181, ra),
            lat = np.linspace(-90, 90, 181),
            mode = 'lines',
            line = dict(color='gray', width=0.5),
            hoverinfo='none',
            showlegend=False
        ))

    # --- Dec grid lines ---
    dec_lines = np.arange(-60, 90, 30)
    for dec in dec_lines:
        fig.add_trace(go.Scattergeo(
            lon = np.linspace(0, 360, 361),
            lat = np.full(361, dec),
            mode = 'lines',
            line = dict(color='gray', width=0.5),
            hoverinfo='none',
            showlegend=False
        ))

    # --- Dec labels at left/right ---
    for dec in dec_lines:
        fig.add_trace(go.Scattergeo(
            lon=[0],  # left
            lat=[dec],
            mode='text',
            text=[f"{dec}°"],
            textfont=dict(color="black", size=10),
            hoverinfo='none',
            showlegend=False
        ))
        fig.add_trace(go.Scattergeo(
            lon=[360],  # right
            lat=[dec],
            mode='text',
            text=[f"{dec}°"],
            textfont=dict(color="black", size=10),
            hoverinfo='none',
            showlegend=False
        ))
    
    # RA labels at the 0 deg line
    ra_labels = np.arange(0, 360, 30)
    fig.add_trace(go.Scattergeo(
        lon=ra_labels,
        lat=np.full_like(ra_labels, 0),  # place at bottom edge
        text=[f"{ra}°" for ra in ra_labels],
        mode="text",
        showlegend=False,
        hoverinfo="none"
    ))


    # --- Layout ---
    fig.update_geos(
        projection_type="mollweide",
        showland=False,
        showcoastlines=False,
    )

    fig.update_layout(
        title="Sky map",
        margin={"r":0,"t":30,"l":0,"b":0},
    )
    fig.update_layout(
        autosize=True
    )
    fig.show()

    # fig, ax = plt.subplots(figsize=(16, 8), subplot_kw=dict(projection="mollweide"))
    # ax.grid(True)
    # ax.scatter(ra_rad, dec_rad, marker="o", s=2, alpha=0.3)
    # fig.subplots_adjust(top=0.95, bottom=0.0)
    # plt.show()
    # print(len(c),'sources after cut')



def filter_unresolved(c,result,tolerance=0.05):
    ratio = result["Ftot"]/result["Fpk"]
    ratio_cut = (ratio > 1. - tolerance) & (ratio < 1. + tolerance)
    racs_compact = c[ratio_cut]
    result = result[ratio_cut]
    # plot_skyview(racs_compact)
    # print(len(racs_compact),"compact sources")
    return racs_compact,result

def filter_demerit(c,result,band='2',cutoff=95):
    ra_rad = c.ra.wrap_at(180 * u.deg).radian
    dec_rad = c.dec.radian
    demerits = demerit.calc_demerit(ra_rad, dec_rad,band)
    demerits_mask = np.asarray(demerits) < cutoff

    c = c[demerits_mask]
    result = result[demerits_mask]

    return c,result

def filter_ICRF3(c,result,tolerance=3*u.arcsec):
    viz = Vizier(columns=["**"])
    viz.ROW_LIMIT=-1
    icrf3 = viz.get_catalogs("J/A+A/644/A159")[0].to_pandas()
    icrf_coords = SkyCoord(
        ra=icrf3["RAICRS"].values,
        dec=icrf3["DEICRS"].values,
        unit=(u.hourangle, u.deg),  # RA in hours, Dec in degrees (sexagesimal accepted)
        frame='icrs'
    )

    mask = (icrf_coords.dec.deg > -90) & (icrf_coords.dec.deg < 30)

    icrf3_cut = icrf3[mask]
    icrf_coords_cut = icrf_coords[mask]  # keep the SkyCoord aligned

    # print(len(icrf3), 'ICRF sources before dec cut')
    # print(len(icrf_coords_cut), 'ICRF sources after dec cut')

    idx, sep2d, _ = c.match_to_catalog_sky(icrf_coords_cut)
    match_mask = sep2d < tolerance  # accept matches within 3" default tolerance
    racs_AND_icrf = c[match_mask]
    result = result[match_mask]
    # print(len(racs_AND_icrf), "RACS unresolved sources matched to ICRF3")
    return racs_AND_icrf,result

def run_search(catname, Fpk=">500", DEJ2000="<30", check_fluxratio=False, check_icrf3=False, check_demerit=False, band='2'):
    result = query_catalog(catname, Fpk=Fpk, DEJ2000=DEJ2000)
    ra_vals = result["RAJ2000"].to_numpy() * u.degree
    dec_vals = result["DEJ2000"].to_numpy() * u.degree
    c = SkyCoord(ra=ra_vals, dec=dec_vals, frame='icrs')
    print(len(c),'catalog sources after initial cut')

    if check_fluxratio:
        c,result = filter_unresolved(c,result)
        # print(len(c),'compact sources')

    if check_icrf3:
        c,result = filter_ICRF3(c,result)
        # print(len(c),'sources matched to ICRF3')

    if check_demerit:
        c,result = filter_demerit(c,result,band=band)
        # print(len(c),'sources after demerit cut')

    plot_skyview(c,result)
    print(len(c),'sources after filtering')

# run_search("J/other/PASA/38.58/galcut", Fpk=">500", DEJ2000="<30", check_fluxratio=False, check_icrf3=False, check_demerit=False, band='2')