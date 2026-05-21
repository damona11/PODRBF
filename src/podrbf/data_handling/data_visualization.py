# src/podrbf/data_handling/data_visualization.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_crosssection(ds, field='u', t_idx=0, z_idx=None):
    """
    Plot field on x-y cross-section at a given time and z index.

    Parameters
    ----------
    ds    : xarray Dataset
    field : 'u', 'v', or 'w'
    t_idx : time index
    z_idx : z index — defaults to mid-plane
    """
    if z_idx is None:
        z_idx = ds.sizes['z'] // 2

    q = ds[field].isel(time=t_idx, z=z_idx).values

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(ds['x'].values, ds['y'].values, q.T,
                       cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label=field)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{field}  t={ds["time"].values[t_idx]:.2f}  z={ds["z"].values[z_idx]:.2f}')
    ax.set_aspect('equal')
    plt.tight_layout()


def plot_axial_slice(ds, field='w', t_idx=0, y_idx=None):
    """
    Plot field on x-z plane at a given time and y index.

    Parameters
    ----------
    ds    : xarray Dataset
    field : 'u', 'v', or 'w'
    t_idx : time index
    y_idx : y index — defaults to mid-plane
    """
    if y_idx is None:
        y_idx = ds.sizes['y'] // 2

    q = ds[field].isel(time=t_idx, y=y_idx).values

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.pcolormesh(ds['z'].values, ds['x'].values, q,
                       cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label=field)
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_title(f'{field}  t={ds["time"].values[t_idx]:.2f}  y={ds["y"].values[y_idx]:.2f}')
    plt.tight_layout()


def plot_time_evolution(ds, field='w', x_idx=None, y_idx=None, z_idx=None):
    """
    Plot field at a single point over all time steps.

    Parameters
    ----------
    ds    : xarray Dataset
    field : 'u', 'v', or 'w'
    x_idx : x index — defaults to mid-plane
    y_idx : y index — defaults to mid-plane
    z_idx : z index — defaults to mid-plane
    """
    if x_idx is None:
        x_idx = ds.sizes['x'] // 2
    if y_idx is None:
        y_idx = ds.sizes['y'] // 2
    if z_idx is None:
        z_idx = ds.sizes['z'] // 2

    q = ds[field].isel(x=x_idx, y=y_idx, z=z_idx).values

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(ds['time'].values, q, linewidth=1.2)
    ax.set_xlabel('time')
    ax.set_ylabel(field)
    ax.set_title(f'{field}  x={ds["x"].values[x_idx]:.2f}  '
                 f'y={ds["y"].values[y_idx]:.2f}  '
                 f'z={ds["z"].values[z_idx]:.2f}')
    plt.tight_layout()


def plot_velocity_magnitude(ds, t_idx=0, z_idx=None):
    """
    Plot velocity magnitude sqrt(u^2 + v^2 + w^2) on x-y cross-section.

    Parameters
    ----------
    ds    : xarray Dataset
    t_idx : time index
    z_idx : z index — defaults to mid-plane
    """
    if z_idx is None:
        z_idx = ds.sizes['z'] // 2

    u   = ds['u'].isel(time=t_idx, z=z_idx).values
    v   = ds['v'].isel(time=t_idx, z=z_idx).values
    w   = ds['w'].isel(time=t_idx, z=z_idx).values
    mag = np.sqrt(u**2 + v**2 + w**2)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(ds['x'].values, ds['y'].values, mag.T,
                       cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label='|U|')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'|U|  t={ds["time"].values[t_idx]:.2f}  z={ds["z"].values[z_idx]:.2f}')
    ax.set_aspect('equal')
    plt.tight_layout()


def plot_summary(ds, t_idx=0):
    """
    Single figure — cross-section of u, v, w and velocity magnitude.

    Parameters
    ----------
    ds    : xarray Dataset
    t_idx : time index
    """
    z_mid = ds.sizes['z'] // 2

    u   = ds['u'].isel(time=t_idx, z=z_mid).values
    v   = ds['v'].isel(time=t_idx, z=z_mid).values
    w   = ds['w'].isel(time=t_idx, z=z_mid).values
    mag = np.sqrt(u**2 + v**2 + w**2)

    fig = plt.figure(figsize=(16, 4))
    gs  = gridspec.GridSpec(1, 4, figure=fig)

    fields = [u,        v,        w,        mag     ]
    labels = ['u',      'v',      'w',      '|U|'   ]
    cmaps  = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'viridis']

    for i, (data, label, cmap) in enumerate(zip(fields, labels, cmaps)):
        ax = fig.add_subplot(gs[i])
        im = ax.pcolormesh(ds['x'].values, ds['y'].values, data.T,
                           cmap=cmap, shading='auto')
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{label}  t={ds["time"].values[t_idx]:.2f}')
        ax.set_aspect('equal')

    plt.tight_layout()