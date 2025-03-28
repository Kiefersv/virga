""" Cloud cloud particle opacities, including mixed particles """
# pylint: disable=R0912,R0913,R0914,R0915,R0917,R1702

import os
import numpy as np
import pandas as pd

from .calc_mie import calc_new_mieff

def _calc_optics_mixed(nwave, qc, rg, ndz, radius, rup, dr, wavelengths, qext, qscat,
                       cos_qscat, sig, rmin, verbose=False, directory=None, mixed=False,
                       quick_mix=True, gas_name=None, rhop=None):
    """
    Calculate spectrally-resolved profiles of optical depth, single-scattering
    albedo, and asymmetry parameter.

    Parameters
    ----------
    nwave : int
        Number of wave points
    qc : ndarray
        Condensate mixing ratio
    rg : ndarray
        Geometric mean radius of condensate
    ndz : ndarray
        Column density of particle concentration in layer (#/cm^2)
    radius : ndarray
        Radius bin centers (cm)
    dr : ndarray
        Width of radius bins (cm)
    qscat : ndarray
        Scattering efficiency
    qext : ndarray
        Extinction efficiency
    cos_qscat : ndarray
        qscat-weighted <cos (scattering angle)>
    sig : float
        Width of the log normal particle distribution
    rmin : float
        minimum radius of Mie grid
    verbose: bool
        print out warnings or not
    mixed : bool, optional
        If true, cloud particles are assumed to be able to mix together.
    quick_mix : bool, optional
        If true, the optical properties of mixed materials are calculated as the weighted
        sum of the individual materials. This is not an accurate solution and should only
        be used for first analysis.


    Returns
    -------
    opd : ndarray
        extinction optical depth due to all condensates in layer
    w0 : ndarray
        single scattering albedo
    g0 : ndarray
        asymmetry parameter = Q_scat wtd avg of <cos theta>
    opd_gas : ndarray
        cumulative (from top) opd by condensing vapor as geometric
        conservative scatterers
    """

    # get length of input arrays
    nz = qc.shape[0]  # atmosphere grid
    ngas = qc.shape[1]  # number of gas phase species
    nrad = len(radius)  # number of Mie grid radii

    # prepare working and output arrays
    opd_layer = np.zeros((nz, ngas))
    scat_gas = np.zeros((nz, nwave, ngas))
    ext_gas = np.zeros((nz, nwave, ngas))
    cqs_gas = np.zeros((nz, nwave, ngas))
    opd = np.zeros((nz, nwave))
    opd_gas = np.zeros((nz, ngas))
    w0 = np.zeros((nz, nwave))
    g0 = np.zeros((nz, nwave))

    # if mixed, preload refindex, igonre mixed entry
    refidx, qe, qs, cos = (None, None, None, None)
    if mixed and not quick_mix:
        refidx = Refrindex(directory, gas_name[:-1])
        wave_in = wavelengths[:, 0]
        qe, qs, cos = _mixed_mie(gas_name, refidx, qc,
                                 radius, rup, wave_in, rhop)

    # warning message string
    warning = ''
    warning0 = ''

    # loop over all grid points and all gasses
    for iz in range(nz):
        for igas in range(ngas):

            # Only if there are cloud particles, do something
            if ndz[iz, igas] > 0:

                # warining message for small cloud radii
                if np.log10(rg[iz, igas]) < np.log10(rmin) + 0.75 * sig:
                    warning0 = ('Take caution in analyzing results. There have been a '
                                'calculated particle radii off the Mie grid, which has '
                                'a min radius of '+str(rmin)+'cm and distribution of '
                                +str(sig)+'.  The following errors:')
                    warning += (str(rg[iz, igas]) + 'cm for the ' + str(igas) + 'th gas '
                                'at the ' + str(iz) + 'th grid point; ')

                # get cloud particle radius adjusted for size distribution
                r2 = rg[iz, igas]**2 * np.exp(2 * np.log(sig)**2)
                # get gemoetric cloud particle absorption
                opd_layer[iz, igas] = 2. * np.pi * r2 * ndz[iz, igas]

                #  Calculate normalization factor (forces lognormal sum = 1.0)
                norm = 0.
                for irad in range(nrad):
                    rr = radius[irad]
                    arg1 = dr[irad] / (np.sqrt(2. * np.pi) * rr * np.log(sig))
                    arg2 = -np.log(rr / rg[iz, igas]) ** 2 / (2 * np.log(sig) ** 2)
                    if arg2 > -300:  # prevent underflow, set to 0 otherwise
                        norm = norm + arg1 * np.exp(arg2)

                # normalization
                if norm > 0:
                    norm = ndz[iz, igas] / norm  # number density distribution
                else:
                    norm = 0

                # iterate over all Mie radii
                for irad in range(nrad):
                    # get geometric cloud particle opacity
                    rr = radius[irad]
                    arg1 = dr[irad] / (np.sqrt(2. * np.pi) * np.log(sig))
                    arg2 = -np.log(rr / rg[iz, igas]) ** 2 / (2 * np.log(sig) ** 2)
                    if arg2 > -300:  # prevent underflow, set to 0 otherwise
                        pir2ndz = norm * np.pi * rr * arg1 * np.exp(arg2)
                    else:
                        pir2ndz = 0.0

                    # mixed particles need to calculated here, cant be precalculated
                    if mixed and igas == ngas-1:
                        # this is a quick and dirty weighted avarage. Only use this if
                        # computation time is non-negotiable
                        if quick_mix:
                            for iw in range(nwave):
                                wei = qc[:-1]/np.sum(qc[:-1]) * pir2ndz
                                scat_gas[iz, iw, -1] += np.sum(wei * qscat[iw, irad, :])
                                ext_gas[iz, iw, -1] += np.sum(wei * qext[iw, irad, :])
                                cqs_gas[iz, iw, -1] += np.sum(wei * cos_qscat[iw, irad, :])
                        else:
                            for iw in range(nwave):
                                scat_gas[iz, iw, -1] += qs[iz, iw, irad] * pir2ndz
                                ext_gas[iz, iw, -1] += qe[iz, iw, irad] * pir2ndz
                                cqs_gas[iz, iw, -1] += cos[iz, iw, irad] * pir2ndz

                    # non mixed particles are multiplied with precalculated mie coeffs
                    else:
                        for iw in range(nwave):
                            scat_gas[iz, iw, igas] += qscat[iw, irad, igas] * pir2ndz
                            ext_gas[iz, iw, igas] += qext[iw, irad, igas] * pir2ndz
                            cqs_gas[iz, iw, igas] += cos_qscat[iw, irad, igas] * pir2ndz

    # perform sublayering which creates a smoother transition at the bottom of the
    # atmosphere to prevent a sudden increase in optical depth which for many radiative
    # transfer codes is difficult to handle
    for igas in range(ngas):
        # find bottom of cloud
        ibot = 0
        for iz in range(nz - 1, -1, -1):
            if np.sum(ext_gas[iz, :, igas]) > 0:
                ibot = iz
                break
            if iz == 0:
                ibot = 0

        # if it is to close to the bottom layer, do not apply the sublayering
        if ibot >= nz - 2:
            print("Not doing sublayer as cloud deck at the bottom of pressure grid")

        # if it is far enough away, add the values
        else:
            opd_layer[ibot + 1, igas] = opd_layer[ibot, igas] * 0.1
            scat_gas[ibot + 1, :, igas] = scat_gas[ibot, :, igas] * 0.1
            ext_gas[ibot + 1, :, igas] = ext_gas[ibot, :, igas] * 0.1
            cqs_gas[ibot + 1, :, igas] = cqs_gas[ibot, :, igas] * 0.1
            opd_layer[ibot + 2, igas] = opd_layer[ibot, igas] * 0.05
            scat_gas[ibot + 2, :, igas] = scat_gas[ibot, :, igas] * 0.05
            ext_gas[ibot + 2, :, igas] = ext_gas[ibot, :, igas] * 0.05
            cqs_gas[ibot + 2, :, igas] = cqs_gas[ibot, :, igas] * 0.05

    # Sum over gases and compute spectral optical depth profile etc
    for iz in range(nz):
        for iwave in range(nwave):
            opd_scat = 0.
            opd_ext = 0.
            cos_qs = 0.
            for igas in range(ngas):
                # for mixed particle, only add the last entry
                if mixed and not igas == ngas-1:
                    continue
                # sum over all gases that need summing
                opd_scat += scat_gas[iz, iwave, igas]
                opd_ext += ext_gas[iz, iwave, igas]
                cos_qs += cqs_gas[iz, iwave, igas]

            # if there are opacities set them
            if opd_scat > 0.:
                opd[iz, iwave] = opd_ext
                w0[iz, iwave] = opd_scat / opd_ext
                g0[iz, iwave] = cos_qs / opd_scat

    # cumulative optical depths for conservative geometric scatterers
    for igas in range(ngas):
        # first value
        opd_gas[0, igas] = opd_layer[0, igas]
        # iterate over the rest
        for iz in range(1, nz):
            opd_gas[iz, igas] = opd_gas[iz - 1, igas] + opd_layer[iz, igas]

    # print warinings if some were encountered
    if warning != '' and verbose:
        print(warning0 + warning + ' Turn off warnings by setting verbose=False.')

    # return values
    return opd, w0, g0, opd_gas


def _mixed_mie(gas_name, refidx, qc, radius, rup, wavelength, rhop):
    """
    Calculate mie coefficients for mixed particles

    Parameters
    ----------
    gas_name : list
        name of all gas species
    refidx : Refrindex
        refractive index database
    qc : ndarray
        mass mixing ratios of cloud material
    radius : ndarray
        radius to calculate mie coefficients
    rup : ndarray
        spacing of radius array
    wavelength : ndarray
        wavelength to calculate mie coefficients
    rhop : ndarray
        densities of cloud particle material

    Returns
    -------
    (qsca, qext, cos_qsca) : tuple
        scattering, extinction and asymmetry coefficients
    """

    # get sizes of input
    nz = qc.shape[0]
    nw = wavelength.shape[0]
    nr = radius.shape[0]

    # set up working arrays
    qext = np.zeros((nz, nw, nr))
    qsca = np.zeros((nz, nw, nr))
    cos_qsca = np.zeros((nz, nw, nr))

    for z in range(nz):

        # check if there is no material, if so, set all 0
        if np.sum(qc[z, :-1]) <= 0:
            qext[z], qsca[z], cos_qsca[z] = (0, 0, 0)
            continue

        # dielectric constant of the mixture
        e_eff = complex(0, 0)
        # read in data that was not previously loded
        for i, gas in enumerate(gas_name[:-1]):
            n, k = refidx.nk(gas, wavelength)
            e_cur = np.asarray([complex(n[i], k[i])**2 for i in range(len(wavelength))])
            volfrac = qc[z, i] / rhop[i] / np.sum(qc[z, :-1]/rhop)
            e_eff += volfrac * e_cur**(1./3.)
        e_eff **= 3  # finish LLL averaging
        m_eff = e_eff**(1./2.)  # convert back to refractive index

        # get Mie coeficients
        qext[z], qsca[z], cos_qsca[z] = calc_new_mieff(wavelength, m_eff.real,
                                                       m_eff.imag, radius, rup)

    # return mie values
    return qext, qsca, cos_qsca


class Refrindex():
    """ Refractive index database handler """
    def __init__(self, directory, gas_names=None):
        """
        Initialise the refracitive index class.

        Parameters
        ----------
        directory : str
            path to refractive index data
        gas_names : list of str
            gas names to be added
        """
        self.refidx_db = {}  # refractive index database
        self.dict = directory  # dictionary of refractive index files
        self.load(gas_names)  # load all initial gases

    def load(self, gas_names):
        """
        Load the gas_names

        Parameters
        ----------
        gas_names : list, str
            gas names to be added
        """

        # if gas_names is a string, make it a list
        if isinstance(gas_names, str):
            gas_names = [gas_names]

        # itterate over all gases in gas_names
        for gas in gas_names:

            # if they are already in the database, skip them
            if gas in self.refidx_db:
                continue

            # load file
            filename = os.path.join(self.dict, gas + ".refrind")

            # first try to load with numpy
            try:
                # put skiprows=1 in loadtxt to skip first line
                _, wave_in, nn, kk = np.loadtxt(open(filename, 'rt').readlines(),
                    unpack=True, usecols=[0, 1, 2, 3])
            # if that does not work, use pandas
            except:
                df = pd.read_csv(filename)
                wave_in = df['micron'].values
                nn = df['real'].values
                kk = df['imaginary'].values

            # store values
            self.refidx_db[gas] = np.vstack((wave_in, nn, kk))

    def nk(self, gas, wave):
        """
        Read out nk data from database

        Parameters
        ----------
        gas : str
            gas name
        wave : float
            wavelength

        Returns
        -------
        (n, k) : tuple
            real and imaginary part of the refractive index
        """

        # if gas is not in the database, try to load it
        if gas not in self.refidx_db:
            self.load(gas)

        # interpolate n and k
        data = self.refidx_db[gas]
        n = np.interp(wave, data[0][::-1], data[1][::-1])
        k = np.interp(wave, data[0][::-1], data[2][::-1])

        # return refractive index values
        return n, k
