"""
Functions to generate a cloud structure by iterating over fsed calculations. For any
questions, pleas contact kiefersv.mail@gmail.com.
"""
# pylint: disable=R0912,R0913,R0914,R0915,R0917,C0302

import os
import csv
from time import time

import numpy as np
import xarray as xr
from scipy.optimize import root
from scipy.special import gamma

from . import gas_properties
from .mixed_clouds import _eddysed_mixed
from .justdoit import get_mie

# pysical constants
RGAS = 8.3143e7  # Gas constant [erg/mol/K]
AVOG = 6.02e23  # avogadro constant [1/mol]
KB = RGAS / AVOG  # boltzmann constant [erg/K]
PF = 1000000  # Reference pressure [dyn / cm2]


# =======================================================================================
#  Main Computational functions
# =======================================================================================
def compute_iterator(atmo, directory=None, as_dict=None, og_solver=None,
                     direct_tol=None, refine_TP=None, og_vfall=True, analytical_rg=None,
                     do_virtual=True, size_distribution='lognormal', mixed=False,
                     iter_max=30, rel_acc=1e-3, nuc_eff=1):
    """
    Top level program to run eddysed. Requires running `Atmosphere` class
    before running this.

    Parameters
    ----------
    atmo : class
        `Atmosphere` class
    directory : str, optional
        Directory string that describes where refrind files are
    as_dict : bool, optional
        Only used in regular VIRGA compute.
    og_solver : bool, optional
        Only used in regular VIRGA compute.
    direct_tol : float , optional
        Only used in regular VIRGA compute.
    refine_TP : bool, optional
        Only used in regular VIRGA compute.
    og_vfall : bool, optional
        Option to use original A&M or new Khan-Richardson method for finding vfall.
    analytical_rg : bool, optional
        Only used in regular VIRGA compute.
    do_virtual : bool
        If the user adds an upper bound pressure that is too low. There are cases where a
        cloud wants to form off the grid towards higher pressures. This enables that.
    size_distribution : str, optional
        Define the size distribution of the cloud particles. Currently supported:
        "lognormal" (default), "exponential", "gamma", and "monodisperse"
    mixed : bool, optional
        If true, cloud particles are assumed to be able to mix together.
    iter_max : int, optional
        Maximum number of iterations to converge on static fsed.
    rel_acc : float, optional
        Maximum relative change in mmr of cloud particles to be considered converged.
    nuc_eff : float, optional
        Nucleation efficiency. This parameter directly reduces the number of cloud
        particles that are created. It should only be used to analyse the effect of
        impressions within nucleation rate calculations and not for actual cloud profile
        calculations.

    Returns
    -------
    dict
        Dictionary output that contains full output. This includes:
        - 'condensate_mmr': qc, mass mixing ratio of solid materials
        - 'cond_plus_gas_mmr': qt, mass mixing ratio of solid and gas
        - 'scalar_inputs': Additional scaler values:
            - 'mmw': mean molecular weight in amu
        - 'pressure': atmospheric pressure in bar
        - 'temperature': atmospheric temperature in K
        - 'layer_thickness': altitude size of each layer in cm
        - 'kz': diffusion coefficient in cm2 / s
    """

    # give warning if unused parameters are given
    w_str = ''
    for key, name in [(as_dict, 'as_dict'), (og_solver, 'og_solver'),
                      (refine_TP, 'refine_TP'), (direct_tol, 'direct_tol'),
                      (analytical_rg, 'analytical_rg')]:
        if not isinstance(key, type(None)):
            w_str += name + '; '
    # print the warning if there are unused variables
    if len(w_str) > 0:
        m_str = '[WARN] The following arguments are unused in compute_iterator: '
        print(m_str + w_str)

    # check if iter_max is correct
    if iter_max < 11:
        print('[WARN] iter_max must be at least 11 (was changed to 11 now).')
        iter_max = 11

    # lengths of arrayes
    nz = atmo.p_level.shape[0]
    ng = len(atmo.condensibles)
    # mixed cloud particles will add an entry later
    if mixed:
        ng += 1

    # set up working variables
    memory_virga = []  # remember virga output
    memory_new = []  # remember newly calculated values
    converged = False  # convergence flag
    qc_old = None  # save old mmr to check for convergence
    fsed_w = np.zeros((nz, ng))  # new fsed for next iteration
    error_max = -1  # additional info on convergence state

    # loop
    i = 0
    start_time = time()
    while i < iter_max:

        # ==== Original VIRGA calculation ===============================================
        # compute cloud structure with given fsed_in, here a lightweight version is used
        # which does not compute opacities just yet.
        all_out = _lightweight_compute(atmo, directory, og_vfall, do_virtual,
                                    size_distribution, mixed)

        # remember output
        memory_virga.append(all_out)

        # ==== Adjustment for additional physics ========================================
        # values needed
        qc = all_out['condensate_mmr']
        qt = all_out['cond_plus_gas_mmr']
        mmw =  all_out['scalar_inputs']['mmw']
        pres = all_out['pressure']*1e6
        temp = all_out['temperature']
        dz = all_out['layer_thickness']
        kzz = all_out['kz']

        # Post-calculate cloud particle properties from virga mmrs
        fsed, ncl_tot, rg, qcl_tot = _top_down_property_calculator(
            atmo.condensibles, qc, qt, temp, pres, atmo.dz_layer, kzz, mmw, atmo.g,
            atmo.sig, size_distribution, mixed, nuc_eff
        )

        # interpolate fsed from mid to edge values for next run
        fsed_w[0] = fsed[0]  # lower limit
        fsed_w[-1] = fsed[-1]  # upper limit
        # inbetween values
        dlnp = np.log(pres[:-1] / pres[1:])  # log pressure thickness
        dfdlnp = (fsed[1:] - fsed[:-1]) / dlnp[:, np.newaxis]  # fsed gradient
        fsed_w[1:-1] = fsed[:-1] + np.log(pres[:-1] / atmo.p_level[1:-1])[:, np.newaxis]*dfdlnp

        # Use dampening if finding a static solution proves difficult
        fsed_damp = fsed_w  # below 10 iterations do not use damping
        # after 10 iterations, use fsed dampening
        if i >= 10:
            mask = fsed_damp != 0
            fsed_damp[mask] = 10**((np.log10(fsed_w[mask]) + np.log10(atmo.fsed[mask]))/2)
        # assign value for next run
        atmo.fsed = fsed_damp

        # save new values
        memory_new.append(all_out)
        memory_new[-1]['fsed'] = fsed
        memory_new[-1]['mean_particle_r'] = rg*1e4
        memory_new[-1]['particle_density'] = ncl_tot
        memory_new[-1]['condensate_mmr'] = qcl_tot

        # ==== Check for convergance and prepare next loop ==============================
        # UI information on progress
        e_str = ''
        if i > 1:
            e_str = ' with max error {:0.3e}'.format(error_max) + '           '
        print('\r[INFO] Iterator loop number ' +str(i) + e_str , end='')

        # increment the loop number
        i += 1

        # check if there is already an old mmr to compare, otherwise start over
        if isinstance(qc_old, type(None)):
            qc_old = qc  # remember current run
            continue

        # Compare to previous value and see if deviations are within tolerance
        error = np.abs((qc[qc > 0] - qc_old[qc > 0]) / qc[qc > 0])
        error_max = np.max(error)  # remember the maximum error for print

        # if error is below accuracy, finish loop
        if (error < rel_acc).all():
            converged = True
            break  # stop iteration
        qc_old = qc  # remember current run

    # calculate evaluation time
    end_time = time()
    time_str = '(' + str(round(end_time - start_time, 1)) + ' s).'

    # in case of convergence, give info
    if converged:
        i_str = '\r[INFO] Iterator successful after ' + str(i) + ' iterations '
        print(i_str + time_str)

    # in case of non-convergence, add the last 10 iterations averaged
    if not converged:
        i_str = '\r[WARN] Cloud profile has not converged, average over last 10 results.'
        print(i_str + time_str)
        # start by assigning values from the 10th last iteration
        avg = memory_new[-10].copy()

        # iterate over all entries
        for key in avg:
            if not isinstance(avg[key], str) and not isinstance(avg[key], dict):
                for i in range(-9, 0, 1):
                    avg[key] = avg[key] + memory_new[i][key]
                avg[key] = avg[key] / 10

        memory_new.append(avg)

    return memory_virga, memory_new


def _lightweight_compute(atmo, directory=None, og_vfall=True, do_virtual=True,
                         size_distribution='lognormal', mixed=False):
    """
    Lightweight version of the top level program to run eddysed.

    Parameters
    ----------
    atmo : class
        `Atmosphere` class
    directory : str, optional
        Directory string that describes where refrind files are
    og_vfall : bool, optional
        Option to use original A&M or new Khan-Richardson method for finding vfall.
    do_virtual : bool
        If the user adds an upper bound pressure that is too low. There are cases where a
        cloud wants to form off the grid towards higher pressures. This enables that.
    size_distribution : str, optional
        Define the size distribution of the cloud particles. Currently supported:
        "lognormal" (default), "exponential", "gamma", and "monodisperse"
    mixed : bool, optional
        If true, cloud particles are assumed to be able to mix together.

    Returns
    -------
    dict
        Dictionary output that contains full output. This includes:
        - 'condensate_mmr': qc, mass mixing ratio of solid materials
        - 'cond_plus_gas_mmr': qt, mass mixing ratio of solid and gas
        - 'scalar_inputs': Additional scaler values:
            - 'mmw': mean molecular weight in amu
        - 'pressure': atmospheric pressure in bar
        - 'temperature': atmospheric temperature in K
        - 'layer_thickness': altitude size of each layer in cm
        - 'kz': diffusion coefficient in cm2 / s
    """

    # if mixed species are already added here remove them, they should only be added later
    if 'mixed' in atmo.condensibles:
        atmo.condensibles.remove('mixed')

    # prepare working arrays
    ngas = len(atmo.condensibles)  # number of gas phase species
    gas_mw = np.zeros(ngas)  # weight of gas species in amu
    gas_mmr = np.zeros(ngas)  # mass mixing ratio
    rho_p = np.zeros(ngas)  # density of the solids in g / cm3

    # get gas phase properties (molecular weight, mixing ratio, and density)
    for i, igas in zip(range(ngas),atmo.condensibles) :
        run_gas = getattr(gas_properties, igas)
        gas_mw[i], gas_mmr[i], rho_p[i] = run_gas(
            atmo.mmw, mh=atmo.mh, gas_mmr=atmo.gas_mmr[igas]
        )

    # get radius from mie grid
    _, _, _, _, radius, _ = get_mie(atmo.condensibles[0], directory)
    nradii = len(radius)
    rmin = np.min(radius)

    # here atmo.param describes the parameterization used for the variable fsed methodology
    if atmo.param == 'exp':
        # scale-height for fsed taken at Teff (default: temp at 1bar)
        scale_h = atmo.r_atmos * atmo.Teff / atmo.g
        # the formalism of this is detailed in Rooney et al. 2021
        atmo.b = 6 * atmo.b * scale_h  # using constant scale-height in fsed
        fsed_in = atmo.fsed - atmo.eps
    elif atmo.param in ['const', 'array']:
        fsed_in = atmo.fsed
    else:
        raise ValueError('Fsed parametrisation "' + atmo.param + '" not supported')

    # call the eddysed function to calculate cloud qc, and qt
    qc, qt, _, _, _, _, _, _ = _eddysed_mixed(atmo.t_level,
        atmo.p_level, atmo.t_layer, atmo.p_layer, atmo.condensibles, gas_mw, gas_mmr,
        rho_p , atmo.mmw, atmo.g, atmo.kz, atmo.mixl, fsed_in, atmo.b, atmo.eps,
        atmo.scale_h, atmo.z_top, atmo.z_alpha, min(atmo.z), atmo.param, atmo.mh,
        atmo.sig, rmin, nradii, atmo.d_molecule, atmo.eps_k, atmo.c_p_factor,
        og_vfall, supsat=atmo.supsat,verbose=atmo.verbose,do_virtual=do_virtual,
        size_distribution=size_distribution, mixed=mixed
    )

    # store all outputs needed in dictionary
    output = {
        "pressure":atmo.p_layer/1e6,
        "pressure_unit":'bar',
        "temperature":atmo.t_layer,
        "temperature_unit":'kelvin',
        "condensate_mmr":qc,
        "cond_plus_gas_mmr":qt,
        "scalar_inputs": {'mmw':atmo.mmw,},
        "layer_thickness":atmo.dz_layer,
        'kz':atmo.kz,
        'kz_unit':'cm^2/s',
    }

    # And you are done
    return output

# =======================================================================================
#  Post processing functions
# =======================================================================================
def _top_down_property_calculator(species, qc, qt, temp, pres, dz, kzz, mmw, gravity,
                                  sig, size_distribution, mixed, nuc_eff=1):
    """
    This function calculates the cloud particle properties top down after
    virga calculated qc bottom up.

    Parameters
    ----------
    species : List
        All species considered in the cloud profile calcualtion.
    qc : ndarray
        Mass mixing ratios of the cloud particle solids.
    qt : ndarray
        Total mass mixing ratio of each cloud particle material
    temp : ndarray
        Temperature structure
    pres : ndarray
        Pressure structure
    dz : ndarray
        Altitude structure
    kzz : ndarray
        Diffusion coefficient
    mmw : float
        mean molecular weight
    gravity : float
        gravity of the planet
    sig : float
        Size distribution factor (meaning depends on the distribution)
    size_distribution : str
        Define the size distribution of the cloud particles. Currently supported:
        "lognormal" (default), "exponential", "gamma", and "monodisperse"
    mixed : bool
        If true, cloud particles are assumed to be able to mix together.
    nuc_eff : float
        Nucleation efficiency. This parameter directly reduces the number of cloud
        particles that are created. It should only be used to analyse the effect of
        impressions within nucleation rate calculations and not for actual cloud profile
        calculations.

    Returns
    -------
    fsed, ncl_tot, rg, qcl_tot
        settling parameter fsed, total cloud particle number density ncl_tot,
        cloud particle mean geometric radius rg, and mass mixing ratio of the
        solid cloud particles
    """

    # ==== Preparation ==================================================================
    # get shape of input
    nz = qc.shape[0]
    ng = qc.shape[1]

    # get working arrays
    ncl_in = np.zeros((nz, ng))  # infalling cloud particle number density
    ncl_tot = np.zeros((nz, ng))  # total cloud particle number density
    qcl_in = np.zeros((nz, ng))  # infalling cloud mass mixing ratio
    qcl_tot = np.zeros((nz, ng))  # total cloud mass mixing ratio
    rg = np.zeros((nz, ng))  # geometric cloud particle radius
    fsed = np.zeros((nz, ng))  # settling parameter

    # physical parameters
    rho_atmo = mmw * pres / temp / RGAS  # atmospheric density

    # calculate size distribution depending factors
    if size_distribution == 'lognormal':
        fac_1 = np.exp(np.log(sig) ** 2 / 2)
        fac_3 = np.exp(9 * np.log(sig) ** 2 / 2)
    elif size_distribution == 'exponential':
        fac_1 = gamma(1)
        fac_3 = gamma(3)
    elif size_distribution == 'gamma':
        fac_1 = gamma(1 + sig) / gamma(sig)
        fac_3 = gamma(3 + sig) / gamma(sig)
    elif size_distribution == 'monodisperse':
        fac_1 = 1
        fac_3 = 1
    else:
        raise ValueError(size_distribution + ' distribution not known.')

    # read in solid densities of materials
    rho_p = np.zeros(ng)
    for i, igas in enumerate(species):
        # if mixed they need to be avaraged from previous read ins
        if igas == 'mixed':
            mask = (qc[:, -1] != 0) # check where qc is not 0
            rho_p[-1] = np.sum(qc[mask, :-1]) / np.sum(qc[mask, :-1] / rho_p[:-1])
        # single material are read in from gas_properties functions
        else:
            run_gas = getattr(gas_properties, igas)
            _, _, rho_p[i] = run_gas(1)  # only the density is needed

    # fsed prefactor calculation
    h = RGAS * temp / gravity / mmw  # scale height
    ct = np.sqrt(2 * RGAS * temp / mmw)  # sound speed
    vsed_pre = np.sqrt(np.pi) * gravity / 2 / rho_atmo / ct  # settling speed pre factor
    fsed_pre = vsed_pre * fac_1 * h / kzz  # added later are: * rho_p[ig] * rg

    # ==== Properties of each layer =====================================================
    for iz in range(nz):  # goes from top to bottom of the atmosphere

        # layer specific prefactor for gravitational settling of number densities and mmr
        fac_ncl = np.sqrt(temp[iz]) / pres[iz] # * dz[iz]
        fac_qcl = 1 / np.sqrt(temp[iz]) # * dz[iz]

        # Loop over all species
        for ig, igas in enumerate(species):

            # ==== Cloud particle number density + mmr ==================================
            # get number of nucleation seeds, start with infall from cell above
            if qc[iz, ig] > 0:
                ncl_tot[iz, ig] = ncl_in[iz, ig] / fac_ncl

            # mixed cloud particles are made from all nucleation seeds
            if igas == 'mixed':
                for im, gas_name in enumerate(species[:-1]):
                    # if there is no material, there will be no seeds
                    if qc[iz, im] <= 0:
                        continue

                    # else calculate the nucleation rate
                    ncl_tot[iz, ig] += _get_ccn_nucleation(
                        gas_name, qt[iz, im], temp[iz], pres[iz], mmw, ncl_tot[iz, ig]
                    ) * nuc_eff

                    # mass mixing ratio
                    qcl_tot[iz, ig] += qc[iz, im] + qcl_in[iz, im] / fac_qcl

            # homogenous particles are made from only their own seeds
            else:
                # if there is no material, there will be no seeds
                if qc[iz, ig] <= 0:
                    continue

                # else calculate the nucleation rate
                ncl_tot[iz, ig] += _get_ccn_nucleation(
                    igas, qt[iz, ig], temp[iz], pres[iz], mmw, ncl_tot[iz, ig]
                ) * nuc_eff

                # mass mixing ratio
                qcl_tot[iz, ig] = qc[iz, ig] + qcl_in[iz, ig] / fac_qcl

            # advect the number density and mmr to the next cell
            if iz < nz-1:  # bottom cell does not need to pass value
                ncl_in[iz + 1, ig] = ncl_tot[iz, ig] * fac_ncl
                qcl_in[iz + 1, ig] = qcl_tot[iz, ig] * fac_qcl

            # ==== Fsed parameter =======================================================
            # now the material can condense onto the particles
            if ncl_tot[iz, ig] > 1e-50:
                # get the mean radius
                rg[iz, ig] = (3 * qcl_tot[iz, ig] * rho_atmo[iz] / 4 / np.pi / rho_p[ig]
                              / ncl_tot[iz, ig] / fac_3)**(1/3)
            # get new fsed parameter
            fsed[iz, ig] = fsed_pre[iz] * rg[iz, ig] * rho_p[ig]

        # if mixed, all cloud particles have the same fsed
        if mixed:
            for im, igas in enumerate(species[:-1]):
                fsed[:, im] = fsed[:, -1]

    # return fsed
    return fsed, ncl_tot, rg, qcl_tot


# =======================================================================================
#  Additional functionalities for the functions above
# =======================================================================================

# =======================================================================================
#  Top level function to calculate nucleation number density
preventer = []  # this variable remembers which warnings were already called
def _get_ccn_nucleation(spec, mmr, temp, pres, mmw, ncl_in,
                        analytic_approximation=False):
    """
    This function balances nucleation rate and growth rate to find how much of the mmr
    material will nucleate and how much condenses.
    Recommended species to add: Cr, ZnS, Na2S, Fe, CH4, Al2O3, CaTiO3, CaAl12O19, SiO2

    Parameters
    ----------
    spec : str
        Name of the species
    mmr : float
        Mass mixing ratio
    temp : float
        temperature
    pres : float
        pressure
    mmw : float
        total mass mixing ratio of spec (qt in eddysed lingo)
    ncl_in : float
        Number density of already present cloud particles.
    analytic_approximation : bool, optional
        If true, a simplified analytic expression is used. This is faster and more
        stable but yields wrong results for mmr close to the saturation value.

    Returns
    -------
    ncl : float
        Cloud particle number density from nucleation.
    """

    # these species are calculated using modified CNT
    if spec in ['SiO', 'Cr', 'KCl', 'NaCl', 'CsCl', 'H2O', 'NH3',
                'H2S', 'CH4']:#, 'MnS']:
        _, ncl = _fracs_elsie(
            spec, mmr, temp, pres, mmw, ncl_in, analytic_approximation
        )

    # these species use the becker-doering method for nucleation
    elif spec in ['TiO2']:
        _, ncl = _non_classical_nucleation(
            spec, mmr, temp, pres, mmw, ncl_in, analytic_approximation
        )

    # some species are known to not nucleate or no data is available
    else:
        if spec not in preventer:
            print('[WARN] No nucleation rate info for "' + spec + '". '
                  'Assumed to be zero.')
            preventer.append(spec)  # this prevents repeated error messages
        ncl = 0

    # and we are done
    return ncl

# =======================================================================================
# General description for non classicle nucleation rate
def _non_classical_nucleation(spec, mmr, temp, pres, mmw, ncl_in=0,
                              analytic_approx=False):
    """
    Here the nucleation rate and the growth rate are calculated. The goal is to find
    the number of cloud particles at which both the nucleation rate J and the growth
    rate R are equal which marks the point after which most material will be used up
    in gorwth before it can nucleate and thus marks an estimate on the number of
    cloud particles 'nc' produced.
      -> Non Classicle Nucleation rate : J = (Sum[1 / (An nu n1 nn)])^(-1)
      -> growth rate: R = Ac nu n1 nc (1 - 1/S)
      => nc^(-1) = Sum[(Ac / An) * (1 / nn)] * (1 - 1/S)

    This equation can be simplified with:
      -> Ac / An = 4 pi rc^2 / 4 pi rn^2 = rc^2 / rn^2
                 = (r1 Nc^(1/3))^2 / (r1 N^(1/3))^2 = (Nc / N)^(2/3)

    The supersaturation is S = p1 / pvap
    The number density nn is derived according to Lee et al. 2013
    The partial pressure is adjusted for the material lost by nucleation
      -> p1_reduced = (n1 - n*nc)/ntot * p = p1 - n*nc*kb*T

    Parameters
    ----------
    spec : str
        Name of the species
    mmr : float
        total mass mixing ratio of spec (qt in eddysed lingo)
    temp : float
        temperature
    pres : float
        pressure
    mmw : float
        mean molecular weight [amu]
    ncl_in : float
        Number density of already present cloud particles.
    analytic_approx : bool, optional
        If true, a simplified analytic expression is used. This is faster and more
        stable but yields wrong results for mmr close to the saturation value.

    Returns
    -------
    sol, ncl : float
        sol is the fraction of mmr that is used up in nucleation and ncl is the number
        density of cloud particles from nucleation.
    """

    # read in data for species
    data = _read_data(spec)  # read from file
    m1 = data['mass'].sel({'N':1}).values  # monomer mass [g]
    n = data['N'].values  # n-mer number (usually incremental from 1, 2, 3, ...)
    n_ccn = data.attrs['n_ccn']  # number of monomers in a single CCN
    gibbs = data['gibbs_energy'].interp(temperature=temp).values  # [erg/K]

    # data of the solid
    data_s = _read_data(spec + '[s]')  # read from file
    gibbs_s = data_s['gibbs_energy'].interp(temperature=temp).values  # [erg/K]
    pvap = PF * np.exp(-(gibbs[0] - gibbs_s) / RGAS / temp)   # vapour pressure [dyn]

    # calculate surface area prefactor, Ac / An
    acdan = (n_ccn / n) ** (2 / 3)

    # get the prefactor for each cluster size nn from Gibbs data
    cl_mass_total = mmr * mmw * pres / RGAS / temp  # total cloud mass [g]
    mol_frac = mmr * mmw / m1 / AVOG  # molar fraction of the monomer
    p1 = mol_frac * pres  # partial pressure of the monomer [dyn]
    n1 = p1 / KB / temp  # monomer number density [1/cm3]
    exp = np.exp(-(gibbs - n * gibbs[0]) / RGAS / temp)  # exponential factor

    # The analytic approximation neglects the depletion from nucleation on the monomer
    # for the calculation of the nucleation and growth rate. USE WITH CAUTION!
    if analytic_approx:
        nn = PF * (p1 / PF) ** n * exp / KB / temp  # N-mer density
        eq_ncl = np.min(np.asarray([1 / np.sum(acdan / nn), n1 / n_ccn]))
        ccn_mass = eq_ncl * n_ccn * m1  # mass of all cluod particle seeds [g]
        eq_sol = min([ccn_mass / cl_mass_total, 1])  # prevent depletion above max
        eq_ncl = min([eq_ncl, n1 / n_ccn])  # max number is everything nucleates
        return eq_sol, eq_ncl

    # function to find nc (in log to optimize minimization)
    def func(lognc):
        """ lognc : ndarray -> log of the cloud particle number density """
        # set up output array
        out = np.zeros_like(lognc)
        # this loop allows lognc to be an array
        for l, lnc in enumerate(lognc):
            # reduced partial pressure, accounting for nucleated material
            ncl_tot = 10 ** lnc  # total cloud particle number density [1/cm3]
            p1r = p1 - n_ccn * ncl_tot * KB * temp  # reduced partial pressure [dyn]
            nn = PF * (p1r / PF) ** n * exp / KB / temp  # N-mer density
            mask = nn != 0  # prevent division by 0
            sfac = 1 - pvap / p1r  # supersaturation factor
            # spot where nucleation and growth are equal
            out[l] = np.abs(1 / np.sum(acdan[mask] / nn[mask]) * sfac - (ncl_tot + ncl_in))
        # return output
        return out

    # starting value is (a tenth of the) maximum value possible
    x0 = n1 / n_ccn
    # execute the minimisation of func to find where growth equals nucleation
    nc_min = root(func, np.log10(x0 * 1e-1))
    ncl = max([10 ** nc_min.x, 0])  # prevent sub zero values

    # and the mass fraction of the nucleation seeds
    ccn_mass = ncl * n_ccn * m1  # mass of all cloud particle seeds
    num_sol = ccn_mass / cl_mass_total  # mass fraction of CCNs created
    sol = min([num_sol, 1])  # prevent division by 0
    return sol, ncl  # mass fraction of CCNs


# =======================================================================================
# Read molecular data from files
def _read_data(spec):
    """
    Read in molecular data for a nucleation species.

    Parameters
    ----------
    spec : str
        name of nucleation species

    Returns
    -------
        ds : xarray
    """

    # define the file path to spec data
    file = os.path.dirname(__file__) + '/nucleation/' + spec.lower() + '.csv'

    # open the files and converting them to numpy arrays
    file_open = open(file)
    data_array = np.array(list(csv.reader(file_open)))
    # split the data into N-mer number, temperature and Gibbs E
    data_n = data_array[3, 1:].astype(int)
    # solid density of material, convert from kg/m3 to g/cm3
    data_rho = data_array[1, 1].astype(float) * 1e-3
    # how many monomer units until the particle is considered solid
    data_nccn = data_array[2, 1].astype(float)
    # sizes, convert from m to cm
    data_size = data_array[4, 1:].astype(float) * 1e2
    # mass, convert from kg to g
    data_mass = data_array[5, 1:].astype(float) * 1e3
    # temperatures already in K
    data_temp = data_array[7:, 0].astype(float)
    # gibbs value, convert from kj / mol to erg / mol
    data_gibs = data_array[7:, 1:].astype(float) * 1e3 * 1e7

    # Assign values to a xarray Dataset
    ds = xr.Dataset(
        data_vars={
            'gibbs_energy': (['temperature', 'N'], data_gibs),
            'size': (['N'], data_size),
            'mass': (['N'], data_mass),
        },
        coords={
            'temperature': data_temp,
            'N': data_n,
        },
        attrs={
            'name': data_array[0][0],
            'density': data_rho,
            'n_ccn': data_nccn,
        },
    )

    # return data array
    return ds


# =======================================================================================
# Solution for specific elements from Lee et al. 2018 (DOI 10.1051/0004-6361/201731977)
def _fracs_elsie(spec, mmr, temp, pres, mmw, ncl_in=0, analytic_approx=False):
    """
    Calculate the number of CCNs created as predicted by Modified Classical Nucleation
    Theory (MCNT).

    Parameters
    ----------
    spec : str
        Name of the species
    mmr : float
        Mass mixing ratio
    temp : float
        temperature
    pres : float
        pressure
    mmw : float
        total mass mixing ratio of spec (qt in eddysed lingo)
    ncl_in : float
        Number density of already present cloud particles.
    analytic_approx : bool, optional
        If true, a simplified analytic expression is used. This is faster and more
        stable but yields wrong results for mmr close to the saturation value.

    Returns
    -------
    sol, ncl : float
        sol is the fraction of mmr that is used up in nucleation and ncl is the number
        density of cloud particles from nucleation.
    """
    # get nucleation rate, vapour pressure, number of monomers per ccn, radius of
    # monomer, and mass of monomer from Lee et al. 2018 (A&A 614, A126).
    j, pvap, n_ccn, r1, m1 = _elsies_vapor_pressures(spec, temp)

    # call balancing function
    result = _balance_with_j(j, temp, pres, mmr, mmw, n_ccn, m1, r1, pvap,
                             ncl_in, analytic_approx)

    # return result
    return result


# =======================================================================================
# Data from Lee et al. 2018 (DOI 10.1051/0004-6361/201731977)
def _elsies_vapor_pressures(spec, temp):
    """
    Data according to Lee et al. 2018 (A&A 614, A126)

    :param spec: Name of species, see below for supported
    :param temp: Temperature in Kelvin
    :return:
        pvap : vapor pressure
        n_ccn : number of monomers to form a ccn
        r1 : radius of the monomer
        m1 : mass of the monomer
    """
    # The number of CCNs needed to form a cloud particle is not discussed
    # in the paper, here 100 is assumed.
    n_ccn = 100

    # ==== data taken from paper
    if spec == 'TiO2':
        pvap = np.exp(35.8027 - 74734.7 / temp)
        r1 = 1.956e-8
        m1 = 79.866 / AVOG
        sig = 589.79 - 0.0708 * temp
    elif spec == 'SiO':
        pvap = np.exp(32.52 - 49520 / temp)
        r1 = 2.001e-8
        m1 = 44.085 / AVOG
        sig = 500  # Gail and Sedlmayr (1986)
    elif spec == 'Cr':
        pvap = 10 ** (7.490 - 20592 / temp) * 1e6
        r1 = 1.421e-8
        m1 = 51.996 / AVOG
        sig = 3330
    elif spec == 'KCl':
        pvap = 10 ** (29.9665 - 23055.3 / temp) * 1e6
        r1 = 2.462e-8
        m1 = 74.551 / AVOG
        sig = 100.3
    elif spec == 'NaCl':
        pvap = 10 ** (29.9665 - 23055.3 / temp) * 1e6
        r1 = 2.205e-8
        m1 = 58.443 / AVOG
        sig = 113.3
    elif spec == 'CsCl':
        pvap = np.exp(29.9665 - 23055.3 / temp)
        r1 = 2.557e-8
        m1 = 168.359 / AVOG
        sig = 100.0
    elif spec == 'H2O':
        temp_c = temp - 273.16
        if temp_c < 0:
            pvap = 6111.5 * np.exp((23.036 * temp_c - temp_c ** 2 / 333.7) / (temp_c + 279.82))
        else:
            pvap = 6112.1 * np.exp((18.729 * temp_c - temp_c ** 2 / 227.3) / (temp_c + 257.87))
        r1 = 1.973e-8
        m1 = 18.015 / AVOG
        sig = 109
    elif spec == 'NH3':
        pvap = np.exp(10.53 - 2161 / temp - 86596 / temp ** 2) * 1e6
        r1 = 1.980e-8
        m1 = 17.031 / AVOG
        sig = 23.4
    elif spec == 'MnS': # MnS is not from the paper
        pvap = 10.0**(11.5315-23810./temp)*1e6  # from Virga
        r1 = 2e-8  # just a guess, find data
        m1 = 87 / AVOG
        sig = 2326.0  # from mini-cloud
    elif spec == 'H2S':
        if temp < 212.8:
            pvap = 10 ** (4.43681 - 829.439 / (temp - 25.412)) * 1e6
        else:
            pvap = 10 ** (4.52887 - 958.587 / (temp - 0.539)) * 1e6
        r1 = 2.293e-8
        m1 = 34.081 / AVOG
        sig = 58.1
    elif spec == 'CH4':
        pvap = 10 ** (3.9895 - 443.028 / (temp - 0.49)) * 1e6
        r1 = 2.383e-8
        m1 = 16.043 / AVOG
        sig = 14.0
    else:
        raise ValueError('Species not found')

    # ==== calcualte nucleation rate
    if spec == 'SiO':
        # SiO has a unique formula
        def j(n1):
            p1 = n1 * KB * temp
            logs = np.log(p1 / pvap)
            return n1 ** 2 * np.exp(1.33 - 4.40e12 / temp ** 3 / logs ** 2)
    else:
        # modified classical nucleation theory from Sindel et al. 2022
        def j(n1):
            # general parameters
            p1 = n1 * KB * temp  # partial pressure
            logs = np.log(p1 / pvap)  # log of saturation
            ti = 4 * np.pi * r1 ** 2 * sig / KB / temp  # theta_inf
            ni = (2 / 3 * ti / logs) ** 3  # N_star_inf
            ni[ni < 1] = 1

            # critical cluster size parameters
            ns = ni / 8 + 1
            rs = ns ** (1 / 3) * r1  # radius of critical cluster
            dgrt = ti * (ns - 1) ** (2 / 3)  # dGibbs/RT
            # number density of critical cluster size
            fs = PF / KB / temp * (p1 / PF) ** ns * np.exp(-dgrt)

            # mcnt factors
            zfac = (ti / (9 * np.pi * (ns - 1) ** (4 / 3))) ** (1 / 2)  # zeldovich factor
            vrel = np.sqrt(KB * temp / 2 / np.pi / m1)  # relative velocity
            tau = 4 * np.pi * rs ** 2 * vrel * fs  # gowth rate

            # nucleation rate
            jrate = n1 * tau * zfac * np.exp((ns - 1) * logs - dgrt)

            # exception handling
            jrate[logs < 0] = 0

            return jrate

    # return results
    return j, pvap, n_ccn, r1, m1


# =======================================================================================
# Wrapper function for a given nucleation rate
def _balance_with_j(j, temp, pres, mmr, mmw, n_ccn, m1, r1, pvap, ncl_in=0,
                    analytic_approx=True):
    """
    Here the nucleation rate and the growth rate are balanced. The goal is to find
    the number of cloud particles at which both the nucleation rate J and the growth
    rate R are equal which marks the point after which most material will be used up
    in gorwth before it can nucleate and thus marks an estimate on the number of
    cloud particles 'nc' produced. This function takes the nucleation rate as input.
      -> Nucleation rate : J(n1)
      -> growth rate: R = Ac nu n1 nc (1 - 1/S)
      => Ac nu n1 nc (1 - 1/S) - J = 0

    Parameters
    ----------
    j : function
        Nucleation function should take 1 argument which is the monomer number density.
    temp : float
        temperature [K]
    pres : float
        pressure [dyn]
    mmr : float
        total mass mixing ratio of spec (qt in eddysed lingo)
    mmw : float
        mean molecular weight
    n_ccn : int
        Number of monomers per CCN
    m1 : float
        mass of the monomer [g]
    r1 : float
        radius of the monomer [cm]
    ncl_in : float
        Number density of already present cloud particles. [1/cm3]
    analytic_approx : bool, optional
        If true, a simplified analytic expression is used. This is faster and more
        stable but yields wrong results for mmr close to the saturation value.
    """

    # calculate growth rate properties
    cl_mass_total = mmr * mmw * pres / RGAS / temp  # total cloud mass [g]
    ac = 4 * np.pi * r1 ** 2 * n_ccn ** (2 / 3)  # cloud particle surface area [cm2]
    nu = np.sqrt(KB * temp / 2 / np.pi / m1)  # relative velocity [cm/s]
    mol_frac = mmr * mmw / m1 / AVOG  # molar fraction of the monomer
    n1 = mol_frac * pres / KB / temp  # monomer number density [1/cm3]

    # The analytic approximation neglects the depletion from nucleation on the monomer
    # for the calculation of the nucleation and growth rate.
    if analytic_approx:
        eq_ncl = j(np.asarray([n1])) / ac * nu * n1
        ccn_mass = eq_ncl * n_ccn * m1  # mass of all cluod particle seeds
        eq_sol = min([ccn_mass / cl_mass_total, 1]) # prevent depletion above max
        eq_ncl = min([eq_ncl, n1 / n_ccn])  # max number is everything nucleates
        return eq_sol, eq_ncl  # mass fraction of CCNs

    # funciton to find nc, nc is in log to optimize minimization
    def minimisation_function(lognc):
        """ lognc : ndarray -> log of the cloud particle number density """
        # reduced monomer number density, accounting for nucleated material
        nc = 10 ** lognc  # cloud particle number density [1/cm3]
        n1r = n1 - nc * n_ccn  # reduced cloud particle number density [1cm3]
        n1r[n1r < 0] = 1e-200  # prevent negative number densities
        p1r = n1r * KB * temp  # reduced partial pressure [dyn]
        sfac = 1 - pvap / p1r  # supersaturation factor
        # spot where nuclation and growth are equal
        jnuc = j(n1r)  # nucleation rate from function
        growth_rate = ac * nu * n1r * (nc + ncl_in) * sfac  # growth rate
        val = growth_rate - jnuc
        # return difference (should be minimized)
        return val

    def poor_mans_minimization(func, xmax, accuracy=int(1e5)):
        """
        A brute force minimisation function optimised for the shape of the
        minimisation_function. It first searches for a local minimia away from xmax.
        If nan was found, it allows to select xmax as minima.

        Parameters
        ----------
        func : function
            the function to minimize
        xmax : float
            maximum value for minimisation
        accuracy : int
            number of points evaluated

        Returns
        -------
        x_min : float
            minial value
        """
        # define log part and linear part of search grid
        xlog = xmax - np.logspace(np.log10(xmax)-5, np.log10(xmax / 2), accuracy // 2)
        xlin = np.linspace(xmax / 2, 1e-50, accuracy // 2)
        x = np.append(xlog, xlin)

        # first search away from maxima
        x_first = x[x < xmax * 0.99]  # exclude the top values, causes problems
        vals = np.abs(func(np.log10(x_first)))  # evaluate all values
        nan_mask = ~np.isnan(vals)  # homemade np.isnotnan
        nonnan_vals = vals[nan_mask]  # all non-nan vlaues
        min_idx = np.min(np.where(nonnan_vals == nonnan_vals.min()))
        # if minima is not close to the boundary, return
        if min_idx > 2:
            return x_first[nan_mask][min_idx]

        # if minima is close to the boundary, evaluate the second part
        x_second = x[x > xmax * 0.9]
        vals = np.abs(func(np.log10(x_second)))
        nan_mask = ~np.isnan(vals)  # homemade np.isnotnan
        nonnan_vals = vals[nan_mask]
        min_idx = np.min(np.where(nonnan_vals == nonnan_vals.min()))
        return x_second[nan_mask][min_idx]

    # maximum number of cloud particle possible to form
    x0 = n1 / n_ccn

    # check if growth is always bigger than nucleation:
    x0_array = np.asarray([x0])
    if minimisation_function(x0_array * 1e-10) > 0:
        return 0, 0

    # poormans minimisation evaluates an array and takes the lowest value,
    # normal minimisation methods have a hard time
    ncl = poor_mans_minimization(minimisation_function, x0)

    # ===== Debuging: Allows to investigate the function structure
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.title(str(np.log10(ncl_in)))
    # xlog = x0 - np.logspace(-15, np.log10(x0/2), 500)
    # xlin = np.linspace(x0/2, 0, 500)
    # test = np.append(xlog, xlin)
    # test_vals = minimisation_function(np.log10(test))
    # plt.plot(test, test_vals)
    # plt.vlines(x0, np.min(test_vals), np.max(test_vals), 'red')
    # plt.vlines(ncl, np.min(test_vals), np.max(test_vals), 'green')
    # #plt.yscale('log')
    # plt.xscale('log')
    # plt.show()

    # and the mass fraction of the nucleation seeds
    ccn_mass = ncl * n_ccn * m1  # mass of all cluod particle seeds
    return min([ccn_mass / cl_mass_total, 1]), ncl  # mass fraction of CCNs
