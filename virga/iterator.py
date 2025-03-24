""" Functions to generate a cloud structure by iterating over fsed calculations """
import os
import csv
import numpy as np
import xarray as xr

from .justdoit import compute
from . import gas_properties
from . import pvaps

from scipy.optimize import minimize, root

#   universal gas constant (erg/mol/K)
RGAS = 8.3143e7
AVOG = 6.02e23
KB = RGAS / AVOG
PF = 1000000  # Reference pressure [dyn / cm2]


def compute_iterator(atmo, directory=None, as_dict=True, og_solver=True,
                     direct_tol=1e-15, refine_TP=True, og_vfall=True, analytical_rg=True,
                     do_virtual=True, size_distribution='lognormal', mixed=False):
    """ TODO """

    # set up working variables
    memory_vir = []  # remember virga output
    memory_new = []  # remember newly calculated values
    converged = False  # convergence flag
    qc_old = None  # save old mmr to check for convergence
    fsed_in = atmo.fsed  # first guess of fsed
    fsed_w = np.zeros((atmo.p_level.shape[0], len(atmo.condensibles)))

    # loop
    i = 0
    while not converged:
        # ==== Original VIRGA calculation ===============================================
        # compute cloud structure with given fsed_in
        all_out = compute(atmo, directory, as_dict, og_solver, direct_tol, refine_TP,
                          og_vfall, analytical_rg, do_virtual, size_distribution, mixed)

        # remember output
        memory_vir.append(all_out)

        # ==== Adjustment for additional physics ========================================
        new_out = {}

        # values needed
        qc = all_out['condensate_mmr']
        mmw =  all_out['scalar_inputs']['mmw']
        pres = all_out['pressure']*1e6
        temp = all_out['temperature']
        dz = all_out['layer_thickness']
        kzz = all_out['kz']

        # # correct masses for rain TODO: proper corrections
        # m_gas = mmw * pres / temp / KB  # get atmospheric mass
        # m_c_tot = qc * m_gas  # get cloud mass in layer
        # v_fall = 1  # when particles slow down, the density increases TODO
        # # qc added through rain per layer:
        # qc_rain = (np.cumsum(m_c_tot*dz) - m_c_tot*dz) / m_gas / dz
        # qc_rain[qc < 1e-20] = 0  # do not add in places where clouds evaporate
        # new_out['condensate_mmr'] = qc + qc_rain  # actual qc present

        # correct number densities for rain

        # calculated new fsed
        fsed = _top_down_property_calculator(atmo.condensibles, qc, temp, pres, kzz, mmw,
                                             atmo.g, atmo.sig, size_distribution, mixed)
        fsed_w[0] = fsed[0]
        fsed_w[-1] = fsed[-1]
        fsed_w[1:-1] = (fsed[1:] + fsed[:-1])/2  # TODO do this in logp space
        # adjust fsed
        atmo.fsed = fsed_w

        # ==== Check for convergance and prepare next loop ==============================
        i += 1
        # check if there is already an old mmr to compare, otherwise start over
        if isinstance(qc_old, type(None)):
            qc_old = qc  # remember current run
            continue

        # Compare to previous value and see if deviations are within tolerance
        error = np.abs((qc[qc > 0] - qc_old[qc > 0]) / qc[qc > 0])
        test = error < 1e-3
        if (error < 1e-3).all():
            break  # stop iteration
        qc_old = qc  # remember current run

    return memory_vir



def _top_down_property_calculator(species, qc, temp, pres, kzz, mmw, gravity, sig,
                                  size_distribution, mixed):
    """
    This function calculates the cloud particle properties top down after
    virga calculated qc bottom up.
    """

    # ==== Preparation ==================================================================
    # get shape of input
    nz = qc.shape[0]
    ngas = qc.shape[1]

    # get working arrays
    ncl_in = np.zeros((nz, ngas))
    fsed = np.zeros((nz, ngas))

    # physical parameters
    rho_atmo = mmw * pres / temp / RGAS

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
    rho_p = np.zeros(ngas)
    for i, igas in enumerate(species):
        if mixed:
            mask = (qc[:, -1] == 0) # check where qc is 0
            rho_p[i] = np.sum(qc[mask, :-1]) / np.sum(qc[mask, :-1] / rho_p[:-1])
        else:
            run_gas = getattr(gas_properties, igas)
            _, _, rho_p[i] = run_gas(1)  # only the density is needed

    # fsed prefactor calculation
    h = RGAS * temp / gravity / mmw  # scale height
    ct = np.sqrt(2 * RGAS * temp / mmw)  # sound speed
    vsed_pre = np.sqrt(np.pi) * gravity / 2 / rho_atmo / ct
    fsed_pre = vsed_pre * fac_1 * h / kzz  # missing: * rho_p[ig] * rg

    # ==== Properties of each layer =====================================================
    for iz in range(nz):  # goes from top to bottom of the atmosphere
        for ig, igas in enumerate(species):  # and for each species
            # get number of nucleation seeds, start with infall from cell above
            ncl_tot = ncl_in[iz, ig]

            # mixed cloud particles are made from all nucleation seeds
            if mixed:
                for im, igas in enumerate(species[:-1]):
                    # if there is no material, there will be no seeds
                    if qc[iz, im] <= 0:
                        continue
                    ncl_tot += _get_ccn_nucleation(igas, qc[iz, im], temp[iz], pres[iz],
                                                   mmw, ncl_in[iz, im])
            # homogenous particles are made from only their own seeds
            else:
                # if there is no material, there will be no seeds
                if qc[iz, ig] > 0:
                    ncl_tot += _get_ccn_nucleation(igas, qc[iz, ig], temp[iz], pres[iz],
                                                   mmw, ncl_in[iz, ig])

            # advect the number to the next cell
            if iz < nz-1:  # bottom cell does not need to pass value
                ncl_in[iz + 1, ig] = ncl_tot

            # now the material can condense onto the particles
            rg = 0
            if ncl_tot > 1e-50:
                # get the mean radius
                rg = (3 * qc[iz, ig] * rho_atmo[iz]
                      / 4 / np.pi / rho_p[-1] / ncl_tot / fac_3)**(1/3)

            # get new fsed parameter
            fsed[iz, ig] = fsed_pre[iz] * rg * rho_p[ig]

    # return fsed
    return fsed[:, -1]  # TODO corerct the routins to require fsed per particle


# =======================================================================================
#  Additional functionalities
# =======================================================================================

def _get_ccn_nucleation(spec, mmr, temp, pres, mmw, ncl_in,
                        analytic_approximation=False):
    # TODO list: Cr, ZnS, Na2S, Fe, CH4, Al2O3, CaTiO3, CaAl12O19, SiO2
    # these species are calculated using modified CNT
    if spec in ['SiO', 'Cr', 'KCl', 'NaCl', 'CsCl', 'H2O', 'NH3',
                'H2S', 'CH4', 'MnS']:
        frac, ncl = _fracs_elsie(spec, mmr, temp, pres, mmw, ncl_in,
                                 analytic_approximation)
    # these species use the becker-doering method for nucleation
    elif spec in ['TiO2']:
        frac, ncl = _non_classical_nucleation(
            spec, mmr, temp, pres, mmw, ncl_in, analytic_approximation
        )
    # some species are known to not nucleate
    elif spec in ['MgSiO3', 'Mg2SiO4']:
        frac, ncl = (0, 0)
    # all other species are not yet implemented
    else:
        raise ValueError(spec + ' is not yet suppoerted.')
    # and we are done
    return ncl

# =======================================================================================
# General description for non classicle nucleation rate
def _non_classical_nucleation(spec, mmr, temp, pres, mmw, ncl_in=0,
                              analytic_approx=False):
    # Here the nucleation rate and the growth rate are calculated. The goal is to find
    # the number of cloud particles at which both the nucleation rate J and the growth
    # rate R are equal which marks the point after which most material will be used up
    # in gorwth before it can nucleate and thus marks an estimate on the number of
    # cloud particles 'nc' produced.
    # -> Non Classicle Nucleation rate : J = (Sum[1 / (An nu n1 nn)])^(-1)
    # -> growth rate: R = Ac nu n1 nc (1 - 1/S)
    # => nc^(-1) = Sum[(Ac / An) * (1 / nn)] * (1 - 1/S)
    # This equation can be simplified with:
    # -> Ac / An = 4 pi rc^2 / 4 pi rn^2 = rc^2 / rn^2
    #            = (r1 Nc^(1/3))^2 / (r1 N^(1/3))^2 = (Nc / N)^(2/3)
    # The supersaturation is S = p1 / pvap
    # The number density nn is derived according to Lee et al. 2013
    # The partial pressure is adjusted for the material lost by nucleation
    # -> p1_reduced = (n1 - n*nc)/ntot * p = p1 - n*nc*kb*T

    # read in data for species
    data = _read_data(spec)
    r1 = data['size'].sel(dict(N=1)).values
    m1 = data['mass'].sel(dict(N=1)).values
    n = data['N'].values
    n_ccn = data.attrs['n_ccn']
    rho = data.attrs['density']
    gibbs = data['gibbs_energy'].interp(temperature=temp).values

    # data of the solid
    data_s = _read_data(spec + '[s]')
    gibbs_s = data_s['gibbs_energy'].interp(temperature=temp).values
    pvap = PF * np.exp(-(gibbs[0] - gibbs_s) / RGAS / temp)

    # calculate surface area prefactor, nc = 100
    acdan = (n_ccn / n) ** (2 / 3)

    # get the prefactor for each cluster size nn from Gibbs data
    cl_mass_total = mmr * mmw * pres / RGAS / temp  # total cloud mass
    mol_frac = mmr * mmw / m1 / AVOG  # molar fraction of the monomer
    n1 = mol_frac * pres / KB / temp  # monomer number density
    p1 = mol_frac * pres  # partial pressure of the monomer
    exp = np.exp(-(gibbs - n * gibbs[0]) / RGAS / temp)  # exponential factor

    # The analytic approximation neglects the depletion from nucleation on the monomer
    # for the calculation of the nucleation and growth rate.
    if analytic_approx:
        nn = PF * (p1 / PF) ** n * exp / KB / temp  # N-mer density
        eq_ncl = np.min(np.asarray([1 / np.sum(acdan / nn), n1 / n_ccn]))
        ccn_mass = eq_ncl * n_ccn * m1  # mass of all cluod particle seeds
        eq_sol = min([ccn_mass / cl_mass_total, 1])
        return eq_sol, eq_ncl

    # funciton to find nc, nc is in log to optimize minimization
    def func(lognc):
        out = np.zeros_like(lognc)
        pir1_save = np.zeros_like(lognc)
        for l, lnc in enumerate(lognc):
            # reduced partial pressure, accounting for nucleated material
            ncl_tot = 10 ** lnc
            p1r = p1 - n_ccn * ncl_tot * KB * temp
            pir1_save[l] = p1r
            nn = PF * (p1r / PF) ** n * exp / KB / temp  # N-mer density
            sfac = (1 - pvap / p1r)  # supersaturation factor
            # spot where nucleation and growth are equal
            out[l] = np.abs(1 / np.sum(acdan / nn) * sfac - (ncl_tot + ncl_in))

        return out

    # execute the minimisation of func to find where growth equals nucleation
    x0 = n1 / n_ccn

    nc_min = root(func, np.log10(x0 * 1e-1))
    ncl = max([10 ** nc_min.x, 0])

    # and the mass fraction of the nucleation seeds
    ccn_mass = ncl * n_ccn * m1  # mass of all cluod particle seeds
    num_sol = ccn_mass / cl_mass_total
    return min([num_sol, 1]), ncl  # mass fraction of CCNs


def _read_data(spec):
    """
    Read in molecular data for a nucleation species.
    :param spec: name of nucleation species
    :return: xarray with all data
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
    # convert from m to cm
    data_size = data_array[4, 1:].astype(float) * 1e2
    # convert from kg to g
    data_mass = data_array[5, 1:].astype(float) * 1e3
    data_temp = data_array[7:, 0].astype(float)
    # convert from kj / mol to erg / mol
    data_gibs = data_array[7:, 1:].astype(float) * 1e3 * 1e7

    # Assign values to a xarray Dataset
    ds = xr.Dataset(
        data_vars=dict(
            gibbs_energy=(['temperature', 'N'], data_gibs),
            size=(['N'], data_size),
            mass=(['N'], data_mass),
        ),
        coords=dict(
            temperature=data_temp,
            N=data_n,
        ),
        attrs=dict(
            name=data_array[0][0],
            density=data_rho,
            n_ccn=data_nccn,
        ),
    )

    # return data array
    return ds


# =======================================================================================
# Wrapper function for a given nucleation rate
def _balance_with_j(j, temp, pres, mmr, mmw, n_ccn, m1, r1, pvap, ncl_in=0,
                    analytic_approx=True):
    # Here the nucleation rate and the growth rate are balanced. The goal is to find
    # the number of cloud particles at which both the nucleation rate J and the growth
    # rate R are equal which marks the point after which most material will be used up
    # in gorwth before it can nucleate and thus marks an estimate on the number of
    # cloud particles 'nc' produced. This function takes the nucleation rate as input.
    # -> Nucleation rate : J(n1)
    # -> growth rate: R = Ac nu n1 nc (1 - 1/S)
    # => Ac nu n1 nc (1 - 1/S) - J = 0

    # calculate growth rate properties
    cl_mass_total = mmr * mmw * pres / RGAS / temp  # total cloud mass
    ac = 4 * np.pi * r1 ** 2 * n_ccn ** (2 / 3)  # cloud particle surface area
    nu = np.sqrt(KB * temp / 2 / np.pi / m1)  # relative velocity
    mol_frac = mmr * mmw / m1 / AVOG  # molar fraction of the monomer
    n1 = mol_frac * pres / KB / temp  # monomer number density

    # The analytic approximation neglects the depletion from nucleation on the monomer
    # for the calculation of the nucleation and growth rate.
    if analytic_approx:
        eq_ncl = j(np.asarray([n1])) / ac * nu * n1
        ccn_mass = eq_ncl * n_ccn * m1  # mass of all cluod particle seeds
        return min([ccn_mass / cl_mass_total, 1]), eq_ncl  # mass fraction of CCNs

    # funciton to find nc, nc is in log to optimize minimization
    def minimisation_function(lognc):
        # reduced monomer number density, accounting for nucleated material
        nc = 10 ** lognc
        n1r = n1 - nc * n_ccn
        n1r[n1r < 0] = 1e-200  # prevent negative number densities
        p1r = n1r * KB * temp  # reduced partial pressure
        sfac = (1 - pvap / p1r)  # supersaturation factor
        # spot where nuclation and growth are equal
        jnuc = j(n1r)
        growth_rate = ac * nu * n1r * (nc + ncl_in) * sfac
        val = (growth_rate - jnuc)

        return val

    def poor_mans_minimization(func, xmax, accuracy=int(1e5)):
        """
        A brute force minimisation function optimised for the shape of the
        minimisation_function. It first searches for a local minimia away from xmax.
        If nan was found, it allows to select xmax as minima.
        """
        # define log part and linear part of search grid
        xlog = xmax - np.logspace(-5, np.log10(xmax / 2), accuracy // 2)
        xlin = np.linspace(xmax / 2, 0, accuracy // 2)
        x = np.append(xlog, xlin)

        # first search away from maxima
        x_first = x[x < xmax * 0.99]
        vals = np.abs(func(np.log10(x_first)))
        nan_mask = ~np.isnan(vals)  # homemade np.isnotnan
        nonnan_vals = vals[nan_mask]
        min_idx = np.min(np.where(nonnan_vals == nonnan_vals.min()))
        # if minima is not close to the boundary, return
        if min_idx > 2:
            return x_first[nan_mask][min_idx]

        # if minima is close to the boundary, evaluate the second part
        x_second = x[x > xmax * 0.8]
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


# =======================================================================================
# Solution for specific elements
def _fracs_elsie(spec, mmr, temp, pres, mmw, ncl_in=0, analytic_approx=False):
    # get nucleation rate, vapour pressure, number of monomers per ccn, radius of
    # monomer, and mass of monomer from Lee et al. 2018 (A&A 614, A126).
    j, pvap, n_ccn, r1, m1 = _elsies_vapor_pressures(spec, temp)

    # call balancing function
    result = _balance_with_j(j, temp, pres, mmr, mmw, n_ccn, m1, r1, pvap,
                             ncl_in, analytic_approx)

    # return result
    return result


def _elsies_vapor_pressures(spec, temp, nf=1):
    """
    Data according to Lee et al. 2018 (A&A 614, A126)

    :param spec: Name of species, see below for supported
    :param temp: Temperature in Kelvin
    :param nf: fitting factor (default nf=1)
    :return:
        pvap : vapor pressure
        n_ccn : number of monomers to form a ccn
        r1 : radius of the monomer
        m1 : mass of the monomer
    """
    # the number of ccns needed to form a cloud particle is not disscussed
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
    elif spec == 'MnS':
        pvap = 10.0**(11.5315-23810./temp)*1e6  # from Virga
        r1 = 2e-8  # TODO just a guess, find data
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
        def j(n1):
            p1 = n1 * KB * temp
            logS = np.log(p1 / pvap)
            return n1 ** 2 * np.exp(1.33 - 4.40e12 / temp ** 3 / logS ** 2)
    else:
        # modified classical nucleation theory
        def j(n1):
            # general parameters
            p1 = n1 * KB * temp  # partial pressure
            logS = np.log(p1 / pvap)  # log of saturation
            ti = 4 * np.pi * r1 ** 2 * sig / KB / temp  # theta_inf
            ni = (2 / 3 * ti / logS) ** 3  # N_star_inf
            ni[ni < 1] = 1

            # critical cluster size parameters
            ns = ni / 8 + 1  # ni / 8 * (1 + np.sqrt(1 + 2*(nf/ni)**(1/3)) - 2*(nf/ni)**(1/3))**3 + 1 # critical cluster size
            rs = ns ** (1 / 3) * r1  # radius of critical cluster
            dgrt = ti * (ns - 1) ** (2 / 3)  # dGibbs/RT
            fs = PF / KB / temp * (p1 / PF) ** ns * np.exp(-dgrt)  # number density of critical cluster size

            # mcnt factors
            zfac = (ti / (9 * np.pi * (ns - 1) ** (4 / 3))) ** (1 / 2)  # zeldovich factor
            vrel = np.sqrt(KB * temp / 2 / np.pi / m1)  # relative velocity
            tau = 4 * np.pi * rs ** 2 * vrel * fs  # gowth rate

            # nucleation rate
            jrate = n1 * tau * zfac * np.exp((ns - 1) * logS - dgrt)

            # exception handling
            jrate[logS < 0] = 0
            # jrate[jrate > 1e200] = 1e200

            return jrate

    # return results
    return j, pvap, n_ccn, r1, m1
