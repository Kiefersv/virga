import numpy as np
from scipy import optimize

from . import pvaps
from .root_functions import vfall_find_root, solve_force_balance, qvs_below_model, vfall

#   universal gas constant (erg/mol/K)
rgas = 8.3143e7
avog = 6.02e23
kb = rgas / avog

def _eddysed_mixed(t_top, p_top, t_mid, p_mid, condensibles, gas_mw, gas_mmr, rho_p,
                   mw_atmos, gravity, kz, mixl, fsed, b, eps, scale_h, z_top, z_alpha,
                   z_min, param, mh, sig, rmin, nrad, d_molecule, eps_k, c_p_factor,
                   og_vfall=True, do_virtual=True, supsat=0, verbose=False, mixed=True):
    """
    Given an atmosphere and condensates, calculate size and concentration
    of condensates in balance between eddy diffusion and sedimentation.

    Parameters
    ----------
    t_top : ndarray[float]
        Temperature at each layer (K)
    p_top : ndarray[float]
        Pressure at each layer (dyn/cm^2)
    t_mid : ndarray[float]
        Temperature at each midpoint (K)
    p_mid : ndarray[float]
        Pressure at each midpoint (dyn/cm^2)
    condensibles : ndarray[str], list[str]
        List or array of condensible gas names
    gas_mw : ndarray[float]
        Array of gas mean molecular weight from `gas_properties`
    gas_mmr : ndarray[float]
        Array of gas mixing ratio from `gas_properties`
    rho_p : ndarray[float]
        density of condensed vapor (g/cm^3)
    mw_atmos : float
        Mean molecular weight of the atmosphere
    gravity : float
        Gravity of planet cgs
    kz : float or ndarray[float]
        Kzz in cgs, either float or ndarray depending of whether or not
        it is set as input
    fsed : float
        Sedimentation efficiency coefficient, unitless
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    scale_h : ndarray
        Scale height of the atmosphere
    z_top : ndarray
        Altitude at each layer
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
    mh : float
        Atmospheric metallicity in NON log units (e.g. 1 for 1x solar)
    sig : float
        Width of the log normal particle distribution
    d_molecule : float
        diameter of atmospheric molecule (cm) (Rosner, 2000)
        (3.711e-8 for air, 3.798e-8 for N2, 2.827e-8 for H2)
        Set in Atmosphere constants
    eps_k : float
        Depth of the Lennard-Jones potential well for the atmosphere
        Used in the viscocity calculation (units are K) (Rosner, 2000)
    c_p_factor : float
        specific heat of atmosphere (erg/K/g) . Usually 7/2 for ideal gas
        diatomic molecules (e.g. H2, N2). Technically does slowly rise with
        increasing temperature
    og_vfall : bool , optional
        optional, default = True. True does the original fall velocity calculation.
        False does the updated one which runs a tad slower but is more consistent.
        The main effect of turning on False is particle sizes in the upper atmosphere
        that are slightly bigger.
    do_virtual : bool,optional
        optional, Default = True which adds a virtual layer if the
        species condenses below the model domain.
    supsat : float, optional
        Default = 0 , Saturation factor (after condensation)
    mixed : bool, optional
        If true, cloud particles are assumed to be able to mix together.

    Returns
    -------
    qc : ndarray
        condenstate mixing ratio (g/g)
    qt : ndarray
        gas + condensate mixing ratio (g/g)
    rg : ndarray
        geometric mean radius of condensate  cm
    reff : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz : ndarray
        number column density of condensate (cm^-3)
    qc_path : ndarray
        vertical path of condensate
    """

    #default for everything is false, will fill in as True as we go
    did_gas_condense = [False for i in condensibles]
    z_cld=None

    # set top of atmosphere values
    t_bot = t_top[-1]
    p_bot = p_top[-1]
    z_bot = z_top[-1]

    # get length of input arrays
    ngas =  len(condensibles)
    nz = len(t_mid)

    # set output arrays
    qc = np.zeros((nz, ngas))
    qt  = np.zeros((nz, ngas))
    rg = np.zeros((nz, ngas))
    reff = np.zeros((nz, ngas))
    ndz = np.zeros((nz, ngas))
    qc_path = np.zeros(ngas)
    z_cld_out = np.zeros(ngas)

    # set working arrays
    q_below = gas_mmr  # mass mixing ratios at bottom of the atmosphere

    # include decrease in condensate mixing ratio below model domain
    if do_virtual:
        for i, igas in zip(range(ngas), condensibles):

            # read in vapour pressure (arguments can be different, thus use kwargs)
            get_pvap = getattr(pvaps, igas)
            if igas in ['Mg2SiO4','CaTiO3','CaAl12O19','FakeHaze','H2SO4','KhareHaze','SteamHaze300K','SteamHaze400K']:
                pvap = get_pvap(t_bot, p_bot, mh=mh)
            else:
                pvap = get_pvap(t_bot, mh=mh)

            # mass mixing ratio of the vapour at saturation
            qvs_factor = (supsat+1)*gas_mw[i]/mw_atmos
            qvs = qvs_factor*pvap/p_bot

            # if the atmosphere is supersaturated at the lowest altitude, remove the
            # additional material
            if qvs <= q_below[i]:

                # temperature gradient
                dtdlnp = (t_top[-2] - t_bot) / np.log(p_bot/p_top[-2])

                # try to find the pressure > p_bot at which gas condenses
                try:
                    p_base = optimize.root_scalar(qvs_below_model,
                                bracket=[p_bot, p_bot * 1e3], method='brentq',
                                args=(dtdlnp, p_bot, t_bot, qvs_factor,
                                      igas, mh, q_below[i])
                                )

                    #Yes, the gas did condense (below the grid)
                    did_gas_condense[i] = True
                    if verbose:
                        print('Virtual Cloud Found for '+ igas)

                    # get values corresponding to the cloud below the layer
                    p_base = p_base.root
                    t_base = t_bot + np.log(p_bot/p_base)*dtdlnp
                    z_base = z_bot + scale_h[-1] * np.log(p_bot/p_base)

                    # Calculate temperature and pressure below bottom layer
                    # by adding a virtual layer
                    p_layer_virtual = 0.5*(p_bot + p_base)
                    t_layer_virtual = t_bot + np.log10(p_bot/p_layer_virtual)*dtdlnp

                    # overwrite q_below from this output for the next routine
                    _, _, _, _, _, q_below[i], _, _ = layer([igas],
                        np.asarray([rho_p[i]]), t_layer_virtual, p_layer_virtual,
                        t_bot, t_base, p_bot, p_base, kz[-1], mixl[-1], gravity,
                        mw_atmos, np.asarray([gas_mw[i]]), np.asarray([q_below[i]]),
                        supsat, fsed, b, eps, z_bot, z_base, z_alpha, z_min, param,
                        sig, mh, rmin, nrad, d_molecule, eps_k, c_p_factor, og_vfall,
                        z_cld, mixed=False
                    )

                # in case no cloud was found, let the user know
                except ValueError:
                    if verbose:
                        print('No virtual Cloud Found for '+ igas)


        # calculate the cloud structure in each layer of the atmosphere
        for iz in range(nz-1,-1,-1): #goes from bottom to top of the atmosphere
            qc[iz], qt[iz], rg[iz], reff[iz], ndz[iz], _, z_cld, _ = layer(condensibles,
                rho_p, t_mid[iz], p_mid[iz], t_top[iz], t_top[iz+1], p_top[iz],
                p_top[iz+1], kz[iz], mixl[iz], gravity, mw_atmos, gas_mw, q_below,
                supsat, fsed, b, eps, z_top[iz], z_top[iz+1], z_alpha, z_min, param,
                sig,mh, rmin, nrad, d_molecule, eps_k, c_p_factor, og_vfall, z_cld, mixed
            )

            qc_path = (qc_path + qc[iz] * (p_top[iz+1] - p_top[iz]) / gravity)

        # assign cloud height for each species
        z_cld_out = z_cld

    return qc, qt, rg, reff, ndz, qc_path, mixl, z_cld_out


def layer(gas_name, rho_p, t_layer, p_layer, t_top, t_bot, p_top, p_bot,
          kz, mixl, gravity, mw_atmos, gas_mw, q_below,
          supsat, fsed, b, eps, z_top, z_bot, z_alpha, z_min, param,
          sig, mh, rmin, nrad, d_molecule, eps_k, c_p_factor,
          og_vfall, z_cld, mixed):
    """
    Calculate layer condensate properties by iterating on optical depth
    in one model layer (convering on optical depth over sublayers)

    Parameters
    ----------
    gas_name : List[str]
        Name of condenstante
    rho_p : ndarray
        density of condensed vapor (g/cm^3)
    t_layer : float
        Temperature of layer mid-pt (K)
    p_layer : float
        Pressure of layer mid-pt (dyne/cm^2)
    t_top : float
        Temperature at top of layer (K)
    t_bot : float
        Temperature at botton of layer (K)
    p_top : float
        Pressure at top of layer (dyne/cm2)
    p_bot : float
        Pressure at botton of layer
    kz : float
        eddy diffusion coefficient (cm^2/s)
    mixl : float
        Mixing length (cm)
    gravity : float
        Gravity of planet cgs
    mw_atmos : float
        Molecular weight of the atmosphere
    gas_mw : ndarray
        Gas molecular weight
    q_below : ndarray
        total mixing ratio (vapor+condensate) below layer (g/g)
    supsat : float
        Super saturation factor
    fsed : float
        Sedimentation efficiency coefficient (unitless)
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    z_top : float
        Altitude at top of layer
    z_bot : float
        Altitude at bottom of layer
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
    sig : float
        Width of the log normal particle distribution
    mh : float
        Metallicity NON log soar (1=1xSolar)
    rmin : float
        Minium radius on grid (cm)
    nrad : int
        Number of radii on Mie grid
    d_molecule : float
        diameter of atmospheric molecule (cm) (Rosner, 2000)
        (3.711e-8 for air, 3.798e-8 for N2, 2.827e-8 for H2)
        Set in Atmosphere constants
    eps_k : float
        Depth of the Lennard-Jones potential well for the atmosphere
        Used in the viscocity calculation (units are K) (Rosner, 2000)
    c_p_factor : float
        specific heat of atmosphere (erg/K/g) . Usually 7/2 for ideal gas
        diatomic molecules (e.g. H2, N2). Technically does slowly rise with
        increasing temperature
    og_vfall : bool
        True = analytic fall speed calculation, False = force balance
    mixed : bool, optional
        If true, cloud particles are assumed to be able to mix together.

    Returns
    -------
    qc_layer : ndarray
        condenstate mixing ratio (g/g)
    qt_layer : ndarray
        gas + condensate mixing ratio (g/g)
    rg_layer : ndarray
        geometric mean radius of condensate  cm
    reff_layer : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz_layer : ndarray
        number column density of condensate (cm^-3)
    q_below : ndarray
        total mixing ratio (vapor+condensate) below layer (g/g)
    z_cld : ndarray
        altitude of the cloud layer
    fsed_layer : ndarray
        fsed of the layer
    """
    # get the correct size for ouput and working arrays
    lg = len(gas_name)
    if mixed:
        lg = 1

    # initialisation
    qc_layer = np.zeros(lg)
    qt_layer = np.zeros(lg)
    qt_top = np.zeros(lg)

    # sublayering parameters
    nsub_max = 128  # max number of sublayers
    nsub = 1  # starting number of sublayers

    # set up output arrays
    reff_layer = np.zeros(lg)
    rg_layer = np.zeros(lg)

    # define pysical parameters
    r_atmos = rgas / mw_atmos  # specific gas constant for atmosphere (erg/K/g)
    r_cloud = rgas / gas_mw  # specific gas constant for cloud (erg/K/g)
    dp_layer = p_bot - p_top  # pressure thickness of layer
    dlnp = np.log(p_bot / p_top)  # log pressure thickness
    dtdlnp = (t_top - t_bot) / dlnp  # temperature gradient
    scale_h = r_atmos * t_layer / gravity  # atmospheric scale height (cm)
    w_convect = kz / mixl   # convective velocity scale (cm/s) from mixing length theory
    n_atmos = p_layer / (kb * t_layer)  # atmospheric number density (molecules/cm^3)
    mfp = 1./(np.sqrt(2.)*n_atmos*np.pi*d_molecule**2) # atmospheric mean free path (cm)

    # atmospheric viscosity (dyne s/cm^2), QN B2 in A & M 2001, originally Rosner+2000
    # Rosner, D.E. 2000, Transport Processes in Chemically Reacting Flow Systems
    visc = (5./16. * np.sqrt(np.pi * kb * t_layer * (mw_atmos / avog)) /
            (np.pi * d_molecule**2) / (1.22 * (t_layer / eps_k)**(-0.16)))

    #   Top of convergence loop
    converge = False
    while not converge:

        # initialise Zero cumulative values
        za = np.zeros(lg)
        qc_layer, qt_layer, ndz_layer, opd_layer = (za, za, za, za)

        # total mixing ratio and pressure at bottom of sub-layer
        qt_bot_sub = q_below
        p_bot_sub = p_bot
        z_bot_sub = z_bot

        # pressures stepping for each sublayer
        dp_sub = dp_layer / nsub

        # loop over the sublayers until convergence is reached
        for isub in range(nsub):

            # structural parameters of the sublayer
            p_top_sub = p_bot_sub - dp_sub  # top pressure
            dz_sub = scale_h * np.log(p_bot_sub / p_top_sub)  # width
            p_sub = 0.5 * (p_bot_sub + p_top_sub)  # midpoint pressure
            z_top_sub = z_bot_sub + dz_sub  # top altitude
            z_sub = z_bot_sub + scale_h * np.log(p_bot_sub / p_sub)  # midpoint altitude
            t_sub = t_bot + np.log(p_bot / p_sub) * dtdlnp  # midpoint temperature

            # calculate cloud structure of sublayer
            qt_top, qc_sub, qt_sub, _, reff_sub, ndz_sub, z_cld, fsed_layer, rho_p_out = calc_qc(
                gas_name, supsat, t_sub, p_sub, r_atmos, r_cloud, qt_bot_sub, mixl,
                dz_sub, gravity, mw_atmos, mfp, visc, rho_p, w_convect, fsed, b, eps,
                param, z_bot_sub, z_sub, z_alpha, z_min, sig, mh, rmin, nrad, og_vfall,
                z_cld, mixed
            )

            # vertical sums
            qc_layer = qc_layer + qc_sub * dp_sub / gravity
            qt_layer = qt_layer + qt_sub * dp_sub / gravity
            ndz_layer = ndz_layer + ndz_sub

            # Increment values at bottom of sub-layer
            qt_bot_sub = qt_top
            p_bot_sub = p_top_sub
            z_bot_sub = z_top_sub

            # calculate odp wher the radius of cloud particles isn't 0
            mask = reff_sub > 0.
            opd_layer[mask] = (opd_layer[mask] + 1.5 * qc_sub[mask] * dp_sub / gravity /
                               (rho_p_out[mask] * reff_sub[mask]))

        # Check convergence on optical depth
        if nsub_max == 1:  # do not use sublayering
            converge = True
        elif nsub == 1:  # 1st it. is used to determine convergence
            opd_test = opd_layer
        elif (opd_layer == 0.).all() or (nsub >= nsub_max):  # break condition
            converge = True
        elif (abs(1. - opd_test / opd_layer) <= 1e-2).all():  # convergence
            converge = True
        else:  # no convergence, start over again
            opd_test = opd_layer

        # increase the number of sublayers
        nsub *= 2

    # Update properties at bottom of next layer
    # !!! Do not change this lane, it will break the code !!!
    for q, qt in enumerate(qt_top):
        q_below[q] = qt

    # Get layer averages where odp is not 0
    mask = opd_layer > 0
    reff_layer[mask] = 1.5 * qc_layer[mask] / (rho_p_out[mask] * opd_layer[mask])
    lnsig2 = 0.5 * np.log(sig) ** 2
    rg_layer[mask] = reff_layer[mask] * np.exp(-5 * lnsig2)

    # readjust for averaging weight
    qc_layer = qc_layer * gravity / dp_layer
    qt_layer = qt_layer * gravity / dp_layer

    return qc_layer, qt_layer, rg_layer, reff_layer, ndz_layer, q_below, z_cld, fsed_layer


def calc_qc(gas_name, supsat, t_layer, p_layer, r_atmos, r_cloud, q_below, mixl,
            dz_layer, gravity, mw_atmos, mfp, visc, rho_p, w_convect, fsed, b,
            eps, param, z_bot, z_layer, z_alpha, z_min, sig, mh, rmin, nrad,
            og_vfall=True, z_cld=None, mixed=False):
    """
    Calculate condensate optical depth and effective radius for a layer,
    assuming geometric scatterers.

    Parameters
    ----------
    gas_name : List[str]
        Name of condensate
    supsat : float
        Super saturation factor
    t_layer : float
        Temperature of layer mid-pt (K)
    p_layer : float
        Pressure of layer mid-pt (dyne/cm^2)
    r_atmos : float
        specific gas constant for atmosphere (erg/K/g)
    r_cloud : float
        specific gas constant for cloud species (erg/K/g)
    q_below : ndarray[float]
        total mixing ratio (vapor+condensate) below layer (g/g)
    mixl : float
        convective mixing length scale (cm): no less than 1/10 scale height
    dz_layer : float
        Thickness of layer cm
    gravity : float
        Gravity of planet cgs
    mw_atmos : float
        Molecular weight of the atmosphere
    mfp : float
        atmospheric mean free path (cm)
    visc : float
        atmospheric viscosity (dyne s/cm^2)
    rho_p : ndarray
        density of condensed vapor (g/cm^3)
    w_convect : float
        convective velocity scale (cm/s)
    fsed : float
        Sedimentation efficiency coefficient (unitless)
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    z_bot : float
        Altitude at bottom of layer
    z_layer : float
        Altitude of midpoint of layer (cm)
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
    sig : float
        Width of the log normal particle distrubtion
    mh : float
        Metallicity NON log solar (1 = 1x solar)
    rmin : float
        Minium radius on grid (cm)
    nrad : int
        Number of radii on Mie grid
    og_vfall : bool
        True = analytic fall speed calculation, False = force balance
    z_cld : flaot
        altitude of cloud
    mixed : bool, optional
        If true, cloud particles are assumed to be able to mix together.

    Returns
    -------
    qt_top : ndarray
        gas + condensate mixing ratio at top of layer(g/g)
    qc_layer : ndarray
        condenstate mixing ratio (g/g)
    qt_layer : ndarray
        gas + condensate mixing ratio (g/g)
    rg_layer : ndarray
        geometric mean radius of condensate  cm
    reff_layer : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz_layer : ndarray
        number column density of condensate (cm^-3)
    z_cld : ndarray
        altitude of the cloud layer
    fsed_layer : ndarray
        fsed of the layer
    rho_p_out : ndarray
        either rho_p or the average density
    """
    # ===================================================================================
    # Initialisation
    # ===================================================================================

    # pysical parameters
    rho_atmos = p_layer / (r_atmos * t_layer)  # atmospheric density (g/cm^3)
    lnsig2 = 0.5 * np.log(sig)**2  # geometric std dev of lognormal size distribution
    rho_p_avg = rho_p

    # length of arrays
    lgas = len(gas_name)

    # working arrays
    qt_layer = np.zeros(lgas)
    qt_top = np.zeros(lgas)
    qc_layer = np.zeros(lgas)
    z_cld = np.zeros(lgas)
    fsed_mid = np.zeros(lgas)

    # ===================================================================================
    # Step 1: calculate the cloud mass mixing ratio for each species individually
    # ===================================================================================
    # read in vapour pressure (arguments can be different, thus use kwargs)
    for i, gas in enumerate(gas_name):
        get_pvap = getattr(pvaps, gas)
        if gas in ['Mg2SiO4','CaTiO3','CaAl12O19','FakeHaze','H2SO4','KhareHaze','SteamHaze300K','SteamHaze400K']:
            pvap = get_pvap(t_layer, p_layer, mh=mh)
        else:
            pvap = get_pvap(t_layer, mh=mh)

        # mass mixing ratio of saturated vapor (g/g)
        qvs = (supsat + 1) * pvap / (r_cloud[i] * t_layer) / rho_atmos

        # if mass mixing ratio is below condensation limit, the layer is cloud free
        if q_below[i] < qvs:
            qt_layer[i] = q_below[i]
            qt_top[i] = q_below[i]
            qc_layer[i] = 0.
            z_cld[i] = z_cld[i]
            fsed_mid[i] = 0

        # Cloudy layer: first calculate qt and qc at top of layer, then calculate the
        # additional cloud properties of the layer
        else:
            # if no cloud layer was found up until now, remember the altitude
            if isinstance(z_cld[i], type(None)):
                z_cld[i] = z_bot

            # solution for constant fsed
            if param == "const":
                qt_top[i] = qvs + (q_below[i] - qvs) * np.exp(-fsed * dz_layer / mixl)
            # solution for exponentially parametrisation
            elif param == "exp":
                fs = fsed / np.exp(z_alpha / b)
                qt_top[i] = qvs + (q_below[i] - qvs) * np.exp(-b*fs/mixl*np.exp(z_bot/b)
                            * (np.exp(dz_layer / b) - 1) + eps * dz_layer / mixl)
            # error if otherwise
            else:
                raise ValueError('Fsed parametrisation not supported')

            # Use trapezoid rule to calculate layer averages
            qt_layer[i] = 0.5 * (q_below[i] + qt_top[i])

            #   Find total condensate mixing ratio
            qc_layer[i] = np.max([0., qt_layer[i] - qvs])

    # prepare for future calculation if particles are mixed
    if mixed:
        qc_layer = np.asarray([np.sum(qc_layer)])
        # first check if there is any cloud material, if not, be done
        if qc_layer == 0:
            rg_layer = np.asarray([0])
            reff_layer = np.asarray([0])
            ndz_layer = np.asarray([0])
            rho_p_avg = np.asarray([0])
            return (qt_top, qc_layer, qt_layer, rg_layer, reff_layer, ndz_layer,
                    z_cld, fsed_mid, rho_p_avg)
        # reduce gas species to one general species
        gas_name = ['cloud']
        # Calculate average cloud particle density
        rho_p_avg = np.asarray([np.sum(qc_layer)/np.sum(qc_layer/rho_p)])

    # ===================================================================================
    # Calculate the radius of cloud particles by balancing the fall out rate
    # ===================================================================================

    # prepare output arrays
    rg_layer = np.zeros(len(gas_name))
    reff_layer = np.zeros(len(gas_name))

    # fsed at middle of layer
    if param == 'exp':
        fs = fsed / np.exp(z_alpha / b)
        fsed_mid = fs * np.exp(z_layer / b) + eps
    else:  # 'const'
        fsed_mid = fsed

    # loop over all cloud particle species
    for i, gas in enumerate(gas_name):
        # range of particle radii to search (cm)
        rlo = 1.e-10
        rhi = 10.

        # calculate rw, if failed, expand search domain
        find_root = True
        rw_layer = 0  # default initialisation
        while find_root:
            try:
                # use the analytic solution to the fall speed
                if og_vfall:
                    rw_temp = optimize.root_scalar(vfall_find_root,
                        bracket=[rlo, rhi], method='brentq', args=(gravity, mw_atmos,
                        mfp, visc, t_layer,p_layer, rho_p_avg[i], w_convect)
                    )
                    rw_layer = rw_temp.root
                # balance forces to arrive at solution
                else:
                    rw_layer = solve_force_balance("rw", w_convect, gravity,
                        mw_atmos, mfp, visc, t_layer, p_layer, rho_p_avg[i], rlo, rhi
                    )

                # the root was found if no error was raised
                find_root = False

            # expend the search radius if the search has failed
            except ValueError:
                rlo = rlo / 10
                rhi = rhi * 10

        # calculate the fall velocity for each radius in the radius bin
        r_, rup, dr = get_r_grid(r_min=rmin, n_radii=nrad)
        vfall_temp = []
        for j in range(len(r_)):
            if og_vfall:
                # use the analytic solution to the fall speed
                vfall_temp.append(vfall(r_[j], gravity, mw_atmos, mfp, visc,
                    t_layer, p_layer, rho_p_avg[i])
                )
            else:
                # balance forces to arrive at solution
                vlo = 1e0
                vhi = 1e6
                find_root = True
                while find_root:
                    try:
                        vfall_temp.append(solve_force_balance("vfall", r_[j],
                            gravity, mw_atmos, mfp, visc, t_layer, p_layer, rho_p_avg[i],
                            vlo, vhi)
                        )
                        find_root = False
                    # if failed, expand search domain
                    except ValueError:
                        vlo = vlo / 10
                        vhi = vhi * 10

        # find alpha for power law fit vf = w(r/rw)^alpha
        def pow_law(r, alpha):
            return np.log(w_convect) + alpha * np.log(r / rw_layer)

        # fit the power law to the fall speeds
        pars, cov = optimize.curve_fit(f=pow_law, xdata=r_, ydata=np.log(vfall_temp),
                                       p0=[0], bounds=(-np.inf, np.inf))
        alpha = pars[0]

        # geometric mean radius of lognormal size distribution, EQN. 13 A&M
        rg_layer[i] = (fsed_mid**(1. / alpha) * rw_layer * np.exp(-(alpha + 6) * lnsig2))

        # droplet effective radius (cm)
        reff_layer[i] = rg_layer[i] * np.exp(5 * lnsig2)

    # ===============================================================================
    # Calculate the cloud particle number densities
    # ===============================================================================

    if mixed:
        # total volume of mixed cloud particles
        vtot = np.asarray([np.sum(qc_layer / rho_p_avg)])
    else:
        # each species individual if not mixed
        vtot = qc_layer / rho_p_avg

    # column droplet number concentration (cm^-2), EQN. 14 A&M
    ndz_layer = (3 * rho_atmos * vtot * dz_layer /
                 (4 * np.pi * rg_layer ** 3) * np.exp(-9 * lnsig2))

    return (qt_top, qc_layer, qt_layer, rg_layer, reff_layer, ndz_layer, z_cld,
            fsed_mid, rho_p_avg)


def get_r_grid(r_min=1e-8, n_radii=60):
    """
    Warning
    -------
    Original code from A&M code.
    Discontinued function. See 'get_r_grid'.

    Get spacing of radii to run Mie code

    r_min : float
        Minimum radius to compute (cm)

    n_radii : int
        Number of radii to compute
    """
    vrat = 2.2
    pw = 1. / 3.
    f1 = ( 2.0*vrat / ( 1.0 + vrat) )**pw
    f2 = (( 2.0 / ( 1.0 + vrat ) )**pw) * (vrat**(pw-1.0))

    radius = r_min * vrat**(np.linspace(0,n_radii-1,n_radii)/3.)
    rup = f1*radius
    dr = f2*radius

    return radius, rup, dr