""" Tests for all functionality of VIRGA """

import numpy as np
import astropy.units as u

from virga import justdoit as jdi

def test_mie_database():
    _, _, _, _, _ = jdi.calc_mie_db(['Cr'],'.', '.', rmin = 1e-5, nradii = 10)
    qext, qsca, asym, radii, wave = jdi.calc_mie_db(['MnS'],
                                '.', '.', rmin = 1e-5, nradii = 10)
    assert np.isclose(np.sum(qext), 3435.3451248302963)
    assert np.isclose(np.sum(qsca), 3295.247186049365)
    assert np.isclose(np.sum(asym), 1093.2539398130693)

def test_virga_cloud():
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile using og_solver
    all_out = jdi.compute(a, as_dict=True, directory='.')
    # check outputs
    assert np.isclose(np.sum(all_out['condensate_mmr']), 6.163947994805619e-05)
    assert np.isclose(np.sum(all_out['mean_particle_r']), 213.68473817971767)
    assert np.isclose(np.sum(all_out['droplet_eff_r']), 710.2622567489823)
    assert np.isclose(np.sum(all_out['column_density']), 2184.11033113616)
    assert np.isclose(np.sum(all_out['single_scattering']), 5752.215299269361)
    assert np.isclose(np.sum(all_out['asymmetry']), 3736.8968648397567)
    assert np.isclose(np.sum(all_out['opd_by_gas']), 0.9849945642140122)
    assert np.isclose(np.sum(all_out['opd_per_layer']), 0.010309205846656488)

def test_virga_direct_solver():
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile using og_solver
    all_out = jdi.compute(a, as_dict=True, directory='.', og_solver=False)
    # check outputs
    assert np.isclose(np.sum(all_out['condensate_mmr']), 1.843949389772658e-05)
    assert np.isclose(np.sum(all_out['mean_particle_r']), 720.4830441320369)
    assert np.isclose(np.sum(all_out['droplet_eff_r']), 2394.7986048690545)
    assert np.isclose(np.sum(all_out['column_density']), 51.23056414336807)
    assert np.isclose(np.sum(all_out['single_scattering']), 5743.974745378395)
    assert np.isclose(np.sum(all_out['asymmetry']), 3910.185884007504)
    assert np.isclose(np.sum(all_out['opd_by_gas']), 0.2613196396359124)
    assert np.isclose(np.sum(all_out['opd_per_layer']), 0.0007309227958875775)

def test_virga_size_distribution():
    # calculate different size distributions
    # -> lognormal is already coverd with the basic test (test_virga_single_cloud)
    # gamma size distribution
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2, size_distribution='gamma')
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    all_out = jdi.compute(a, as_dict=True, directory='.')
    assert np.isclose(np.sum(all_out['droplet_eff_r']), 940.9847777172278)
    assert np.isclose(np.sum(all_out['opd_per_layer']), 0.001924747698604046)
    # exponential distribution
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2, size_distribution='exponential')
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    all_out = jdi.compute(a, as_dict=True, directory='.')
    assert np.isclose(np.sum(all_out['droplet_eff_r']), 776.5807857795131)
    assert np.isclose(np.sum(all_out['opd_per_layer']), 0.003930891126005702)
    # monodisperse distributions
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2, size_distribution='monodisperse')
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    all_out = jdi.compute(a, as_dict=True, directory='.')
    assert np.isclose(np.sum(all_out['droplet_eff_r']), 1195.0939223236237)
    assert np.isclose(np.sum(all_out['opd_per_layer']), 0.0005444735981582805)

def test_virga_mixed_cloud():
    # mixed cloud atmosphere
    a = jdi.Atmosphere(['MnS', 'Cr'], fsed=1)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    all_out = jdi.compute(a, as_dict=True, directory='.', mixed=True)
    assert np.isclose(np.sum(all_out['condensate_mmr'][:,-1]), 0.00012194708770283248)
    assert np.isclose(np.sum(all_out['mean_particle_r'][:,-1]), 184.1291922153259)
    assert np.isclose(np.sum(np.nan_to_num(all_out['column_density'][:,-1])), 60029.405846054025)

def test_virga_fsed():
    # Note: constant fsed is tested in test_virga_single_cloud()

    # TODO: add test for exp

    # fsed array (one fsed for all species)
    fsed_in = np.linspace(2, 0.1, 53)
    a = jdi.Atmosphere(['MnS'], fsed=fsed_in, param='array')
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    all_out = jdi.compute(a, directory='.')
    assert np.isclose(np.sum(all_out['condensate_mmr']), 5.7569695488823735e-05)
    assert np.isclose(np.sum(all_out['mean_particle_r']), 198.05075000254712)
    assert np.isclose(np.sum(np.nan_to_num(all_out['column_density'])), 3403.5371544131235)

    # fsed array (different fsed for all species)
    fsed_in = np.asarray([np.linspace(2, 0.1, 53), np.linspace(1, 0.05, 53)]).T
    a = jdi.Atmosphere(['MnS', 'Cr'], fsed=fsed_in, param='array')
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    all_out = jdi.compute(a, directory='.')
    assert np.isclose(np.sum(all_out['condensate_mmr']), 0.00017752695437599523)
    assert np.isclose(np.sum(all_out['mean_particle_r']), 193.98902426602106)
    assert np.isclose(np.sum(np.nan_to_num(all_out['column_density'])), 421573.4502528833)

