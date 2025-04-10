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
    assert np.isclose(np.sum(all_out['single_scattering']), 5550.9344884880975)
    assert np.isclose(np.sum(all_out['asymmetry']), 2102.9942713110036)
    assert np.isclose(np.sum(all_out['opd_by_gas']), 0.9849945642140122)
    assert np.isclose(np.sum(all_out['opd_per_layer']), 0.01236885455767609)

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
    assert np.isclose(np.sum(all_out['single_scattering']), 5561.577845960359)
    assert np.isclose(np.sum(all_out['asymmetry']), 2196.782156982542)
    assert np.isclose(np.sum(all_out['opd_by_gas']), 0.26131963963591226)
    assert np.isclose(np.sum(all_out['opd_per_layer']), 0.0008936391095493811)

def test_virga_size_distribution():
    # calculate different size distributions
    # -> lognormal is already coverd with the basic test (test_virga_single_cloud)
    # gamma size distribution
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2, size_distribution='gamma')
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    all_out = jdi.compute(a, as_dict=True, directory='.')
    assert np.isclose(np.sum(all_out['droplet_eff_r']), 940.9847777172278)
    assert np.isclose(np.sum(all_out['opd_per_layer']), 0.0022617745054825046)
    # exponential distribution
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2, size_distribution='exponential')
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    all_out = jdi.compute(a, as_dict=True, directory='.')
    assert np.isclose(np.sum(all_out['droplet_eff_r']), 776.5807857795131)
    assert np.isclose(np.sum(all_out['opd_per_layer']), 0.004498364511911136)
    # monodisperse distributions
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2, size_distribution='monodisperse')
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    all_out = jdi.compute(a, as_dict=True, directory='.')
    assert np.isclose(np.sum(all_out['droplet_eff_r']), 1195.0939223236237)
    assert np.isclose(np.sum(all_out['opd_per_layer']), 0.0006611566799597155)

def test_virga_mixed_cloud():
    # mixed cloud atmosphere for single species
    a = jdi.Atmosphere(['MnS', 'Cr'], fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    # calculate cloud profile using og_solver
    all_out = jdi.compute(a, as_dict=True, directory='.', mixed=True)
    # check outputs
    assert np.isclose(np.sum(all_out['condensate_mmr'][:,0]), 6.163947994805619e-05)
    assert np.isclose(np.sum(all_out['condensate_mmr'][:,-1]), 0.0001220546204504057)
    assert np.isclose(np.sum(all_out['mean_particle_r'][:,0]), 213.68473817971767)
    assert np.isclose(np.sum(all_out['mean_particle_r'][:,-1]), 271.6210390747892)
    assert np.isclose(np.sum(all_out['droplet_eff_r'][:,0]), 710.2622567489823)
    assert np.isclose(np.sum(all_out['droplet_eff_r'][:,-1]), 902.8355222613407)
    assert np.isclose(np.sum(all_out['column_density'][:,0]), 2184.11033113616)
    assert np.isclose(np.sum(all_out['column_density'][:,-1]), 6339.705942962247)
    assert np.isclose(np.sum(all_out['single_scattering']), 3699.098557198065)
    assert np.isclose(np.sum(all_out['asymmetry']), 3263.267445382071)
    assert np.isclose(np.sum(all_out['opd_by_gas'][:,0]), 0.9849945642140122)
    assert np.isclose(np.sum(all_out['opd_by_gas'][:,-1]), 3.575627014252061)
    assert np.isclose(np.sum(all_out['opd_per_layer']), 0.062410121071994985)

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

