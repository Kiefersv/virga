import numpy as np
import astropy.units as u

from virga import justdoit as jdi

def test_mie_database():
    qext, qsca, asym, radii, wave = jdi.calc_mie_db(['MnS'],
                                '.', '.', rmin = 1e-5, nradii = 10)

    assert np.isclose(np.sum(qext), 3943.0645661036983)
    assert np.isclose(np.sum(qsca), 3761.3770094896213)
    assert np.isclose(np.sum(asym), 1918.452490845249)

def test_virga_cloud():
    # initialise atmosphere
    a = jdi.Atmosphere(['MnS'], fsed=1, mh=1, mmw=2.2)
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())

    # calculate cloud profile
    all_out = jdi.compute(a, as_dict=True, directory='.')

    assert np.isclose(np.sum(all_out['condensate_mmr']), 6.163947994805619e-05)
    assert np.isclose(np.sum(all_out['mean_particle_r']), 213.68473817971767)
    assert np.isclose(np.sum(all_out['droplet_eff_r']), 710.2622567489823)
    assert np.isclose(np.sum(all_out['column_density']), 2184.11033113616)
    assert np.isclose(np.sum(all_out['single_scattering']), 5752.215299269361)
    assert np.isclose(np.sum(all_out['asymmetry']), 3736.8968648397567)
    assert np.isclose(np.sum(all_out['opd_by_gas']), 0.9849945642140122)

