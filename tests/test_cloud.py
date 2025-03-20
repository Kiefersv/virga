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

def test_virga_fsed():
    # Note: constant fsed is tested in test_virga_single_cloud()

    # TODO: add test for exp

    # fsed array
    a = jdi.Atmosphere(['MnS'], fsed=np.linspace(2, 0.1, 53), param='array')
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    a.ptk(df=jdi.hot_jupiter())
    all_out = jdi.compute(a, directory='.')
    assert np.isclose(np.sum(all_out['condensate_mmr']), 5.7569695488823735e-05)
    assert np.isclose(np.sum(all_out['mean_particle_r']), 250.514075712179)
    assert np.isclose(np.sum(np.nan_to_num(all_out['column_density'])), 643.8594768055566)

