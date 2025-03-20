""" Functions to generate a cloud structure by iterating over fsed calculations """
import numpy as np
from .justdoit import compute

def compute_iterator(atmo, directory=None, as_dict=True, og_solver=True,
                     direct_tol=1e-15, refine_TP=True, og_vfall=True, analytical_rg=True,
                     do_virtual=True, mixed=False):
    """ TODO """

    # set up working variables
    mmr_old = None
    memory = []
    converged = False
    fsed_in = atmo.fsed

    while not converged:
        # compute cloud structure with given fsed_in
        all_out = compute(atmo, directory, as_dict, og_solver, direct_tol, refine_TP,
                          og_vfall, analytical_rg, do_virtual, mixed)

        # remember output
        memory.append(all_out)

        # get cloud particle mmrs
        mmr = all_out['condensate_mmr']

        # adjust fsed
        # TODO

        # check if there is already an old mmr to compare, otherwise start over
        if isinstance(mmr_old, type(None)):
            mmr_old = mmr  # remember current run
            continue
        # Compare to previous value and see if deviations are within tolerance
        error = np.abs((mmr[mmr>0] - mmr_old[mmr>0])/mmr[mmr>0])
        if (error < 1e-3).all():
            break  # stop iteration
        mmr_old = mmr  # remember current run




    print('YAAAY')
    exit()