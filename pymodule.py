import psi4
import re
import os
import inputparser
import math
import warnings
from driver import *
from wrappers import *
from molutil import *
import p4util
from p4xcpt import *

plugdir = os.path.split(os.path.abspath(__file__))[0]
sofile = plugdir + "/cdft.so"

def run_cdft(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    CDFT can be called via :py:func:`~driver.energy`.

    >>> energy('cdft')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    scf_molecule = kwargs.get('molecule', psi4.get_active_molecule())
    ref_wfn = psi4.new_wavefunction(scf_molecule, psi4.get_global_option('BASIS'))

    if ref_wfn is None:
        #ref_wfn = scf_helper(name, **kwargs)
	psi4.set_local_option('CDFT','METHOD','CDFT')
        returnvalue = psi4.plugin(sofile,ref_wfn)

    # Run CDFT
    #psi4.set_local_option('CDFT','METHOD','CDFT')
    #returnvalue = psi4.plugin(sofile,ref_wfn)

    return returnvalue


def run_noci(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    NOCI can be called via :py:func:`~driver.energy`.

    >>> energy('noci')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    scf_molecule = kwargs.get('molecule', psi4.get_active_molecule())
    ref_wfn = psi4.new_wavefunction(scf_molecule, psi4.get_global_option('BASIS'))

    if ref_wfn is None:
        ref_wfn = scf_helper(name, **kwargs)

    # Run CDFT
    psi4.set_local_option('CDFT','METHOD','NOCI')
    returnvalue = psi4.plugin(sofile,ref_wfn)

    return returnvalue




def run_ocdft(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    OCDFT can be called via :py:func:`~driver.energy`.

    >>> energy('ocdft')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    scf_molecule = kwargs.get('molecule', psi4.get_active_molecule())
    ref_wfn = psi4.new_wavefunction(scf_molecule, psi4.get_global_option('BASIS'))

    if ref_wfn is None:
        ref_wfn = scf_helper(name, **kwargs)

    # Run OCDFT
    psi4.set_local_option('CDFT','METHOD','OCDFT')
    returnvalue = psi4.plugin(sofile, ref_wfn)

    return returnvalue

def run_fasnocis(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    OCDFT can be called via :py:func:`~driver.energy`.

    >>> energy('fasnocis')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    scf_molecule = kwargs.get('molecule', psi4.get_active_molecule())
    ref_wfn = psi4.new_wavefunction(scf_molecule, psi4.get_global_option('BASIS'))


    if ref_wfn is None:
        ref_wfn = scf_helper(name, **kwargs)

    # Run OCDFT
    psi4.set_local_option('CDFT','METHOD','FASNOCIS')
    returnvalue = psi4.plugin(sofile,ref_wfn)

    return returnvalue


# Integration with driver routines
procedures['energy']['cdft'] = run_cdft
procedures['energy']['ocdft'] = run_ocdft
procedures['energy']['fasnocis'] = run_fasnocis
procedures['energy']['noci'] = run_noci
