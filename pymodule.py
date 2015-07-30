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

def run_cdft(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    CDFT can be called via :py:func:`~driver.energy`.

    >>> energy('cdft')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Run CDFT
    psi4.set_local_option('CDFT','METHOD','CDFT')
    returnvalue = psi4.plugin('cdft.so')

    return returnvalue

def run_ocdft(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    OCDFT can be called via :py:func:`~driver.energy`.

    >>> energy('ocdft')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Run OCDFT
    psi4.set_local_option('CDFT','METHOD','OCDFT')
    returnvalue = psi4.plugin('cdft.so')

    return returnvalue

def run_fasnocis(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    OCDFT can be called via :py:func:`~driver.energy`.

    >>> energy('fasnocis')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Run OCDFT
    psi4.set_local_option('CDFT','METHOD','FASNOCIS')
    returnvalue = psi4.plugin('cdft.so')

    return returnvalue


# Integration with driver routines
procedures['energy']['cdft'] = run_cdft
procedures['energy']['ocdft'] = run_ocdft
procedures['energy']['fasnocis'] = run_fasnocis
