import psi4
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util

def run_cdft(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    CDFT can be called via :py:func:`~driver.energy`.

    >>> energy('ocdft')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    scf_molecule = kwargs.get('molecule', psi4.get_active_molecule())
    ref_wfn = psi4.new_wavefunction(scf_molecule, psi4.get_global_option('BASIS'))

    if ref_wfn is None:
        ref_wfn = scf_helper(name, **kwargs)

    if (psi4.core.get_option('OCDFT','MINAO_BASIS')):
        minao_basis = psi4.core.BasisSet.build(ref_wfn.molecule(), 'MINAO_BASIS',
                                                   psi4.core.get_option('FORTE','MINAO_BASIS'))
        ref_wfn.set_basisset('MINAO_BASIS', minao_basis)

    # Run CDFT
    psi4.set_local_option('OCDFT','METHOD','CDFT')
    returnvalue = psi4.plugin(sofile,ref_wfn)

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

    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # Run OCDFT
    psi4.core.set_local_option('OCDFT','METHOD','OCDFT')
    if (psi4.core.get_option('OCDFT','MINAO_BASIS')):
        minao_basis = psi4.core.BasisSet.build(ref_wfn.molecule(), 'MINAO_BASIS',
                                                   psi4.core.get_option('OCDFT','MINAO_BASIS'))
        ref_wfn.set_basisset('MINAO_BASIS', minao_basis)

    # Call the Psi4 plugin
    # Please note that setting the reference wavefunction in this way is ONLY for plugins

    ocdft_wfn = psi4.core.plugin('ocdft.so', ref_wfn)

    return ocdft_wfn

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
psi4.driver.procedures['energy']['cdft'] = run_cdft
psi4.driver.procedures['energy']['ocdft'] = run_ocdft
psi4.driver.procedures['energy']['fasnocis'] = run_fasnocis
psi4.driver.procedures['energy']['noci'] = run_noci
