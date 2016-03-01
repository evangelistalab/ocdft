#ifndef SRC_LIB_FASNOCIS_H
#define SRC_LIB_FASNOCIS_H

#include "boost/tuple/tuple.hpp"

#include <libscf_solver/hf.h>

#include "constraint.h"
#include "determinant.h"

namespace psi{
class Options;
namespace scf{

/// A class for unrestricted constrained Kohn-Sham theory
class FASNOCIS : public UHF {
public:
    explicit FASNOCIS(SharedWavefunction ref_scf, Options &options, boost::shared_ptr<PSIO> psio);
    explicit FASNOCIS(Options &options, boost::shared_ptr<PSIO> psio,
                      SharedWavefunction ref_scf,
                      std::vector<std::pair<size_t,size_t>> vec_frozen_mos,
                      std::vector<size_t> aocc,
                      std::vector<size_t> bocc, Dimension nadoccpi);
    virtual ~FASNOCIS();
protected:

    boost::shared_ptr<Wavefunction> ref_scf_;
    /// Vector of frozen orbitals stored as pairs of (irrep,rel mo)
    std::vector<std::pair<size_t,size_t>> vec_frozen_mos_;
    /// Vector of active orbitals stored as pairs of (irrep,rel mo)
    std::vector<std::pair<size_t,size_t>> vec_active_mos_;
    std::vector<size_t> afocc_;
    std::vector<size_t> bfocc_;
    std::vector<size_t> afvir_;
    std::vector<size_t> bfvir_;


    /// Number of frozen MOs per irrep
    Dimension nfmopi_;
    /// Number of active MOs per irrep
    Dimension namopi_;
    /// Number of doubly occupied active MOs per irrep
    Dimension nadoccpi_;
    /// Number of unoccupied active MOs per irrep
    Dimension nauoccpi_;

    SharedMatrix Ca_f_;
    SharedMatrix Cb_f_;
    SharedMatrix Ca_a_;
    SharedMatrix Cb_a_;

    /// The total Fock matrix in the SO basis
    SharedMatrix Ft_;
    SharedMatrix Ft_amo_;
    SharedMatrix U_amo_;
    SharedVector lambda_amo_;

    bool do_excitation;

    /// Form the frozen part of C
    void form_C_noscf();

    void form_C_frozen();
    void form_C_active();
    void combine_C_frozen_active();


    virtual void form_C();
    virtual double compute_E();
    virtual bool diis();
};

}} // Namespaces

#endif // Header guard
