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
    explicit FASNOCIS(Options &options, boost::shared_ptr<PSIO> psio);
    explicit FASNOCIS(Options &options, boost::shared_ptr<PSIO> psio,
                      boost::shared_ptr<Wavefunction> ref_scf,
                      std::vector<std::pair<int,int>> active_mos,
                      std::vector<int> aocc,
                      std::vector<int> bocc);
    virtual ~FASNOCIS();
protected:

    boost::shared_ptr<Wavefunction> ref_scf_;
    std::vector<std::pair<int,int>> active_mos_;
    std::vector<int> aocc_;
    std::vector<int> bocc_;

    virtual void form_C();
};

}} // Namespaces

#endif // Header guard
