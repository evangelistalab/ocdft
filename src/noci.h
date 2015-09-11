#ifndef _noci_h_
#define _noci_h_

#include "boost/tuple/tuple.hpp"

#include <libscf_solver/hf.h>

#include "determinant.h"

namespace psi{
class Options;
namespace scf{


class NOCI : public UHF {
public:
    explicit NOCI(Options &options, boost::shared_ptr<PSIO> psio);
    explicit NOCI(Options &options, boost::shared_ptr<PSIO> psio, int state_a,std::pair<int,int> fmo, int state_b, std::vector<std::pair<int,int>> frozen_occ_a,std::vector<std::pair<int,int>> frozen_occ_b, std::vector<std::pair<int,int>> frozen_mos,
                  std::vector<int> occ_frozen,
                  std::vector<int> vir_frozen, SharedMatrix Ca_gs, SharedMatrix Cb_gs,bool valence_in);
    virtual ~NOCI();

protected:

      bool valence;
     bool do_noci;
     bool do_alpha_states;
     int state_a_;
     std::pair<int,int> fmo_;
     int state_b_;
     int occ_;

     std::vector<std::pair<int,int>> frozen_mos_;
     std::vector<std::pair<int,int>> frozen_occ_a_;
     std::vector<std::pair<int,int>> frozen_occ_b_;
     std::vector<int> occ_frozen_;
     std::vector<int> vir_frozen_;


    Dimension zero_dim_;
//    Dimension saved_alpha_;
//    Dimension saved_beta_;



    double ground_state_energy;


    /// A temporary matrix
    SharedMatrix TempMatrix;
    /// A temporary matrix
    SharedMatrix TempMatrix2;
    /// A temporary vector
    SharedVector TempVector;

    /// The old alpha density matrix
    SharedMatrix Dolda_;
    /// The old beta density matrix
    SharedMatrix Doldb_;

    SharedMatrix Fpq_a;
    SharedMatrix Fpq_b;

     SharedMatrix Fpq;
     SharedMatrix Dt_diff;

    double den_diff;


    SharedMatrix Cocc_bN;
    SharedMatrix Cvrt_bN;

     SharedMatrix Ua_;
     SharedMatrix Ub_;

     SharedMatrix Ca0;
     SharedMatrix Cb0;


     SharedMatrix Ca_gs;
     SharedMatrix Cb_gs;



     SharedMatrix U_fmo;
     SharedMatrix U_swap;

     SharedMatrix Ca_fmo;
     SharedMatrix Cb_fmo;
     SharedMatrix rFpq_a;
     SharedMatrix rFpq_b;
     SharedMatrix rFpq;
     SharedMatrix rUa_;
     SharedMatrix rUb_;

      SharedVector Repsilon_a_;

      SharedMatrix rCa0;
      SharedMatrix rCb0;

      SharedMatrix rCa_;
      SharedMatrix rCb_;

     SharedMatrix Ft_;
    /// A copy of the one-electron potential

    /// Flag to save the one-electron part of the Hamiltonian


    // UKS specific functions
    /// Class initializer
    void init();
    void init_excitation();
    void form_C_noci();


    // Overloaded UKS function
    virtual void form_C();
 virtual bool test_convergency();
    virtual void guess();

};

}} // Namespaces

#endif // Header guard
