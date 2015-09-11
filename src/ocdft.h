#ifndef _ocdft_h_
#define _ocdft_h_

#include "boost/tuple/tuple.hpp"

#include <libscf_solver/ks.h>

#include "determinant.h"

namespace psi{
class Options;
namespace scf{

/// A class for unrestricted Orthogonality Constrained DFT
class UOCDFT : public UKS {
public:
    explicit UOCDFT(Options &options, boost::shared_ptr<PSIO> psio);
    explicit UOCDFT(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<Wavefunction> ref_scf, int state);
    explicit UOCDFT(Options &options, boost::shared_ptr<PSIO> psio, boost::shared_ptr<Wavefunction> ref_scf, int state, int symmetry);
    virtual ~UOCDFT();

    double singlet_exc_energy_s_plus() {return singlet_exc_energy_s_plus_;}
    double oscillator_strength_s_plus() {return oscillator_strength_s_plus_;}

protected:
    /// The fragment constraint matrices in the SO basis
    std::vector<SharedMatrix> W_frag;

    // Excitation energy
    /// Compute an excited state?
    bool do_excitation;
    /// Compute a state of a given symmetry?
    bool do_symmetry;
    /// Optimized the holes
    bool do_holes;
    /// Optimized the particles
    bool do_parts;
    /// Optimize the spectators
    bool do_opt_spectators;
    /// For multiple excited state project out previous holes
    bool do_project_out_holes;
    /// For multiple excited state project out previous particles
    bool do_project_out_particles;
    /// For multiple excited state project out previous holes
    bool do_save_holes;
    /// For multiple excited state project out previous particles
    bool do_save_particles;
    /// Ground state energy
    double ground_state_energy;
    /// Ground state symmetry
    int ground_state_symmetry_;
    /// Symmetry of the excited state, if specified
    int excited_state_symmetry_;
    /// Excited state number, starting from one
    int state_;

    double singlet_exc_energy_s_plus_;
    double triplet_exc_energy_s_plus;
    double singlet_exc_energy_ci;
    double triplet_exc_energy_ci;
    double oscillator_strength_s_plus_;
    double oscillator_strength_ci;

    // Information about the excited states
    /// Determinant information for each electronic state
    std::vector<SharedDeterminant> dets;

    /// The Fock matrix projected onto the spectator space
    SharedMatrix QFQ_;
    /// The hole orbitals coefficients
    SharedMatrix Ch_;
    /// The particle orbitals coefficients
    SharedMatrix Cp_;
    /// The holes
    SharedMatrix saved_Ch_;
    /// The particles
    SharedMatrix saved_Cp_;
    /// The effective alpha Fock matrix in the MO basis
    SharedMatrix moFeffa_;
    /// The effective beta Fock matrix in the MO basis
    SharedMatrix moFeffb_;
    /// The Fock matrix projected onto the occupied space
    SharedMatrix PoFaPo_;
    /// The Fock matrix projected onto the virtual space
    SharedMatrix PvFaPv_;
    /// The eigenvectors of PoFPo
    SharedMatrix Ua_o_;
    /// The eigenvectors of PvFPv
    SharedMatrix Ua_v_;
    /// The eigenvalues of PoFPo
    SharedVector lambda_a_o_;
    /// The eigenvalues of PvFPv
    SharedVector lambda_a_v_;
    /// The Fock matrix projected onto the occupied space
    SharedMatrix PoFbPo_;
    /// The Fock matrix projected onto the virtual space
    SharedMatrix PvFbPv_;
    /// The eigenvectors of PoFPo
    SharedMatrix Ub_o_;
    /// The eigenvectors of PvFPv
    SharedMatrix Ub_v_;
    /// The eigenvalues of PoFPo
    SharedVector lambda_b_o_;
    /// The eigenvalues of PvFPv
    SharedVector lambda_b_v_;

    /// The list of alpha holes (irrep,mo,energy)
    std::vector<boost::tuple<int,int,double> > aholes;
    /// The list of alpha particles (irrep,mo,energy)
    std::vector<boost::tuple<int,int,double> > aparts;
    /// The ground state alpha occupation numbers per irrep
    Dimension gs_nalphapi_;
    /// The ground state alpha virtual mos per irrep
    Dimension gs_navirpi_;
    /// The ground state alpha occupation numbers per irrep
    Dimension gs_nbetapi_;
    /// The ground state alpha virtual mos per irrep
    Dimension gs_nbvirpi_;
    /// The number of alpha holes saved from previous computations
    Dimension saved_naholepi_;
    /// The number of alpha particles saved from previous computations
    Dimension saved_napartpi_;
    /// The number of alpha holes to project out
    Dimension project_naholepi_;
    /// A dimension vector with all zeros
    Dimension zero_dim_;

    /// Number of alpha holes per irrep
    Dimension naholepi_;
    /// Number of alpha particles per irrep
    Dimension napartpi_;

    /// The ground state alpha Fock matrix
    SharedMatrix gs_Fa_;
    /// The ground state beta Fock matrix
    SharedMatrix gs_Fb_;

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

    /// A copy of the one-electron potential
    SharedMatrix H_copy;
    /// Flag to save the one-electron part of the Hamiltonian
    bool save_H_;
    /// A shift to apply to the virtual orbitals to improve convergence
    double level_shift_;

    int nW_opt;

    // UKS specific functions
    /// Class initializer
    void init();
    /// Initialize the exctiation functions
    void init_excitation( boost::shared_ptr<Wavefunction> ref_scf);
    /// The constrained hole/particle algorithm for computing the orbitals
    void form_C_ee();
    /// Finds the optimal holes
    void compute_holes();
    /// Finds the optimal particles
    void compute_particles();
    /// Finds the optimal hole and particle pair
    void find_ee_occupation(SharedVector lambda_o,SharedVector lambda_v);
    ///
    void compute_hole_particle_mos();
    /// Form the Fock matrix for the spectator orbitals
    void diagonalize_F_spectator_relaxed();
    /// Form the Fock matrix for the spectator orbitals
    void diagonalize_F_spectator_unrelaxed();
    void sort_ee_mos();

    /// Analyze excitations
    void analyze_excitations();
    /// Compute the transition dipole moment between the ground and excited states
    void compute_transition_moments();
    /// Compute a correction for the mixed excited states
    double compute_triplet_correction();
    /// Compute a correction for the mixed excited state based on a triplet state generated by acting with the S+ operator
    double compute_S_plus_triplet_correction();
    /// Compute the singlet and triplet energy of a mixed excited state
    void spin_adapt_mixed_excitation();
    /// Compute the CIS excitation energy
    void cis_excitation_energy();
    /// Compute the corresponding orbitals for a pair of MO sets
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double> corresponding_orbitals(SharedMatrix A, SharedMatrix B, Dimension dima, Dimension dimb);
    /// Form_C for the beta MOs
    void form_C_beta();
    /// Checks if the orbital defined by the matrix are orthogonal with respect to the metric S
    void orthogonality_check(SharedMatrix C, SharedMatrix S);

    std::pair<double,double> matrix_element(SharedDeterminant A, SharedDeterminant B);

    // Overloaded UKS function
    virtual void save_density_and_energy();
    virtual void form_F();
    virtual void form_C();
    virtual double compute_E();
    virtual void damp_update();
    virtual bool test_convergency();
    virtual void guess();
    virtual void save_information();
    virtual void compute_orbital_gradient(bool save_fock);
};

}} // Namespaces

#endif // Header guard
