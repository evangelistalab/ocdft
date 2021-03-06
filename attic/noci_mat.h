#ifndef _noci_hamiltonian_h_
#define _noci_hamiltonian_h_

#include "boost/tuple/tuple.hpp"

#include "determinant.h"

namespace psi{
class Options;
 namespace scf{

class NOCI_Hamiltonian{
public:

    // ==> Class Interface <==

    /// Constructor
    explicit NOCI_Hamiltonian(Options &options, std::vector<SharedDeterminant> dets);    
    /// Diagonalize the Hamiltonian in the basis of determinants
    double compute_energy(std::vector<double> energies);
    /// Return the eigenvalues
    SharedVector evals() {return evals_;}
    /// Return the eigenvectors
    SharedMatrix evecs() {return evecs_;}

protected:

    // ==> Class Private Functions <==

    /// Read the two-electron integrals from disk
    void read_tei();
    /// Compute the Coulomb operator
    void J(SharedMatrix D);
    /// Compute the Exchange operator
    void K(SharedMatrix D);
    /// Compute the Coulomb and Exchange operators using PSI4's JK builder
    void fast_JK(SharedMatrix Cl,SharedMatrix Cr);
    /// Compute W, the averaged density matrix
    SharedMatrix build_W_c1(SharedMatrix CA, SharedMatrix CB, SharedVector s, size_t nocc);
    /// Compute D_i, the one-orbital transition density matrix
    SharedMatrix build_D_i_c1(SharedMatrix CA, SharedMatrix CB, size_t i, size_t j);
    /// Compute the left/right C matrix necessary to bild W, the averaged density matrix
    void build_W_JK_c1(SharedMatrix CA, SharedMatrix CB, SharedVector s, size_t n);
    void build_D_i_JK_c1(SharedMatrix CA, SharedMatrix CB, size_t i);


    /// Compute the matrix element between determinants A and B assuming C1 symmetry
    std::vector<double> matrix_element_c1(SharedDeterminant A, SharedDeterminant B);
    /// Compute the matrix elements of a vector of one-body operators between determinants A and B assuming C1 symmetry
    std::vector<double> matrix_element_one_body_c1(SharedDeterminant A, SharedDeterminant B, std::vector<SharedMatrix> Ops);
    /// Compute the corresponding orbitals between determinant A and B
    std::tuple<SharedMatrix,SharedMatrix,SharedVector,double>
    corresponding_orbitals(SharedMatrix A, SharedMatrix B, Dimension dima, Dimension dimb);

    /// The number of irreps
    int nirrep_;
    /// The number of symmetry-adapted orbitals
    size_t nso;
    /// Use PSI4's JK object
    bool use_fast_jk_ = false;

    /// Matrix factory
    std::shared_ptr<Molecule> molecule_;
    /// Matrix factory
    std::shared_ptr<MatrixFactory> factory_;
    /// The integral factory
    std::shared_ptr<IntegralFactory> integral_;
    /// Number of symmetry adapted AOs per irrep
    Dimension nsopi_;
    /// The one-electron integrals
    SharedMatrix Hso_;
    /// The overalp integrals
    SharedMatrix Sso_;
    SharedMatrix Jso_libfock_;
    SharedMatrix Kso_libfock_;
    /// The two-electron integrals
    std::vector<double> tei_ints_;
    /// The JK builder object
    std::shared_ptr<JK> jk_;
    /// The nuclear repulsion energy
    double nuclearrep_;
    /// Temporary matrices
    SharedMatrix TempMatrix, TempMatrix2;

    Options& options_;
    std::vector<SharedDeterminant> dets_;
    /// The Hamiltonian matrix
    SharedMatrix H_;
    /// The overlap matrix
    SharedMatrix S_;
    /// The S^2 operator
    SharedMatrix S2_;
    SharedMatrix evecs_;
    SharedVector evals_;
};

}} // Namespaces

#endif // Header guard
