#ifndef _noci_hamiltonian_h_
#define _noci_hamiltonian_h_

#include "boost/tuple/tuple.hpp"

#include "determinant.h"

namespace psi{
class Options;
 namespace scf{

class NOCI_Hamiltonian{
public:
    explicit NOCI_Hamiltonian(Options &options, std::vector<SharedDeterminant> dets);

    double compute_energy();

    ~NOCI_Hamiltonian();
    void print();
protected:
    /// Read the two-electron integrals
    void read_tei();
    /// Compute the Coulomb operator
    void J(SharedMatrix D);
    /// Compute the Exchange operator
    void K(SharedMatrix D);

    /// Compute the matrix element between determinants A and B assuming C1 symmetry
    std::pair<double,double> matrix_element_c1(SharedDeterminant A, SharedDeterminant B);
//    /// Compute the matrix element between determinants A and B
//    std::pair<double,double> matrix_element(SharedDeterminant A, SharedDeterminant B);
    /// Compute the corresponding orbitals between determinant A and B
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double>
    corresponding_orbitals(SharedMatrix A, SharedMatrix B, Dimension dima, Dimension dimb);

    int nirrep_;
    boost::shared_ptr<BasisSet> basisset_;

    int nso;

    /// Matrix factory
    boost::shared_ptr<MatrixFactory> factory_;
    /// Number of symmetry adapted AOs per irrep
    Dimension nsopi_;
    /// The one-electron integrals
    SharedMatrix Hao_;
    /// The overalp integrals
    SharedMatrix Sao_;
    /// The two-electron integrals
    std::vector<double> tei_ints_;
    boost::shared_ptr<JK> jk_;
    /// The nuclear repulsion energy
    double nuclearrep_;
    /// Temporary matrices
    SharedMatrix TempMatrix, TempMatrix2;

    Options& options_;
    std::vector<SharedDeterminant> dets_;
    SharedMatrix H_;
    SharedMatrix S_;
    SharedMatrix evecs_;
    SharedVector evals_;
};

}} // Namespaces

#endif // Header guard
