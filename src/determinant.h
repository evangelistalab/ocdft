#ifndef DETERMINANT_H
#define DETERMINANT_H

//#include <psi4/libscf_solver/ks.h>
#include <psi4/libmints/matrix.h>
//#include <psi4/libmints/matrix.h>

namespace psi{
namespace scf{

class Determinant
{
public:
    Determinant(SharedMatrix Ca, SharedMatrix Cb, const Dimension& nalphapi, const Dimension& nbetapi,int fmo,
    int state_a, int state_b,int vir_frozen);

    Determinant(double energy, SharedMatrix Ca, SharedMatrix Cb, const Dimension &nalphapi, const Dimension &nbetapi);
    Determinant(SharedMatrix Ca, SharedMatrix Cb, const Dimension &nalphapi, const Dimension &nbetapi);
    Determinant(const Determinant& det);
    ~Determinant();
    double energy() {return energy_;}
    SharedMatrix Ca() {return Ca_;}
    SharedMatrix Cb() {return Cb_;}
    const Dimension& nalphapi() {return nalphapi_;}
    const Dimension& nbetapi() {return nbetapi_;}
    int symmetry();
    void print();
    void occup();
private:
    double energy_;
    Dimension nalphapi_;
    Dimension nbetapi_;
    SharedMatrix Ca_;
    SharedMatrix Cb_;

   int vir_fmo;
   int o_fmo;
   int v_fmoA;
   int v_fmoB;
};

}
typedef std::shared_ptr<psi::scf::Determinant> SharedDeterminant;
} // Namespaces

#endif // Header guard
