#include <libmints/mints.h>

#include "determinant.h"

using namespace psi;

namespace psi{
namespace scf{

Determinant::Determinant(double energy, SharedMatrix Ca, SharedMatrix Cb, const Dimension& nalphapi, const Dimension& nbetapi)
    : energy_(energy),nalphapi_(nalphapi), nbetapi_(nbetapi), Ca_(Ca->clone()), Cb_(Cb->clone())
{
}


Determinant::Determinant(SharedMatrix Ca, SharedMatrix Cb, const Dimension& nalphapi, const Dimension& nbetapi,int fmo,
    int state_a, int state_b,int vir_frozen)
    :energy_(0.0),nalphapi_(nalphapi), nbetapi_(nbetapi), Ca_(Ca->clone()), Cb_(Cb->clone()),o_fmo(fmo),v_fmoA(state_a),
    v_fmoB(state_b),vir_fmo(vir_frozen)
{
}

Determinant::Determinant(SharedMatrix Ca, SharedMatrix Cb, const Dimension& nalphapi, const Dimension& nbetapi)
    : energy_(0.0), nalphapi_(nalphapi), nbetapi_(nbetapi), Ca_(Ca->clone()), Cb_(Cb->clone())
{
}

Determinant::Determinant(const Determinant& det)
{
    energy_ = det.energy_;
    Ca_ = det.Ca_->clone();
    Cb_ = det.Cb_->clone();
    nalphapi_ = det.nalphapi_;
    nbetapi_ = det.nbetapi_;

    o_fmo=0;
    v_fmoA=0;
    v_fmoB=0;
    vir_fmo=0;

}

Determinant::~Determinant()
{}

int Determinant::symmetry()
{
    int symm = 0;
    int nirrep = nalphapi_.n();
    for (int h = 0; h < nirrep; ++h){
        // Check if there is an odd number of electrons in h
        if( std::abs(nalphapi_[h] - nbetapi_[h]) % 2 == 1){
            symm ^= h;
        }
    }
    return symm;
}

void Determinant::print()
{
    nalphapi_.print();
    nbetapi_.print();
}


void Determinant::occup()
{
  outfile->Printf("\n|");
  outfile->Printf(">\n");
}


}} // namespaces
