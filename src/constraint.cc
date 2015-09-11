#include "constraint.h"

using namespace psi;

namespace psi{ namespace scf{

Constraint::Constraint(SharedMatrix W_so,double Nc,double weight_alpha,double weight_beta,std::string type)
    : W_so_(W_so), Nc_(Nc), weight_alpha_(weight_alpha), weight_beta_(weight_beta), type_(type)
{
}

}} // Namespaces
