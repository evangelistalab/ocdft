#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include <libscf_solver/ks.h>

namespace psi{
namespace scf{

class Constraint
{
public:
    Constraint(SharedMatrix W_so,double Nc,double weight_alpha,double weight_beta,std::string type);
    SharedMatrix W_so() {return W_so_;}
    double Nc() {return Nc_;}
    double weight_alpha() {return weight_alpha_;}
    double weight_beta() {return weight_beta_;}
    std::string type() {return type_;}
protected:
    /// The alpha constraint matrices in the SO basis
    SharedMatrix W_so_;
    /// The value of the constraint
    double Nc_;
    /// Weight of the alpha constraint
    double weight_alpha_;
    /// Weight of the beta constraint
    double weight_beta_;
    /// The type of constraint
    std::string type_;
};

}
typedef boost::shared_ptr<psi::scf::Constraint> SharedConstraint;
} // Namespaces

#endif // Header guard
