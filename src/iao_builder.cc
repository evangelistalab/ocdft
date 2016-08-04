/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2016 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include <libqt/qt.h>
#include <libmints/mints.h>
#include "iao_builder.h"

using namespace boost;
using namespace psi;

namespace psi {

IAOBuilder::IAOBuilder(
    boost::shared_ptr<BasisSet> primary, 
    boost::shared_ptr<BasisSet> minao, 
    boost::shared_ptr<Matrix> C) :
    primary_(primary),
    minao_(minao),
    C_(C)
{
    if (C->nirrep() != 1) {
        throw PSIEXCEPTION("Localizer: C matrix is not C1");
    }
    if (C->rowspi()[0] != primary->nbf()) {
        throw PSIEXCEPTION("Localizer: C matrix does not match basis");
    }
    common_init();
}
IAOBuilder::~IAOBuilder()
{
}
void IAOBuilder::common_init()
{
    print_ = 0;
    debug_ = 0;
    bench_ = 0;
    convergence_ = 1.0E-12;
    maxiter_ = 50;
    use_ghosts_ = false;
    power_ = 4;
    condition_ = 1.0E-7; 
    use_stars_ = false;
    stars_completeness_ = 0.9;
    stars_.clear();
}
boost::shared_ptr<IAOBuilder> IAOBuilder::build(
    boost::shared_ptr<BasisSet> primary, 
    boost::shared_ptr<Matrix> C,
    Options& options)
{
//    Options& options = Process::environment.options;    

    boost::shared_ptr<BasisSet> minao = BasisSet::pyconstruct_orbital(primary->molecule(),
        "BASIS", options.get_str("MINAO_BASIS"));

    boost::shared_ptr<IAOBuilder> local(new IAOBuilder(primary, minao, C)); 

    local->set_print(options.get_int("PRINT"));
    local->set_debug(options.get_int("DEBUG"));
    local->set_bench(options.get_int("BENCH"));
    local->set_convergence(options.get_double("LOCAL_CONVERGENCE"));
    local->set_maxiter(options.get_int("LOCAL_MAXITER"));
    local->set_use_ghosts(options.get_bool("LOCAL_USE_GHOSTS"));
    local->set_condition(options.get_double("LOCAL_IBO_CONDITION"));
    local->set_power(options.get_double("LOCAL_IBO_POWER"));
    local->set_use_stars(options.get_bool("LOCAL_IBO_USE_STARS"));
    local->set_stars_completeness(options.get_double("LOCAL_IBO_STARS_COMPLETENESS"));

    std::vector<int> stars;
    for (int ind = 0; ind < options["LOCAL_IBO_STARS"].size(); ind++) {
        stars.push_back(options["LOCAL_IBO_STARS"][ind].to_integer() - 1);
    }
    local->set_stars(stars);
    
    return local;
}

std::map<std::string, SharedMatrix> IAOBuilder::build_iaos()
{
    // => Ghosting <= //
    boost::shared_ptr<Molecule> mol = minao_->molecule();
    true_atoms_.clear();
    true_iaos_.clear();
    iaos_to_atoms_.clear();
    for (int A = 0; A < mol->natom(); A++) {
        if (!use_ghosts_ && mol->Z(A) == 0.0) continue;
        int Atrue = true_atoms_.size(); 
        int nPshells = minao_->nshell_on_center(A);
        int sPshells = minao_->shell_on_center(A, 0);        
        for (int P = sPshells; P < sPshells + nPshells; P++) {
            int nP = minao_->shell(P).nfunction();
            int oP = minao_->shell(P).function_index();
            for (int p = 0; p < nP; p++) {
                true_iaos_.push_back(p + oP);
                iaos_to_atoms_.push_back(Atrue);
            }
        }
        true_atoms_.push_back(A);
    }

    // => Overlap Integrals <= //

    boost::shared_ptr<IntegralFactory> fact11(new IntegralFactory(primary_,primary_));
    boost::shared_ptr<IntegralFactory> fact12(new IntegralFactory(primary_,minao_));
    boost::shared_ptr<IntegralFactory> fact22(new IntegralFactory(minao_,minao_));

    boost::shared_ptr<OneBodyAOInt> ints11(fact11->ao_overlap());
    boost::shared_ptr<OneBodyAOInt> ints12(fact12->ao_overlap());
    boost::shared_ptr<OneBodyAOInt> ints22(fact22->ao_overlap());

    boost::shared_ptr<Matrix> S11(new Matrix("S11", primary_->nbf(), primary_->nbf()));
    boost::shared_ptr<Matrix> S12f(new Matrix("S12f", primary_->nbf(), minao_->nbf()));
    boost::shared_ptr<Matrix> S22f(new Matrix("S22f", minao_->nbf(), minao_->nbf()));

    ints11->compute(S11); 
    ints12->compute(S12f); 
    ints22->compute(S22f); 

    ints11.reset();
    ints12.reset();
    ints22.reset();

    fact11.reset();
    fact12.reset();
    fact22.reset();

    // => Ghosted Overlap Integrals <= //

    boost::shared_ptr<Matrix> S12(new Matrix("S12", primary_->nbf(), true_iaos_.size()));
    boost::shared_ptr<Matrix> S22(new Matrix("S22", true_iaos_.size(), true_iaos_.size()));
    
    double** S12p  = S12->pointer();
    double** S12fp = S12f->pointer();
    for (int m = 0; m < primary_->nbf(); m++) {
        for (int p = 0; p < true_iaos_.size(); p++) {
            S12p[m][p] = S12fp[m][true_iaos_[p]];
        }
    }
    
    double** S22p  = S22->pointer();
    double** S22fp = S22f->pointer();
    for (int p = 0; p < true_iaos_.size(); p++) {
        for (int q = 0; q < true_iaos_.size(); q++) {
            S22p[p][q] = S22fp[true_iaos_[p]][true_iaos_[q]];

        }
    }

    // => Metric Inverses <= //

    boost::shared_ptr<Matrix>S11_m12(S11->clone());
    boost::shared_ptr<Matrix>S22_m12(S22->clone());
    S11_m12->copy(S11);
    S22_m12->copy(S22);
    S11_m12->power(-1.0/2.0, condition_);
    S22_m12->power(-1.0/2.0, condition_);

    // => Tilde C <= //

    boost::shared_ptr<Matrix> C = C_;
    boost::shared_ptr<Matrix> T1 = Matrix::doublet(S22_m12, S12, false, true);
    boost::shared_ptr<Matrix> T2 = Matrix::doublet(S11_m12, Matrix::triplet(T1, T1, C, true, false, false), false, false);
    boost::shared_ptr<Matrix> T3 = Matrix::doublet(T2, T2, true, false);
    T3->power(-1.0/2.0, condition_);
    boost::shared_ptr<Matrix> Ctilde = Matrix::triplet(S11_m12, T2, T3, false, false, false);

    // => D and Tilde D <= //

    boost::shared_ptr<Matrix> D = Matrix::doublet(C, C, false, true);
    boost::shared_ptr<Matrix> Dtilde = Matrix::doublet(Ctilde, Ctilde, false, true);

    // => A (Before Orthogonalization) <= //

    boost::shared_ptr<Matrix> DSDtilde = Matrix::triplet(D, S11, Dtilde,false, false, false);
    DSDtilde->scale(2.0);
    
    boost::shared_ptr<Matrix> L = Matrix::doublet(S11_m12, S11_m12, false, false); // TODO: Possibly Unstable
    L->add(DSDtilde);
    L->subtract(D);
    L->subtract(Dtilde);
    
    boost::shared_ptr<Matrix> AN = Matrix::doublet(L, S12, false, false);
    
    // => A (After Orthogonalization) <= //

    boost::shared_ptr<Matrix> V = Matrix::triplet(AN, S11, AN, true, false, false);
    V->power(-1.0/2.0, condition_);

    boost::shared_ptr<Matrix> A = Matrix::doublet(AN, V, false, false);

    // => Assignment <= //

    S_ = S11;
    A_ = A;

    SharedMatrix Acoeff(A->clone());
    SharedMatrix S_min(S22->clone());
    std::map<std::string, SharedMatrix > ret; 
    ret["A"] = Acoeff;
    ret["S_min"] = S_min;

    //ret["A"] = set_name("A")
    //ret["S_min"] = set_name("S_min")
    
    return ret;
}

} // Namespace psi
