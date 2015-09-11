#include "ucks.h"
#include <physconst.h>
#include <libmints/view.h>
#include <libmints/mints.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <libdisp/dispersion.h>
#include <liboptions/liboptions.h>
#include <libciomr/libciomr.h>
#include <libqt/qt.h>
#include "boost/tuple/tuple_comparison.hpp"
#include <boost/format.hpp>
#include <boost/algorithm/string/join.hpp>
#include <libiwl/iwl.hpp>
#include <psifiles.h>
//#include <libscf_solver/integralfunctors.h>
//#include <libscf_solver/omegafunctors.h>

#define DEBUG_OCDFT 0

#define DEBUG_THIS2(EXP) \
    outfile->Printf("\n  Starting " #EXP " ..."); fflush(outfile); \
    EXP \
    outfile->Printf("  done."); fflush(outfile); \


using namespace psi;

namespace psi{ namespace scf{

UCKS::UCKS(Options &options, boost::shared_ptr<PSIO> psio)
: UKS(options, psio),
  optimize_Vc(false),
  gradW_threshold_(1.0e-9),
  nW_opt(0),
  ground_state_energy(0.0),
  ground_state_symmetry_(0)
{
    init();
}

void UCKS::init()
{
    outfile->Printf("\n  ==> Unrestricted Constrained DFT (CDFT) <==\n\n");

    optimize_Vc = false;
    if(KS::options_["CHARGE"].size() > 0 or KS::options_["SPIN"].size() > 0){
        KS::options_.get_bool("OPTIMIZE_VC");
    }
    gradW_threshold_ = KS::options_.get_double("W_CONVERGENCE");
    outfile->Printf("  gradW threshold = :%9.2e\n",gradW_threshold_);
    nfrag = basisset()->molecule()->nfragments();
    outfile->Printf("  Number of fragments: %d\n",nfrag);
    level_shift_ = KS::options_.get_double("LEVEL_SHIFT");
    outfile->Printf("  Level shift: %f\n",level_shift_);

    build_W_frag();

    // Check the option CHARGE, if it is defined use it to define the constrained charges, "-" skips the constraint
    for (int f = 0; f < int(KS::options_["CHARGE"].size()); ++f){
        if(KS::options_["CHARGE"][f].to_string() != "-"){
            double constrained_charge = KS::options_["CHARGE"][f].to_double();
            double Nc = frag_nuclear_charge[f] - constrained_charge;
            SharedConstraint constraint(new Constraint(W_frag[f],Nc,1.0,1.0,"charge(" + to_string(f) + ")"));
            constraints.push_back(constraint);
            outfile->Printf("  Fragment %d: constrained charge = %f .\n",f,constrained_charge);
        }else{
            outfile->Printf("  Fragment %d: no charge constraint specified .\n",f);
        }
    }
    // Check the option SPIN, if it is defined use it to define the constrained spins, "-" skips the constraint
    for (int f = 0; f < int(KS::options_["SPIN"].size()); ++f){
        if(KS::options_["SPIN"][f].to_string() != "-"){
            double constrained_spin = KS::options_["SPIN"][f].to_double();
            double Nc = constrained_spin;
            SharedConstraint constraint(new Constraint(W_frag[f],Nc,0.5,-0.5,"spin(" + to_string(f) + ")"));
            constraints.push_back(constraint);
            outfile->Printf("  Fragment %d: constrained spin   = %f .\n",f,constrained_spin);
        }else{
            outfile->Printf("  Fragment %d: no spin constraint specified .\n",f);
        }
    }

//    zero_dim_ = Dimension(nirrep_);

    nconstraints = static_cast<int>(constraints.size());

    // Allocate vectors
    TempVector = factory_->create_shared_vector("SVD sigma");
    gradW = SharedVector(new Vector("gradW",nconstraints));
    gradW_old = SharedVector(new Vector("gradW_old",nconstraints));
    gradW_mo_resp = SharedVector(new Vector("gradW_mo_resp",nconstraints));
    Vc = SharedVector(new Vector("Vc",nconstraints));
    Vc_old = SharedVector(new Vector("Vc_old",nconstraints));

    // Allocate matrices
    H_copy = factory_->create_shared_matrix("H_copy");
    TempMatrix = factory_->create_shared_matrix("Temp");
    TempMatrix2 = factory_->create_shared_matrix("Temp2");
    Dolda_ = factory_->create_shared_matrix("Dold alpha");
    Doldb_ = factory_->create_shared_matrix("Dold beta");
    hessW = SharedMatrix(new Matrix("hessW",nconstraints,nconstraints));
    hessW_BFGS = SharedMatrix(new Matrix("hessW_BFGS",nconstraints,nconstraints));

    for (int f = 0; f < std::min(int(KS::options_["VC"].size()),nconstraints); ++f){
        if(KS::options_["VC"][f].to_string() != "-"){
            Vc->set(f,KS::options_["VC"][f].to_double());
        }else{
            Vc->set(f,0.0);
        }
        outfile->Printf("  The Lagrange multiplier for constraint %d will be initialized to Vc = %f .\n",f,KS::options_["VC"][f].to_double());
    }

    if(optimize_Vc){
        outfile->Printf("  The constraint(s) will be optimized.\n");
    }
    save_H_ = true;
}


UCKS::~UCKS()
{
}

void UCKS::build_W_frag()
{
    // Compute the overlap matrix
    boost::shared_ptr<BasisSet> basisset_ = basisset();
    boost::shared_ptr<Molecule> mol = basisset_->molecule();
    boost::shared_ptr<OneBodyAOInt> overlap(integral_->ao_overlap());
    SharedMatrix S_ao(new Matrix("S_ao",basisset_->nbf(),basisset_->nbf()));
    overlap->compute(S_ao);

    // Form the S^(1/2) matrix
    S_ao->power(1.0/2.0);

    boost::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
    SharedMatrix AO2SO_ = pet->aotoso();

    // Compute the W matrix for each fragment
    int min_a = 0;
    int max_a = 0;
    for (int f = 0; f < nfrag; ++f){
        std::vector<int> flist;
        std::vector<int> glist;
        flist.push_back(f);
        boost::shared_ptr<Molecule> frag = mol->extract_subsets(flist,glist);

        // Compute the nuclear charge on each fragment
        double frag_Z = 0.0;
        for (int a = 0; a < frag->natom(); ++a){
            frag_Z += frag->Z(a);
        }
        frag_nuclear_charge.push_back(frag_Z);

        // Form a copy of S_ao and zero the rows that are not on this fragment
        max_a = min_a + frag->natom();
        SharedMatrix S_f(S_ao->clone());
        for (int rho = 0; rho < basisset_->nbf(); rho++) {
            int shell = basisset_->function_to_shell(rho);
            int A = basisset_->shell_to_center(shell);
            if (A < min_a or max_a <= A){
                S_f->scale_row(0,rho,0.0);
            }
        }

        // Form W_f = (S_f)^T * S_f and transform it to the SO basis
        SharedMatrix W_f(new Matrix("W_f",basisset_->nbf(),basisset_->nbf()));
        SharedMatrix W_f_so(factory_->create_matrix("W_f_so"));
        W_f->gemm(true, false, 1.0, S_f, S_f, 0.0);
        W_f_so->apply_symmetry(W_f,AO2SO_);

        // Save W_f_so to the W_so vector
        W_frag.push_back(W_f_so);
        min_a = max_a;
    }
}

void UCKS::save_density_and_energy()
{
    Dtold_->copy(Dt_);
    Dolda_->copy(Da_);
    Doldb_->copy(Db_);
    Eold_ = E_;
}

void UCKS::form_G()
{
    timer_on("Form V");
    form_V();
    timer_off("Form V");

    // Push the C matrix on
    std::vector<SharedMatrix> & C = jk_->C_left();
    C.clear();
    C.push_back(Ca_subset("SO", "OCC"));
    C.push_back(Cb_subset("SO", "OCC"));

    // Addition to standard call, make sure that C_right is not set
    std::vector<SharedMatrix> & C_right = jk_->C_right();
    C_right.clear();

    // Run the JK object
    jk_->compute();

    // Pull the J and K matrices off
    const std::vector<SharedMatrix> & J = jk_->J();
    const std::vector<SharedMatrix> & K = jk_->K();
    const std::vector<SharedMatrix> & wK = jk_->wK();
    J_->copy(J[0]);
    J_->add(J[1]);
    if (functional_->is_x_hybrid()) {
        Ka_ = K[0];
        Kb_ = K[1];
    }
    if (functional_->is_x_lrc()) {
        wKa_ = wK[0];
        wKb_ = wK[1];
    }
    Ga_->copy(J_);
    Gb_->copy(J_);

    Ga_->add(Va_);
    Gb_->add(Vb_);

    double alpha = functional_->x_alpha();
    double beta = 1.0 - alpha;
    if (alpha != 0.0) {
        Ka_->scale(alpha);
        Kb_->scale(alpha);
        Ga_->subtract(Ka_);
        Gb_->subtract(Kb_);
        Ka_->scale(1.0/alpha);
        Kb_->scale(1.0/alpha);
    } else {
        Ka_->zero();
        Kb_->zero();
    }

    std::string functional_prefix = functional_->name().substr(0,2);
    if (functional_prefix == "sr"){
        wKa_->scale(-alpha);
        wKb_->scale(-alpha);
        Ga_->subtract(wKa_);
        Gb_->subtract(wKb_);
        wKa_->scale(-1.0/alpha);
        wKb_->scale(-1.0/alpha);
    } else{
        if (functional_->is_x_lrc()) {
            wKa_->scale(beta);
            wKb_->scale(beta);
            Ga_->subtract(wKa_);
            Gb_->subtract(wKb_);
            wKa_->scale(1.0/beta);
            wKb_->scale(1.0/beta);
        } else {
            wKa_->zero();
            wKb_->zero();
        }
    }

    if (debug_ > 2) {
        J_->print();
        Ka_->print();
        Kb_->print();
        wKa_->print();
        wKb_->print();
        Va_->print();
        Vb_->print();
    }
}

void UCKS::form_F()
{
    // On the first iteration save H_
    if (save_H_){
        H_copy->copy(H_);
        save_H_ = false;
    }

    // Augement the one-electron potential (H_) with the CDFT terms
    H_->copy(H_copy);
    for (int c = 0; c < nconstraints; ++c){
        TempMatrix->copy(constraints[c]->W_so());
        TempMatrix->scale(Vc->get(c) * constraints[c]->weight_alpha());
        H_->add(TempMatrix);
    }
    Fa_->copy(H_);
    Fa_->add(Ga_);

    H_->copy(H_copy);
    for (int c = 0; c < nconstraints; ++c){
        TempMatrix->copy(constraints[c]->W_so());
        TempMatrix->scale(Vc->get(c) * constraints[c]->weight_beta());
        H_->add(TempMatrix);
    }
    Fb_->copy(H_);
    Fb_->add(Gb_);

    gradient_of_W();

    if (debug_) {
        Fa_->print();
        Fb_->print();
    }
}

void UCKS::form_C()
{
    // Ground state: use the default form_C
    UKS::form_C();
    if(iteration_ == 4 and KS::options_["CDFT_BREAK_SYMMETRY"].has_changed()){
        // Mix the alpha and beta homo
        int np = KS::options_["CDFT_BREAK_SYMMETRY"][0].to_integer();
        int nq = KS::options_["CDFT_BREAK_SYMMETRY"][1].to_integer();
        double angle = KS::options_["CDFT_BREAK_SYMMETRY"][2].to_double();
        outfile->Printf("\n  Mixing the alpha orbitals %d and %d by %f.1 degrees\n\n",np,nq,angle);
        outfile->Flush();
        Ca_->rotate_columns(0,np-1,nq-1,pc_pi * angle / 180.0);
        Cb_->rotate_columns(0,np-1,nq-1,-pc_pi * angle / 180.0);
        // Reset the DIIS subspace
        diis_manager_->reset_subspace();
    }
}


/// Gradient of W
///
/// Implements Eq. (6) of Phys. Rev. A 72, 024502 (2005) and estimates a correction to the
/// gradient from the orbital response
void UCKS::gradient_of_W()
{
    for (int c = 0; c < nconstraints; ++c){
        double grad = 0.0;
        grad  = constraints[c]->weight_alpha() * Da_->vector_dot(constraints[c]->W_so());
        grad += constraints[c]->weight_beta()  * Db_->vector_dot(constraints[c]->W_so());
        grad -= constraints[c]->Nc();
        gradW->set(c,grad);
    }
    for (int c = 0; c < nconstraints; ++c){
        double grad = 0.0;
        // Transform W_so to the MO basis (alpha)
        TempMatrix->transform(constraints[c]->W_so(),Ca_);
        TempMatrix->scale(constraints[c]->weight_alpha());
        // Transform Fa_ to the MO basis
        TempMatrix2->transform(Fa_,Ca_);
        for (int h = 0; h < nirrep_; h++) {
            int nmo  = nmopi_[h];
            int nocc = nalphapi_[h];
            int nvir = nmo - nocc;
            if (nvir == 0 or nocc == 0) continue;
            double** Temp_h = TempMatrix->pointer(h);
            double** Temp2_h = TempMatrix2->pointer(h);
            for (int i = 0; i < nocc; ++i){
                for (int a = nocc; a < nmo; ++a){
                    grad += 2.0 * Temp_h[a][i] * Temp2_h[a][i] / (Temp2_h[i][i] - Temp2_h[a][a]);
                }
            }
        }
        // Transform W_so to the MO basis (beta)
        TempMatrix->transform(constraints[c]->W_so(),Cb_);
        TempMatrix->scale(constraints[c]->weight_beta());
        // Transform Fb_ to the MO basis
        TempMatrix2->transform(Fb_,Cb_);
        for (int h = 0; h < nirrep_; h++) {
            int nmo  = nmopi_[h];
            int nocc = nbetapi_[h];
            int nvir = nmo - nocc;
            if (nvir == 0 or nocc == 0) continue;
            double** Temp_h = TempMatrix->pointer(h);
            double** Temp2_h = TempMatrix2->pointer(h);
            for (int i = 0; i < nocc; ++i){
                for (int a = nocc; a < nmo; ++a){
                    grad += 2.0 * Temp_h[a][i] * Temp2_h[a][i] / (Temp2_h[i][i] - Temp2_h[a][a]);
                }
            }
        }
        gradW_mo_resp->set(c,grad);
    }
    if(nconstraints > 0){
        for (int c = 0; c < nconstraints; ++c){
            outfile->Printf("   %-10s: grad = %10.7f    grad (resp) = %10.7f    Vc = %10.7f\n",constraints[c]->type().c_str(),
                    gradW->get(c),gradW_mo_resp->get(c),Vc->get(c));
        }
    }
}

/// Hessian of W
///
/// Implements Eq. (7) of Phys. Rev. A 72, 024502 (2005).
void UCKS::hessian_of_W()
{
    outfile->Printf("\n  COMPUTE THE HESSIAN\n\n");
    for (int c1 = 0; c1 < nconstraints; ++c1){
        for (int c2 = 0; c2 <= c1; ++c2){
            double hess = 0.0;
            // Transform W_so to the MO basis (alpha)
            TempMatrix->transform(constraints[c1]->W_so(),Ca_);
            TempMatrix->scale(constraints[c1]->weight_alpha());
            // Transform W_so to the MO basis (alpha)
            TempMatrix2->transform(constraints[c2]->W_so(),Ca_);
            TempMatrix2->scale(constraints[c2]->weight_alpha());
            for (int h = 0; h < nirrep_; h++) {
                int nmo  = nmopi_[h];
                int nocc = nalphapi_[h];
                int nvir = nmo - nocc;
                if (nvir == 0 or nocc == 0) continue;
                double** Temp_h = TempMatrix->pointer(h);
                double** Temp2_h = TempMatrix2->pointer(h);
                double* eps = epsilon_a_->pointer(h);
                for (int i = 0; i < nocc; ++i){
                    for (int a = nocc; a < nmo; ++a){
                        hess += 2.0 * Temp_h[i][a] * Temp2_h[a][i] /  (eps[i] - eps[a]);
                    }
                }
            }
            // Transform W_so to the MO basis (beta)
            TempMatrix->transform(constraints[c1]->W_so(),Cb_);
            TempMatrix->scale(constraints[c1]->weight_beta());
            // Transform W_so to the MO basis (beta)
            TempMatrix2->transform(constraints[c2]->W_so(),Cb_);
            TempMatrix2->scale(constraints[c2]->weight_beta());
            for (int h = 0; h < nirrep_; h++) {
                int nmo  = nmopi_[h];
                int nocc = nbetapi_[h];
                int nvir = nmo - nocc;
                if (nvir == 0 or nocc == 0) continue;
                double** Temp_h = TempMatrix->pointer(h);
                double** Temp2_h = TempMatrix2->pointer(h);
                double* eps = epsilon_b_->pointer(h);
                for (int i = 0; i < nocc; ++i){
                    for (int a = nocc; a < nmo; ++a){
                        hess += 2.0 * Temp_h[i][a] * Temp2_h[a][i] /  (eps[i] - eps[a]);
                    }
                }
            }
            hessW->set(c1,c2,hess);
            hessW->set(c2,c1,hess);
        }
    }
}

/// Apply the BFGS update to the approximate Hessian h[][].
/// h[][] = Hessian matrix from previous iteration
/// dx[]  = Step from previous iteration (dx[] = x[] - xp[] where xp[] is docc previous point)
/// dg[]  = gradient difference (dg = g - gp)
void UCKS::hessian_update(SharedMatrix h, SharedVector dx, SharedVector dg)
{
    SharedVector hdx = SharedVector(new Vector("hdx",nconstraints));
    hdx->gemv(false, 1.0, h.get(), dx.get(), 0.0);
    double dxhdx = dx->dot(hdx.get());
    double dxdx  = dx->dot(dx.get());
    double dxdg  = dx->dot(dg.get());
    double dgdg  = dg->dot(dg.get());

    if ( (dxdx > 0.0) && (dgdg > 0.0) && (std::fabs(dxdg / std::sqrt(dxdx * dgdg)) > 1.e-8) ) {
        for (int i = 0; i < nconstraints; ++i) {
            for (int j = 0; j < nconstraints; ++j) {
                h->add(i,j,dg->get(i) * dg->get(j) / dxdg - hdx->get(i) * hdx->get(j) / dxhdx);
            }
        }
    }
    else {
        outfile->Printf("  BFGS not updating dxdg (%e), dgdg (%e), dxhdx (%f), dxdx(%e)\n" , dxdg, dgdg, dxhdx, dxdx);
    }
}

/// Optimize the Lagrange multiplier
void UCKS::constraint_optimization()
{
    if(KS::options_.get_str("W_ALGORITHM") == "NEWTON"){
        // Optimize Vc once you have a good gradient and the gradient is not converged
        double max_error = 0.0;
        for (int c = 0; c < nconstraints; ++c){
            if(std::fabs(gradW->get(c)) > gradW_threshold_){
                max_error = std::max(max_error,std::fabs(gradW_mo_resp->get(c) / gradW->get(c)));
            }
        }
        // Make sure the component with the largest error is within 10% of the estimate
        if (max_error < 0.1 and gradW->norm() > gradW_threshold_){
            outfile->Printf( "  ==> Constraint optimization <==\n");
            if(nW_opt > 0){
                SharedVector dVc = SharedVector(new Vector("dVc",nconstraints));
                dVc->copy(Vc.get());
                dVc->subtract(Vc_old);
                SharedVector dgradW = SharedVector(new Vector("dgradW",nconstraints));
                dgradW->copy(gradW.get());
                dgradW->subtract(gradW_old);
                hessian_update(hessW_BFGS, dVc, dgradW);
                outfile->Printf( "  Hessian update.\n");
            }else{
                hessian_of_W();
                hessW_BFGS->copy(hessW);
            }
            hessW_BFGS->print();
            hessW->print();
            // First step, do a Newton with the hessW information
            Vc_old->copy(Vc.get());
            gradW_old->copy(gradW.get());
            // Use the Hessian to do a Newton step with trust radius
            bool warning;
            SharedMatrix hessW_inv = hessW_BFGS->pseudoinverse(1.0E-10, &warning);
            if (warning) {
                outfile->Printf( "  Warning, the inverse Hessian had to be conditioned.\n\n");
            }
            SharedVector h_inv_g = SharedVector(new Vector("h_inv_g",nconstraints));
            h_inv_g->gemv(false, 1.0, hessW_inv.get(), gradW.get(), 0.0);

            // Apply trust radius
            double step_size = h_inv_g->norm();
            double threshold = 0.5;
            if(step_size > threshold){
                h_inv_g->scale(threshold / step_size);
            }
            Vc->subtract(h_inv_g);

            // Reset the DIIS subspace
            diis_manager_->reset_subspace();
            nW_opt += 1;
        }
    }
}

double UCKS::compute_E()
{
    // E_CDFT = 2.0 D*H + D*J - \alpha D*K + E_xc - Nc * Vc
    double one_electron_E = Da_->vector_dot(H_);
    one_electron_E += Db_->vector_dot(H_);
    for (int c = 0; c < nconstraints; ++c){
        one_electron_E -= Vc->get(c) * constraints[c]->Nc(); // Added the CDFT contribution that is not included in H_
    }
    double coulomb_E = Da_->vector_dot(J_);
    coulomb_E += Db_->vector_dot(J_);

    std::map<std::string, double>& quad = potential_->quadrature_values();
    double XC_E = quad["FUNCTIONAL"];
    double exchange_E = 0.0;
    double alpha = functional_->x_alpha();
    double beta = 1.0 - alpha;
    if (functional_->is_x_hybrid()) {
        exchange_E -= alpha*Da_->vector_dot(Ka_);
        exchange_E -= alpha*Db_->vector_dot(Kb_);
    }

    std::string functional_prefix = functional_->name().substr(0,2);
    if (functional_prefix == "sr"){
        exchange_E +=  alpha * Da_->vector_dot(wKa_);
        exchange_E +=  alpha * Db_->vector_dot(wKb_);
    }else{
        if (functional_->is_x_lrc()) {
            exchange_E -=  beta*Da_->vector_dot(wKa_);
            exchange_E -=  beta*Db_->vector_dot(wKb_);
        }
    }

    double dashD_E = 0.0;
    boost::shared_ptr<Dispersion> disp;
    if (disp) {
        dashD_E = disp->compute_energy(HF::molecule_);
    }

    double Etotal = 0.0;
    Etotal += nuclearrep_;
    Etotal += one_electron_E;
    Etotal += 0.5 * coulomb_E;
    Etotal += 0.5 * exchange_E;
    Etotal += XC_E;
    Etotal += dashD_E;

    if (debug_) {
        outfile->Printf( "   => Energetics <=\n\n");
        outfile->Printf( "    Nuclear Repulsion Energy = %24.14f\n", nuclearrep_);
        outfile->Printf( "    One-Electron Energy =      %24.14f\n", one_electron_E);
        outfile->Printf( "    Coulomb Energy =           %24.14f\n", 0.5 * coulomb_E);
        outfile->Printf( "    Hybrid Exchange Energy =   %24.14f\n", 0.5 * exchange_E);
        outfile->Printf( "    XC Functional Energy =     %24.14f\n", XC_E);
        outfile->Printf( "    -D Energy =                %24.14f\n", dashD_E);
    }
    return Etotal;
}

void UCKS::damp_update()
{
//    double damping = damping_percentage_;
//    for(int h = 0; h < nirrep_; ++h){
//        for(int row = 0; row < Da_->rowspi(h); ++row){
//            for(int col = 0; col < Da_->colspi(h); ++col){
//                double Dolda = damping * Dolda_->get(h, row, col);
//                double Dnewa = (1.0 - damping) * Da_->get(h, row, col);
//                Da_->set(h, row, col, Dolda+Dnewa);
//                double Doldb = damping * Doldb_->get(h, row, col);
//                double Dnewb = (1.0 - damping) * Db_->get(h, row, col);
//                Db_->set(h, row, col, Doldb+Dnewb);
//            }
//        }
//    }
//    // Update Dt_
//    Dt_->copy(Da_);
//    Dt_->add(Db_);
}

bool UCKS::test_convergency()
{
    // energy difference
    double ediff = E_ - Eold_;

    // RMS of the density
    Matrix D_rms;
    D_rms.copy(Dt_);
    D_rms.subtract(Dtold_);
    Drms_ = 0.5 * D_rms.rms();

    bool energy_test = fabs(ediff) < energy_threshold_;
    bool density_test = Drms_ < density_threshold_;
    bool cycle_test = iteration_ > 5;

    if(optimize_Vc){
        bool constraint_test = gradW->norm() < gradW_threshold_;
        constraint_optimization();
        if(energy_test and density_test and constraint_test and cycle_test){
            return true;
        }else{
            return false;
        }
    }else{
        if(energy_test and density_test and cycle_test){
            return true;
        }
        return false;
    }
}

}} // Namespaces
