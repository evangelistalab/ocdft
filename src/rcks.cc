#include "rcks.h"
#include <libmints/view.h>
#include <libmints/mints.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <libdisp/dispersion.h>
#include <liboptions/liboptions.h>
#include <libciomr/libciomr.h>
#include <libqt/qt.h>

using namespace psi;

namespace psi{ namespace scf{

RCKS::RCKS(Options& options, boost::shared_ptr<PSIO> psio)
    : RKS(options, psio), Vc(0.0), optimize_Vc(false), gradW_threshold_(1.0e-9),nW_opt(0), old_gradW(0.0), BFGS_hessW(0.0)
{
    outfile->Printf("\n  ==> Constrained DFT (RCKS) <==\n\n");

    Vc = options.get_double("VC");
    optimize_Vc = options.get_bool("OPTIMIZE_VC");

    if(optimize_Vc){
        outfile->Printf("  The constraint will be optimized.\n");
    }else{
        outfile->Printf("  The Lagrange multiplier for the constraint will be fixed: Vc = %f .\n",Vc);
    }

    gradW_threshold_ = options.get_double("W_CONVERGENCE");
    outfile->Printf("  gradW threshold = :%9.2e\n",gradW_threshold_);
    nfrag = basisset()->molecule()->nfragments();
    outfile->Printf("  Number of fragments: %d\n",nfrag);

    // Check the option CHARGE, if it is defined use it to define the constrained charges
    int charge_size = options["CHARGE"].size();
    if (charge_size > 0){
        if (charge_size == nfrag){
            for (int n = 0; n < charge_size; ++n){
                constrained_charges.push_back(options["CHARGE"][n].to_double());
            }
        }else{
            throw InputException("The number of charge constraints does not match the number of fragments", "CHARGE", __FILE__, __LINE__);
        }
    }

    build_W_so();

    Temp = SharedMatrix(factory_->create_matrix("Temp"));
    Temp2 = SharedMatrix(factory_->create_matrix("Temp2"));

    save_H_ = true;

    for (int f = 0; f < nfrag; ++f){
        outfile->Printf("  Fragment %d: constrained charge = %f .\n",f,constrained_charges[f]);
    }
    Nc  = frag_nuclear_charge[0] - constrained_charges[0];
    Nc -= frag_nuclear_charge[1] - constrained_charges[1];
    outfile->Printf("  Constraining Tr[w rho] = Nc = %f .\n\n",Nc);
}


RCKS::~RCKS()
{
}

void RCKS::build_W_so()
{
    // Allocate the total constraint matrix
    W_tot = SharedMatrix(factory_->create_matrix("W_tot"));

    //    Compute the overlap matrix
    boost::shared_ptr<BasisSet> basisset_ = basisset();
    boost::shared_ptr<Molecule> mol = basisset_->molecule();
    boost::shared_ptr<OneBodyAOInt> overlap(integral_->ao_overlap());
    SharedMatrix S_ao(new Matrix("S_ao",basisset_->nbf(),basisset_->nbf()));
    overlap->compute(S_ao);

    //    Form the S^(1/2) matrix
    S_ao->power(1.0/2.0);

    boost::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
    SharedMatrix AO2SO_ = pet->aotoso();

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

        constrained_charges.push_back(double(frag->molecular_charge())); // TODO Remove the factor of 10.0!!!

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

        // If CHARGE is not defined, then read the charges from the fragment input
        if(constrained_charges.size() != nfrag){
            constrained_charges.push_back(double(frag->molecular_charge()));
        }

        // Form W_f = (S_f)^T * S_f and transform it to the SO basis
        SharedMatrix W_f(new Matrix("W_f",basisset_->nbf(),basisset_->nbf()));
        SharedMatrix W_f_so(factory_->create_matrix("W_f_so"));
        W_f->gemm(true, false, 1.0, S_f, S_f, 0.0);
        W_f_so->apply_symmetry(W_f,AO2SO_);

        // Save W_f_so to the W_so vector
        W_so.push_back(W_f_so);
        min_a = max_a;
    }
    W_tot->zero();
    W_tot->add(W_so[0]);
    W_tot->subtract(W_so[1]);
}

void RCKS::form_F()
{
    // On the first iteration save H_
    if (save_H_){
        H_copy = SharedMatrix(factory_->create_matrix("H_copy"));
        H_copy->copy(H_);
        save_H_ = false;
    }

    // Augement the one-electron potential (H_) with the CDFT terms
    H_->copy(H_copy);
    Temp->copy(W_tot);
    Temp->scale(Vc);
    H_->add(Temp);  // Temp = Vc * W_tot

    Fa_->copy(H_);
    Fa_->add(G_);

    gradient_of_W();
    hessian_of_W();

    outfile->Printf("   @CDFT                Vc = %.7f  gradW = %.7f  hessW = %.7f\n",Vc,gradW,hessW);

    if (debug_) {
        Fa_->print();
        J_->print();
        K_->print();
        G_->print();
    }
}

double RCKS::compute_E()
{
    // E_DFT = 2.0 D*H + 2.0 D*J - \alpha D*K + E_xc - Vc * Nc
    double one_electron_E = 2.0*D_->vector_dot(H_) - Vc * Nc;  // Added the CDFT contribution that is not included in H_
    double coulomb_E = D_->vector_dot(J_);

    std::map<std::string, double>& quad = potential_->quadrature_values();
    double XC_E = quad["FUNCTIONAL"];
    double exchange_E = 0.0;
    double alpha = functional_->x_alpha();
    double beta = 1.0 - alpha;
    if (functional_->is_x_hybrid()) {
        exchange_E -= alpha*Da_->vector_dot(K_);
    }
    if (functional_->is_x_lrc()) {
        exchange_E -=  beta*Da_->vector_dot(wK_);
    }

    double dashD_E = 0.0;
    boost::shared_ptr<Dispersion> disp;
    if (disp) {
        dashD_E = disp->compute_energy(HF::molecule_);
    }

    double Etotal = 0.0;
    Etotal += nuclearrep_;
    Etotal += one_electron_E;
    Etotal += coulomb_E;
    Etotal += exchange_E;
    Etotal += XC_E;
    Etotal += dashD_E;

    if (debug_) {
        outfile->Printf( "   => Energetics <=\n\n");
        outfile->Printf( "    Nuclear Repulsion Energy = %24.14f\n", nuclearrep_);
        outfile->Printf( "    One-Electron Energy =      %24.14f\n", one_electron_E);
        outfile->Printf( "    Coulomb Energy =           %24.14f\n", coulomb_E);
        outfile->Printf( "    Hybrid Exchange Energy =   %24.14f\n", exchange_E);
        outfile->Printf( "    XC Functional Energy =     %24.14f\n", XC_E);
        outfile->Printf( "    -D Energy =                %24.14f\n\n", dashD_E);
    }

    return Etotal;
}

bool RCKS::test_convergency()
{
    // energy difference
    double ediff = E_ - Eold_;

    // RMS of the density
    Matrix D_rms;
    D_rms.copy(D_);
    D_rms.subtract(Dold_);
    Drms_ = D_rms.rms();

    if(optimize_Vc){
        constraint_optimization();
        return (fabs(ediff) < energy_threshold_ and Drms_ < density_threshold_ and std::fabs(gradW) < gradW_threshold_);
    }
    return (fabs(ediff) < energy_threshold_ and Drms_ < density_threshold_);
}

/// Gradient of W
///
/// Implements Eq. (6) of Phys. Rev. A 72, 024502 (2005).
void RCKS::gradient_of_W()
{
//    gradW  = 2.0 * D_->vector_dot(W_so[0]) - (frag_nuclear_charge[0] - constrained_charges[0]);
//    gradW -= 2.0 * D_->vector_dot(W_so[1]) - (frag_nuclear_charge[1] - constrained_charges[1]);
    gradW = 2.0 * D_->vector_dot(W_tot) - Nc;
    outfile->Printf("  gradW(1) = %.9f\n",gradW);

    // Transform W_tot to the MO basis
    Temp->transform(W_tot,Ca_);
    // Transform Fa_ to the MO basis
    Temp2->transform(Fa_,Ca_);

    gradW_mo_resp = 0.0;
    for (int h = 0; h < nirrep_; h++) {
        int nmo = nmopi_[h];
        int nvir = nmopi_[h]-doccpi_[h];
        int nocc = doccpi_[h];
        if (nvir == 0 or nocc == 0) continue;
        double** Temp_h = Temp->pointer(h);
        double** Temp2_h = Temp2->pointer(h);
        double* eps = epsilon_a_->pointer(h);
        for (int i = 0; i < nocc; ++i){
            for (int a = nocc; a < nmo; ++a){
                gradW_mo_resp += 4.0 * Temp_h[a][i] * Temp2_h[a][i] / (Temp2_h[i][i] - Temp2_h[a][a]);//  ;(eps[i] - eps[a]);
            }
        }
    }
}


//    SharedMatrix eigvec= factory_->create_shared_matrix("L");
//    SharedVector eigval(factory_->create_vector());
//    // Transform W_tot to the MO basis
//    Temp->transform(W_tot,Ca_);
//    // Diagonalize W
//    Temp->diagonalize(eigvec,eigval);
//    eigvec->print();
//    eigval->print();


/// Hessian of W
///
/// Implements Eq. (7) of Phys. Rev. A 72, 024502 (2005).
void RCKS::hessian_of_W()
{
    hessW = 0.0;
    // Transform W_tot to the MO basis
    Temp->transform(W_tot,Ca_);
    for (int h = 0; h < nirrep_; h++) {
        int nmo = nmopi_[h];
        int nvir = nmopi_[h]-doccpi_[h];
        int nocc = doccpi_[h];
        if (nvir == 0 or nocc == 0) continue;
        double** Temp_h = Temp->pointer(h);
        double* eps = epsilon_a_->pointer(h);
        for (int i = 0; i < nocc; ++i){
            for (int a = nocc; a < nmo; ++a){
                //outfile->Printf("  -> (%d,%d): (%f)^2 / (%f - %f) = %f\n",i,a,Temp_h[a][i],eps[i],eps[a],std::pow(Temp_h[a][i],2.0) /  (eps[i] - eps[a]));
                hessW += Temp_h[a][i] * Temp_h[a][i] /  (eps[i] - eps[a]);
            }
        }
    }
    hessW *= 4.0; // 2 for the complex conjugate and 2 for the spin cases
}

/// Optimize the Lagrange multiplier
void RCKS::constraint_optimization()
{   
    outfile->Printf( "  ==> Constraint optimization <==\n");
    if(psi::scf::KS::options_.get_str("W_ALGORITHM") == "NEWTON"){
        // Optimize Vc once you have a good gradient and the gradient is not converged
        if (std::fabs(gradW_mo_resp / gradW) < 0.1 and std::fabs(gradW) > gradW_threshold_){
            // First step, do a Newton with the hessW information
            if(nW_opt == 0){
                old_Vc = Vc;
                old_gradW = gradW;
                // Use the crappy Hessian to do a Newton step with trust radius
                double new_Vc = Vc - gradW / hessW;
                double threshold = 0.5;
                if(std::fabs(new_Vc - Vc) > threshold){
                    new_Vc = Vc + (new_Vc > Vc ? threshold : -threshold);
                }
                outfile->Printf( "  hessW = %f\n",hessW);
                Vc = new_Vc;
            }else{
                if(std::fabs(Vc - old_Vc) > 1.0e-3){
                    // We landed somewhere else, update the Hessian
                    BFGS_hessW = (gradW - old_gradW) / (Vc - old_Vc);
                    outfile->Printf( "  BFGS_hessW = %f\n",BFGS_hessW);
                    old_Vc = Vc;
                    old_gradW = gradW;
                }
                double new_Vc = Vc - gradW / BFGS_hessW;
                Vc = new_Vc;
            }
            // Reset the DIIS subspace
            diis_manager_->reset_subspace();
        }
    }else if (psi::scf::KS::options_.get_str("W_ALGORITHM") == "QUADRATIC"){
        if (std::fabs(gradW_mo_resp / gradW) < 0.25 and std::fabs(gradW) > gradW_threshold_){
        // Transform W_tot to the MO basis
        Temp->transform(W_tot,Ca_);
        double numerator = 0.0;
        for (int h = 0; h < nirrep_; h++) {
            int nocc = doccpi_[h];
            double** Temp_h = Temp->pointer(h);
            for (int i = 0; i < nocc; ++i){
                numerator += 2.0 * Temp_h[i][i];
            }
        }
        numerator -= Nc;
        Temp->power(2.0);
        double denominator = 0.0;
        for (int h = 0; h < nirrep_; h++) {
            int nocc = doccpi_[h];
            double** Temp_h = Temp->pointer(h);
            for (int i = 0; i < nocc; ++i){
                denominator += 2.0 * Temp_h[i][i];
            }
        }
        Vc += 0.5 * numerator / denominator;
        }
    }
    nW_opt += 1;
}

void RCKS::Lowdin2()
{
    outfile->Printf( "  ==> Lowdin Charges <==\n\n");
    for (int f = 0; f < nfrag; ++f){
        double Q_f = -2.0 * D_->vector_dot(W_so[f]);
        Q_f += frag_nuclear_charge[f];
        outfile->Printf( "  Fragment %d: charge = %.6f\n",f,Q_f);
    }
}

void RCKS::Lowdin()
{
    //    Compute the overlap matrix
    boost::shared_ptr<BasisSet> basisset_ = basisset();
    boost::shared_ptr<OneBodyAOInt> overlap(integral_->ao_overlap());
    SharedMatrix S_ao(new Matrix("S_ao",basisset_->nbf(),basisset_->nbf()));
    SharedMatrix D_ao(new Matrix("D_ao",basisset_->nbf(),basisset_->nbf()));
    SharedMatrix L_ao(new Matrix("L_ao",basisset_->nbf(),basisset_->nbf()));
    overlap->compute(S_ao);

    //    Form the S^(1/2) matrix
    S_ao->power(1.0/2.0);

    boost::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
    SharedMatrix SO2AO_ = pet->sotoao();
    D_ao->remove_symmetry(D_,SO2AO_);
    L_ao->transform(D_ao,S_ao);
    L_ao->print();

    boost::shared_ptr<Molecule> mol = basisset_->molecule();
    SharedVector Qa(new Vector(mol->natom()));
    double* Qa_pointer = Qa->pointer();


    for (int a = 0; a < mol->natom(); ++a){
        Qa->set(a,mol->Z(a));
    }

    for (int mu = 0; mu < basisset_->nbf(); mu++) {
        double charge = L_ao->get(0,mu,mu);
        int shell = basisset_->function_to_shell(mu);
        int A = basisset_->shell_to_center(shell);

        Qa_pointer[A] -= 2.0 * charge;
      }
    Qa->print();

    int nfrag = mol->nfragments();
    outfile->Printf( "\n  There are %d fragments in this molecule\n", nfrag);
    int a = 0;
    for (int f = 0; f < nfrag; ++f){
        std::vector<int> flist;
        std::vector<int> glist;
        flist.push_back(f);
        boost::shared_ptr<Molecule> frag = mol->extract_subsets(flist,glist);
        double fcharge = 0.0;
        for (int n = 0; n < frag->natom(); ++n){
            fcharge += Qa_pointer[a];
            ++a;
          }
        outfile->Printf("  Fragment %d, charge = %.8f, constrained charge = %.8f:\n",f,fcharge,double(frag->molecular_charge()));
    }
}

}} // Namespaces
