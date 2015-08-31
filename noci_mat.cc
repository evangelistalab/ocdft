#include <physconst.h>
#include <psifiles.h>
#include <libmints/mints.h>
#include <libmints/wavefunction.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <liboptions/liboptions.h>
#include <libciomr/libciomr.h>
#include <libiwl/iwl.hpp>
#include <libpsio/psio.hpp>
#include "noci_mat.h"
#include "helpers.h"

#define DEBUG_NOCI 0

using namespace psi;

namespace psi{ namespace scf{

NOCI_Hamiltonian::NOCI_Hamiltonian(Options &options, std::vector<SharedDeterminant> dets)
    : options_(options), dets_(dets)
{
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    boost::shared_ptr<Molecule> mol = Process::environment.molecule();


    nirrep_ = wfn->nirrep();
    factory_ = wfn->matrix_factory();
    nsopi_ = wfn->nsopi();
    Sao_ = wfn->S()->clone();
    Hao_ = wfn->H()->clone();
    jk_ = JK::build_JK();
    nuclearrep_ = mol->nuclear_repulsion_energy();
    basisset_ = wfn->basisset();
    nso   = wfn->nso();

    TempMatrix = factory_->create_shared_matrix("TempMatrix");
    TempMatrix2 = factory_->create_shared_matrix("TempMatrix2");
    read_tei();
}

NOCI_Hamiltonian::~NOCI_Hamiltonian()
{
}

double NOCI_Hamiltonian::compute_energy()
{
    outfile->Printf("\n  ==> Computing the NOCI energy <==");

    size_t ndets = dets_.size();
    S_ = SharedMatrix(new Matrix("S",ndets,ndets));
    H_ = SharedMatrix(new Matrix("H",ndets,ndets));
    for (size_t i = 0; i < ndets; ++i){
        for (size_t j = 0; j < ndets; ++j){
            outfile->Printf("\n  Element (%d,%d)",i,j);
            std::pair<double,double> S_H = matrix_element_c1(dets_[i],dets_[j]);
            S_->set(i,j,S_H.first);
            H_->set(i,j,S_H.second);
        }
    }
    S_->print();
    H_->print();

    return 0.0;
}

void NOCI_Hamiltonian::print()
{
}

void NOCI_Hamiltonian::read_tei()
{
    outfile->Printf("\n  ==> Reading the two-electron integrals (C1 symmetry) <==");
    size_t maxi4 = INDEX4(nsopi_[0]+1,nsopi_[0]+1,nsopi_[0]+1,nsopi_[0]+1)+nsopi_[0]+1;
    tei_ints_.resize(maxi4);
    for (size_t l = 0; l < maxi4; ++l){
        tei_ints_[l] = 0.0;
    }
    double integral_threshold_ = 1.0e-12;
    IWL *iwl = new IWL(PSIO::shared_object().get(), PSIF_SO_TEI, integral_threshold_, 1, 1);
    Label *lblptr = iwl->labels();
    Value *valptr = iwl->values();
    int labelIndex, p,q,r,s;
    double value;
    bool lastBuffer;
    do{
        lastBuffer = iwl->last_buffer();
        for(int index = 0; index < iwl->buffer_count(); ++index){
            labelIndex = 4 * index;
            p  = abs((int) lblptr[labelIndex++]);
            q  = (int) lblptr[labelIndex++];
            r  = (int) lblptr[labelIndex++];
            s  = (int) lblptr[labelIndex++];
            value = (double) valptr[index];
            tei_ints_[INDEX4(p,q,r,s)] = value;
        } /* end loop through current buffer */
        if(!lastBuffer) iwl->fetch();
    }while(!lastBuffer);
    iwl->set_keep_flag(1);

    delete iwl;
}

void NOCI_Hamiltonian::J(SharedMatrix D)
{
    TempMatrix->zero();

    // Number of symmetry-adapted AOs
    size_t nso  = nsopi_[0];

    double** Tp = TempMatrix->pointer(0);
    double** Dp = D->pointer(0);
    for (size_t k = 0; k < nso; ++k){
        for (size_t l = 0; l < nso; ++l){
            double Jkl = 0.0;
            for (size_t m = 0; m < nso; ++m){
                for (size_t n = 0; n < nso; ++n){
                    Jkl += tei_ints_[INDEX4(k,l,m,n)] * Dp[m][n];
                }
            }
            Tp[k][l] = Jkl;
        }
    }
}

void NOCI_Hamiltonian::K(SharedMatrix D)
{
    TempMatrix->zero();

    // Number of symmetry-adapted AOs
    size_t nso  = nsopi_[0];

    double** Tp = TempMatrix->pointer(0);
    double** Dp = D->pointer(0);
    for (size_t k = 0; k < nso; ++k){
        for (size_t l = 0; l < nso; ++l){
            double Kkl = 0.0;
            for (size_t m = 0; m < nso; ++m){
                for (size_t n = 0; n < nso; ++n){
                    Kkl += tei_ints_[INDEX4(k,m,l,n)] * Dp[m][n];
                }
            }
            Tp[k][l] = Kkl;
        }
    }
}

std::pair<double,double> NOCI_Hamiltonian::matrix_element_c1(SharedDeterminant A, SharedDeterminant B)
{
    double overlap = 0.0;
    double hamiltonian = 0.0;
    nirrep_ = 1;
    // Number of alpha occupied orbitals
    size_t nocc_a = A->nalphapi()[0];
    // Number of beta occupied orbitals
    size_t nocc_b = A->nbetapi()[0];

    // I. Form the corresponding alpha and beta orbitals
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double> calpha = corresponding_orbitals(A->Ca(),B->Ca(),A->nalphapi(),B->nalphapi());
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double> cbeta  = corresponding_orbitals(A->Cb(),B->Cb(),A->nbetapi(),B->nbetapi());
    SharedMatrix ACa = calpha.get<0>();
    SharedMatrix BCa = calpha.get<1>();
    SharedMatrix ACb = cbeta.get<0>();
    SharedMatrix BCb = cbeta.get<1>();
    double detUValpha = calpha.get<3>();
    double detUVbeta = cbeta.get<3>();
    SharedVector s_a = calpha.get<2>();
    SharedVector s_b = cbeta.get<2>();

    // Compute the number of noncoincidences
    double noncoincidence_threshold = 1.0e-9;

    double Sta = 1.0;
    std::vector<std::pair<size_t,double>> alpha_nonc;
    for (size_t p = 0; p < nocc_a; ++p){
        double s_p = s_a->get(0,p);
        if(std::fabs(s_p) >= noncoincidence_threshold){
            Sta *= s_p;
        }else{
            alpha_nonc.push_back(std::make_pair(p,s_p));
        }
    }

    double Stb = 1.0;
    std::vector<std::pair<size_t,double>> beta_nonc;
    for (size_t p = 0; p < nocc_b; ++p){
        double s_p = s_b->get(0,p);
        if(std::fabs(s_p) >= noncoincidence_threshold){
            Stb *= s_p;
        }else{
            beta_nonc.push_back(std::make_pair(p,s_p));
        }
    }

    outfile->Printf("\n  Corresponding orbitals:\n");
    outfile->Printf("\n  Alpha: ");
    for (auto& kv : alpha_nonc){
        outfile->Printf(" MO = %zu, s = %e",kv.first,kv.second);
    }
    outfile->Printf("\n  Beta: ");
    for (auto& kv : beta_nonc){
        outfile->Printf(" MO = %zu, s = %e",kv.first,kv.second);
    }

    double Stilde = Sta * Stb * detUValpha * detUVbeta;

    size_t num_alpha_nonc = alpha_nonc.size();
    size_t num_beta_nonc = beta_nonc.size();

    outfile->Printf("\n  Stilde = %.6f\n",Stilde);
    outfile->Printf("\n  Noncoincidences: %da + %db \n",
                    num_alpha_nonc,num_beta_nonc);

    if(num_alpha_nonc == 0 and num_beta_nonc == 0){
        overlap = Stilde;
        // Build the W^BA alpha density matrix
        SharedMatrix W_BA_a = build_W_c1(ACa,BCa,s_a,nocc_a);
        SharedMatrix W_BA_b = build_W_c1(ACb,BCb,s_b,nocc_b);

        // Contract h with W^BA
        double WH_a = W_BA_a->vector_dot(Hao_);
        double WH_b  = W_BA_b->vector_dot(Hao_);
        double one_body = WH_a + WH_b;

        J(W_BA_a);
        double WaJWa = 0.5 * W_BA_a->vector_dot(TempMatrix);
        double WbJWa = W_BA_b->vector_dot(TempMatrix);
        J(W_BA_b);
        double WbJWb = 0.5 * W_BA_b->vector_dot(TempMatrix);

        K(W_BA_a);
        double WaKWa = -0.5 * W_BA_a->vector_dot(TempMatrix);
        K(W_BA_b);
        double WbKWb = -0.5 * W_BA_b->vector_dot(TempMatrix);

        double two_body = WaJWa + WbJWa + WbJWb + WaKWa + WbKWb;
        hamiltonian = Stilde * (nuclearrep_ + one_body + two_body);
    }
    else if(num_alpha_nonc == 1 and num_beta_nonc == 0){
        overlap = 0.0;
        // Build the W^BA alpha density matrix
        size_t i = alpha_nonc[0].first;
        SharedMatrix D_BA_i_a = build_D_i_c1(ACa,BCa,i);
        double one_body = D_BA_i_a->vector_dot(Hao_);

        // Build the W^BA alpha density matrix
        SharedMatrix W_BA_a = build_W_c1(ACa,BCa,s_a,nocc_a - 1); // <- exclude noncoincidence
        SharedMatrix W_BA_b = build_W_c1(ACb,BCb,s_b,nocc_b);

        J(W_BA_a);
        double DaJWa = D_BA_i_a->vector_dot(TempMatrix);
        J(W_BA_b);
        double DaJWb = D_BA_i_a->vector_dot(TempMatrix);
        K(W_BA_a);
        double DaKWa = -D_BA_i_a->vector_dot(TempMatrix);

        double two_body = DaJWa + DaKWa + DaJWb;
        hamiltonian = Stilde * (one_body + two_body);
    }
    else if(num_alpha_nonc == 0 and num_beta_nonc == 1){
        overlap = 0.0;
        // Build the W^BA alpha density matrix
        size_t i = beta_nonc[0].first;
        SharedMatrix D_BA_i_b = build_D_i_c1(ACb,BCb,i);
        double one_body = D_BA_i_b->vector_dot(Hao_);

        // Build the W^BA alpha density matrix
        SharedMatrix W_BA_a = build_W_c1(ACa,BCa,s_a,nocc_a);
        SharedMatrix W_BA_b = build_W_c1(ACb,BCb,s_b,nocc_b - 1); // <- exclude noncoincidence

        J(W_BA_a);
        double DbJWa = D_BA_i_b->vector_dot(TempMatrix);
        J(W_BA_b);
        double DbJWb = D_BA_i_b->vector_dot(TempMatrix);
        K(W_BA_b);
        double DbKWb = -D_BA_i_b->vector_dot(TempMatrix);

        double two_body = DbJWa + DbKWb + DbJWb;
        hamiltonian = Stilde * (one_body + two_body);
    }
    else if(num_alpha_nonc == 2 and num_beta_nonc == 0){
        overlap = 0.0;
        // Build the W^BA alpha density matrix
        size_t i = alpha_nonc[0].first;
        size_t j = alpha_nonc[1].first;
        SharedMatrix D_BA_i_a = build_D_i_c1(ACa,BCa,i);
        SharedMatrix D_BA_j_a = build_D_i_c1(ACa,BCa,j);

        J(D_BA_j_a);
        double DaJDa = D_BA_i_a->vector_dot(TempMatrix);
        K(D_BA_j_a);
        double DaKDa = -D_BA_i_a->vector_dot(TempMatrix);

        double two_body = DaJDa + DaKDa;
        hamiltonian = Stilde * two_body;
    }
    else if(num_alpha_nonc == 0 and num_beta_nonc == 2){
        overlap = 0.0;
        // Build the W^BA alpha density matrix
        size_t i = beta_nonc[0].first;
        size_t j = beta_nonc[1].first;
        SharedMatrix D_BA_i_b = build_D_i_c1(ACb,BCb,i);
        SharedMatrix D_BA_j_b = build_D_i_c1(ACb,BCb,j);

        J(D_BA_j_b);
        double DbJDb = D_BA_i_b->vector_dot(TempMatrix);
        K(D_BA_j_b);
        double DbKDb = -D_BA_i_b->vector_dot(TempMatrix);

        double two_body = DbJDb + DbKDb;
        hamiltonian = Stilde * two_body;
    }
    else if(num_alpha_nonc == 1 and num_beta_nonc == 1){
        overlap = 0.0;
        // Build the W^BA alpha density matrix
        size_t i = alpha_nonc[0].first;
        size_t j = beta_nonc[0].first;
        SharedMatrix D_BA_i_a = build_D_i_c1(ACa,BCa,i);
        SharedMatrix D_BA_j_b = build_D_i_c1(ACb,BCb,j);

        J(D_BA_j_b);
        double DaJDb = D_BA_i_a->vector_dot(TempMatrix);

        hamiltonian = Stilde * DaJDb;
    }
    return std::make_pair(overlap,hamiltonian);
}

SharedMatrix NOCI_Hamiltonian::build_W_c1(SharedMatrix CA, SharedMatrix CB, SharedVector s, size_t nocc)
{
    SharedMatrix W_BA = factory_->create_shared_matrix("W_BA");
    {
        double** Wp = W_BA->pointer(0);
        double** CAp = CA->pointer(0);
        double** CBp = CB->pointer(0);
        double* sp = s->pointer(0);
        for (size_t m = 0; m < nso; ++m){
            for (size_t n = 0; n < nso; ++n){
                double Wmn = 0.0;
                for (size_t i = 0; i < nocc; ++i){
                    Wmn += CBp[m][i] * CAp[n][i] / sp[i];
                }
                Wp[m][n] = Wmn;
            }
        }
    }
    return W_BA;
}

SharedMatrix NOCI_Hamiltonian::build_D_i_c1(SharedMatrix CA, SharedMatrix CB, size_t i)
{
    SharedMatrix D_BA = factory_->create_shared_matrix("D_BA");
    {
        double** Dp = D_BA->pointer(0);
        double** CAp = CA->pointer(0);
        double** CBp = CB->pointer(0);
        for (size_t m = 0; m < nso; ++m){
            for (size_t n = 0; n < nso; ++n){
                Dp[m][n] += CBp[m][i] * CAp[n][i];
            }
        }
    }
    return D_BA;
}

boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double>
NOCI_Hamiltonian::corresponding_orbitals(SharedMatrix A, SharedMatrix B, Dimension dima, Dimension dimb)
{
    // Form <B|S|A>
    TempMatrix->gemm(false,false,1.0,Sao_,A,0.0);
    TempMatrix2->gemm(true,false,1.0,B,TempMatrix,0.0);

    // Extract the occupied blocks only
    SharedMatrix Sba = SharedMatrix(new Matrix("Sba",dimb,dima));
    for (int h = 0; h < nirrep_; ++h) {
        int naocc = dima[h];
        int nbocc = dimb[h];
        double** Sba_h = Sba->pointer(h);
        double** S_h = TempMatrix2->pointer(h);
        for (int i = 0; i < nbocc; ++i){
            for (int j = 0; j < naocc; ++j){
                Sba_h[i][j] = S_h[i][j];
            }
        }
    }
    //    Sba->print();

    // SVD <B|S|A>
    boost::tuple<SharedMatrix, SharedVector, SharedMatrix> UsV = Sba->svd_a_temps();
    SharedMatrix U = UsV.get<0>();
    SharedVector sigma = UsV.get<1>();
    SharedMatrix V = UsV.get<2>();
    Sba->svd_a(U,sigma,V);
    //    sigma->print();
    //    U->print();
    //    V->print();

    // II. Transform the occupied orbitals to the new representation
    // Transform A with V (need to transpose V since svd returns V^T)
    // Extract the
    TempMatrix->identity();
    for (int h = 0; h < nirrep_; ++h) {
        int rows = V->rowdim(h);
        int cols = V->coldim(h);
        double** V_h = V->pointer(h);
        double** T_h = TempMatrix->pointer(h);
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j){
                T_h[i][j] = V_h[i][j];
            }
        }
    }
    TempMatrix2->gemm(false,true,1.0,A,TempMatrix,0.0);
    SharedMatrix cA = SharedMatrix(new Matrix("Corresponding " + A->name(),A->rowspi(),dima));
    copy_subblock(TempMatrix2,cA,cA->rowspi(),dima,true);

    // Transform B with U
    TempMatrix->identity();
    for (int h = 0; h < nirrep_; ++h) {
        int rows = U->rowdim(h);
        int cols = U->coldim(h);
        double** U_h = U->pointer(h);
        double** T_h = TempMatrix->pointer(h);
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j){
                T_h[i][j] = U_h[i][j];
            }
        }
    }
    TempMatrix2->gemm(false,false,1.0,B,TempMatrix,0.0);
    SharedMatrix cB = SharedMatrix(new Matrix("Corresponding " + B->name(),B->rowspi(),dimb));
    copy_subblock(TempMatrix2,cB,cB->rowspi(),dimb,true);


    // Find the product of the determinants of U and V
    double detU = 1.0;
    for (int h = 0; h < nirrep_; ++h) {
        int nmo = U->rowdim(h);
        if(nmo > 1){
            double d = 1.0;
            int* indx = new int[nmo];
            double** ptrU = U->pointer(h);
            ludcmp(ptrU,nmo,indx,&d);
            detU *= d;
            for (int i = 0; i < nmo; ++i){
                detU *= ptrU[i][i];
            }
            delete[] indx;
        }
    }
    double detV = 1.0;
    for (int h = 0; h < nirrep_; ++h) {
        int nmo = V->rowdim(h);
        if(nmo > 1){
            double d = 1.0;
            int* indx = new int[nmo];
            double** ptrV = V->pointer(h);
            ludcmp(ptrV,nmo,indx,&d);
            detV *= d;
            for (int i = 0; i < nmo; ++i){
                detV *= ptrV[i][i];
            }
            delete[] indx;
        }
    }
    outfile->Printf("\n det U = %f, det V = %f",detU,detV);
    double detUV = detU * detV;
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double> result(cA,cB,sigma,detUV);
    return result;
}

/*
std::pair<double,double> NOCI_Hamiltonian::matrix_element(SharedDeterminant A, SharedDeterminant B)
{
    double overlap = 0.0;
    double hamiltonian = 0.0;

    // I. Form the corresponding alpha and beta orbitals
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double> calpha = corresponding_orbitals(A->Ca(),B->Ca(),A->nalphapi(),B->nalphapi());
    boost::tuple<SharedMatrix,SharedMatrix,SharedVector,double> cbeta  = corresponding_orbitals(A->Cb(),B->Cb(),A->nbetapi(),B->nbetapi());
    SharedMatrix ACa = calpha.get<0>();
    SharedMatrix BCa = calpha.get<1>();
    SharedMatrix ACb = cbeta.get<0>();
    SharedMatrix BCb = cbeta.get<1>();
    double detUValpha = calpha.get<3>();
    double detUVbeta = cbeta.get<3>();
    SharedVector s_a = calpha.get<2>();
    SharedVector s_b = cbeta.get<2>();

    // Compute the number of noncoincidences
    double noncoincidence_threshold = 1.0e-9;

    std::vector<boost::tuple<int,int,double> > Aalpha_nonc;
    std::vector<boost::tuple<int,int,double> > Balpha_nonc;
    double Sta = 1.0;
    for (int h = 0; h < nirrep_; ++h){
        // Count all the numerical noncoincidences
        int nmin = std::min(A->nalphapi()[h],B->nalphapi()[h]);
        for (int p = 0; p < nmin; ++p){
            if(std::fabs(s_a->get(h,p)) >= noncoincidence_threshold){
                Sta *= s_a->get(h,p);
            }else{
                Aalpha_nonc.push_back(boost::make_tuple(h,p,s_a->get(h,p)));
                Balpha_nonc.push_back(boost::make_tuple(h,p,s_a->get(h,p)));
            }
        }
        // Count all the symmetry noncoincidences
        int nmax = std::max(A->nalphapi()[h],B->nalphapi()[h]);
        bool AgeB = A->nalphapi()[h] >= B->nalphapi()[h] ? true : false;
        for (int p = nmin; p < nmax; ++p){
            if(AgeB){
                Aalpha_nonc.push_back(boost::make_tuple(h,p,0.0));
            }else{
                Balpha_nonc.push_back(boost::make_tuple(h,p,0.0));
            }
        }
    }

    std::vector<boost::tuple<int,int,double> > Abeta_nonc;
    std::vector<boost::tuple<int,int,double> > Bbeta_nonc;
    double Stb = 1.0;
    for (int h = 0; h < nirrep_; ++h){
        // Count all the numerical noncoincidences
        int nmin = std::min(A->nbetapi()[h],B->nbetapi()[h]);
        for (int p = 0; p < nmin; ++p){
            if(std::fabs(s_b->get(h,p)) >= noncoincidence_threshold){
                Stb *= s_b->get(h,p);
            }else{
                Abeta_nonc.push_back(boost::make_tuple(h,p,s_b->get(h,p)));
                Bbeta_nonc.push_back(boost::make_tuple(h,p,s_b->get(h,p)));
            }
        }
        // Count all the symmetry noncoincidences
        int nmax = std::max(A->nbetapi()[h],B->nbetapi()[h]);
        bool AgeB = A->nbetapi()[h] >= B->nbetapi()[h] ? true : false;
        for (int p = nmin; p < nmax; ++p){
            if(AgeB){
                Abeta_nonc.push_back(boost::make_tuple(h,p,0.0));
            }else{
                Bbeta_nonc.push_back(boost::make_tuple(h,p,0.0));
            }
        }
    }
    outfile->Printf("\n  Corresponding orbitals:\n");
    outfile->Printf("  A(alpha): ");
    for (size_t k = 0; k < Aalpha_nonc.size(); ++k){
        int i_h = Aalpha_nonc[k].get<0>();
        int i_mo = Aalpha_nonc[k].get<1>();
        outfile->Printf(" (%1d,%2d)",i_h,i_mo);
    }
    outfile->Printf("\n  B(alpha): ");
    for (size_t k = 0; k < Balpha_nonc.size(); ++k){
        int i_h = Balpha_nonc[k].get<0>();
        int i_mo = Balpha_nonc[k].get<1>();
        outfile->Printf(" (%1d,%2d)",i_h,i_mo);
    }
    outfile->Printf("\n  s(alpha): ");
    for (size_t k = 0; k < Balpha_nonc.size(); ++k){
        double i_s = Balpha_nonc[k].get<2>();
        outfile->Printf(" %6e",i_s);
    }
    outfile->Printf("\n  A(beta):  ");
    for (size_t k = 0; k < Abeta_nonc.size(); ++k){
        int i_h = Abeta_nonc[k].get<0>();
        int i_mo = Abeta_nonc[k].get<1>();
        outfile->Printf(" (%1d,%2d)",i_h,i_mo);
    }
    outfile->Printf("\n  B(beta):  ");
    for (size_t k = 0; k < Bbeta_nonc.size(); ++k){
        int i_h = Bbeta_nonc[k].get<0>();
        int i_mo = Bbeta_nonc[k].get<1>();
        outfile->Printf(" (%1d,%2d)",i_h,i_mo);
    }
    outfile->Printf("\n  s(beta):  ");
    for (size_t k = 0; k < Balpha_nonc.size(); ++k){
        double i_s = Bbeta_nonc[k].get<2>();
        outfile->Printf(" %6e",i_s);
    }

    double Stilde = Sta * Stb * detUValpha * detUVbeta;
    outfile->Printf("\n  Stilde = %.6f\n",Stilde);

    int num_alpha_nonc = static_cast<int>(Aalpha_nonc.size());
    int num_beta_nonc = static_cast<int>(Abeta_nonc.size());

    if(num_alpha_nonc + num_beta_nonc != 2){
        s_a->print();
        s_b->print();
    }
    outfile->Flush();

    if(num_alpha_nonc == 0 and num_beta_nonc == 0){
        overlap = Stilde;
        // Build the W^BA alpha density matrix
        SharedMatrix W_BA_a = factory_->create_shared_matrix("W_BA_a");
        SharedMatrix W_BA_b = factory_->create_shared_matrix("W_BA_b");
        for (int h = 0; h < nirrep_; ++h){
            int nocc = A->nalphapi()[h];  // NB in this case there cannot be symmetry noncoincidences
            int nso = nsopi_[h];
            double** W = W_BA_a->pointer(h);
            double** CA = ACa->pointer(h);
            double** CB = BCa->pointer(h);
            double* s = s_a->pointer(h);
            for (int m = 0; m < nso; ++m){
                for (int n = 0; n < nso; ++n){
                    double Wmn = 0.0;
                    for (int i = 0; i < nocc; ++i){
                        Wmn += CB[m][i] * CA[n][i] / s[i];
                    }
                    W[m][n] = Wmn;
                }
            }
        }
        for (int h = 0; h < nirrep_; ++h){
            int nocc = A->nbetapi()[h];  // NB in this case there cannot be symmetry noncoincidences
            int nso = nsopi_[h];
            double** W = W_BA_b->pointer(h);
            double** CA = ACb->pointer(h);
            double** CB = BCb->pointer(h);
            double* s = s_b->pointer(h);
            for (int m = 0; m < nso; ++m){
                for (int n = 0; n < nso; ++n){
                    double Wmn = 0.0;
                    for (int i = 0; i < nocc; ++i){
                        Wmn += CB[m][i] * CA[n][i] / s[i];
                    }
                    W[m][n] = Wmn;
                }
            }
        }


        double WH_a = W_BA_a->vector_dot(Hao_);
        double WH_b  = W_BA_b->vector_dot(Hao_);
        double one_body = WH_a + WH_b;

        SharedMatrix scaled_BCa = BCa->clone();
        SharedMatrix scaled_BCb = BCb->clone();
        SharedMatrix scaled_ACa = ACa->clone();
        SharedMatrix scaled_ACb = ACb->clone();


        for (int h = 0; h < nirrep_; ++h){
            int nocc = A->nalphapi()[h];  // NB in this case there cannot be symmetry noncoincidences
            int nso = nsopi_[h];
            double** CA = scaled_ACa->pointer(h);
            double** CB = scaled_BCa->pointer(h);
            double* s = s_a->pointer(h);
            for (int m = 0; m < nso; ++m){
                for (int i = 0; i < nocc; ++i){
                    CB[m][i] /= s[i];
                    //                    CA[m][i] /= std::sqrt(s[i]);
                    //                    CB[m][i] /= std::sqrt(s[i]);
                }
            }
        }
        for (int h = 0; h < nirrep_; ++h){
            int nocc = A->nbetapi()[h];  // NB in this case there cannot be symmetry noncoincidences
            int nso = nsopi_[h];
            double** CA = scaled_ACb->pointer(h);
            double** CB = scaled_BCb->pointer(h);
            double* s = s_b->pointer(h);
            for (int m = 0; m < nso; ++m){
                for (int i = 0; i < nocc; ++i){
                    CB[m][i] /= s[i];
                    //                    CA[m][i] /= std::sqrt(s[i]);
                    //                    CB[m][i] /= std::sqrt(s[i]);
                }
            }
        }
        boost::shared_ptr<IntegralFactory> integral_(new IntegralFactory(basisset_));
        boost::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
//        int nbf = basisset_->nbf();
        SharedMatrix SO2AO_ = pet->sotoao();


        double c2 = 0.0;

        int a_alpha_h  = Aalpha_nonc[0].get<0>();
        int b_alpha_h  = Balpha_nonc[0].get<0>();
        int a_alpha_mo = Aalpha_nonc[0].get<1>();
        int b_alpha_mo  = Balpha_nonc[0].get<1>();
        int a_beta_h   = Abeta_nonc[0].get<0>();
        int a_beta_mo  = Abeta_nonc[0].get<1>();
        int b_beta_h   = Bbeta_nonc[0].get<0>();
        int b_beta_mo  = Bbeta_nonc[0].get<1>();

        SharedVector Ava = ACa->get_column(a_alpha_h,a_alpha_mo);
        SharedVector Bva = BCa->get_column(b_alpha_h,b_alpha_mo);
        SharedVector Avb = ACb->get_column(a_beta_h,a_beta_mo);
        SharedVector Bvb = BCb->get_column(b_beta_h,b_beta_mo);


        double* Ci = Bva->pointer();
        double* Cj = Ava->pointer();
        double* Ck = Bvb->pointer();
        double* Cl = Avb->pointer();
        for (int i = 0; i < nsopi_[0]; ++i){
            for (int j = 0; j < nsopi_[0]; ++j){
                for (int k = 0; k < nsopi_[0]; ++k){
                    for (int l = 0; l < nsopi_[0]; ++l){
                        c2 += tei_ints_[INDEX4(i,j,k,l)] * Ci[i] * Cj[j] * Ck[k] * Cl[l];
                    }
                }
            }
        }
        outfile->Printf("  Matrix element from ints    = %20.12f\n",c2);

    }
    /*
        std::vector<SharedMatrix>& C_left = jk_->C_left();
        C_left.clear();
        C_left.push_back(scaled_BCa);
        C_left.push_back(scaled_BCb);
        std::vector<SharedMatrix>& C_right = jk_->C_right();
        C_right.clear();
        C_right.push_back(scaled_ACa);
        C_right.push_back(scaled_ACb);
        jk_->compute();
        const std::vector<SharedMatrix >& Dn = jk_->D();

        SharedMatrix Ja = jk_->J()[0];
        SharedMatrix Jb = jk_->J()[1];
        SharedMatrix Ka = jk_->K()[0];
        SharedMatrix Kb = jk_->K()[1];
        double WJW_aa = Ja->vector_dot(W_BA_a);
        double WJW_bb = Jb->vector_dot(W_BA_b);
        double WJW_ba = Jb->vector_dot(W_BA_a);
        W_BA_a->transpose_this();
        W_BA_b->transpose_this();
        double WKW_aa = Ka->vector_dot(W_BA_a);
        double WKW_bb = Kb->vector_dot(W_BA_b);

        double two_body = 0.5 * (WJW_aa + WJW_bb + 2.0 * WJW_ba - WKW_aa - WKW_bb);

        double interaction = nuclearrep_ + one_body + two_body;

        hamiltonian = interaction * Stilde;
        outfile->Printf("  Matrix element from libfock = %14.6f (Stilde) * %14.6f (int) = %20.12f\n", Stilde, interaction, hamiltonian);
        outfile->Printf("  W_a . h = %20.12f\n", WH_a);
        outfile->Printf("  W_b . h = %20.12f\n", WH_b);
        outfile->Printf("  W_a . J(W_a) = %20.12f\n", WJW_aa);
        outfile->Printf("  W_b . J(W_b) = %20.12f\n", WJW_bb);
        outfile->Printf("  W_b . J(W_a) = %20.12f\n", WJW_ba);
        outfile->Printf("  W_a . K(W_a) = %20.12f\n", WKW_aa);
        outfile->Printf("  W_b . K(W_b) = %20.12f\n", WKW_bb);

        outfile->Printf("  W . h = %20.12f\n", one_body);
        outfile->Printf("  1/2 W . J - 1/2 Wa . Ka - 1/2 Wb . Kb = %20.12f\n", two_body);

//        outfile->Printf("  E1 = %20.12f\n", pc_hartree2ev * ((E_ + hamiltonian)/(1+overlap) - ground_state_energy) );
//        outfile->Printf("  E2 = %20.12f\n", pc_hartree2ev * ((E_ - hamiltonian)/(1-overlap) - ground_state_energy) );
//        outfile->Printf("  E1-E2 = %20.12f\n", pc_hartree2ev * ((E_ + hamiltonian)/(1+overlap) - (E_ - hamiltonian)/(1-overlap)));

        C_left = jk_->C_left();
        C_left.clear();
        C_left.push_back(Ca_subset("SO", "OCC"));
        C_left.push_back(Cb_subset("SO", "OCC"));
        C_right = jk_->C_right();
        C_right.clear();
        jk_->compute();

        Ja = jk_->J()[0];
        Jb = jk_->J()[1];
        Ka = jk_->K()[0];
        Kb = jk_->K()[1];

        double one_electron_E = Da_->vector_dot(H_);
        one_electron_E += Db_->vector_dot(H_);
        J_->copy(Ja);
        J_->add(Jb);
        double coulomb_E = 0.5 * Da_->vector_dot(J_);
        coulomb_E += 0.5 * Db_->vector_dot(J_);
        double exchange_E = - Da_->vector_dot(Ka_);
        exchange_E -= Db_->vector_dot(Kb_);
        double two_electron_E = coulomb_E + exchange_E;

        double E_HF_Phip = nuclearrep_ + one_electron_E + coulomb_E + 0.5 * exchange_E;
        outfile->Printf("  nuclearrep_ = %20.12f\n",nuclearrep_);
        outfile->Printf("  one_electron_E = %20.12f\n",one_electron_E);
        outfile->Printf("  two_electron_E = %20.12f\n",two_electron_E);
        outfile->Printf("  coulomb_E = %20.12f\n",coulomb_E);
        outfile->Printf("  exchange_E = %20.12f\n",exchange_E);
        outfile->Printf("  E_HF_Phi' = %20.12f\n",E_HF_Phip);

        double perfected_coupling = Stilde * (E_ + interaction - E_HF_Phip);
        outfile->Printf("  Matrix element from libfock = %14.6f (Stilde) * %14.6f (int) = %20.12f\n", Stilde, E_ + interaction - E_HF_Phip, perfected_coupling);
        hamiltonian = perfected_coupling;

//        SharedMatrix h_BA_a = SharedMatrix(new Matrix("h_BA_a",A->nalphapi(),B->nalphapi()));
//        h_BA_a->transform(BCa,Hao_,ACa);
//        double alpha_one_body2 = 0.0;
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = A->nalphapi()[h];  // NB in this case there cannot be symmetry noncoincidences
//            double** h_BA = h_BA_a->pointer(h);
//            double* s = s_a->pointer(h);
//            for (int i = 0; i < nocc; ++i){
//                alpha_one_body2 += h_BA[i][i] / s[i];
//            }
//        }

//        SharedMatrix h_BA_b = SharedMatrix(new Matrix("h_BA_b",A->nbetapi(),B->nbetapi()));
//        h_BA_b->transform(BCb,Hao_,ACb);
//        double beta_one_body2 = 0.0;
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = A->nbetapi()[h];  // NB in this case there cannot be symmetry noncoincidences
//            double** h_BA = h_BA_b->pointer(h);
//            double* s = s_b->pointer(h);
//            for (int i = 0; i < nocc; ++i){
//                beta_one_body2 += h_BA[i][i] / s[i];
//            }
//        }
//        double one_body2 = alpha_one_body2 + beta_one_body2;
//        double interaction2 = one_body2;
//        outfile->Printf("  Matrix element from libfock = %14.6f (Stilde) * %14.6f (int) = %20.12f\n", Stilde, interaction2, interaction2 * Stilde);
//        outfile->Printf("  W_a . h = %20.12f\n", alpha_one_body2);
//        outfile->Printf("  W_b . h = %20.12f\n", beta_one_body2);
//        fflush(outfile);






    }else if(num_alpha_nonc == 1 and num_beta_nonc == 0){
        overlap = 0.0;
//        throw FeatureNotImplemented("CKS", "H in the case of one noncoincidence", __FILE__, __LINE__);
    }else if(num_alpha_nonc == 0 and num_beta_nonc == 1){
        overlap = 0.0;
//        throw FeatureNotImplemented("CKS", "H in the case of one noncoincidence", __FILE__, __LINE__);
    }else if(num_alpha_nonc == 1 and num_beta_nonc == 1){
        overlap = 0.0;
        int a_alpha_h  = Aalpha_nonc[0].get<0>();
        int a_alpha_mo = Aalpha_nonc[0].get<1>();
        int b_alpha_h  = Balpha_nonc[0].get<0>();
        int b_alpha_mo = Balpha_nonc[0].get<1>();
        int a_beta_h   = Abeta_nonc[0].get<0>();
        int a_beta_mo  = Abeta_nonc[0].get<1>();
        int b_beta_h   = Bbeta_nonc[0].get<0>();
        int b_beta_mo  = Bbeta_nonc[0].get<1>();

        // A_b absorbs the symmetry of B_b
        Dimension A_beta_dim(nirrep_,"A_beta_dim");
        A_beta_dim[b_beta_h] = 1;
        // B_b is total symmetric
        Dimension B_beta_dim(nirrep_,"B_beta_dim");
        B_beta_dim[b_beta_h] = 1;
        SharedMatrix A_b(new Matrix("A_b", nsopi_, A_beta_dim,a_beta_h ^ b_beta_h));
        SharedMatrix B_b(new Matrix("B_b", nsopi_, B_beta_dim));
        for (int m = 0; m < nsopi_[a_beta_h]; ++m){
            A_b->set(a_beta_h,m,0,ACb->get(a_beta_h,m,a_beta_mo));
        }
        for (int m = 0; m < nsopi_[b_beta_h]; ++m){
            B_b->set(b_beta_h,m,0,BCb->get(b_beta_h,m,b_beta_mo));
        }

        std::vector<SharedMatrix>& C_left = jk_->C_left();
        C_left.clear();
        C_left.push_back(B_b);
        std::vector<SharedMatrix>& C_right = jk_->C_right();
        C_right.clear();
        C_right.push_back(A_b);
        jk_->compute();
        const std::vector<SharedMatrix >& Dn = jk_->D();

        SharedMatrix Jnew = jk_->J()[0];

        SharedMatrix D = SharedMatrix(new Matrix("D",nirrep_, nsopi_, nsopi_, a_alpha_h ^ b_alpha_h));
        D->zero();
        double** D_h = D->pointer(b_alpha_h);
        double* Dp = &(D_h[0][0]);
        for (int n = 0; n < nsopi_[a_alpha_h]; ++n){
            for (int m = 0; m < nsopi_[b_alpha_h]; ++m){
                D_h[m][n] = BCa->get(b_alpha_h,m,b_alpha_mo) * ACa->get(a_alpha_h,n,a_alpha_mo);
            }
        }
        double twoelint = Jnew->vector_dot(D);
        hamiltonian = twoelint * std::fabs(Stilde);
        outfile->Printf("\n\n  Warning, the code is using the absolute value of Stilde.  Hope for the best!\n\n");
        outfile->Printf("  Matrix element from libfock = |%.6f| (Stilde) * %14.6f (int) = %20.12f\n", Stilde, twoelint, hamiltonian);
    }else if(num_alpha_nonc == 2 and num_beta_nonc == 0){
        overlap = 0.0;
//        throw FeatureNotImplemented("CKS", "H in the case of two alpha noncoincidences", __FILE__, __LINE__);
    }else if(num_alpha_nonc == 0 and num_beta_nonc == 2){
        overlap = 0.0;
//        throw FeatureNotImplemented("CKS", "H in the case of two beta noncoincidences", __FILE__, __LINE__);
    }
    outfile->Flush();

    return std::make_pair(overlap,hamiltonian);
}
*/


//        SharedMatrix W_BA_a = factory_->create_shared_matrix("W_BA_a");
//        SharedMatrix W_BA_a = factory_->create_shared_matrix("W_BA_a");
//        {
//            double** W = W_BA_a->pointer(0);
//            double** CA = ACa->pointer(0);
//            double** CB = BCa->pointer(0);
//            double* s = s_a->pointer(0);
//            for (size_t m = 0; m < nso; ++m){
//                for (size_t n = 0; n < nso; ++n){
//                    double Wmn = 0.0;
//                    for (size_t i = 0; i < nocc_a; ++i){
//                        Wmn += CB[m][i] * CA[n][i] / s[i];
//                    }
//                    W[m][n] = Wmn;
//                }
//            }
//        }
//        SharedMatrix W_BA_b = factory_->create_shared_matrix("W_BA_b");
//        {
//            double** W = W_BA_b->pointer(0);
//            double** CA = ACb->pointer(0);
//            double** CB = BCb->pointer(0);
//            double* s = s_b->pointer(0);
//            for (size_t m = 0; m < nso; ++m){
//                for (size_t n = 0; n < nso; ++n){
//                    double Wmn = 0.0;
//                    for (size_t i = 0; i < nocc_b; ++i){
//                        Wmn += CB[m][i] * CA[n][i] / s[i];
//                    }
//                    W[m][n] = Wmn;
//                }
//            }
//        }
}} // Namespaces

