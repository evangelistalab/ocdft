#include "fasnocis.h"

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
#include "helpers.h"

#define DEBUG_NOSCF 1

#define DEBUG_THIS2(EXP) \
    outfile->Printf("\n  Starting " #EXP " ..."); fflush(outfile); \
    EXP \
    outfile->Printf("  done."); fflush(outfile); \


using namespace psi;

namespace psi{ namespace scf{

FASNOCIS::FASNOCIS(Options &options, boost::shared_ptr<PSIO> psio)
    : UHF(options, psio), do_excitation(false)
{
}

FASNOCIS::FASNOCIS(Options &options, boost::shared_ptr<PSIO> psio,
                   boost::shared_ptr<Wavefunction> ref_scf,
                   std::vector<std::pair<size_t,size_t>> vec_frozen_mos,
                   std::vector<size_t> aocc,
                   std::vector<size_t> bocc,
                   Dimension nadoccpi)
    : UHF(options,psio), ref_scf_(ref_scf), vec_frozen_mos_(vec_frozen_mos),
      afocc_(aocc), bfocc_(bocc), nadoccpi_(nadoccpi), do_excitation(true)
{
    Ft_ = factory_->create_shared_matrix("Total Fock matrix (SO basis)");
    form_C_frozen();
    form_C_active();
    combine_C_frozen_active();
}

FASNOCIS::~FASNOCIS()
{
}

void FASNOCIS::form_C_frozen()
{
    SharedMatrix Ca_ref = ref_scf_->Ca();
    SharedMatrix Cb_ref = ref_scf_->Cb();

    nfmopi_ = Dimension(nirrep_,"Number of frozen MOs per irrep");
    for (auto& kv : vec_frozen_mos_){
        nfmopi_[kv.first] += 1;
    }
    Ca_f_ = SharedMatrix(new Matrix("C alpha frozen",nsopi_,nfmopi_));
    Cb_f_ = SharedMatrix(new Matrix("C beta frozen",nsopi_,nfmopi_));

    for (size_t p = 0; p < vec_frozen_mos_.size(); ++p){
        if (std::find(afocc_.begin(), afocc_.end(), p) == afocc_.end()){
            afvir_.push_back(p);
        }
        if (std::find(bfocc_.begin(), bfocc_.end(), p) == bfocc_.end()){
            bfvir_.push_back(p);
        }
    }

    for (size_t i : afocc_){
        outfile->Printf("\n  Alpha frozen occupied MO: %zu",i);
    }
    for (size_t i : bfocc_){
        outfile->Printf("\n  Beta  frozen occupied MO: %zu",i);
    }
    for (size_t i : afvir_){
        outfile->Printf("\n  Alpha frozen occupied MO: %zu",i);
    }
    for (size_t i : bfvir_){
        outfile->Printf("\n  Beta  frozen occupied MO: %zu",i);
    }

    // Copy the elements of C
    Dimension afmo_count(nirrep_);
    for (auto& vec : {afocc_,afvir_}){
        for (size_t i : vec){
            size_t h = vec_frozen_mos_[i].first;
            size_t mo = vec_frozen_mos_[i].second;
            outfile->Printf("\n  Copying orbital %zu (%zu,%zu)",i,h,mo);
            SharedVector column = Ca_ref->get_column(h,mo);
            Ca_f_->set_column(h,afmo_count[h],column);
            afmo_count[h] += 1;
        }
    }

    Dimension bfmo_count(nirrep_);
    for (auto& vec : {bfocc_,bfvir_}){
        for (size_t i : vec){
            size_t h = vec_frozen_mos_[i].first;
            size_t mo = vec_frozen_mos_[i].second;
            SharedVector column = Cb_ref->get_column(h,mo);
            Cb_f_->set_column(h,bfmo_count[h],column);
            bfmo_count[h] += 1;
        }
    }
#if DEBUG_NOSCF
    Ca_ref->print();
    Ca_f_->print();
    Cb_f_->print();
#endif
}

void FASNOCIS::form_C_active()
{
    SharedMatrix Ca_ref = ref_scf_->Ca();
    SharedMatrix Cb_ref = ref_scf_->Cb();

    // Recompute the number of alpha and beta electrons per irrep
    for (int h = 0; h < nirrep_; ++h){
        nalphapi_[h] = nadoccpi_[h];
        nbetapi_[h] = nadoccpi_[h];
    }
    for (size_t i : afocc_){
        size_t h = vec_frozen_mos_[i].first;
        nalphapi_[h] += 1;
    }
    for (size_t i : bfocc_){
        size_t h = vec_frozen_mos_[i].first;
        nbetapi_[h] += 1;
    }

    nalphapi_.print();
    nbetapi_.print();

    namopi_ = Dimension(nirrep_,"Number of active MOs per irrep");
    nauoccpi_ = Dimension(nirrep_,"Number of unoccupied active MOs per irrep");
    // Form the list of active MOs
    for (size_t h = 0; h < nirrep_; ++h){
        for (size_t p = 0; p < nmopi_[h]; ++p){
            std::pair<size_t,size_t> h_p(h,p);
            if (std::find(vec_frozen_mos_.begin(), vec_frozen_mos_.end(), h_p) == vec_frozen_mos_.end()){
                vec_active_mos_.push_back(h_p);
                namopi_[h] += 1;
#if DEBUG_NOSCF
                outfile->Printf("\n  Adding MO (h = %zu, mo = %zu) to the vector of active MOs",h,p);
#endif
            }
        }
        nauoccpi_[h] = namopi_[h] - nadoccpi_[h];
    }

    namopi_.print();
    nauoccpi_.print();

    Ca_a_ = SharedMatrix(new Matrix("C alpha active",nsopi_,namopi_));
    Cb_a_ = SharedMatrix(new Matrix("C beta active",nsopi_,namopi_));
    Ft_amo_ = SharedMatrix(new Matrix("Total Fock matrix in the active MOs",namopi_,namopi_));
    U_amo_ = SharedMatrix(new Matrix("Eigenvectors of Ft_amo",namopi_,namopi_));
    lambda_amo_ = SharedVector(new Vector("Eigenvalues of Ft_amo",namopi_));


    // Copy the elements of C
    Dimension amo_count(nirrep_);
    for (auto h_p : vec_active_mos_){
        size_t h = h_p.first;
        size_t mo = h_p.second;
#if DEBUG_NOSCF
        outfile->Printf("\n  Copying orbital (%zu,%zu) to C_active",h,mo);
#endif
        SharedVector column = Ca_ref->get_column(h,mo);
        Ca_a_->set_column(h,amo_count[h],column);
        Cb_a_->set_column(h,amo_count[h],column);
        amo_count[h] += 1;
    }
#if DEBUG_NOSCF
    Ca_a_->print();
    Cb_a_->print();
#endif
}

void FASNOCIS::combine_C_frozen_active()
{
    Dimension zerodim(nirrep_);
    Ca_->zero();
    Cb_->zero();
    // Insert all the doubly occupied active MOs
    copy_block(Ca_a_,1.0,Ca_,1.0,nsopi_,nadoccpi_);
    copy_block(Cb_a_,1.0,Cb_,1.0,nsopi_,nadoccpi_);
#if DEBUG_NOSCF
    Ca_->print();
    Cb_->print();
#endif
    // Insert all the frozen MOs
    copy_block(Ca_f_,1.0,Ca_,1.0,nsopi_,nfmopi_,zerodim,zerodim,zerodim,nadoccpi_);
    copy_block(Cb_f_,1.0,Cb_,1.0,nsopi_,nfmopi_,zerodim,zerodim,zerodim,nadoccpi_);
#if DEBUG_NOSCF
    Ca_->print();
    Cb_->print();
#endif
    // Insert all the unoccupied active MOs
    Dimension offset = nadoccpi_ + nfmopi_;
    copy_block(Ca_a_,1.0,Ca_,1.0,nsopi_,nauoccpi_,zerodim,nadoccpi_,zerodim,offset);
    copy_block(Cb_a_,1.0,Cb_,1.0,nsopi_,nauoccpi_,zerodim,nadoccpi_,zerodim,offset);
#if DEBUG_NOSCF
    Ca_->print();
    Cb_->print();
#endif
}

void FASNOCIS::form_C()
{
    if(not do_excitation){
        UHF::form_C();
    }
    else{
        form_C_noscf();
    }
}

void FASNOCIS::form_C_noscf()
{
    Ft_->copy(Fa_);
    Ft_->add(Fb_);

    // Compute the Fock matrix in the MO basis
    Ft_amo_->zero();
    Ft_amo_->transform(Ft_,Ca_a_);

#if DEBUG_NOSCF
    Ft_amo_->print();
#endif

    Ft_amo_->diagonalize(U_amo_,lambda_amo_);

#if DEBUG_NOSCF
    U_amo_->print();
    lambda_amo_->print();
#endif

    Cb_a_->gemm(false,false,1.0,Ca_a_,U_amo_,0.0);
    Ca_a_->copy(Cb_a_);

#if DEBUG_NOSCF
    Cb_a_->print();
#endif

    combine_C_frozen_active();

#if DEBUG_NOSCF
    Ft_->print();
    Ft_amo_->print();
#endif
}

double FASNOCIS::compute_E()
{
    double one_electron_E = Dt_->vector_dot(H_);
    double two_electron_E = 0.5 * (Da_->vector_dot(Fa_) + Db_->vector_dot(Fb_) - one_electron_E);

    energies_["Nuclear"] = nuclearrep_;
    energies_["One-Electron"] = one_electron_E;
    energies_["Two-Electron"] = two_electron_E;
    energies_["XC"] = 0.0;
    energies_["-D"] = 0.0;

    double DH  = Dt_->vector_dot(H_);
    double DFa = Da_->vector_dot(Fa_);
    double DFb = Db_->vector_dot(Fb_);
    double Eelec = 0.5 * (DH + DFa + DFb);
    // outfile->Printf( "electronic energy = %20.14f\n", Eelec);
    double Etotal = nuclearrep_ + Eelec;
    return Etotal;
}

bool FASNOCIS::diis()
{
    if(not do_excitation){
        return UHF::diis();
    }
    return false;
}

}} // Namespaces
