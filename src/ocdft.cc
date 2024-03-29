#include <algorithm>

#include "fmt/format.h"

#include "psi4/libpsi4util/PsiOutStream.h"
#include <psi4/libmints/local.h>
#include <psi4/libmints/factory.h>
#include <psi4/libmints/pointgrp.h>
#include <psi4/libmints/petitelist.h>
#include <psi4/liboptions/liboptions.h>
#include "psi4/libfunctional/superfunctional.h"
#include <psi4/physconst.h>
#include <psi4/psifiles.h>
#include <psi4/libmints/oeprop.h>
#include <psi4/libmints/onebody.h>
#include <psi4/libmints/integral.h>
//#include <libfock/apps.h>
#include <psi4/libfock/jk.h>
#include <psi4/libfock/v.h>

#include <psi4/libcubeprop/cubeprop.h>
#include <psi4/libdisp/dispersion.h>
//#include <liboptions/liboptions.h>
#include <psi4/libciomr/libciomr.h>
#include <psi4/libiwl/iwl.hpp>
#include <psi4/libqt/qt.h>
#include <psi4/libscf_solver/hf.h>
#include <psi4/libdiis/diismanager.h>
#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/libpsi4util.h"
#include "psi4/libpsi4util/process.h"

#include "aosubspace.h"
#include "helpers.h"
#include "iao_builder.h"
#include "ocdft.h"

#define DEBUG_OCDFT 0

#define DEBUG_THIS2(EXP)                                                                           \
    outfile->Printf("\n  Starting " #EXP " ...");                                                  \
    fflush(outfile);                                                                               \
    EXP outfile->Printf("  done.");                                                                \
    fflush(outfile);

using namespace psi;

namespace psi {
namespace scf {

UOCDFT::UOCDFT(SharedWavefunction ref_scf, std::shared_ptr<SuperFunctional> functional,
               Options& options, std::shared_ptr<PSIO> psio)
    : UHF(ref_scf, functional, options, psio), do_excitation(false), do_symmetry(false),
      ground_state_energy(0.0), ground_state_symmetry_(0), excited_state_symmetry_(0), state_(0),
      singlet_exc_energy_s_plus_(0.0), triplet_exc_energy_s_plus(0.0), singlet_exc_energy_ci(0.0),
      triplet_exc_energy_ci(0.0), oscillator_strength_s_plus_(0.0),
      oscillator_strength_s_plus_x_(0.0), oscillator_strength_s_plus_y_(0.0),
      oscillator_strength_s_plus_z_(0.0), oscillator_strength_ci(0.0) {
    init();
    gs_Fa_ = Fa_;
    gs_Fb_ = Fb_;
    wfn_ = ref_scf;
    // std::vector<int> accepted_virts;
    // Ca_gs_ = dets[0]->Ca();
}

UOCDFT::UOCDFT(std::shared_ptr<SuperFunctional> functional, Options& options,
               std::shared_ptr<PSIO> psio, SharedWavefunction ref_scf, int state)
    : UHF(ref_scf, functional, options, psio), do_excitation(true), do_symmetry(false),
      ground_state_energy(0.0), ground_state_symmetry_(0), excited_state_symmetry_(0),
      state_(state), singlet_exc_energy_s_plus_(0.0), triplet_exc_energy_s_plus(0.0),
      singlet_exc_energy_ci(0.0), triplet_exc_energy_ci(0.0), oscillator_strength_s_plus_(0.0),
      oscillator_strength_s_plus_x_(0.0), oscillator_strength_s_plus_y_(0.0),
      oscillator_strength_s_plus_z_(0.0), oscillator_strength_ci(0.0) {
    init();
    init_excitation(ref_scf);
    wfn_ = ref_scf;
    ground_state_energy = dets[0]->energy();
    // std::vector<int> accepted_virts;
    // Ca_gs_ = dets[0]->Ca();
}

UOCDFT::UOCDFT(std::shared_ptr<SuperFunctional> functional, Options& options,
               std::shared_ptr<PSIO> psio, SharedWavefunction ref_scf, int state, int symmetry)
    : UHF(ref_scf, functional, options, psio), do_excitation(true), do_symmetry(true),
      ground_state_energy(0.0), ground_state_symmetry_(0), excited_state_symmetry_(symmetry),
      state_(state), singlet_exc_energy_s_plus_(0.0), triplet_exc_energy_s_plus(0.0),
      singlet_exc_energy_ci(0.0), triplet_exc_energy_ci(0.0), oscillator_strength_s_plus_(0.0),
      oscillator_strength_s_plus_x_(0.0), oscillator_strength_s_plus_y_(0.0),
      oscillator_strength_s_plus_z_(0.0), oscillator_strength_ci(0.0) {
    do_excitation = true;
    init();
    init_excitation(ref_scf);
    wfn_ = ref_scf;
    ground_state_energy = dets[0]->energy();
    ground_state_symmetry_ = dets[0]->symmetry();
    // std::vector<int> accepted_virts;
    // Ca_gs_ = dets[0]->Ca();
}

void UOCDFT::init() {
    outfile->Printf("\n  ==> Unrestricted Orthogonality Constrained DFT (OCDFT) <==\n\n");

    //    level_shift_ = options_.get_double("LEVEL_SHIFT");
    //    outfile->Printf("  Level shift: %f\n",level_shift_);

    saved_naholepi_ = Dimension(nirrep_, "Saved number of holes per irrep");
    saved_napartpi_ = Dimension(nirrep_, "Saved number of particles per irrep");
    zero_dim_ = Dimension(nirrep_);

    MOM_enabled_ = (UHF::options_["MOM_START"].has_changed());
    MOM_started_ = false;
    MOM_performed_ = false;

    // Allocate vectors
    TempVector = factory_->create_shared_vector("SVD sigma");
    AcceptedVirtuals =
        factory_->create_shared_vector("Accepted Localized Adsorbate Virtual Orbitals");

    // Allocate matrices
    Ca_gs_ = factory_->create_shared_matrix("Ground State Ca Coefficients");
    H_copy = factory_->create_shared_matrix("H_copy");
    gs_Ca = factory_->create_shared_matrix("Ground State MO Coefficients");
    TempMatrix = factory_->create_shared_matrix("Temp");
    TempMatrix2 = factory_->create_shared_matrix("Temp2");
    TempMatrix3 = factory_->create_shared_matrix("Temp3");
    TempMatrix4 = factory_->create_shared_matrix("Temp4");
    // L_ibo = factory_->create_shared_matrix("IBO Coefficients");
    // A_iao = factory_->create_shared_matrix("IAOs in Primary basis size
    // Matrix");
    Dolda_ = factory_->create_shared_matrix("Dold alpha");
    Doldb_ = factory_->create_shared_matrix("Dold beta");
    NTOs_ = factory_->create_shared_matrix("Natural Transition Orbitals");
    // L_vvo = factory_->create_shared_matrix("Localized Ground State VVOs");
    nvvos = 0;
    save_H_ = true;
}

void UOCDFT::init_excitation(SharedWavefunction ref_scf) {
    // Never recalculate the socc and docc arrays
    input_socc_ = true;
    input_docc_ = true;

    // Default: CHP algorithm
    do_holes = true;
    do_parts = true;
    do_opt_spectators = true;

    std::string exc_method = options_.get_str("CDFT_EXC_METHOD");
    std::string exc_type = options_.get_str("CDFT_EXC_TYPE");
    std::string nroots = options_.get_str("NROOTS");
    std::string roots_per_hole = options_.get_str("CDFT_NUM_PROJECT_OUT");
    std::string exc_project = options_.get_str("CDFT_PROJECT_OUT");
    if (exc_method == "CHP-F") {
        throw InputException("Frozen CHP algorithm has not been implemented", "REFERENCE to UKS",
                             __FILE__, __LINE__);
        // do_opt_spectators = false;
    } else if (exc_method == "CH") {
        do_parts = false;
    } else if (exc_method == "CP") {
        do_holes = false;
    }

    outfile->Printf("\n  ==> OCDFT Algorithm Details <==");
    outfile->Printf("\n  Excited State Type: %s", exc_type.c_str());
    outfile->Printf("\n  Total Number of Roots: %s", nroots.c_str());
    outfile->Printf("\n  Number of Roots Per Hole: %s", roots_per_hole.c_str());
    outfile->Printf("\n  Excited State Algorithm: ", exc_method.c_str());

    // Save the reference state MOs and occupation numbers
    // outfile->Printf("  Saving the reference orbitals for an excited state
    // computation\n");
    UOCDFT* ucks_ptr = dynamic_cast<UOCDFT*>(ref_scf.get());
    // Ca_gs_ =  SharedMatrix(new Matrix("Ca_gs_",nsopi_,nmopi_));
    // Ca_gs_ = dets[0]->Ca();
    gs_Fa_ = ucks_ptr->gs_Fa_;
    gs_Fb_ = ucks_ptr->gs_Fb_;
    Fa_->copy(gs_Fa_);
    Fb_->copy(gs_Fb_);
    naholepi_ = Dimension(nirrep_, "Number of holes per irrep");
    napartpi_ = Dimension(nirrep_, "Number of particles per irrep");
    gs_nalphapi_ = ucks_ptr->dets[0]->nalphapi();
    gs_navirpi_ = nmopi_ - gs_nalphapi_;
    gs_nbetapi_ = ucks_ptr->dets[0]->nbetapi();
    gs_nbvirpi_ = nmopi_ - gs_nbetapi_;

    // Grab the saved number of alpha holes/particles
    project_naholepi_ = Dimension(nirrep_, "project_naholepi_");
    ;
    if (state_ == 1) {
        dets.push_back(ucks_ptr->dets[0]);
        saved_Ch_ = SharedMatrix(new Matrix("Ch_", nsopi_, gs_nalphapi_));
        saved_Cp_ = SharedMatrix(new Matrix("Cp_", nsopi_, gs_navirpi_));
    } else {
        dets = ucks_ptr->dets;
        saved_Ch_ = ucks_ptr->saved_Ch_;
        saved_Cp_ = ucks_ptr->Cp_;
        saved_naholepi_ = ucks_ptr->saved_naholepi_;
        saved_napartpi_ = ucks_ptr->napartpi_;
    }

    PoFaPo_ = SharedMatrix(new Matrix("PoFPo", gs_nalphapi_, gs_nalphapi_));
    PvFaPv_ = SharedMatrix(new Matrix("PvFPo", gs_navirpi_, gs_navirpi_));
    Ua_o_ = SharedMatrix(new Matrix("Ua_o_", gs_nalphapi_, gs_nalphapi_));
    Ua_v_ = SharedMatrix(new Matrix("Ua_v_", gs_navirpi_, gs_navirpi_));
    lambda_a_o_ = SharedVector(new Vector("lambda_a_o_", gs_nalphapi_));
    lambda_a_v_ = SharedVector(new Vector("lambda_a_v_", gs_navirpi_));
    // A_iao = SharedMatrix(new Matrix("IAO Coefficients",nsopi_, nsopi_));
    // L_ibo = SharedMatrix(new Matrix("IBO Coefficients",nsopi_, nsopi_));

    Ch_ = SharedMatrix(new Matrix("Ch_", nsopi_, gs_nalphapi_));
    Cp_ = SharedMatrix(new Matrix("Cp_", nsopi_, gs_navirpi_));

    QFQ_ = factory_->create_shared_matrix("QFQ");
    moFeffa_ = factory_->create_shared_matrix("MO alpha Feff");
    moFeffb_ = factory_->create_shared_matrix("MO beta Feff");
    // SharedWavefunction wfn_ = ref_scf;

    std::string project_out = options_.get_str("CDFT_PROJECT_OUT");
    do_project_out_holes = false;
    do_project_out_particles = false;
    if (project_out == "H") {
        outfile->Printf("\n  Projection Algorithm: H (Hole Projection)");
        do_project_out_holes = true;
        do_save_holes = true;
        // Project out all the holes
        project_naholepi_ = saved_naholepi_;
    } else if (project_out == "P") {
        outfile->Printf("\n  Projection Algorithm: P (Particle Projection)");
        do_project_out_particles = true;
        do_save_particles = true;
    } else if (project_out == "HP") {
        outfile->Printf("\n  Projection Algorithm: HP (Hole/Particle Projection)");
        if (nirrep_ != 1) {
            outfile->Printf("\n  The HP algorithm is only implemented for C1 symmetry.\n");
            exit(1);
        }
        do_project_out_holes = true;
        do_project_out_particles = true;
        do_save_particles = true;
        do_save_holes = true;
        outfile->Printf("\n\n  STATE %d of %s", state_, nroots.c_str());
        // Check if we computed enough hole/particles
        if (state_ > 0) {
            project_naholepi_[0] =
                (state_ - 1) /
                options_.get_int("CDFT_NUM_PROJECT_OUT"); // TODO generalize to symmetry
            if (state_ % options_.get_int("CDFT_NUM_PROJECT_OUT") == 1) {
                // Compute a new particle and save it, project out the holes but don't
                // save it
                // outfile->Printf("\n  Save the first hole in the series.  Compute a
                // new particle and save it.");
                //                saved_naholepi_[0] = (state_-1)/
                //                options_.get_int("CDFT_NUM_PROJECT_OUT");
                // outfile->Printf("\n  Saved holes %d",saved_naholepi_[0]);
                do_project_out_holes = true;
                do_save_particles = true;
                do_save_holes = true;
                for (int h = 0; h < nirrep_; ++h)
                    saved_napartpi_[h] = 0;
                Cp_->zero();
                saved_Cp_->zero();
            } else {
                // Reset the cycle, compute a new hole and zero the previous particles
                do_save_particles = true;
                do_save_holes = false;
                do_project_out_holes = true;
                // outfile->Printf("\n  Compute a new particle and save it, project out
                // the holes but don't save it.  Saved holes %d",saved_naholepi_[0]);
            }
        }
    }
    if (options_.get_bool("VALENCE_TO_CORE") and state_ == 1) {
        do_project_out_holes = true;
        do_save_particles = true;
        do_save_holes = false;
    }
    if (options_.get_bool("VALENCE_TO_CORE") and state_ > 1) {
        do_project_out_holes = true;
        do_save_particles = true;
        do_save_holes = true;
    }
    outfile->Printf("\n  ");
    // saved_naholepi_.print();
    outfile->Printf("\n  ");
    // saved_napartpi_.print();
    outfile->Printf("\n  ");
    // project_naholepi_.print();
}

UOCDFT::~UOCDFT() {}

void UOCDFT::guess() {
    if (do_excitation) {
        iteration_ = 0;
        form_initial_C();
        // find_occupation();
        //        Ca_ = dets[0]->Ca();
        //        Cb_ = dets[0]->Cb();
        form_D();
        compute_initial_E();
    } else {
        UHF::guess();
    }
}

void UOCDFT::save_density_and_energy() {

    Dt_old_->copy(Dt_);
    Dolda_->copy(Da_);
    Doldb_->copy(Db_);
    Eold_ = E_;
}

// void UOCDFT::form_G()
//{
//    timer_on("Form V");
//    form_V();
//    timer_off("Form V");

//    // Push the C matrix on
//    std::vector<SharedMatrix> & C = jk_->C_left();
//    C.clear();
//    C.push_back(Ca_subset("SO", "OCC"));
//    C.push_back(Cb_subset("SO", "OCC"));

//    // Addition to standard call, make sure that C_right is not set
//    std::vector<SharedMatrix> & C_right = jk_->C_right();
//    C_right.clear();

//    // Run the JK object
//    jk_->compute();

//    // Pull the J and K matrices off
//    const std::vector<SharedMatrix> & J = jk_->J();
//    const std::vector<SharedMatrix> & K = jk_->K();
//    const std::vector<SharedMatrix> & wK = jk_->wK();
//    J_->copy(J[0]);
//    J_->add(J[1]);
//    if (functional_->is_x_hybrid()) {
//        Ka_ = K[0];
//        Kb_ = K[1];
//    }
//    if (functional_->is_x_lrc()) {
//        wKa_ = wK[0];
//        wKb_ = wK[1];
//    }
//    Ga_->copy(J_);
//    Gb_->copy(J_);

//    Ga_->add(Va_);
//    Gb_->add(Vb_);

//    double alpha = functional_->x_alpha();
//    double beta = 1.0 - alpha;
//    if (alpha != 0.0) {
//        Ka_->scale(alpha);
//        Kb_->scale(alpha);
//        Ga_->subtract(Ka_);
//        Gb_->subtract(Kb_);
//        Ka_->scale(1.0/alpha);
//        Kb_->scale(1.0/alpha);
//    } else {
//        Ka_->zero();
//        Kb_->zero();
//    }

//    std::string functional_prefix = functional_->name().substr(0,2);
//    if (functional_prefix == "sr"){
//        wKa_->scale(-alpha);
//        wKb_->scale(-alpha);
//        Ga_->subtract(wKa_);
//        Gb_->subtract(wKb_);
//        wKa_->scale(-1.0/alpha);
//        wKb_->scale(-1.0/alpha);
//    } else{
//        if (functional_->is_x_lrc()) {
//            wKa_->scale(beta);
//            wKb_->scale(beta);
//            Ga_->subtract(wKa_);
//            Gb_->subtract(wKb_);
//            wKa_->scale(1.0/beta);
//            wKb_->scale(1.0/beta);
//        } else {
//            wKa_->zero();
//            wKb_->zero();
//        }
//    }

//    if (debug_ > 2) {
//        J_->print();
//        Ka_->print();
//        Kb_->print();
//        wKa_->print();
//        wKb_->print();
//        Va_->print();
//        Vb_->print();
//    }
//}

void UOCDFT::form_F() {
    // On the first iteration save H_
    if (save_H_) {
        H_copy->copy(H_);
        save_H_ = false;
    }

    // Augement the one-electron potential (H_) with the CDFT terms
    H_->copy(H_copy);
    Fa_->copy(H_);
    Fa_->add(Ga_);

    H_->copy(H_copy);
    Fb_->copy(H_);
    Fb_->add(Gb_);

    // Form the effective Fock matrix
    if (do_excitation) {
        // Form the projector onto the orbitals orthogonal to the holes and
        // particles in the excited state mo representation
        TempMatrix->zero();
        if (do_holes) {
            TempMatrix->gemm(false, true, 1.0, Ch_, Ch_, 1.0);
        }
        if (do_parts) {
            TempMatrix->gemm(false, true, 1.0, Cp_, Cp_, 1.0);
        }
        TempMatrix->transform(S_);
        TempMatrix->transform(Ca_);
        TempMatrix2->identity();
        TempMatrix2->subtract(TempMatrix);
        // Form the Fock matrix in the excited state basis, project out the h/p
        QFQ_->transform(Fa_, Ca_);
        QFQ_->transform(TempMatrix2);
        moFeffa_->copy(QFQ_);
        // Form the Fock matrix in the excited state basis, project out the h/p
        TempMatrix->transform(Fb_, Cb_);
        moFeffb_->copy(TempMatrix);
        // If user has requested varying occupation numbers, start it on 5th
        // iteration
        if (options_["PFON_TEMP"].has_changed() and iteration_ > 5) {
            pFON();
        }
        ////        QFQ_->print();
        //        // Form the projector onto the ground state occuppied space in the
        //        excited state mo representation
        //        TempMatrix->zero();
        //        TempMatrix->gemm(false,true,1.0,Ch_,Ch_,0.0);
        //        TempMatrix->gemm(false,true,1.0,Cp_,Cp_,1.0);
        //        TempMatrix->transform(S_);
        //        TempMatrix->transform(Ca_);
        //        TempMatrix2->identity();
        //        TempMatrix2->subtract(TempMatrix);
    }
    if (debug_) {
        Fa_->print();
        Fb_->print();
    }
}

void UOCDFT::form_C() {
    if (not do_excitation) {
        // Ground state: use the default
        UHF::form_C();
        // if(state_==0){
        //    gs_Ca->copy(Ca_);
        //    gs_Ca->print();
        //}
        // gs_Ca->print();
        SharedMatrix L_vvo;
        SharedMatrix L_occ;
        // std::shared_ptr<Localizer> loc_vvo = Localizer::build("PIPEK_MEZEY",
        // wfn_->basisset(), VVO_Space_Matrix);
        // std::shared_ptr<Localizer> loc_occ = Localizer::build("PIPEK_MEZEY",
        // wfn_->basisset(), Occ_Space_Matrix);
        // loc_vvo->localize();
        // loc_occ->localize();
        // L_vvo = loc_vvo->L();
        // L_occ = loc_occ->L();
        if (state_ == 0 and iteration_ == 1 and options_["CDFT_BREAK_SYMMETRY"].has_changed()) {
            // Mix the alpha and beta homo
            int np = options_["CDFT_BREAK_SYMMETRY"][0].to_integer();
            int nq = options_["CDFT_BREAK_SYMMETRY"][1].to_integer();
            double angle = options_["CDFT_BREAK_SYMMETRY"][2].to_double();
            outfile->Printf("\n  Mixing the alpha orbitals %d and %d by %f.1 degrees\n\n", np, nq,
                            angle);
            // SharedMatrix L_a;
            // std::shared_ptr<Localizer> loc_a = Localizer::build("PIPEK_MEZEY",
            // wfn_->basisset(), Ca_);
            // loc_a->localize();
            // L_a = loc_a->L();
            // Ca_->set_column(0,np-1,L_a->get_column(0,np-1));
            // saved_Ch_->set_column(h,saved_naholepi_[h] + i,Ch_->get_column(h,i));
            Ca_->rotate_columns(0, np - 1, nq - 1, (pc_pi * angle) / 180.0);
            // Cb_->rotate_columns(0,np-1,nq-1,pc_pi * angle / 180.0);
            // Reset the DIIS subspace
            diis_manager_->reset_subspace();
        }
    } else {
        // Excited state: use a special form_C
        form_C_ee();
    }

    // Check the orthogonality of the MOs
    orthogonality_check(Ca_, S_);
}

void UOCDFT::form_C_ee() {
    // Compute the hole and the particle states
    compute_holes();
    compute_particles();

    // Find the hole/particle pair to follow
    find_ee_occupation(lambda_a_o_, lambda_a_v_);

    // Build the Ch and Cp matrices
    compute_hole_particle_mos();

    // Form and diagonalize the Fock matrix for the spectator orbitals
    if (do_opt_spectators) {
        diagonalize_F_spectator_relaxed();
    } else {
        diagonalize_F_spectator_unrelaxed();
    }

    // Update the occupation and sort the MOs
    sort_ee_mos();

    // Beta always fully relaxed
    form_C_beta();
}

void UOCDFT::compute_holes() {
    if (iteration_ > 1) {
        // Form the projector Ca Ca^T S Ca_gs
        TempMatrix->zero();
        // Copy the occupied block of Ca
        copy_block(Ca_, 1.0, TempMatrix, 0.0, nsopi_, nalphapi_);
        // Copy Ch
        copy_block(Ch_, 1.0, TempMatrix, 0.0, nsopi_, naholepi_, zero_dim_, zero_dim_, zero_dim_,
                   nalphapi_);

        TempMatrix2->gemm(false, true, 1.0, TempMatrix, TempMatrix, 0.0);
        TempMatrix->gemm(false, false, 1.0, TempMatrix2, S_, 0.0);
        // If performing a valence-to-core transition, transform the projectors and
        // Fa in the MO basis
        // of the intermediate state (dets[1]) otherwise transform them in the basis
        // of the ground state
        if (options_.get_bool("VALENCE_TO_CORE") and state_ > 1) {
            TempMatrix2->gemm(false, false, 1.0, TempMatrix, dets[1]->Ca(), 0.0);
        } else {
            TempMatrix2->gemm(false, false, 1.0, TempMatrix, dets[0]->Ca(), 0.0);
        }

        TempMatrix->transform(Fa_, TempMatrix2);
    } else {
        // Transform Fa to the MO basis of the ground state
        if (options_.get_bool("VALENCE_TO_CORE") and state_ > 1) {
            TempMatrix->transform(Fa_, dets[1]->Ca());
        } else {
            TempMatrix->transform(Fa_, dets[0]->Ca());
        }
    }

    // Grab the occ block of Fa
    copy_block(TempMatrix, 1.0, PoFaPo_, 0.0, gs_nalphapi_, gs_nalphapi_);

    // Form the projector 1 - Ph = 1 - (Ch^T S Ca_gs)^T Ch^T S Ca_gs
    TempMatrix->zero();
    SharedMatrix project_Ch(new Matrix("project_Ch_", nsopi_, project_naholepi_));
    // Copy only the orbitals that need to be projected out
    copy_subblock(saved_Ch_, project_Ch, nsopi_, project_naholepi_, true);

    TempMatrix->gemm(false, true, 1.0, project_Ch, project_Ch, 0.0);
    TempMatrix->transform(S_);
    if (options_.get_bool("VALENCE_TO_CORE") and state_ > 1) {
        TempMatrix->transform(dets[1]->Ca());
    } else {
        TempMatrix->transform(dets[0]->Ca());
    }
    SharedMatrix Ph = SharedMatrix(new Matrix("Ph", gs_nalphapi_, gs_nalphapi_));
    Ph->identity();
    copy_block(TempMatrix, -1.0, Ph, 1.0, gs_nalphapi_, gs_nalphapi_);

    if (do_project_out_holes) {
        // Project out the previous holes
        PoFaPo_->transform(Ph);
        // outfile->Printf("  Projecting out %d previous
        // holes\n",project_naholepi_.sum());
    }

    // Diagonalize the occ block
    PoFaPo_->diagonalize(Ua_o_, lambda_a_o_);
// Ua_o_->print();
#if DEBUG_OCDFT
    lambda_a_o_->print();
#endif
}

void UOCDFT::compute_particles() {
    if (iteration_ > 1) {
        // Form the projector Ca Ca^T S Ca_gs
        TempMatrix->zero();
        // Copy Cp
        copy_block(Cp_, 1.0, TempMatrix, 0.0, nsopi_, napartpi_);
        // Copy the virtual block of Ca
        copy_block(Ca_, 1.0, TempMatrix, 0.0, nsopi_, nmopi_ - nalphapi_, zero_dim_, nalphapi_,
                   zero_dim_, napartpi_);

        TempMatrix2->gemm(false, true, 1.0, TempMatrix, TempMatrix, 0.0);
        TempMatrix->gemm(false, false, 1.0, TempMatrix2, S_, 0.0);
        // If performing a valence-to-core transition, transform the projectors and
        // Fa in the MO basis
        // of the intermediate state (dets[1]) otherwise transform them in the basis
        // of the ground state
        if (options_.get_bool("VALENCE_TO_CORE") and state_ > 1) {
            TempMatrix2->gemm(false, false, 1.0, TempMatrix, dets[1]->Ca(), 0.0);
        } else {
            TempMatrix2->gemm(false, false, 1.0, TempMatrix, dets[0]->Ca(), 0.0);
        }

        TempMatrix->transform(Fa_, TempMatrix2);
    } else {
        // Transform Fa to the MO basis of the ground state
        if (options_.get_bool("VALENCE_TO_CORE") and state_ > 1) {
            TempMatrix->transform(Fa_, dets[1]->Ca());
        } else {
            TempMatrix->transform(Fa_, dets[0]->Ca());
        }
    }

    // Grab the vir block of Fa
    copy_block(TempMatrix, 1.0, PvFaPv_, 0.0, gs_navirpi_, gs_navirpi_, gs_nalphapi_, gs_nalphapi_);

    // Form the projector Pp = 1 - (Cp^T S Ca_gs)^T Cp^T S Ca_gs
    TempMatrix->zero();
    SharedMatrix project_Cp(new Matrix("project_Cp_", nsopi_, gs_navirpi_));
    // Copy only the orbitals that need to be projected out
    copy_subblock(saved_Cp_, project_Cp, nsopi_, saved_napartpi_, true);
    TempMatrix->gemm(false, true, 1.0, project_Cp, project_Cp, 0.0);
    TempMatrix->transform(S_);
    if (options_.get_bool("VALENCE_TO_CORE") and state_ > 1) {
        TempMatrix->transform(dets[1]->Ca());
    } else {
        TempMatrix->transform(dets[0]->Ca());
    }
    SharedMatrix Pp = SharedMatrix(new Matrix("Pp", gs_navirpi_, gs_navirpi_));
    Pp->identity();
    copy_block(TempMatrix, -1.0, Pp, 1.0, gs_navirpi_, gs_navirpi_, gs_nalphapi_, gs_nalphapi_);

    if (do_project_out_particles) {
        // Project out the previous particles
        PvFaPv_->transform(Pp);
        // outfile->Printf("  Projecting out %d previous
        // particles\n",saved_napartpi_.sum());
    }

    // Diagonalize the vir block
    PvFaPv_->diagonalize(Ua_v_, lambda_a_v_);

#if DEBUG_OCDFT
    lambda_a_v_->print();
#endif
}

void UOCDFT::find_ee_occupation(SharedVector lambda_o, SharedVector lambda_v) {
    // Find the hole/particle pair to follow
    std::tuple<double, int, int> hole;
    std::tuple<double, int, int> particle;
    int count = 1;
    std::vector<std::tuple<double, int, int, double, int, int, double>> sorted_hp_pairs;
    std::vector<int> accepted_virts;
    std::vector<std::tuple<double, int>> accepted_holes;
    if (options_["P_SUBSPACE"].size() > 0 and options_.get_bool("VALENCE_TO_CORE") and
        state_ != 1) {
        accepted_virts = particle_subspace(wfn_, dets[0]->Ca());
    }
    if (options_["H_SUBSPACE"].size() > 0) {
        accepted_holes = hole_subspace(wfn_, dets[0]->Ca());
    }
    // If we are doing core excitation just take the negative of the hole energy
    bool do_core_excitation = false;
    if (options_.get_str("CDFT_EXC_TYPE") == "CORE") {
        do_core_excitation = true;
    }
    if (options_.get_str("CDFT_EXC_TYPE") == "CORE" and options_["H_SUBSPACE"].size() > 0) {
        std::sort(accepted_holes.rbegin(), accepted_holes.rend());
    }
    // If performing a valence-to-core transition, it is now desirable to obtain
    // the "lowest" energy solutions,
    // since the excitation energies will be negative.
    if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1 and
        options_.get_str("CDFT_EXC_TYPE") == "CORE") {
        do_core_excitation = false;
    }
    // Compute the symmetry adapted hole/particle pairs
    for (int occ_h = 0; occ_h < nirrep_; ++occ_h) {
        int nocc = gs_nalphapi_[occ_h];
        for (int i = 0; i < nocc; ++i) {
            double e_h = lambda_o->get(occ_h, i);
            for (int vir_h = 0; vir_h < nirrep_; ++vir_h) {
                int nvir = gs_navirpi_[vir_h];
                for (int a = 0; a < nvir; ++a) {
                    double e_p = lambda_v->get(vir_h, a);
                    double e_hp = do_core_excitation ? (e_p + e_h) : (e_p - e_h);
                    int symm = occ_h ^ vir_h ^ ground_state_symmetry_;
                    bool use_vir = true;
                    bool use_occ = true;
                    if (options_["P_SUBSPACE"].size() > 0 and
                        options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
                        // bool use_vir = true;
                        int vir_size = accepted_virts.size();
                        int vir_num = 0;
                        do {
                            use_vir = false;
                            if (a == accepted_virts[vir_num]) {
                                use_vir = true;
                                break;
                            }
                            vir_num++;
                        } while (vir_num <= vir_size);
                    }
                    if (options_["H_SUBSPACE"].size() > 0) {
                        int occ_num = 0;
                        int occ_size = accepted_holes.size();
                        do {
                            use_occ = false;
                            if (i == std::get<1>(accepted_holes[occ_num])) {
                                use_occ = true;
                                break;
                            }
                            occ_num++;
                        } while (occ_num <= occ_size);
                    }
                    if (not do_symmetry or (symm == excited_state_symmetry_)) { // Test for symmetry
                        // Make sure we are not adding excitations to holes/particles that
                        // have been projected out
                        if (options_.get_double("REW") >
                            0.0) { // Perform Restricted Excitation Window Calculation
                            double rew_cutoff = options_.get_double("REW");
                            bool rew_criteria = false;
                            if (std::fabs(e_h) < rew_cutoff) {
                                rew_criteria = true;
                            }
                            if (std::fabs(e_h) > 1.0e-6 and rew_criteria) {
                                sorted_hp_pairs.push_back(
                                    std::make_tuple(e_hp, occ_h, i, e_h, vir_h, a, e_p));
                            }
                        } else if (do_core_excitation and options_["H_SUBSPACE"].size() > 0) {
                            if (std::fabs(e_h) > 1.0e-6 and std::fabs(e_p) > 1.0e-6 and use_vir and
                                i == std::get<1>(accepted_holes[0])) {
                                // if(std::fabs(e_h) > 1.0e-6 and std::fabs(e_p) > 1.0e-6 and
                                // use_vir and use_occ and e_h!=e_p){
                                sorted_hp_pairs.push_back(
                                    std::make_tuple(e_hp, occ_h, i, e_h, vir_h, a, e_p));
                            }
                        } else {
                            if (std::fabs(e_h) > 1.0e-6 and std::fabs(e_p) > 1.0e-6 and use_vir and
                                use_occ) { // Use Full Excitation Space
                                sorted_hp_pairs.push_back(
                                    std::make_tuple(e_hp, occ_h, i, e_h, vir_h, a,
                                                    e_p)); // N.B. shifted wrt to full indexing
                                // sorted_hp_pairs.push_back(std::make_tuple(e_hp,vir_h,a,e_p,occ_h,i,e_h));
                            }
                        }
                    }
                }
            }
        }
    }
    count = count + 1;
    // If we are using MOM, let MOM sort the orbitals via overlap criteria
    if (MOM_started_) {
        HF::MOM();
    }
    // Otherwise, just sort the hole/particle pairs according to the energy
    else {
        std::sort(sorted_hp_pairs.begin(), sorted_hp_pairs.end());
    }
    CharacterTable ct = molecule_->point_group()->char_table();
    if (iteration_ == 0) {
        // outfile->Printf( "\n  Ground state symmetry:
        // %s\n",ct.gamma(ground_state_symmetry_).symbol());
        // outfile->Printf( "  Excited state symmetry:
        // %s\n",ct.gamma(excited_state_symmetry_).symbol());
        outfile->Printf("\n\n  ==> Chosen Hole/Particle pairs <==\n");
        outfile->Printf("  --------------------------------------\n");
        outfile->Printf("    N   Occupied     Virtual     E(eV)  \n");
        outfile->Printf("  --------------------------------------\n");
        int maxstates = std::min(10, static_cast<int>(sorted_hp_pairs.size()));
        for (int n = 0; n < maxstates; ++n) {
            double energy_hp = std::get<6>(sorted_hp_pairs[n]) - std::get<3>(sorted_hp_pairs[n]);
            outfile->Printf(
                "   %2d   %4d%-3s  -> %4d%-3s   %9.3f\n", n + 1,
                std::get<2>(sorted_hp_pairs[n]) + 1,
                ct.gamma(std::get<1>(sorted_hp_pairs[n])).symbol(),
                gs_nalphapi_[std::get<4>(sorted_hp_pairs[n])] + std::get<5>(sorted_hp_pairs[n]) + 1,
                ct.gamma(std::get<4>(sorted_hp_pairs[n])).symbol(), energy_hp * pc_hartree2ev);
        }
        outfile->Printf("  --------------------------------------\n");

        int select_pair = 0;
        // Select the excitation pair using the energetic ordering
        if (options_["CDFT_EXC_SELECT"].has_changed()) {
            int input_select = options_["CDFT_EXC_SELECT"][excited_state_symmetry_].to_integer();
            if (input_select > 0) {
                select_pair = input_select - 1;
                outfile->Printf("\n  Following excitation #%d: ", input_select);
            }
        }
        // Select the excitation pair using the symmetry of the hole
        if (options_["CDFT_EXC_HOLE_SYMMETRY"].has_changed()) {
            int input_select =
                options_["CDFT_EXC_HOLE_SYMMETRY"][excited_state_symmetry_].to_integer();
            if (input_select > 0) {
                int maxstates = static_cast<int>(sorted_hp_pairs.size());
                for (int n = 0; n < maxstates; ++n) {
                    if (std::get<1>(sorted_hp_pairs[n]) == input_select - 1) {
                        select_pair = n;
                        break;
                    }
                }
                outfile->Printf("\n  Following excitation #%d:\n", select_pair + 1);
            }
        }
        aholes.clear();
        aparts.clear();
        // accepted_virts.clear();

        int ahole_h = std::get<1>(sorted_hp_pairs[select_pair]);
        int ahole_mo = std::get<2>(sorted_hp_pairs[select_pair]);
        double ahole_energy = std::get<3>(sorted_hp_pairs[select_pair]);
        std::tuple<int, int, double> ahole = std::make_tuple(ahole_h, ahole_mo, ahole_energy);
        aholes.push_back(ahole);

        int apart_h = std::get<4>(sorted_hp_pairs[select_pair]);
        int apart_mo = std::get<5>(sorted_hp_pairs[select_pair]);
        double apart_energy = std::get<6>(sorted_hp_pairs[select_pair]);
        std::tuple<int, int, double> apart = std::make_tuple(apart_h, apart_mo, apart_energy);
        aparts.push_back(apart);
    } else {
        if (not(options_["CDFT_EXC_SELECT"].has_changed() or
                options_["CDFT_EXC_HOLE_SYMMETRY"].has_changed())) {
            aholes.clear();
            aparts.clear();

            int ahole_h = std::get<1>(sorted_hp_pairs[0]);
            int ahole_mo = std::get<2>(sorted_hp_pairs[0]);
            double ahole_energy = std::get<3>(sorted_hp_pairs[0]);
            std::tuple<int, int, double> ahole = std::make_tuple(ahole_h, ahole_mo, ahole_energy);
            aholes.push_back(ahole);

            int apart_h = std::get<4>(sorted_hp_pairs[0]);
            int apart_mo = std::get<5>(sorted_hp_pairs[0]);
            double apart_energy = std::get<6>(sorted_hp_pairs[0]);
            std::tuple<int, int, double> apart = std::make_tuple(apart_h, apart_mo, apart_energy);
            aparts.push_back(apart);
        }
    }

    for (int h = 0; h < nirrep_; ++h) {
        naholepi_[h] = 0;
        napartpi_[h] = 0;
    }

    // Compute the number of hole and/or particle orbitals to compute
    outfile->Printf("HOLE:   ");
    size_t naholes = aholes.size();
    for (size_t n = 0; n < naholes; ++n) {
        naholepi_[std::get<0>(aholes[n])] += 1;
        outfile->Printf("%4d%-3s (%+.6f)", std::get<1>(aholes[n]) + 1,
                        ct.gamma(std::get<0>(aholes[n])).symbol(), std::get<2>(aholes[n]));
    }
    outfile->Printf("      PARTICLE:");
    size_t naparts = aparts.size();
    for (size_t n = 0; n < naparts; ++n) {
        napartpi_[std::get<0>(aparts[n])] += 1;
        outfile->Printf("%4d%-3s (%+.6f)",
                        gs_nalphapi_[std::get<0>(aparts[n])] + std::get<1>(aparts[n]) + 1,
                        ct.gamma(std::get<0>(aparts[n])).symbol(), std::get<2>(aparts[n]));
    }
    outfile->Printf("\n");
    // Check if we are using MOM
    if (options_["MOM_START"].has_changed()) {
        HF::MOM_start();
    }
}

void UOCDFT::compute_hole_particle_mos() {
    SharedMatrix Ca0;
    // If performing valence-to-core transition, the reference C matrix is now
    // that of the
    // intermediate state (dets[1]), otherwise the reference is the ground state.
    if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
        Ca0 = dets[1]->Ca();
    } else {
        Ca0 = dets[0]->Ca();
    }

    Ch_->zero();
    Cp_->zero();
    // Compute the hole orbitals
    size_t naholes = aholes.size();
    std::vector<int> hoffset(nirrep_, 0);
    for (size_t n = 0; n < naholes; ++n) {
        int ahole_h = std::get<0>(aholes[n]);
        int ahole_mo = std::get<1>(aholes[n]);
        int nhole = hoffset[ahole_h];
        int maxi = gs_nalphapi_[ahole_h];
        for (int p = 0; p < nsopi_[ahole_h]; ++p) {
            double c_h = 0.0;
            for (int i = 0; i < maxi; ++i) {
                c_h += Ca0->get(ahole_h, p, i) * Ua_o_->get(ahole_h, i, ahole_mo);
            }
            Ch_->set(ahole_h, p, nhole, c_h);
        }
        hoffset[ahole_h] += 1;
    }

    // Compute the particle orbital
    size_t naparts = aparts.size();
    std::vector<int> poffset(nirrep_, 0);
    for (size_t n = 0; n < naparts; ++n) {
        int apart_h = std::get<0>(aparts[n]);
        int apart_mo = std::get<1>(aparts[n]);
        int npart = poffset[apart_h];
        int maxa = gs_navirpi_[apart_h];
        for (int p = 0; p < nsopi_[apart_h]; ++p) {
            double c_p = 0.0;
            for (int a = 0; a < maxa; ++a) {
                c_p += Ca0->get(apart_h, p, gs_nalphapi_[apart_h] + a) *
                       Ua_v_->get(apart_h, a, apart_mo);
            }
            Cp_->set(apart_h, p, npart, c_p);
        }
        poffset[apart_h] += 1;
    }
}

std::vector<std::tuple<double, int>> UOCDFT::hole_subspace(SharedWavefunction wfn,
                                                           SharedMatrix Ca) {
    // Allocate Necessary Vectors and Matrices.
    std::vector<std::tuple<double, int>> accepted_holes;
    std::vector<double> m_sub_diag;
    std::vector<std::string> subspace_str;
    std::vector<int> selected_holes;
    SharedMatrix S_ao =
        SharedMatrix(new Matrix("AO Overlap matrix", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix S_inv_half_ao = SharedMatrix(new Matrix(
        "Inverse square root of the AO Overlap matrix", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix X_sub = SharedMatrix(
        new Matrix("Subset of Inverse Square Root Matrix", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix M_sub = SharedMatrix(
        new Matrix("Projected Subset Overlap Matrix", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix Sub_cont = SharedMatrix(
        new Matrix("Transformaed Subset Overlap Matrix", basisset_->nbf(), basisset_->nbf()));
    std::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
    // Transform overlap Matrix to AO basis
    SharedMatrix SO2AO_ = pet->sotoao();
    S_ao->remove_symmetry(S_, SO2AO_);
    S_inv_half_ao->copy(S_ao);
    // Build S^-1/2 Matrix
    S_inv_half_ao->power(-0.5);
    // Only grab hole subspace information if subspace atoms have been requested
    if (options_["H_SUBSPACE"].size() > 0) {
        for (int entry = 0; entry < (int)options_["H_SUBSPACE"].size(); ++entry) {
            std::string s = options_["H_SUBSPACE"][entry].to_string();
            subspace_str.push_back(s);
        }
        // Create an AOSubspace object
        AOSubspace aosub(subspace_str, wfn->molecule(), wfn->basisset());
        if (iteration_ == 0) {
            outfile->Printf("\n\n   => Hole Orbital Subspace <=\n");
        }
        // Compute the subspaces
        aosub.find_subspace();

        // Get the subspaces
        selected_holes = aosub.subspace();
    }
    int nocc = gs_nalphapi_[0];
    // Only print subspace information on initial iteration
    if (iteration_ == 0) {
        outfile->Printf("\n    Overlap of Occupied Orbitals with Atomic Orbital "
                        "Subspace              ");
        outfile->Printf("\n    ----------------------------------------------------------");
        outfile->Printf("\n\n    %d Occupied MOs: \n \n", nocc);
    }

    int num_funcs = selected_holes.size();
    int print_count = 1;
    // Build X matrix for orthonormal transformation of the basis, essentially the
    // S^-1/2 matrix for the given subset
    for (int mu = 0; mu < basisset_->nbf(); ++mu) {
        double x_sub_element = 0.0;
        for (int nu = 0; nu < num_funcs; ++nu) {
            x_sub_element = S_inv_half_ao->get(mu, selected_holes[nu]);
            X_sub->set(mu, selected_holes[nu], x_sub_element);
        }
    }
    TempMatrix->zero();
    TempMatrix2->zero();
    // Compute projection C^T S X X^T S C
    TempMatrix->gemm(false, true, 1.0, X_sub, X_sub, 0.0);
    TempMatrix2->gemm(false, false, 1.0, S_ao, TempMatrix, 0.0);
    M_sub->gemm(false, false, 1.0, TempMatrix2, S_ao, 0.0);
    M_sub->transform(dets[0]->Ca());
    // Grab the diagonal elements of the subset projection matrix
    for (int i = 0; i < basisset_->nbf(); i++) {
        for (int j = 0; j < basisset_->nbf(); j++) {
            if (i == j) {
                m_sub_diag.push_back(M_sub->get(i, j));
            }
        }
    }
    // Print each molecular orbital's overlap with the subset and store the values
    // above the specified threshold.

    double hole_threshold;
    if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
        hole_threshold = 0.0;
    } else {
        hole_threshold = options_.get_double("HOLE_THRESHOLD");
    }
    for (int i = 0; i < nocc; i++) {
        if (iteration_ == 0) {
            if (print_count == 4) {
                auto temp_string =
                    fmt::format("     {:2d}A       {:-5f}", (i + 1) , m_sub_diag[i]);
                outfile->Printf("%s", temp_string.c_str());
                outfile->Printf("\n");
                print_count = 0;
            } else {
                auto temp_string =
                    fmt::format("     {:2d}A       {:-5f}", (i + 1) , m_sub_diag[i]);
                // outfile->Printf("  %d-A      %f  ",mu+1,kappa);
                outfile->Printf("%s", temp_string.c_str());
            }
            print_count++;
        }
        if (m_sub_diag[i] > hole_threshold) {
            accepted_holes.push_back(std::make_tuple(m_sub_diag[i], i));
        }
    }
    //}
    return accepted_holes;
}

std::vector<int> UOCDFT::particle_subspace(SharedWavefunction wfn, SharedMatrix Ca) {
    // Find the AO subset
    if (iteration_ == 0) {
        outfile->Printf("\n  ==> Hole/Particle Atomic Orbital Subspace "
                        "Localization Routine <== \n");
    }
    std::vector<int> subspace;
    std::vector<double> m_sub_diag;
    std::vector<int> selected_particles;
    SharedMatrix S_ao =
        SharedMatrix(new Matrix("AO Overlap matrix", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix S_inv_half_ao = SharedMatrix(new Matrix(
        "Inverse square root of the AO Overlap matrix", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix X_sub = SharedMatrix(
        new Matrix("Subset of Inverse Square Root Matrix", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix M_sub = SharedMatrix(
        new Matrix("Projected Subset Overlap Matrix", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix Sub_cont = SharedMatrix(
        new Matrix("Transformaed Subset Overlap Matrix", basisset_->nbf(), basisset_->nbf()));
    std::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
    SharedMatrix SO2AO_ = pet->sotoao();
    S_ao->remove_symmetry(S_, SO2AO_);
    S_inv_half_ao->copy(S_ao);
    S_inv_half_ao->power(-0.5);

    std::vector<std::string> subspace_str;
    if (options_["P_SUBSPACE"].size() > 0) {
        for (int entry = 0; entry < (int)options_["P_SUBSPACE"].size(); ++entry) {
            std::string s = options_["P_SUBSPACE"][entry].to_string();
            subspace_str.push_back(s);
        }
    }

    // Create an AOSubspace object
    AOSubspace aosub(subspace_str, wfn->molecule(), wfn->basisset());
    if (iteration_ == 0) {
        outfile->Printf("\n   => Particle Orbital Subspace <= \n");
    }
    // Compute the subspaces
    aosub.find_subspace();

    // Get the subspaces
    selected_particles = aosub.subspace();
    std::vector<int> accepted_virts;

    int nocc = gs_nalphapi_[0];
    if (iteration_ == 0) {
        outfile->Printf("\n    Overlap of Virtual Orbitals with Atomic Orbital "
                        "Subspace              ");
        outfile->Printf("\n    --------------------------------------------------------");
        // outfile->Printf("\n\n    %d Particle MOs: \n \n", basisset_->nbf() -
        // nocc);
        outfile->Printf("\n\n    %d Particle MOs: \n \n", nocc);
    }

    int num_funcs = selected_particles.size();
    int print_count = 1;

    for (int mu = 0; mu < basisset_->nbf(); ++mu) {
        double x_sub_element = 0.0;
        for (int nu = 0; nu < num_funcs; ++nu) {
            x_sub_element = S_inv_half_ao->get(mu, selected_particles[nu]);
            X_sub->set(mu, selected_particles[nu], x_sub_element);
        }
    }

    TempMatrix->zero();
    TempMatrix2->zero();
    TempMatrix->gemm(false, true, 1.0, X_sub, X_sub, 0.0);
    TempMatrix2->gemm(false, false, 1.0, S_ao, TempMatrix, 0.0);
    M_sub->gemm(false, false, 1.0, TempMatrix2, S_ao, 0.0);
    M_sub->transform(dets[0]->Ca());

    for (int i = 0; i < basisset_->nbf(); i++) {
        for (int j = 0; j < basisset_->nbf(); j++) {
            if (i == j) {
                m_sub_diag.push_back(M_sub->get(i, j));
            }
        }
    }
    // for(int i = nocc; i < basisset_->nbf(); i++){
    for (int i = 0; i < nocc; i++) {
        if (iteration_ == 0) {
            if (print_count == 4) {
                auto temp_string =
                    fmt::format("     {:2d}A       {:-5f}",i + 1,m_sub_diag[i]);
                outfile->Printf("%s", temp_string.c_str());
                outfile->Printf("\n");
                print_count = 0;
            } else {
                auto temp_string =
                    fmt::format("     {:2d}A       {:-5f}",i + 1,m_sub_diag[i]);
                outfile->Printf("%s", temp_string.c_str());
            }
            print_count++;
        }
        if (m_sub_diag[i] > options_.get_double("PARTICLE_THRESHOLD")) {
            accepted_virts.push_back(i);
        }
    }

    return accepted_virts;
}

void UOCDFT::diagonalize_F_spectator_relaxed() {
    // Form the projector onto the orbitals orthogonal to the holes and particles
    // in the excited state mo representation
    TempMatrix->zero();
    // Project the hole, the particles, or both depending on the method
    if (do_holes) {
        TempMatrix->gemm(false, true, 1.0, Ch_, Ch_, 1.0);
        if (do_project_out_holes) {
            SharedMatrix project_Ch(new Matrix("project_Ch_", nsopi_, gs_nalphapi_));
            // Copy only the orbitals that need to be projected out
            //            copy_subblock(saved_Ch_,project_Ch,nsopi_,saved_naholepi_,true);
            copy_subblock(saved_Ch_, project_Ch, nsopi_, project_naholepi_, true);
            //            Dimension one_offset_(nirrep_,"Offset");

            ////            The problem is here.
            //            one_offset_[0] = 1;
            //            Dimension saved_naholepi_min_one_(nirrep_,"Offset");
            //            saved_naholepi_min_one_[0] = saved_naholepi_[0] - 1;
            //            copy_block(saved_Ch_,1.0,project_Ch,0.0,nsopi_,saved_naholepi_,
            //                                  zero_dim_,one_offset_,
            //                                  zero_dim_,zero_dim_);
            //            saved_Ch_->print();
            //            project_Ch->print();
            //            TempMatrix->gemm(false,true,1.0,saved_Ch_,saved_Ch_,1.0);
            TempMatrix->gemm(false, true, 1.0, project_Ch, project_Ch, 1.0);
        }
    }
    if (do_parts) {
        TempMatrix->gemm(false, true, 1.0, Cp_, Cp_, 1.0);
        if (do_project_out_particles) {
            // outfile->Printf("\n  Projecting out particles:");
            TempMatrix->gemm(false, true, 1.0, saved_Cp_, saved_Cp_, 1.0);
        }
    }

    TempMatrix->transform(S_);
    if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
        TempMatrix->transform(dets[1]->Ca());
    } else {
        TempMatrix->transform(dets[0]->Ca());
    }
    TempMatrix2->identity();
    TempMatrix2->subtract(TempMatrix);

    // Form the Fock matrix in the excited state basis, project out the h/p
    if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
        TempMatrix->transform(Fa_, dets[1]->Ca());
    } else {
        TempMatrix->transform(Fa_, dets[0]->Ca());
    }
    TempMatrix->transform(TempMatrix2);

    // Diagonalize the Fock matrix and transform the MO coefficients
    TempMatrix->diagonalize(TempMatrix2, epsilon_a_);
    TempMatrix->zero();
    if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
        TempMatrix->gemm(false, false, 1.0, dets[1]->Ca(), TempMatrix2, 0.0);
    } else {
        TempMatrix->gemm(false, false, 1.0, dets[0]->Ca(), TempMatrix2, 0.0);
    }

    //    outfile->Printf("\n  Epsilons for spectators");
    //    epsilon_a_->print();
    Ca_->copy(TempMatrix);
}

void UOCDFT::sort_ee_mos() {
    // Set the occupation
    nalphapi_ = gs_nalphapi_ + napartpi_ - naholepi_;
    nbetapi_ = gs_nbetapi_;

    int old_socc[8];
    int old_docc[8];
    for (int h = 0; h < nirrep_; ++h) {
        old_socc[h] = soccpi_[h];
        old_docc[h] = doccpi_[h];
    }

    for (int h = 0; h < nirrep_; ++h) {
        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
        doccpi_[h] = std::min(nalphapi_[h], nbetapi_[h]);
    }

    bool occ_changed = false;
    for (int h = 0; h < nirrep_; ++h) {
        if (old_socc[h] != soccpi_[h] || old_docc[h] != doccpi_[h]) {
            occ_changed = true;
            break;
        }
    }

    // At this point the orbitals are sorted according to the energy but we
    // want to make sure that the hole and particle MO appear where they should,
    // that is
    // |(particles) (occupied spectators) | (virtual spectators) (hole)>
    TempMatrix->zero();
    //    TempVector->zero();
    SharedVector temp_epsilon_a_(new Vector("e_a", nsopi_));

    //    saved_Ch_->print();
    for (int h = 0; h < nirrep_; ++h) {
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        if (nso == 0 or nmo == 0)
            continue;
        double** T_h = TempMatrix->pointer(h);
        double** C_h = Ca_->pointer(h);
        double** Cp_h = Cp_->pointer(h);
        double** Ch_h = Ch_->pointer(h);
        double** saved_Cp_h = saved_Cp_->pointer(h);
        double** saved_Ch_h = saved_Ch_->pointer(h);

        int m = 0;
        // First place the (previously) projected holes and the particles
        if (do_project_out_holes) {
            for (int p = 0; p < project_naholepi_[h]; ++p) {
                temp_epsilon_a_->set(h, m, -200.0);
                for (int q = 0; q < nso; ++q) {
                    T_h[q][m] = saved_Ch_h[q][p];
                }
                m += 1;
            }
            //            outfile->Printf("\n %d saved alpha
            //            holes",saved_naholepi_[h]);
        }
        if (do_parts) {
            for (int p = 0; p < napartpi_[h]; ++p) {
                temp_epsilon_a_->set(h, m, -100.0);
                for (int q = 0; q < nso; ++q) {
                    T_h[q][m] = Cp_h[q][p];
                }
                m += 1;
            }
            //            outfile->Printf("\n %d particles",napartpi_[h]);
        }
        // Then the spectators
        int nspect = 0;
        for (int p = 0; p < nmo; ++p) {
            // Is this MO a hole or a particle?
            if (std::fabs(epsilon_a_->get(h, p)) > 1.0e-6) {
                temp_epsilon_a_->set(h, m, epsilon_a_->get(h, p));
                for (int q = 0; q < nso; ++q) {
                    T_h[q][m] = C_h[q][p];
                }
                m += 1;
                nspect += 1;
            }
        }
        //        outfile->Printf("\n %d spectators",nspect);
        // Then the (previously) projected particles and the holes
        if (do_holes) {
            for (int p = 0; p < naholepi_[h]; ++p) {
                temp_epsilon_a_->set(h, m, 100.0);
                for (int q = 0; q < nso; ++q) {
                    T_h[q][m] = Ch_h[q][p];
                }
                m += 1;
            }
            //            outfile->Printf("\n %d holes",naholepi_[h]);
        }
        if (do_project_out_particles) {
            for (int p = 0; p < saved_napartpi_[h]; ++p) {
                temp_epsilon_a_->set(h, m, 200.0);
                for (int q = 0; q < nso; ++q) {
                    T_h[q][m] = saved_Cp_h[q][p];
                }
                m += 1;
            }
            //            outfile->Printf("\n %d saved alpha
            //            particles",saved_napartpi_[h]);
        }

        //        outfile->Printf("\n Irrep %d has %d mos
        //        (%d,%d)",h,m,nsopi_[h],nmopi_[h]);
        //        outfile->Flush();
    }
    Ca_->copy(TempMatrix);
    epsilon_a_->copy(*temp_epsilon_a_);
}

void UOCDFT::diagonalize_F_spectator_unrelaxed() {
    //    // Frozen spectator orbital algorithm
    //    // Transform the ground state orbitals to the representation which
    //    diagonalizes the
    //    // the PoFaPo and PvFaPv blocks
    //    // Form the transformation matrix that diagonalizes the PoFaPo and
    //    PvFaPv blocks
    //    // |----|----|
    //    // | Uo | 0  |
    //    // |----|----|
    //    // | 0  | Uv |
    //    // |----|----|
    //    TempMatrix->zero();
    //    for (int h = 0; h < nirrep_; ++h){
    //        int nocc = dets[0]->nalphapi()[h];
    //        int nvir = nmopi_[h] - nocc;
    //        if (nocc != 0){
    //            double** Temp_h = TempMatrix->pointer(h);
    //            double** Uo_h = Ua_o_->pointer(h);
    //            for (int i = 0; i < nocc; ++i){
    //                epsilon_a_->set(h,i,lambda_a_o_->get(h,i));
    //                for (int j = 0; j < nocc; ++j){
    //                    Temp_h[i][j] = Uo_h[i][j];
    //                }
    //            }
    //        }
    //        if (nvir != 0){
    //            double** Temp_h = TempMatrix->pointer(h);
    //            double** Uv_h = Ua_v_->pointer(h);
    //            for (int i = 0; i < nvir; ++i){
    //                epsilon_a_->set(h,i + nocc,lambda_a_v_->get(h,i));
    //                for (int j = 0; j < nvir; ++j){
    //                    Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
    //                }
    //            }
    //        }
    //    }
    //    // Get the excited state orbitals: Ca(ex) = Ca(gs) * (Uo | Uv)
    //    Ca_->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix,0.0);

    //    // Form the projector onto the orbitals orthogonal to the holes and
    //    particles in the excited state mo representation
    //    TempMatrix->zero();
    //    TempMatrix->gemm(false,true,1.0,Ch_,Ch_,0.0);
    //    TempMatrix->gemm(false,true,1.0,Cp_,Cp_,1.0);
    //    TempMatrix->transform(S_);
    //    TempMatrix->transform(Ca_);
    //    TempMatrix2->identity();
    //    TempMatrix2->subtract(TempMatrix);

    //    // Form the Fock matrix in the excited state basis, project out the h/p
    //    TempMatrix->transform(Fa_,Ca_);
    //    TempMatrix->transform(TempMatrix2);
    //    // If we want the relaxed orbitals diagonalize the Fock matrix and
    //    transform the MO coefficients
    //    if(options_.get_str("CDFT_EXC_METHOD") == "CHP" or
    //    options_.get_str("CDFT_EXC_METHOD") == "CHP-FB"){
    //        TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
    //        TempMatrix->zero();
    //        TempMatrix->gemm(false,false,1.0,Ca_,TempMatrix2,0.0);
    //        Ca_->copy(TempMatrix);
    //    }else{
    //        // The orbitals don't change, but make sure that epsilon_a_ has the
    //        correct eigenvalues (some which are zero)
    //        for (int h = 0; h < nirrep_; ++h){
    //            for (int p = 0; p < nmopi_[h]; ++p){
    //                epsilon_a_->set(h,p,TempMatrix->get(h,p,p));
    //            }
    //        }
    //    }

    //    std::vector<std::tuple<double,int,int> > sorted_spectators;
    //    for (int h = 0; h < nirrep_; ++h){
    //        for (int p = 0; p < nmopi_[h]; ++p){
    //            sorted_spectators.push_back(std::make_tuple(epsilon_a_->get(h,p),h,p));
    //        }
    //    }
    //    std::sort(sorted_spectators.begin(),sorted_spectators.end());

    //    // Find the alpha occupation
    //    int assigned = 0;
    //    for (int h = 0; h < nirrep_; ++h){
    //        nalphapi_[h] = apartpi[h];
    //        assigned += apartpi[h];
    //    }
    //    for (int p = 0; p < nmo_; ++p){
    //        if (assigned < nalpha_){
    //            if(std::fabs(sorted_spectators[p].get<0>()) > 1.0e-6){  // !!!
    //            Check this out NB WARNING
    //                int h = sorted_spectators[p].get<1>();
    //                nalphapi_[h] += 1;
    //                assigned += 1;
    //            }
    //        }
    //    }
    //    nbetapi_ = dets[0]->nbetapi();
    //    int old_socc[8];
    //    int old_docc[8];
    //    for(int h = 0; h < nirrep_; ++h){
    //        old_socc[h] = soccpi_[h];
    //        old_docc[h] = doccpi_[h];
    //    }

    //    for (int h = 0; h < nirrep_; ++h) {
    //        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
    //        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
    //    }

    //    bool occ_changed = false;
    //    for(int h = 0; h < nirrep_; ++h){
    //        if( old_socc[h] != soccpi_[h] || old_docc[h] != doccpi_[h]){
    //            occ_changed = true;
    //            break;
    //        }
    //    }

    //    // At this point the orbitals are sorted according to the energy but we
    //    // want to make sure that the hole and particle MO appear where they
    //    should, that is
    //    // |(particles) (occupied spectators) | (virtual spectators) (hole)>
    //    TempMatrix->zero();
    //    TempVector->zero();
    //    for (int h = 0; h < nirrep_; ++h){
    //        int nso = nsopi_[h];
    //        int nmo = nmopi_[h];
    //        double** T_h = TempMatrix->pointer(h);
    //        double** C_h = Ca_->pointer(h);
    //        double** Cp_h = Cp_->pointer(h);
    //        double** Ch_h = Ch_->pointer(h);
    //        // First place the particles
    //        int m = 0;
    //        for (int p = 0; p < apartpi[h]; ++p){
    //            for (int q = 0; q < nso; ++q){
    //                T_h[q][m] = Cp_h[q][p];
    //            }
    //            m += 1;
    //        }
    //        // Then the spectators
    //        for (int p = 0; p < nmo; ++p){
    //            // Is this MO a hole or a particle?
    //            if(std::fabs(epsilon_a_->get(h,p)) > 1.0e-6){
    //                TempVector->set(h,m,epsilon_a_->get(h,p));
    //                for (int q = 0; q < nso; ++q){
    //                    T_h[q][m] = C_h[q][p];
    //                }
    //                m += 1;
    //            }
    //        }
    //        // Then the holes
    //        for (int p = 0; p < aholepi[h]; ++p){
    //            for (int q = 0; q < nso; ++q){
    //                T_h[q][m] = Ch_h[q][p];
    //            }
    //            m += 1;
    //        }
    //    }
    //    Ca_->copy(TempMatrix);
    //    epsilon_a_->copy(TempVector.get());

    //    // BETA
    //    if(options_.get_str("CDFT_EXC_METHOD") == "CHP"){
    //        diagonalize_F(Fb_, Cb_, epsilon_b_);
    //    }else{
    //        // Unrelaxed procedure, but still find MOs which diagonalize the
    //        occupied block
    //        // Transform Fb to the MO basis of the ground state
    //        TempMatrix->transform(Fb_,dets[0]->Cb());
    //        // Grab the occ block of Fb
    //        extract_square_subblock(TempMatrix,PoFaPo_,true,dets[0]->nbetapi(),1.0e9);
    //        // Grab the vir block of Fa
    //        extract_square_subblock(TempMatrix,PvFaPv_,false,dets[0]->nbetapi(),1.0e9);
    //        // Diagonalize the hole block
    //        PoFaPo_->diagonalize(Ua_o_,lambda_a_o_);
    //        // Diagonalize the particle block
    //        PvFaPv_->diagonalize(Ua_v_,lambda_a_v_);
    //        // Form the transformation matrix that diagonalizes the PoFaPo and
    //        PvFaPv blocks
    //        // |----|----|
    //        // | Uo | 0  |
    //        // |----|----|
    //        // | 0  | Uv |
    //        // |----|----|
    //        TempMatrix->zero();
    //        for (int h = 0; h < nirrep_; ++h){
    //            int nocc = dets[0]->nbetapi()[h];
    //            int nvir = nmopi_[h] - nocc;
    //            if (nocc != 0){
    //                double** Temp_h = TempMatrix->pointer(h);
    //                double** Uo_h = Ua_o_->pointer(h);
    //                for (int i = 0; i < nocc; ++i){
    //                    epsilon_b_->set(h,i,lambda_a_o_->get(h,i));
    //                    for (int j = 0; j < nocc; ++j){
    //                        Temp_h[i][j] = Uo_h[i][j];
    //                    }
    //                }
    //            }
    //            if (nvir != 0){
    //                double** Temp_h = TempMatrix->pointer(h);
    //                double** Uv_h = Ua_v_->pointer(h);
    //                for (int i = 0; i < nvir; ++i){
    //                    epsilon_b_->set(h,i + nocc,lambda_a_v_->get(h,i));
    //                    for (int j = 0; j < nvir; ++j){
    //                        Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
    //                    }
    //                }
    //            }
    //        }
    //        // Get the excited state orbitals: Cb(ex) = Cb(gs) * (Uo | Uv)
    //        Cb_->gemm(false,false,1.0,dets[0]->Cb(),TempMatrix,0.0);
    //    }
    //    if (debug_) {
    //        Ca_->print(outfile);
    //        Cb_->print(outfile);
    //    }
}

void UOCDFT::form_C_beta() {
    // BETA
    if (options_.get_str("CDFT_EXC_METHOD") != "CHP-FB") {
        diagonalize_F(Fb_, Cb_, epsilon_b_);
    } else {
        outfile->Printf("\n  Frozen beta algorithm\n");
        if (!PoFbPo_) {
            PoFbPo_ = SharedMatrix(new Matrix("PoFbPo", gs_nbetapi_, gs_nbetapi_));
            PvFbPv_ = SharedMatrix(new Matrix("PvFbPo", gs_nbvirpi_, gs_nbvirpi_));
            Ub_o_ = SharedMatrix(new Matrix("Ub_o_", gs_nbetapi_, gs_nbetapi_));
            Ub_v_ = SharedMatrix(new Matrix("Ub_v_", gs_nbvirpi_, gs_nbvirpi_));
            lambda_b_o_ = SharedVector(new Vector("lambda_b_o_", gs_nbetapi_));
            lambda_b_v_ = SharedVector(new Vector("lambda_b_v_", gs_nbvirpi_));
            outfile->Printf("\n  Allocated beta matrices!!!\n");
        }

        // Unrelaxed procedure, but still find MOs which diagonalize the occupied
        // block
        // Transform Fb to the MO basis of the ground state
        if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
            TempMatrix->transform(Fb_, dets[1]->Cb());
        } else {
            TempMatrix->transform(Fb_, dets[0]->Cb());
        }

        // Grab the occ block of Fb
        copy_block(TempMatrix, 1.0, PoFbPo_, 0.0, gs_nbetapi_, gs_nbetapi_);

        // Diagonalize the occ block
        PoFbPo_->diagonalize(Ub_o_, lambda_b_o_);

        // Grab the vir block of Fb
        copy_block(TempMatrix, 1.0, PvFbPv_, 0.0, gs_nbvirpi_, gs_nbvirpi_, gs_nbetapi_,
                   gs_nbetapi_);

        // Diagonalize the vir block
        PvFbPv_->diagonalize(Ub_v_, lambda_b_v_);

        TempMatrix->zero();
        copy_block(Ub_o_, 1.0, TempMatrix, 0.0, gs_nbetapi_, gs_nbetapi_);
        copy_block(Ub_v_, 1.0, TempMatrix, 0.0, gs_nbvirpi_, gs_nbvirpi_, Dimension(nirrep_),
                   Dimension(nirrep_), gs_nbetapi_, gs_nbetapi_);

        // Get the excited state orbitals: Cb(ex) = Cb(gs) * (Uo | Uv)
        if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
            Cb_->gemm(false, false, 1.0, dets[1]->Cb(), TempMatrix, 0.0);
        } else {
            Cb_->gemm(false, false, 1.0, dets[0]->Cb(), TempMatrix, 0.0);
        }
    }
}

double UOCDFT::compute_E() {

    // E_CDFT = 2.0 D*H + D*J - \alpha D*K + E_xc - Nc * Vc
    double one_electron_E = Da_->vector_dot(H_);
    one_electron_E += Db_->vector_dot(H_);

    double coulomb_E = Da_->vector_dot(J_);
    coulomb_E += Db_->vector_dot(J_);

    double XC_E = 0.0;
    double VV10_E = 0.0;
    if (functional_->needs_xc()) {
        XC_E = potential_->quadrature_values()["FUNCTIONAL"];
    }
    if (functional_->needs_vv10()) {
        VV10_E = potential_->quadrature_values()["VV10"];
    }

    //    std::map<std::string, double>& quad = potential_->quadrature_values();
    //    double XC_E = quad["FUNCTIONAL"];
    //    double exchange_E = 0.0;
    //    double alpha = functional_->x_alpha();
    //    double beta = 1.0 - alpha;
    //    if (functional_->is_x_hybrid()) {
    //        exchange_E -= alpha * Da_->vector_dot(Ka_);
    //        exchange_E -= alpha * Db_->vector_dot(Kb_);
    //    }

    double exchange_E = 0.0;
    double alpha = functional_->x_alpha();
    double beta = functional_->x_beta();
    if (functional_->is_x_hybrid()) {
        exchange_E -= alpha * Da_->vector_dot(Ka_);
        exchange_E -= alpha * Db_->vector_dot(Kb_);
    }
    if (functional_->is_x_lrc()) {
        exchange_E -= beta * Da_->vector_dot(wKa_);
        exchange_E -= beta * Db_->vector_dot(wKb_);
    }

    energies_["Nuclear"] = nuclearrep_;
    energies_["One-Electron"] = one_electron_E;
    energies_["Two-Electron"] = 0.5 * (coulomb_E + exchange_E);
    energies_["XC"] = XC_E;
    energies_["VV10"] = VV10_E;
    energies_["-D"] = variables_["-D Energy"];
    double dashD_E = energies_["-D"];

    double Etotal = 0.0;
    Etotal += nuclearrep_;
    Etotal += one_electron_E;
    Etotal += 0.5 * coulomb_E;
    Etotal += 0.5 * exchange_E;
    Etotal += XC_E;
    Etotal += dashD_E;

    Process::environment.globals["CURRENT ENERGY"] = Etotal;
    return Etotal;

    //    std::string functional_prefix = functional_->name().substr(0, 2);
    //    if (functional_prefix == "sr") {
    //        exchange_E += alpha * Da_->vector_dot(wKa_);
    //        exchange_E += alpha * Db_->vector_dot(wKb_);
    //    } else {
    //        if (functional_->is_x_lrc()) {
    //            exchange_E -= beta * Da_->vector_dot(wKa_);
    //            exchange_E -= beta * Db_->vector_dot(wKb_);
    //        }
    //    }

    //    double dashD_E = 0.0;
    //    std::shared_ptr<Dispersion> disp;
    //    if (disp) {
    //        dashD_E = disp->compute_energy(HF::molecule_);
    //    }

    //    double Etotal = 0.0;
    //    Etotal += nuclearrep_;
    //    Etotal += one_electron_E;
    //    Etotal += 0.5 * coulomb_E;
    //    Etotal += 0.5 * exchange_E;
    //    Etotal += XC_E;
    //    Etotal += dashD_E;

    // SharedMatrix Vmo(Va_->clone());
    // Vmo->transform(Va_,Ca_);
    // Vmo->print();
    // for (int h = 0; h < 20; ++h){
    // double print_it = 0.0;
    // print_it = Vmo->get(h,h);
    // outfile->Printf( "\n V_%d-%d: %f \n", h,h,print_it);
    //}

    //    energies_["Nuclear"] = nuclearrep_;
    //    energies_["One-Electron"] = one_electron_E;
    //    energies_["Two-Electron"] = 0.5 * (coulomb_E + exchange_E);
    //    energies_["XC"] = XC_E;
    //    energies_["-D"] = dashD_E;

    // outfile->Printf( "   => Energetics <=\n\n");
    // outfile->Printf( "    Nuclear Repulsion Energy = %24.14f\n", nuclearrep_);
    // outfile->Printf( "    One-Electron Energy =      %24.14f\n",
    // one_electron_E);
    // outfile->Printf( "    Coulomb Energy =           %24.14f\n", 0.5 *
    // coulomb_E);
    // outfile->Printf( "    Hybrid Exchange Energy =   %24.14f\n", 0.5 *
    // exchange_E);
    // outfile->Printf( "    XC Functional Energy =     %24.14f\n", XC_E);
    // outfile->Printf( "    -D Energy =                %24.14f\n", dashD_E);
    // Da_->print();
    // screen_virtuals();
    //    return Etotal;
}

void UOCDFT::pFON() {

    // Pseudo-Fractional Occupation Method (pFON) follows the formulas detailed in
    // J.Chem.Phys.,Vol.110,No.2,8 January 1999

    // Grab initial temp from input and "cool off" 50K each iteration
    double init_temp = options_.get_double("PFON_TEMP") - 50.0 * iteration_;
    if (init_temp > 0.0) {
        // Grab DIIS error matrix
        SharedMatrix errveca(moFeffa_);
        errveca->zero_diagonal();
        errveca->back_transform(Ca_);
        SharedMatrix errvecb(moFeffb_);
        errvecb->zero_diagonal();
        errvecb->back_transform(Cb_);
        double max_erra = 0.0;
        double max_errb = 0.0;
        for (int i = 0; i < basisset_->nbf(); ++i) {
            for (int j = 0; j < basisset_->nbf(); ++j) {
                max_erra = std::max(errveca->get(i, j), max_erra);
                max_errb = std::max(errvecb->get(i, j), max_errb);
            }
        }
        outfile->Printf("\n\n ----------------------------");
        outfile->Printf("\n pFON Stats for iteration %d", iteration_ - 1);
        outfile->Printf("\n ----------------------------");
        outfile->Printf("\n Initial Temperature: %f", init_temp);
        double tempa = 0.5 * init_temp * max_erra;
        // If DIIS error is very low, set temp to zero to preserve integer
        // occupations
        if (max_erra < 1E-4) {
            tempa = 0.0;
        }
        double k = 8.617e-5;
        double betaa = 1.0 / (k * tempa);
        double tempb = 0.5 * init_temp * max_errb;
        if (max_errb < 1E-4) {
            tempb = 0.0;
        }
        double betab = 1.0 / (k * tempb);
        outfile->Printf("\n Current Temperature (alpha): %f", tempa);
        outfile->Printf("\n Current Temperature (beta): %f", tempb);
        outfile->Printf("\n DIIS Error (alpha): %f", max_erra);
        outfile->Printf("\n DIIS Error (beta): %f", max_errb);
        int nocc = gs_nalphapi_[0];
        double e_homo = epsilon_b_->get(nocc - 1);
        outfile->Printf("\n Homo: %f", e_homo);
        double e_lumo = epsilon_b_->get(nocc);
        outfile->Printf("\n Lumo: %f \n\n", e_lumo);
        // Fermi level is set as halfway b/w HOMO and LUMO
        double e_fermi = (e_homo + e_lumo) / 2;
        double occ_suma = 0.0;
        double occ_sumb = 0.0;
        for (int i = 0; i < nocc; ++i) {
            double e_i = epsilon_b_->get(i);
            double occupation_numbera = 1.0 / (1.0 + std::exp(betaa * (e_i - e_fermi)));
            occ_suma += occupation_numbera;
            double occupation_numberb = 1.0 / (1.0 + std::exp(betab * (e_i - e_fermi)));
            occ_sumb += occupation_numberb;
        }
        for (int p = 0; p < basisset_->nbf(); ++p) {
            for (int q = 0; q < basisset_->nbf(); ++q) {
                double da_element = 0.0;
                double db_element = 0.0;
                for (int i = 0; i < basisset_->nbf(); ++i) {
                    double e_i = epsilon_b_->get(i);
                    // outfile->Printf("\n eigenvalue %d: %f", i, e_i);
                    double occupation_numbera = 1.0 / (1.0 + std::exp(betaa * (e_i - e_fermi)));
                    double occupation_numbera_norm = (occupation_numbera * nocc) / (occ_suma);
                    // outfile->Printf("\n Occupation of orbital %d: %f", i,
                    // occupation_numbera_norm);
                    da_element += occupation_numbera_norm * Ca_->get(p, i) * Ca_->get(q, i);
                    double occupation_numberb = 1.0 / (1.0 + std::exp(betab * (e_i - e_fermi)));
                    double occupation_numberb_norm = (occupation_numberb * nocc) / (occ_sumb);
                    db_element += occupation_numberb_norm * Cb_->get(p, i) * Cb_->get(q, i);
                }
                Da_->set(p, q, da_element);
                Db_->set(p, q, db_element);
            }
        }
    }
    // Update Density Matrix
    Dt_->copy(Da_);
    Dt_->add(Db_);
    // Dt_->print();
}
void UOCDFT::damp_update() {
    // Turn on damping only for excited state computations
    if (do_excitation) {
        double damping_percentage_ = options_.get_double("DAMPING_PERCENTAGE");
        double damping = damping_percentage_ / 100.0;
        for (int h = 0; h < nirrep_; ++h) {
            for (int row = 0; row < Da_->rowspi(h); ++row) {
                for (int col = 0; col < Da_->colspi(h); ++col) {
                    double Dolda = damping * Dolda_->get(h, row, col);
                    double Dnewa = (1.0 - damping) * Da_->get(h, row, col);
                    Da_->set(h, row, col, Dolda + Dnewa);
                    double Doldb = damping * Doldb_->get(h, row, col);
                    double Dnewb = (1.0 - damping) * Db_->get(h, row, col);
                    Db_->set(h, row, col, Doldb + Dnewb);
                }
            }
        }
        // Update Dt_
        Dt_->copy(Da_);
        Dt_->add(Db_);
        SharedMatrix errveca(moFeffa_);
        errveca->zero_diagonal();
        errveca->back_transform(Ca_);
        // errveca->print();
        double maximum = 0.0;
        for (int i = 0; i < basisset_->nbf(); ++i)
            for (int j = 0; j < basisset_->nbf(); ++j)
                maximum = std::max(errveca->get(i, j), maximum);
        // double err_max = std::max(errveca);
        outfile->Printf("Max_err: %f", maximum);
    }
}

bool UOCDFT::test_convergency() {
    // energy difference
    double ediff = E_ - Eold_;

    // RMS of the density
    Matrix D_rms;
    D_rms.copy(Dt_);
    D_rms.subtract(Dt_old_);
    Drms_ = 0.5 * D_rms.rms();

    bool energy_test = fabs(ediff) < energy_threshold_;
    bool density_test = Drms_ < density_threshold_;
    bool cycle_test = iteration_ > 5;

    if (state_ > 0) {
        if (iteration_ > options_.get_int("OCDFT_MAX_ITER"))
            return true;
    }
    if (energy_test and density_test and cycle_test) {
        return true;
    }
    return false;
}

void UOCDFT::save_information() {
    
    //    saved_naholepi_ = naholepi_;
    //    saved_napartpi_ = napartpi_;

    dets.push_back(SharedDeterminant(new Determinant(E_, Ca_, Cb_, nalphapi_, nbetapi_)));

    if (do_excitation) {
        Process::environment.globals["DFT ENERGY"] = ground_state_energy;

        // if (options_.get_bool("ANALYZE_EXCITATIONS")){
        if (nirrep_ == 1) {
            analyze_excitations();
        } else {
            outfile->Printf(
                "\n\n  Skipping analysis of excitations. To enable run in C1 symmetry.\n");
        }
        // }
        std::vector<std::tuple<double, int, int>> info_a_;
        int nirrep = wfn_->nirrep();
        Dimension nmopi = wfn_->nmopi();
        // Gather orbital information
        for (int h = 0; h < nirrep; h++) {
            for (int i = 0; i < (int)nmopi[h]; i++) {
                info_a_.push_back(std::tuple<double, int, int>(wfn_->epsilon_a()->get(h, i), i, h));
            }
        }
        std::sort(info_a_.begin(), info_a_.end(),
                  std::less<std::tuple<double, int, int>>()); // Sort as in wfn
        std::vector<int> indsp0;
        std::vector<int> indsh0;
        int orb_tup_size = info_a_.size();

        for (int ind = 0; ind < Ca_->colspi()[0]; ind++) {
            if (epsilon_a_->get(ind) == -100.0) {
                indsp0.push_back(ind);
            }
            if (epsilon_a_->get(ind) == 100.0) {
                indsh0.push_back(ind);
            }
        }
        std::vector<std::string> labelsp;
        std::vector<std::string> labelsh;
        CharacterTable ct = basisset_->molecule()->point_group()->char_table();
        for (size_t ind = 0; ind < indsp0.size(); ++ind) {
            int i = std::get<1>(info_a_[indsp0[ind]]);
            int h = std::get<2>(info_a_[indsp0[ind]]);
            labelsp.push_back(ct.gamma(h).symbol());
        }
        for (size_t ind = 0; ind < indsh0.size(); ++ind) {
            int i = std::get<1>(info_a_[indsh0[ind]]);
            int h = std::get<2>(info_a_[indsh0[ind]]);
            labelsh.push_back(ct.gamma(h).symbol());
        }
        CubeProperties cube = CubeProperties(wfn_);

        SharedMatrix Dp_ =
            SharedMatrix(new Matrix("Particle Attachment Density", nsopi_, gs_navirpi_));
        SharedMatrix Ddiff_(Da_->clone());
        Dp_->gemm(false, true, 1.0, Cp_, Cp_, 0.0);
        SharedMatrix Dh_ =
            SharedMatrix(new Matrix("Hole Detachment Density", nsopi_, gs_nalphapi_));
        Dh_->gemm(false, true, 1.0, Ch_, Ch_, 0.0);
        std::string particle_str = fmt::format("particle_{:d}", state_);
        std::string hole_str = fmt::format("hole_{:d}", state_);
        if (options_.get_bool("CUBE_HP")) {
            outfile->Printf("\n  Computing cube files\n");
            cube.compute_orbitals(Ca_, indsp0, labelsp, particle_str);
            cube.compute_orbitals(Ca_, indsh0, labelsh, hole_str);

            // cube.compute_density(Dp_, "Dp");
            // cube.compute_density(Dh_, "Dh");
        }
        // cube.compute_properties();
        // grid_->print_header();
        compute_transition_moments(wfn_);

        double mixlet_exc_energy = E_ - ground_state_energy;
        outfile->Printf("  Excited mixed state   : excitation energy = %9.6f Eh = "
                        "%8.4f eV = %9.1f cm**-1 \n",
                        mixlet_exc_energy, mixlet_exc_energy * pc_hartree2ev,
                        mixlet_exc_energy * pc_hartree2wavenumbers);
        if (options_.get_bool("CDFT_SPIN_ADAPT_CI")) {
            spin_adapt_mixed_excitation();
        }
        if (options_.get_bool("CDFT_SPIN_ADAPT_SP")) {
            compute_S_plus_triplet_correction();
        }

        if (do_save_holes) {
            // Add the saved holes to the Ch_ matrix
            // The information about previous holes is passed via saved_Ch_
            saved_naholepi_.print();
            for (int h = 0; h < nirrep_; ++h) {
                for (int i = 0; i < naholepi_[h]; ++i) {
                    saved_Ch_->set_column(h, saved_naholepi_[h] + i, Ch_->get_column(h, i));
                }
            }
            saved_naholepi_ += naholepi_;
        }
        if (do_save_particles) {
            // Add the saved particles to the Cp_ matrix
            saved_napartpi_.print();
            for (int h = 0; h < nirrep_; ++h) {
                for (int i = 0; i < saved_napartpi_[h]; ++i) {
                    Cp_->set_column(h, napartpi_[h] + i, saved_Cp_->get_column(h, i));
                }
            }
            napartpi_ += saved_napartpi_;
        }
    }
    //    if(options_.get_str("CDFT_EXC_METHOD") == "CIS")
    //        cis_excitation_energy();
    //    if(options_["CDFT_BREAK_SYMMETRY"].has_changed()){
    //        spin_adapt_mixed_excitation();
    //        compute_S_plus_triplet_correction();
    //    }
}

void UOCDFT::compute_transition_moments(SharedWavefunction ref_scf) {
    double overlap = 0.0;
    double hamiltonian = 0.0;

    outfile->Printf("\n  Computing transition dipole moments");
    outfile->Printf("\n  %d determinants stored", static_cast<int>(dets.size()));
    int temp_index;

    if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
        temp_index = 1;
    } else {
        temp_index = 0;
    }

    std::tuple<SharedMatrix, SharedMatrix, SharedVector, double> calpha;
    std::tuple<SharedMatrix, SharedMatrix, SharedVector, double> cbeta;

    SharedDeterminant A = dets[temp_index];
    SharedDeterminant B = dets[dets.size() - 1];
    if (nirrep_ != 1) {
        // I. Form the corresponding alpha and beta orbitals
        calpha = corresponding_orbitals(A->Ca(), B->Ca(), A->nalphapi(), B->nalphapi());
        cbeta = corresponding_orbitals(A->Cb(), B->Cb(), A->nbetapi(), B->nbetapi());
    } else {
        TempMatrix->zero();
        TempMatrix2->zero();
        copy_block(Ca_, 1.0, TempMatrix, 0.0, nsopi_, nalphapi_);
        std::shared_ptr<IAOBuilder> iao = IAOBuilder::build(
            wfn_->basisset(), wfn_->get_basisset("MINAO_BASIS"), TempMatrix, options_);
        std::map<std::string, SharedMatrix> ret;
        ret = iao->build_iaos();

        SharedMatrix IAO =
            SharedMatrix(new Matrix("IAO coeffs in primary", basisset_->nbf(), basisset_->nbf()));
        SharedMatrix iao_coeffs2 = ret["A"];
        int nmin = iao_coeffs2->colspi()[0];
        for (int i = 0; i < basisset_->nbf(); ++i) {
            for (int j = 0; j < nmin; ++j) {
                IAO->set(i, j, iao_coeffs2->get(i, j));
            }
        }
        SharedMatrix Ca_A_iao = SharedMatrix(
            new Matrix("Determinant A alpha coeffs IAO Basis", basisset_->nbf(), basisset_->nbf()));
        SharedMatrix Ca_B_iao = SharedMatrix(
            new Matrix("Determinant B alpha coeffs IAO Basis", basisset_->nbf(), basisset_->nbf()));
        SharedMatrix Cb_A_iao = SharedMatrix(
            new Matrix("Determinant A beta coeffs IAO Basis", basisset_->nbf(), basisset_->nbf()));
        SharedMatrix Cb_B_iao = SharedMatrix(
            new Matrix("Determinant B beta coeffs IAO Basis", basisset_->nbf(), basisset_->nbf()));
        TempMatrix->gemm(true, false, 1.0, IAO, S_, 0.0);
        SharedDeterminant A = dets[temp_index];
        SharedDeterminant B = dets[dets.size() - 1];
        // I. Form the corresponding alpha and beta orbitals
        SharedMatrix Ca_A(A->Ca()->clone());
//        outfile->Printf("\n HERE!");
        Ca_A_iao->gemm(false, false, 1.0, TempMatrix, Ca_A, 0.0);
//        outfile->Printf("\n HERE!");
        // Ca_A_iao->print();
        SharedMatrix Ca_B(B->Ca()->clone());
        Ca_B_iao->gemm(false, false, 1.0, TempMatrix, Ca_B, 0.0);
        SharedMatrix Cb_A(A->Cb()->clone());
        Cb_A_iao->gemm(false, false, 1.0, TempMatrix, Cb_A, 0.0);
        SharedMatrix Cb_B(B->Cb()->clone());
        Cb_B_iao->gemm(false, false, 1.0, TempMatrix, Cb_B, 0.0);
        calpha = corresponding_orbitals(A->Ca(), B->Ca(), A->nalphapi(), B->nalphapi());
        cbeta = corresponding_orbitals(A->Cb(), B->Cb(), A->nbetapi(), B->nbetapi());
    }
    SharedMatrix ACa = std::get<0>(calpha);
    SharedMatrix BCa = std::get<1>(calpha);
    SharedMatrix ACb = std::get<0>(cbeta);
    SharedMatrix BCb = std::get<1>(cbeta);
    double detUValpha = std::get<3>(calpha);
    double detUVbeta = std::get<3>(cbeta);
    SharedVector s_a = std::get<2>(calpha);
    SharedVector s_b = std::get<2>(cbeta);

    // Compute the number of noncoincidences
    double noncoincidence_threshold = 1.0e-9;

    std::vector<std::tuple<int, int, double>> Aalpha_nonc;
    std::vector<std::tuple<int, int, double>> Balpha_nonc;
    std::vector<std::tuple<double, int, int>> nonc;
    double Sta = 1.0;
    for (int h = 0; h < nirrep_; ++h) {
        // Count all the numerical noncoincidences
        int nmin = std::min(A->nalphapi()[h], B->nalphapi()[h]);
        for (int p = 0; p < nmin; ++p) {
            if (std::fabs(s_a->get(h, p)) >= noncoincidence_threshold) {
                Sta *= s_a->get(h, p);
                // outfile->Printf("Not Accepted: %f \n ", std::fabs(s_a->get(h,p)));
            } else {
                // outfile->Printf("Accepted: %f \n ", std::fabs(s_a->get(h,p)));
                nonc.push_back(std::make_tuple(std::fabs(s_a->get(h, p)), h, p));
                std::sort(nonc.begin(), nonc.end());
                Aalpha_nonc.push_back(std::make_tuple(std::get<1>(nonc[0]), std::get<2>(nonc[0]),
                                                      std::get<0>(nonc[0])));
                Balpha_nonc.push_back(std::make_tuple(std::get<1>(nonc[0]), std::get<2>(nonc[0]),
                                                      std::get<0>(nonc[0])));
            }
        }
        // Count all the symmetry noncoincidences
        int nmax = std::max(A->nalphapi()[h], B->nalphapi()[h]);
        bool AgeB = A->nalphapi()[h] >= B->nalphapi()[h] ? true : false;
        for (int p = nmin; p < nmax; ++p) {
            if (AgeB) {
                Aalpha_nonc.push_back(std::make_tuple(h, p, 0.0));
            } else {
                Balpha_nonc.push_back(std::make_tuple(h, p, 0.0));
            }
        }
    }

    std::vector<std::tuple<int, int, double>> Abeta_nonc;
    std::vector<std::tuple<int, int, double>> Bbeta_nonc;
    double Stb = 1.0;
    for (int h = 0; h < nirrep_; ++h) {
        // Count all the numerical noncoincidences
        int nmin = std::min(A->nbetapi()[h], B->nbetapi()[h]);
        for (int p = 0; p < nmin; ++p) {
            if (std::fabs(s_b->get(h, p)) >= noncoincidence_threshold) {
                Stb *= s_b->get(h, p);
            } else {
                Abeta_nonc.push_back(std::make_tuple(h, p, s_b->get(h, p)));
                Bbeta_nonc.push_back(std::make_tuple(h, p, s_b->get(h, p)));
            }
        }
        // Count all the symmetry noncoincidences
        int nmax = std::max(A->nbetapi()[h], B->nbetapi()[h]);
        bool AgeB = A->nbetapi()[h] >= B->nbetapi()[h] ? true : false;
        for (int p = nmin; p < nmax; ++p) {
            if (AgeB) {
                Abeta_nonc.push_back(std::make_tuple(h, p, 0.0));
            } else {
                Bbeta_nonc.push_back(std::make_tuple(h, p, 0.0));
            }
        }
    }
    outfile->Printf("\n  Corresponding orbitals:\n");
    outfile->Printf("  A(alpha): ");
    for (size_t k = 0; k < Aalpha_nonc.size(); ++k) {
        int i_h = std::get<0>(Aalpha_nonc[k]);
        int i_mo = std::get<1>(Aalpha_nonc[k]);
        outfile->Printf(" (%1d,%2d)", i_h, i_mo);
    }
    outfile->Printf("\n  B(alpha): ");
    for (size_t k = 0; k < Balpha_nonc.size(); ++k) {
        int i_h = std::get<0>(Balpha_nonc[k]);
        int i_mo = std::get<1>(Balpha_nonc[k]);
        outfile->Printf(" (%1d,%2d)", i_h, i_mo);
    }
    outfile->Printf("\n  s(alpha): ");
    for (size_t k = 0; k < Balpha_nonc.size(); ++k) {
        double i_s = std::get<2>(Balpha_nonc[k]);
        outfile->Printf(" %6e", i_s);
    }
    outfile->Printf("\n  A(beta):  ");
    for (size_t k = 0; k < Abeta_nonc.size(); ++k) {
        int i_h = std::get<0>(Abeta_nonc[k]);
        int i_mo = std::get<1>(Abeta_nonc[k]);
        outfile->Printf(" (%1d,%2d)", i_h, i_mo);
    }
    outfile->Printf("\n  B(beta):  ");
    for (size_t k = 0; k < Bbeta_nonc.size(); ++k) {
        int i_h = std::get<0>(Bbeta_nonc[k]);
        int i_mo = std::get<1>(Bbeta_nonc[k]);
        outfile->Printf(" (%1d,%2d)", i_h, i_mo);
    }
    outfile->Printf("\n  s(beta):  ");
    for (size_t k = 0; k < Bbeta_nonc.size(); ++k) {
        double i_s = std::get<2>(Bbeta_nonc[k]);
        outfile->Printf(" %6e", i_s);
    }

    double Stilde = Sta * Stb * detUValpha * detUVbeta;
    outfile->Printf("\n  Stilde = %.6f\n", Stilde);

    // Irreps and MO index of the alpha corresponding orbitals
    int i_A_h = std::get<0>(Aalpha_nonc[0]);
    int i_A_mo = std::get<1>(Aalpha_nonc[0]);

    int i_B_h = std::get<0>(Balpha_nonc[0]);
    int i_B_mo = std::get<1>(Balpha_nonc[0]);

    int i_AB_h = i_A_h ^ i_B_h;

    // For a single noncoincidence the matrix element of <A|mu|B> = tilde{S}_AB *
    // <phi^A_i|mu|phi^B_i>.
    // There is no contribution from the beta orbitals.

    SharedMatrix trDa_c1 = SharedMatrix(new Matrix("Transition Density Matrix (alpha, C1 symmetry)",
                                                   basisset_->nbf(), basisset_->nbf()));
    SharedMatrix SO2AO_c1 =
        SharedMatrix(new Matrix("SO2AO (C1 symmetry)", basisset_->nbf(), basisset_->nbf()));

    SharedMatrix trDa =
        SharedMatrix(new Matrix("Transition Density Matrix (alpha)", nsopi_, nsopi_, i_AB_h));
    SharedMatrix trDb =
        SharedMatrix(new Matrix("Transition Density Matrix (beta)", nsopi_, nsopi_));
    trDb->zero();
    double** ca = ACa->pointer(i_A_h);
    double** cb = BCa->pointer(i_B_h);
    double** da = trDa->pointer(i_AB_h);
    double** da_c1 = trDa_c1->pointer();

    size_t offset_A = 0;
    size_t offset_B = 0;
    for (int h = 0; h < i_A_h; ++h) {
        offset_A += nsopi_[h];
    }
    for (int h = 0; h < i_B_h; ++h) {
        offset_B += nsopi_[h];
    }
    for (int mu = 0; mu < nsopi_[i_A_h]; ++mu) {
        for (int nu = 0; nu < nsopi_[i_B_h]; ++nu) {
            da[nu][mu] = Stilde * ca[mu][i_A_mo] * cb[nu][i_B_mo];
            da_c1[offset_B + nu][offset_A + mu] = Stilde * ca[mu][i_A_mo] * cb[nu][i_B_mo];
        }
    }
    SharedMatrix trDa_ao =
        SharedMatrix(new Matrix("AO Density", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix trDa_iao =
        SharedMatrix(new Matrix("IAO Density", basisset_->nbf(), basisset_->nbf()));

    SharedMatrix S_ao =
        SharedMatrix(new Matrix("AO Overlap matrix", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix S_half_ao = SharedMatrix(
        new Matrix("Square root of the AO Overlap matrix", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix S_inv_half_ao = SharedMatrix(new Matrix(
        "Inverse square root of the AO Overlap matrix", basisset_->nbf(), basisset_->nbf()));

    double de[3];
    std::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
    SharedMatrix SO2AO_ = pet->sotoao();

    double** so2ao_c1 = SO2AO_c1->pointer();
    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        double** so2ao = SO2AO_->pointer(h);
        for (int mu = 0; mu < nsopi_[h]; ++mu) {
            for (int nu = 0; nu < basisset_->nbf(); ++nu) {
                so2ao_c1[offset + mu][nu] = so2ao[mu][nu];
            }
        }
        offset += nsopi_[h];
    }

    trDa_ao->transform(trDa_c1, SO2AO_c1);
    // trDa_iao->transform(trDa_c1,IAO_coeffs);
    //    trDa_ao->remove_symmetry(trDa,SO2AO_);
    S_ao->remove_symmetry(S_, SO2AO_);
    S_half_ao->copy(S_ao);
    S_half_ao->power(0.5);
    S_inv_half_ao->copy(S_ao);
    S_inv_half_ao->power(-0.5);

    // Contract the dipole moment operators with the pseudo-density matrix
    std::shared_ptr<OEProp> oe(new OEProp(ref_scf));
    oe->set_title("OCDFT TRANSITION");
    oe->add("TRANSITION_DIPOLE");
    oe->set_Da_so(trDa);
    oe->set_Db_so(trDb);
    outfile->Printf("  ==> Transition dipole moment computed with OCDFT <==\n\n");
    oe->compute();

    // SVD To obtain NTOs
    // std::tuple<SharedMatrix, SharedVector, SharedMatrix> UtrdV =
    // trDa->svd_temps();
    // SharedMatrix U = UtrdV.get<0>();
    // SharedVector sigma = UtrdV.get<1>();
    // SharedMatrix V = UtrdV.get<2>();
    // trDa->svd(U,sigma,V);
    // sigma->print();
    // U->print();
    // V->print();
    // double nto_participation_ratio = 0.0;
    // double square_of_sums = 0.0;
    // double sum_of_squares = 0.0;

    // for (int i = 0; i < basisset_->nbf(); i++){
    //    sum_of_squares += sigma->get(i)*sigma->get(i);
    //    square_of_sums += sigma->get(i);
    //}

    // nto_participation_ratio = square_of_sums*square_of_sums/(sum_of_squares);
    // outfile->Printf( " PRnto = %f \n\n", nto_participation_ratio);
    //    Dt_->copy(Da_);
    //    Dt_->add(Db_);

    std::vector<SharedMatrix> dipole_ints;
    dipole_ints.push_back(
        SharedMatrix(new Matrix("AO Dipole X", basisset_->nbf(), basisset_->nbf())));
    dipole_ints.push_back(
        SharedMatrix(new Matrix("AO Dipole Y", basisset_->nbf(), basisset_->nbf())));
    dipole_ints.push_back(
        SharedMatrix(new Matrix("AO Dipole Z", basisset_->nbf(), basisset_->nbf())));
    std::shared_ptr<OneBodyAOInt> aodOBI(integral_->ao_dipole());
    Vector3 origin(0.0, 0.0, 0.0);
    aodOBI->set_origin(origin);
    aodOBI->compute(dipole_ints);

    de[0] = trDa_ao->vector_dot(dipole_ints[0]);
    de[1] = trDa_ao->vector_dot(dipole_ints[1]);
    de[2] = trDa_ao->vector_dot(dipole_ints[2]);

    std::tuple<SharedMatrix, SharedVector, SharedMatrix> UtrdV = trDa_ao->svd_temps();
    SharedMatrix U = std::get<0>(UtrdV);
    SharedVector sigma = std::get<1>(UtrdV);
    SharedMatrix V = std::get<2>(UtrdV);
    trDa_ao->svd(U, sigma, V);
    // sigma->print();
    double nto_participation_ratio = 0.0;
    double square_of_sums = 0.0;
    double sum_of_squares = 0.0;

    for (int i = 0; i < basisset_->nbf(); i++) {
        sum_of_squares += sigma->get(i) * sigma->get(i);
        // outfile->Printf("\n sum%d - %f \n", i, sum_of_squares);
        square_of_sums += sigma->get(i);
    }

    nto_participation_ratio = square_of_sums * square_of_sums / (sum_of_squares);

    double full_abs_dipole = std::sqrt(de[0] * de[0] + de[1] * de[1] + de[2] * de[2]);
    outfile->Printf("\n  Dipole moments from AO integrals: %.4f %.4f %.4f", de[0], de[1], de[2]);

    // Form a map that lists all functions on a given atom and with a given ang.
    // momentum
    std::map<std::pair<int, int>, std::vector<int>> atom_am_to_f;
    int sum = 0;
    for (int A = 0; A < molecule_->natom(); A++) {
        // outfile->Printf("\n Atom %d",A);
        int n_shell = basisset_->nshell_on_center(A);
        for (int Q = 0; Q < n_shell; Q++) {
            const GaussianShell& shell = basisset_->shell(A, Q);
            int nfunction = shell.nfunction();
            int am = shell.am();
            std::pair<int, int> atom_am(A, am);
            for (int p = sum; p < sum + nfunction; ++p) {
                atom_am_to_f[atom_am].push_back(p);
            }
            sum += nfunction;
        }
    }

    std::vector<std::pair<int, int>> keys;
    for (auto& kv : atom_am_to_f) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());

    std::vector<std::tuple<std::string, std::string, std::string, std::string, double, double,
                           double, double, double, double>>
        all_atomic_trans;
    std::vector<std::pair<double, std::string>> restricted_sums;

    std::vector<std::string> l_to_symbol{"s", "p", "d", "f", "g", "h"};

    std::vector<std::pair<double, std::string>> sorted_contributions;
    SharedMatrix trDa_ao_atom =
        SharedMatrix(new Matrix("AO Density", basisset_->nbf(), basisset_->nbf()));
    SharedMatrix trDa_atom_iao =
        SharedMatrix(new Matrix("AO Density", basisset_->nbf(), basisset_->nbf()));
    // SharedMatrix trDa_ao_atom_iao = SharedMatrix(new Matrix("AO Density [IAO
    // Basis]", nmin,nmin));
    for (auto& i : keys) {
        auto& ifn = atom_am_to_f[i];
        for (auto& f : keys) {
            trDa_ao_atom->zero();
            trDa_atom_iao->zero();
            auto& ffn = atom_am_to_f[f];
            for (int iao : ifn) {
                for (int fao : ffn) {
                    double value = trDa_ao->get(fao, iao);
                    trDa_ao_atom->set(fao, iao, value);
                }
            }
            // trDa_ao_atom_iao->transform(IAO_coeffs);
            // trDa_ao_atom_iao->print();
            de[0] = trDa_ao_atom->vector_dot(dipole_ints[0]);
            de[1] = trDa_ao_atom->vector_dot(dipole_ints[1]);
            de[2] = trDa_ao_atom->vector_dot(dipole_ints[2]);

            std::tuple<SharedMatrix, SharedVector, SharedMatrix> UtrdaV =
                trDa_atom_iao->svd_temps();
            SharedMatrix U_atom = std::get<0>(UtrdaV);
            SharedVector sigma_atom = std::get<1>(UtrdaV);
            SharedMatrix V_atom = std::get<2>(UtrdaV);
            trDa_ao_atom->svd(U_atom, sigma_atom, V_atom);

            double sum_of_squares_atom = 0.0;
            double square_of_sums_atom = 0.0;
            double charge_transfer_number = 0.0;
            for (int i = 0; i < basisset_->nbf(); i++) {
                sum_of_squares_atom += sigma_atom->get(i) * sigma_atom->get(i);
                // outfile->Printf("\n ao%d - %f \n", i, sum_of_squares_atom);
                // outfile->Printf("\n ao%d - %f \n", i, sum_of_squares);
                square_of_sums_atom += sigma_atom->get(i);
            }
            // outfile->Printf("\ntotal: %f \n", sum_of_squares_atom);
            double participation_percentage = ((sum_of_squares_atom) / (sum_of_squares)) * 100.0;
            // charge_transfer_number = (square_of_sums_atom*square_of_sums_atom);
            double abs_dipole = std::sqrt(de[0] * de[0] + de[1] * de[1] + de[2] * de[2]);
            double dipole_percent = abs_dipole / (full_abs_dipole)*100.0;
            if (dipole_percent >= 2.0) {
                std::string outstr = fmt::format("  {:3d} {:2s} {:1s}  {:3d} {:2s} {:1s}  {:9f}  {:9f}  {:9f}  {:9f} {:9.2f}% {:9f}",
                    (i.first + 1) , molecule_->symbol(i.first).c_str() ,
                    l_to_symbol[i.second].c_str() , (f.first + 1) ,
                    molecule_->symbol(f.first).c_str() , l_to_symbol[f.second].c_str() , de[0] ,
                    de[1] , de[2] , abs_dipole , participation_percentage , charge_transfer_number);
                sorted_contributions.push_back(std::make_pair(dipole_percent, outstr));
                all_atomic_trans.push_back(std::make_tuple(
                    molecule_->symbol(i.first).c_str(), l_to_symbol[i.second].c_str(),
                    molecule_->symbol(f.first).c_str(), l_to_symbol[f.second].c_str(), abs_dipole,
                    participation_percentage, de[0], de[1], de[2], dipole_percent));
            }
        }
    }
    outfile->Printf("\n    ==> Restricted Sum from Atomic Population Analysis of "
                    "NTO contributions <==\n");
    outfile->Printf("\n   "
                    "============================================================"
                    "=======================");
    outfile->Printf("\n   Atomic Transition     PR%%       |mu|%%       |mu|     "
                    "mu(x)      mu(y)      mu(z)  ");
    outfile->Printf("\n   "
                    "============================================================"
                    "=======================");
    std::vector<std::string> duplicates;
    int atomic_trans = all_atomic_trans.size();
    double total_abs_dipole_all = 0.0;
    for (int i = 0; i < atomic_trans; ++i) {
        total_abs_dipole_all += std::get<4>(all_atomic_trans[i]);
    }
    for (int i = 0; i < atomic_trans; ++i) {
        double total_dipole = 0.0;
        double total_pr = 0.0;
        double total_ct = 0.0;
        double total_mux = 0.0;
        double total_muy = 0.0;
        double total_muz = 0.0;
        double total_dipole_percent = 0.0;
        std::string istring = fmt::format("{:s} {:s} {:s} {:s}", std::get<0>(all_atomic_trans[i]).c_str(),
            std::get<1>(all_atomic_trans[i]).c_str(), std::get<2>(all_atomic_trans[i]).c_str(),
            std::get<3>(all_atomic_trans[i]).c_str());
        if (std::find(duplicates.begin(), duplicates.end(), istring.c_str()) != duplicates.end()) {
            // outfile->Printf("\n%s is a duplicate!\n", istring.c_str());
        } else {
            for (int j = 0; j < atomic_trans; ++j) {
                std::string jstring = fmt::format("{:s} {:s} {:s} {:s}",
                                                 std::get<0>(all_atomic_trans[j]).c_str(),
                                                 std::get<1>(all_atomic_trans[j]).c_str(),
                                                 std::get<2>(all_atomic_trans[j]).c_str(),
                                                 std::get<3>(all_atomic_trans[j]).c_str());
                if (istring == jstring) {
                    // outfile->Printf("\n%s = %s\n", istring.c_str(), jstring.c_str());
                    // //DEBUG statement, may be useful if anything looks weird with the
                    // promotion numbers.
                    total_dipole += std::get<4>(all_atomic_trans[j]);
                    total_pr += std::get<5>(all_atomic_trans[j]);
                    total_mux += std::get<6>(all_atomic_trans[j]);
                    total_muy += std::get<7>(all_atomic_trans[j]);
                    total_muz += std::get<8>(all_atomic_trans[j]);
                    // total_dipole_percent += all_atomic_trans[j].get<9>();
                }
            }
            duplicates.push_back(istring.c_str());
            total_dipole_percent = (total_dipole / total_abs_dipole_all) * 100.0;
            std::string trans_str = fmt::format("    {:2s} {:1s} ---> {:2s} {:1s}  {:9.2f}%  {:9.2f}  {:9f}  {:9f}  {:9f}  {:9f}",
                std::get<0>(all_atomic_trans[i]).c_str() ,
                std::get<1>(all_atomic_trans[i]).c_str() ,
                std::get<2>(all_atomic_trans[i]).c_str() ,
                std::get<3>(all_atomic_trans[i]).c_str() , total_pr , total_dipole_percent ,
                total_dipole , total_mux , total_muy , total_muz);
            restricted_sums.push_back(std::make_pair(total_dipole_percent, trans_str.c_str()));
        }
    }
    std::sort(restricted_sums.rbegin(), restricted_sums.rend());
    for (auto& kv : restricted_sums) {
        outfile->Printf("\n%s", kv.second.c_str());
    }
    outfile->Printf("\n   "
                    "============================================================"
                    "=======================\n\n");
    outfile->Printf("\n         ==> Mulliken Population Analysis of the "
                    "Transition Dipole Moment <==\n");
    outfile->Printf("\n   "
                    "============================================================"
                    "=====================");
    outfile->Printf("\n    Initial     Final     mu(x)      mu(y)      mu(z)     "
                    "  |mu|     PRnto     CT");
    outfile->Printf("\n   "
                    "------------------------------------------------------------"
                    "---------------------");
    std::sort(sorted_contributions.rbegin(), sorted_contributions.rend());
    for (auto& kv : sorted_contributions) {
        outfile->Printf("\n%s", kv.second.c_str());
    }
    outfile->Printf("\n  "
                    "============================================================"
                    "======================\n\n");

    trDa_iao->zero();
    sorted_contributions.clear();
    trDa_ao->transform(S_half_ao);
    for (int i = 0; i < 3; ++i) {
        dipole_ints[i]->transform(S_inv_half_ao);
    }

    de[0] = trDa_ao->vector_dot(dipole_ints[0]);
    de[1] = trDa_ao->vector_dot(dipole_ints[1]);
    de[2] = trDa_ao->vector_dot(dipole_ints[2]);

    // outfile->Printf("\n  Dipole moments from AO integrals: %.4f %.4f
    // %.4f",de[0],de[1],de[2]);

    // outfile->Printf("\n  Lowdin Population Analysis of the Transition Dipole
    // Moment:\n");
    // outfile->Printf("\n
    // ===============================================================");
    // outfile->Printf("\n   Initial     Final     mu(x)      mu(y)      mu(z)
    // |mu|");
    // outfile->Printf("\n
    // ---------------------------------------------------------------");
    // for (auto& i : keys){
    //    for (auto& f : keys){
    //        trDa_ao_atom->zero();
    //        auto& ifn = atom_am_to_f[i];
    //        auto& ffn = atom_am_to_f[f];
    //        for (int iao : ifn){
    //            for (int fao : ffn){
    //                double value = trDa_ao->get(fao,iao);
    //                trDa_ao_atom->set(fao,iao,value);
    //            }
    //        }
    //        de[0] = trDa_ao_atom->vector_dot(dipole_ints[0]);
    //        de[1] = trDa_ao_atom->vector_dot(dipole_ints[1]);
    //        de[2] = trDa_ao_atom->vector_dot(dipole_ints[2]);

    //        double abs_dipole = std::sqrt(de[0] * de[0] + de[1] * de[1] + de[2]
    //        * de[2]);
    //        if (abs_dipole >= 1.0e-4){
    //            std::string outstr = boost::str(boost::format("  %3d %2s %1s
    //            %3d %2s %1s  %9f  %9f  %9f  %9f") %
    //                        (i.first + 1) %
    //                        molecule_->symbol(i.first).c_str() %
    //                        l_to_symbol[i.second].c_str() %
    //                        (f.first + 1) %
    //                        molecule_->symbol(f.first).c_str() %
    //                        l_to_symbol[f.second].c_str() %
    //                        de[0] % de[1] % de[2] % abs_dipole);
    //            sorted_contributions.push_back(std::make_pair(abs_dipole,outstr));
    //        }
    //    }
    //}

    std::sort(sorted_contributions.rbegin(), sorted_contributions.rend());
    for (auto& kv : sorted_contributions) {
        outfile->Printf("\n%s", kv.second.c_str());
    }
    // outfile->Printf("\n
    // ===============================================================\n\n");
}

// void UCsave_fock()
//{
//    if(not do_excitation){
//        UHF::save_fock();
//    }else{
//        if (initialized_diis_manager_ == false) {
//            diis_manager_ = std::shared_ptr<DIISManager>(new
//            DIISManager(max_diis_vectors_, "HF DIIS vector",
//            DIISManager::LargestError, DIISManager::OnDisk));
//            diis_manager_->set_error_vector_size(2,
//                                                 DIISEntry::Matrix, Fa_.get(),
//                                                 DIISEntry::Matrix,
//                                                 Fb_.get());
//            diis_manager_->set_vector_size(2,
//                                           DIISEntry::Matrix, Fa_.get(),
//                                           DIISEntry::Matrix, Fb_.get());
//            initialized_diis_manager_ = true;
//        }

//        SharedMatrix errveca(moFeffa_);
//        errveca->zero_diagonal();
//        errveca->back_transform(Ca_);
//        SharedMatrix errvecb(moFeffb_);
//        errvecb->zero_diagonal();
//        errvecb->back_transform(Cb_);
//        diis_manager_->add_entry(4, errveca.get(), errvecb.get(), Fa_.get(),
//        Fb_.get());
//    }
//}

void UOCDFT::compute_orbital_gradient(bool save_fock) {
    if (not do_excitation) {
        UHF::compute_orbital_gradient(save_fock);
    } else {
        SharedMatrix gradient_a = form_FDSmSDF(Fa_, Da_);
        SharedMatrix gradient_b = form_FDSmSDF(Fb_, Db_);
        Drms_ = 0.5 * (gradient_a->rms() + gradient_b->rms());

        //        if(save_fock){
        //            if (initialized_diis_manager_ == false) {
        //                diis_manager_ = std::shared_ptr<DIISManager>(new
        //                DIISManager(max_diis_vectors_, "HF DIIS vector",
        //                DIISManager::LargestError, DIISManager::OnDisk));
        //                diis_manager_->set_error_vector_size(2,
        //                                                     DIISEntry::Matrix,
        //                                                     gradient_a.get(),
        //                                                     DIISEntry::Matrix,
        //                                                     gradient_b.get());
        //                diis_manager_->set_vector_size(2,
        //                                               DIISEntry::Matrix,
        //                                               Fa_.get(),
        //                                               DIISEntry::Matrix,
        //                                               Fb_.get());
        //                initialized_diis_manager_ = true;
        //            }

        //            diis_manager_->add_entry(4, gradient_a.get(),
        //            gradient_b.get(), Fa_.get(), Fb_.get());
        //        }

        if (save_fock) {
            if (initialized_diis_manager_ == false) {
                diis_manager_ = std::shared_ptr<DIISManager>(
                    new DIISManager(max_diis_vectors_, "HF DIIS vector", DIISManager::LargestError,
                                    DIISManager::OnDisk));
                diis_manager_->set_error_vector_size(2, DIISEntry::Matrix, Fa_.get(),
                                                     DIISEntry::Matrix, Fb_.get());
                diis_manager_->set_vector_size(2, DIISEntry::Matrix, Fa_.get(), DIISEntry::Matrix,
                                               Fb_.get());
                initialized_diis_manager_ = true;
            }
            SharedMatrix errveca(moFeffa_);
            errveca->zero_diagonal();
            errveca->back_transform(Ca_);
            SharedMatrix errvecb(moFeffb_);
            errvecb->zero_diagonal();
            errvecb->back_transform(Cb_);
            diis_manager_->add_entry(4, errveca.get(), errvecb.get(), Fa_.get(), Fb_.get());
        }
    }
}

void UOCDFT::spin_adapt_mixed_excitation() {
    CharacterTable ct = molecule_->point_group()->char_table();
    SharedDeterminant D1 = SharedDeterminant(new Determinant(E_, Ca_, Cb_, nalphapi_, nbetapi_));
    SharedDeterminant D2 = SharedDeterminant(new Determinant(E_, Cb_, Ca_, nbetapi_, nalphapi_));
    std::pair<double, double> M12 = matrix_element(D1, D2);
    double alpha = options_.get_double("ALPHA_CI");
    double S12 = alpha * M12.first;
    double H12 = alpha * M12.second;
    double triplet_energy = (E_ - H12) / (1.0 - S12);
    double singlet_energy = (E_ + H12) / (1.0 + S12);
    double triplet_exc_energy = (E_ - H12) / (1.0 - S12) - ground_state_energy;
    double singlet_exc_energy = (E_ + H12) / (1.0 + S12) - ground_state_energy;
    outfile->Printf("\n\n  H12  %d-%s = %15.9f\n",
                    state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
                    ct.gamma(excited_state_symmetry_).symbol(), H12);
    outfile->Printf("  S12  %d-%s = %15.9f\n\n",
                    state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
                    ct.gamma(excited_state_symmetry_).symbol(), S12);

    outfile->Printf("\n  Triplet state energy (CI) %d-%s %20.9f Eh \n",
                    state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
                    ct.gamma(excited_state_symmetry_).symbol(), triplet_energy);

    outfile->Printf("\n  Singlet state energy (CI) %d-%s %20.9f Eh \n",
                    state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
                    ct.gamma(excited_state_symmetry_).symbol(), singlet_energy);

    outfile->Printf("\n  Excited triplet state %d-%s : excitation energy (CI) = %9.6f Eh = "
                    "%8.4f eV = %9.1f cm**-1 \n",
                    state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
                    ct.gamma(excited_state_symmetry_).symbol(), triplet_exc_energy,
                    triplet_exc_energy * pc_hartree2ev,
                    triplet_exc_energy * pc_hartree2wavenumbers);

    outfile->Printf("  Excited singlet state %d-%s : excitation energy (CI) = %9.6f Eh = "
                    "%8.4f eV = %9.1f cm**-1 \n",
                    state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
                    ct.gamma(excited_state_symmetry_).symbol(), singlet_exc_energy,
                    singlet_exc_energy * pc_hartree2ev,
                    singlet_exc_energy * pc_hartree2wavenumbers);
}

double UOCDFT::compute_triplet_correction() {
    // I. Form the corresponding alpha and beta orbitals, this gives us a way
    // to identify the alpha and beta paired orbitals (singular value ~ 1)
    // and distinguish them from singly occupied MOs (singular value ~ 0).

    // Form <phi_b|S|phi_a>
    TempMatrix->gemm(false, false, 1.0, S_, Ca_, 0.0);
    TempMatrix2->gemm(true, false, 1.0, Cb_, TempMatrix, 0.0);

    // Scale it down to the occupied blocks only
    SharedMatrix Sba = SharedMatrix(new Matrix("Sba", nbetapi_, nalphapi_));
    nalphapi_.print();
    nbetapi_.print();
    for (int h = 0; h < nirrep_; ++h) {
        int nmo = nmopi_[h];
        int naocc = nalphapi_[h];
        int nbocc = nbetapi_[h];
        double** Sba_h = Sba->pointer(h);
        double** S_h = TempMatrix2->pointer(h);
        for (int i = 0; i < nbocc; ++i) {
            for (int j = 0; j < naocc; ++j) {
                Sba_h[i][j] = S_h[i][j];
            }
        }
    }

    // SVD <phi_b|S|phi_a>
    std::tuple<SharedMatrix, SharedVector, SharedMatrix> UsV = Sba->svd_temps();
    SharedMatrix U = std::get<0>(UsV);
    SharedVector sigma = std::get<1>(UsV);
    SharedMatrix V = std::get<2>(UsV);
    Sba->svd(U, sigma, V);
    sigma->print();
    U->print();
    V->print();

    // II. Transform the occupied alpha and beta orbitals to the new
    // representation
    // and compute the energy of the high-spin state.  The singly occupied MOs can
    // be used to guide the selection of the occupation numbers

    // Transform Ca_ with V (need to transpose V since svd returns V^T)
    TempMatrix->identity();
    for (int h = 0; h < nirrep_; ++h) {
        int rows = V->rowdim(h);
        int cols = V->coldim(h);
        double** V_h = V->pointer(h);
        double** T_h = TempMatrix->pointer(h);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                T_h[i][j] = V_h[i][j];
            }
        }
    }
    TempMatrix2->copy(Ca_);
    Ca_->gemm(false, true, 1.0, TempMatrix2, TempMatrix, 0.0);

    // Transform Cb_ with U
    TempMatrix->identity();
    for (int h = 0; h < nirrep_; ++h) {
        int rows = U->rowdim(h);
        int cols = U->coldim(h);
        double** U_h = U->pointer(h);
        double** T_h = TempMatrix->pointer(h);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                T_h[i][j] = U_h[i][j];
            }
        }
    }
    TempMatrix2->copy(Cb_);
    Cb_->gemm(false, false, 1.0, TempMatrix2, TempMatrix, 0.0);

    std::vector<std::pair<int, int>> noncoincidences;
    double noncoincidence_threshold = 1.0e-5;
    double Stilde = 1.0;
    // Compute the number of noncoincidences
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < sigma->dim(h); ++p) {
            if (std::fabs(sigma->get(h, p)) >= noncoincidence_threshold) {
                Stilde *= sigma->get(h, p);
            } else {
                noncoincidences.push_back(std::make_pair(h, p));
            }
        }
    }
    int num_noncoincidences = static_cast<int>(noncoincidences.size());
    for (int k = 0; k < num_noncoincidences; ++k) {
        int i_h = noncoincidences[k].first;
        int i_mo = noncoincidences[k].second;
        outfile->Printf("  Found a noncoincidence: irrep %d mo %d\n", i_h, i_mo);
    }
    outfile->Printf("  Stilde = %.6f\n", Stilde);
    double overlap = 1.0;
    if (num_noncoincidences == 0) {
        throw FeatureNotImplemented("CKS", "Overlap in the case of zero noncoincidences", __FILE__,
                                    __LINE__);
    }

    int i_h = noncoincidences[0].first;
    int i_mo = noncoincidences[0].second;
    Dimension nnonc(1, "");
    nnonc[0] = 1;
    SharedMatrix Cnca(new Matrix("Cnca", nsopi_, nnonc));
    Cnca->set_column(0, 0, Ca_->get_column(i_h, i_mo));
    SharedMatrix Cncb(new Matrix("Cncb", nsopi_, nnonc));
    Cncb->set_column(0, 0, Cb_->get_column(i_h, i_mo));
    Cnca->print();
    Cncb->print();

    //    if (options_.get_str("SCF_TYPE") == "DF") {
    //        jk = JK::build_JK(get_basisset("ORBITAL"), get_basisset("DF_BASIS_SCF"),
    //        options_);
    //    } else {
    //        jk = JK::build_JK(get_basisset("ORBITAL"), BasisSet::zero_ao_basis_set(),
    //        options_);
    //    }
    //    //    std::shared_ptr<JK> jk = JK::build_JK(basisset_, options_);
    //    jk->initialize();
    std::vector<SharedMatrix>& C_left = jk_->C_left();
    C_left.clear();
    C_left.push_back(Cncb);
    std::vector<SharedMatrix>& C_right = jk_->C_right();
    C_right.clear();
    C_right.push_back(Cnca);
    jk_->compute();
    SharedMatrix Jnew = jk_->J()[0];

    double coupling = 0.0;
    for (int m = 0; m < nsopi_[0]; ++m) {
        for (int n = 0; n < nsopi_[0]; ++n) {
            double Dvalue = Cncb->get(0, m) * Cnca->get(0, n);
            double Jvalue = Jnew->get(m, n);
            coupling += Dvalue * Jvalue;
        }
    }
    outfile->Printf("  Matrix element from libfock = %20.12f\n", coupling);

    coupling *= Stilde * Stilde;

    //    jk_->finalize();

    int maxi4 = INDEX4(nsopi_[0] + 1, nsopi_[0] + 1, nsopi_[0] + 1, nsopi_[0] + 1) + nsopi_[0] + 1;
    double* integrals = new double[maxi4];
    for (int l = 0; l < maxi4; ++l) {
        integrals[l] = 0.0;
    }

    IWL* iwl = new IWL(psio_.get(), PSIF_SO_TEI, integral_threshold_, 1, 1);
    Label* lblptr = iwl->labels();
    Value* valptr = iwl->values();
    int labelIndex, pabs, qabs, rabs, sabs, prel, qrel, rrel, srel, psym, qsym, rsym, ssym;
    double value;
    bool lastBuffer;
    do {
        lastBuffer = iwl->last_buffer();
        for (int index = 0; index < iwl->buffer_count(); ++index) {
            labelIndex = 4 * index;
            pabs = abs((int)lblptr[labelIndex++]);
            qabs = (int)lblptr[labelIndex++];
            rabs = (int)lblptr[labelIndex++];
            sabs = (int)lblptr[labelIndex++];
            prel = so2index_[pabs];
            qrel = so2index_[qabs];
            rrel = so2index_[rabs];
            srel = so2index_[sabs];
            psym = so2symblk_[pabs];
            qsym = so2symblk_[qabs];
            rsym = so2symblk_[rabs];
            ssym = so2symblk_[sabs];
            value = (double)valptr[index];
            integrals[INDEX4(prel, qrel, rrel, srel)] = value;
        } /* end loop through current buffer */
        if (!lastBuffer)
            iwl->fetch();
    } while (!lastBuffer);
    iwl->set_keep_flag(1);
    delete iwl;
    double c2 = 0.0;
    double* Ci = Cncb->get_pointer();
    double* Cj = Cnca->get_pointer();
    double* Ck = Cnca->get_pointer();
    double* Cl = Cncb->get_pointer();
    for (int i = 0; i < nsopi_[0]; ++i) {
        for (int j = 0; j < nsopi_[0]; ++j) {
            for (int k = 0; k < nsopi_[0]; ++k) {
                for (int l = 0; l < nsopi_[0]; ++l) {
                    c2 += integrals[INDEX4(i, j, k, l)] * Ci[i] * Cj[j] * Ck[k] * Cl[l];
                }
            }
        }
    }
    delete[] integrals;
    outfile->Printf("  Matrix element from functor = %20.12f\n", c2);

    return coupling;
}

double UOCDFT::compute_S_plus_triplet_correction() {
    outfile->Printf("\n  ==> Spin-adaptation correction using S+ <==\n");
    CharacterTable ct = molecule_->point_group()->char_table();
    // A. Form the corresponding virtual alpha and occupied beta orbitals
    SharedMatrix Sba = SharedMatrix(new Matrix("Sba", nbetapi_, nmopi_ - nalphapi_));

    // Form <phi_b|S|phi_a>
    TempMatrix->gemm(false, false, 1.0, S_, Ca_, 0.0);
    TempMatrix2->gemm(true, false, 1.0, Cb_, TempMatrix, 0.0);

    // Grab the virtual alpha and occupied beta blocks
    for (int h = 0; h < nirrep_; ++h) {
        int nmo = nmopi_[h];
        int naocc = nalphapi_[h];
        int navir = nmo - naocc;
        int nbocc = nbetapi_[h];
        double** Sba_h = Sba->pointer(h);
        double** S_h = TempMatrix2->pointer(h);
        for (int i = 0; i < nbocc; ++i) {
            for (int a = 0; a < navir; ++a) {
                Sba_h[i][a] = S_h[i][a + naocc];
            }
        }
    }

    // SVD <phi_b|S|phi_a>
    std::tuple<SharedMatrix, SharedVector, SharedMatrix> UsV = Sba->svd_a_temps();
    SharedMatrix U = std::get<0>(UsV);
    SharedVector sigma = std::get<1>(UsV);
    SharedMatrix V = std::get<2>(UsV);
    Sba->svd_a(U, sigma, V);

    // B. Find the corresponding alpha and beta orbitals
    std::vector<std::tuple<double, int, int>> sorted_pair; // (singular value,irrep,mo in irrep)
    for (int h = 0; h < nirrep_; ++h) {
        int npairs = sigma->dim(h);
        for (int p = 0; p < npairs; ++p) {
            sorted_pair.push_back(
                std::make_tuple(sigma->get(h, p), h, p)); // N.B. shifted wrt to full indexing
        }
    }
    std::sort(sorted_pair.begin(), sorted_pair.end(), std::greater<std::tuple<double, int, int>>());

    // Print some useful information
    int npairs = std::min(10, static_cast<int>(sorted_pair.size()));
    // outfile->Printf("  Most important corresponding occupied/virtual
    // orbitals:\n\n");
    // outfile->Printf("  Pair  Irrep  MO  <phi_b|phi_a>\n");
    // for (int p = 0; p < npairs; ++p){
    //    outfile->Printf("\n %f \n", npairs);
    //    outfile->Printf("    %2d     %3s %4d
    //    %9.6f\n",p,ct.gamma(sorted_pair[p].get<1>()).symbol(),sorted_pair[p].get<2>(),sorted_pair[p].get<0>());
    //}

    // C. Transform the alpha virtual and beta occupied orbitals to the new
    // representation
    // Transform Ca_ with V (need to transpose V since svd returns V^T)
    TempMatrix->identity();
    for (int h = 0; h < nirrep_; ++h) {
        int rows = V->rowdim(h);
        int cols = V->coldim(h);
        int naocc = nalphapi_[h];
        double** V_h = V->pointer(h);
        double** T_h = TempMatrix->pointer(h);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                T_h[i + naocc][j + naocc] = V_h[i][j]; // Offset by the number of occupied MOs
            }
        }
    }
    TempMatrix2->copy(Ca_);
    Ca_->gemm(false, true, 1.0, TempMatrix2, TempMatrix, 0.0);

    // Transform Cb_ with U (reversing the order so that the corresponding orbital
    // is the first to be excluded)
    TempMatrix->identity();
    for (int h = 0; h < nirrep_; ++h) {
        int rows = U->rowdim(h);
        int cols = U->coldim(h);
        double** U_h = U->pointer(h);
        double** T_h = TempMatrix->pointer(h);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                T_h[i][j] = U_h[i][rows - j - 1]; // invert the order
            }
        }
    }
    TempMatrix2->copy(Cb_);
    Cb_->gemm(false, false, 1.0, TempMatrix2, TempMatrix, 0.0);

    outfile->Printf("\n  Original occupation numbers:\n");
    outfile->Printf("\tNA   [ ");
    for (int h = 0; h < nirrep_ - 1; ++h)
        outfile->Printf(" %4d,", nalphapi_[h]);
    outfile->Printf(" %4d ]\n", nalphapi_[nirrep_ - 1]);
    outfile->Printf("\tNB   [ ");
    for (int h = 0; h < nirrep_ - 1; ++h)
        outfile->Printf(" %4d,", nbetapi_[h]);
    outfile->Printf(" %4d ]\n", nbetapi_[nirrep_ - 1]);
    int mo_h = std::get<1>(sorted_pair[0]);
    outfile->Printf("\n  Final occupation numbers:\n");
    // Update the occupation numbers
    nalphapi_[mo_h] += 1;
    nbetapi_[mo_h] -= 1;
    nalpha_ = 0;
    nbeta_ = 0;
    for (int h = 0; h < nirrep_; ++h) {
        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
        doccpi_[h] = std::min(nalphapi_[h], nbetapi_[h]);
        nalpha_ += nalphapi_[h];
        nbeta_ += nalphapi_[h];
    }
    outfile->Printf("\tNA   [ ");
    for (int h = 0; h < nirrep_ - 1; ++h)
        outfile->Printf(" %4d,", nalphapi_[h]);
    outfile->Printf(" %4d ]\n", nalphapi_[nirrep_ - 1]);
    outfile->Printf("\tNB   [ ");
    for (int h = 0; h < nirrep_ - 1; ++h)
        outfile->Printf(" %4d,", nbetapi_[h]);
    outfile->Printf(" %4d ]\n", nbetapi_[nirrep_ - 1]);

    // Compute the density matrices with the new occupation

    form_D();
    form_G();
    form_F();
    // Compute the triplet energy from the density matrices
    double triplet_energy = compute_E();
    double triplet_exc_energy = triplet_energy - ground_state_energy;

    outfile->Printf("\n  Triplet state energy (S+) %d-%s %20.9f Eh \n",
                    state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
                    ct.gamma(excited_state_symmetry_).symbol(), triplet_energy);

    outfile->Printf("\n  Singlet state energy (S+) %d-%s %20.9f Eh \n",
                    state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
                    ct.gamma(excited_state_symmetry_).symbol(), 2.0 * E_ - triplet_energy);
    std::map<std::string, double>& quad = potential_->quadrature_values();
    outfile->Printf("\n  Excited triplet state %d-%s : excitation energy (S+) = %9.6f Eh = "
                    "%8.4f eV = %9.1f cm**-1 \n",
                    state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
                    ct.gamma(excited_state_symmetry_).symbol(), triplet_exc_energy,
                    triplet_exc_energy * pc_hartree2ev,
                    triplet_exc_energy * pc_hartree2wavenumbers);

    double singlet_exc_energy = 2.0 * E_ - triplet_energy - ground_state_energy;
    outfile->Printf("  Excited singlet state %d-%s : excitation energy (S+) = %9.6f Eh = "
                    "%8.4f eV = %9.1f cm**-1 \n",
                    state_ + (ground_state_symmetry_ == excited_state_symmetry_ ? 1 : 0),
                    ct.gamma(excited_state_symmetry_).symbol(), singlet_exc_energy,
                    singlet_exc_energy * pc_hartree2ev,
                    singlet_exc_energy * pc_hartree2wavenumbers);

    Process::environment.globals["OCDFT TRIPLET ENERGY"] = triplet_energy;
    Process::environment.globals["OCDFT SINGLET ENERGY"] = singlet_exc_energy + ground_state_energy;

    Process::environment.globals["OCDFT TRIPLET ENERGY STATE " + std::to_string(state_)] =
        triplet_energy;
    Process::environment.globals["OCDFT SINGLET ENERGY STATE " + std::to_string(state_)] =
        singlet_exc_energy + ground_state_energy;

    double xes_energy;
    if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
        xes_energy = (dets[state_]->energy() - dets[1]->energy());
        outfile->Printf("\n first state energy: %f\n", (dets[1]->energy()));
        outfile->Printf("\n XES Energy = %f\n",
                        std::abs(((2.0 * xes_energy - triplet_energy) * pc_hartree2ev)));
    }

    // Save the excitation energy
    if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
        singlet_exc_energy_s_plus_ = std::abs(xes_energy);
    } else {
        singlet_exc_energy_s_plus_ = singlet_exc_energy;
    }
    triplet_exc_energy_s_plus = triplet_energy - ground_state_energy;

    double dx = Process::environment.globals["OCDFT TRANSITION DIPOLE X"] / pc_dipmom_au2debye;
    double dy = Process::environment.globals["OCDFT TRANSITION DIPOLE Y"] / pc_dipmom_au2debye;
    double dz = Process::environment.globals["OCDFT TRANSITION DIPOLE Z"] / pc_dipmom_au2debye;

    oscillator_strength_s_plus_ =
        2. / 3. * singlet_exc_energy_s_plus_ * (dx * dx + dy * dy + dz * dz);
    oscillator_strength_s_plus_x_ = 2. / 3. * singlet_exc_energy_s_plus_ * (dx * dx);
    oscillator_strength_s_plus_y_ = 2. / 3. * singlet_exc_energy_s_plus_ * (dy * dy);
    oscillator_strength_s_plus_z_ = 2. / 3. * singlet_exc_energy_s_plus_ * (dz * dz);
    outfile->Printf("\n  Transition Dipole Moment = (%f,%f,%f) \n ", dx, dy, dz);
    ;
    compute_spin_contamination();
    outfile->Printf("\n");

    // Revert to the mixed state occupation numbers
    nalphapi_[mo_h] -= 1;
    nbetapi_[mo_h] += 1;
    nalpha_ = 0;
    nbeta_ = 0;
    for (int h = 0; h < nirrep_; ++h) {
        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
        doccpi_[h] = std::min(nalphapi_[h], nbetapi_[h]);
        nalpha_ += nalphapi_[h];
        nbeta_ += nalphapi_[h];
    }
    Ca_->copy(dets.back()->Ca());
    Cb_->copy(dets.back()->Cb());
    return singlet_exc_energy;
}

void UOCDFT::cis_excitation_energy() {

    //    CharacterTable ct = molecule_->point_group()->char_table();

    //    int symmetry = 0;
    //    SharedMatrix ra = SharedMatrix(new Matrix("r
    //    Amplitudes",nmopi_,nmopi_,symmetry));
    //    SharedMatrix rb = SharedMatrix(new Matrix("r
    //    Amplitudes",nmopi_,nmopi_,symmetry));
    //    SharedMatrix genDa = factory_->create_shared_matrix("genDa");
    //    SharedMatrix genDb = factory_->create_shared_matrix("genDb");
    //    // Determine the hole/particle pair to follow
    //    // Compute the symmetry adapted hole/particle pairs
    //    std::vector<std::tuple<double,int,int,double,int,int,double> >
    //    sorted_hp_pairs;
    //    for (int h = 0; h < nirrep_; ++h){
    //        int h_i = h;
    //        int h_a = h ^ symmetry;
    //        int nocc_i = nalphapi_[h_i];
    //        int nocc_a = nalphapi_[h_a];
    //        int nvir_a = nmopi_[h_a] - nalphapi_[h_a];
    //        for (int i = 0; i < nocc_i; ++i){
    //            for (int a = 0; a < nvir_a; ++a){
    //                double e_i = epsilon_a_->get(h_i,i);
    //                double e_a = epsilon_a_->get(h_a,a + nocc_a);
    //                double delta_ai = e_a - e_i;
    //                sorted_hp_pairs.push_back(std::make_tuple(delta_ai,h_i,i,e_i,h_a,a
    //                + nocc_a,e_a));
    //            }
    //        }
    //    }

    //    std::sort(sorted_hp_pairs.begin(),sorted_hp_pairs.end());
    ////    if(iteration_ == 0){
    //    outfile->Printf( "\n  Ground state symmetry:
    //    %s\n",ct.gamma(ground_state_symmetry_).symbol());
    //    outfile->Printf( "  Excited state symmetry:
    //    %s\n",ct.gamma(excited_state_symmetry_).symbol());
    //    outfile->Printf( "\n  Lowest energy excitations:\n");
    //    outfile->Printf( "  --------------------------------------\n");
    //    outfile->Printf( "    N   Occupied     Virtual     E(eV)  \n");
    //    outfile->Printf( "  --------------------------------------\n");
    //    int maxstates = std::min(10,static_cast<int>(sorted_hp_pairs.size()));
    //    for (int n = 0; n < maxstates; ++n){
    //        double energy_hp = sorted_hp_pairs[n].get<6>() -
    //        sorted_hp_pairs[n].get<3>();
    //        outfile->Printf("   %2d:  %4d%-3s  -> %4d%-3s   %9.3f\n",n + 1,
    //                sorted_hp_pairs[n].get<2>() + 1,
    //                ct.gamma(sorted_hp_pairs[n].get<1>()).symbol(),
    //                sorted_hp_pairs[n].get<5>() + 1,
    //                ct.gamma(sorted_hp_pairs[n].get<4>()).symbol(),
    //                energy_hp * _hartree2ev);
    //    }
    //    outfile->Printf( "  --------------------------------------\n");

    //    int select_pair = 0;
    //    aholes_h = sorted_hp_pairs[select_pair].get<1>();
    //    aholes_mo = sorted_hp_pairs[select_pair].get<2>();
    //    aparts_h = sorted_hp_pairs[select_pair].get<4>();
    //    aparts_mo = sorted_hp_pairs[select_pair].get<5>();
    //    ra->set(aholes_h,aholes_mo,aparts_mo,1.0 / std::sqrt(2.0));
    //    rb->set(aholes_h,aholes_mo,aparts_mo,-1.0 / std::sqrt(2.0));

    //    // Compute the density matrix
    //    genDa->zero();
    //    genDb->zero();

    //    // C1 symmetry
    //    {
    //    int naocc = nalphapi_[0];
    //    int navir = nmopi_[0] - nalphapi_[0];
    //    int nmo = nmopi_[0];
    //    double** ra_h = ra->pointer(0);
    //    for (int i = 0; i < naocc; ++i){
    //        for (int j = 0; j < naocc; ++j){
    //            double da = (i == j ? 1.0 : 0.0);
    //            for (int c = naocc; c < nmo; ++c){
    //                da -= ra_h[i][c] * ra_h[j][c];
    //            }
    //            genDa->set(0,i,j,da);
    //        }
    //    }
    //    for (int a = naocc; a < nmo; ++a){
    //        for (int b = naocc; b < nmo; ++b){
    //            double da = 0.0;
    //            for (int k = 0; k < naocc; ++k){
    //                da += ra_h[k][a] * ra_h[k][b];
    //            }
    //            genDa->set(0,a,b,da);
    //        }
    //    }

    //    }

    //    {
    //    int nbocc = nbetapi_[0];
    //    int nbvir = nmopi_[0] - nbetapi_[0];
    //    int nmo = nmopi_[0];
    //    double** rb_h = rb->pointer(0);
    //    for (int i = 0; i < nbocc; ++i){
    //        for (int j = 0; j < nbocc; ++j){
    //            double db = (i == j ? 1.0 : 0.0);
    //            for (int c = nbocc; c < nmo; ++c){
    //                db -= rb_h[i][c] * rb_h[j][c];
    //            }
    //            genDb->set(0,i,j,db);
    //        }
    //    }
    //    for (int a = nbocc; a < nmo; ++a){
    //        for (int b = nbocc; b < nmo; ++b){
    //            double db = 0.0;
    //            for (int k = 0; k < nbocc; ++k){
    //                db += rb_h[k][a] * rb_h[k][b];
    //            }
    //            genDb->set(0,a,b,db);
    //        }
    //    }
    //    }

    ////    // Off-diagonal terms
    ////    for (int h = 0; h < nirrep_; ++h){
    ////        int g = h ^ symmetry;
    ////        int nmo_h = nmopi_[h];
    ////        int nmo_g = nmopi_[h];
    ////        double** ra_h = ra->pointer(h);
    ////        double** ra_g = ra->pointer(g);
    ////        int navir_g = nmopi_[g] - nalphapi_[g];
    ////        for (int p = 0; p < nmo; ++p){
    ////            for (int q = 0; q < nmo; ++q){
    ////                double da = 0.0;
    ////                for (int c = navir_g; c < nmo_; ++c){
    ////                    da -= ra_h[p][c] * ra_h[q][c];
    ////                }
    ////            }
    ////        }

    ////        int navir = nmopi_[h] - nalphapi_[h];
    ////        int h_a = h ^ symmetry;
    ////        double** ra_h = ra->pointer(h);
    ////        int nvir_ac = nmopi_[h_a] - nalphapi_[h_a];
    ////        int nocc_ak = nalphapi_[h_a];
    ////        for (int i = 0; i < naocc; ++i){
    ////            for (int j = 0; j < naocc; ++j){
    ////                double da = (i == j ? 1.0 : 0.0);
    ////                for (int c = 0; c < nvir_ac; ++c){
    ////                        da -= ra_h[i][c] * ra_h[j][c];
    ////                }
    ////                genDa->set(h,i,j,da);
    ////            }
    ////        }
    ////        double** ra_h_a = ra->pointer(h_a);
    ////        for (int a = 0; a < navir; ++a){
    ////            for (int b = 0; b < navir; ++b){
    ////                double da = 0.0;
    ////                for (int k = 0; k < nocc_ak; ++k){
    ////                        da += ra_h_a[k][a] * ra_h_a[k][b];
    ////                }
    ////                genDa->set(h,a + naocc,b + naocc,da);
    ////            }
    ////        }

    ////        int nbocc = nbetapi_[h];
    ////        int nbvir = nmopi_[h] - nbetapi_[h];
    ////        double** rb_h = rb->pointer(h);
    ////        int nvir_bc = nmopi_[h_a] - nbetapi_[h_a];
    ////        int nocc_bk = nbetapi_[h_a];
    ////        for (int i = 0; i < nbocc; ++i){
    ////            for (int j = 0; j < nbocc; ++j){
    ////                double db = (i == j ? 1.0 : 0.0);
    ////                for (int c = 0; c < nvir_bc; ++c){
    ////                        db -= rb_h[i][c] * rb_h[j][c];
    ////                }
    ////                genDb->set(h,i,j,db);
    ////            }
    ////        }
    ////        double** rb_h_a = rb->pointer(h_a);
    ////        for (int a = 0; a < nbvir; ++a){
    ////            for (int b = 0; b < nbvir; ++b){
    ////                double db = 0.0;
    ////                for (int k = 0; k < nocc_ak; ++k){
    ////                        db += rb_h_a[k][a] * rb_h_a[k][b];
    ////                }
    ////                genDb->set(h,a + nbocc,b + nbocc,db);
    ////            }
    ////        }
    ////    }

    ////    genDa->zero();
    ////    genDb->zero();
    ////    for (int n = 0; n < 5; ++n){

    ////    }

    //    Da_->back_transform(genDa,Ca_);
    //    Db_->back_transform(genDb,Cb_);
    //    Dt_->copy(Da_);
    //    Dt_->add(Db_);
    ////    nalphapi_[7] = 0;
    ////    nalphapi_[2] = 1;

    ////    form_D();
    ////    for (int h = 0; h < nirrep_; ++h) {
    ////        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
    ////        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
    ////    }
    //    form_G();
    //    form_F();

    //    // Compute the energy
    //    double cis_energy = compute_E() - E_;
    //    outfile->Printf("\n  CIS excited state = %9.6f Eh = %8.4f eV = %9.1f
    //    cm**-1 \n",
    //            cis_energy,cis_energy * _hartree2ev, cis_energy *
    //            _hartree2wavenumbers);
}

// void UCform_D_cis()
//{
//    for (int h = 0; h < nirrep_; ++h) {
//        int nso = nsopi_[h];
//        int nmo = nmopi_[h];
//        int na = nalphapi_[h];
//        int nb = nbetapi_[h];

//        if (nso == 0 || nmo == 0) continue;

//        double** Ca = Ca_->pointer(h);
//        double** Cb = Cb_->pointer(h);
//        double** Da = Da_->pointer(h);
//        double** Db = Db_->pointer(h);

//        if (na == 0)
//            ::memset(static_cast<void*>(Da[0]), '\0', sizeof(double)*nso*nso);
//        if (nb == 0)
//            ::memset(static_cast<void*>(Db[0]), '\0', sizeof(double)*nso*nso);

//        C_DGEMM('N','T',nso,nso,na,1.0,Ca[0],nmo,Ca[0],nmo,0.0,Da[0],nso);
//        C_DGEMM('N','T',nso,nso,nb,1.0,Cb[0],nmo,Cb[0],nmo,0.0,Db[0],nso);

//    }

//    Dt_->copy(Da_);
//    Dt_->add(Db_);

//    if (debug_) {
//        outfile->Printf( "in UHF::form_D:\n");
//        Da_->print();
//        Db_->print();
//    }
//}

void UOCDFT::orthogonality_check(SharedMatrix C, SharedMatrix S) {
    SharedMatrix CSC(S->clone());
    CSC->transform(C);
    double diag = 0.0;
    double off_diag = 0.0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nsopi_[h]; ++i) {
            for (int j = 0; j < nsopi_[h]; ++j)
                if (i == j) {
                    diag += std::fabs(CSC->get(h, i, j));
                } else {
                    off_diag += std::fabs(CSC->get(h, i, j));
                }
        }
    }
    if (off_diag > 1e-4) {
        outfile->Printf("\n***** WARNING!: ORBITALS HAVE LOST ORTHOGONALITY ******");
        outfile->Printf("\n***** Sum of the Off-Diagonal Elements of S: %f  "
                        "*******************",
                        off_diag);
    }
}
}
} // Namespaces
