//#include <algorithm>
#include <numeric>
//#include <boost/format.hpp>
//#include <boost/tuple/tuple_comparison.hpp>

#include "psi4/libpsi4util/PsiOutStream.h"
#include <psi4/libmints/basisset.h>
#include <psi4/libmints/vector.h>
#include <psi4/libmints/molecule.h>
//#include <psi4/libmints/local.h>
//#include <psi4/libmints/factory.h>
//#include <psi4/libmints/pointgrp.h>
//#include <psi4/libmints/petitelist.h>
//#include <psi4/liboptions/liboptions.h>
//#include "psi4/libfunctional/superfunctional.h"
//#include <psi4/physconst.h>
//#include <psi4/psifiles.h>
//#include <psi4/libmints/oeprop.h>
//#include <psi4/libmints/onebody.h>
//#include <psi4/libmints/integral.h>
////#include <libfock/apps.h>
//#include <psi4/libfock/jk.h>
//#include <psi4/libfock/v.h>

#include <psi4/libcubeprop/cubeprop.h>
//#include <psi4/libdisp/dispersion.h>
////#include <liboptions/liboptions.h>
//#include <psi4/libciomr/libciomr.h>
//#include <psi4/libiwl/iwl.hpp>
//#include <psi4/libqt/qt.h>
//#include <psi4/libscf_solver/hf.h>
//#include <psi4/libdiis/diismanager.h>
//#include "psi4/psi4-dec.h"
//#include "psi4/libpsi4util/libpsi4util.h"
//#include "psi4/libpsi4util/process.h"

//#include "aosubspace.h"
#include "helpers.h"
#include "iao_builder.h"
#include "ocdft.h"

using namespace psi;

namespace psi {
namespace scf {

void UOCDFT::livvo_analysis(SharedMatrix refC) {
    outfile->Printf("\n\n  ==> Analysis of OCDFT orbitals <==\n\n");

    // Get the virtual block of refC
    int nocc = gs_nalphapi_[0];
    int nvir = gs_navirpi_[0];
    int nbf = basisset()->nbf();
    Dimension dim_zero(1);
    Dimension dim_bf(1);
    Dimension dim_occ(1);
    Dimension dim_vir(1);
    dim_bf[0] = nbf;
    dim_occ[0] = nocc;
    dim_vir[0] = nvir;
    SharedMatrix Cvir = std::make_shared<Matrix>("Cvir", dim_bf, dim_vir);
    copy_block(refC, 1.0, Cvir, 0.0, dim_bf, dim_vir, dim_zero, dim_occ);

    // Build the IAOs
    auto aios = build_aios(refC);

    // Build the LIVVOs
    SharedMatrix Clivvo = generate_livvos(aios, Cvir);

    // Assign the character of the IAOs
    auto IAO_character = find_iao_character(aios.first);

    // Assign the LIVVOs
    find_particle_orbital_character(Cp_, Clivvo, aios.first, IAO_character);
}

std::pair<SharedMatrix, std::shared_ptr<IAOBuilder>> UOCDFT::build_aios(SharedMatrix refC) {
    // Build the IAOs using the MINAO_BASIS basis
    std::shared_ptr<BasisSet> minao = get_basisset("MINAO_BASIS");
    std::shared_ptr<IAOBuilder> iao = IAOBuilder::build(wfn_->basisset(), minao, refC, options_);
    std::map<std::string, SharedMatrix> iao_info = iao->build_iaos();
    SharedMatrix C_IAO(iao_info["A"]->clone());

    // Code to plot IAOs (checked)
    CubeProperties cube = CubeProperties(wfn_);
    int niao = C_IAO->ncol();
    std::vector<int> indices(niao);
    std::vector<std::string> labels(niao);
    std::iota(indices.begin(), indices.end(), 1);
    std::iota(indices.begin(), indices.end(), 1);
    cube.compute_orbitals(C_IAO, indices, labels, "iao");

    // returns a nbsf x niao matrix
    return std::make_pair(C_IAO, iao);
}

SharedMatrix UOCDFT::generate_livvos(std::pair<SharedMatrix, std::shared_ptr<IAOBuilder>> iaos,
                                     SharedMatrix Cvir) {
    SharedMatrix LIVVO;

    // Compute the overlap of IAOs with virtual orbitals C_IAO^t S C_vir
    SharedMatrix C_IAO = iaos.first;
    SharedMatrix CiaoSCvir = Matrix::triplet(C_IAO, S_, Cvir, true, false, false);

    // Do SVD of C_IAO^t S C_vir and determine the number of vvos
    std::tuple<SharedMatrix, SharedVector, SharedMatrix> svd = CiaoSCvir->svd_temps();
    SharedMatrix U = std::get<0>(svd);
    SharedVector sigma = std::get<1>(svd);
    SharedMatrix V = std::get<2>(svd);
    CiaoSCvir->svd(U, sigma, V);
    sigma->print();

    int nvvo = 0;
    for (int i = 0; i < sigma->dim(); ++i) {
        double eigen = sigma->get(i);
        if (eigen >= 0.9 and eigen <= 1.1) {
            nvvo += 1;
        }
    }
    outfile->Printf("\n  Found %d VVOs", nvvo);

    Dimension dim_vvo(1);
    dim_vvo[0] = nvvo;
    Dimension dim_bf(1);
    dim_bf[0] = basisset()->nbf();

    // Extract the VVO block
    SharedMatrix Cvir_p = Matrix::doublet(Cvir, V, false, true);
    //    Verified with the following lines
    //    SharedMatrix C_IAO_p = Matrix::doublet(C_IAO,U,false,false);
    //    SharedMatrix CiaoSCvir_p = Matrix::triplet(C_IAO_p, S_, Cvir_p, true, false, false);
    //    CiaoSCvir_p->print();

    SharedMatrix Cvvo = std::make_shared<Matrix>("Cvvo", dim_bf, dim_vvo);
    copy_block(Cvir_p, 1.0, Cvvo, 0.0, dim_bf, dim_vvo);
    Cvvo->print();

    CubeProperties cube = CubeProperties(wfn_);
    std::vector<int> indices(nvvo);
    std::vector<std::string> labels(nvvo);
    std::iota(indices.begin(), indices.end(), 0);
    cube.compute_orbitals(Cvvo, indices, labels, "vvo");

    // Transform the Fock matrix to the VVO basis
    SharedMatrix Fvvo = Matrix::triplet(Cvvo, gs_Fa_, Cvvo, true, false, false);

    // Localize the VVOs
    std::vector<int> ranges;
    std::map<std::string, std::shared_ptr<Matrix>> livvo_info =
        iaos.second->localize(Cvvo, Fvvo, ranges);

    // Grab the LIVVOs and plot them
    SharedMatrix Clivvo = livvo_info["L"];
    cube.compute_orbitals(Clivvo, indices, labels, "livvo");
    return Clivvo;
}

std::vector<std::tuple<int, int, std::string>> UOCDFT::find_iao_character(SharedMatrix Ciao) {

    // Find the value of (atom,n,l) for each basis function
    std::vector<std::tuple<int, int, int>> atom_n_l;
    int count = 0;
    for (int A = 0; A < molecule_->natom(); A++) {
        int Z = static_cast<int>(round(molecule_->Z(A)));

        int n_shell = basisset_->nshell_on_center(A);

        std::vector<int> n_count(10, 1);
        std::iota(n_count.begin(), n_count.end(), 1);

        std::vector<int> ao_list;

        //      if (debug_)
        outfile->Printf("\n  Atom %d (Z = %d) has %d shells\n", A, Z, n_shell);

        for (int Q = 0; Q < n_shell; Q++) {
            const GaussianShell& shell = basisset_->shell(A, Q);
            int nfunction = shell.nfunction();
            int l = shell.am();
            //            if (debug_)
            outfile->Printf("    Shell %d: L = %d, N = %d (%d -> %d)\n", Q, l, nfunction, count,
                            count + nfunction);
            for (int m = 0; m < nfunction; ++m) {
                atom_n_l.push_back(std::make_tuple(A, n_count[l], l));
                count += 1;
            }
            n_count[l] += 1; // increase the angular momentum count
        }

        //        atom_to_aos_[Z].push_back(ao_list);

        //        element_count[Z] += 1; // increase the element count
    }

    int niao = Ciao->ncol();
    int nbf = Ciao->nrow();
    std::vector<std::string> l_to_symbol{"s", "p", "d", "f", "g", "h"};
    std::vector<std::tuple<int, int, std::string>> result;

    for (int rho = 0; rho < niao; rho++) {
        std::map<std::pair<int, int>, double> GP_Al;
        for (int mu = 0; mu < nbf; mu++) {
            double Dmu_rho = 0.0;
            for (int nu = 0; nu < nbf; nu++) {
                Dmu_rho += Ciao->get(mu, rho) * S_->get(mu, nu) * Ciao->get(nu, rho);
            }
            int A, n, l;
            std::tie(A, n, l) = atom_n_l[mu];
            GP_Al[std::make_pair(A, l)] += Dmu_rho;
        }
        std::vector<std::tuple<double, int, int>> sorted_GP;
        for (auto& kv : GP_Al) {
            int A = kv.first.first;
            int l = kv.first.second;
            double GP = kv.second;
            sorted_GP.push_back(std::make_tuple(std::fabs(GP), A, l));
        }
        std::sort(sorted_GP.rbegin(), sorted_GP.rend());

        outfile->Printf("\n\n     =====> IAO %d: Population Analysis <=====", rho + 1);
        outfile->Printf("\n   =================================================");
        outfile->Printf("\n   Atom Number    Symbol     l            population");
        outfile->Printf("\n   =================================================");
        double pop_sum = 0.0;
        for (auto& gp_A_L : sorted_GP) {
            int A = std::get<1>(gp_A_L);
            int l = std::get<2>(gp_A_L);
            double GP = GP_Al[std::make_pair(A, l)];
            pop_sum += GP;
            outfile->Printf("\n   %3d            %3s      %3s           %9.2f ", A + 1,
                            molecule_->symbol(A).c_str(), l_to_symbol[l].c_str(), GP);
        }
        outfile->Printf("\n Population Sum: %f\n", pop_sum);
        int character_A = std::get<1>(sorted_GP[0]);
        int character_l = std::get<2>(sorted_GP[0]);
        std::string character_symbol = molecule_->symbol(character_A);
        result.push_back(std::make_tuple(character_A, character_l, character_symbol));
    }
    return result;
}

void UOCDFT::find_particle_orbital_character(
    SharedMatrix Cmo, SharedMatrix Clivvo, SharedMatrix Ciao,
    std::vector<std::tuple<int, int, std::string>>& IAO_character) {
    std::vector<std::string> l_to_symbol{"s", "p", "d", "f", "g", "h"};

    int np = Cmo->ncol();
    outfile->Printf("\n\n =====> LIVVO Population Analysis <=====\n");

    SharedMatrix particle_livvo_overlap = Matrix::triplet(Cmo, S_, Clivvo, true, false, false);
    SharedMatrix livvo_iao_overlap = Matrix::triplet(Clivvo, S_, Ciao, true, false, false);
    int niao = Ciao->ncol();
    int nlivvo = Clivvo->ncol();
    for (int l = 0; l < nlivvo; l++) {
        double omega_p_l = std::pow(particle_livvo_overlap->get(0, l), 2.0);
        std::vector<double> l_composition(10, 0.0);
        outfile->Printf("\n  |<psi_p|psi_LIVVO(%2d)>|^2 = %9.5f\n", l, omega_p_l);
        std::vector<std::tuple<double, std::string, int, int>> contributions;
        for (int rho = 0; rho < niao; rho++) {
            double S_l_rho = std::pow(livvo_iao_overlap->get(l, rho), 2.0);
            int character_A = std::get<0>(IAO_character[rho]);
            int character_l = std::get<1>(IAO_character[rho]);
            std::string character_symbol = std::get<2>(IAO_character[rho]);
            if (S_l_rho > 0.1) {
                outfile->Printf("\n  |<psi_LIVVO(%2d)|psi_IAO(%2d)>|^2 = %9.5f -> %s%d(%s)", l, rho,
                                S_l_rho, character_symbol.c_str(), character_A + 1,
                                l_to_symbol[character_l].c_str());
                contributions.push_back(
                    std::make_tuple(S_l_rho, character_symbol, character_A, character_l));
                l_composition[character_l] += S_l_rho;
            }
        }
        std::sort(contributions.rbegin(), contributions.rend());
        for (const auto& contribution : contributions) {
            double S_l_rho;
            std::string character_symbol;
            int character_A, character_l;
            std::tie(S_l_rho, character_symbol, character_A, character_l) = contribution;
            outfile->Printf("\n                   %9.5f : %s%d(%s)", l, S_l_rho,
                            character_symbol.c_str(), character_A + 1,
                            l_to_symbol[character_l].c_str());
        }
        for (int character_l = 0; character_l < 3; character_l++) {
            outfile->Printf("\n  Total %s character = %7.2f%%", l_to_symbol[character_l].c_str(),
                            l_composition[character_l] * 100.0);
        }
        //        bool if (l_composition[character_l])
    }
}

/*
{
CharacterTable ct = molecule_->point_group()->char_table();
SharedMatrix S_ao =
    SharedMatrix(new Matrix("AO Overlap matrix", basisset_->nbf(), basisset_->nbf()));
std::shared_ptr<PetiteList> pet(new PetiteList(basisset_, integral_));
SharedMatrix SO2AO_ = pet->sotoao();
int nocc = gs_nalphapi_[0];
int nvir = gs_navirpi_[0];
S_ao->remove_symmetry(S_, SO2AO_);
SharedMatrix iao_coeffs;

TempMatrix->zero();
TempMatrix3->zero();
TempMatrix4->zero();

// Copy the occupied block of Ca and Fa
copy_block(Ca_, 1.0, TempMatrix, 0.0, nsopi_, nalphapi_);
copy_block(dets[0]->Ca(), 1.0, TempMatrix3, 0.0, nsopi_, nalphapi_);
copy_block(Fa_, 1.0, TempMatrix2, 0.0, nsopi_, nalphapi_);
copy_block(gs_Fa_, 1.0, TempMatrix4, 0.0, nsopi_, nalphapi_);

std::shared_ptr<BasisSet> minao = wfn_->get_basisset("MINAO_BASIS");
// Bring in IAO Coefficients (nbfs x niaos)
std::shared_ptr<IAOBuilder> iao =
    IAOBuilder::build(wfn_->basisset(), minao, dets[0]->Ca(), options_);
std::map<std::string, SharedMatrix> ret;
std::map<std::string, std::shared_ptr<Matrix>> ret2;
ret = iao->build_iaos();
std::vector<int> ranges;
ranges.push_back(0);
ranges.push_back(nocc);
ret2 = iao->localize(TempMatrix3, TempMatrix4, ranges);
SharedMatrix L =
    SharedMatrix(new Matrix("AO Overlap matrix", basisset_->nbf(), basisset_->nbf()));
L = ret2["L"];
iao_coeffs = ret["A"];
int nmin = iao_coeffs->colspi()[0];
std::vector<std::string> iao_labels;
// iao_labels = iao->print_IAO(iao_coeffs, nmin, basisset_->nbf(), wfn_);
SharedMatrix IAO_print =
    SharedMatrix(new Matrix("AO Overlap matrix", basisset_->nbf(), basisset_->nbf()));
for (int i = 0; i < basisset_->nbf(); ++i) {
    for (int j = 0; j < nmin; ++j) {
        IAO_print->set(i, j, iao_coeffs->get(i, j));
    }
}




SharedMatrix S_min = SharedMatrix(new Matrix("AO Overlap matrix", nmin, nmin));
SharedMatrix iao_dens =
    SharedMatrix(new Matrix("IAO Density", basisset_->nbf(), basisset_->nbf()));
SharedMatrix iao_population_matrix = SharedMatrix(
    new Matrix("IAO Population Matrix (Overall)", basisset_->nbf(), basisset_->nbf()));

// Form a map that lists all functions on a given atom and with a given ang.
// momentum
std::map<std::tuple<int, int, int>, std::vector<int>> atom_am_to_f;
int sum = 0;
for (int A = 0; A < molecule_->natom(); A++) {
    int principal_qn = 0;
    int n_shell = minao->nshell_on_center(A);
    outfile->Printf("\n N on Shell %d = %d", A + 1, n_shell);
    for (int Q = 0; Q < n_shell; Q++) {
        const GaussianShell& shell = minao->shell(A, Q);
        int nfunction = shell.nfunction();
        int am = shell.am();
        if (am == 0) {
            principal_qn = principal_qn + 1;
        }
        std::tuple<int, int, int> atom_am;
        atom_am = std::make_tuple(A, am, (principal_qn));
        for (int p = sum; p < sum + nfunction; ++p) {
            outfile->Printf("\n nfunction_minao", p);
            atom_am_to_f[atom_am].push_back(p);
        }
        sum += nfunction;
    }
}

std::vector<std::tuple<int, int, int>> keys;
for (auto& kv : atom_am_to_f) {
    keys.push_back(kv.first);
}
std::sort(keys.begin(), keys.end());
std::vector<std::string> l_to_symbol{"s", "p", "d", "f", "g", "h"};
std::vector<std::string> ind_to_orb{"1s", "2s", "2p", "2p", "2p", "3s", "3p",
                                    "3p", "3p", "3d", "3d", "3d", "3d", "3d"};

// Form IAO Fock Matrix //
SharedMatrix F_iao = SharedMatrix(new Matrix("IAO Fock Matrix", nmin, nmin));
SharedMatrix CtFao = SharedMatrix(new Matrix("C^T*F_ao Matrix", nmin, basisset_->nbf()));
CtFao->gemm(true, false, 1.0, iao_coeffs, Fa_, 0.0);
F_iao->gemm(false, false, 1.0, CtFao, iao_coeffs, 0.0);
// F_iao->print();
SharedMatrix F_iao_eigvec =
    SharedMatrix(new Matrix("Eigenvectors of IAO Fock Matrix", nmin, nmin));
SharedVector F_iao_eigvals = SharedVector(new Vector("Eigenvalues of IAO Fock Matrix", nmin));
F_iao->diagonalize(F_iao_eigvec, F_iao_eigvals);
// F_iao_eigvals->print();
for (auto& i : keys) {
    auto& ifn = atom_am_to_f[i];
    for (int iao : ifn) {
        outfile->Printf("\n IAO_%d:%d%s%s(%.3f)", iao + 1, std::get<0>(i) + 1,
                        molecule_->symbol(std::get<0>(i)).c_str(),
                        l_to_symbol[std::get<1>(i)].c_str(), F_iao->get(iao, iao));
    }
}
/////////////////////////

S_min = ret["S_min"];
SharedMatrix hole_iao = SharedMatrix(new Matrix("Hole IAO Density", nmin, nmin));
SharedMatrix part_iao = SharedMatrix(new Matrix("Particle IAO Density", nmin, nmin));
TempMatrix2->zero();
TempMatrix3->zero();
TempMatrix4->zero();
TempMatrix2->gemm(true, false, 1.0, Ca_, S_, 0.0);
SharedMatrix coeffs_transpose =
    SharedMatrix(new Matrix("Particle IAO Density", nmin, basisset_->nbf()));
for (int i = 0; i < nmin; ++i) {
    for (int j = 0; j < basisset_->nbf(); ++j) {
        coeffs_transpose->set(i, j, iao_coeffs->get(j, i));
    }
}
SharedMatrix rhs =
    SharedMatrix(new Matrix("Particle IAO Density (RHS)", nmin, basisset_->nbf()));
for (int i = 0; i < nmin; ++i) {
    for (int j = 0; j < basisset_->nbf(); ++j) {
        double num_in = 0.0;
        for (int k = 0; k < basisset_->nbf(); ++k) {
            num_in += coeffs_transpose->get(i, k) * S_ao->get(k, j);
        }
        rhs->set(i, j, num_in);
    }
}

SharedMatrix c_iao_hole = SharedMatrix(new Matrix("C_iao_hole", nmin, nocc));
for (int i = 0; i < nmin; ++i) {
    for (int j = 0; j < nocc; ++j) {
        double num_in = 0.0;
        for (int k = 0; k < basisset_->nbf(); ++k) {
            num_in += rhs->get(i, k) * Ch_->get(k, j);
        }
        c_iao_hole->set(i, j, num_in);
    }
}
SharedMatrix c_iao_part = SharedMatrix(new Matrix("C_iao_particle", nmin, nvir));
for (int i = 0; i < nmin; ++i) {
    for (int j = 0; j < nvir; ++j) {
        double num_in = 0.0;
        for (int k = 0; k < basisset_->nbf(); ++k) {
            num_in += rhs->get(i, k) * Cp_->get(k, j);
        }
        c_iao_part->set(i, j, num_in);
    }
}

SharedMatrix c_iao_hole_transpose = SharedMatrix(new Matrix("Hole IAO Transpose", nocc, nmin));
for (int i = 0; i < nocc; ++i) {
    for (int j = 0; j < nmin; ++j) {
        c_iao_hole_transpose->set(i, j, c_iao_hole->get(j, i));
    }
}
SharedMatrix c_iao_part_transpose =
    SharedMatrix(new Matrix("Particle IAO Transpose", nvir, nmin));
for (int i = 0; i < nvir; ++i) {
    for (int j = 0; j < nmin; ++j) {
        c_iao_part_transpose->set(i, j, c_iao_part->get(j, i));
    }
}

for (int i = 0; i < nmin; ++i) {
    for (int j = 0; j < nmin; ++j) {
        double num_in = 0.0;
        for (int k = 0; k < nocc; ++k) {
            num_in += c_iao_hole->get(i, k) * c_iao_hole_transpose->get(k, j);
        }
        hole_iao->set(i, j, num_in);
    }
}

std::map<std::tuple<int, int, int>, std::vector<int>> atom_am_to_f_ao;
sum = 0;
for (int A = 0; A < molecule_->natom(); A++) {
    int principal_qn = 0;
    int n_shell = basisset_->nshell_on_center(A);
    // outfile->Printf("\n n_shell: %d", n_shell);
    for (int Q = 0; Q < n_shell; Q++) {
        // outfile->Printf("\n Q: %d", Q);
        const GaussianShell& shell = basisset_->shell(A, Q);
        int nfunction = shell.nfunction();
        int am = shell.am();
        int principal_qn = shell.ncartesian();
        std::tuple<int, int, int> atom_am;
        // principal_qn_s = principal_qn_s + 1;
        atom_am = std::make_tuple(A, am, (principal_qn));
        // outfile->Printf("\n Atom:%d, AM:%d, PQN:%d", atom_am.get<0>(),
        // atom_am.get<1>(), atom_am.get<2>());
        for (int p = sum; p < sum + nfunction; ++p) {
            // outfile->Printf("\n nfunction: %d", p);
            atom_am_to_f_ao[atom_am].push_back(p);
        }
        sum += nfunction;
        // prev_am = am;
    }
}
SharedMatrix Left_hand = SharedMatrix(new Matrix("Left-Hand_side", nmin, basisset_->nbf()));
SharedMatrix Total = SharedMatrix(new Matrix("Total", nmin, nmin));
Left_hand->gemm(true, false, 1.0, iao_coeffs, S_, 0.0);
// TempMatrix->zero();
Total->gemm(false, false, 1.0, Left_hand, iao_coeffs, 0.0);
// Total->print();
std::vector<std::tuple<int, int, int>> keys_ao;

for (auto& kv : atom_am_to_f_ao) {
    keys_ao.push_back(kv.first);
}
std::sort(keys_ao.begin(), keys_ao.end());
// std::vector<std::string> l_to_symbol{"s","p","d","f","g","h"};
for (int k = 0; k < nmin; ++k) {
    for (int i = 0; i < basisset_->nbf(); ++i) {
        for (int j = 0; j < basisset_->nbf(); ++j) {
            iao_dens->set(i, j, iao_coeffs->get(i, k) * iao_coeffs->get(j, k));
        }
    }
    iao_population_matrix->gemm(false, false, 1.0, iao_dens, S_, 0.0);
    // iao_population_matrix->print();
    double pop_sum = 0.0;
    for (int i = 0; i < basisset_->nbf(); ++i) {
        for (int j = 0; j < basisset_->nbf(); ++j) {
            if (i == j) {
                pop_sum += iao_population_matrix->get(i, j);
            }
        }
    }
    // iao_population_matrix->print();
    std::vector<std::pair<double, std::string>> ao_iao_coeff_contributions;
    std::vector<std::pair<double, std::string>> ao_iao_coeff_contributions_print;
    outfile->Printf("\n Population Sum: %f\n", pop_sum);
    outfile->Printf("\n\n     =====> IAO %d: Population Analysis <=====", k + 1);
    outfile->Printf("\n   =================================================");
    outfile->Printf("\n   Atom Number    Symbol     l            population");
    outfile->Printf("\n   =================================================");
    for (auto& i : keys_ao) {
        auto& ifn = atom_am_to_f_ao[i];
        double sum = 0.0;
        for (int iao : ifn) {
            // outfile->Printf("\n IAO Number: %d", iao);
            // if(iao < basisset_->nbf()){
            // outfile->Printf("\n IAO Number: %d", iao);
            sum += iao_population_matrix->get(iao, iao);
            // outfile->Printf("\n Executed Sum");
            //}
        }
        double norm_sum = sum;
        if (std::fabs(norm_sum) >= 0.00001) {
            // outfile->Printf("\n In here");
            std::string outstr =
                boost::str(boost::format("   %3d            %3s      %3s           %9.2f ") %
                           (std::get<0>(i) + 1) % molecule_->symbol(std::get<0>(i)).c_str() %
                           l_to_symbol[std::get<1>(i)].c_str() % norm_sum);

            std::string outstr_compact =
                boost::str(boost::format("%3d %3s(%s,%4.2f)") % (std::get<0>(i) + 1) %
                           molecule_->symbol(std::get<0>(i)).c_str() %
                           l_to_symbol[std::get<1>(i)].c_str() % norm_sum);
            std::string label_string =
                boost::str(boost::format("%d%s%s_%d") % (std::get<0>(i) + 1) %
                           molecule_->symbol(std::get<0>(i)).c_str() %
                           l_to_symbol[std::get<1>(i)].c_str() % (k + 1));
            ao_iao_coeff_contributions_print.push_back(std::make_pair(norm_sum, outstr));
            ao_iao_coeff_contributions.push_back(std::make_pair(norm_sum, outstr_compact));
        }
    }
    std::sort(ao_iao_coeff_contributions_print.rbegin(),
              ao_iao_coeff_contributions_print.rend());
    std::sort(ao_iao_coeff_contributions.rbegin(), ao_iao_coeff_contributions.rend());
    // outfile->Printf("\n Sorted");
    for (auto& kv : ao_iao_coeff_contributions_print) {
        outfile->Printf("\n%s", kv.second.c_str());
    }
    iao_labels.push_back(ao_iao_coeff_contributions[0].second.c_str());
}
for (int i = 0; i < nmin; ++i) {
    for (int j = 0; j < nmin; ++j) {
        double num_in = 0.0;
        for (int k = 0; k < nvir; ++k) {
            num_in += c_iao_part->get(i, k) * c_iao_part_transpose->get(k, j);
        }
        part_iao->set(i, j, num_in);
    }
}
std::vector<std::pair<double, std::string>> iao_hole_contributions;
std::vector<std::pair<double, std::string>> iao_part_contributions;
std::vector<std::pair<double, std::string>> ao_hole_contributions;
std::vector<std::pair<double, std::string>> ao_part_contributions;

// Form a map that lists all functions on a given atom and with a given ang.
// momentum
// std::map<std::tuple<int,int,int>,std::vector<int>> atom_am_to_f;
// sum = 0;
// for (int A = 0; A < molecule_->natom(); A++) {
//    int principal_qn = 0;
//    int n_shell = minao->nshell_on_center(A);
//    for (int Q = 0; Q < n_shell; Q++){
//        const GaussianShell& shell = minao->shell(A,Q);
//        int nfunction = shell.nfunction();
//        int am = shell.am();
//        if(am==0){
//    	principal_qn = principal_qn + 1;
//        }
//        std::tuple<int,int,int> atom_am;
//        atom_am = std::make_tuple(A,am,(principal_qn));
//        for (int p = sum; p < sum + nfunction; ++p){
//            atom_am_to_f[atom_am].push_back(p);
//        }
//        sum += nfunction;
//    }
//}
//
//// "I got the keys, the keys, the keys"
// std::vector<std::tuple<int,int,int>> keys;
// for (auto& kv : atom_am_to_f){
//    keys.push_back(kv.first);
//}
// std::sort(keys.begin(),keys.end());
outfile->Printf("\n\n     =====> IAO Analysis of Hole Orbital <=====");
outfile->Printf("\n   =====================================================");
outfile->Printf("\n   Atom Number    Symbol         l            population");
outfile->Printf("\n   =====================================================");

// std::vector<std::string> l_to_symbol{"s","p","d","f","g","h"};
for (auto& i : keys) {
    auto& ifn = atom_am_to_f[i];
    // Mulliken Analysis from IAO Density Matrix
    // 1. Form Population Matrix
    // 2. Take Trace of Population Matrix
    // 3. Decompose by atom

    SharedMatrix population_matrix_hole =
        SharedMatrix(new Matrix("IAO Population Matrix ", nmin, nmin));
    population_matrix_hole->gemm(false, false, 1.0, hole_iao, S_min, 0.0);
    double trace = 0.0;
    double reg_trace = 0.0;
    for (int iao : ifn) {
        trace += population_matrix_hole->get(iao, iao);
    }
    // outfile->Printf("\n        %d            %s          %s           %9.2f
    // ",
    // i.first+1,molecule_->symbol(i.first).c_str(),l_to_symbol[i.second].c_str(),trace);
    if (std::fabs(trace) >= 0.01) {
        std::string outstr =
            boost::str(boost::format("   %3d            %3s          %3s           %9.2f ") %
                       (std::get<0>(i) + 1) % molecule_->symbol(std::get<0>(i)).c_str() %
                       l_to_symbol[std::get<1>(i)].c_str() % trace);
        iao_hole_contributions.push_back(std::make_pair(trace, outstr));
    }
}
std::sort(iao_hole_contributions.rbegin(), iao_hole_contributions.rend());
for (auto& kv : iao_hole_contributions) {
    outfile->Printf("\n%s", kv.second.c_str());
}
outfile->Printf("\n   -----------------------------------------------------");

outfile->Printf("\n\n     ===> IAO Analysis of Particle Orbital <===");
outfile->Printf("\n   =====================================================");
outfile->Printf("\n   Atom Number     Symbol       l             population");
outfile->Printf("\n   =====================================================");
for (auto& i : keys) {
    auto& ifn = atom_am_to_f[i];
    SharedMatrix population_matrix_part =
        SharedMatrix(new Matrix("IAO Particle Population Matrix ", nmin, nmin));
    population_matrix_part->gemm(false, false, 1.0, part_iao, S_min, 0.0);
    double trace = 0.0;
    for (int iao : ifn) {
        trace += population_matrix_part->get(iao, iao);
    }
    if (std::fabs(trace) >= 0.01) {
        std::string outstr =
            boost::str(boost::format("   %3d            %3s          %3s           %9.2f ") %
                       (std::get<0>(i) + 1) % molecule_->symbol(std::get<0>(i)).c_str() %
                       l_to_symbol[std::get<1>(i)].c_str() % trace);
        iao_part_contributions.push_back(std::make_pair(trace, outstr));
    }
}
std::sort(iao_part_contributions.rbegin(), iao_part_contributions.rend());
for (auto& kv : iao_part_contributions) {
    outfile->Printf("\n%s", kv.second.c_str());
}
outfile->Printf("\n   -----------------------------------------------------");
if (options_.get_bool("FULL_MULLIKEN_PRINT")) {
    std::map<std::pair<int, int>, std::vector<int>> atom_am_to_f_full;
    sum = 0;
    for (int A = 0; A < molecule_->natom(); A++) {
        int n_shell = basisset_->nshell_on_center(A);
        for (int Q = 0; Q < n_shell; Q++) {
            const GaussianShell& shell = basisset_->shell(A, Q);
            int nfunction = shell.nfunction();
            int am = shell.am();
            std::pair<int, int> atom_am(A, am);
            for (int p = sum; p < sum + nfunction; ++p) {
                atom_am_to_f_full[atom_am].push_back(p);
            }
            sum += nfunction;
        }
    }

    // "I got the keys, the keys, the keys"
    std::vector<std::pair<int, int>> keys_full;
    for (auto& kv : atom_am_to_f_full) {
        keys_full.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());
    outfile->Printf("\n\n     =====> AO Analysis of Hole Orbital <=====");
    outfile->Printf("\n   =====================================================");
    outfile->Printf("\n   Atom Number     Symbol       l             population");
    outfile->Printf("\n   =====================================================");

    std::vector<std::string> l_to_symbol_full{"s", "p", "d", "f", "g", "h"};
    int hole_index = 0;
    for (auto& i : keys_full) {
        auto& ifn = atom_am_to_f_full[i];
        // Mulliken Analysis from IAO Density Matrix
        // 1. Form Population Matrix
        // 2. Take Trace of Population Matrix
        // 3. Decompose by atom

        SharedMatrix population_matrix_hole_full = SharedMatrix(
            new Matrix("IAO Population Matrix ", basisset_->nbf(), basisset_->nbf()));
        SharedMatrix Dh_ = SharedMatrix(new Matrix("Hole Detachment Density", nsopi_, nsopi_));
        Dh_->gemm(false, true, 1.0, Ch_, Ch_, 0.0);
        population_matrix_hole_full->gemm(false, false, 1.0, Dh_, S_ao, 0.0);
        double trace = 0.0;
        for (int iao : ifn) {
            trace += population_matrix_hole_full->get(iao, iao);
        }
        if (std::fabs(trace) >= 0.01) {
            std::string outstr = boost::str(
                boost::format("   %3d            %3s          %3s           %9.2f ") %
                (i.first + 1) % molecule_->symbol(i.first).c_str() %
                l_to_symbol[i.second].c_str() % trace);
            ao_hole_contributions.push_back(std::make_pair(trace, outstr));
        }
        if (std::fabs(trace) >= 0.85) {
            hole_index = i.first;
        }
    }
    std::sort(ao_hole_contributions.rbegin(), ao_hole_contributions.rend());
    for (auto& kv : ao_hole_contributions) {
        outfile->Printf("\n%s", kv.second.c_str());
    }
    outfile->Printf("\n   -----------------------------------------------------");
    outfile->Printf("\n\n     ==> AO Analysis of Particle Orbital <==");
    outfile->Printf("\n   =====================================================");
    outfile->Printf("\n   Atom Number     Symbol       l             population");
    outfile->Printf("\n   =====================================================");
    double local_p_character_sum;
    for (auto& i : keys_full) {
        auto& ifn = atom_am_to_f_full[i];

        SharedMatrix population_matrix_particle_full = SharedMatrix(
            new Matrix("IAO Population Matrix ", basisset_->nbf(), basisset_->nbf()));
        SharedMatrix Dp_ =
            SharedMatrix(new Matrix("Particle Detachment Density", nsopi_, nsopi_));
        Dp_->gemm(false, true, 1.0, Cp_, Cp_, 0.0);
        population_matrix_particle_full->gemm(false, false, 1.0, Dp_, S_ao, 0.0);
        double trace = 0.0;
        if (i.first == hole_index and l_to_symbol[i.second] == "p") {
            for (int iao : ifn) {
                local_p_character_sum += Cp_->get(iao, 0) * Cp_->get(iao, 0);
            }
        }
        for (int iao : ifn) {
            trace += population_matrix_particle_full->get(iao, iao);
        }
        if (std::fabs(trace) >= 0.01) {
            std::string outstr = boost::str(
                boost::format("   %3d            %3s          %3s           %9.2f ") %
                (i.first + 1) % molecule_->symbol(i.first).c_str() %
                l_to_symbol[i.second].c_str() % trace);
            ao_part_contributions.push_back(std::make_pair(trace, outstr));
        }
    }
    std::sort(ao_part_contributions.rbegin(), ao_part_contributions.rend());
    for (auto& kv : ao_part_contributions) {
        outfile->Printf("\n%s", kv.second.c_str());
    }
    outfile->Printf("\n   -----------------------------------------------------");
    outfile->Printf("\n Local P Character: %f \n", local_p_character_sum);
}
// outfile->Printf("\n Local P Character: %f \n", local_p_character_sum);
// outfile->Printf("\n\n  Analysis of the hole/particle MOs in terms of the
// ground state DFT MOs");
if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
    TempMatrix->gemm(false, false, 1.0, S_, dets[1]->Ca(), 0.0);
} else {
    TempMatrix->gemm(false, false, 1.0, S_, dets[0]->Ca(), 0.0);
}
int temp_int_new = 0;
if (options_.get_bool("VALENCE_TO_CORE") and state_ != 1) {
    temp_int_new = 1;
} else {
    temp_int_new = 0;
}
SharedMatrix Temporary_hole =
    SharedMatrix(new Matrix("Hole Transformation Matrix", basisset_->nbf(), nocc));
SharedMatrix Temporary_particle =
    SharedMatrix(new Matrix("Particle Transformation Matrix", basisset_->nbf(), nvir));
SharedMatrix Temporary_particle_rev =
    SharedMatrix(new Matrix("Particle Transformation Matrix", nvir, basisset_->nbf()));
SharedMatrix Temporary_particle_iao =
    SharedMatrix(new Matrix("Particle Transformation Matrix", nmin, nvir));
SharedMatrix Temporary_iao_overlap =
    SharedMatrix(new Matrix("IAO hole transform Matrix", nvir, basisset_->nbf()));
SharedMatrix Temporary_iao_transform =
    SharedMatrix(new Matrix("IAO hole transform Matrix", nmin, basisset_->nbf()));
Temporary_hole->gemm(false, false, 1.0, S_, Ch_, 0.0);
TempMatrix->zero();
copy_block(dets[0]->Ca(), 1.0, TempMatrix, 0.0, nsopi_, nmopi_ - nalphapi_, zero_dim_,
           nalphapi_, zero_dim_, napartpi_);
// TempMatrix->print();
std::vector<SharedMatrix> overlaps;
SharedMatrix current_overlap(wfn_->S()->clone());
overlaps.push_back(current_overlap);
Temporary_particle->gemm(false, false, 1.0, overlaps[0], Cp_, 0.0);
Temporary_particle_rev->gemm(true, false, 1.0, Cp_, overlaps[0], 0.0);
Temporary_iao_overlap->gemm(true, false, 1.0, TempMatrix, overlaps[0], 0.0);
Temporary_iao_transform->gemm(true, false, 1.0, iao_coeffs, overlaps[0], 0.0);

// FORM MATRICES FOR ss^t matrix in valence virtual orbitals! //
SharedMatrix CtS = SharedMatrix(new Matrix("CtS ", nvir, basisset_->nbf()));
SharedMatrix CtSA = SharedMatrix(new Matrix("CtSA", nvir, nmin));
SharedMatrix CtSAAt = SharedMatrix(new Matrix("CtSAAt", nvir, basisset_->nbf()));
SharedMatrix CtSAAtS = SharedMatrix(new Matrix("CtSAAtS ", nvir, basisset_->nbf()));
SharedMatrix CtSAAtSC = SharedMatrix(new Matrix("CtSAAtSC ", nvir, nvir));
SharedMatrix CtSAAtSC_eigvec = SharedMatrix(new Matrix("CtSAAtSC Eigen Vector", nvir, nvir));
SharedVector CtSAAtSC_eigvals = SharedVector(new Vector("CtSAAtSC Eigen Values", nvir));

SharedMatrix Ground_State_Overlap =
    SharedMatrix(new Matrix("Rotated Particle Orbital", basisset_->nbf(), basisset_->nbf()));
Ground_State_Overlap->gemm(true, false, 1.0, dets[0]->Ca(), dets[0]->Ca(), 0.0);

CtS->gemm(true, false, 1.0, TempMatrix, S_, 0.0);
CtSA->gemm(false, false, 1.0, CtS, iao_coeffs, 0.0);
CtSAAt->gemm(false, true, 1.0, CtSA, iao_coeffs, 0.0);
CtSAAtS->gemm(false, false, 1.0, CtSAAt, S_, 0.0);
CtSAAtSC->gemm(false, false, 1.0, CtSAAtS, TempMatrix, 0.0);
// CtSAAtSC->print();
CtSAAtSC->diagonalize(CtSAAtSC_eigvec, CtSAAtSC_eigvals);
// CtSAAtSC_eigvals->print();
for (int i = 0; i < nvir; ++i) {
    outfile->Printf("\n %f \n", CtSAAtSC_eigvals->get(i));
}
// END //
SharedMatrix LSCh = SharedMatrix(new Matrix("IBO Hole Overlap ", basisset_->nbf(), nocc));
SharedMatrix LSCp = SharedMatrix(new Matrix("IBO Particle Overlap ", basisset_->nbf(), nvir));
SharedMatrix ASCh = SharedMatrix(new Matrix("IAO Hole Overlap ", nmin, nocc));
SharedMatrix CpSA = SharedMatrix(new Matrix("Hole Overlap with IAO Basis ", nvir, nmin));
SharedMatrix LvvoSCp =
    SharedMatrix(new Matrix("Particle Overlap with VVO basis", basisset_->nbf(), nvir));
SharedMatrix LvvoSCp_iao =
    SharedMatrix(new Matrix("Particle Overlap with VVO basis in IAO basis", nmin, nvir));
SharedMatrix CpSLvvo =
    SharedMatrix(new Matrix("Particle Overlap with VVO basis", nvir, basisset_->nbf()));
SharedMatrix ASCp = SharedMatrix(new Matrix("IAO Particle Overlap ", nmin, nvir));
SharedMatrix ASLvvo = SharedMatrix(new Matrix("IAO Particle Overlap ", nmin, basisset_->nbf()));
SharedMatrix ASCp_print =
    SharedMatrix(new Matrix("IAO Particle Overlap ", basisset_->nbf(), basisset_->nbf()));
CpSA->gemm(false, false, 1.0, Temporary_iao_overlap, iao_coeffs, 0.0);
// CpSA->print();
std::tuple<SharedMatrix, SharedVector, SharedMatrix> UCpSAV = CpSA->svd_temps();
SharedMatrix U = std::get<0>(UCpSAV);
SharedVector sigma = std::get<1>(UCpSAV);
SharedMatrix V = std::get<2>(UCpSAV);
CpSA->svd(U, sigma, V);
// sigma->print();
// U->print();
// V->print();
// int nvvos = 0;
for (int i = 0; i < nmin; ++i) {
    double eigen = 0.0;
    eigen = sigma->get(i);
    if (eigen >= 0.9 and eigen <= 1.1) {
        outfile->Printf("\n ADDING ONE %f", sigma->get(i));
        nvvos = nvvos + 1;
    }
}
outfile->Printf("\n There are %d Valence Virtual Orbitals", nvvos);
SharedMatrix Rotated_particle =
    SharedMatrix(new Matrix("Rotated Particle Orbital", basisset_->nbf(), nvir));
// SharedMatrix Ground_State_Overlap = SharedMatrix(new Matrix("Rotated
// Particle Orbital", basisset_->nbf(),basisset_->nbf()));
SharedMatrix Occ_Space_Matrix =
    SharedMatrix(new Matrix("Occupied Space", basisset_->nbf(), basisset_->nbf()));
SharedMatrix VVO_Space_Matrix =
    SharedMatrix(new Matrix("VVO Space", basisset_->nbf(), basisset_->nbf()));
SharedMatrix EXT_Space_Matrix =
    SharedMatrix(new Matrix("External Orbital Space", basisset_->nbf(), basisset_->nbf()));
SharedMatrix Full_Space_Matrix =
    SharedMatrix(new Matrix("Occ+VVO+Ext Space", basisset_->nbf(), basisset_->nbf()));
// std::string local_type_ = options().get_str("LOCALIZE_TYPE");
// Ground_State_Overlap->gemm(true,false,1.0,dets[0]->Ca(),dets[0]->Ca(),0.0);
for (int i = 0; i < basisset_->nbf(); ++i) {
    for (int j = 0; j < nvir; ++j) {
        double post_svd_sum = 0.0;
        for (int k = 0; k < nvir; ++k) {
            post_svd_sum += TempMatrix->get(i, k) * U->get(k, j);
        }
        Rotated_particle->set(i, j, post_svd_sum);
    }
}

for (int i = 0; i < nocc; ++i) {
    SharedVector vec;
    vec = dets[0]->Ca()->get_column(0, i);
    Occ_Space_Matrix->set_column(0, i, vec);
}
for (int i = 0; i < nvvos; ++i) {
    SharedVector vec;
    vec = Rotated_particle->get_column(0, i);
    VVO_Space_Matrix->set_column(0, i + nocc, vec);
}
for (int i = 0; i < (nvir - nvvos); ++i) {
    SharedVector vec;
    vec = Rotated_particle->get_column(0, i + (nvvos));
    EXT_Space_Matrix->set_column(0, i + (nocc + nvvos), vec);
}
SharedMatrix L_vvo;
SharedMatrix L_occ;

// Copy the virtual block of Ca and Fa
std::map<std::string, std::shared_ptr<Matrix>> ret_vvo;
TempMatrix3->zero();
TempMatrix4->zero();
copy_block(dets[0]->Ca(), 1.0, TempMatrix3, 0.0, nsopi_, nmopi_ - nalphapi_, zero_dim_,
           nalphapi_, zero_dim_, napartpi_);
copy_block(gs_Fa_, 1.0, TempMatrix4, 0.0, nsopi_, nmopi_ - nalphapi_, zero_dim_, nalphapi_,
           zero_dim_, napartpi_);

ranges.push_back(nocc);
ranges.push_back(nocc + nvvos);
// VVO_Space_Matrix->print();
ret_vvo = iao->localize(VVO_Space_Matrix, TempMatrix4, ranges);
SharedMatrix L_livvo = SharedMatrix(new Matrix("Localized Intrinsic Valence Virtual Orbitals",
                                               basisset_->nbf(), basisset_->nbf()));
SharedMatrix L_livvo_transpose = SharedMatrix(new Matrix(
    "Localized Intrinsic Valence Virtual Orbitals", basisset_->nbf(), basisset_->nbf()));
// L_livvo->print();
L_livvo = ret_vvo["L"];
// outfile->Printf("\n Printing L_livvo Matrix \n");
// L_livvo->print();
for (int i = 0; i < basisset_->nbf(); ++i) {
    for (int j = 0; j < basisset_->nbf(); ++j) {
        L_livvo_transpose->set(i, j, L_livvo->get(j, i));
    }
}
std::shared_ptr<Localizer> loc_vvo =
    Localizer::build("PIPEK_MEZEY", wfn_->basisset(), VVO_Space_Matrix);
std::shared_ptr<Localizer> loc_occ =
    Localizer::build("PIPEK_MEZEY", wfn_->basisset(), Occ_Space_Matrix);
loc_vvo->localize();
loc_occ->localize();
L_vvo = loc_vvo->L();
// L_vvo->print();
L_occ = loc_occ->L();
// VVO_Space_Matrix->print();
VVO_Space_Matrix->zero();
Occ_Space_Matrix->zero();
TempMatrix->zero();
TempMatrix2->zero();
TempMatrix->gemm(true, false, 1.0, L_livvo, S_, 0.0);
TempMatrix2->gemm(false, false, 1.0, TempMatrix, L_vvo, 0.0);
//    TempMatrix2->print();
// L_livvo_transpose->print();
// VVO_Space_Matrix->print();
for (int i = 0; i < nvvos; ++i) {
    SharedVector vec;
    vec = L_vvo->get_column(0, i + nocc);
    VVO_Space_Matrix->set_column(0, i + nocc, vec);
}
for (int i = 0; i < nocc; ++i) {
    SharedVector vec;
    vec = L_occ->get_column(0, i);
    Occ_Space_Matrix->set_column(0, i, vec);
}

Full_Space_Matrix->add(Occ_Space_Matrix);
Full_Space_Matrix->add(VVO_Space_Matrix);
Full_Space_Matrix->add(EXT_Space_Matrix);
// Psuedo-Canonicalization Procedure //
// TempMatrix2->zero();
// TempMatrix3->zero();
// TempMatrix2->gemm(true,false,1.0,VVO_Space_Matrix,Fa_,0.0);
// TempMatrix3->gemm(false,false,1.0,TempMatrix2,VVO_Space_Matrix,0.0);
// TempMatrix3->print();
// Full_Space_Matrix->print();
// Ca_ = Full_Space_Matrix;
// Cb_ = Full_Space_Matrix;
// dets[0]->Ca()->print();
// Rotated_particle->print();
// Full_VVO_Matrix->print();
// std::shared_ptr<Localizer> loc_vvo = Localizer::build("PIPEK_MEZEY",
// wfn_->basisset(), VVO_Space_Matrix);
// loc_vvo->localize();
// SharedMatrix L_vvo = loc_vvo->L();
// L_vvo->print();
LSCh->gemm(true, false, 1.0, L, Temporary_hole, 0.0);
ASCh->gemm(true, false, 1.0, iao_coeffs, Temporary_hole, 0.0);
// ASCh->print();
LSCp->gemm(true, false, 1.0, L, Temporary_particle, 0.0);
SharedMatrix Localized_particle =
    SharedMatrix(new Matrix("PM Localized Particle Orbital", nmin, basisset_->nbf()));
ASCp->gemm(true, false, 1.0, iao_coeffs, Temporary_particle, 0.0);
TempMatrix4->zero();
TempMatrix4 = VVO_Space_Matrix;
TempMatrix4->add(EXT_Space_Matrix);
LvvoSCp->gemm(true, false, 1.0, L_vvo, Temporary_particle, 0.0);
CpSLvvo->gemm(false, false, 1.0, Temporary_particle_rev, L_vvo, 0.0);
ASLvvo->gemm(false, false, 1.0, Temporary_iao_transform, L_vvo, 0.0);
for (int i = 0; i < nmin; ++i) {
    for (int j = 0; j < nvir; ++j) {
        ASCp_print->set(i, j, ASCp->get(i, j));
    }
}
ASCp->gemm(true, false, 1.0, iao_coeffs, Temporary_particle, 0.0);
// std::shared_ptr<Localizer> loc_Cp = Localizer::build("PIPEK_MEZEY",
// wfn_->basisset(), ASCp_print);
// Localized_particle = loc_Cp->L();
TempMatrix4->zero();
TempMatrix4 = VVO_Space_Matrix;
TempMatrix4->add(EXT_Space_Matrix);
LvvoSCp->gemm(true, false, 1.0, L_livvo, Temporary_particle, 0.0);
CpSLvvo->gemm(false, false, 1.0, Temporary_particle_rev, L_livvo, 0.0);
ASLvvo->gemm(false, false, 1.0, Temporary_iao_transform, L_livvo, 0.0);
// Temporary_particle_iao->gemm(false,false,1.0,S_min,ASCp,0.0);
Temporary_particle_iao->gemm(false, false, 1.0, S_min, ASCp, 0.0);
LvvoSCp_iao->gemm(true, false, 1.0, ASLvvo, Temporary_particle_iao, 0.0);
// for (int i = 0; i < nmin; ++i){
//    for (int j = 0; j < nvir; ++j){
//        ASCp_print->set(i,j,ASCp->get(i,j));
//    }
//}
// ASCp_print->print();
// CubeProperties cube = CubeProperties(wfn_);
// std::vector<int> indsp0;
// std::vector<std::string> labelsp;
// indsp0.push_back(0);
// labelsp.push_back("part_iao");
// cube.compute_orbitals(ASCp_print, indsp0,labelsp, "1");
// ASCp->print();
// LSCp->gemm(true,false,1.0,L,Temporary_particle,0.0);
// ASCp->gemm(true,false,1.0,IAO_print,Temporary_particle,0.0);
std::vector<std::pair<double, int>> pair_occ_ibo_hole;
for (int i = 0; i < nocc; ++i) {
    double overlap = std::pow(LSCh->get(i, 0), 2.0);
    // outfile->Printf("\n %f", overlap);
    if (overlap > 0.01) {
        pair_occ_ibo_hole.push_back(std::make_pair(overlap, i + 1));
    }
}
CubeProperties cube_livvo = CubeProperties(wfn_);
if (options_.get_bool("IBO_ANALYSIS")) {
    outfile->Printf("\n\n  Analysis of the hole/particle MOs in terms of the "
                    "Intrinsic Bond Orbitals (IBOs)");
    std::sort(pair_occ_ibo_hole.begin(), pair_occ_ibo_hole.end());
    std::reverse(pair_occ_ibo_hole.begin(), pair_occ_ibo_hole.end());
    outfile->Printf("\n Hole: ");
    for (auto occ_ibo : pair_occ_ibo_hole) {
        outfile->Printf("|%.1f%% IBO%d|", occ_ibo.first * 100.0, occ_ibo.second);
    }
    outfile->Printf("\n");
    std::vector<std::pair<double, int>> pair_occ_ibo_part;
    for (int i = 0; i < nocc; ++i) {
        double overlap = std::pow(LSCp->get(i, 0), 2.0);
        if (overlap > 0.01) {
            pair_occ_ibo_part.push_back(std::make_pair(overlap, i + 1));
        }
    }
    std::sort(pair_occ_ibo_part.begin(), pair_occ_ibo_part.end());
    std::reverse(pair_occ_ibo_part.begin(), pair_occ_ibo_part.end());
    outfile->Printf("\n Particle: ");
    for (auto occ_ibo : pair_occ_ibo_part) {
        outfile->Printf("|%.1f%% IBO%d|", occ_ibo.first * 100.0, occ_ibo.second);
    }
    // CubeProperties cube_livvo = CubeProperties(wfn_);
    std::vector<int> indsp0_occ;
    std::vector<std::string> labelsp_occ;
    for (int i = 0; i < nocc; ++i) {
        indsp0_occ.push_back(i);
    }
    for (int i = 0; i < nocc; ++i) {
        labelsp_occ.push_back("part_iao");
    }
    cube_livvo.compute_orbitals(L, indsp0_occ, labelsp_occ, "ibo");
}
// Printing LIVVO Cube file
// CubeProperties cube = CubeProperties(wfn_);
std::vector<int> indsp0_livvo;
std::vector<std::string> labelsp_livvo;
for (int i = nocc; i < (nocc + nvvos); ++i) {
    indsp0_livvo.push_back(i);
}
for (int i = nocc; i < (nocc + nvvos); ++i) {
    labelsp_livvo.push_back("livvo");
}
std::string livvo_label = "state_" + to_string(state_) + "_";
cube_livvo.compute_orbitals(L_livvo, indsp0_livvo, labelsp_livvo, livvo_label);
std::vector<std::pair<double, std::string>> pair_occ_iao_hole;
SharedMatrix C_h_cont = SharedMatrix(
    new Matrix("Sum of IAO Contributions Hole ", basisset_->nbf(), basisset_->nbf()));
for (auto& i : keys) {
    auto& ifn = atom_am_to_f[i];
    for (auto& iao : ifn) {
        double overlap = std::pow(ASCh->get(iao, 0), 2.0);
        std::string outstr = boost::str(boost::format("%d%s(%d%s)") % (std::get<0>(i) + 1) %
                                        molecule_->symbol(std::get<0>(i)).c_str() %
                                        std::get<2>(i) % l_to_symbol[std::get<1>(i)].c_str());
        // outfile->Printf("\n %s",outstr.c_str());
        TempMatrix->zero();
        if (overlap > 0.01) {
            for (int j = 0; j < basisset_->nbf(); ++j) {
                TempMatrix->set(j, 0, overlap * iao_coeffs->get(j, iao));
            }
            pair_occ_iao_hole.push_back(std::make_pair(overlap, outstr));
            C_h_cont->add(TempMatrix);
        }
    }
}
// std::sort(pair_occ_iao_part.begin(),pair_occ_iao_part.end());
// C_h_cont->print();
CubeProperties cube = CubeProperties(wfn_);
std::vector<int> indsp0_hole;
std::vector<std::string> labelsp_hole;
indsp0_hole.push_back(0);
labelsp_hole.push_back("hole_iao");
cube.compute_orbitals(C_h_cont, indsp0_hole, labelsp_hole, "1");
// std::vector<std::pair<double,int> > pair_occ_iao_hole;
// for (int i = 0; i < basisset_->nbf(); ++i){
//        double overlap = std::pow(ASCh->get(i,0),2.0);
//        if(overlap>0.01){
//            pair_occ_iao_hole.push_back(std::make_pair(overlap,i+1));
//        }
//}

outfile->Printf("\n\n  Analysis of the hole/particle MOs in terms of the "
                "Intrinsic Atomic Orbitals (IAOs)");
std::sort(pair_occ_iao_hole.begin(), pair_occ_iao_hole.end());
std::reverse(pair_occ_iao_hole.begin(), pair_occ_iao_hole.end());
outfile->Printf("\n Hole:     ");
for (auto occ_iao : pair_occ_iao_hole) {
    outfile->Printf("|%.1f%% %s|", occ_iao.first * 100.0, occ_iao.second.c_str());
}
outfile->Printf("\n");

std::vector<std::pair<double, std::string>> pair_occ_iao_part;
std::vector<std::pair<double, std::string>> pair_occ_vvoiao_part;
SharedMatrix C_p_cont =
    SharedMatrix(new Matrix("Sum of IAO Contributions ", basisset_->nbf(), basisset_->nbf()));
for (auto& i : keys) {
    auto& ifn = atom_am_to_f[i];
    for (auto& iao : ifn) {
        double overlap = std::pow(ASCp->get(iao, 0), 2.0);
        std::string outstr = boost::str(boost::format("%d%s(%d%s)") % (std::get<0>(i) + 1) %
                                        molecule_->symbol(std::get<0>(i)).c_str() %
                                        std::get<2>(i) % l_to_symbol[std::get<1>(i)].c_str());
        // outfile->Printf("\n %s",outstr.c_str());
        TempMatrix->zero();
        if (overlap > 0.01) {
            for (int j = 0; j < basisset_->nbf(); ++j) {
                TempMatrix->set(j, 0, overlap * iao_coeffs->get(j, iao));
            }
            pair_occ_iao_part.push_back(std::make_pair(overlap, outstr));
            C_p_cont->add(TempMatrix);
        }
    }
}
// LvvoSCp->print();
std::vector<std::pair<double, int>> pair_occ_vvo_part;
// LvvoSCp_iao->print();
for (int i = 0; i < nvir; ++i) {
    double overlap = std::pow(LvvoSCp->get(i + nocc, 0), 2.0);
    if (overlap > 0.01) {
        pair_occ_vvo_part.push_back(std::make_pair(overlap, (i + nocc) + 1));
    }
}
std::sort(pair_occ_vvo_part.begin(), pair_occ_vvo_part.end());
std::reverse(pair_occ_vvo_part.begin(), pair_occ_vvo_part.end());
double valence_sum = 0.0;
outfile->Printf("\n\n         ===> LIVVO Analysis of Particle Orbital <===");
outfile->Printf("\n   "
                "============================================================"
                "===============");
outfile->Printf("\n    %%Contribution    MO Character           IAO Contributions");
outfile->Printf("\n   "
                "============================================================"
                "===============");
std::string total_string;
for (auto occ_vvo : pair_occ_vvo_part) {
    double overlap_iao = 0.0;
    bool is_sigma = false;
    total_string = boost::str(boost::format(""));
    std::vector<std::pair<double, std::string>> iao_cont;
    // outfile->Printf("\n     %d                       %.1f%%",occ_vvo.second,
    // occ_vvo.first*100.0);
    for (auto& i : keys) {
        auto& ifn = atom_am_to_f[i];
        for (auto& iao : ifn) {
            overlap_iao = std::pow(ASLvvo->get(iao, occ_vvo.second - 1), 2.0);
            // std::string outstr = boost::str(boost::format("%.3f_%s") %
            // overlap_iao % (i.get<0>() + 1) %
            // molecule_->symbol(i.get<0>()).c_str()  %
            // l_to_symbol[i.get<1>()].c_str());
            std::string check_string = iao_labels[iao].c_str();
            // outfile->Printf("%s \n", check_string.c_str());
            // if(check_string.find('s') != std::string::npos){
            //	is_sigma = true;
            //}
            std::string outstr =
                boost::str(boost::format("%.2f_%s +") % overlap_iao % iao_labels[iao].c_str());
            if (overlap_iao >= 0.01) {
                iao_cont.push_back(std::make_pair(overlap_iao, outstr));
                total_string.append(outstr.c_str());
            }
        }
    }
    std::sort(iao_cont.begin(), iao_cont.end());
    std::reverse(iao_cont.begin(), iao_cont.end());
    std::string current_string = total_string.c_str();
    current_string.pop_back();
    if (current_string.find('s') != std::string::npos) {
        is_sigma = true;
    }
    if (occ_vvo.first > 0.001) {
        if (is_sigma == true) {
            outfile->Printf("\n          %.1f%%          \u03C3*            %s",
                            occ_vvo.first * 100.0, current_string.c_str());
        } else if (is_sigma == false) {
            outfile->Printf("\n          %.1f%%          \u03C0*            %s",
                            occ_vvo.first * 100.0, current_string.c_str());
        }
    }
    valence_sum += occ_vvo.first * 100.0;
}
outfile->Printf("\n                  -------------------------------------");
outfile->Printf("\n                    Total Valence Character: %.1f%%    ", valence_sum);
outfile->Printf("\n                  -------------------------------------");
outfile->Printf("\n   "
                "============================================================"
                "==============\n\n");
std::sort(pair_occ_iao_part.begin(), pair_occ_iao_part.end());
// C_p_cont->print();
// CubeProperties cube = CubeProperties(wfn_);
if (state_ == 1) {
    std::vector<int> indsp0;
    std::vector<std::string> labelsp;
    std::vector<int> indsp0_occ;
    std::vector<std::string> labelsp_occ;
    for (int i = 0; i < nvvos; ++i) {
        indsp0.push_back(i + nocc);
    }
    for (int i = 0; i < nocc; ++i) {
        indsp0_occ.push_back(i);
    }
    for (int i = 0; i < nvvos; ++i) {
        labelsp.push_back("part_iao");
    }
    for (int i = 0; i < nocc; ++i) {
        labelsp_occ.push_back("part_iao");
    }
    cube.compute_orbitals(VVO_Space_Matrix, indsp0, labelsp, "vvo");
    cube.compute_orbitals(Occ_Space_Matrix, indsp0_occ, labelsp_occ, "occ");
}
std::reverse(pair_occ_iao_part.begin(), pair_occ_iao_part.end());
outfile->Printf("\n Particle: ");
for (auto occ_iao : pair_occ_iao_part) {
    outfile->Printf("|%.1f%% %s|", occ_iao.first * 100.0, occ_iao.second.c_str());
}

outfile->Printf("\n\n  Analysis of the hole/particle MOs in terms of the "
                "ground state DFT MOs");
SharedMatrix ChSCa(new Matrix(Ch_->colspi(), dets[temp_int_new]->Ca()->colspi()));
ChSCa->gemm(true, false, 1.0, Ch_, TempMatrix, 0.0);
for (int h = 0; h < nirrep_; ++h) {
    for (int p = 0; p < Ch_->colspi()[h]; ++p) {
        double sum = 0.0;
        for (int q = 0; q < Ch_->rowspi()[h]; ++q) {
            sum += std::fabs(Ch_->get(h, q, p));
        }
        if (sum > 1.0e-3) {
            std::vector<std::pair<double, int>> pair_occ_mo;
            for (int q = 0; q < Ca_->colspi()[h]; ++q) {
                pair_occ_mo.push_back(
                    std::make_pair(std::pow(ChSCa->get(h, p, q), 2.0), q + 1));
            }
            std::sort(pair_occ_mo.begin(), pair_occ_mo.end());
            std::reverse(pair_occ_mo.begin(), pair_occ_mo.end());
            std::vector<std::string> vec_str;
            for (auto occ_mo : pair_occ_mo) {
                if (occ_mo.first > 0.01) {
                    vec_str.push_back(boost::str(boost::format(" %.1f%% %d%s") %
                                                 (occ_mo.first * 100.0) % occ_mo.second %
                                                 ct.gamma(h).symbol()));
                }
            }
            outfile->Printf("\n  Hole:     %6d%s = %s", p, ct.gamma(h).symbol(),
                            to_string(vec_str, " + ").c_str());
        }
    }
}

TempMatrix->gemm(false, false, 1.0, S_, dets[temp_int_new]->Ca(), 0.0);
SharedMatrix CpSCa(new Matrix(Cp_->colspi(), dets[temp_int_new]->Ca()->colspi()));
CpSCa->gemm(true, false, 1.0, Cp_, TempMatrix, 0.0);
for (int h = 0; h < nirrep_; ++h) {
    for (int p = 0; p < Cp_->colspi()[h]; ++p) {
        double sum = 0.0;
        for (int q = 0; q < Cp_->rowspi()[h]; ++q) {
            sum += std::fabs(Cp_->get(h, q, p));
        }
        if (sum > 1.0e-3) {
            std::vector<std::pair<double, int>> pair_occ_mo;
            for (int q = 0; q < Ca_->colspi()[h]; ++q) {
                pair_occ_mo.push_back(
                    std::make_pair(std::pow(CpSCa->get(h, p, q), 2.0), q + 1));
            }
            std::sort(pair_occ_mo.begin(), pair_occ_mo.end());
            std::reverse(pair_occ_mo.begin(), pair_occ_mo.end());
            std::vector<std::string> vec_str;
            for (auto occ_mo : pair_occ_mo) {
                if (occ_mo.first > 0.01) {
                    vec_str.push_back(boost::str(boost::format(" %.1f%% %d%s") %
                                                 (occ_mo.first * 100.0) % occ_mo.second %
                                                 ct.gamma(h).symbol()));
                }
            }
            outfile->Printf("\n  Particle: %6d%s = %s", p, ct.gamma(h).symbol(),
                            to_string(vec_str, " + ").c_str());
        }
    }
}
outfile->Printf("\n\n");
}
*/
}
} // Namespaces
