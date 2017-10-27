#include <numeric>

#include "psi4/libpsi4util/PsiOutStream.h"
#include <psi4/libmints/basisset.h>
#include <psi4/libmints/vector.h>
#include <psi4/libmints/molecule.h>
#include <psi4/libcubeprop/cubeprop.h>

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
    //    sigma->print();

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
    //    Cvvo->print();

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

std::vector<std::tuple<int, int, int>> UOCDFT::analyze_basis_set() {
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
    return atom_n_l;
}
std::vector<std::tuple<int, int, std::string>> UOCDFT::find_iao_character(SharedMatrix Ciao) {
    // Find the value of (atom,n,l) for each basis function
    auto atom_n_l = analyze_basis_set();

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

    // Compute overlap of particle orbitals with LIVVOs
    SharedMatrix particle_livvo_overlap = Matrix::triplet(Cmo, S_, Clivvo, true, false, false);
    // Compute overlap of IAOs with LIVVOs
    SharedMatrix livvo_iao_overlap = Matrix::triplet(Clivvo, S_, Ciao, true, false, false);

    outfile->Printf("\n\n =====> LIVVO Population Analysis <=====\n");
    int niao = Ciao->ncol();
    int nlivvo = Clivvo->ncol();
    for (int l = 0; l < nlivvo; l++) {
        double omega_p_l = std::pow(particle_livvo_overlap->get(0, l), 2.0);
        std::vector<double> l_composition(10, 0.0);
        // Save info only if this LIVVO has |<psi_p|psi_LIVVO>|^2 >= 0.01
        if (omega_p_l >= 0.01) {
            outfile->Printf("\n  |<psi_p|psi_LIVVO(%2d)>|^2 = %7.2f%%\n", l, omega_p_l * 100.0);
            // Compute contributions from each IAO to this LIVVO
            std::vector<std::tuple<double, std::string, int, int>> contributions;
            for (int rho = 0; rho < niao; rho++) {
                double S_l_rho = std::pow(livvo_iao_overlap->get(l, rho), 2.0);
                int character_A = std::get<0>(IAO_character[rho]);
                int character_l = std::get<1>(IAO_character[rho]);
                std::string character_symbol = std::get<2>(IAO_character[rho]);
                if (S_l_rho > 0.1) {
                    contributions.push_back(
                        std::make_tuple(S_l_rho, character_symbol, character_A, character_l));
                    l_composition[character_l] += S_l_rho;
                }
            }

            // Determine if this is a sigma or pi LIVVO
            for (int character_l = 0; character_l < 3; character_l++) {
                outfile->Printf(" %7.2f%% %s", l_composition[character_l] * 100.0,
                                l_to_symbol[character_l].c_str());
            }
            int type = 0;
            if (l_composition[1] / l_composition[0] >= 3.0) {
                type = 1;
            }
            outfile->Printf("   -> %s* :", type == 0 ? "sigma" : "pi");

            // Print important contributions sorted
            std::sort(contributions.rbegin(), contributions.rend());
            for (const auto& contribution : contributions) {
                double S_l_rho;
                std::string character_symbol;
                int character_A, character_l;
                std::tie(S_l_rho, character_symbol, character_A, character_l) = contribution;
                outfile->Printf("  %7.2f%% %s%d(%s)", 100.0 * S_l_rho, character_symbol.c_str(),
                                character_A + 1, l_to_symbol[character_l].c_str());
            }
        }
    }
    outfile->Printf("\n\n =======================================\n");
}
}
} // Namespaces
