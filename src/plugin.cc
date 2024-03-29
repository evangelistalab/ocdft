#include <string>

#include "format.h"

#include "psi4/libpsi4util/PsiOutStream.h"
#include <psi4/physconst.h>
#include <psi4/psi4-dec.h>
#include <psi4/libcubeprop/cubeprop.h>
#include <psi4/libmints/oeprop.h>
#include <psi4/libmints/wavefunction.h>
#include <psi4/libmints/writer.h>
#include <psi4/libmints/writer_file_prefix.h>
#include <psi4/liboptions/liboptions.h>
#include <psi4/libscf_solver/hf.h>

#include "ocdft.h"
#include "ucks.h"

namespace psi {
namespace ocdft {

void CDFT(SharedWavefunction ref_wfn, Options& options);
void OCDFT(SharedWavefunction ref_wfn, Options& options);

extern "C" PSI_API int read_options(std::string name, Options& options) {
    if (name == "OCDFT" or options.read_globals()) {
        /*- Select the constrained DFT method.  The valid options are:
        ``CDFT`` Constrained DFT;
        ``OCDFT`` Constrained DFT;   Default is ``OCDFTHP``. -*/
        options.add_str("METHOD", "OCDFT", "OCDFT CDFT FASNOCIS NOCI");

        ////////////////////////////////////////
        // Options for Constrained DFT (CDFT) //
        ///////////////////////////////////////

        /*- Charge constraints -*/
        options.add("CONSTRAIN_CHARGE", new ArrayType());
        /*- Spin constraints -*/
        options.add("SPIN", new ArrayType());
        /*- Select the way the charges are computed -*/
        options.add_str("CONSTRAINT_TYPE", "LOWDIN", "LOWDIN");
        /*- Select the algorithm to optimize the constraints -*/
        options.add_str("W_ALGORITHM", "NEWTON", "NEWTON QUADRATIC");
        /*- The threshold for the gradient of the constraint -*/
        options.add_double("W_CONVERGENCE", 1.0e-5);
        /*- The Lagrange multiplier for the SUHF formalism -*/
        options.add_double("CDFT_SUHF_LAMBDA", 0.0);
        /*- Charge constraints -*/
        options.add_double("LEVEL_SHIFT", 0.0);
        /*- Apply a fixed Lagrange multiplier -*/
        options.add_bool("OPTIMIZE_VC", true);
        /*- Value of the Lagrange multiplier -*/
        options.add("VC", new ArrayType());

        ///////////////////////////////////////////////////////
        // Options for Orthogonality Constrained DFT (OCDFT) //
        ///////////////////////////////////////////////////////

        // Options for Specifying Excited States and Algorithms //
        /*- Number of excited states -*/
        options.add_int("NROOTS", 0);
        /*- Number of excited states per irrep, ROOTS_PER_IRREP has priority over
         * NROOTS -*/
        options.add("ROOTS_PER_IRREP", new ArrayType());
        /*- Perform a correction of the triplet excitation energies -*/
        options.add_bool("TRIPLET_CORRECTION", true);
        options.add_bool("VALENCE_TO_CORE", false);
        options.add_bool("FULL_MULLIKEN_PRINT", false);
        /*- Perform a correction of the triplet excitation energies using the S+
         * formalism -*/
        options.add_bool("CDFT_SPIN_ADAPT_SP", true);
        /*- Show analysis of hole and particle orbitals in terms of Intrinsic Bond
         * Orbitals -*/
        options.add_bool("IBO_ANALYSIS", false);
        /*- Perform a correction of the triplet excitation energies using a CI
         * formalism -*/
        options.add_bool("CDFT_SPIN_ADAPT_CI", false);
        /*- Break the symmetry of the HOMO/LUMO pair (works only in C1 symmetry) -*/
        options.add("CDFT_BREAK_SYMMETRY", new ArrayType());
        /*- Select the excited state method.  The valid options are:
        ``CP`` (constrained particle) which finds the optimal particle orbital
        while relaxing the other orbitals;
        ``CH`` (constrained hole) which finds the optimal hole orbital
        while relaxing the other orbitals;
        ``CHP`` (constrained hole/particle) which finds the optimal hole and
        particle orbitals while relaxing the other orbitals;
        ``CHP-F`` (frozen CHP) which is CHP without orbital relaxation.
        ``CHP-Fb`` (frozen beta CHP) which is CHP without beta orbital relaxation.
        Default is ``CHP``. -*/
        options.add_str("CDFT_EXC_METHOD", "CHP", "CP CH CHP CHP-F CHP-FB CIS");
        /*- An array of dimension equal to the number of irreps that allows to
         * select a given hole/particle excitation -*/
        options.add("CDFT_EXC_SELECT", new ArrayType());
        /*- An array of dimension equal to the number of irreps that allows to
         * select an excitation with given symmetry-*/
        options.add("CDFT_EXC_HOLE_SYMMETRY", new ArrayType());
        /*- Select the type of excited state to target -*/
        options.add_str("CDFT_EXC_TYPE", "VALENCE", "VALENCE CORE IP EA");
        /*- Select the type of excited state to target -*/
        options.add_str("CDFT_PROJECT_OUT", "H", "H P HP");
        /*- Select the type of excited state to target -*/
        options.add_int("CDFT_NUM_PROJECT_OUT", 1);
        /*- Select the maximum number of iterations in an OCDFT computation -*/
        options.add_int("OCDFT_MAX_ITER", 1000000);
        /*- The number of holes per irrep-*/
        options.add("OCDFT_HOLES_PER_IRREP", new ArrayType());
        /*- The number of particles per irrep-*/
        options.add("OCDFT_PARTS_PER_IRREP", new ArrayType());

        // Convergence Tools //
        /*- Maximum Overlap Method for convergence, specifies starting iteration -*/
        options.add_int("MOM_START", 0);
        /*- Use Damping for SCF Density in Excited States -*/
        options.add_double("DAMPING_PERCENTAGE", 0.0);
        /*- Select "Temperature" for use in Scuseria partial Fractional Occupation
         * Number Method (PFON) -*/
        options.add_double("PFON_TEMP", 0.0);

        // Selecting Specific Excitations (Expert)//
        // CAUTION: This must be done with great care, the orbitals you select must
        // be
        // sufficiently decoupled from other orbitals in the system (ex. localized
        // 1s orbs)
        // attempting to target other orbitals (ex. degenerate pi* valence orbitals)
        // may
        // result in convergence issues and/or variational collapse.
        /*- Orbital Energy Cutoff for selecting hole orbital -*/
        options.add_double("REW", 0.0);
        /*- Obtain Desired Hole AO Subspace from user -*/
        options.add("H_SUBSPACE", new ArrayType());
        /*- Obtain Desired Particle AO Subspace from user -*/
        options.add("P_SUBSPACE", new ArrayType());
        /**
        *    Valid options for P and H Subspace are of the following form:
        *
        *    ["C"] - all carbon atoms
        *    ["C","N"] - all carbon and nitrogen atoms
        *    ["C1"] - carbon atom #1
        *    ["C1-3"] - carbon atoms #1, #2, #3
        *    ["C(2p)"] - the 2p subset of all carbon atoms
        *    ["C(1s,2s)"] - the 1s/2s subsets of all carbon atoms
        *    ["C1-3(2s)"] - the 2s subsets of carbon atoms #1, #2, #3
        **/
        /*- Threshold for overlap of MOs with AO hole subspace-*/
        options.add_double("HOLE_THRESHOLD", 0.2);
        /*- Threshold for overlap of MOs with AO particle subspace-*/
        options.add_double("PARTICLE_THRESHOLD", 0.1);
        /*- Scale off-diagonal terms in CI spin adaptation -*/
        options.add_double("ALPHA_CI", 1.0);
        /*- Print cube files for the particle and hole orbitals -*/
        //  Cube files are EXTREMELY large files and thus this should be done
        //  carefully. Even small molecules produce cube files that are 5-8 MB each.
        //  Ensure that your computer has enough memory before printing cube files.
        options.add_bool("CUBE_HP", false);

        /*- Enable analysis of excitations -*/
        options.add_bool("ANALYZE_EXCITATIONS", false);

        options.add_str("MINAO_BASIS", "CC-PVTZ-MINAO");

        /////////////////////////////////////////////////////////////////
        // Options for Non-Orthogonal Configuration Interaction (NOCI) //
        /////////////////////////////////////////////////////////////////

        options.add_bool("USE_FAST_JK", false);

        options.add_bool("REF_MIX", true);

        options.add_bool("DIAG_DFT_E", false);

        options.add("AOCC_FROZEN", new ArrayType());
        options.add("AVIR_FROZEN", new ArrayType());

        /*- Would you like to perform an NOCI calculation as well? -*/
        options.add_bool("DO_NOCI_AND_OCDFT", false);

        /*- Would you like to perform an NOCI calculation as well? -*/
        options.add_bool("DO_NOCI_AND_OCDFT", false);

        /*- TODOPRAKASH: add description -*/
        options.add("OCC_FROZEN", new ArrayType());
        options.add("VIR_FROZEN", new ArrayType());

        // Expert options
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
    }
    return true;
}

extern "C" PSI_API SharedWavefunction ocdft(SharedWavefunction ref_wfn, Options& options) {
if (options.get_str("METHOD") == "OCDFT") {
        outfile->Printf("\n  ==> Orthogonality Constrained DFT <==\n");
        OCDFT(ref_wfn, options);
    }  
    // if (options.get_str("METHOD") == "CDFT") {
    //     // outfile->Printf("\n  ==> Constrained DFT <==\n");
    //     // CDFT(ref_wfn, options);
    // } else if (options.get_str("METHOD") == "OCDFT") {
    //     outfile->Printf("\n  ==> Orthogonality Constrained DFT <==\n");
    //     OCDFT(ref_wfn, options);
    // } else if (options.get_str("METHOD") == "FASNOCIS") {
    //     // outfile->Printf("\n  ==> Frozen-active-space NOCI Singles <==\n");
    //     // FASNOCIS(ref_wfn, options);
    // } else if (options.get_str("METHOD") == "NOCI") {
    //     // outfile->Printf("\n  ==> NON-Orthogonality CI <==\n");
    //     // NOCI(ref_wfn, options);
    // }

    // Set some environment variables
    //    Process::environment.globals["SCF TOTAL ENERGY"] = energies.back();
    //    Process::environment.globals["CURRENT ENERGY"] = energies.back();
    //    Process::environment.globals["CURRENT REFERENCE ENERGY"] = energies[0];
    return ref_wfn;
}

void OCDFT(SharedWavefunction ref_wfn, Options& options) {
    std::shared_ptr<PSIO> psio = PSIO::shared_object();

    std::string reference = options.get_str("REFERENCE");
    std::vector<double> energies;
    std::vector<SharedDeterminant> dets;
    // Store the irrep, multiplicity, total energy, excitation energy, oscillator
    // strength
    std::vector<std::tuple<int, int, double, double, double, double, double, double>> state_info;

    scf::HF* hf_obj = dynamic_cast<scf::HF*>(ref_wfn.get());
    std::shared_ptr<SuperFunctional> functional = hf_obj->functional();

    if (reference == "RKS") {
        throw InputException("Constrained RKS is not implemented ", "REFERENCE to UKS", __FILE__,
                             __LINE__);
    } else if (reference == "UKS") {
        // Run a ground state computation first
        SharedWavefunction ref_scf =
            SharedWavefunction(new scf::UOCDFT(ref_wfn, functional, options, psio));
        double gs_energy = ref_scf->compute_energy();
        SharedMatrix Ca_solab = ref_scf->Ca();
        SharedMatrix Cb_solab = ref_scf->Cb();
        energies.push_back(gs_energy);
        dets.push_back(SharedDeterminant(new scf::Determinant(
            ref_scf->Ca(), ref_scf->Cb(), ref_scf->nalphapi(), ref_scf->nbetapi())));
        SharedMatrix GroundDens(ref_scf->Da()->clone());
        state_info.push_back(std::make_tuple(0, 1, gs_energy, 0.0, 0.0, 0.0, 0.0, 0.0));

        // Print a molden file
        if (options["MOLDEN_WRITE"].has_changed()) {
            std::shared_ptr<MoldenWriter> molden(new MoldenWriter(ref_scf));
            std::string filename = get_writer_file_prefix("OCDFT") + "." + to_string(0) + ".molden";
            psi::scf::HF* hf = (psi::scf::HF*)ref_scf.get();
            SharedVector occA = hf->occupation_a();
            SharedVector occB = hf->occupation_b();
            molden->write(filename, ref_scf->Ca(), ref_scf->Cb(), ref_scf->epsilon_a(),
                          ref_scf->epsilon_b(), occA, occB, true);
        }

        if (options["ROOTS_PER_IRREP"].has_changed() and options["NROOTS"].has_changed()) {
            throw InputException("NROOTS and ROOTS_PER_IRREP are simultaneously defined",
                                 "Please specify either NROOTS or ROOTS_PER_IRREP", __FILE__,
                                 __LINE__);
        }
        // Compute a number of excited states without specifying the symmetry
        if (options["NROOTS"].has_changed()) {
            int nstates = options["NROOTS"].to_integer();
            std::vector<std::string> labels;
            labels.push_back("placeholder");
            for (int state = 1; state <= nstates; ++state) {
                SharedWavefunction new_scf =
                    SharedWavefunction(new scf::UOCDFT(functional, options, psio, ref_scf, state));
                // Process::environment.wavefunction().reset();
                // Process::environment.set_wavefunction(new_scf);
                double new_energy = new_scf->compute_energy();
                SharedMatrix Ddiff(GroundDens->clone());
                Ddiff->subtract(new_scf->Da());
                CubeProperties cube = CubeProperties(new_scf);
                if (options.get_bool("CUBE_HP")) {
                    std::string label_str = fmt::format("Ddiff{:d}", state);
                    labels.push_back(label_str);
                    cube.compute_density(Ddiff, label_str);
                }
                // Ddiff->print();
                energies.push_back(new_energy);
                if (options.get_bool("DIAG_DFT_E")) {
                    energies.push_back(new_energy);
                }
                dets.push_back(SharedDeterminant(new scf::Determinant(
                    new_scf->Ca(), new_scf->Cb(), new_scf->nalphapi(), new_scf->nbetapi())));
                dets.push_back(SharedDeterminant(new scf::Determinant(
                    new_scf->Cb(), new_scf->Ca(), new_scf->nbetapi(), new_scf->nalphapi())));

                scf::UOCDFT* uocdft_scf = dynamic_cast<scf::UOCDFT*>(new_scf.get());
                double singlet_exc_energy_s_plus = uocdft_scf->singlet_exc_energy_s_plus();
                double oscillator_strength_s_plus = uocdft_scf->oscillator_strength_s_plus();
                double oscillator_strength_s_plus_x = uocdft_scf->oscillator_strength_s_plus_x();
                double oscillator_strength_s_plus_y = uocdft_scf->oscillator_strength_s_plus_y();
                double oscillator_strength_s_plus_z = uocdft_scf->oscillator_strength_s_plus_z();
                state_info.push_back(
                    std::make_tuple(state, 1, new_energy, singlet_exc_energy_s_plus,
                                    oscillator_strength_s_plus, oscillator_strength_s_plus_x,
                                    oscillator_strength_s_plus_y, oscillator_strength_s_plus_z));

                // Print a molden file
                if (options["MOLDEN_WRITE"].has_changed()) {
                    std::shared_ptr<MoldenWriter> molden(new MoldenWriter(new_scf));
                    std::string filename =
                        get_writer_file_prefix("OCDFT") + "." + to_string(state) + ".molden";
                    psi::scf::HF* hf = (psi::scf::HF*)new_scf.get();
                    SharedVector occA = hf->occupation_a();
                    SharedVector occB = hf->occupation_b();
                    molden->write(filename, new_scf->Ca(), new_scf->Cb(), new_scf->epsilon_a(),
                                  new_scf->epsilon_b(), occA, occB, true);
                }

                ref_scf = new_scf;
            }
        }
        // Compute a number of excited states of a given symmetry
        else if (options["ROOTS_PER_IRREP"].has_changed()) {
            int maxnirrep = ref_scf->nirrep();
            int nirrep = options["ROOTS_PER_IRREP"].size();
            if (nirrep != maxnirrep) {
                throw InputException(
                    "The number of irreps specified in the option ROOTS_PER_IRREP does "
                    "not match the number of irreps",
                    "Please specify a correct number of irreps in ROOTS_PER_IRREP", __FILE__,
                    __LINE__);
            }
            for (int h = 0; h < nirrep; ++h) {
                int nstates = options["ROOTS_PER_IRREP"][h].to_integer();
                if (nstates > 0) {
                    outfile->Printf("\n\n  ==== Computing %d state%s of symmetry %d ====\n",
                                    nstates, nstates > 1 ? "s" : "", h);
                }
                int hole_num = 0;
                int part_num = -1;
                for (int state = 1; state <= nstates; ++state) {
                    part_num += 1;
                    SharedWavefunction new_scf = SharedWavefunction(
                        new scf::UOCDFT(functional, options, psio, ref_scf, state, h));
                    // Process::environment.wavefunction().reset();
                    // Process::environment.set_wavefunction(new_scf);
                    double new_energy = new_scf->compute_energy();
                    energies.push_back(new_energy);

                    scf::UOCDFT* uocdft_scf = dynamic_cast<scf::UOCDFT*>(new_scf.get());
                    double singlet_exc_energy_s_plus = uocdft_scf->singlet_exc_energy_s_plus();
                    double oscillator_strength_s_plus = uocdft_scf->oscillator_strength_s_plus();
                    double oscillator_strength_s_plus_x =
                        uocdft_scf->oscillator_strength_s_plus_x();
                    double oscillator_strength_s_plus_y =
                        uocdft_scf->oscillator_strength_s_plus_y();
                    double oscillator_strength_s_plus_z =
                        uocdft_scf->oscillator_strength_s_plus_z();
                    state_info.push_back(std::make_tuple(
                        state, 1, new_energy, singlet_exc_energy_s_plus, oscillator_strength_s_plus,
                        oscillator_strength_s_plus_x, oscillator_strength_s_plus_y,
                        oscillator_strength_s_plus_z));
                    // Print a molden file
                    if (options.get_bool("MOLDEN_WRITE")) {
                        std::shared_ptr<MoldenWriter> molden(new MoldenWriter(new_scf));
                        std::string filename = get_writer_file_prefix("OCDFT") + "." +
                                               to_string(h) + "." + to_string(state) + ".molden";
                        psi::scf::HF* hf = (psi::scf::HF*)new_scf.get();
                        SharedVector occA = hf->occupation_a();
                        SharedVector occB = hf->occupation_b();
                        molden->write(filename, new_scf->Ca(), new_scf->Cb(), new_scf->epsilon_a(),
                                      new_scf->epsilon_b(), occA, occB, true);
                    }
                    ref_scf = new_scf;
                    if (part_num > hole_num) {
                        hole_num = part_num;
                        part_num = -1;
                    }
                }
            }
        }
    } else {
        throw InputException("Unknown reference " + reference, "REFERENCE", __FILE__, __LINE__);
    }
    outfile->Printf("\n       ==> OCDFT Excited State Information <==\n");

    outfile->Printf("\n    "
                    "------------------------------------------------------------"
                    "----------------------------------------------");
    outfile->Printf("\n      State       Energy (Eh)    Omega (eV)   Osc. Str.   "
                    "  Osc. Str. (x)  Osc. Str. (y)  Osc. Str. (z) ");
    outfile->Printf("\n    "
                    "------------------------------------------------------------"
                    "----------------------------------------------");
    for (size_t n = 0; n < state_info.size(); ++n) {
        double singlet_exc_en = std::get<3>(state_info[n]);
        double osc_strength = std::get<4>(state_info[n]);
        double osc_strength_x = std::get<5>(state_info[n]);
        double osc_strength_y = std::get<6>(state_info[n]);
        double osc_strength_z = std::get<7>(state_info[n]);
        if (n == 0) {
            outfile->Printf("\n     _OCDFT-%-3d %13.7f %11.4f   %12.8f   %12.8f   "
                            "%12.8f   %12.8f (Ground State)",
                            n, energies[n], (singlet_exc_en)*pc_hartree2ev, osc_strength,
                            osc_strength_x, osc_strength_y, osc_strength_z);
        } else if (options["VALENCE_TO_CORE"].has_changed() and n == 1) {
            outfile->Printf("\n     _OCDFT-%-3d %13.7f %11.4f   %12.8f   %12.8f   "
                            "%12.8f   %12.8f (Intermediate State)",
                            n, energies[n], (singlet_exc_en)*pc_hartree2ev, osc_strength,
                            osc_strength_x, osc_strength_y, osc_strength_z);
            outfile->Printf("\n                                    =====> X-Ray "
                            "Emission Spectrum <=====                                "
                            "        ");
        } else {
            outfile->Printf("\n     @OCDFT-%-3d %13.7f %11.4f   %12.8f   %12.8f   "
                            "%12.8f   %12.8f",
                            n - 1, energies[n], (singlet_exc_en)*pc_hartree2ev, osc_strength,
                            osc_strength_x, osc_strength_y, osc_strength_z);
        }
    }
    outfile->Printf("\n    "
                    "------------------------------------------------------------"
                    "-----------------------------------------------\n");
    //    if (options["DO_NOCI_AND_OCDFT"].has_changed()) {
    //        scf::NOCI_Hamiltonian noci_H(options, dets);
    //        noci_H.compute_energy(energies);
    //    }
    // Set this early because the callback mechanism uses it.
    // Process::environment.wavefunction().reset();
}

void CDFT(SharedWavefunction ref_wfn, Options& options) {
    std::string reference = options.get_str("REFERENCE");
    if (reference == "RKS") {
        // std::shared_ptr<PSIO> psio = PSIO::shared_object();

        // SharedWavefunction ref_scf = SharedWavefunction(new scf::RCKS(ref_wfn, options, psio));
        // double gs_energy = ref_scf->compute_energy();
        throw std::runtime_error("CDFT is not implemented for RKS references");
    } 
    // else 
    if (reference == "UKS") {
        // Run a ground state computation first
        std::shared_ptr<PSIO> psio = PSIO::shared_object();

        SharedWavefunction ref_scf = SharedWavefunction(new scf::UCKS(ref_wfn, options, psio));
        std::shared_ptr<OEProp> oe(new OEProp(ref_scf));
        oe->set_Da_so(ref_scf->Da());
        oe->add("MULLIKEN_CHARGES");
        oe->add("LOWDIN_CHARGES");
        oe->compute();
        double gs_energy = ref_scf->compute_energy();
        // std::shared_ptr<OEProp> oe(new OEProp(ref_scf));
        oe->set_Da_so(ref_scf->Da());
        oe->add("MULLIKEN_CHARGES");
        oe->add("LOWDIN_CHARGES");
        oe->compute();
        // If requested, write a molden file
        if (options["MOLDEN_WRITE"].has_changed()) {
            std::shared_ptr<MoldenWriter> molden(new MoldenWriter(ref_scf));
            std::string filename = get_writer_file_prefix("OCDFT") + "." + to_string(0) + ".molden";
            psi::scf::HF* hf = (psi::scf::HF*)ref_scf.get();
            SharedVector occA = hf->occupation_a();
            SharedVector occB = hf->occupation_b();
            molden->write(filename, ref_scf->Ca(), ref_scf->Cb(), ref_scf->epsilon_a(),
                          ref_scf->epsilon_b(), occA, occB, true);
        }
    }
}

}
} // End namespaces
