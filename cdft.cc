#include <psi4-dec.h>
#include <physconst.h>

#include <libplugin/plugin.h>
#include <liboptions/liboptions.h>
#include <libmints/mints.h>
#include <libpsio/psio.hpp>
#include <libciomr/libciomr.h>
#include <libscf_solver/hf.h>

#include <libmints/wavefunction.h>
#include <libmints/writer.h>
#include <libmints/writer_file_prefix.h>
#include<vector>

#include "ucks.h"
#include "ocdft.h"

#include "noci.h"
#include "noci_mat.h"
#include "determinant.h"

#include "fasnocis.h"


INIT_PLUGIN

using namespace boost;

namespace psi{ namespace cdft {

void CDFT(Options& options);
void OCDFT(Options& options);
void FASNOCIS(Options& options);
void NOCI(Options& options);

extern "C"
int read_options(std::string name, Options& options)
{
    if (name == "CDFT" or options.read_globals()) {
        /*- Select the constrained DFT method.  The valid options are:
        ``CDFT`` Constrained DFT;
        ``OCDFT`` Constrained DFT;   Default is ``OCDFTHP``. -*/
        options.add_str("METHOD","OCDFT", "OCDFT CDFT FASNOCIS NOCI");

        // Options for Constrained DFT (CDFT)

        /*- Charge constraints -*/
        options.add("CHARGE", new ArrayType());
        /*- Spin constraints -*/
        options.add("SPIN", new ArrayType());
        /*- Select the way the charges are computed -*/
        options.add_str("CONSTRAINT_TYPE","LOWDIN", "LOWDIN");
        /*- Select the algorithm to optimize the constraints -*/
        options.add_str("W_ALGORITHM","NEWTON","NEWTON QUADRATIC");
        /*- The threshold for the gradient of the constraint -*/
        options.add_double("W_CONVERGENCE",1.0e-5);
        /*- The Lagrange multiplier for the SUHF formalism -*/
        options.add_double("CDFT_SUHF_LAMBDA",0.0);
        /*- Charge constraints -*/
        options.add_double("LEVEL_SHIFT",0.0);
	/*- Restrict Core -*/
	options.add_double("REW", 0.0);
        /*- Apply a fixed Lagrange multiplier -*/
        options.add_bool("OPTIMIZE_VC", true);
        /*- Value of the Lagrange multiplier -*/
        options.add("VC", new ArrayType());


        // Options for Orthogonality Constrained DFT (OCDFT)

        /*- Number of excited states -*/
        options.add_int("NROOTS", 0);
        /*- Number of excited states per irrep, ROOTS_PER_IRREP has priority over NROOTS -*/
        options.add("ROOTS_PER_IRREP", new ArrayType());
        /*- Perform a correction of the triplet excitation energies -*/
        options.add_bool("TRIPLET_CORRECTION", true);
        /*- Perform a correction of the triplet excitation energies using the S+ formalism -*/
        options.add_bool("CDFT_SPIN_ADAPT_SP", true);
        /*- Perform a correction of the triplet excitation energies using a CI formalism -*/
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
        ``CHP-Fb`` (frozen beta CHP) which is CHP without beta orbital relaxation. Default is ``CHP``. -*/
        options.add_str("CDFT_EXC_METHOD","CHP","CP CH CHP CHP-F CHP-FB CIS");
        /*- An array of dimension equal to then number of irreps that allows to select a given hole/particle excitation -*/
        options.add("CDFT_EXC_SELECT", new ArrayType());
        /*- An array of dimension equal to then number of irreps that allows to select an excitation with given symmetry-*/
        options.add("CDFT_EXC_HOLE_SYMMETRY", new ArrayType());
        /*- Select the type of excited state to target -*/
        options.add_str("CDFT_EXC_TYPE","VALENCE","VALENCE CORE IP EA");
        /*- Select the type of excited state to target -*/
        options.add_str("CDFT_PROJECT_OUT","H","H P HP");
        /*- Select the type of excited state to target -*/
        options.add_int("CDFT_NUM_PROJECT_OUT",1);

        options.add_bool("USE_FAST_JK", false);

        options.add("AOCC_FROZEN", new ArrayType());
        options.add("AVIR_FROZEN", new ArrayType());

        /*- Select the maximum number of iterations in an OCDFT computation -*/
        options.add_int("OCDFT_MAX_ITER",1000000);


        /*- TODOPRAKASH: add description -*/
        options.add("OCC_FROZEN", new ArrayType());
        options.add("VIR_FROZEN", new ArrayType());


        // Expert options
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
    }
    return true;
}

extern "C"
PsiReturnType cdft(Options& options)
{
    if (options.get_str("METHOD") == "CDFT"){
        outfile->Printf("\n  ==> Constrained DFT <==\n");
        CDFT(options);
    }else if (options.get_str("METHOD") == "OCDFT"){
        outfile->Printf("\n  ==> Orthogonality Constrained DFT <==\n");
        OCDFT(options);
    }else if (options.get_str("METHOD") == "FASNOCIS"){
        outfile->Printf("\n  ==> Frozen-active-space NOCI Singles <==\n");
        FASNOCIS(options);
    }else if (options.get_str("METHOD") == "NOCI"){
        outfile->Printf("\n  ==> NON-Orthogonality CI <==\n");
        NOCI(options);
    }

    // Set some environment variables
    //    Process::environment.globals["SCF TOTAL ENERGY"] = energies.back();
    //    Process::environment.globals["CURRENT ENERGY"] = energies.back();
    //    Process::environment.globals["CURRENT REFERENCE ENERGY"] = energies[0];
    return Success;
}

void CDFT(Options& options)
{
    std::string reference = options.get_str("REFERENCE");

    if (reference == "RKS") {
        throw InputException("Constrained RKS is not implemented ", "REFERENCE to UKS", __FILE__, __LINE__);
    }else if (reference == "UKS") {
        // Run a ground state computation first
        boost::shared_ptr<PSIO> psio = PSIO::shared_object();
        boost::shared_ptr<Wavefunction> ref_scf(new scf::UCKS(options, psio));
        Process::environment.set_wavefunction(ref_scf);
        double gs_energy = ref_scf->compute_energy();

        // If requested, write a molden file
        if ( options["MOLDEN_WRITE"].has_changed() ) {
            boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(ref_scf));
            std::string filename = get_writer_file_prefix() + "." + to_string(0) + ".molden";
            psi::scf::HF* hf = (psi::scf::HF*)ref_scf.get();
            SharedVector occA = hf->occupation_a();
            SharedVector occB = hf->occupation_b();
            molden->write(filename,ref_scf->Ca(),ref_scf->Cb(),ref_scf->epsilon_a(),ref_scf->epsilon_b(),occA,occB);
        }
    }
}


void NOCI(Options& options)
{
    boost::shared_ptr<PSIO> psio = PSIO::shared_object();
    boost::shared_ptr<Wavefunction> ref_scf;
    std::string reference = options.get_str("REFERENCE");
    std::vector<double> energies;
    std::vector<SharedDeterminant> dets;
    // Store the irrep, multiplicity, total energy, excitation energy, oscillator strength
    std::vector<boost::tuple<int,int,double,double,double>> state_info;
    if (reference == "RHF") {
        throw InputException("NOCI based on a RHF reference is not implemented ", "REFERENCE to UHF", __FILE__, __LINE__);
    }else if (reference == "UHF") {
        // Run a ground state computation first
        outfile->Printf(" PV this is first done.\n");
        ref_scf = boost::shared_ptr<Wavefunction>(new scf::NOCI(options, psio));
        Process::environment.set_wavefunction(ref_scf);
        double gs_energy = ref_scf->compute_energy();
        outfile->Printf("\n  %11.4f",gs_energy);
        energies.push_back(gs_energy);

        // Push the ground state determinant
        dets.push_back(SharedDeterminant(new scf::Determinant(ref_scf->Ca(),ref_scf->Cb(),ref_scf->nalphapi(),ref_scf->nbetapi())));

        // I am going to ask user to give me
        //  OCC_ACTIVE based on each irrep
        //  VIR_ACTIVE based on each irrep

        int nirrep = Process::environment.wavefunction()->nirrep();
        std::vector<int> occ_frozen, vir_frozen;
        std::vector<boost::tuple<int,int,double>> occup_a;


        for (int h = 0; h < nirrep; ++h){
            occ_frozen.push_back(options["AOCC_FROZEN"][h].to_integer());
            vir_frozen.push_back(options["AVIR_FROZEN"][h].to_integer());
        }

        for (auto &i : occ_frozen){
            outfile->Printf("\n  occ_frozen = %d",i);
        }

        // find out how many are occupied alpha and beta
        Dimension nalphapi = ref_scf->nalphapi();
        Dimension nbetapi = ref_scf->nbetapi();

        Dimension nsopi_ = ref_scf->nsopi();
        Dimension nmopi_ = ref_scf->nmopi();

        // need to know my active mos based on each irrep
        // active_mos have (irrep, mo_number_info)

        std::vector<std::pair<int,int>> frozen_mos;
        std::vector<std::pair<int,int>> frozen_occ_a;
        std::vector<std::pair<int,int>> frozen_occ_b;

        for (int h = 0; h < nirrep; ++h){
            for (int i = 0; i < occ_frozen[h]; ++i){
                frozen_mos.push_back(std::make_pair(h,nalphapi[h] - 1 - i));
                frozen_occ_a.push_back(std::make_pair(h,nalphapi[h] - 1 - i));
                frozen_occ_b.push_back(std::make_pair(h,nbetapi[h] - 1 - i));
            }
        }


        for (int h = 0; h < nirrep; ++h){
            for (int i = 0; i < vir_frozen[h]; ++i){
                frozen_mos.push_back(std::make_pair(h,nalphapi[h] + i));
            }
        }


        for (auto &h_p : frozen_mos){
            outfile->Printf("\n  ireep = %d mo = %d",h_p.first,h_p.second);

        }

        outfile->Printf("\n not sure abott this %d",frozen_mos[1].second);




        //       for (int h = 0; h < nirrep; ++h){
        //            outfile->Printf("\n  irrep = %d Nalpha %d FNa = %d",h,nalphapi[h],occ_frozen[h]);
        //           for (int i = 0; i < nmopi_[h]; ++i){

        //               if (i < nalphapi[h] - occ_frozen[h]){
        //                  occup_a.push_back(boost::make_tuple(h,i,1.0));
        //               }
        //               else if (i >= nalphapi[h] - occ_frozen[h] && i < nalphapi[h] ){
        //                   occup_a.push_back(boost::make_tuple(h,i,0.0));
        //                }
        //               else if (i >= nalphapi[h] - occ_frozen[h] && i < nalphapi[h] ){
        //                                  occup_a.push_back(boost::make_tuple(h,i,0.0));
        //                               }
        //               else{
        //                   occup_a.push_back(boost::make_tuple(h,i,0.0));
        //               }
        //           }
        //       }

        for (size_t n = 0; n < occup_a.size(); ++n){
            outfile->Printf("\n  occup_a = %d %d mo = %f\n",occup_a[n].get<0>(),occup_a[n].get<1>(),occup_a[n].get<2>());

        }


        state_info.push_back(boost::make_tuple(0,1,gs_energy,0.0,0.0));

        SharedMatrix Ca_gs_;
        SharedMatrix Cb_gs_;


        Ca_gs_ =  SharedMatrix(new Matrix("Ca_gs_",nsopi_,nmopi_));
        Cb_gs_ =  SharedMatrix(new Matrix("Cb_gs_",nsopi_,nmopi_));

        Ca_gs_->copy(ref_scf->Ca());
        Cb_gs_->copy(ref_scf->Cb());

        int nstates = 0;
        for (int h = 0; h < nirrep; ++h){
            nstates +=occ_frozen[h]*vir_frozen[h];
        }

        outfile->Printf("\n no. of states involved single excitations only %d\n", nstates);


        //         for (auto &h_p : frozen_occ_a){
        //             int irrep = h_p.first;
        //             int fmo   = h_p.second;
        //          //   outfile->Printf("frozen_occ_a %d %d %d\n", irrep, fmo,vir_frozen[irrep],nalphapi);
        //         }

        for (auto &h_p : frozen_occ_a){
            int irrep = h_p.first;
            int fmo   = h_p.second;
            std::pair<int,int> swap_occ (irrep,fmo);

            //for (int h = 0; h < nirrep; ++h){
            //   for (int i=1; i<= occ_frozen[h]; ++i){

            for (int state_a=1; state_a <= vir_frozen[irrep];++state_a){
                int state_b=0;
                boost::shared_ptr<Wavefunction> new_scf = boost::shared_ptr<Wavefunction>(new scf::NOCI(options,psio,state_a,swap_occ,state_b,
                                                                                                        frozen_occ_a,frozen_occ_b,
                                                                                                        frozen_mos,
                                                                                                        occ_frozen,vir_frozen,
                                                                                                        Ca_gs_,Cb_gs_));
                Process::environment.wavefunction().reset();
                Process::environment.set_wavefunction(new_scf);
                double new_energy = new_scf->compute_energy();
                energies.push_back(new_energy);
                dets.push_back(SharedDeterminant(new scf::Determinant(new_scf->Ca(),new_scf->Cb(),new_scf->nalphapi(),new_scf->nbetapi())));
            }
            //   }//occup
        } //irrep

        scf::NOCI_Hamiltonian noci_H(options,dets);
        noci_H.compute_energy();
    }
}



void OCDFT(Options& options)
{
    boost::shared_ptr<PSIO> psio = PSIO::shared_object();
    boost::shared_ptr<Wavefunction> ref_scf;
    std::string reference = options.get_str("REFERENCE");
    std::vector<double> energies;
    // Store the irrep, multiplicity, total energy, excitation energy, oscillator strength
    std::vector<boost::tuple<int,int,double,double,double>> state_info;

    if (reference == "RKS") {
        throw InputException("Constrained RKS is not implemented ", "REFERENCE to UKS", __FILE__, __LINE__);
    }else if (reference == "UKS") {
        // Run a ground state computation first
        ref_scf = boost::shared_ptr<Wavefunction>(new scf::UOCDFT(options, psio));
        Process::environment.set_wavefunction(ref_scf);
        double gs_energy = ref_scf->compute_energy();

        energies.push_back(gs_energy);

        state_info.push_back(boost::make_tuple(0,1,gs_energy,0.0,0.0));

        // Print a molden file
        if ( options["MOLDEN_WRITE"].has_changed() ) {
            boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(ref_scf));
            std::string filename = get_writer_file_prefix() + "." + to_string(0) + ".molden";
            psi::scf::HF* hf = (psi::scf::HF*)ref_scf.get();
            SharedVector occA = hf->occupation_a();
            SharedVector occB = hf->occupation_b();
            molden->write(filename,ref_scf->Ca(),ref_scf->Cb(),ref_scf->epsilon_a(),ref_scf->epsilon_b(),occA,occB);
        }

        if(options["ROOTS_PER_IRREP"].has_changed() and options["NROOTS"].has_changed()){
            throw InputException("NROOTS and ROOTS_PER_IRREP are simultaneously defined", "Please specify either NROOTS or ROOTS_PER_IRREP", __FILE__, __LINE__);
        }
        // Compute a number of excited states without specifying the symmetry
        if(options["NROOTS"].has_changed()){
            int nstates = options["NROOTS"].to_integer();
            for(int state = 1; state <= nstates; ++state){
                boost::shared_ptr<Wavefunction> new_scf = boost::shared_ptr<Wavefunction>(new scf::UOCDFT(options,psio,ref_scf,state));
                Process::environment.wavefunction().reset();
                Process::environment.set_wavefunction(new_scf);
                double new_energy = new_scf->compute_energy();
                energies.push_back(new_energy);

                scf::UOCDFT* uocdft_scf = dynamic_cast<scf::UOCDFT*>(new_scf.get());
                double singlet_exc_energy_s_plus = uocdft_scf->singlet_exc_energy_s_plus();
                double oscillator_strength_s_plus = uocdft_scf->oscillator_strength_s_plus();
                state_info.push_back(boost::make_tuple(state,1,new_energy,singlet_exc_energy_s_plus,oscillator_strength_s_plus));

                // Print a molden file
                if ( options["MOLDEN_WRITE"].has_changed() ) {
                    boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(new_scf));
                    std::string filename = get_writer_file_prefix() + "." + to_string(state) + ".molden";
                    psi::scf::HF* hf = (psi::scf::HF*)new_scf.get();
                    SharedVector occA = hf->occupation_a();
                    SharedVector occB = hf->occupation_b();
                    molden->write(filename,new_scf->Ca(),new_scf->Cb(),new_scf->epsilon_a(),new_scf->epsilon_b(),occA,occB);
                }

                ref_scf = new_scf;
            }
        }
        // Compute a number of excited states of a given symmetry
        else if(options["ROOTS_PER_IRREP"].has_changed()){
            int maxnirrep = Process::environment.wavefunction()->nirrep();
            int nirrep = options["ROOTS_PER_IRREP"].size();
            if (nirrep != maxnirrep){
                throw InputException("The number of irreps specified in the option ROOTS_PER_IRREP does not match the number of irreps",
                                     "Please specify a correct number of irreps in ROOTS_PER_IRREP", __FILE__, __LINE__);
            }
            for (int h = 0; h < nirrep; ++h){
                int nstates = options["ROOTS_PER_IRREP"][h].to_integer();
                if (nstates > 0){
                    outfile->Printf("\n\n  ==== Computing %d state%s of symmetry %d ====\n",nstates,nstates > 1 ? "s" : "",h);
                }
                int hole_num =  0;
                int part_num = -1;
                for (int state = 1; state <= nstates; ++state){
                    part_num += 1;
                    boost::shared_ptr<Wavefunction> new_scf = boost::shared_ptr<Wavefunction>(new scf::UOCDFT(options,psio,ref_scf,state,h));
                    Process::environment.wavefunction().reset();
                    Process::environment.set_wavefunction(new_scf);
                    double new_energy = new_scf->compute_energy();
                    energies.push_back(new_energy);

                    scf::UOCDFT* uocdft_scf = dynamic_cast<scf::UOCDFT*>(new_scf.get());
                    double singlet_exc_energy_s_plus = uocdft_scf->singlet_exc_energy_s_plus();
                    double oscillator_strength_s_plus = uocdft_scf->oscillator_strength_s_plus();
                    state_info.push_back(boost::make_tuple(0,1,new_energy,singlet_exc_energy_s_plus,oscillator_strength_s_plus));

                    // Print a molden file
                    if ( options.get_bool("MOLDEN_WRITE") ) {
                        boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(new_scf));
                        std::string filename = get_writer_file_prefix() + "." + to_string(h) + "." + to_string(state) + ".molden";
                        psi::scf::HF* hf = (psi::scf::HF*)new_scf.get();
                        SharedVector occA = hf->occupation_a();
                        SharedVector occB = hf->occupation_b();
                        molden->write(filename,new_scf->Ca(),new_scf->Cb(),new_scf->epsilon_a(),new_scf->epsilon_b(),occA,occB);
                    }
                    ref_scf = new_scf;
                    if (part_num > hole_num){
                        hole_num = part_num;
                        part_num = -1;
                    }
                }
            }
        }
    }else {
        throw InputException("Unknown reference " + reference, "REFERENCE", __FILE__, __LINE__);
    }

    outfile->Printf("\n       ==> OCDFT Excited State Information <==\n");

    outfile->Printf("\n    ----------------------------------------------------");
    outfile->Printf("\n      State       Energy (Eh)    Omega (eV)   Osc. Str.");
    outfile->Printf("\n    ----------------------------------------------------");
    for (size_t n = 0; n < state_info.size(); ++n){
        double singlet_exc_en = state_info[n].get<3>();
        double osc_strength = state_info[n].get<4>();
        outfile->Printf("\n     @OCDFT-%-3d %13.7f %11.4f %11.4f",n,energies[n],(singlet_exc_en) * pc_hartree2ev,osc_strength);
    }
    outfile->Printf("\n    ----------------------------------------------------\n");

    // Set this early because the callback mechanism uses it.
    Process::environment.wavefunction().reset();
}


void FASNOCIS(Options& options)
{
    boost::shared_ptr<PSIO> psio = PSIO::shared_object();
    boost::shared_ptr<Wavefunction> ref_scf;
    std::string reference = options.get_str("REFERENCE");
    std::vector<double> energies;
    std::vector<SharedDeterminant> dets;

    // Store the irrep, multiplicity, total energy, excitation energy, oscillator strength
    std::vector<boost::tuple<int,int,double,double,double>> state_info;

    if (reference == "RHF") {
        throw InputException("Constrained RKS is not implemented ", "REFERENCE to UKS", __FILE__, __LINE__);
    }else if (reference == "UHF") {
        // Run a ground state computation first
        ref_scf = boost::shared_ptr<Wavefunction>(new scf::FASNOCIS(options, psio));
        Process::environment.set_wavefunction(ref_scf);
        double gs_energy = ref_scf->compute_energy();
        dets.push_back(SharedDeterminant(new scf::Determinant(ref_scf->Ca(),ref_scf->Cb(),ref_scf->nalphapi(),ref_scf->nbetapi())));


        // Print a molden file
        if ( options["MOLDEN_WRITE"].has_changed() ) {
            boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(ref_scf));
            std::string filename = get_writer_file_prefix() + "." + to_string(0) + ".molden";
            psi::scf::HF* hf = (psi::scf::HF*)ref_scf.get();
            SharedVector occA = hf->occupation_a();
            SharedVector occB = hf->occupation_b();
            molden->write(filename,ref_scf->Ca(),ref_scf->Cb(),ref_scf->epsilon_a(),ref_scf->epsilon_b(),occA,occB);
        }

        // Optimize the orbitals of each frozen configuration
        Dimension nalphapi = ref_scf->nalphapi();
        Dimension nbetapi = ref_scf->nbetapi();

        int nirrep = Process::environment.wavefunction()->nirrep();
        Dimension adoccpi(nirrep,"Number of doubly occupied active MOs per irrep");
        std::vector<int> occ_frozen, vir_frozen;
        size_t nofrz = 0;
        size_t nvfrz = 0;
        for (int h = 0; h < nirrep; ++h){
            size_t no_h = options["OCC_FROZEN"][h].to_integer();
            size_t nv_h = options["VIR_FROZEN"][h].to_integer();
            occ_frozen.push_back(no_h);
            vir_frozen.push_back(nv_h);
            nofrz += no_h;
            nvfrz += nv_h;
            adoccpi[h] = nalphapi[h] - no_h;
        }


        size_t nfrz = nofrz + nvfrz;

        std::vector<std::pair<size_t,size_t>> frozen_mos;
        for (int h = 0; h < nirrep; ++h){
            for (int i = 0; i < occ_frozen[h]; ++i){
                frozen_mos.push_back(std::make_pair(h,nalphapi[h] - 1 - i));
            }
        }
        for (int h = 0; h < nirrep; ++h){
            for (int i = 0; i < vir_frozen[h]; ++i){
                frozen_mos.push_back(std::make_pair(h,nalphapi[h] + i));
            }
        }

        for (auto &h_p : frozen_mos){
            outfile->Printf("\n  irrep = %d mo = %d",h_p.first,h_p.second);
        }

        outfile->Printf("\n  nofrz = %d nfrz = %d",nofrz,nfrz);


        // Create all alpha-alpha singly-excited determinants
        for (size_t i = 0; i < nofrz; ++i){
            for (size_t a = nofrz; a < nfrz; ++a){
                outfile->Printf("\n  ==> State %d -> %d\n",i,a);

                // Build the vector of frozen occupied orbitals
                std::vector<size_t> aocc, bocc;
                for (size_t j = 0; j < nofrz; ++j){
                    if (j != i) aocc.push_back(j);
                    bocc.push_back(j);
                }
                aocc.push_back(a);

                boost::shared_ptr<Wavefunction> new_scf =
                        boost::shared_ptr<Wavefunction>(
                            new scf::FASNOCIS(options,psio,ref_scf,frozen_mos,
                                              aocc,bocc,adoccpi));

                new_scf->compute_energy();
                dets.push_back(SharedDeterminant(new scf::Determinant(new_scf->Ca(),new_scf->Cb(),new_scf->nalphapi(),new_scf->nbetapi())));
            }
        }
        // Create all beta-beta singly-excited determinants
        for (size_t i = 0; i < nofrz; ++i){
            for (size_t a = nofrz; a < nfrz; ++a){
                outfile->Printf("\n  ==> State %d -> %d\n",i,a);

                // Build the vector of frozen occupied orbitals
                std::vector<size_t> aocc, bocc;
                for (size_t j = 0; j < nofrz; ++j){
                    if (j != i) bocc.push_back(j);
                    aocc.push_back(j);
                }
                bocc.push_back(a);

                boost::shared_ptr<Wavefunction> new_scf =
                        boost::shared_ptr<Wavefunction>(
                            new scf::FASNOCIS(options,psio,ref_scf,frozen_mos,
                                              aocc,bocc,adoccpi));

                new_scf->compute_energy();
                dets.push_back(SharedDeterminant(new scf::Determinant(new_scf->Ca(),new_scf->Cb(),new_scf->nalphapi(),new_scf->nbetapi())));
            }
        }
        scf::NOCI_Hamiltonian noci_H(options,dets);
        noci_H.compute_energy();
    }else {
        throw InputException("Unknown reference " + reference, "REFERENCE", __FILE__, __LINE__);
    }

    // Set this early because the callback mechanism uses it.
    Process::environment.wavefunction().reset();
}

}} // End namespaces
