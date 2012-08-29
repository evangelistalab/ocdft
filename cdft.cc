#include <libplugin/plugin.h>
#include <psi4-dec.h>
#include <libparallel/parallel.h>
#include <liboptions/liboptions.h>
#include <libmints/mints.h>
#include <libpsio/psio.hpp>
#include <libciomr/libciomr.h>
#include <ucks.h>

INIT_PLUGIN

using namespace boost;

namespace psi{ namespace cdft {

extern "C"
int read_options(std::string name, Options& options)
{
    if (name == "CDFT" or options.read_globals()) {
        /*- Charge constraints -*/
        options.add("CHARGE", new ArrayType());
        /*- Spin constraints -*/
        options.add("SPIN", new ArrayType());
        /*- Number of excited states -*/
        options.add_int("NROOTS", 0);
        /*- Number of excited states per irrep, ROOTS_PER_IRREP has priority over NROOTS -*/
        options.add("ROOTS_PER_IRREP", new ArrayType());
        /*- Perform a correction of the triplet excitation energies -*/
        options.add_bool("TRIPLET_CORRECTION", true);
        /*- Perform a correction of the triplet excitation energies -*/
        options.add_bool("CDFT_SPIN_ADAPT", true);
        /*- Select the way the charges are computed -*/
        options.add_str("CONSTRAINT_TYPE","LOWDIN", "LOWDIN");
        /*- Select the algorithm to optimize the constraints -*/
        options.add_str("W_ALGORITHM","NEWTON","NEWTON QUADRATIC");
        /*- Select the excited state method.  The valid options are:
        ``CP`` (constrained particle) which finds the optimal particle orbital
        while relaxing the other orbitals;
        ``CH`` (constrained hole) which finds the optimal hole orbital
        while relaxing the other orbitals;
        ``CHP`` (constrained hole/particle) which finds the optimal hole and
        particle orbitals while relaxing the other orbitals;
        ``CHP-F`` (frozen CHP) which is CHP without orbital relaxation.  Default is ``CHP``. -*/
        options.add_str("CDFT_EXC_METHOD","CH","CP CH CHP CHP-F");// CHP CHP-F");
        // CP Constrained particle: find the optimal particle without relaxing
        /*- Select the excited hole to target -*/
        options.add_str("CDFT_EXC_HOLE","VALENCE","VALENCE CORE");
        /*- The threshold for the gradient of the constraint -*/
        options.add_double("W_CONVERGENCE",1.0e-5);
        /*- The Lagrange multiplier for the SUHF formalism -*/
        options.add_double("CDFT_SUHF_LAMBDA",0.0);
        /*- Charge constraints -*/
        options.add_double("LEVEL_SHIFT",0.0);


        // Expert options
        /*- Apply a fixed Lagrange multiplier -*/
        options.add_bool("OPTIMIZE_VC", true);
        /*- Value of the Lagrange multiplier -*/
        options.add("VC", new ArrayType());
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
        /*- Compute an excited state by adding a penalty to the HOMO -*/
        options.add_bool("HOMO_PENALTY", false);
    }
    return true;
}

extern "C"
PsiReturnType cdft(Options& options)
{
    tstart();

    boost::shared_ptr<PSIO> psio = PSIO::shared_object();
    boost::shared_ptr<Wavefunction> ref_scf;
    std::string reference = options.get_str("REFERENCE");
    std::vector<double> energies;

    if (reference == "RKS") {
        throw InputException("Constrained RKS is not implemented ", "REFERENCE to UKS", __FILE__, __LINE__);
    }else if (reference == "UKS") {
        // Run a ground state computation first
        ref_scf = boost::shared_ptr<Wavefunction>(new scf::UCKS(options, psio));
        Process::environment.set_wavefunction(ref_scf);
        double gs_energy = ref_scf->compute_energy();
        // Print a molden file
        if ( options["MOLDEN_FILE"].has_changed() ) {
           boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(ref_scf));
           molden->write("0." + options.get_str("MOLDEN_FILE"));
        }
        energies.push_back(gs_energy);

        // Count the number of excited state computations
        int nexcited = 0;
        if(options["ROOTS_PER_IRREP"].has_changed() and options["NROOTS"].has_changed()){
            throw InputException("NROOTS and ROOTS_PER_IRREP are simultaneously defined", "Please specify either NROOTS or ROOTS_PER_IRREP", __FILE__, __LINE__);
        }else{
            nexcited = options["NROOTS"].to_integer();
        }
        int nholes = 0;
        int nparticles = 0;
        for(int state = 0; state < nexcited; ++state){
            boost::shared_ptr<scf::UCKS> ref_ucks = boost::shared_ptr<scf::UCKS>( static_cast<scf::UCKS*>(ref_scf.get()) );
            boost::shared_ptr<Wavefunction> new_scf = boost::shared_ptr<Wavefunction>(new scf::UCKS(options,psio,ref_ucks));
            Process::environment.wavefunction().reset();
            Process::environment.set_wavefunction(new_scf);
            double new_energy = new_scf->compute_energy();
            // Print a molden file
            if ( options["MOLDEN_FILE"].has_changed() ) {
               boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(new_scf));
               molden->write(to_string(state + 1) + "." + options.get_str("MOLDEN_FILE"));
            }
            energies.push_back(new_energy);
            ref_scf = new_scf;
        }
    }else {
        throw InputException("Unknown reference " + reference, "REFERENCE", __FILE__, __LINE__);
    }
//    // Print the excitation energies
//    for(int state = 1; state < static_cast<int>(energies.size()); ++state){
//        double exc_energy = energies[state] - energies[0];
//        fprintf(outfile,"  Excited state %d : excitation energy = %9.6f Eh = %8.4f eV = %9.1f cm**-1 \n",
//                state,exc_energy,exc_energy * _hartree2ev, exc_energy * _hartree2wavenumbers);
//    }

    // Set this early because the callback mechanism uses it.
    Process::environment.wavefunction().reset();

    Communicator::world->sync();

//    // Set some environment variables
//    Process::environment.globals["SCF TOTAL ENERGY"] = energies.back();
//    Process::environment.globals["CURRENT ENERGY"] = energies.back();
//    Process::environment.globals["CURRENT REFERENCE ENERGY"] = energies[0];

    // Shut down psi.

    tstop();
    return Success;
}

}} // End namespaces