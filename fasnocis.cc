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


#define DEBUG_THIS2(EXP) \
    outfile->Printf("\n  Starting " #EXP " ..."); fflush(outfile); \
    EXP \
    outfile->Printf("  done."); fflush(outfile); \


using namespace psi;

namespace psi{ namespace scf{

FASNOCIS::FASNOCIS(Options &options, boost::shared_ptr<PSIO> psio)
: UHF(options, psio)
{
}

FASNOCIS::FASNOCIS(Options &options, boost::shared_ptr<PSIO> psio,
                  boost::shared_ptr<Wavefunction> ref_scf,
                  std::vector<std::pair<int,int>> active_mos,
                  std::vector<int> aocc,
                  std::vector<int> bocc)
    : UHF(options,psio), ref_scf_(ref_scf), active_mos_(active_mos),
      aocc_(aocc), bocc_(bocc)
{
}

FASNOCIS::~FASNOCIS()
{
}

void FASNOCIS::form_C()
{
//  if(not do_noci){
//        UHF::form_C();
//     }
//   else{
//         form_C_noci();
//  }
}

}} // Namespaces
