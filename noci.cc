#include <physconst.h>
#include <psifiles.h>
#include <libmints/mints.h>
#include <libfock/apps.h>
#include <libfock/v.h>
#include <libfock/jk.h>
#include <liboptions/liboptions.h>

#include "noci.h"

#define DEBUG_NOCI 0


using namespace psi;

namespace psi{ namespace scf{

NOCI::NOCI(Options &options, boost::shared_ptr<PSIO> psio): UHF(options, psio),do_noci(false),state_a_(0),state_b_(0)
{
    init();
}

NOCI::NOCI(Options &options, boost::shared_ptr<PSIO> psio, int state_a, std::pair<int,int> fmo, int state_b,
           std::vector<std::pair<int,int>>frozen_occ_a,std::vector<std::pair<int,int>>frozen_occ_b,
           std::vector<std::pair<int,int>>frozen_mos,
           std::vector<int>occ_frozen,std::vector<int>vir_frozen,
           SharedMatrix Ca_gs_, SharedMatrix Cb_gs_,bool valence_in)
: UHF(options, psio),
  do_noci(true),do_alpha_states(state_a),
  state_a_(state_a),
  fmo_(fmo),
  state_b_(state_b),occ_(1),
  frozen_occ_a_(frozen_occ_a),
  frozen_occ_b_(frozen_occ_b),
  frozen_mos_(frozen_mos),
  occ_frozen_(occ_frozen),
  vir_frozen_(vir_frozen),
  Ca_gs(Ca_gs_),
  Cb_gs(Cb_gs_),
    valence(valence_in)
{
    //outfile->Printf("\n  ==> first excitation CI OFFFFGGG    <==\n\n");
    init();
    init_excitation();

}

void NOCI::init()
{
    // Allocate matrices
    Dolda_ = factory_->create_shared_matrix("Dold alpha");
    Doldb_ = factory_->create_shared_matrix("Dold beta");
    Dt_diff = factory_->create_shared_matrix("Density_diff");

}

NOCI::~NOCI()
{
}

void NOCI::init_excitation()
{

        Ft_ = SharedMatrix(new Matrix("Ft_",nsopi_,nsopi_));

        Ua_ = SharedMatrix(new Matrix("C_aN",nsopi_,nmopi_));
        Ub_ = SharedMatrix(new Matrix("C_bN",nsopi_,nmopi_));

        Ca0  = SharedMatrix(new Matrix("Ca0",nsopi_,nmopi_));
        Cb0  = SharedMatrix(new Matrix("Cb0",nsopi_,nmopi_));

        U_swap =    SharedMatrix(new Matrix("U_fmo",nsopi_,nmopi_));

        Ca_fmo =    SharedMatrix(new Matrix("Ca_fmo",nsopi_,nmopi_));
        Cb_fmo=    SharedMatrix(new Matrix("Cb_fmo",nsopi_,nmopi_));

        rFpq_a = SharedMatrix(new Matrix("rFpq_a",nmopi_-(occ_frozen_+vir_frozen_),nmopi_-(occ_frozen_+vir_frozen_)));
        rFpq_b = SharedMatrix(new Matrix("rFpq_b",nmopi_-(occ_frozen_+vir_frozen_),nmopi_-(occ_frozen_+vir_frozen_)));

        rFpq = SharedMatrix(new Matrix("rFpq",nmopi_-(occ_frozen_+vir_frozen_),nmopi_-(occ_frozen_+vir_frozen_)));
        rUa_ = SharedMatrix(new Matrix("rUa_",nmopi_-(occ_frozen_+vir_frozen_),nmopi_-(occ_frozen_+vir_frozen_)));
        rUb_ = SharedMatrix(new Matrix("rUa_",nmopi_-(occ_frozen_+vir_frozen_),nmopi_-(occ_frozen_+vir_frozen_)));
        Repsilon_a_= SharedVector(new Vector("Dirac EigenValues",nmopi_-(occ_frozen_+vir_frozen_)));


        rCa0 = SharedMatrix(new Matrix("rCa0",nsopi_, nmopi_-(occ_frozen_+vir_frozen_)));
        rCb0 = SharedMatrix(new Matrix("rCb0",nsopi_, nmopi_-(occ_frozen_+vir_frozen_)));


        rCa_ = SharedMatrix(new Matrix("rCa0",nsopi_,nmopi_-(occ_frozen_+vir_frozen_)));
        rCb_ = SharedMatrix(new Matrix("rCb0",nsopi_,nmopi_-(occ_frozen_+vir_frozen_)));



} //end of the function


void NOCI::guess()
{
if(do_noci){
        iteration_ = 0;
       load_orbitals();
        U_swap->identity();
        if(do_alpha_states){
            int irrep = std::get<0>(fmo_);
            int fmo   = std::get<1>(fmo_);
            int state = nalphapi_[irrep]-1+state_a_;
            outfile->Printf("irrep %d fmo %d state_a %d \n", irrep, fmo, state);
            U_swap->set(irrep,fmo,fmo,0.0);
            U_swap->set(irrep,state,state,0.0);
            U_swap->set(irrep,fmo,state,1.0);
            U_swap->set(irrep,state,fmo,1.0);
            Ca_fmo->gemm(false,false,1.0,Ca_gs,U_swap,0.0);
            Cb_fmo->copy(Cb_gs);
        }else{
            int irrep = std::get<0>(fmo_);
            int fmo   = std::get<1>(fmo_);
            int state = nbetapi_[irrep]-1+state_b_;
            U_swap->set(irrep,fmo,fmo,0.0);
            U_swap->set(irrep,state,state,0.0);
            U_swap->set(irrep,fmo,state,1.0);
            U_swap->set(irrep,state,fmo,1.0);
        Cb_fmo->gemm(false,false,1.0,Cb_gs,U_swap,0.0);
        Ca_fmo->copy(Ca_gs);
    }
//    Ca_->copy(Ca_gs);
//    Cb_->copy(Cb_gs); this is incorrect

 //   Ca_->copy(Ca_fmo);
 //   Cb_->copy(Cb_fmo);
    UHF::guess();
    form_D();
    E_ = compute_initial_E();
    for(int h=0; h <nirrep_; ++h){
        int oo;
        if(valence ){
        oo=nalphapi_[h]-occ_frozen_[h];
        int aa=nalphapi_[h]+vir_frozen_[h];
        int ov;
        for (int i=0; i <nsopi_[h]; ++i){
            for (int j=0; j < oo; ++j){
                double Cpq =Ca_gs->get(h,i,j);
                rCa0->set(h,i,j,Cpq);
            }//j
            ov=oo;
            for (int a=aa;a <nmopi_[h];++a){
                double Cpq=Ca_gs->get(h,i,a);
                rCa0->set(h,i,ov,Cpq);
                ov+=1;
            }//a
        }//i
        }
        else
        {
        oo=occ_frozen_[h];
        int aa=nalphapi_[h]+vir_frozen_[h];
        int ov;
        for (int i=0; i <nsopi_[h]; ++i){
            int n=0;
            for (int j=oo; j < nalphapi_[h]; ++j){
                double Cpq =Ca_gs->get(h,i,j);
                rCa0->set(h,i,n,Cpq);
                n=n+1;
            }//j
            ov=nalphapi_[h]-oo;
            for (int a=aa;a <nmopi_[h];++a){
                double Cpq=Ca_gs->get(h,i,a);
                rCa0->set(h,i,ov,Cpq);
                ov+=1;
            }//a
        }//i
        }
    } //irrep


    for(int h=0; h <nirrep_; ++h){
        int oo;
        if(valence ){
            oo=nbetapi_[h]-occ_frozen_[h];
            int aa=nbetapi_[h]+vir_frozen_[h];
            int ov;
            for (int i=0; i <nsopi_[h]; ++i){
                for (int j=0; j < oo; ++j){
                    double Cpq =Cb_gs->get(h,i,j);
                    rCb0->set(h,i,j,Cpq);
                }//j
                ov=oo;
                for (int a=aa;a <nmopi_[h];++a){
                    double Cpq=Cb_gs->get(h,i,a);
                    rCb0->set(h,i,ov,Cpq);
                    ov+=1;
                }//a
            }//i
        }
        else{
            oo=occ_frozen_[h];
            int aa=nbetapi_[h]+vir_frozen_[h];
            int ov;
            for (int i=0; i <nsopi_[h]; ++i){
                int n=0;
                for (int j=oo; j < nbetapi_[h]; ++j){
                    double Cpq =Cb_gs->get(h,i,j);
                    rCb0->set(h,i,n,Cpq);
                    n=n+1;
                }//j
                ov=nbetapi_[h]-oo;
                for (int a=aa;a <nmopi_[h];++a){
                    double Cpq=Cb_gs->get(h,i,a);
                    rCb0->set(h,i,ov,Cpq);
                    ov+=1;
                }//a
            }//i
        }

    } //irrep
}else{
UHF::guess();
}
}


bool NOCI::test_convergency()
{
    double ediff = E_ - Eold_;
    Dt_diff->copy(Dtold_);
    Dt_diff->subtract(Dt_);
    den_diff=0.0;
    den_diff=Dt_diff->rms();
    if (fabs(ediff) < energy_threshold_ && den_diff < density_threshold_)
        return true;
    else
        return false;
}

void NOCI::form_C()
{
  if(not do_noci){
        UHF::form_C();
     }
   else{
         form_C_noci();
  }
}



void NOCI::form_C_noci()
{
rFpq->zero();
    rFpq_a->transform(Fa_,rCa0);
    rFpq_b->transform(Fb_,rCb0);


    rFpq->copy(rFpq_a);
    rFpq->add(rFpq_b);
    rFpq->scale(0.5);
    rFpq->diagonalize(rUa_,Repsilon_a_);

    rUb_->copy(rUa_);
    rCa_->gemm(false,false,1.0,rCa0,rUa_,0.0);
    rCb_->gemm(false,false,1.0,rCb0,rUb_,0.0);

    rCa0->copy(rCa_);
    rCb0->copy(rCb_);

    Ca_->zero();
    Cb_->zero();
    for(int h=0; h <nirrep_; ++h){
        int oo;
        if(valence){
        oo=nalphapi_[h]-occ_frozen_[h];
        for (int i=0; i < nsopi_[h]; ++i){
            for(int j=0; j <(nalphapi_[h]-occ_frozen_[h]); ++j){
                Ca_->set(h,i,j,rCa_->get(h,i,j));
            }
            for(int jf=(nalphapi_[h]-occ_frozen_[h]); jf <nalphapi_[h];++jf){
                Ca_->set(h,i,jf,Ca_fmo->get(h,i,jf));
            }
            for(int af=nalphapi_[h]; af <(nalphapi_[h]+vir_frozen_[h]);++af){
                Ca_->set(h,i,af,Ca_fmo->get(h,i,af));
            }
            int m=oo;
            for (int a=(nalphapi_[h]+vir_frozen_[h]);a<nmopi_[h];++a){
                 Ca_->set(h,i,a,rCa_->get(h,i,m));
                 m=m+1;
            }
        }
        }
        else{
            oo=occ_frozen_[h];
            for (int i=0; i < nsopi_[h]; ++i){
                for(int j=0; j <oo; ++j){
                    Ca_->set(h,i,j,Ca_fmo->get(h,i,j));
                }
                int n=0;
                for(int jf=oo; jf <nalphapi_[h];++jf){
                    Ca_->set(h,i,jf,rCa_->get(h,i,n));
                    n=n+1;
                }
                for(int af=nalphapi_[h]; af <(nalphapi_[h]+vir_frozen_[h]);++af){
                    Ca_->set(h,i,af,Ca_fmo->get(h,i,af));
                }
                int m=nalphapi_[h]-oo;
                for (int a=(nalphapi_[h]+vir_frozen_[h]);a<nmopi_[h];++a){
                     Ca_->set(h,i,a,rCa_->get(h,i,m));
                     m=m+1;
                }
            }
        }


    }

    for(int h=0; h <nirrep_; ++h){
        int oo;
        if(valence){
        oo=nbetapi_[h]-occ_frozen_[h];
        for (int i=0; i < nsopi_[h]; ++i){
            for(int j=0; j <nbetapi_[h]-occ_frozen_[h]; ++j){
                Cb_->set(h,i,j,rCb_->get(h,i,j));
            }
            for(int j=nbetapi_[h]-occ_frozen_[h]; j <nbetapi_[h];++j){
                Cb_->set(h,i,j,Cb_fmo->get(h,i,j));
            }
            for(int af=nbetapi_[h]; af <(nbetapi_[h]+vir_frozen_[h]);++af){
                Cb_->set(h,i,af,Cb_fmo->get(h,i,af));
            }
            int m=oo;
            for (int a=(nbetapi_[h]+vir_frozen_[h]);a<nmopi_[h];++a){
                 Cb_->set(h,i,a,rCb_->get(h,i,m));
                 m=m+1;
            }
        }
        }
        else{
            oo=occ_frozen_[h];
            for (int i=0; i < nsopi_[h]; ++i){
                for(int j=0; j <oo; ++j){
                    Cb_->set(h,i,j,Cb_fmo->get(h,i,j));
                }
                int n=0;
                for(int j=oo; j <nbetapi_[h];++j){
                    Cb_->set(h,i,j,rCb_->get(h,i,n));
                    n=n+1;
                }
                for(int af=nbetapi_[h]; af <(nbetapi_[h]+vir_frozen_[h]);++af){
                    Cb_->set(h,i,af,Cb_fmo->get(h,i,af));
                }
                int m=nbetapi_[h]-oo;
                for (int a=(nbetapi_[h]+vir_frozen_[h]);a<nmopi_[h];++a){
                     Cb_->set(h,i,a,rCb_->get(h,i,m));
                     m=m+1;
                }
            }
        }

    }

}


}} // Namespaces

