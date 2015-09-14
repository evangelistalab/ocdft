#ifndef JUNK_H
#define JUNK_H




void UCKS::form_C_CH_algorithm()
{
    int nstate = static_cast<int>(dets.size());
    outfile->Printf("  Computing %d optimal hole orbitals\n",nstate);fflush(outfile);

    // Data structures to save the hole information
    Dimension aholepi(nirrep_,"Alpha holes per irrep");
    std::vector<SharedVector> holes_Ca;
    std::vector<int> holes_h;

    // Compute the hole states
    for (int m = 0; m < nstate; ++m){
        // Transform Fa to the MO basis of state m
        TempMatrix->transform(Fa_,dets[m]->Ca());
        // Grab the occ block of Fa
        extract_square_subblock(TempMatrix,PoFaPo_,true,dets[m]->nalphapi(),1.0e9);
        PoFaPo_->diagonalize(Ua_o_,lambda_a_o_);
        std::vector<boost::tuple<double,int,int> > sorted_holes; // (energy,irrep,mo in irrep)
        for (int h = 0; h < nirrep_; ++h){
            int nmo = nmopi_[h];
            for (int p = 0; p < nmo; ++p){
                if (lambda_a_o_->get(h,p) < 1.0e6){
                    sorted_holes.push_back(boost::make_tuple(lambda_a_o_->get(h,p),h,p));
                }
            }
        }
        std::sort(sorted_holes.begin(),sorted_holes.end());
        boost::tuple<double,int,int> hole;
        // Extract the hole alpha orbital according to an energy criteria (this needs a generalization)
        if (KS::options_.get_str("CDFT_EXC_TYPE") == "VALENCE"){
            // For valence excitations select the highest lying orbital (HOMO-like)
            hole = sorted_holes.back();
        }else if(KS::options_.get_str("CDFT_EXC_TYPE") == "CORE"){
            // For core excitations select the lowest lying orbital (1s-like)
            hole = sorted_holes.front();
        }
        double hole_energy = hole.get<0>();
        int hole_h = hole.get<1>();
        int hole_mo = hole.get<2>();
        outfile->Printf("   constrained hole %d :(irrep = %d,mo = %d,energy = %.6f)\n",
                        m,hole_h,hole_mo,hole_energy);

        // Compute the hole orbital
        SharedVector hole_Ca = factory_->create_shared_vector("Hole");
        for (int p = 0; p < nsopi_[hole_h]; ++p){
            double c_p = 0.0;
            for (int i = 0; i < dets[m]->nalphapi()[hole_h]; ++i){
                c_p += dets[m]->Ca()->get(hole_h,p,i) * Ua_o_->get(hole_h,i,hole_mo) ;
            }
            hole_Ca->set(hole_h,p,c_p);
        }
        holes_Ca.push_back(hole_Ca);
        holes_h.push_back(hole_h);
        aholepi[hole_h] += 1;
    }

    // Put the hole orbitals in Ch
    SharedMatrix Ch = SharedMatrix(new Matrix("Ch",nsopi_,aholepi));
    SharedMatrix Cho = SharedMatrix(new Matrix("Cho",nsopi_,aholepi));
    std::vector<int> offset(nirrep_,0);
    for (int m = 0; m < nstate; ++m){
        //int h = current_excited_state->ah_sym(m);
        int h = holes_h[m];
        Ch->set_column(h,offset[h],holes_Ca[m]);
        offset[h] += 1;
    }

    // Orthogonalize the hole orbitals
    SharedMatrix Spp = SharedMatrix(new Matrix("Spp",aholepi,aholepi));
    SharedMatrix Upp = SharedMatrix(new Matrix("Upp",aholepi,aholepi));
    SharedVector spp = SharedVector(new Vector("spp",aholepi));
    Spp->transform(S_,Ch);
    Spp->diagonalize(Upp,spp);
    double S_cutoff = 1.0e-3;
    // Form the transformation matrix X (in place of Upp)
    for (int h = 0; h < nirrep_; ++h) {
        //in each irrep, scale significant cols i by 1.0/sqrt(s_i)
        for (int i = 0; i < aholepi[h]; ++i) {
            if (std::fabs(spp->get(h,i)) > S_cutoff) {
                double scale = 1.0 / std::sqrt(spp->get(h,i));
                Upp->scale_column(h,i,scale);
            } else {
                throw FeatureNotImplemented("CKS", "Cannot yet deal with linear dependent particle orbitals", __FILE__, __LINE__);
            }
        }
    }
    Cho->zero();
    Cho->gemm(false,false,1.0,Ch,Upp,0.0);
    Ch_->zero();
    copy_block(Cho,Ch_,nsopi_,aholepi);

    // Form the projector onto the orbitals orthogonal to the particles in the ground state mo representation
    TempMatrix->zero();
    TempMatrix->gemm(false,true,1.0,Cho,Cho,0.0);
    TempMatrix->transform(S_);
    TempMatrix->transform(dets[0]->Ca());
    TempMatrix2->identity();
    TempMatrix2->subtract(TempMatrix);

    // Form the Fock matrix in the GS basis, project out the holes, diagonalize it, and transform the MO coefficients
    TempMatrix->transform(Fa_,dets[0]->Ca());
    TempMatrix->transform(TempMatrix2);

    TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
    Ca_->zero();
    Ca_->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix2,0.0);

//    epsilon_a_->print();

    std::vector<boost::tuple<double,int,int> > sorted_spectators;
    for (int h = 0; h < nirrep_; ++h){
        for (int p = 0; p < nmopi_[h]; ++p){
            sorted_spectators.push_back(boost::make_tuple(epsilon_a_->get(h,p),h,p));
        }
    }
    std::sort(sorted_spectators.begin(),sorted_spectators.end());

    for (int h = 0; h < nirrep_; ++h){
        nalphapi_[h] = 0;
    }
    nbetapi_ = dets[0]->nbetapi();
    int assigned = 0;
    for (int p = 0; p < nmo_; ++p){
        if (assigned < nalpha_){
            if(std::fabs(sorted_spectators[p].get<0>()) > 1.0e-6){  // !!! Check this out NB WARNING
                int h = sorted_spectators[p].get<1>();
                nalphapi_[h] += 1;
                assigned += 1;
            }
        }
    }
    // At this point the orbitals are sorted according to the energy but we
    // want to make sure that the hole MO appear where they should, that is
    // the holes in the virtual space.
    // |(1) (2) ... (hole) | ...> will become
    // |(particle) (1) (2) ... | ... (hole)>
    TempMatrix->zero();
    TempVector->zero();
    for (int h = 0; h < nirrep_; ++h){
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        double** T_h = TempMatrix->pointer(h);
        double** C_h = Ca_->pointer(h);
        double** Cho_h = Cho->pointer(h);
        // First place the holes
        int m = 0;
        for (int p = 0; p < nmo; ++p){
            // Is this MO a hole or a particle?
            if(std::fabs(epsilon_a_->get(h,p)) > 1.0e-6){
                TempVector->set(h,m,epsilon_a_->get(h,p));
                for (int q = 0; q < nso; ++q){
                    T_h[q][m] = C_h[q][p];
                }
                m += 1;
            }
        }
        for (int p = 0; p < aholepi[h]; ++p){
            for (int q = 0; q < nso; ++q){
                T_h[q][m] = Cho_h[q][p];
            }
            m += 1;
        }
    }

    Ca_->copy(TempMatrix);
    epsilon_a_->copy(TempVector.get());

//    Ca_->print();
    int old_socc[8];
    int old_docc[8];
    for(int h = 0; h < nirrep_; ++h){
        old_socc[h] = soccpi_[h];
        old_docc[h] = doccpi_[h];
    }

    for (int h = 0; h < nirrep_; ++h) {
        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
    }

    bool occ_changed = false;
    for(int h = 0; h < nirrep_; ++h){
        if( old_socc[h] != soccpi_[h] || old_docc[h] != doccpi_[h]){
            occ_changed = true;
            break;
        }
    }

    // BETA
    diagonalize_F(Fb_, Cb_, epsilon_b_);

    if (debug_) {
        Ca_->print(outfile);
        Cb_->print(outfile);
    }
}

void UCKS::form_C_CP_algorithm()
{
    int nstate = static_cast<int>(dets.size());
    outfile->Printf("  Computing %d optimal particle orbitals\n",nstate);

    // Data structures to save the particle information
    Dimension apartpi(nirrep_,"Alpha particles per irrep");
    std::vector<SharedVector> parts_Ca;
    std::vector<int> parts_h;

    // Compute the particle states
    for (int m = 0; m < nstate; ++m){
        // Transform Fa to the MO basis of state m
        TempMatrix->transform(Fa_,dets[m]->Ca());
        // Grab the vir block of Fa
        extract_square_subblock(TempMatrix,PvFaPv_,false,dets[m]->nalphapi(),1.0e9);
        PvFaPv_->diagonalize(Ua_v_,lambda_a_v_);
        std::vector<boost::tuple<double,int,int> > sorted_vir; // (energy,irrep,mo in irrep)
        for (int h = 0; h < nirrep_; ++h){
            int nmo = nmopi_[h];
            for (int p = 0; p < nmo; ++p){
                sorted_vir.push_back(boost::make_tuple(lambda_a_v_->get(h,p),h,p));  // N.B. shifted to full indexing
            }
        }
        std::sort(sorted_vir.begin(),sorted_vir.end());
        // In the case of particle, we assume that we are always interested in the lowest lying orbitals
        boost::tuple<double,int,int> particle = sorted_vir.front();
        int part_h = particle.get<1>();
        int part_mo = particle.get<2>();
        outfile->Printf("   constrained particle %d :(irrep = %d,mo = %d,energy = %.6f)\n",
                m,particle.get<1>(),particle.get<2>(),particle.get<0>());

        // Compute the particle orbital
        SharedVector part_Ca = factory_->create_shared_vector("Particle");
        for (int p = 0; p < nsopi_[part_h]; ++p){
            double c_p = 0.0;
            int maxa = nmopi_[part_h] - dets[m]->nalphapi()[part_h];
            for (int a = 0; a < maxa; ++a){
                c_p += dets[m]->Ca()->get(part_h,p,dets[m]->nalphapi()[part_h] + a) * Ua_v_->get(part_h,a,part_mo) ;
            }
            part_Ca->set(part_h,p,c_p);
        }
        parts_Ca.push_back(part_Ca);
        parts_h.push_back(part_h);
        apartpi[part_h] += 1;
    }

    // Put the particle orbitals in Cp
    SharedMatrix Cp = SharedMatrix(new Matrix("Cp",nsopi_,apartpi));
    SharedMatrix Cpo = SharedMatrix(new Matrix("Cpo",nsopi_,apartpi));
    std::vector<int> offset(nirrep_,0);
    for (int m = 0; m < nstate; ++m){
        int h = parts_h[m];
        Cp->set_column(h,offset[h],parts_Ca[m]);
        offset[h] += 1;
    }
    SharedMatrix Spp = SharedMatrix(new Matrix("Spp",apartpi,apartpi));
    SharedMatrix Upp = SharedMatrix(new Matrix("Upp",apartpi,apartpi));
    SharedVector spp = SharedVector(new Vector("spp",apartpi));
    Spp->transform(S_,Cp);
    Spp->print();
    Spp->diagonalize(Upp,spp);

    double S_cutoff = 1.0e-2;
    // Form the transformation matrix X (in place of Upp)
    for (int h = 0; h < nirrep_; ++h) {
        //in each irrep, scale significant cols i  by 1.0/sqrt(s_i)
        for (int i = 0; i < apartpi[h]; ++i) {
            if (S_cutoff  < spp->get(h,i)) {
                double scale = 1.0 / sqrt(spp->get(h, i));
                Upp->scale_column(h, i, scale);
            } else {
                throw FeatureNotImplemented("CKS", "Cannot yet deal with linear dependent particle orbitals", __FILE__, __LINE__);
            }
        }
    }
    Cpo->zero();
    Cpo->gemm(false,false,1.0,Cp,Upp,0.0);
    Cp_->zero();
    copy_block(Cpo,Cp_,nsopi_,apartpi);

    // Form the projector onto the orbitals orthogonal to the particles in the ground state mo representation
    TempMatrix->zero();
    TempMatrix->gemm(false,true,1.0,Cpo,Cpo,0.0);
    TempMatrix->transform(S_);
    TempMatrix->transform(dets[0]->Ca());
    TempMatrix2->identity();
    TempMatrix2->subtract(TempMatrix);

    // Form the Fock matrix in the GS basis, diagonalize it, and transform the MO coefficients
    TempMatrix->transform(Fa_,dets[0]->Ca());
    TempMatrix->transform(TempMatrix2);
    TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
    Ca_->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix2,0.0);

    std::vector<boost::tuple<double,int,int> > sorted_spectators;
    for (int h = 0; h < nirrep_; ++h){
        for (int p = 0; p < nmopi_[h]; ++p){
            sorted_spectators.push_back(boost::make_tuple(epsilon_a_->get(h,p),h,p));
        }
    }
    std::sort(sorted_spectators.begin(),sorted_spectators.end());

    for (int h = 0; h < nirrep_; ++h){
        nalphapi_[h] = apartpi[h];
    }
    nbetapi_ = dets[0]->nbetapi();
    int assigned = 0;
    for (int p = 0; p < nmo_; ++p){
        if (assigned < nalpha_ - nstate){
            if(std::fabs(sorted_spectators[p].get<0>()) > 1.0e-6){  // !!! Check this out NB WARNING
                int h = sorted_spectators[p].get<1>();
                nalphapi_[h] += 1;
                assigned += 1;
            }
        }
    }

    // At this point the orbitals are sorted according to the energy but we
    // want to make sure that the hole and the particle MO appear where they
    // should, that is the holes in the virtual space and the particles in
    // the occupied space.
    // |(1) (2) ... (hole) | (particle) ...> will become
    // |(particle) (1) (2) ...  | ... (hole)>
    TempMatrix->zero();
    TempVector->zero();
    for (int h = 0; h < nirrep_; ++h){
        int m = apartpi[h];  // Offset by the number of holes
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        double** T_h = TempMatrix->pointer(h);
        double** C_h = Ca_->pointer(h);
        double** Cpo_h = Cpo->pointer(h);
        for (int p = 0; p < m; ++p){
            for (int q = 0; q < nso; ++q){
                T_h[q][p] = Cpo_h[q][p];
            }
        }
        for (int p = 0; p < nmo; ++p){
            // Is this MO a hole or a particle?
            if(std::fabs(epsilon_a_->get(h,p)) > 1.0e-6){
                TempVector->set(h,m,epsilon_a_->get(h,p));
                for (int q = 0; q < nso; ++q){
                    T_h[q][m] = C_h[q][p];
                }
                m += 1;
            }
        }
    }

    Ca_->copy(TempMatrix);
    epsilon_a_->copy(TempVector.get());

    int old_socc[8];
    int old_docc[8];
    for(int h = 0; h < nirrep_; ++h){
        old_socc[h] = soccpi_[h];
        old_docc[h] = doccpi_[h];
    }

    for (int h = 0; h < nirrep_; ++h) {
        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
    }

    bool occ_changed = false;
    for(int h = 0; h < nirrep_; ++h){
        if( old_socc[h] != soccpi_[h] || old_docc[h] != doccpi_[h]){
            occ_changed = true;
            break;
        }
    }

    // BETA
    diagonalize_F(Fb_, Cb_, epsilon_b_);

    if (debug_) {
        Ca_->print(outfile);
        Cb_->print(outfile);
    }
}

void UCKS::form_C_CHP_algorithm()
{
    int nstate = static_cast<int>(dets.size());
    if (nstate > 1)
        throw FeatureNotImplemented("CKS", "Cannot treat more than one excited state in the CHP method", __FILE__, __LINE__);

    // Compute the hole and particle states
    for (int m = 0; m < nstate; ++m){
        // Transform Fa to the MO basis of state m
        TempMatrix->transform(Fa_,dets[m]->Ca());
        // Grab the occ and vir blocks
        // |--------|--------|
        // |        |        |
        // | PoFaPo |        |
        // |        |        |
        // |--------|--------|
        // |        |        |
        // |        | PvFaPv |
        // |        |        |
        // |--------|--------|
        // Grab the occ block of Fa
        extract_square_subblock(TempMatrix,PoFaPo_,true,dets[m]->nalphapi(),1.0e9);
        // Grab the vir block of Fa
        extract_square_subblock(TempMatrix,PvFaPv_,false,dets[m]->nalphapi(),1.0e9);

        // Diagonalize the hole block
        PoFaPo_->diagonalize(Ua_o_,lambda_a_o_);
        std::vector<boost::tuple<double,int,int> > sorted_holes; // (energy,irrep,mo in irrep)
        for (int h = 0; h < nirrep_; ++h){
            int nmo = nmopi_[h];
            for (int p = 0; p < nmo; ++p){
                if (lambda_a_o_->get(h,p) < 1.0e6){
                    sorted_holes.push_back(boost::make_tuple(lambda_a_o_->get(h,p),h,p));
                }
            }
        }
        std::sort(sorted_holes.begin(),sorted_holes.end());

        // Diagonalize the particle block
        PvFaPv_->diagonalize(Ua_v_,lambda_a_v_);
        std::vector<boost::tuple<double,int,int> > sorted_vir; // (energy,irrep,mo in irrep)
        for (int h = 0; h < nirrep_; ++h){
            int nmo = nmopi_[h];
            for (int p = 0; p < nmo; ++p){
                sorted_vir.push_back(boost::make_tuple(lambda_a_v_->get(h,p),h,p));  // N.B. shifted wrt to full indexing
            }
        }
        std::sort(sorted_vir.begin(),sorted_vir.end());

        boost::tuple<double,int,int> hole;
        boost::tuple<double,int,int> particle;
        std::vector<boost::tuple<double,int,int,double,int,int,double> > sorted_hp_pairs;

        // Extract the hole alpha orbital according to an energy criteria (this needs a generalization)
        bool do_core_excitation = false;
        double hole_energy_shift = 0.0;
        if (KS::options_.get_str("CDFT_EXC_TYPE") == "VALENCE"){
            // For valence excitations select the highest lying orbital (HOMO-like)
        }else if(KS::options_.get_str("CDFT_EXC_TYPE") == "CORE"){
            do_core_excitation = true;
            // Get the energy of the lowest lying orbital (1s-like)
            hole_energy_shift = sorted_holes.front().get<0>();
        }
        CharacterTable ct = KS::molecule_->point_group()->char_table();

        // Determine the hole/particle pair to follow
        // Compute the symmetry adapted hole/particle pairs
        for (int h_h = 0; h_h < nirrep_; ++h_h){
            int nmo_h = nmopi_[h_h];
            for (int h = 0; h < nmo_h; ++h){
                double e_h = lambda_a_o_->get(h_h,h);
                for (int h_p = 0; h_p < nirrep_; ++h_p){
                    int nmo_p = nmopi_[h_p];
                    for (int p = 0; p < nmo_p; ++p){
                        double e_p = lambda_a_v_->get(h_p,p);
                        if ((e_h < 1.0e6) and (e_p < 1.0e6)){  // Test to eliminate the fake eigenvalues added to the PFP matrices
                            double e_hp = do_core_excitation ? (e_p + e_h - hole_energy_shift) : (e_p - e_h);
                            int symm = h_h ^ h_p ^ ground_state_symmetry_;
                            if(not do_symmetry or (symm == excited_state_symmetry_)){ // Test for symmetry
                                sorted_hp_pairs.push_back(boost::make_tuple(e_hp,h_h,h,e_h,h_p,p,e_p));  // N.B. shifted wrt to full indexing
//                                outfile->Printf( "  %s  gamma(h) = %s, gamma(p) = %s, gamma(hp) = %s, gamma(Phi-hp) = %s \n",do_symmetry ? "true" : "false",
//                                        ct.gamma(h_h).symbol(),ct.gamma(h_p).symbol(),ct.gamma(h_h ^ h_p).symbol(),
//                                        ct.gamma(symm).symbol());
                            }
                        }
                    }
                }
            }
        }

        std::sort(sorted_hp_pairs.begin(),sorted_hp_pairs.end());
        if(iteration_ == 0){
            outfile->Printf( "\n  Ground state symmetry: %s\n",ct.gamma(ground_state_symmetry_).symbol());
            outfile->Printf( "  Excited state symmetry: %s\n",ct.gamma(excited_state_symmetry_).symbol());
            outfile->Printf( "\n  Lowest energy excitations:\n");
            outfile->Printf( "  --------------------------------------\n");
            outfile->Printf( "    N   Occupied     Virtual     E(eV)  \n");
            outfile->Printf( "  --------------------------------------\n");
            int maxstates = std::min(10,static_cast<int>(sorted_hp_pairs.size()));
            for (int n = 0; n < maxstates; ++n){
                double energy_hp = sorted_hp_pairs[n].get<6>() - sorted_hp_pairs[n].get<3>();
                outfile->Printf("   %2d:  %4d%-3s  -> %4d%-3s   %9.3f\n",n + 1,
                        sorted_hp_pairs[n].get<2>() + 1,
                        ct.gamma(sorted_hp_pairs[n].get<1>()).symbol(),
                        dets[m]->nalphapi()[sorted_hp_pairs[n].get<4>()] + sorted_hp_pairs[n].get<5>() + 1,
                        ct.gamma(sorted_hp_pairs[n].get<4>()).symbol(),
                        energy_hp * _hartree2ev);
            }
            outfile->Printf( "  --------------------------------------\n");

            int select_pair = 0;
            // Select the excitation pair using the energetic ordering
            if(KS::options_["CDFT_EXC_SELECT"].has_changed()){
                int input_select = KS::options_["CDFT_EXC_SELECT"][excited_state_symmetry_].to_integer();
                if (input_select > 0){
                    select_pair = input_select - 1;
                    outfile->Printf( "\n  Following excitation #%d: ",input_select);
                }
            }
            // Select the excitation pair using the symmetry of the hole
            if(KS::options_["CDFT_EXC_HOLE_SYMMETRY"].has_changed()){
                int input_select = KS::options_["CDFT_EXC_HOLE_SYMMETRY"][excited_state_symmetry_].to_integer();
                if (input_select > 0){
                    int maxstates = static_cast<int>(sorted_hp_pairs.size());
                    for (int n = 0; n < maxstates; ++n){
                        if(sorted_hp_pairs[n].get<1>() == input_select - 1){
                            select_pair = n;
                            break;
                        }
                    }
                    outfile->Printf( "\n  Following excitation #%d:\n",select_pair + 1);
                }
            }
            hole_h = sorted_hp_pairs[select_pair].get<1>();
            hole_mo = sorted_hp_pairs[select_pair].get<2>();
//            hole_energy = sorted_hp_pairs[select_pair].get<3>();

            part_h = sorted_hp_pairs[select_pair].get<4>();
            part_mo = sorted_hp_pairs[select_pair].get<5>();
//            part_energy = sorted_hp_pairs[select_pair].get<6>();
        }else{
            if(not (KS::options_["CDFT_EXC_SELECT"].has_changed() or
                    KS::options_["CDFT_EXC_HOLE_SYMMETRY"].has_changed())){
                hole_h = sorted_hp_pairs[0].get<1>();
                hole_mo = sorted_hp_pairs[0].get<2>();
                part_h = sorted_hp_pairs[0].get<4>();
                part_mo = sorted_hp_pairs[0].get<5>();
    //            double hole_energy = sorted_hp_pairs[0].get<3>();
    //            double part_energy = sorted_hp_pairs[0].get<6>();
            }
        }

//        int hole_h = sorted_hp_pairs[0].get<1>();
//        int hole_mo = sorted_hp_pairs[0].get<2>();
        double hole_energy = lambda_a_o_->get(hole_h,hole_mo);

//        int part_h = sorted_hp_pairs[0].get<4>();
//        int part_mo = sorted_hp_pairs[0].get<5>();
        double part_energy = lambda_a_v_->get(part_h,part_mo);
        outfile->Printf("                     "
                "%4d%-3s (%9.3f) -> %4d%-3s (%9.3f)\n",
                hole_mo + 1,
                ct.gamma(hole_h).symbol(),
                hole_energy,
                dets[m]->nalphapi()[part_h] + part_mo + 1,
                ct.gamma(part_h).symbol(),
                part_energy);

        // Compute the hole orbital
        SharedVector hole_Ca = factory_->create_shared_vector("Hole");
        for (int p = 0; p < nsopi_[hole_h]; ++p){
            double c_p = 0.0;
            for (int i = 0; i < dets[m]->nalphapi()[hole_h]; ++i){
                c_p += dets[m]->Ca()->get(hole_h,p,i) * Ua_o_->get(hole_h,i,hole_mo) ;
            }
            hole_Ca->set(hole_h,p,c_p);
        }
        holes_Ca.push_back(hole_Ca);
        holes_h.push_back(hole_h);
        aholepi[hole_h] += 1;

        // Compute the particle orbital
        SharedVector part_Ca = factory_->create_shared_vector("Particle");
        for (int p = 0; p < nsopi_[part_h]; ++p){
            double c_p = 0.0;
            int maxa = nmopi_[part_h] - dets[m]->nalphapi()[part_h];
            for (int a = 0; a < maxa; ++a){
                c_p += dets[m]->Ca()->get(part_h,p,dets[m]->nalphapi()[part_h] + a) * Ua_v_->get(part_h,a,part_mo) ;
            }
            part_Ca->set(part_h,p,c_p);
        }
        parts_Ca.push_back(part_Ca);
        parts_h.push_back(part_h);
        apartpi[part_h] += 1;
    }

    // Put the hole and particle orbitals in Ch_ and Cp_
    std::vector<int> hole_offset(nirrep_,0);
    std::vector<int> part_offset(nirrep_,0);
    Cp_->zero();
    Ch_->zero();
    for (int m = 0; m < nstate; ++m){
        int hole_h = holes_h[m];
        Ch_->set_column(hole_h,hole_offset[hole_h],holes_Ca[m]);
        hole_offset[hole_h] += 1;
        int part_h = parts_h[m];
        Cp_->set_column(part_h,part_offset[part_h],parts_Ca[m]);
        part_offset[part_h] += 1;
    }

    // Frozen spectator orbital algorithm
    // Transform the ground state orbitals to the representation which diagonalizes the
    // the PoFaPo and PvFaPv blocks
    // Form the transformation matrix that diagonalizes the PoFaPo and PvFaPv blocks
    // |----|----|
    // | Uo | 0  |
    // |----|----|
    // | 0  | Uv |
    // |----|----|
    TempMatrix->zero();
    for (int h = 0; h < nirrep_; ++h){
        int nocc = dets[0]->nalphapi()[h];
        int nvir = nmopi_[h] - nocc;
        if (nocc != 0){
            double** Temp_h = TempMatrix->pointer(h);
            double** Uo_h = Ua_o_->pointer(h);
            for (int i = 0; i < nocc; ++i){
                epsilon_a_->set(h,i,lambda_a_o_->get(h,i));
                for (int j = 0; j < nocc; ++j){
                    Temp_h[i][j] = Uo_h[i][j];
                }
            }
        }
        if (nvir != 0){
            double** Temp_h = TempMatrix->pointer(h);
            double** Uv_h = Ua_v_->pointer(h);
            for (int i = 0; i < nvir; ++i){
                epsilon_a_->set(h,i + nocc,lambda_a_v_->get(h,i));
                for (int j = 0; j < nvir; ++j){
                    Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
                }
            }
        }
    }
    // Get the excited state orbitals: Ca(ex) = Ca(gs) * (Uo | Uv)
    Ca_->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix,0.0);

    // Form the projector onto the orbitals orthogonal to the holes and particles in the excited state mo representation
    TempMatrix->zero();
    TempMatrix->gemm(false,true,1.0,Ch_,Ch_,0.0);
    TempMatrix->gemm(false,true,1.0,Cp_,Cp_,1.0);
    TempMatrix->transform(S_);
    TempMatrix->transform(Ca_);
    TempMatrix2->identity();
    TempMatrix2->subtract(TempMatrix);

    // Form the Fock matrix in the excited state basis, project out the h/p
    TempMatrix->transform(Fa_,Ca_);
    TempMatrix->transform(TempMatrix2);
    // If we want the relaxed orbitals diagonalize the Fock matrix and transform the MO coefficients
    if(KS::options_.get_str("CDFT_EXC_METHOD") == "CHP" or KS::options_.get_str("CDFT_EXC_METHOD") == "CHP-FB"){
        TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
        TempMatrix->zero();
        TempMatrix->gemm(false,false,1.0,Ca_,TempMatrix2,0.0);
        Ca_->copy(TempMatrix);
    }else{
        // The orbitals don't change, but make sure that epsilon_a_ has the correct eigenvalues (some which are zero)
        for (int h = 0; h < nirrep_; ++h){
            for (int p = 0; p < nmopi_[h]; ++p){
                epsilon_a_->set(h,p,TempMatrix->get(h,p,p));
            }
        }
    }

    std::vector<boost::tuple<double,int,int> > sorted_spectators;
    for (int h = 0; h < nirrep_; ++h){
        for (int p = 0; p < nmopi_[h]; ++p){
            sorted_spectators.push_back(boost::make_tuple(epsilon_a_->get(h,p),h,p));
        }
    }
    std::sort(sorted_spectators.begin(),sorted_spectators.end());

    // Find the alpha occupation
    int assigned = 0;
    for (int h = 0; h < nirrep_; ++h){
        nalphapi_[h] = apartpi[h];
        assigned += apartpi[h];
    }
    for (int p = 0; p < nmo_; ++p){
        if (assigned < nalpha_){
            if(std::fabs(sorted_spectators[p].get<0>()) > 1.0e-6){  // !!! Check this out NB WARNING
                int h = sorted_spectators[p].get<1>();
                nalphapi_[h] += 1;
                assigned += 1;
            }
        }
    }
    nbetapi_ = dets[0]->nbetapi();
    int old_socc[8];
    int old_docc[8];
    for(int h = 0; h < nirrep_; ++h){
        old_socc[h] = soccpi_[h];
        old_docc[h] = doccpi_[h];
    }

    for (int h = 0; h < nirrep_; ++h) {
        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
    }

    bool occ_changed = false;
    for(int h = 0; h < nirrep_; ++h){
        if( old_socc[h] != soccpi_[h] || old_docc[h] != doccpi_[h]){
            occ_changed = true;
            break;
        }
    }

    // At this point the orbitals are sorted according to the energy but we
    // want to make sure that the hole and particle MO appear where they should, that is
    // |(particles) (occupied spectators) | (virtual spectators) (hole)>
    TempMatrix->zero();
    TempVector->zero();
    for (int h = 0; h < nirrep_; ++h){
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        double** T_h = TempMatrix->pointer(h);
        double** C_h = Ca_->pointer(h);
        double** Cp_h = Cp_->pointer(h);
        double** Ch_h = Ch_->pointer(h);
        // First place the particles
        int m = 0;
        for (int p = 0; p < apartpi[h]; ++p){
            for (int q = 0; q < nso; ++q){
                T_h[q][m] = Cp_h[q][p];
            }
            m += 1;
        }
        // Then the spectators
        for (int p = 0; p < nmo; ++p){
            // Is this MO a hole or a particle?
            if(std::fabs(epsilon_a_->get(h,p)) > 1.0e-6){
                TempVector->set(h,m,epsilon_a_->get(h,p));
                for (int q = 0; q < nso; ++q){
                    T_h[q][m] = C_h[q][p];
                }
                m += 1;
            }
        }
        // Then the holes
        for (int p = 0; p < aholepi[h]; ++p){
            for (int q = 0; q < nso; ++q){
                T_h[q][m] = Ch_h[q][p];
            }
            m += 1;
        }
    }
    Ca_->copy(TempMatrix);
    epsilon_a_->copy(TempVector.get());

    // BETA
    if(KS::options_.get_str("CDFT_EXC_METHOD") == "CHP"){
        diagonalize_F(Fb_, Cb_, epsilon_b_);
    }else{
        // Unrelaxed procedure, but still find MOs which diagonalize the occupied block
        // Transform Fb to the MO basis of the ground state
        TempMatrix->transform(Fb_,dets[0]->Cb());
        // Grab the occ block of Fb
        extract_square_subblock(TempMatrix,PoFaPo_,true,dets[0]->nbetapi(),1.0e9);
        // Grab the vir block of Fa
        extract_square_subblock(TempMatrix,PvFaPv_,false,dets[0]->nbetapi(),1.0e9);
        // Diagonalize the hole block
        PoFaPo_->diagonalize(Ua_o_,lambda_a_o_);
        // Diagonalize the particle block
        PvFaPv_->diagonalize(Ua_v_,lambda_a_v_);
        // Form the transformation matrix that diagonalizes the PoFaPo and PvFaPv blocks
        // |----|----|
        // | Uo | 0  |
        // |----|----|
        // | 0  | Uv |
        // |----|----|
        TempMatrix->zero();
        for (int h = 0; h < nirrep_; ++h){
            int nocc = dets[0]->nbetapi()[h];
            int nvir = nmopi_[h] - nocc;
            if (nocc != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** Uo_h = Ua_o_->pointer(h);
                for (int i = 0; i < nocc; ++i){
                    epsilon_b_->set(h,i,lambda_a_o_->get(h,i));
                    for (int j = 0; j < nocc; ++j){
                        Temp_h[i][j] = Uo_h[i][j];
                    }
                }
            }
            if (nvir != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** Uv_h = Ua_v_->pointer(h);
                for (int i = 0; i < nvir; ++i){
                    epsilon_b_->set(h,i + nocc,lambda_a_v_->get(h,i));
                    for (int j = 0; j < nvir; ++j){
                        Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
                    }
                }
            }
        }
        // Get the excited state orbitals: Cb(ex) = Cb(gs) * (Uo | Uv)
        Cb_->gemm(false,false,1.0,dets[0]->Cb(),TempMatrix,0.0);
    }
    if (debug_) {
        Ca_->print(outfile);
        Cb_->print(outfile);
    }
}


//    // Excited state: use specialized code

//    // Transform Fa to the ground state MO basis
//    TempMatrix->transform(Fa_,dets[0]->Ca());

//    // Set the orbital transformation matrices for the occ and vir blocks
//    // equal to the identity so that if we decide to optimize only the hole
//    // or the particle all is ok
//    Uo_->identity();
//    Uv_->identity();
//    boost::tuple<double,int,int> hole;
//    boost::tuple<double,int,int> particle;
//    // Grab the occ and vir blocks
//    // |--------|--------|
//    // |        |        |
//    // | PoFaPo |        |
//    // |        |        |
//    // |--------|--------|
//    // |        |        |
//    // |        | PvFaPv |
//    // |        |        |
//    // |--------|--------|
//    if(do_constrained_hole){
//        extract_square_subblock(TempMatrix,PoFPo_,true,dets[0]->nalphapi(),1.0e9);
//        PoFPo_->diagonalize(Uo_,lambda_o_);
//        // Sort the orbitals according to the eigenvalues of PoFaPo
//        std::vector<boost::tuple<double,int,int> > sorted_occ;
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = dets[0]->nalphapi()[h];
//            for (int i = 0; i < nocc; ++i){
//                sorted_occ.push_back(boost::make_tuple(lambda_o_->get(h,i),h,i));
//            }
//        }
//        std::sort(sorted_occ.begin(),sorted_occ.end());
//        // Extract the hole alpha orbital according to an energy criteria (this needs a generalization)
//        if (KS::options_.get_str("CDFT_EXC_TYPE") == "VALENCE"){
//            // For valence excitations select the highest lying orbital (HOMO-like)
//            hole = sorted_occ.back();
//        }else if(KS::options_.get_str("CDFT_EXC_TYPE") == "CORE"){
//            // For core excitations select the lowest lying orbital (1s-like)
//            hole = sorted_occ.front();
//        }

//    }

//    if(do_constrained_part){
//        extract_square_subblock(TempMatrix,PvFPv_,false,dets[0]->nalphapi(),1.0e9);
//        PvFPv_->diagonalize(Uv_,lambda_v_);
//        // Sort the orbitals according to the eigenvalues of PvFaPv
//        std::vector<boost::tuple<double,int,int> > sorted_vir;
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = dets[0]->nalphapi()[h];
//            int nvir = nmopi_[h] - nocc;
//            for (int i = 0; i < nvir; ++i){
//                sorted_vir.push_back(boost::make_tuple(lambda_v_->get(h,i),h,i + nocc));  // N.B. shifted to full indexing
//            }
//        }
//        std::sort(sorted_vir.begin(),sorted_vir.end());
//        // In the case of particle, we assume that we are always interested in the lowest lying orbitals
//        particle = sorted_vir.front();
//    }

//    // Form the transformation matrix that diagonalizes the PoFaPo and PvFaPv blocks
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
//            double** Uo_h = Uo_->pointer(h);
//            for (int i = 0; i < nocc; ++i){
//                epsilon_a_->set(h,i,lambda_o_->get(h,i));
//                for (int j = 0; j < nocc; ++j){
//                    Temp_h[i][j] = Uo_h[i][j];
//                }
//            }
//        }
//        if (nvir != 0){
//            double** Temp_h = TempMatrix->pointer(h);
//            double** Uv_h = Uv_->pointer(h);
//            for (int i = 0; i < nvir; ++i){
//                epsilon_a_->set(h,i + nocc,lambda_v_->get(h,i));
//                for (int j = 0; j < nvir; ++j){
//                    Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
//                }
//            }
//        }
//    }

//    // Get the excited state orbitals: Ca(ex) = Ca(gs) * (Uo | Uv)
//    Ca_->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix,0.0);
//    if(do_constrained_hole and do_constrained_part){
//        outfile->Printf("   constrained hole/particle pair :(irrep = %d,mo = %d,energy = %.6f) -> (irrep = %d,mo = %d,energy = %.6f)\n",
//                hole.get<1>(),hole.get<2>(),hole.get<0>(),
//                particle.get<1>(),particle.get<2>(),particle.get<0>());
//    }else if(do_constrained_hole and not do_constrained_part){
//        outfile->Printf("   constrained hole :(irrep = %d,mo = %d,energy = %.6f)\n",
//                hole.get<1>(),hole.get<2>(),hole.get<0>());
//    }else if(not do_constrained_hole and do_constrained_part){
//        outfile->Printf("   constrained particle :(irrep = %d,mo = %d,energy = %.6f)\n",
//                particle.get<1>(),particle.get<2>(),particle.get<0>());
//    }

//    // Save the hole and particle information and at the same time zero the columns in Ca_
//    current_excited_state = SharedExcitedState(new ExcitedState(nirrep_));
//    if(do_constrained_hole){
//        SharedVector hole_mo = Ca_->get_column(hole.get<1>(),hole.get<2>());
//        Ca_->zero_column(hole.get<1>(),hole.get<2>());
//        epsilon_a_->set(hole.get<1>(),hole.get<2>(),0.0);
//        current_excited_state->add_hole(hole.get<1>(),hole_mo,hole.get<0>(),true);
//    }
//    if(do_constrained_part){
//        SharedVector particle_mo = Ca_->get_column(particle.get<1>(),particle.get<2>());
//        Ca_->zero_column(particle.get<1>(),particle.get<2>());
//        epsilon_a_->set(particle.get<1>(),particle.get<2>(),0.0);
//        current_excited_state->add_particle(particle.get<1>(),particle_mo,particle.get<0>(),true);
//    }

//    // Adjust the occupation (nalphapi_,nbetapi_)
//    for (int h = 0; h < nirrep_; ++h){
//        nalphapi_[h] = dets[0]->nalphapi()[h];
//    }
//    nbetapi_ = dets[0]->nbetapi();
//    nalphapi_[hole.get<1>()] -= 1;
//    nalphapi_[particle.get<1>()] += 1;

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

//    // If print > 2 (diagnostics), print always
//    if((print_ > 2 || (print_ && occ_changed)) && iteration_ > 0){
//        if (Communicator::world->me() == 0)
//            outfile->Printf( "\tOccupation by irrep:\n");
//        print_occupation();
//    }

//    // Optionally, include relaxation effects
//    if(do_relax_spectators){
//        // Transform Fa to the excited state MO basis, this includes the hole and particle states
//        TempMatrix->transform(Fa_,Ca_);

//        // Zero the terms that couple the hole, particle, and the rest of the orbitals
//        // |--------|--------|
//        // |       0|0       |
//        // |       0|0       |
//        // |00000000|00000000|
//        // |--------|--------|
//        // |00000000|00000000|
//        // |       0|0       |
//        // |       0|0       |
//        // |--------|--------|
//        if(do_constrained_hole){
//            // Zero the hole couplings
//            int h = hole.get<1>();
//            int i = hole.get<2>();
//            int nmo = nmopi_[h];
//            if (nmo != 0){
//                double** Temp_h = TempMatrix->pointer(h);
//                for (int p = 0; p < nmo; ++p){
//                    if(p != i){
//                        Temp_h[i][p] = Temp_h[p][i] = 0.0;
//                    }
//                }
//            }

//        }
//        if(do_constrained_part){
//            // Zero the LUMO couplings
//            int h = particle.get<1>();
//            int i = particle.get<2>();
//            int nmo = nmopi_[h];
//            if (nmo != 0){
//                double** Temp_h = TempMatrix->pointer(h);
//                for (int p = 0; p < nmo; ++p){
//                    if(p != i){
//                        Temp_h[i][p] = Temp_h[p][i] = 0.0;
//                    }
//                }
//            }
//        }

//        TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
//        TempMatrix->copy(Ca_);
//        Ca_->gemm(false,false,1.0,TempMatrix,TempMatrix2,0.0);
//    }

//    // At this point the orbitals are sorted according to the energy but we
//    // want to make sure that the hole and the particle MO appear where they
//    // should, that is the holes in the virtual space and the particles in
//    // the occupied space.
//    // |(1) (2) ... (hole) | (particle) ...> will become
//    // |(particle) (1) (2) ...  | ... (hole)>
//    std::vector<int> naholepi = current_excited_state->aholepi();
//    std::vector<int> napartpi = current_excited_state->apartpi();
//    TempMatrix->zero();
//    TempVector->zero();
//    for (int h = 0; h < nirrep_; ++h){
//        int m = napartpi[h];  // Offset by the number of holes
//        int nso = nsopi_[h];
//        int nmo = nmopi_[h];
//        double** T_h = TempMatrix->pointer(h);
//        double** C_h = Ca_->pointer(h);
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
//    }
//    if(do_constrained_hole){
//        // Place the hole orbital in the last MO of its irrep
//        TempMatrix->set_column(hole.get<1>(),nmopi_[hole.get<1>()]-1,current_excited_state->get_hole(0,true));
//        TempVector->set(hole.get<1>(),nmopi_[hole.get<1>()]-1,hole.get<0>());
//    }
//    if(do_constrained_part){
//        // Place the particle orbital in the first MO of its irrep
//        TempMatrix->set_column(particle.get<1>(),0,current_excited_state->get_particle(0,true));
//        TempVector->set(particle.get<1>(),0,particle.get<0>());
//    }
//    Ca_->copy(TempMatrix);
//    epsilon_a_->copy(TempVector.get());

//    Ca_->print();
//    epsilon_a_->print();

//    // BETA
//    diagonalize_F(Fb_, Cb_, epsilon_b_);

//    //find_occupation();

//    if (debug_) {
//        Ca_->print(outfile);
//        Cb_->print(outfile);
//    }
double UCKS::compute_overlap(int n)
{
//    // Orthogonality test
//    Temp->gemm(false,false,1.0,state_Da[0],S_,0.0);
//    // DSC'
//    Ua->gemm(false,false,1.0,Temp,Ca_,0.0);
//    Ua->print();

//    // Temp = 1 - DS
//    Temp2->identity();
//    Temp2->subtract(Temp);
//    // (1 - DS)C'
//    Ua->gemm(false,false,1.0,Temp2,Ca_,0.0);
//    Ua->print();


//    Temp->gemm(false,false,1.0,S_,Ca_,0.0);
//    Ua->gemm(true,false,1.0,state_Da[n],Temp,0.0);
//    Ua->print();
//    // Orthogonality test
//    Temp->gemm(false,false,1.0,PvFaPv_,Ca_,0.0);
//    Ua->gemm(true,false,1.0,Ca_,Temp,0.0);
//    Ua->print();

//    // Alpha block
//    TempMatrix->gemm(false,false,1.0,S_,Ca_,0.0);
//    Ua->gemm(true,false,1.0,state_Ca[n],TempMatrix,0.0);
//    //Ua->print();
//    // Grab S_aa from Ua
//    SharedMatrix S_aa = SharedMatrix(new Matrix("S_aa",state_nalphapi[n],nalphapi_));
//    for (int h = 0; h < nirrep_; ++h) {
//        int ngs_occ = dets[0]->nalphapi()[h];
//        int nex_occ = nalphapi_[h];
//        if (ngs_occ == 0 or nex_occ == 0) continue;
//        double** Ua_h = Ua->pointer(h);
//        double** S_aa_h = S_aa->pointer(h);
//        for (int i = 0; i < ngs_occ; ++i){
//            for (int j = 0; j < nex_occ; ++j){
//                S_aa_h[i][j] = Ua_h[i][j];
//            }
//        }
//    }

//    double detS_aa = 1.0;
//    double traceS2_aa = 0.0;
//    {
//        boost::tuple<SharedMatrix, SharedVector, SharedMatrix> UsV = S_aa->svd_temps();
//        S_aa->svd(UsV.get<0>(),UsV.get<1>(),UsV.get<2>());
//        if(dets[0]->nalphapi() == nalphapi_){
//            for (int h = 0; h < nirrep_; ++h) {
//                for (int i = 0; i < UsV.get<1>()->dim(h); ++i){
//                    detS_aa *= UsV.get<1>()->get(h,i);
//                }
//            }
//        }else{
//            detS_aa = 0.0;
//        }
//        for (int h = 0; h < nirrep_; ++h) {
//            for (int i = 0; i < UsV.get<1>()->dim(h); ++i){
//                traceS2_aa += std::pow(UsV.get<1>()->get(h,i),2.0);
//            }
//        }
//    }

//    // Beta block
//    TempMatrix->gemm(false,false,1.0,S_,Cb_,0.0);
//    Ub->gemm(true,false,1.0,state_Cb[n],TempMatrix,0.0);

//    // Grab S_bb from Ub
//    SharedMatrix S_bb = SharedMatrix(new Matrix("S_bb",state_nbetapi[n],nbetapi_));

//    for (int h = 0; h < nirrep_; ++h) {
//        int ngs_occ = state_nbetapi[0][h];
//        int nex_occ = nbetapi_[h];
//        if (ngs_occ == 0 or nex_occ == 0) continue;
//        double** Ub_h = Ub->pointer(h);
//        double** S_bb_h = S_bb->pointer(h);
//        for (int i = 0; i < ngs_occ; ++i){
//            for (int j = 0; j < nex_occ; ++j){
//                S_bb_h[i][j] = Ub_h[i][j];
//            }
//        }
//    }
//    double detS_bb = 1.0;
//    double traceS2_bb = 0.0;
//    {
//        boost::tuple<SharedMatrix, SharedVector, SharedMatrix> UsV = S_bb->svd_temps();
//        S_bb->svd(UsV.get<0>(),UsV.get<1>(),UsV.get<2>());
//        if(state_nbetapi[0] == nbetapi_){
//            for (int h = 0; h < nirrep_; ++h) {
//                for (int i = 0; i < UsV.get<1>()->dim(h); ++i){
//                    detS_bb *= UsV.get<1>()->get(h,i);
//                }
//            }
//        }else{
//            detS_bb = 0.0;
//        }
//        for (int h = 0; h < nirrep_; ++h) {
//            for (int i = 0; i < UsV.get<1>()->dim(h); ++i){
//                traceS2_bb += std::pow(UsV.get<1>()->get(h,i),2.0);
//            }
//        }
//    }
//    outfile->Printf("   det(S_aa) = %.6f det(S_bb) = %.6f  <Phi|Phi'> = %.6f\n",detS_aa,detS_bb,detS_aa * detS_bb);
//    outfile->Printf("   <Phi'|Poa|Phi'> = %.6f  <Phi'|Pob|Phi'> = %.6f  <Phi'|Po|Phi'> = %.6f\n",nalpha_ - traceS2_aa,nbeta_ - traceS2_bb,nalpha_ - traceS2_aa + nbeta_ - traceS2_bb);
//    TempMatrix->transform(state_Da[0],S_);
//    TempMatrix2->transform(TempMatrix,Ca_);
//    return (detS_aa * detS_bb);
    return 0;
}s

void UCKS::form_C_CP_algorithm()
{

    // Excited state: use specialized code
    int nstate = static_cast<int>(state_Ca.size());
    outfile->Printf("  Computing %d optimal particle orbitals\n",nstate);

    for (int m = 0; m < nstate; ++m){

    }
    // Transform Fa to the ground state MO basis
    TempMatrix->transform(Fa_,dets[0]->Ca());

    // Set the orbital transformation matrices for the occ and vir blocks
    // equal to the identity so that if we decide to optimize only the hole
    // or the particle all is ok
    Uo_->identity();
    Uv_->identity();
    boost::tuple<double,int,int> hole;
    boost::tuple<double,int,int> particle;
    // Grab the occ and vir blocks
    // |--------|--------|
    // |        |        |
    // | PoFaPo |        |
    // |        |        |
    // |--------|--------|
    // |        |        |
    // |        | PvFaPv |
    // |        |        |
    // |--------|--------|
    if(do_constrained_hole){
        PoFPo_->identity();
        PoFPo_->scale(1.0e9);
        for (int h = 0; h < nirrep_; ++h){
            int nocc = dets[0]->nalphapi()[h];
            if (nocc != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** PoFaPo_h = PoFPo_->pointer(h);
                for (int i = 0; i < nocc; ++i){
                    for (int j = 0; j < nocc; ++j){
                        PoFaPo_h[i][j] = Temp_h[i][j];
                    }
                }
            }
        }
        PoFPo_->diagonalize(Uo_,lambda_o_);
        // Sort the orbitals according to the eigenvalues of PoFaPo
        std::vector<boost::tuple<double,int,int> > sorted_occ;
        for (int h = 0; h < nirrep_; ++h){
            int nocc = dets[0]->nalphapi()[h];
            for (int i = 0; i < nocc; ++i){
                sorted_occ.push_back(boost::make_tuple(lambda_o_->get(h,i),h,i));
            }
        }
        std::sort(sorted_occ.begin(),sorted_occ.end());
        // Extract the hole alpha orbital according to an energy criteria (this needs a generalization)
        if (KS::options_.get_str("CDFT_EXC_TYPE") == "VALENCE"){
            // For valence excitations select the highest lying orbital (HOMO-like)
            hole = sorted_occ.back();
        }else if(KS::options_.get_str("CDFT_EXC_TYPE") == "CORE"){
            // For core excitations select the lowest lying orbital (1s-like)
            hole = sorted_occ.front();
        }

    }

    if(do_constrained_part){
        PvFPv_->identity();
        PvFPv_->scale(1.0e9);
        for (int h = 0; h < nirrep_; ++h){
            int nocc = dets[0]->nalphapi()[h];
            int nvir = nmopi_[h] - nocc;
            if (nvir != 0){
                double** Temp_h = TempMatrix->pointer(h);
                double** PvFaPv_h = PvFPv_->pointer(h);
                for (int i = 0; i < nvir; ++i){
                    for (int j = 0; j < nvir; ++j){
                        PvFaPv_h[i][j] = Temp_h[i + nocc][j + nocc];
                    }
                }
            }
        }
        PvFPv_->diagonalize(Uv_,lambda_v_);
        // Sort the orbitals according to the eigenvalues of PvFaPv
        std::vector<boost::tuple<double,int,int> > sorted_vir;
        for (int h = 0; h < nirrep_; ++h){
            int nocc = dets[0]->nalphapi()[h];
            int nvir = nmopi_[h] - nocc;
            for (int i = 0; i < nvir; ++i){
                sorted_vir.push_back(boost::make_tuple(lambda_v_->get(h,i),h,i + nocc));  // N.B. shifted to full indexing
            }
        }
        std::sort(sorted_vir.begin(),sorted_vir.end());
        // In the case of particle, we assume that we are always interested in the lowest lying orbitals
        particle = sorted_vir.front();
    }

    // Form the transformation matrix that diagonalizes the PoFaPo and PvFaPv blocks
    // |----|----|
    // | Uo | 0  |
    // |----|----|
    // | 0  | Uv |
    // |----|----|
    TempMatrix->zero();
    for (int h = 0; h < nirrep_; ++h){
        int nocc = dets[0]->nalphapi()[h];
        int nvir = nmopi_[h] - nocc;
        if (nocc != 0){
            double** Temp_h = TempMatrix->pointer(h);
            double** Uo_h = Uo_->pointer(h);
            for (int i = 0; i < nocc; ++i){
                epsilon_a_->set(h,i,lambda_o_->get(h,i));
                for (int j = 0; j < nocc; ++j){
                    Temp_h[i][j] = Uo_h[i][j];
                }
            }
        }
        if (nvir != 0){
            double** Temp_h = TempMatrix->pointer(h);
            double** Uv_h = Uv_->pointer(h);
            for (int i = 0; i < nvir; ++i){
                epsilon_a_->set(h,i + nocc,lambda_v_->get(h,i));
                for (int j = 0; j < nvir; ++j){
                    Temp_h[i + nocc][j + nocc] = Uv_h[i][j];
                }
            }
        }
    }

    // Get the excited state orbitals: Ca(ex) = Ca(gs) * (Uo | Uv)
    Ca_->gemm(false,false,1.0,dets[0]->Ca(),TempMatrix,0.0);
    if(do_constrained_hole and do_constrained_part){
        outfile->Printf("   constrained hole/particle pair :(irrep = %d,mo = %d,energy = %.6f) -> (irrep = %d,mo = %d,energy = %.6f)\n",
                hole.get<1>(),hole.get<2>(),hole.get<0>(),
                particle.get<1>(),particle.get<2>(),particle.get<0>());
    }else if(do_constrained_hole and not do_constrained_part){
        outfile->Printf("   constrained hole :(irrep = %d,mo = %d,energy = %.6f)\n",
                hole.get<1>(),hole.get<2>(),hole.get<0>());
    }else if(not do_constrained_hole and do_constrained_part){
        outfile->Printf("   constrained particle :(irrep = %d,mo = %d,energy = %.6f)\n",
                particle.get<1>(),particle.get<2>(),particle.get<0>());
    }

    // Save the hole and particle information and at the same time zero the columns in Ca_
    current_excited_state = SharedExcitedState(new ExcitedState);
    if(do_constrained_hole){
        SharedVector hole_mo = Ca_->get_column(hole.get<1>(),hole.get<2>());
        Ca_->zero_column(hole.get<1>(),hole.get<2>());
        epsilon_a_->set(hole.get<1>(),hole.get<2>(),0.0);
        current_excited_state->add_hole(hole.get<1>(),hole_mo,true);
    }
    if(do_constrained_part){
        SharedVector particle_mo = Ca_->get_column(particle.get<1>(),particle.get<2>());
        Ca_->zero_column(particle.get<1>(),particle.get<2>());
        epsilon_a_->set(particle.get<1>(),particle.get<2>(),0.0);
        current_excited_state->add_particle(particle.get<1>(),particle_mo,true);
    }

    // Adjust the occupation (nalphapi_,nbetapi_)
    for (int h = 0; h < nirrep_; ++h){
        nalphapi_[h] = dets[0]->nalphapi()[h];
        nbetapi_[h] = state_nbetapi[0][h];
    }
    nalphapi_[hole.get<1>()] -= 1;
    nalphapi_[particle.get<1>()] += 1;

    int old_socc[8];
    int old_docc[8];
    for(int h = 0; h < nirrep_; ++h){
        old_socc[h] = soccpi_[h];
        old_docc[h] = doccpi_[h];
    }

    for (int h = 0; h < nirrep_; ++h) {
        soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
        doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
    }

    bool occ_changed = false;
    for(int h = 0; h < nirrep_; ++h){
        if( old_socc[h] != soccpi_[h] || old_docc[h] != doccpi_[h]){
            occ_changed = true;
            break;
        }
    }

    // If print > 2 (diagnostics), print always
    if((print_ > 2 || (print_ && occ_changed)) && iteration_ > 0){
        if (Communicator::world->me() == 0)
            outfile->Printf( "\tOccupation by irrep:\n");
        print_occupation();
    }

    // Optionally, include relaxation effects
    if(do_relax_spectators){
        // Transform Fa to the excited state MO basis, this includes the hole and particle states
        TempMatrix->transform(Fa_,Ca_);

        // Zero the terms that couple the hole, particle, and the rest of the orbitals
        // |--------|--------|
        // |       0|0       |
        // |       0|0       |
        // |00000000|00000000|
        // |--------|--------|
        // |00000000|00000000|
        // |       0|0       |
        // |       0|0       |
        // |--------|--------|
        if(do_constrained_hole){
            // Zero the hole couplings
            int h = hole.get<1>();
            int i = hole.get<2>();
            int nmo = nmopi_[h];
            if (nmo != 0){
                double** Temp_h = TempMatrix->pointer(h);
                for (int p = 0; p < nmo; ++p){
                    if(p != i){
                        Temp_h[i][p] = Temp_h[p][i] = 0.0;
                    }
                }
            }

        }
        if(do_constrained_part){
            // Zero the LUMO couplings
            int h = particle.get<1>();
            int i = particle.get<2>();
            int nmo = nmopi_[h];
            if (nmo != 0){
                double** Temp_h = TempMatrix->pointer(h);
                for (int p = 0; p < nmo; ++p){
                    if(p != i){
                        Temp_h[i][p] = Temp_h[p][i] = 0.0;
                    }
                }
            }
        }

        TempMatrix->diagonalize(TempMatrix2,epsilon_a_);
        TempMatrix->copy(Ca_);
        Ca_->gemm(false,false,1.0,TempMatrix,TempMatrix2,0.0);
    }

    // At this point the orbitals are sorted according to the energy but we
    // want to make sure that the hole and the particle MO appear where they
    // should, that is the holes in the virtual space and the particles in
    // the occupied space.
    // |(1) (2) ... (hole) | (particle) ...> will become
    // |(particle) (1) (2) ...  | ... (hole)>
    std::vector<int> naholepi = current_excited_state->aholepi();
    std::vector<int> napartpi = current_excited_state->apartpi();
    TempMatrix->zero();
    TempVector->zero();
    for (int h = 0; h < nirrep_; ++h){
        int m = napartpi[h];  // Offset by the number of holes
        int nso = nsopi_[h];
        int nmo = nmopi_[h];
        double** T_h = TempMatrix->pointer(h);
        double** C_h = Ca_->pointer(h);
        for (int p = 0; p < nmo; ++p){
            // Is this MO a hole or a particle?
            if(std::fabs(epsilon_a_->get(h,p)) > 1.0e-6){
                TempVector->set(h,m,epsilon_a_->get(h,p));
                for (int q = 0; q < nso; ++q){
                    T_h[q][m] = C_h[q][p];
                }
                m += 1;
            }

        }
    }
    if(do_constrained_hole){
        // Place the hole orbital in the last MO of its irrep
        TempMatrix->set_column(hole.get<1>(),nmopi_[hole.get<1>()]-1,current_excited_state->get_hole(0,true));
        TempVector->set(hole.get<1>(),nmopi_[hole.get<1>()]-1,hole.get<0>());
    }
    if(do_constrained_part){
        // Place the particle orbital in the first MO of its irrep
        TempMatrix->set_column(particle.get<1>(),0,current_excited_state->get_particle(0,true));
        TempVector->set(particle.get<1>(),0,particle.get<0>());
    }
    Ca_->copy(TempMatrix);
    epsilon_a_->copy(TempVector.get());

    // BETA
    diagonalize_F(Fb_, Cb_, epsilon_b_);

    //find_occupation();

    if (debug_) {
        Ca_->print(outfile);
        Cb_->print(outfile);
    }
}


//void UCKS::form_D()
//{
//    for (int h = 0; h < nirrep_; ++h) {
//        int nso = nsopi_[h];
//        int nmo = nmopi_[h];
//        int na = nalphapi_[h];
//        int nb = nbetapi_[h];

//        if (nso == 0 || nmo == 0) continue;

//        double* aocc_num_h = aocc_num_->pointer(h);
//        double* bocc_num_h = bocc_num_->pointer(h);
//        double** Ca = Ca_->pointer(h);
//        double** Cb = Cb_->pointer(h);
//        double** Da = Da_->pointer(h);
//        double** Db = Db_->pointer(h);

//        if (na == 0)
//            ::memset(static_cast<void*>(Da[0]), '\0', sizeof(double)*nso*nso);
//        if (nb == 0)
//            ::memset(static_cast<void*>(Db[0]), '\0', sizeof(double)*nso*nso);
//        for (int mu = 0; mu < nso; ++mu){
//            for (int nu = 0; nu < nso; ++nu){
//                for (int p = 0; p < nmo; ++p){
//                    Da[mu][nu] += Ca[mu][p] * Ca[nu][p] * aocc_num_h[p];
//                    Db[mu][nu] += Cb[mu][p] * Cb[nu][p] * bocc_num_h[p];
//                }
//            }
//        }
//    }

//    Dt_->copy(Da_);
//    Dt_->add(Db_);

//    if (debug_) {
//        outfile->Printf( "in UCKS::form_D:\n");
//        Da_->print();
//        Db_->print();
//    }
//}

//        if(do_penalty){
//            // Find the alpha HOMO of the ground state wave function
//            int homo_h = 0;
//            int homo_p = 0;
//            double homo_e = -1.0e9;
//            for (int h = 0; h < nirrep_; ++h) {
//                int nocc = dets[0]->nalphapi()[h] - 1;
//                if (nocc < 0) continue;
//                if(state_epsilon_a[0]->get(h,nocc) > homo_e){
//                    homo_h = h;
//                    homo_p = nocc;
//                    homo_e = state_epsilon_a[0]->get(h,nocc);
//                }
//            }
//            outfile->Printf("  The HOMO orbital has energy %.9f and is %d of irrep %d.\n",homo_e,homo_p,homo_h);
//            Pa = SharedMatrix(factory_->create_matrix("Penalty matrix alpha"));
//            for (int mu = 0; mu < nsopi_[homo_h]; ++mu){
//                for (int nu = 0; nu < nsopi_[homo_h]; ++nu){
//                    double P_mn = 1000000.0 * dets[0]->Ca()->get(homo_h,mu,homo_p) * dets[0]->Ca()->get(homo_h,nu,homo_p);
//                    Pa->set(homo_h,mu,nu,P_mn);
//                }
//            }
//            Pa->transform(S_);
//        }



//    if(gs_scf_ and do_excitation){
//        // Form the projected Fock matrices
//        // Po = DS
//        Temp->gemm(false,false,1.0,state_Da[0],S_,0.0);
//        // SDFDS
//        PoFaPo_->transform(Fa_,Temp);
//        // Temp = 1 - DS
//        Temp2->identity();
//        Temp2->subtract(Temp);
//        PvFaPv_->transform(Fa_,Temp2);
//    }



////// EXCITED STATES
//        Temp->copy(Fa_);
//        Temp->transform(Ca_);

//        Temp2->copy(Fb_);
//        Temp2->transform(Ca_);

//        Ub->copy(Temp);
//        Ub->add(Temp2);
//        Ub->scale(0.5);

//        for (int h = 0; h < nirrep_; ++h) {
//          // CO
//          for (int i = 0; i < doccpi_[h]; ++i) {
//            for (int j = doccpi_[h]; j < doccpi_[h] + soccpi_[h]; ++j) {
//                Ub->set(h,i,j,Temp2->get(h,i,j));
//                Ub->set(h,j,i,Temp2->get(h,j,i));
//            }
//          }
//          for (int i = doccpi_[h]; i < doccpi_[h] + soccpi_[h]; ++i) {
//            for (int j = doccpi_[h] + soccpi_[h]; j < nmopi_[h]; ++j) {
//                Ub->set(h,i,j,Temp->get(h,i,j));
//                Ub->set(h,j,i,Temp->get(h,j,i));
//            }
//          }
//        }
//        Ub->diagonalize(Ua,epsilon_a_);
//        Temp->copy(Ca_);
//        Ca_->gemm(false,false,1.0,Temp,Ua,0.0);
//        Cb_->copy(Ca_);
//        epsilon_b_->copy(epsilon_a_.get());

//        Temp->diagonalize(Ua,epsilon_a_);
//        Temp2->diagonalize(Ub,epsilon_b_);
//        Temp->copy(Ca_);
//        Temp2->copy(Cb_);
//        Ca_->gemm(false,false,1.0,Temp,Ua,0.0);
//        Cb_->gemm(false,false,1.0,Temp2,Ub,0.0);


//        diagonalize_F(Fa_, Ca_, epsilon_a_);
//        Temp->copy(Fb_);
//        Temp->transform(Ca_);
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = doccpi_[h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0 and nvir!= 0){
//                double** Temp_h = Temp->pointer(h);
//                for (int i = 0; i < nocc; ++i){
////                    for (int j = 0; j < nvir; ++j){
////                        Temp_h[i][j + nocc] = Temp_h[j + nocc][i] = 0.0;
//                    for (int j = doccpi_[h] + soccpi_[h]; j < nmopi_[h]; ++j){
//                        Temp_h[i][j] = Temp_h[j][i] = 0.0;
//                    }
//                }
//            }
//        }
//        Temp->diagonalize(Temp2,epsilon_b_);
//        Temp->copy(Ca_);
//        Cb_->gemm(false,false,1.0,Temp,Temp2,0.0);

//        for (int h = 0; h < nirrep_; ++h){
//            int nso = nsopi_[h];
//            if (nso != 0){
//                double** Temp_h = Temp->pointer(h);
//                for (int p = 0; p < nso; ++p){
//                    epsilon_b_->set(h,p,Temp_h[p][p]);
//                }
//            }
//        }


//        // Transform Fa to the ground state MO basis
//        Temp->transform(Fb_,state_Cb[0]);
//        // Grab the occ and vir blocks
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = state_nbetapi[0][h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** PoFbPo_h = PoFbPo_->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    for (int j = 0; j < nocc; ++j){
//                        PoFbPo_h[i][j] = Temp_h[i][j];
//                    }
//                }
//            }
//            if (nvir != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** PvFbPv_h = PvFbPv_->pointer(h);
//                for (int i = 0; i < nvir; ++i){
//                    for (int j = 0; j < nvir; ++j){
//                        PvFbPv_h[i][j] = Temp_h[i + nocc][j + nocc];
//                    }
//                }
//            }
//        }
//        PoFbPo_->diagonalize(Uob,lambda_ob);
//        PvFbPv_->diagonalize(Uvb,lambda_vb);

//        Temp->zero();
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = state_nbetapi[0][h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Uob_h = Uob->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    epsilon_b_->set(h,i,lambda_ob->get(h,i));
//                    for (int j = 0; j < nocc; ++j){
//                        Temp_h[i][j] = Uob_h[i][j];
//                    }
//                }
//            }
//            if (nvir != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Uvb_h = Uvb->pointer(h);
//                for (int i = 0; i < nvir; ++i){
//                    epsilon_b_->set(h,i + nocc,lambda_vb->get(h,i));
//                    for (int j = 0; j < nvir; ++j){
//                        Temp_h[i + nocc][j + nocc] = Uvb_h[i][j];
//                    }
//                }
//            }
//        }
//        // Get the new orbitals
//        Cb_->gemm(false,false,1.0,state_Cb[0],Temp,0.0);
//        Temp->transform(Fb_,Cb_);
//        Temp->print();



//        // Grab the occ and vir blocks
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = new_occupation[h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Fcan_h = Fcan->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    for (int j = 0; j < nocc; ++j){
//                        Fcan_h[i][j] = Temp_h[i][j];
//                    }
//                }
//            }
//        }
//        Fcan->diagonalize(Ucan,lambda_can);
//        Temp->zero();
//        for (int h = 0; h < nirrep_; ++h){
//            int nocc = new_occupation[h];
//            int nvir = nmopi_[h] - nocc;
//            if (nocc != 0){
//                double** Temp_h = Temp->pointer(h);
//                double** Ucan_h = Ucan->pointer(h);
//                for (int i = 0; i < nocc; ++i){
//                    epsilon_a_->set(h,i,lambda_can->get(h,i));
//                    for (int j = 0; j < nocc; ++j){
//                        Temp_h[i][j] = Ucan_h[i][j];
//                    }
//                }
//            }
//            if (nvir != 0){
//                double** Temp_h = Temp->pointer(h);
//                for (int i = 0; i < nvir; ++i){
//                    Temp_h[i + nocc][i + nocc] = 1.0;
//                }
//            }
//        }

//        Temp->transform(Fa_,Ca_);
//        // Get the new orbitals
//        Temp2->copy(Ca_);
//        Ca_->gemm(false,false,1.0,Temp2,Temp,0.0);

//        Dimension new
//        SharedVector epsilon_ao_ = SharedVector(factory_->create_vector());
//        SharedVector epsilon_av_ = SharedVector(factory_->create_vector());
//        diagonalize_F(PoFaPo_, Temp,  epsilon_ao_);
//        diagonalize_F(PvFaPv_, Temp2, epsilon_av_);
//        int homo_h = 0;
//        int homo_p = 0;
//        double homo_energy = -1.0e10;
//        int lumo_h = 0;
//        int lumo_p = 0;
//        double lumo_energy = +1.0e10;
//        for (int h = 0; h < nirrep_; ++h){
//            int nmo  = nmopi_[h];
//            int nso  = nsopi_[h];
//            if (nmo == 0 or nso == 0) continue;
//            double** Ca_h  = Ca_->pointer(h);
//            double** Cao_h = Temp->pointer(h);
//            double** Cav_h = Temp2->pointer(h);
//            int no = 0;
//            for (int p = 0; p < nmo; ++p){
//                if(std::fabs(epsilon_ao_->get(h,p)) > 1.0e-6 ){
//                    for (int mu = 0; mu < nmo; ++mu){
//                        Ca_h[mu][no] = Cao_h[mu][p];
//                    }
//                    epsilon_a_->set(h,no,epsilon_ao_->get(h,p));
//                    if (epsilon_ao_->get(h,p) > homo_energy){
//                        homo_energy = epsilon_ao_->get(h,p);
//                        homo_h = h;
//                        homo_p = no;
//                    }
//                    no++;
//                }
//            }
//            for (int p = 0; p < nmo; ++p){
//                if(std::fabs(epsilon_av_->get(h,p)) > 1.0e-6 ){
//                    for (int mu = 0; mu < nmo; ++mu){
//                        Ca_h[mu][no] = Cav_h[mu][p];
//                    }
//                    epsilon_a_->set(h,no,epsilon_av_->get(h,p));
//                    if (epsilon_av_->get(h,p) < lumo_energy){
//                        lumo_energy = epsilon_av_->get(h,p);
//                        lumo_h = h;
//                        lumo_p = no;
//                    }
//                    no++;
//                }
//            }
//        }
//        // Shift the HOMO orbital in the occupied space


//int old_socc[8];
//int old_docc[8];
//for(int h = 0; h < nirrep_; ++h){
//    old_socc[h] = soccpi_[h];
//    old_docc[h] = doccpi_[h];
//}

//for (int h = 0; h < nirrep_; ++h) {
//    soccpi_[h] = std::abs(nalphapi_[h] - nbetapi_[h]);
//    doccpi_[h] = std::min(nalphapi_[h] , nbetapi_[h]);
//}

//bool occ_changed = false;
//for(int h = 0; h < nirrep_; ++h){
//    if( old_socc[h] != soccpi_[h] || old_docc[h] != doccpi_[h]){
//        occ_changed = true;
//        break;
//    }
//}

//// If print > 2 (diagnostics), print always
//if((print_ > 2 || (print_ && occ_changed)) && iteration_ > 0){
//    if (Communicator::world->me() == 0)
//        outfile->Printf( "\tOccupation by irrep:\n");
//    print_occupation();
//}

//outfile->Printf( "\tNA   [ ");
//for(int h = 0; h < nirrep_-1; ++h) outfile->Printf( " %4d,", nalphapi_[h]);
//outfile->Printf( " %4d ]\n", nalphapi_[nirrep_-1]);
//outfile->Printf( "\tNB   [ ");
//for(int h = 0; h < nirrep_-1; ++h) outfile->Printf( " %4d,", nbetapi_[h]);
//outfile->Printf( " %4d ]\n", nbetapi_[nirrep_-1]);

//// Compute the density matrices with the new occupation
//form_D();

//// Compute the triplet energy from the density matrices
//double triplet_energy = compute_E();

//        if(nexclude_occ == 0 and nexclude_vir){
//            // Find the lowest single excitations
//            std::vector<boost::tuple<double,int,int,int,int> > sorted_exc;
//            // Loop over occupied MOs
//            for (int hi = 0; hi < nirrep_; ++hi){
//                int nocci = ref_scf_->nalphapi_[0][hi];
//                for (int i = 0; i < nocc; ++i){
//                    for (int ha = 0; ha < nirrep_; ++ha){
//                        int nocca = ref_scf_->nalphapi_[0][ha];
//                        for (int a = nocca; a < nmopi_[ha]; ++a){
//                            sorted_exc.push_back(boost::make_tuple(
//                        }
//                    }
//                    int nocc = dets[0]->nalphapi()[h];

//                int nvir = nmopi_[h] - nocc;
//                for (int i = 0; i < nocc; ++i){
//                    sorted_occ.push_back(boost::make_tuple(lambda_o->get(h,i),h,i));
//                }
//                for (int i = 0; i < nvir; ++i){
//                    sorted_vir.push_back(boost::make_tuple(lambda_v->get(h,i),h,i));
//                }
//            }
//            std::sort(sorted_occ.begin(),sorted_occ.end());
//            std::sort(sorted_vir.begin(),sorted_vir.end());
//        }

//    if(do_roks){
//        moFa_->transform(Fa_, Ca_);
//        moFb_->transform(Fb_, Ca_);
//        /*
//        * Fo = open-shell fock matrix = 0.5 Fa
//        * Fc = closed-shell fock matrix = 0.5 (Fa + Fb)
//        *
//        * Therefore
//        *
//        * 2(Fc-Fo) = Fb
//        * 2Fo = Fa
//        *
//        * Form the effective Fock matrix, too
//        * The effective Fock matrix has the following structure
//        *          |  closed     open    virtual
//        *  ----------------------------------------
//        *  closed  |    Fc     2(Fc-Fo)    Fc
//        *  open    | 2(Fc-Fo)     Fc      2Fo
//        *  virtual |    Fc       2Fo       Fc
//        */
//        Feff_->copy(moFa_);
//        Feff_->add(moFb_);
//        Feff_->scale(0.5);
//        for (int h = 0; h < nirrep_; ++h) {
//            for (int i = doccpi_[h]; i < doccpi_[h] + soccpi_[h]; ++i) {
//                // Set the open/closed portion
//                for (int j = 0; j < doccpi_[h]; ++j) {
//                    double val = moFb_->get(h, i, j);
//                    Feff_->set(h, i, j, val);
//                    Feff_->set(h, j, i, val);
//                }
//                // Set the open/virtual portion
//                for (int j = doccpi_[h] + soccpi_[h]; j < nmopi_[h]; ++j) {
//                    double val = moFa_->get(h, i, j);
//                    Feff_->set(h, i, j, val);
//                    Feff_->set(h, j, i, val);
//                }
//            }
//        }

//        // Form the orthogonalized SO basis Feff matrix, for use in DIIS
//        soFeff_->copy(Feff_);
//        soFeff_->back_transform(Ct_);
//    }


//    // SUHF Contributions
//    double suhf_weight = std::pow(0.5,std::max(10.0 - iteration_,1.0));
//    TempMatrix->transform(Db_,S_);
//    TempMatrix->scale(- 4.0 * suhf_weight * KS::options_.get_double("CDFT_SUHF_LAMBDA"));
//    Fa_->add(TempMatrix);
//    TempMatrix->transform(Da_,S_);
//    TempMatrix->scale(- 4.0 * suhf_weight * KS::options_.get_double("CDFT_SUHF_LAMBDA"));
//    Fb_->add(TempMatrix);


#endif // JUNK_H
