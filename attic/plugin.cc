void FASNOCIS(SharedWavefunction ref_wfn, Options& options);
void NOCI(SharedWavefunction ref_wfn, Options& options);

void NOCI(SharedWavefunction ref_wfn, Options& options) {
    std::shared_ptr<PSIO> psio = PSIO::shared_object();
    std::string reference = options.get_str("REFERENCE");
    std::vector<double> energies;
    bool valence = true;
    if (options.get_str("CDFT_EXC_TYPE") == "CORE") {
        valence = false;
    }
    std::vector<SharedDeterminant> dets;
    // Store the irrep, multiplicity, total energy, excitation energy, oscillator
    // strength
    std::vector<std::tuple<int, int, double, double, double, double, double, double>> state_info;
    if (reference == "RHF") {
        throw InputException("NOCI based on a RHF reference is not implemented ",
                             "REFERENCE to UHF", __FILE__, __LINE__);
    } else if (reference == "UHF") {
        // Run a ground state computation first
        outfile->Printf(" PV this is first done.\n");
        SharedWavefunction ref_scf = SharedWavefunction(new scf::NOCI(ref_wfn, options, psio));
        double gs_energy = ref_scf->compute_energy();
        outfile->Printf("\n  %11.4f", gs_energy);
        energies.push_back(gs_energy);

        // Push the ground state determinant
        dets.push_back(SharedDeterminant(new scf::Determinant(
            ref_scf->Ca(), ref_scf->Cb(), ref_scf->nalphapi(), ref_scf->nbetapi())));
        // I am going to ask user to give me
        //  OCC_ACTIVE based on each irrep
        //  VIR_ACTIVE based on each irrep

        int nirrep = ref_scf->nirrep();
        std::vector<int> occ_frozen, vir_frozen;
        std::vector<std::tuple<int, int, double>> occup_a;

        for (int h = 0; h < nirrep; ++h) {
            occ_frozen.push_back(options["AOCC_FROZEN"][h].to_integer());
            vir_frozen.push_back(options["AVIR_FROZEN"][h].to_integer());
        }

        for (auto& i : occ_frozen) {
            outfile->Printf("\n  occ_frozen = %d", i);
        }

        // find out how many are occupied alpha and beta
        Dimension nalphapi = ref_scf->nalphapi();
        Dimension nbetapi = ref_scf->nbetapi();

        Dimension nsopi_ = ref_scf->nsopi();
        Dimension nmopi_ = ref_scf->nmopi();

        // need to know my active mos based on each irrep
        // active_mos have (irrep, mo_number_info)

        std::vector<std::pair<int, int>> frozen_mos;
        std::vector<std::pair<int, int>> frozen_occ_a;
        std::vector<std::pair<int, int>> frozen_occ_b;

        for (int h = 0; h < nirrep; ++h) {
            for (int i = 0; i < occ_frozen[h]; ++i) {
                frozen_mos.push_back(std::make_pair(h, nalphapi[h] - 1 - i));
                if (valence) {
                    frozen_occ_a.push_back(std::make_pair(h, nalphapi[h] - 1 - i));
                    frozen_occ_b.push_back(std::make_pair(h, nbetapi[h] - 1 - i));
                } else {
                    frozen_occ_a.push_back(std::make_pair(h, i));
                    frozen_occ_b.push_back(std::make_pair(h, i));
                }
            }
        }

        for (int h = 0; h < nirrep; ++h) {
            for (int i = 0; i < vir_frozen[h]; ++i) {
                frozen_mos.push_back(std::make_pair(h, nalphapi[h] + i));
            }
        }

        for (size_t n = 0; n < occup_a.size(); ++n) {
            outfile->Printf("\n  occup_a = %d %d mo = %f\n", std::get<0>(occup_a[n]),
                            std::get<1>(occup_a[n]), std::get<2>(occup_a[n]));
        }

        state_info.push_back(std::make_tuple(0, 1, gs_energy, 0.0, 0.0, 0.0, 0.0, 0.0));

        SharedMatrix Ca_gs_;
        SharedMatrix Cb_gs_;

        Ca_gs_ = SharedMatrix(new Matrix("Ca_gs_", nsopi_, nmopi_));
        Cb_gs_ = SharedMatrix(new Matrix("Cb_gs_", nsopi_, nmopi_));

        Ca_gs_->copy(ref_scf->Ca());
        Cb_gs_->copy(ref_scf->Cb());

        int nstates = 0;
        for (int h = 0; h < nirrep; ++h) {
            nstates += occ_frozen[h] * vir_frozen[h];
        }

        for (auto& h_p : frozen_occ_a) {
            int irrep = h_p.first;
            int fmo = h_p.second;
            std::pair<int, int> swap_occ(irrep, fmo);
            int vrt = vir_frozen[irrep];
            for (int state_a = 1; state_a <= vir_frozen[irrep]; ++state_a) {
                int state_b = 0;
                SharedWavefunction new_scf;
                std::shared_ptr<Wavefunction> new_scf_ =
                    std::shared_ptr<Wavefunction>(new scf::NOCI(
                        new_scf, options, psio, state_a, swap_occ, state_b, frozen_occ_a,
                        frozen_occ_b, frozen_mos, occ_frozen, vir_frozen, Ca_gs_, Cb_gs_, valence));
                new_scf = new_scf_;
                // Process::environment.wavefunction().reset();
                // Process::environment.set_wavefunction(new_scf);
                double new_energy = new_scf->compute_energy();
                energies.push_back(new_energy);
                dets.push_back(SharedDeterminant(
                    new scf::Determinant(new_scf->Ca(), new_scf->Cb(), new_scf->nalphapi(),
                                         new_scf->nbetapi(), fmo, state_a, state_b, vrt)));
            }
            //   }//occup
        } // irrep

        for (auto& h_p : frozen_occ_b) {
            int irrep = h_p.first;
            int fmo = h_p.second;
            std::pair<int, int> swap_occ(irrep, fmo);

            int vrt = vir_frozen[irrep];
            for (int state_b = 1; state_b <= vir_frozen[irrep]; ++state_b) {
                int state_a = 0;
                SharedWavefunction new_scf;
                std::shared_ptr<Wavefunction> new_scf_ =
                    std::shared_ptr<Wavefunction>(new scf::NOCI(
                        new_scf, options, psio, state_a, swap_occ, state_b, frozen_occ_a,
                        frozen_occ_b, frozen_mos, occ_frozen, vir_frozen, Ca_gs_, Cb_gs_, valence));
                new_scf = new_scf_;
                // Process::environment.wavefunction().reset();
                // Process::environment.set_wavefunction(new_scf);
                double new_energy = new_scf->compute_energy();
                energies.push_back(new_energy);
                dets.push_back(SharedDeterminant(
                    new scf::Determinant(new_scf->Ca(), new_scf->Cb(), new_scf->nalphapi(),
                                         new_scf->nbetapi(), fmo, state_a, state_b, vrt)));
            }
            //   }//occup
        } // irrep
        scf::NOCI_Hamiltonian noci_H(options, dets);
        noci_H.compute_energy(energies);
    }
}

void FASNOCIS(SharedWavefunction ref_wfn, Options &options) {
  std::shared_ptr<PSIO> psio = PSIO::shared_object();
  std::string reference = options.get_str("REFERENCE");
  std::vector<double> energies;
  std::vector<SharedDeterminant> dets;

  // Store the irrep, multiplicity, total energy, excitation energy, oscillator
  // strength
  std::vector<std::tuple<int, int, double, double, double>> state_info;

  if (reference == "RHF") {
    throw InputException("Constrained RKS is not implemented ",
                         "REFERENCE to UKS", __FILE__, __LINE__);
  } else if (reference == "UHF") {
    // Run a ground state computation first
    SharedWavefunction ref_scf =
        SharedWavefunction(new scf::FASNOCIS(ref_wfn, options, psio));

    double gs_energy = ref_scf->compute_energy();
    energies.push_back(gs_energy);

    // Push the ground state determinant
    dets.push_back(SharedDeterminant(
        new scf::Determinant(ref_scf->Ca(), ref_scf->Cb(), ref_scf->nalphapi(),
                             ref_scf->nbetapi())));

    Process::environment.globals["HF NOCI energy"] = gs_energy;
    // Print a molden file
    if (options["MOLDEN_WRITE"].has_changed()) {
      std::shared_ptr<MoldenWriter> molden(new MoldenWriter(ref_scf));
      std::string filename =
          get_writer_file_prefix("OCDFT") + "." + to_string(0) + ".molden";
      psi::scf::HF *hf = (psi::scf::HF *)ref_scf.get();
      SharedVector occA = hf->occupation_a();
      SharedVector occB = hf->occupation_b();
      molden->write(filename, ref_scf->Ca(), ref_scf->Cb(),
                    ref_scf->epsilon_a(), ref_scf->epsilon_b(), occA, occB);
    }

    // Optimize the orbitals of each frozen configuration
    Dimension nalphapi = ref_scf->nalphapi();
    Dimension nbetapi = ref_scf->nbetapi();

    int nirrep = ref_scf->nirrep();
    Dimension adoccpi(nirrep, "Number of doubly occupied active MOs per irrep");
    std::vector<int> occ_frozen, vir_frozen;
    size_t nofrz = 0;
    size_t nvfrz = 0;
    for (int h = 0; h < nirrep; ++h) {
      size_t no_h = options["OCC_FROZEN"][h].to_integer();
      size_t nv_h = options["VIR_FROZEN"][h].to_integer();
      occ_frozen.push_back(no_h);
      vir_frozen.push_back(nv_h);
      nofrz += no_h;
      nvfrz += nv_h;
      adoccpi[h] = nalphapi[h] - no_h;
    }

    size_t nfrz = nofrz + nvfrz;

    std::vector<std::pair<size_t, size_t>> frozen_mos;
    for (int h = 0; h < nirrep; ++h) {
      for (int i = 0; i < occ_frozen[h]; ++i) {
        frozen_mos.push_back(std::make_pair(h, nalphapi[h] - 1 - i));
      }
    }
    for (int h = 0; h < nirrep; ++h) {
      for (int i = 0; i < vir_frozen[h]; ++i) {
        frozen_mos.push_back(std::make_pair(h, nalphapi[h] + i));
      }
    }

    for (auto &h_p : frozen_mos) {
      outfile->Printf("\n  irrep = %d mo = %d", h_p.first, h_p.second);
    }

    outfile->Printf("\n  nofrz = %d nfrz = %d", nofrz, nfrz);

    // Create all alpha-alpha singly-excited determinants
    for (size_t i = 0; i < nofrz; ++i) {
      for (size_t a = nofrz; a < nfrz; ++a) {
        outfile->Printf("\n  ==> State %d -> %d\n", i, a);

        // Build the vector of frozen occupied orbitals
        std::vector<size_t> aocc, bocc;
        for (size_t j = 0; j < nofrz; ++j) {
          if (j != i)
            aocc.push_back(j);
          bocc.push_back(j);
        }
        aocc.push_back(a);

        std::shared_ptr<Wavefunction> new_scf =
            std::shared_ptr<Wavefunction>(new scf::FASNOCIS(
                options, psio, ref_scf, frozen_mos, aocc, bocc, adoccpi));

        new_scf->compute_energy();
        dets.push_back(SharedDeterminant(
            new scf::Determinant(new_scf->Ca(), new_scf->Cb(),
                                 new_scf->nalphapi(), new_scf->nbetapi())));
      }
    }
    // Create all beta-beta singly-excited determinants
    for (size_t i = 0; i < nofrz; ++i) {
      for (size_t a = nofrz; a < nfrz; ++a) {
        outfile->Printf("\n  ==> State %d -> %d\n", i, a);

        // Build the vector of frozen occupied orbitals
        std::vector<size_t> aocc, bocc;
        for (size_t j = 0; j < nofrz; ++j) {
          if (j != i)
            bocc.push_back(j);
          aocc.push_back(j);
        }
        bocc.push_back(a);

        std::shared_ptr<Wavefunction> new_scf =
            std::shared_ptr<Wavefunction>(new scf::FASNOCIS(
                options, psio, ref_scf, frozen_mos, aocc, bocc, adoccpi));

        new_scf->compute_energy();
        dets.push_back(SharedDeterminant(
            new scf::Determinant(new_scf->Ca(), new_scf->Cb(),
                                 new_scf->nalphapi(), new_scf->nbetapi())));
      }
    }
    scf::NOCI_Hamiltonian noci_H(options, dets);
    noci_H.compute_energy(energies);

    SharedVector evals = noci_H.evals();

    // Save the state energy to
    for (size_t n = 0; n < evals->dim(); ++n) {
      std::string str = "NOCI ENERGY STATE " + std::to_string(n);
      Process::environment.globals[str] = evals->get(n);
    }

  } else {
    throw InputException("Unknown reference " + reference, "REFERENCE",
                         __FILE__, __LINE__);
  }

  // Set this early because the callback mechanism uses it.
  // Process::environment.wavefunction().reset();
}