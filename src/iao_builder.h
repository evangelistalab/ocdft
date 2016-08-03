/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2016 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#ifndef IAO_BUILDER_H
#define IAO_BUILDER_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

namespace psi {

class Matrix;
class BasisSet;

class IAOBuilder {

protected:

    // => Overall Parameters <= //

    /// Print flag
    int print_;
    /// Debug flug
    int debug_;
    /// Bench flag
    int bench_;

    /// Relative convergence criteria
    double convergence_;
    /// Maximum number of iterations
    int maxiter_;

    // => IAO Parameters <= //

    /// Use ghost IAOs?
    bool use_ghosts_;
    /// IAO localization power (4 or 2)
    int power_; 
    /// Metric condition for IAO
    double condition_;
    
    /// Occupied orbitals, in primary basis
    boost::shared_ptr<Matrix> C_;
    /// Primary orbital basis set
    boost::shared_ptr<BasisSet> primary_;
    /// MinAO orbital baiss set
    boost::shared_ptr<BasisSet> minao_;

    // => Stars Parameters <= //
    
    /// Do stars treatment?
    bool use_stars_;
    /// Charge completeness for two-center orbitals
    double stars_completeness_;
    /// List of centers for stars
    std::vector<int> stars_;

    // => IAO Data <= //

    /// Map from non-ghosted to full atoms: true_atoms[ind_true] = ind_full
    std::vector<int> true_atoms_; 
    /// Map from non-ghosted IAOs to full IAOs: true_iaos[ind_true] = ind_full
    std::vector<int> true_iaos_;
    /// Map from non-ghosted IAOs to non-ghosted atoms 
    std::vector<int> iaos_to_atoms_;

    /// Overlap matrix in full basis
    boost::shared_ptr<Matrix> S_;
    /// Non-ghosted IAOs in full basis
    boost::shared_ptr<Matrix> A_;
    
    
    /// Set defaults
    void common_init();
    

public:

    // => Constructors <= //

    IAOBuilder(
        boost::shared_ptr<BasisSet> primary, 
        boost::shared_ptr<BasisSet> minao, 
        boost::shared_ptr<Matrix> C);
    
    virtual ~IAOBuilder();

    /// Build IBO with defaults from Options object (including MINAO_BASIS)
    static boost::shared_ptr<IAOBuilder> build(
        boost::shared_ptr<BasisSet> primary, 
        boost::shared_ptr<Matrix> C,
        Options& options);
    /// Build the IAOs for exporting
    SharedMatrix build_iaos();

    // => Knobs <= //

    void set_print(int print) { print_ = print; }
    void set_debug(int debug) { debug_ = debug; }
    void set_bench(int bench) { bench_ = bench; }
    void set_convergence(double convergence) { convergence_ = convergence; }
    void set_maxiter(int maxiter) { maxiter_ = maxiter; }
    void set_use_ghosts(bool use_ghosts) { use_ghosts_ = use_ghosts; }
    void set_condition(double condition) { condition_ = condition; }
    void set_power(double power) { power_ = power; }
    void set_use_stars(bool use_stars) { use_stars_ = use_stars; }
    void set_stars_completeness(double stars_completeness) { stars_completeness_ = stars_completeness; }
    void set_stars(const std::vector<int>& stars) { stars_ = stars; }

};

} // Namespace psi

#endif
