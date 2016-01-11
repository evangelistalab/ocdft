#include <map>
#include <numeric>
#include <vector>
#include <regex>
#include <boost/regex.hpp>
#include <libscf_solver/hf.h>

#include <masses.h>
#include <Z_to_element.h>
#include <element_to_Z.h>

#include "boost/format.hpp"

#include "aosubspace.h"

std::vector<std::string> mysplit(const std::string& input, const std::string& regex);

std::vector<std::string> mysplit(const std::string& input, const std::string& regex) {
    // passing -1 as the submatch index parameter performs splitting
    boost::regex re(regex);
    boost::sregex_token_iterator
        first{input.begin(), input.end(), re, -1},
        last;
    return {first, last};
}

namespace psi{ namespace aosubspace {

AOSubspace::AOSubspace(boost::shared_ptr<Molecule> molecule,boost::shared_ptr<BasisSet> basis)
{
    startup();
}

AOSubspace::AOSubspace(std::vector<std::string> subspace_str,boost::shared_ptr<Molecule> molecule, boost::shared_ptr<BasisSet> basis)
    : subspace_str_(subspace_str), molecule_(molecule), basis_(basis)
{
    startup();
}

int AOSubspace::find_subspace(int iteration)
{
    parse_basis_set();
    parse_subspace(iteration);
    return 0;
}

void AOSubspace::startup()
{
    //outfile->Printf("  ---------------------------------------\n");
    //outfile->Printf("    Atomic Orbital Subspace\n");
    //outfile->Printf("    written by Francesco A. Evangelista\n");
    //outfile->Printf("  ---------------------------------------\n");

    lm_labels_cartesian_ = {{"S"},
                            {"PX","PY","PZ"},
                            {"DX2","DXY","DXZ","DY2","DYZ","DZ2"},
                            {"FX3","FX2Y","FX2Z","FXY2","FXYZ","FXZ2","FY3","FY2Z","FYZ2","FZ3"}};

    l_labels_ = {"S","P","D","F","G","H","I","K","L","M"};

    lm_labels_sperical_ = {{"S"},
                           {"PZ","PX","PY"},
                           {"DZ2","DXZ","DYZ","DX2Y2","DXY"},
                           {"FZ3","FXZ2","FYZ2","FZX2-ZY2","FXYZ","FX3-3XY2","F3X2Y-Y3"},
                           {"G1","G2","G3","G4","G5","G6","G7","G8","G9"},
                           {"H1","H2","H3","H4","H5","H6","H7","H8","H9","H10","H11"}};

    for (int l = 0; l < (int)lm_labels_sperical_.size(); ++l){
        for (int m = 0; m < (int)lm_labels_sperical_[l].size(); ++m){
            labels_sperical_to_lm_[lm_labels_sperical_[l][m]] = {std::make_pair(l,m)};
        }
    }
    for (int l = 0; l < (int)l_labels_.size(); ++l){
        std::vector<std::pair<int,int>> lm_vec;
        for (int m = 0; m < 2 * l + 1; ++m){
            lm_vec.push_back(std::make_pair(l,m));
        }
        labels_sperical_to_lm_[l_labels_[l]] = lm_vec;
    }
}

const std::vector<int>& AOSubspace::subspace()
{
    return subspace_;
}

std::vector<std::string> AOSubspace::aolabels(std::string str_format) const
{
    std::vector<std::string> aolbl;
    for (const AOInfo& aoinfo : aoinfo_vec_){
        std::string s = boost::str( boost::format(str_format)
                                    % (aoinfo.A() + 1)
                                    % atomic_labels[aoinfo.Z()]
                                    % (aoinfo.element_count() + 1)
                                    % aoinfo.n()
                                    % lm_labels_sperical_[aoinfo.l()][aoinfo.m()]);
        aolbl.push_back(s);
    }
    return aolbl;
}

const std::vector<AOInfo>& AOSubspace::aoinfo() const
{
    return aoinfo_vec_;
}

int AOSubspace::parse_subspace(int iteration)
{   
    if(iteration==0){
	    outfile->Printf("\n   List of subspaces:");
	    for (const std::string& s : subspace_str_){
		outfile->Printf(" %s",s.c_str());
	    }
    	    outfile->Printf("\n");
    }
    for (const std::string& s : subspace_str_){
        parse_subspace_entry(s);
    }
    if(iteration==0){
	    outfile->Printf("\n   Subspace contains AOs:\n");
	    for (size_t i = 0; i < subspace_.size(); ++i){
		outfile->Printf("  %6d",subspace_[i] + 1);
		if ((i + 1) % 8 == 0)
		    outfile->Printf("\n");
	    }
	    outfile->Printf("\n");
    }
    return 0;
}

void AOSubspace::parse_subspace_entry(const std::string& s)
{
    // The regex to parse the entries
    boost::regex re("([a-zA-Z]{1,2})([1-9]+)?-?([1-9]+)?\\(?((?:\\/?[1-9]{1}[SPDF]{1}[a-zA-Z]*)*)\\)?");
    boost::smatch match;

    Element_to_Z etoZ;

    boost::regex_match(s,match,re);
    if (debug_){
        outfile->Printf("\n  Parsing entry: '%s'\n",s.c_str());
        for (boost::ssub_match base_sub_match : match){
            std::string base = base_sub_match.str();
            outfile->Printf("  --> '%s'\n",base.c_str());
        }
    }
    if (match.size() == 5){
        // Find Z
        int Z = static_cast<int>(etoZ[match[1].str()]);

        // Find the range
        int minA = 0;
        int maxA = atom_to_aos_[Z].size();
        if (match[2].str().size() != 0){
            minA = stoi(match[2].str()) - 1;
            if (match[3].str().size() == 0){
                maxA = minA + 1;
            }else{
                maxA = stoi(match[3].str());
            }
        }

        if (debug_){
            outfile->Printf("  Element %s -> %d\n",match[1].str().c_str(),Z);
            outfile->Printf("  Range %d -> %d\n",minA,maxA);
        }

        // Find the subset of AOs
        if (match[4].str().size() != 0){
            // Include some of the AOs
            std::vector<std::string> vec_str = mysplit(match[4].str(),"/");
            for (std::string str : vec_str){
                int n = atoi(&str[0]);
                str.erase(0,1);
                if (labels_sperical_to_lm_.count(str) > 0){
                    for (std::pair<int,int> lm : labels_sperical_to_lm_[str]){
                        int l = lm.first;
                        int m = lm.second;
                        if (debug_) outfile->Printf("     -> %s (n = %d,l = %d, m = %d)\n",str.c_str(),n,l,m);
                        for (int A = minA; A < maxA; A++){
                            for (int pos : atom_to_aos_[Z][A]){
                                if ((aoinfo_vec_[pos].n() == n) and (aoinfo_vec_[pos].l() == l) and (aoinfo_vec_[pos].m() == m)){
                                    if (debug_) outfile->Printf("     + found at position %d\n",pos);
                                    subspace_.push_back(pos);
                                }
                            }
                        }

                    }
                }else{
                    outfile->Printf("  AO label '%s' is not valid.\n",str.c_str());
                }
            }
        }else{
            // Include all the AOs
            for (int A = minA; A < maxA; A++){
                for (int pos : atom_to_aos_[Z][A]){
                    if (debug_) outfile->Printf("     + found at position %d\n",pos);
                    subspace_.push_back(pos);
                }
            }
        }
    }
}

void AOSubspace::parse_basis_set()
{
    // Form a map that lists all functions on a given atom and with a given ang. momentum
    std::map<std::pair<int,int>,std::vector<int>> atom_am_to_f;
    bool pure_am = basis_->has_puream();

    if (debug_){
        outfile->Printf("\n  Parsing basis set\n");
        outfile->Printf("  Pure Angular Momentum: %s\n",pure_am ? "True" : "False");
    }

    int count = 0;

    std::vector<int> element_count(130);

    for (int A = 0; A < molecule_->natom(); A++) {
        int Z = static_cast<int>(round(molecule_->Z(A)));

        int n_shell = basis_->nshell_on_center(A);

        std::vector<int> n_count(10,1);
        std::iota(n_count.begin(),n_count.end(),1);

        std::vector<int> ao_list;

        if (debug_) outfile->Printf("\n  Atom %d (%s) has %d shells\n",A,Z_to_element[Z].c_str(),n_shell);

        for (int Q = 0; Q < n_shell; Q++){
            const GaussianShell& shell = basis_->shell(A,Q);
            int nfunction = shell.nfunction();
            int l = shell.am();
            if (debug_) outfile->Printf("    Shell %d: L = %d, N = %d (%d -> %d)\n",Q,l,nfunction,count,count + nfunction);
            for (int m = 0; m < nfunction; ++m){
                AOInfo ao(A,Z,element_count[Z],n_count[l],l,m);
                aoinfo_vec_.push_back(ao);
                ao_list.push_back(count);
                count += 1;
            }
            n_count[l] += 1;  // increase the angular momentum count
        }

        atom_to_aos_[Z].push_back(ao_list);

        element_count[Z] += 1; // increase the element count
    }
}

}}
