#
#@BEGIN LICENSE
#
# myplugin by Psi4 Developer, a plugin to:
#
# PSI4: an ab initio quantum chemistry software package
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
#@END LICENSE
#

#"""Plugin docstring.

#"""
#__version__ = '0.1'
#__author__  = 'Psi4 Developer'

## Load Python modules
#from pymodule import *

## Load C++ plugin
#import os
#import psi4
#plugdir = os.path.split(os.path.abspath(__file__))[0]
#sofile = plugdir + '/cdft.so'
#psi4.plugin_load(sofile)


"""Plugin docstring.

"""
__version__ = '0.1'
__author__  = 'Psi4 Developer'

# Load Python modules
from .pymodule import *

# Load C++ plugin
import os
import psi4
plugdir = os.path.split(os.path.abspath(__file__))[0]
sofile = plugdir + '/' + os.path.split(plugdir)[1] + '.so'
psi4.core.plugin_load(sofile)
