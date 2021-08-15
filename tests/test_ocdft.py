#!/usr/bin/env python

import sys
import os
import subprocess

import argparse


psi4command = ""
cmd = ["which","psi4"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
res = p.stdout.readlines()
psi4command = res[0][:-1]

#if len(sys.argv) == 1:
#    cmd = ["which","psi4"]
#    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
#    res = p.stdout.readlines()
#    psi4command = res[0][:-1]
#elif len(sys.argv) == 2:
#    psi4command = sys.argv[1]

print "Running test using psi4 executable found in:\n%s" % psi4command

options = {"OCDFT"    : [None,"OCDFT arg"],
           "NOCI"    : [None,"NOCI arg"]}

parser = argparse.ArgumentParser(description='Control which test cases will run')
parser.add_argument("--ocdft", const=True, default=False, nargs='?',
                 help='Run only OCDFT test cases(string, default = false)')
parser.add_argument("--noci", const=True, default=False, nargs='?',
                 help='Run only OCDFT test cases(string, default = false)')
args = parser.parse_args()

options["OCDFT"][0] = str(args.ocdft)
options["NOCI"][0] = str(args.noci)


# "noci-1","noci-ocdft-hf-core-ex-CO","noci-ocdft-hf-val-ex-CO","noci-val-ex-CO","ocdft-rew",
:qocdft_tests = ["ocdft-ch","ocdft-chp",  "ocdft-chpfb", "ocdft-cis", "ocdft-core","ocdft-cp", "ocdft-cube"]

if options["NOCI"][0] == 'True':
	ocdft_tests = ["noci-1","noci-ocdft-hf-core-ex-CO","noci-ocdft-hf-val-ex-CO","noci-val-ex-CO"]
if options["OCDFT"][0] == 'True':
        ocdft_tests = ["ocdft-ch","ocdft-chp", "ocdft-cp", "ocdft-cis", "ocdft-chpfb", "ocdft-core",  "ocdft-cube", "ocdft-vtc"]


tests = ocdft_tests
maindir = os.getcwd()
for d in tests:
    print "\nRunning test %s\n" % d
    os.chdir(d)
    subprocess.call([psi4command])
    os.chdir(maindir)
