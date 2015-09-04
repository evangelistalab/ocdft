#!/usr/bin/env python

import sys
import os
import subprocess

psi4command = ""

if len(sys.argv) == 1:
    cmd = ["which","psi4"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    res = p.stdout.readlines()
    psi4command = res[0][:-1]
elif len(sys.argv) == 2:
    psi4command = sys.argv[1]

print "Running test using psi4 executable found in:\n%s" % psi4command

ocdft_tests = ["ocdft-ch","ocdft-chp", "ocdft-cp", "ocdft-cis", "ocdft-chpfb", "ocdft-core", "ocdft-rew"]

tests = ocdft_tests
maindir = os.getcwd()
for d in tests:
    print "\nRunning test %s\n" % d
    os.chdir(d)
    subprocess.call([psi4command])
    os.chdir(maindir)
