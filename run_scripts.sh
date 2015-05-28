#!/usr/bin/sh
##
## Script for running the scripts
##
git pull origin master
python analyzeData.py
mv distributions.png /afs/cern.ch/user/j/jbadillo/www/plots/ml/distributions.png
