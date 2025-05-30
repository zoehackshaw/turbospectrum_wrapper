# turbospectrum_wrapper
This repository contains my wrapper for creating synthetic spectra (although ... *shudder* python 2.7) with turbospectrum

Main things to edit when downloading and running for yourself:
- defined paths on lines 20-25
- contopac dir on line 125
- lammin, lammax, and deltalam in make_babsmabysn_file
- linelist paths on lines 255-266
- specpath on 342
- the BACCHUS paths on 375 - 406

## Quick Start:
1. In the same directory as this file `ipython`
2. `import turbospec_synthetic_spec as ts`
3. Define at least teff, logg, and feh
4. `ts.Turbosynth(stellarpar=[teff,logg,feh],specfile='test_spec_new.txt')`
5. You can also edit free abundances, vsini, and any other parameters you desire, ex: `ts.Turbosynth(stellarpar=[4300,1.5,-0.5],specfile='test_spec_c1.txt',free_abund=[['C',0.1]],vsini=5)`
Note that the inputs for the free abundances are in [X/H]

## To Do for future Zoe:
- make generalizable
- probably update to a python version from this century

Feel free to email me with any questions zoehackshaw@utexas.edu because this repository is sure to be forgotten about!
