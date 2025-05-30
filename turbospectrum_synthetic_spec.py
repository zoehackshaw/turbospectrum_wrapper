from numpy import *
import pylab as p
#from matplotlib import *
#import pyfits
import astropy.io.fits as pyfits
import numpy.random as rand
import pickle
import subprocess
import os
import sys
from subprocess import Popen, PIPE
import warnings
import logging
import numpy as np
from scipy.special import j1


# adopted from group github file abund_mod/turbospectrum_spec.py

turbo_path = '/home/zoeh/Code/BACCHUS/Turbo-v15.1/EXPORT-15.1/exec-gf-v15.1/'
modpath = '/home/zoeh/Code/synthetic_spec/models/' 
interp_path = '/home/zoeh/Code/BACCHUS/INTERPOL/'
solarabu_file = '/home/zoeh/Code/BACCHUS/solabu.dat'
sys.path.insert(0, os.path.abspath(turbo_path))
faltbo3dir =  '/home/zoeh/Code/BACCHUS/bin/'

# this is useful for the free abundances later!
elements = np.array([
    ["H", 1], ["He", 2], ["Li", 3], ["Be", 4], ["B", 5], ["C", 6], ["N", 7], ["O", 8], 
    ["F", 9], ["Ne", 10], ["Na", 11], ["Mg", 12], ["Al", 13], ["Si", 14], ["P", 15], ["S", 16], 
    ["Cl", 17], ["Ar", 18], ["K", 19], ["Ca", 20], ["Sc", 21], ["Ti", 22], ["V", 23], ["Cr", 24], 
    ["Mn", 25], ["Fe", 26], ["Co", 27], ["Ni", 28], ["Cu", 29], ["Zn", 30], ["Ga", 31], ["Ge", 32], 
    ["As", 33], ["Se", 34], ["Br", 35], ["Kr", 36], ["Rb", 37], ["Sr", 38], ["Y", 39], ["Zr", 40], 
    ["Nb", 41], ["Mo", 42], ["Tc", 43], ["Ru", 44], ["Rh", 45], ["Pd", 46], ["Ag", 47], ["Cd", 48], 
    ["In", 49], ["Sn", 50], ["Sb", 51], ["Te", 52], ["I", 53], ["Xe", 54], ["Cs", 55], ["Ba", 56], 
    ["La", 57], ["Ce", 58], ["Pr", 59], ["Nd", 60], ["Pm", 61], ["Sm", 62], ["Eu", 63], ["Gd", 64], 
    ["Tb", 65], ["Dy", 66], ["Ho", 67], ["Er", 68], ["Tm", 69], ["Yb", 70], ["Lu", 71], ["Hf", 72], 
    ["Ta", 73], ["W", 74], ["Re", 75], ["Os", 76], ["Ir", 77], ["Pt", 78], ["Au", 79], ["Hg", 80], 
    ["Tl", 81], ["Pb", 82], ["Bi", 83], ["Po", 84], ["At", 85], ["Rn", 86], ["Fr", 87], ["Ra", 88], 
    ["Ac", 89], ["Th", 90], ["Pa", 91], ["U", 92], ["Np", 93], ["Pu", 94], ["Am", 95], ["Cm", 96], 
    ["Bk", 97], ["Cf", 98], ["Es", 99], ["Fm", 100], ["Md", 101], ["No", 102], ["Lr", 103], ["Rf", 104], 
    ["Db", 105], ["Sg", 106], ["Bh", 107], ["Hs", 108], ["Mt", 109], ["Ds", 110], ["Rg", 111], ["Cn", 112], 
    ["Nh", 113], ["Fl", 114], ["Mc", 115], ["Lv", 116], ["Ts", 117], ["Og", 118]
])

def vsini_calc(wave, flux, vsini): #from starfish
	dv = 2.99792458e5 * np.median(np.diff(wave) / wave[:-1])

	freq = np.fft.rfftfreq(flux.shape[-1], dv)
	ub = 2.0 * np.pi * vsini * freq
	# Remove 0th frequency
	ub = ub[1:]
	sb = np.empty(len(ub)+1)
	sb[0] = 1
	sb[1:] = j1(ub) / ub - 3.0 * np.cos(ub) / (2 * ub ** 2) + 3.0 * np.sin(ub) / (2 * ub ** 3)

	flux_final = np.fft.irfft(np.fft.rfft(flux)*sb, n=flux.shape[-1])
	return flux_final

def generate_model_atmo(T, g, f, modpath=modpath, starname=None, **kwargs):
    '''
    PURPOSE:
    Generates an interpolated model atmosphere from the MARCs grid using the interpol.f routine developed by
    T. Masseron (Masseron, PhD Thesis, 2006). This is a python wrapper for that Fortran code.

    INPUT:
    (1) Temperature, (2) Surface gravity, (3) Metallicity of the model atmosphere

    OUTPUT:
    Full path of model atmosphere and the type of interpolation done: Plane-parallel (P), or Spherical (S)

    OPTIONAL:
    modpath = the path to write the model atmosphere
    '''
    if kwargs.get('verbose', True):
        stdout = open('/dev/null', 'w')
        stderr = subprocess.STDOUT
    else:
        stdout = open('/dev/null', 'w')
        stderr = subprocess.STDOUT

    # ----- defines the point at which Plane-Parallel vs spherical model atmosphere models are used
    if g >= 3.0:
        if starname is not None:
            modelname = '%s_%ig%.2fm0.0z%.2f.int' % (starname, T, g, f)  # modelname
        else:
            modelname = '%ig%.2fm0.0z%.2f.int' % (T, g, f)

        try:
            p = subprocess.Popen([interp_path + 'interpol_planparallel_in.com', str(T), str(g), str(f), modelname],
                                 stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
            stdout, stderr = p.communicate()
            print(stdout)
        except subprocess.CalledProcessError:
            raise RuntimeError('Plane-Parallel Model atmosphere interpolation failed ....')
        sphere = 'F'
    else:
        if starname is not None:
            modelname = '%s_%ig%.2fm1.0z%.2f.int' % (starname, T, g, f)
        else:
            modelname = '%ig%.2fm1.0z%.2f.int' % (T, g, f)

        try:
            p = subprocess.Popen([interp_path + 'interpol_spherical_in.com', str(T), str(g), str(f), modelname],
                                 stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
            stdout, stderr = p.communicate()
        except subprocess.CalledProcessError:
            raise RuntimeError('Spherical Model atmosphere interpolation failed ....')
        sphere = 'T'

    print("./%s" % modelname)
    print(modpath + '%s' % modelname)

    # Move the generated file to the modpath
    import shutil
    try:
        shutil.move("./%s" % modelname, modpath + '%s' % modelname)
    except IOError as e:
        raise RuntimeError("Failed to move file: %s" % e)

    return modelname, sphere


def make_babsmabysn_file(lammin=3700,lammax=5000,deltalam=0.01, modelname='5777g4.44m0.0z-1.50.int', \
	contopacdir ='/home/zoeh/Code/BACCHUS/contopac/', modpath=modpath, alpha=None, free_abund=None , freeisotope=None, c12_c13 = None,**kwargs):
	'''
	--------------------------------------------------------------------------------
	| PURPOSE:
	| Generate the parameter file for both the babsma and bsyn codes for turbospectrum
	|
	| INPUT:
	| free_abund = [[element,log(eps)]]
	|
	| OUTPUT:
	|
    | HETDEX: lammin=3470, lammax=5540, deltalam=1
    | 2.7m TS: lammin=3700, lammax=8999, deltalam=~0.05  
	|
	| turbvel was set to 1.0
	|
	--------------------------------------------------------------------------------
	'''
	sphere = kwargs.get('sphere','F'
	turbvel=kwargs.get('turbvel',1.5)
	T, g, metal = kwargs.get('stellarpar')
	modpath = kwargs.get('modpath',modpath)
	specpath = kwargs.get('specpath','/home/zoeh/Code/synthetic_spec/specs/')
	specfile= kwargs.get('specfile')
	spectype = kwargs.get('spectype','Flux') #FLUX OR INTENSITY
	if spectype != 'Flux' and spectype != 'Intensity':
		raise ValueError('spectype must be Flux or INTENSITY only (currently it is %s)!'%spectype)


	#-----checks that model metallicity and input metallicity are consistent
	modelmetal = float(modelname.split('z')[1].split('.int')[0])
	if modelmetal > metal+0.05 or modelmetal < metal-0.05:
		warnings.warn('Atmopshere model (%.2f) and input metallicity (%.2f) not consistent; proceed with caution'%(modelmetal,metal),RuntimeWarning)
		print('--WARNING: Atmopshere model (%.2f) and input metallicity (%.2f) not consistent; proceed with caution'%(modelmetal,metal))
	#-----------

	#------ assume an alpha enhancment ----
	if alpha is None:
		if metal < -1.0:
			alpha_set = 0.4
		elif metal > -1.0 and metal <0.0:
			alpha_set = -0.4*metal
		else:
			alpha_set = 0
		alpha = kwargs.get('alpha',alpha_set)

	sprocess = kwargs.get('s-process',0)
        rprocess = kwargs.get('r-process',0)



	#-----build bysn----
	s1 ="'LAMBDA_MIN:'    '%.3f'\n"%lammin
	s1 += "'LAMBDA_MAX:'    '%.3f'\n"%lammax
	s1 += "'LAMBDA_STEP:'    '%.3f'\n"%deltalam
	s1 += "'INTENSITY/FLUX:' '%s'\n"%spectype
	s1 += "'COS(THETA)    :' '1.00'\n"
	s1 += "'ABFIND        :' '.false.'\n"
	s1 += "'MODELOPAC:' '%s%sopac'\n"%(contopacdir,modelname.split('.int')[0]+'t%.2fl%i-%i'%(turbvel,lammin,lammax))
	s1 += "'RESULTFILE :' '%s'\n"%(specpath+specfile)
	s1 += "'METALLICITY:'    '%.2f'\n"%metal
	s1 += "'ALPHA/Fe   :'    '%.2f'\n"%alpha
	s1 += "'HELIUM     :'    '0.00'\n"
	s1 += "'R-PROCESS  :'    '%.2f'\n"%rprocess
	s1 += "'S-PROCESS  :'    '%.2f'\n"%sprocess

	#---- Allow for user input abundances in the form [Abund, value]
    
	if free_abund is None:
		ind_abunds = "'INDIVIDUAL ABUNDANCES:'   '0'\n"
	else:
		ind_abunds = "'INDIVIDUAL ABUNDANCES:'   '%i'\n"%(shape(free_abund)[0])
		#els, Z, logeps  =  loadtxt(solarabu_file,dtype=str,usecols=(0,1,2),unpack=True)
		els, logeps  =  loadtxt(solarabu_file,dtype=str,usecols=(0,1),unpack=True)
                if shape(free_abund)[1] ==2:
			for i in arange(len(free_abund)):
				elind = where(els == free_abund[i][0])[0]
				if len(elind) != 1:
					raise ValueError('element %s not found in %s'%(free_abund[i][0],solarabu_file))
					print(type(free_abund[i][0]))
				else:
					elind = elind[0]
				#ind_abunds += '%i  %.2f\n'%(int(Z[elind]),float(logeps[elind])+float(free_abund[i][1]))
                                # hardcoding Carbon for now
                                Z = int(elements[elements[:,0] == free_abund[i][0]][0][1])
		                ind_abunds += '%i  %.2f\n'%(int(Z),float(logeps[elind])+float(free_abund[i][1]))
                else:
			raise ValueError('shape of free_abundances is not correct... Please try again... free_abund = [[element,log(eps)]]')

	#---- Define ISTOPES ----- TODO : make these user inputs
	s1 += ind_abunds + '\n'

	if c12_c13 == None:
		if freeisotope != None:
			print('here222')
			s1 += "'ISOTOPES : ' '%i'\n"%(len(freeisotope)+2)
			s1 += '3.006  0.075\n'
			s1 += '3.007  0.925\n'
			for i in range(len(freeisotope)):
				s1 += str(list(freeisotope.keys())[i])+'   '+str(freeisotope[list(freeisotope.keys())[i]])+'\n'
		else:
			print('not here')
			s1 += "'ISOTOPES : ' '2'\n"
			s1 += '3.006  0.075\n'
			s1 += '3.007  0.925\n'
	else:
		if freeisotope != None:
			print('here222')
			s1 += "'ISOTOPES : ' '%i'\n"%(len(freeisotope)+4)
			s1 += '3.006  0.075\n'
			s1 += '3.007  0.925\n'

			c12 = c12_c13/(c12_c13+1.)
			c13 = 1.- c12_c13/(c12_c13+1.)
			s1 += '6.012  '+str(c12)+'\n'
			s1 += '6.013  '+str(c13)+'\n'
			for i in range(len(freeisotope)):
				s1 += str(list(freeisotope.keys())[i])+'   '+str(freeisotope[list(freeisotope.keys())[i]])+'\n'
		else:
			print('not here')
			s1 += "'ISOTOPES : ' '4'\n"
			s1 += '3.006  0.075\n'
			s1 += '3.007  0.925\n'
			c12 = c12_c13/(c12_c13+1.)
			c13 = 1.- c12_c13/(c12_c13+1.)
			s1 += '6.012  '+str(c12)+'\n'
			s1 += '6.013  '+str(c13)+'\n'



	linelistdir = kwargs.get('linelistdir','/home/zoeh/Code/BACCHUS/linelists/GAIA-ESO/v5/')
	molecular_linelistdir = kwargs.get('molecular_linelistdir','/home/zoeh/Code/BACCHUS/linelists/GAIA-ESO/v5/bsyn/')
	#linelist_files = kwargs.get('linelist',['vald-6700-6720.list'])
	linelist_files = kwargs.get('linelist',['3000-4200_Nov11.list','ges_atom_hfs-iso_v5_t1.txt_newalternate.bsyn'])
	# linelist_files = kwargs.get('linelist',['ges_atom_hfs-iso_v5_t1.txt_newalternate.bsyn'])
	#linelist_files = kwargs.get('linelist',['turbo_line_300_1100.bsyn','turbo_vald_infrared.bsyn'])
	molecular_files= []
	molecular_files = kwargs.get('mlinelist',['12C12C_GESv5.bsyn','12C13C_GESv5.bsyn','12C14N_GESv5.bsyn','12C15N_GESv5.bsyn',
		'12CH_GESv5.bsyn','13C13C_GESv5.bsyn','13C14N_GESv5.bsyn','13CH_GESv5.bsyn','16OH_GESv5.bsyn','24MgH_GESv5.bsyn','25MgH_GESv5.bsyn','26MgH_GESv5.bsyn'])

	s1 += "'NFILES   :' '%i'\n"%(len(linelist_files)+1+len(molecular_files))
	s1 +='/home/zoeh/Code/BACCHUS/DATA/Hlinedata\n'
	for i in arange(len(linelist_files)):
		s1 += linelistdir+linelist_files[i] + '\n'
	for i in arange(len(molecular_files)):
		s1 += molecular_linelistdir+molecular_files[i] + '\n'
	s1 +="'SPHERICAL:'  '%s'\n"%sphere
  	s1 += '  30\n'
  	s1 += '  300.00\n'
 	s1+= '  15\n'
 	s1 += '  1.30\n'




 	#------- babsma----
	s2 ="'LAMBDA_MIN:'    '%.3f'\n"%lammin
	s2 += "'LAMBDA_MAX:'    '%.3f'\n"%lammax
	s2 += "'LAMBDA_STEP:'    '%.3f'\n"%deltalam
	s2 += "'MODELINPUT:' '%s'\n"%(modpath+modelname)
	s2 += "'MARCS-FILE:' '.false.'\n" #if it is not a marcs model atmo (i.e. it is a interoplated one) needs to be false
	s2 += "'MODELOPAC:' '%s%sopac'\n"%(contopacdir,modelname.split('.int')[0]+'t%.2fl%i-%i'%(turbvel,lammin,lammax))
	s2 += "'METALLICITY:'    '%.2f'\n"%metal
	s2 += "'ALPHA/Fe   :'    '%.2f'\n"%alpha
	s2 += "'HELIUM     :'    '0.00'\n"
	s2 += "'R-PROCESS  :'    '%.2f'\n"%rprocess
	s2 += "'S-PROCESS  :'    '%.2f'\n"%sprocess
	s2 += ind_abunds + '\n'
	s2 += "'XIFIX:' 'T'\n"
	s2 += '%.2f'%turbvel
	try:
		os.remove('%s%sopac'%(contopacdir,modelname.split('.int')[0]+'t%.2fl%i-%i'%(turbvel,lammin,lammax)))
	except OSError:
		pass

	f = open('/home/zoeh/Code/synthetic_spec/babsma.par', "w") ; f.write(s2) ; f.close()
	f1 = open('/home/zoeh/Code/synthetic_spec/bysn.par', "w") ; f1.write(s1) ; f1.close()
        print('babsma and bysn files created successfully!')
	return s2,s1




def run_faltbo3(infile, output, FWHM, profile=2, binpath=faltbo3dir, verbose=False):
    """
    Runs faltbo3.f code which convolves spectrum to appropriate resolution. 
    Can be replaced in future with Python code.
    """
    import re
    import subprocess

    if verbose:
        stdout = None
        stderr = subprocess.STDOUT
    else:
        stdout = open('/dev/null', 'w')
        stderr = subprocess.STDOUT

    X = '%s\n%s\n%2f\n%i' % (infile, output, FWHM, profile)
    print(infile)

    try:
        p = subprocess.Popen(
            [binpath + 'faltbo3'], 
            stdin=subprocess.PIPE, 
            stdout=stdout, 
            stderr=stderr, 
            shell=True
        )
        for line in re.split('\n', X):
            p.stdin.write(line + '\n')
        stdout, stderr = p.communicate()
    except subprocess.CalledProcessError:
        raise RuntimeError('Faltbo3 (convolution) Failed')

def Turbosynth(*args, **kwargs):
    starname = kwargs.get('starname', None)
    specpath = kwargs.get('specpath', '/home/zoeh/Code/synthetic_spec/specs/')
    vsini = kwargs.get('vsini', None)
    verbose = kwargs.get('verbose', True)
    sphere = kwargs.get('sphere', 'F')
    # specfile = kwargs.get('specfile', 'test1_1.spec')
    specfile = kwargs.get('specfile')
    R = kwargs.get('R', 60000)
    # R = kwargs.get('R',750)
    conv_profile = kwargs.get('conv_profile', 2)

    
    if kwargs.get('stellarpar', None) == None:
        modelname = kwargs.get('modelname', None)
        bab_in, bysn_in = make_babsmabysn_file(modelname=modelname, sphere=sphere, **kwargs)
    else:
        stellarpar = kwargs.get('stellarpar')
        if len(stellarpar) != 3:
            raise ValueError('The stellarpar variable must be a list/array of length 3 = (Teff, logg, [Fe/H])')
        else:
            print('help!')
            T, g, f = kwargs.get('stellarpar')
            print('Generating model atmosphere with T=%.1f, log g = %.2f, metallicity = %.2f' % (T, g, f))
            modelname, sphere = generate_model_atmo(T, g, f, modpath=modpath, starname=starname)
            bab_in, bysn_in = make_babsmabysn_file(modelname=modelname, sphere=sphere, **kwargs)

    if kwargs.get('verbose', True):
        stdout = None
        stderr = subprocess.STDOUT
    else:
        stdout = open('/dev/null', 'w')
        stderr = subprocess.STDOUT

    try:
        os.chdir('/home/zoeh/Code/BACCHUS/')
        pr1 = subprocess.Popen([turbo_path + 'babsma_lu'], stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
        with open('/home/zoeh/Code/synthetic_spec/babsma.par', 'r') as parfile:
            for line in parfile:
                pr1.stdin.write(line)
        stdout, stderr = pr1.communicate()
        print('here')
    except subprocess.CalledProcessError:
        raise RuntimeError('babsma failed ...')
    print('------')
    print(pr1.returncode, stderr)

    os.chdir('/home/zoeh/Code/BACCHUS/')

    if kwargs.get('verbose', True):
        stdout = None
        stderr = subprocess.STDOUT
    else:
        stdout = open('/dev/null', 'w')
        stderr = None

    try:
        os.chdir('/home/zoeh/Code/BACCHUS/')
        pr = subprocess.Popen([turbo_path + 'bsyn_lu'], stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
        with open('/home/zoeh/Code/synthetic_spec/bysn.par', 'r') as parfile:
            for line in parfile:
                pr.stdin.write(line)
        stdout, stderr = pr.communicate()
        print('here')
    except subprocess.CalledProcessError:
        raise RuntimeError('bsyn failed ....')
    os.chdir('/home/zoeh/Code/BACCHUS/')

    if vsini != None:
        w, f, fc = np.loadtxt(specpath + specfile, usecols=(0, 1, 2), unpack=True)
        nf = vsini_calc(w, f, vsini)
        nfc = vsini_calc(w, f, vsini)
        with open(specpath + specfile, 'w') as f:
            for i in range(len(w)):
                f.write("%.4f \t %.5f \t %.5f \n" % (w[i], nf[i], nfc[i]))

    print('about to run faltbo...')
    speed_of_light = 2.99792458E5
    FWHM = -1 * (speed_of_light / R)
    run_faltbo3(specpath+specfile, specpath+specfile+'.conv', FWHM, profile=conv_profile, binpath = faltbo3dir,verbose=verbose)
    if kwargs.get('verbose', True):
        print('------')
        print(pr.returncode, stderr)
    print('synthetic spectrum created!')
    print('R is', R)
    return pr.returncode
