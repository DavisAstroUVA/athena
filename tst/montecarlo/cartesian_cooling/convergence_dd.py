#! /usr/bin/env python

"""
Read in athena results to asses whether various radiation moments
and cooling are being computed correctly
"""

# python standard modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpmath import polylog
from os import system

# Athena++ modules
import athena_read

def bnu_int(en,tgas):
    kb = 1.380649e-16
    kt = kb * tgas
    x = np.exp(-en/(kt))
    if (en == 0.):
        # avoids evaluation of 0 * -inf in log
        return float(-3.*kt*en**2*polylog(2,x)-6.*en*kt**2*polylog(3,x)
                     -6*kt**3*polylog(4,x))
    else:
        return float(en**3*np.log(1.-x)-3.*kt*en**2*polylog(2,x)
                     -6.*en*kt**2*polylog(3,x)-6*kt**3*polylog(4,x))

def eta_kappap_ff(rho,tgas,emin,emax):
    """
    Computes the frequency integrated emmisivity for a gas of completely ionized
    H and He over an energy range extending from emin to emax, where theose energies
    are measured in eV
    """
    kb = 1.380649e-16
    mp = 1.67e-24
    me = 9.109e-28
    ec = 4.8032e-10
    c = 2.99792458e10
    h = 6.62607015e-27
    everg = 1.602176634e-12

    # efac accoutns for integration limits that are not 0, infinity
    efac = np.exp(-emin*everg/(kb*tgas))-np.exp(-emax*everg/(kb*tgas))
    ef0 = 32.*np.pi*ec**6/(3.*me*h*c**3)*(2.*np.pi*kb/(3.*me))**0.5*efac

    heabund = 0.09
    nh = rho/mp/(1.+4.*heabund)
    nhe = nh*heabund
    nel = nh+2.*nhe

    eta = ef0*nel*(nh+4.*nhe)*tgas**0.5

    # Integrates plank function yields sigma*T^4/pi in limit that
    # emin -> 0 and emax -> infinity
    planck_int = np.zeros_like(tgas)
    for index, temp in np.ndenumerate(tgas):
      planck_int[index] = 2.*kb*temp/h**3/c**2*(bnu_int(emax*everg,temp)-
                                         bnu_int(emin*everg,temp))
    #planck_int = 2.*kb*tgas/h**3/c**2*(bnu_int(emax*everg,tgas)-
    #                                   bnu_int(emin*everg,tgas))
    kap = eta/(planck_int*4.*np.pi*rho)
    return eta, kap

def cooling(tgas,rho,Ermc,hnu,kapj,emin,emax,scatflag=True):
    """
    Computes the cooling assuming free-free emission and absorption, and
    Compton scattering
    """
    kb = 1.3806580e-16
    c = 2.9979e10
    me = 9.1094e-28
    sigmat = 6.65248e-25
    mp = 1.6726e-24
    heabund = 0.09
    if (scatflag):
        kappa_es = sigmat * (1.+2.*heabund) / (mp*(1.+4.*heabund))
    else:
        kappa_es = 0.

    eta, kappap = eta_kappap_ff(rho,tgas,emin,emax) #includes factor of 4pi

    cool = (-eta + c*kapj*rho*Ermc - (c*rho*kappa_es*4.*kb*tgas*Ermc / (me*c**2))
            + (c*rho*kappa_es*hnu*Ermc / (me*c**2)))
    coolderiv = -0.5*eta/tgas - (c*rho*kappa_es*4.*kb*Ermc / (me*c**2))

    return cool, coolderiv

def write_athinput(iseed,nphot,vel,frame,dens,tgas,emin,emax,length,periodic,
                   scattering,file='athinput.mciso'):
    """
    Write the remainder of the athinput file for convergence test
    """
    if vel is None:
        vel = 0.
        boosts = False
    else:
        boosts = True
    outfile = open(file,'w')
    outfile.write("<comment>\n")
    outfile.write("problem   =  Uniform periodic box\n")
    outfile.write("reference =\n")
    outfile.write("configure = --prob=mc_isoth -mc\n")
    outfile.write("\n")
    outfile.write("<job>\n")
    outfile.write("problem_id = mciso   # problem ID: basename of output filenames\n")
    outfile.write("\n")
    outfile.write("<output1>\n")
    outfile.write("file_type  = hdf5\n")
    outfile.write("variable   = mclab\n")
    outfile.write("id         = lab\n")
    outfile.write("\n")
    outfile.write("\n")
    outfile.write("<output2>\n")
    outfile.write("file_type  = hdf5\n")
    outfile.write("variable   = uom\n")
    outfile.write("id         = uom\n")
    outfile.write("\n")
    outfile.write("<output3>\n")
    outfile.write("file_type  = hdf5\n")
    outfile.write("variable   = mcsrc\n")
    outfile.write("id         = src\n")
    outfile.write("\n")
    outfile.write("<mesh>\n")
    outfile.write("nx1        = 8       # Number of zones in X1-direction\n")
    outfile.write("x1min      = {:e}    # minimum value of X1\n".format(-length/2.))
    outfile.write("x1max      = {:e}    # maximum value of X1\n".format(length/2.))
    if (periodic):
        outfile.write("ix1_mc_bc     = periodic  # inner-X1 boundary flag\n")
        outfile.write("ox1_mc_bc     = periodic  # outer-X1 boundary flag\n")
    else:
        outfile.write("ix1_mc_bc     = reflect  # inner-X1 boundary flag\n")
        outfile.write("ox1_mc_bc     = reflect  # outer-X1 boundary flag\n")
    outfile.write("\n")
    outfile.write("nx2        = 8         # Number of zones in X2-direction\n")
    outfile.write("x2min      = {:e}      # minimum value of X2\n".format(-length/2.))
    outfile.write("x2max      = {:e}      # maximum value of X2\n".format(length/2.))
    if (periodic):
        outfile.write("ix2_mc_bc  = periodic  # inner-X2 boundary flag\n")
        outfile.write("ox2_mc_bc  = periodic  # outer-X2 boundary flag\n")
    else:
        outfile.write("ix2_mc_bc  = reflect  # inner-X2 boundary flag\n")
        outfile.write("ox2_mc_bc  = reflect  # outer-X2 boundary flag\n")
    outfile.write("\n")
    outfile.write("nx3        = 8         # Number of zones in X3-direction\n")
    outfile.write("x3min      = {:e}      # minimum value of X3\n".format(-length/2.))
    outfile.write("x3max      = {:e}       # maximum value of X3\n".format(length/2.))
    if (periodic):
        outfile.write("ix3_mc_bc  = periodic  # inner-X3 boundary flag\n")
        outfile.write("ox3_mc_bc  = periodic  # outer-X3 boundary flag\n")
    else:
        outfile.write("ix3_mc_bc  = reflect  # inner-X3 boundary flag\n")
        outfile.write("ox3_mc_bc  = reflect  # outer-X3 boundary flag\n")
    outfile.write("\n")
    outfile.write("<meshblock>\n")
    outfile.write("nx1 = 4\n")
    outfile.write("nx2 = 4\n")
    outfile.write("nx3 = 4\n")
    outfile.write("\n")
    outfile.write("<hydro>\n")
    outfile.write("gamma = 1.666666666666667 # gamma = C_p/C_v\n")
    outfile.write("\n")
    outfile.write("<montecarlo>\n")
    outfile.write("nphot     = {:d}\n".format(nphot))
    outfile.write("iseed     = {:d}\n".format(iseed))
    if scattering:
        outfile.write("scattering = compton\n")
    else:
        outfile.write("scattering = none\n")
    outfile.write("emission   = freefree\n")
    outfile.write("absorption = freefree\n")
    outfile.write("polarized = none\n")
    outfile.write("comptonio = 0\n")
    if boosts:
        outfile.write("boosts     = true\n")
    else:
        outfile.write("boosts     = false\n")
    #outfile.write("abs_weight = false\n")
    outfile.write("\n")
    outfile.write("<problem>\n")
    outfile.write("temp     = {:e}\n".format(tgas))
    outfile.write("dens     = {:e}\n".format(dens))
    outfile.write("velocity = {:e}\n".format(vel))
    outfile.write("constdens = true\n")
    outfile.write("emin      = {:e}\n".format(emin))
    outfile.write("emax      = {:e}\n".format(emax))
    outfile.close()

# Main function
def main(**kwargs):

    path =  kwargs.pop("path")
    athena_path = path+"/bin"
    infile = "athinput.mciso"

    mcranks = kwargs['mcranks']
    nstep = kwargs['nstep']
    nphots = [kwargs['nmin']]
    step = kwargs['step']

    # input energy ranges and physical parameters
    tgas = kwargs.pop("tgas")
    dens = kwargs.pop("dens")
    length = kwargs.pop("length")

    # set energy limits based on temperature
    kb = 1.380649e-16
    emin = 0.001*kb*tgas
    emax = 50.*kb*tgas

    c = 2.99792458e10
    h = 6.62607015e-27
    planck_int = 2.*kb*tgas/h**3/c**2*(bnu_int(emax,tgas)-
                                      bnu_int(emin,tgas))
    # convert to ev for athinput file
    everg = 1.602176634e-12
    emin /= everg
    emax /= everg

    er0 = float(4.*np.pi*planck_int/c)
    fr0 = 0.

    if ((kwargs['vel'] is not None) and (kwargs['frame'] == 'eulerian')):
        beta = kwargs['vel']
        gamma=1./(1.-beta**2)**0.5
        er = er0*gamma**2*(1+beta**2/3.)
        fr = er0*beta*c*4./3.*gamma**2
    else:
        er = er0
        fr = fr0


    scatflag = not kwargs["noscat"]
    periodic = not kwargs["reflect"]
    iseed = []
    for i in range(nstep-1):
        nphots.append(nphots[i]*step)
    iseed = kwargs['iseed']

    # Evaluate emissivity and Planck mean opacity
    eta, kappap = eta_kappap_ff(dens,tgas,emin,emax)

    # Set up array to store norm for convergence evaluation
    output = np.zeros((nstep,6))
    for i,nphot in enumerate(nphots):
        write_athinput(iseed+99*i,nphot,kwargs['vel'],kwargs['frame'],dens,tgas,
                       emin,emax,length,
                       periodic,scatflag,file=infile)
        if (mcranks == 1):
            com=athena_path+"/athena -i athinput.mciso"
        else:
            com="mpirun -np {:d} ".format(mcranks)+athena_path+"/athena -i "+infile
        print(com)
        system(com)
        # read hdf5 output
        data_lab = athena_read.athdf("mciso.lab.00000.athdf",quantities=['Ermc','Frmc1',
                                     'Frmc2','Frmc3','tgas','rho'])
        data_src = athena_read.athdf("mciso.src.00000.athdf",quantities=['Cooling'])
        data_usr = athena_read.athdf("mciso.uom.00000.athdf",quantities=['kapJ','eave'])
        output[i,0] = float(nphot)
        ermc = np.average(data_lab['Ermc'])
        eave = data_usr['eave']/ermc
        eave_ave = np.average(eave)
        kapj = data_usr['kapJ']/ermc/dens
        kapj_ave = np.average(kapj)
        cool = np.average(data_src['Cooling'])
        # zone-by-zone mean absolute cooling, so that column 3 is directly
        # comparable to the analytic estimate in column 4.  Averaging the signed
        # Cooling array first (as `cool` does, for the summary print below) cancels
        # to ~zero in a uniform box and is not a convergence measure.
        cool_abs = np.average(abs(data_src['Cooling']))
        cdot, cdot_tgas = cooling(data_lab['tgas'],data_lab['rho'],data_lab['Ermc'],
                                  eave,kapj,emin,emax,scatflag=scatflag)
        cdot_ave = np.average(cdot)
        dt_ave = np.average(abs(-cdot/cdot_tgas/tgas))

        output[i,1] = np.average(abs(ermc-er))/er
        output[i,2] = np.average(abs(kapj-kappap))/kappap
        output[i,3] = cool_abs/eta
        output[i,4] = np.average(abs(cdot))/eta
        output[i,5] = np.average(abs(eave-(3.83223*kb*tgas)))/(3.83223*kb*tgas)

    # save plot to outfile
    np.savetxt(kwargs['outfile'],output)

    print(output)
    print("kapp, kapj, ratio:",kappap, kapj_ave, kapj_ave/kappap)
    print("er, ermc, ratio:",er,ermc,ermc/er)
    print("<hnu>, <hnu>/4kT, <hnu>/3.83223kT:",eave_ave,eave_ave/(4*kb*tgas),
          eave_ave/(3.83223*kb*tgas))
    print("mccool, cdot, mccool/eta, cdot/eta:",cool, cdot_ave, cool/eta, cdot_ave/eta)
    print("Temperature correction:",dt_ave)

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('iseed',
        type = int,
        default = None,
        help='random seed')
    parser.add_argument('nmin',
        type = int,
        default = None,
        help='minimum photon number')
    parser.add_argument('nstep',
        type = int,
        default = None,
        help='number of steps')
    parser.add_argument('step',
        type = int,
        default = 2,
        help='factor by which to increase nphot')
    parser.add_argument('--mcranks',
        type = int,
        default = 8,
        help='mpi ranks to use')
    parser.add_argument('--vel',
        type = float,
        default = None,
        help='boost velocity')
    parser.add_argument('--dens',
        type = float,
        default = 3.e-4,
        help='density')
    parser.add_argument('--tgas',
        type = float,
        default = 3.e6,
        help='gas temperature')
    parser.add_argument('--length',
        type = float,
        default = 1.e11,
        help='length of box')
    parser.add_argument('--frame',
        default = "eulerian",
        help='boost velocity')
    parser.add_argument('--noscat',
        action = 'store_true',
        help='do not include scattering')
    parser.add_argument('--reflect',
        action = 'store_true',
        help='use reflecting boundaries')
    parser.add_argument('--path',
        default = "/home/swd8g/athena-swdavis",
        help='path to Athena++ distribution')
    parser.add_argument('--outfile',
        default="conv.out",
        help='output filename for storing convergence rate')

    args = parser.parse_args()
    main(**vars(args))
