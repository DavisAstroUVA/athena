#! /usr/bin/env python

"""
Read in athena results to judge convergence towards blackbody emission
as photon number increases
"""

# python standard modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os import system

# Athena++ modules
import athena_mc as mcspec

def write_athinput(iseed,nphot,nen,emin,emax,abs_meth,file='athinput.mciso'):
    """
    Write the remainder of the athinput file for convergence test
    """

    outfile = open(file,'w')
    outfile.write("<comment>\n")
    outfile.write("problem   =  Simple isothermal atmosphere\n")
    outfile.write("reference =\n")
    outfile.write("configure = --prob=mc_isoth -mc\n")
    outfile.write("\n")
    outfile.write("<job>\n")
    outfile.write("problem_id = mciso  # problem ID: basename of output filenames\n")
    outfile.write("\n")
    outfile.write("<output1>\n")
    outfile.write("file_type  = spec\n")
    outfile.write("face      = outer_x3\n")
    outfile.write("ne        = {:d}\n".format(nen))
    outfile.write("emin      = {:e}\n".format(emin))
    outfile.write("emax      = {:e}\n".format(emax))
    outfile.write("ncth      = 8\n")
    outfile.write("nphi      = 8\n")
    outfile.write("\n")
    outfile.write("<mesh>\n")
    outfile.write("nx1        = 64       # Number of zones in X1-direction\n")
    outfile.write("x1min      = -5.0e10       # minimum value of X1\n")
    outfile.write("x1max      = 5.0e10      # maximum value of X1\n")
    outfile.write("ix1_mc_bc     = periodic  # inner-X1 boundary flag\n")
    outfile.write("ox1_mc_bc     = periodic  # outer-X1 boundary flag\n")
    outfile.write("\n")
    outfile.write("nx2        = 64         # Number of zones in X2-direction\n")
    outfile.write("x2min      = -5.0e10      # minimum value of X2\n")
    outfile.write("x2max      = 5.0e10       # maximum value of X2\n")
    outfile.write("ix2_mc_bc  = periodic  # inner-X2 boundary flag\n")
    outfile.write("ox2_mc_bc  = periodic  # outer-X2 boundary flag\n")
    outfile.write("\n")
    outfile.write("nx3        = 128         # Number of zones in X3-direction\n")
    outfile.write("x3min      = 0.0        # minimum value of X3\n")
    outfile.write("x3max      = 1.0e11       # maximum value of X3\n")
    outfile.write("ix3_mc_bc  = absorb  # inner-X3 boundary flag\n")
    outfile.write("ox3_mc_bc  = escape  # outer-X3 boundary flag\n")
    outfile.write("\n")
    outfile.write("<meshblock>\n")
    outfile.write("nx1 = 32\n")
    outfile.write("nx2 = 32\n")
    outfile.write("nx3 = 64\n")
    outfile.write("\n")
    outfile.write("<hydro>\n")
    outfile.write("gamma = 1.666666666666667 # gamma = C_p/C_v\n")
    outfile.write("\n")
    outfile.write("<montecarlo>\n")
    outfile.write("nphot     = {:d}\n".format(nphot))
    outfile.write("iseed     = {:d}\n".format(iseed))
    outfile.write("scattering = none\n")
    outfile.write("emission   = freefree\n")
    outfile.write("absorption = freefree\n")
    outfile.write("polarized = none\n")
    outfile.write("abs_method = "+abs_meth+"\n")
    outfile.write("\n<problem>\n")
    outfile.write("temp     = 1.0e5\n")
    outfile.write("taumin   = 1.0e-3\n")
    outfile.write("taumax   = 1.0e4\n")
    outfile.write("emin      = {:e}\n".format(emin))
    outfile.write("emax      = {:e}\n".format(emax))
    outfile.close()

def get_blackbody(temp,nu):
    c = 2.99792458e10
    kb = 1.380649e-16
    h = 6.62607015e-27
    return 2*h/c**2*nu**3/(np.exp(h*nu/(kb*temp)) - 1.0)

# Main function
def main(**kwargs):

    path =  kwargs.pop("path")
    athena_path = path+"/bin"
    area = 1.e22

    mcranks = kwargs['mcranks']
    nstep = kwargs['nstep']
    nphots = [kwargs['nmin']]
    step = kwargs['step']
    iseed = []
    for i in range(nstep-1):
        nphots.append(nphots[i]*step)
    iseed = kwargs['iseed']
    # Set up array to store norm for convergence evaluation
    lnorm = np.zeros((nstep,2))

    for i,nphot in enumerate(nphots):
        write_athinput(iseed+99*i,nphot,kwargs['nen'],kwargs['emin'],kwargs['emax'],kwargs['abs_meth'])
        if (mcranks == 1):
            com=athena_path+"/athena -i athinput.mciso"
        else:
            com="mpirun -np {:d} ".format(mcranks)+athena_path+"/athena -i athinput.mciso"
        system(com)
        # read spectrum as dict from infile
        spectrum = mcspec.read_spectrum("mciso.out1.00000.spec")
        # compute bin center frequencies on which spectrum is tabulated
        xfaces = spectrum['xfaces']
        h = 6.62607015e-27
        everg = 1.6021772e-12
        nu = 0.5*(xfaces[1:]+xfaces[:-1])*everg/h
        # compute blackbody comparison function at bin centers
        ibb = get_blackbody(1.e5,nu)*area
        # average spectrum over phi
        intensity = spectrum['intensity']
        intensity = np.sum(intensity[0,:,:,:],axis=0)/float(spectrum['nphi'])
        norm = 0.
        n = 0
        for j in range(spectrum['nmu']):
            for k in range(spectrum['nx']):
                norm += abs(intensity[j,k]-ibb[k])/ibb[k]
                n += 1
        lnorm[i,0] = float(nphot)
        lnorm[i,1] = norm/float(n)

    # save plot to outfile
    np.savetxt(kwargs['outfile'],lnorm)

    plt.plot(lnorm[:,0],lnorm[:,1],'+')
    plt.plot(lnorm[:,0],lnorm[-1,1]*(lnorm[-1,0]/lnorm[:,0])**0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("conv.pdf")

    print(lnorm)


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
    parser.add_argument('--nen',
        type = int,
        default = 100,
        help='number of photon energy bins')
    parser.add_argument('--emin',
        type = float,
        default = 1.,
        help='minimum photon energy')
    parser.add_argument('--emax',
        type = float,
        default = 100.,
        help='maximum photon energy')
    parser.add_argument('--path',
        default = "/home/swd8g/athena-swdavis",
        help='path to Athena++ distribution')   
    parser.add_argument('--abs_meth',
        default = "weight",
        help='method for absorption')
    parser.add_argument('--outfile',
        default="conv.out",
        help='output filename for storing convergence rate')

    args = parser.parse_args()
    main(**vars(args))
