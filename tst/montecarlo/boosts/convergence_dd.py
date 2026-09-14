#! /usr/bin/env python

"""
Read in athena results to judge convergence towards blackbody emission
as photon number increases
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
        return (-3.*kt*en**2*polylog(2,x)-6.*en*kt**2*polylog(3,x)
                -6*kt**3*polylog(4,x))
    else:
        return (en**3*np.log(1.-x)-3.*kt*en**2*polylog(2,x)
                -6.*en*kt**2*polylog(3,x)-6*kt**3*polylog(4,x))

def write_athinput(iseed,nphot,vel,dens,tgas,emin,emax,absmeth='weight',
                   scattering=False,file='athinput.mciso'):
    """
    Write the remainder of the athinput file for convergence test
    """
    if (vel is None):
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
    outfile.write("problem_id = mciso # problem ID: basename of output filenames\n")
    outfile.write("\n")
    outfile.write("<output1>\n")
    outfile.write("file_type  = hdf5\n")
    outfile.write("variable   = mclab\n")
    outfile.write("id         = lab\n")
    outfile.write("\n")
    if boosts:
        outfile.write("<output2>\n")
        outfile.write("file_type  = hdf5\n")
        outfile.write("variable   = mccom\n")
        outfile.write("id         = com\n")
        outfile.write("\n")
    outfile.write("<output3>\n")
    outfile.write("file_type  = hdf5\n")
    outfile.write("variable   = uom\n")
    outfile.write("id         = uom\n")
    outfile.write("\n")
    outfile.write("<mesh>\n")
    outfile.write("nx1        = 8       # Number of zones in X1-direction\n")
    outfile.write("x1min      = -5.0e10       # minimum value of X1\n")
    outfile.write("x1max      = 5.0e10      # maximum value of X1\n")
    outfile.write("ix1_mc_bc     = periodic  # inner-X1 boundary flag\n")
    outfile.write("ox1_mc_bc     = periodic  # outer-X1 boundary flag\n")
    outfile.write("\n")
    outfile.write("nx2        = 8         # Number of zones in X2-direction\n")
    outfile.write("x2min      = -5.0e10      # minimum value of X2\n")
    outfile.write("x2max      = 5.0e10       # maximum value of X2\n")
    outfile.write("ix2_mc_bc  = periodic  # inner-X2 boundary flag\n")
    outfile.write("ox2_mc_bc  = periodic  # outer-X2 boundary flag\n")
    outfile.write("\n")
    outfile.write("nx3        = 8         # Number of zones in X3-direction\n")
    outfile.write("x3min      = -5.0e10      # minimum value of X2\n")
    outfile.write("x3max      = 5.0e10       # maximum value of X2\n")
    outfile.write("ix3_mc_bc  = periodic  # inner-X3 boundary flag\n")
    outfile.write("ox3_mc_bc  = periodic  # outer-X3 boundary flag\n")
    outfile.write("\n")
    outfile.write("<hydro>\n")
    outfile.write("gamma = 1.666666666666667 # gamma = C_p/C_v\n")
    outfile.write("\n")
    outfile.write("<meshblock>\n")
    outfile.write("nx1 = 4\n")
    outfile.write("nx2 = 4\n")
    outfile.write("nx3 = 4\n")
    outfile.write("\n")
    outfile.write("<montecarlo>\n")
    outfile.write("nphot     = {:d}\n".format(nphot))
    outfile.write("iseed     = {:d}\n".format(iseed))
    if scattering:
        outfile.write("scattering = thomson\n")
    else:
        outfile.write("scattering = none\n")
    outfile.write("emission   = freefree\n")
    outfile.write("absorption = freefree\n")
    outfile.write("polarized = none\n")
    outfile.write("abs_method = "+absmeth+"\n")
    if boosts:
        outfile.write("boosts     = true\n")
    else:
        outfile.write("boosts     = false\n")
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

    # set limits
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

    if kwargs['vel'] is not None:
        beta = kwargs['vel']
        gamma=1./(1.-beta**2)**0.5
        er = er0*gamma**2*(1+beta**2/3.)
        fr = er0*beta*c*4./3.*gamma**2
        boosts = True
    else:
        er = er0
        fr = fr0
        boosts = False

    iseed = []
    for i in range(nstep-1):
        nphots.append(nphots[i]*step)
    iseed = kwargs['iseed']

    # Set up array to store norm for convergence evaluation
    if boosts:
        res = np.zeros((nstep,20))
    else:
        res = np.zeros((nstep,12))
    for i,nphot in enumerate(nphots):
        write_athinput(iseed+99*i,nphot,kwargs['vel'],dens,tgas,emin,emax,
                       scattering=kwargs['scat'],absmeth=kwargs['absmeth'],file=infile)
        if (mcranks == 1):
            com=athena_path+"/athena -i athinput.mciso"
        else:
            com="mpirun -np {:d} ".format(mcranks)+athena_path+"/athena -i "+infile
        print(com)
        system(com)
        # read hdf5 outputs
        data = athena_read.athdf("mciso.lab.00000.athdf",quantities=['Ermc','Frmc1',
                                                                      'Frmc2','Frmc3'])

        res[i,0] = float(nphot)
        res[i,1] = er0
        res[i,2] = er
        res[i,3] = fr
        res[i,4] = np.average(data['Ermc'])
        res[i,5] = np.average(data['Frmc1'])
        res[i,6] = np.average(data['Frmc2'])
        res[i,7] = np.average(data['Frmc3'])
        res[i,8] = np.average(abs(data['Ermc']-er))/er
        res[i,9] = np.average(abs(data['Frmc1']))/(er*c)
        res[i,10] = np.average(abs(data['Frmc2']))/(er*c)
        if (boosts):
            res[i,11] = np.average(abs(data['Frmc3']-fr))/fr
            data = athena_read.athdf("mciso.com.00000.athdf",quantities=['Ermc0','Frmc01',
                                     'Frmc02','Frmc03'])
            res[i,12] = np.average(data['Ermc0'])
            res[i,13] = np.average(data['Frmc01'])
            res[i,14] = np.average(data['Frmc02'])
            res[i,15] = np.average(data['Frmc03'])
            res[i,16] = np.average(abs(data['Ermc0']-er0))/er0
            res[i,17] = np.average(abs(data['Frmc01']))/(er0*c)
            res[i,18] = np.average(abs(data['Frmc02']))/(er0*c)
            res[i,19] = np.average(abs(data['Frmc03']))/(er0*c)
        else:
            res[i,11] = np.average(abs(data['Frmc3']))/(er*c)

    # save subset of results to outfile
    if boosts:
        output = np.zeros((nstep,9))
        output[:,0] = res[:,0]
        output[:,1:5] = res[:,8:12]
        output[:,5:9] = res[:,16:20]
    else:
        output = np.zeros((nstep,5))
        output[:,0] = res[:,0]
        output[:,1:5] = res[:,8:12]
    np.savetxt(kwargs['outfile'],output)

    # save plot to file
    plt.plot(output[:,0],abs(output[:,4]-output[:,1]),'r+')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("conv.pdf")

    #print results
    for i in range(nstep):
        print("Results for {:d} samples: ".format(int(res[i,0])))
        print(" Expected: Er0 {:e}, Er {:e}, Fr {:e}".format(res[i,1],res[i,2],res[i,3]))
        print(" Averages: Er: {:e}, Fr1 {:e}, Fr2 {:e}, Fr3 {:e}".format(res[i,4],
              res[i,5],res[i,6],res[i,7]))
        print(" Convergence: Er: {:e}, Fr1 {:e}, Fr2 {:e}, Fr3 {:e}".format(res[i,8],
              res[i,9], res[i,10], res[i,11]))
        if boosts:
            print(" Comoving moments:")
            print(" Averages: Er: {:e}, Fr1 {:e}, Fr2 {:e}, Fr3 {:e}".format(res[i,12],
                  res[i,13],res[i,14],res[i,15]))
            print(" Convergence: Er: {:e}, Fr1 {:e}, Fr2 {:e}, Fr3 {:e}".format(
                  res[i,16],res[i,17], res[i,18], res[i,19]))
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
        default = 3.e-8,
        help='density')
    parser.add_argument('--tgas',
        type = float,
        default = 1.e5,
        help='gas temperature')
    parser.add_argument('--scat',
        action = 'store_true',
        help='use thomson scattering')
    parser.add_argument('--absmeth',
        default = "weight",
        help='absorption method: weight, destroy, tau')
    parser.add_argument('--path',
        default = "/home/swd8g/athena-swdavis",
        help='path to Athena++ distribution')
    parser.add_argument('--outfile',
        default="conv.out",
        help='output filename for storing convergence rate')

    args = parser.parse_args()
    main(**vars(args))
