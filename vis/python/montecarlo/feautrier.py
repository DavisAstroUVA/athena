"""
Feautrier solution for a plane-parallel scattering atmosphere, used as the reference for
the polarized Monte Carlo tests.

Polarization is reported as Q/I in the same convention the Monte Carlo code stores, so the
two can be compared directly with no sign change: Q > 0 for polarization along the meridian
direction l, which makes Q < 0 for a scattering atmosphere.

Lives here, beside athena_mc, rather than in any one test directory: three separate tests
depend on it, and all of them already need this directory on PYTHONPATH for athena_mc.
"""

# standard python modules
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def transfer(tconst=1.e5,trange=[1e-3,1.0e4],l0=1.e11,nd=64,na=8,nf=96,
             outfile="feautrier.out",dconst=None,numin=None,numax=None,
             nu=None):
    """
    Set up 1D atmosphere and then perform Feautrier solution for the
    radiation transfer
    """
            
    # physical constants
    h = 6.62607015e-27
    kb = 1.380649e-16
    sigmab = 5.6703e-5

    grid = {}
    rad = {}
    #rad = container()

    # Set up depth grid
    kapes = 0.33
    
    grid['temp'] = np.empty(nd)
    grid['temp'].fill(tconst)

    grid['nd'] = nd
    grid['md'] = np.empty(nd+1)

    if dconst is None:
        # compute density, md from tau grid chosen to to match
        # mctest.cpp
        step = np.log10(trange[1]/trange[0])/float(nd-1)
        tau = 10**(np.log10(trange[0])+step*np.arange(nd))
        grid['md'][0] = 0.0
        grid['md'][1:] = tau/kapes
        
        dz = l0 / nd
        grid['dens'] = np.empty(nd)
        for i in range(nd):
            grid['dens'][i] = (grid['md'][i+1]-grid['md'][i])/dz
    else:
        grid['dens'] = np.empty(nd)
        grid['dens'].fill(dconst)
        dz = l0 / nd
        grid['md'][0] = 0.0
        for i in range(1,nd+1):
            grid['md'][i] = grid['md'][i-1] + dz * grid['dens'][i-1]

    #grid['flux'] = sigmab * tconst**4
    grid['flux'] = 0.0

    # Set up frequency grid using featupol method:
    if nu is not None:
        nf = len(nu)
        rad['nf'] = nf
        rad['nu'] = nu
    else:
        rad['nf'] = nf
        rad['nu'] = np.empty(nf)
        if (numin is None): numin = 1.e-3*kb*grid['temp'][0]/h
        if (numax is None): numax = 1.e2*kb*grid['temp'][nd-1]/h
        rad['nu'] = numin*10**((np.arange(nf))/float(nf-1)*np.log10(numax/numin))

    rad['wnu'] = np.empty(nf)  
    rad['wnu'][0] = 0.5*(rad['nu'][2] - rad['nu'][1])
    rad['wnu'][nf-1] = 0.5*(rad['nu'][nf-1] - rad['nu'][nf-2])
    for i in range(1,nf-1):
        rad['wnu'][i] = 0.5*(rad['nu'][i+1] - rad['nu'][i-1])

    # Set up angular grid
    rad['na'] = na
    rad['mu'] = np.empty(na)
    rad['wmu'] = np.empty(na)

    # Use Gauss-Legendre weights and ordinates
    rad['mu'],rad['wmu'] = gauleg(0.0,1.0,na)

    # Set external radiation to zero
    rad['intex'] = np.zeros([4*na,nf])

    # Run feautrier calculation
    feautrier(grid,rad)

    # Write results to file
    write_intensity(outfile,rad)


def write_intensity(outfname,rad):
    """
    Write intensity to output file
    """
    outfile = open(outfname,'w')
    outfile.write("{:d} {:d}\n".format(rad['nf'],rad['na']))
    for i in range(rad['nf']):
        outfile.write("{:e} ".format(rad['nu'][i]))
    outfile.write("\n")
    for i in range(rad['nf']):
        outfile.write("{:e} ".format(rad['wnu'][i]))
    outfile.write("\n")
    for i in range(rad['na']):
        outfile.write("{:e} ".format(rad['mu'][i]))
    outfile.write("\n")
    for i in range(rad['na']):
        outfile.write("{:e} ".format(rad['wmu'][i]))
    outfile.write("\n")
    for i in range(rad['nf']):
        for j in range (rad['na']):
            outfile.write("{:e} {:e} ".format(rad['intens'][i,j],rad['pol'][i,j]))
        outfile.write("\n")
    outfile.close()


def feautrier(grid,rad,heabund=0.09):
    """
    Compute the Feautrier solution of temperature and density structure
    contained in grid using the frequency and angular arrays contained
    in rad
    """

    # physical constants
    h = 6.62607015e-27
    kb = 1.380649e-16
    c = 2.99792458e10
    mp = 1.6726e-24
    sigmat = 6.65248e-25

    nf = rad['nf']
    na = rad['na']
    nd = grid['nd']

    # Set up opacity and emissivity arrays
    eta = np.empty([nf,nd])
    kappa = np.empty([nf,nd])
    chi = np.empty([nf,nd])
    gaunt = 1.0
    nh = grid['dens']/mp/(1.+4.*heabund)
    ne = (1.+2*heabund) * nh
    for i in range(nf):
        for j in range(nd):
            emhnu = np.exp(-h*rad['nu'][i]/kb/grid['temp'][j])
            aff = 3.69e8/rad['nu'][i]**3/np.sqrt(grid['temp'][j])*gaunt
            kappa[i,j] = ne[j]*nh[j]*aff*(1.-emhnu)
            eta[i,j] = 2.*h*rad['nu'][i]**3/c**2*emhnu*ne[j]*nh[j]*aff
            chi[i,j] = kappa[i,j] + ne[j] * sigmat
    # Set up optical depth grid
    dtau = np.zeros([nf,nd+1])
    for i in range(nf):
        for j in range(1,nd):
            dtau[i,j] = 0.5*(chi[i,j-1]*(grid['md'][j]-grid['md'][j-1])
                             /grid['dens'][j-1]+chi[i,j]*(grid['md'][j+1]
                             -grid['md'][j])/grid['dens'][j])

    # Set up scattering matrix
    scatm = np.empty([2*na,2*na])
    for i in range(na):
        for j in range(na):
            scatm[i,j] = 0.75*rad['wmu'][j]*(2.*(1.-rad['mu'][i]**2)*
                         (1.-rad['mu'][j]**2)+rad['mu'][i]**2.*rad['mu'][j]**2)
            scatm[i,j+na] = 0.75*rad['wmu'][j]*rad['mu'][i]**2
            scatm[i+na,j] = 0.75*rad['wmu'][j]*rad['mu'][j]**2
            scatm[i+na,j+na] = 0.75*rad['wmu'][j]

    # Set up double mu matrix
    mu2 = np.empty(2*na)
    mu2[0:na] = rad['mu']
    mu2[na:2*na] = rad['mu']

    #=========== Feautrier solution starts here =========================

    Mat = {}
    Mat['A'] = np.empty([nd,2*na,2*na])
    Mat['B'] = np.empty([nd,2*na,2*na])
    Mat['C'] = np.empty([nd,2*na,2*na])
    Mat['L'] = np.empty([nd,2*na])
    Jrad = np.empty([nd,2*na])

    # Set up matrixes for Gauss-Joran elimination
    delt = chi[:,0]*grid['md'][1]/grid['dens'][0]


    # Needed for boundary condtions
    ehnu = np.exp(h*rad['nu']/kb/grid['temp'][nd-1])
    dbdt = 2*h**2*rad['nu']**4/(c**2*kb*grid['temp'][nd-1]**2)*ehnu/(ehnu-1.)**2
    dbint = np.sum(rad['wnu']*dbdt/chi[:,nd-1])

    # Create array to store output intensities and polarization
    rad['intens'] = np.empty([nf,na])
    rad['pol'] = np.empty([nf,na])
    for i in range(nf):
    
        # upper boundary condition
        Mat['A'][0,:,:].fill(0.)
        Mat['B'][0,:,:] = -ne[0]*sigmat/chi[i,0]*scatm+np.diag(1.+mu2**2/ \
                          delt[i]*(1./dtau[i,1]+1./(0.5*delt[i]+mu2)))
        Mat['C'][0,:,:] = np.diag(mu2**2/delt[i]/dtau[i,1])
        Mat['L'][0,:] = (0.5*eta[i,0]/chi[i,0]+mu2*rad['intex'][0:2*na,i]/ \
                         delt[i]/(0.5*delt[i]+mu2))
        
        # interior depth points
        for l in range(1,nd-1):
            Mat['A'][l,:,:] = np.diag(2.*mu2**2/(dtau[i,l+1]+dtau[i,l])/ \
                                      dtau[i,l])
            Mat['B'][l,:,:] = -ne[l]*sigmat/chi[i,l]*scatm+np.diag(1.+2.* \
                              mu2**2/(dtau[i,l+1]+dtau[i,l])*(1./dtau[i,l] \
                              +1./dtau[i,l+1]))
            Mat['C'][l,:,:] = np.diag(2.*mu2**2/(dtau[i,l+1]+dtau[i,l])/ \
                              dtau[i,l+1])
            Mat['L'][l,:].fill(0.5*eta[i,l]/chi[i,l])
     

        #lower boundary condtion (assumes unpolarized blackbody)
        #NOT CORRECT FOR DISKS

        Mat['A'][nd-1,:,:] = np.identity(2*na)/dtau[i,nd-1]
        Mat['B'][nd-1,:,:] = np.identity(2*na)/dtau[i,nd-1]
        Mat['C'][nd-1:,:].fill(0.)
        Mat['L'][nd-1,:].fill(0.375*grid['flux']*dbdt[i]/np.pi/chi[i,nd-1] \
                              /dbint)

        gauss_elim(Mat,Jrad)
    
        intensl = ((Jrad[0,0:na]*2*rad['mu']+(0.5*delt[i]-rad['mu'])* \
                  rad['intex'][0:na,i])/(rad['mu']+0.5*delt[i])+ \
                  rad['intex'][2*na:3*na,i])
        intensr = ((Jrad[0,na:2*na]*2*rad['mu']+(0.5*delt[i]-rad['mu'])* \
                  rad['intex'][na:2*na,i])/(rad['mu']+0.5*delt[i])+ \
                  rad['intex'][3*na:4*na,i])
                
        rad['intens'][i,:] = intensl + intensr
        # Q/I, with Q > 0 for polarization along the meridian direction l
        rad['pol'][i,:] = (intensl - intensr)/rad['intens'][i,:]

def gauss_elim(Mat,X):
    """
    Perform Gaussian elimination on a set of matrices
    """
    A = Mat['A']
    B = Mat['B']
    C = Mat['C']
    L = Mat['L']

    nd,Np = X.shape

    D = np.zeros((nd,Np,Np))
    Z = np.zeros((nd,Np))
 
    D[0,:,:] = np.dot(linalg.inv(B[0,:,:]),C[0,:,:])
    Z[0,:] = np.dot(linalg.inv(B[0,:,:]),L[0,:])
    for i in range(1,nd):
        INVmat = linalg.inv(B[i,:,:]-np.dot(A[i,:,:],D[i-1,:,:]))
        D[i,:,:] = np.dot(INVmat,C[i,:,:])
        Z[i,:] = np.dot(INVmat,(L[i,:]+np.dot(A[i,:,:],Z[i-1,:])))

    X[nd-1,:] = Z[nd-1,:]
    for i in range(nd-2,-1,-1):
        X[i,:] = np.dot(D[i,:,:],X[i+1,:]) + Z[i,:]

def gauleg(x1,x2,n):
    """
    Numerical Recipes style implementation of Gauss-Legendre weights -- may
    be better implemented in numpy.polynomial.legendre.
    """

    x = np.zeros(n+1) # x[0] unused
    w = np.zeros(n+1) # w[0] unused
    eps = 3.0E-14
    m = (n+1)//2
    xm = 0.5*(x2+x1)
    xl = 0.5*(x2-x1)
    for i in range(1,m+1):
        z = np.cos(np.pi*(i-0.25)/(n+0.5))
        while True:
            p1 = 1.0
            p2 = 0.0
            for j in range(1,n+1):
                p3 = p2
                p2 = p1
                p1 = ((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j

            pp = n*(z*p1-p2)/(z*z-1.0)
            z1 = z
            z = z1 - p1/pp
            if abs(z-z1) <= eps:
                break

        x[i] = xm - xl*z
        x[n+1-i] = xm + xl*z
        w[i] = 2.0*xl/((1.0-z*z)*pp*pp)
        w[n+1-i] = w[i]

    return x[1:n+1], w[1:n+1]


def read_feautrier(infname="feautrier.out"):
    """
    Read output of the feautire.py
    """
    
    infile = open(infname,'r')
    line = infile.readline().split()
    nf = int(line[0])
    na = int(line[1])
    nu = np.empty(nf)
    mu = np.empty(na)
    intens = np.empty([nf,na])
    pol = np.empty([nf,na])
    # Read frequencies
    line = infile.readline().split()
    for i in range(nf):
        nu[i] = float(line[i])
    # Read frequency weights
    line = infile.readline().split()
    # Read angles
    line = infile.readline().split()
    for i in range(na):
        mu[i] = float(line[i])
    # Read angle weightss
    line = infile.readline().split()
    for i in range(nf):
        line = infile.readline().split()
        for j in range(na):
            intens[i,j] = float(line[2*j])
            pol[i,j] = float(line[2*j+1])

    return nu,mu,intens,pol


def plot_feautrier(imu,infile="feautrier.out",xlim=None,ylim=None,
                   outfile="feautrier.pdf"):
  
    """
    Plot intensity of feautrier output for specific polar angle.
    """
  
    plt.rc('text',usetex=True)
    plt.rc('font', **{'family' :"serif"})

    h = 6.62607015e-27

    plt.yscale('log')
    plt.xscale('log')
    if (ylim is not None): plt.ylim(ylim)
    if (xlim is not None): plt.xlim(xlim)

    plt.ylabel(r"$\nu L_\nu {\rm (erg/s)}$",fontsize=16)
    plt.xlabel(r"$E {\rm (keV)}$",fontsize=16)

    nuf,muf,intensf,polf = read_feautrier(infile)
    plt.plot(6.2415069e8*h*nuf,nuf*intensf[:,imu])

    plt.savefig(outfile)
    plt.close()
