"""
Reading an opacity table in Python the way OpacityTable does, for the tests that need
analytic targets on an arbitrary table: the grids, the Planck and Rosseland group means,
the optional Rosseland absorption block, the free electrons per atom, the bracket and
weights of a (T, rho), and the two-band split of tops_regrid.py.
"""
import numpy as np

H = 6.62607015e-27
C = 2.99792458e10
KB = 1.380649e-16
KEV = 1.602176634e-9
MP = 1.67262192369e-24
MU = 1.66053906660e-24
SIGMA_T = 6.65248e-25
KEV_K = KEV/KB


class Table:
    """an opacity table and, if present beside it, its free electrons per atom"""

    def __init__(self, path, nfree_path=None):
        with open(path) as f:
            vals = np.fromstring(f.read(), sep=' ')
        ng, nt, nr = (int(v) for v in vals[:3])
        pos = 3
        self.lo = vals[pos:pos+ng]*KEV
        pos += ng
        self.tg = vals[pos:pos+nt]*KEV_K
        pos += nt
        self.rg = vals[pos:pos+nr]
        pos += nr
        n2 = nt*nr
        self.gray_ross = vals[pos:pos+n2].reshape(nt, nr)
        pos += n2
        self.gray_plan = vals[pos:pos+n2].reshape(nt, nr)
        pos += n2
        n3 = ng*nt*nr
        self.ross = vals[pos:pos+n3].reshape(ng, nt, nr)
        pos += n3
        self.plan = vals[pos:pos+n3].reshape(ng, nt, nr)
        pos += n3
        self.rossabs = None
        if len(vals) >= pos + n3:
            self.rossabs = vals[pos:pos+n3].reshape(ng, nt, nr)
        self.ng, self.nt, self.nr = ng, nt, nr
        self.dle = (np.log10(self.lo[-1]) - np.log10(self.lo[0]))/(ng - 1)
        self.edges = 10.**(np.log10(self.lo[0]) + self.dle*np.arange(ng + 1))
        self.edges[:ng] = self.lo
        self.nfree = self.abar = None
        if nfree_path is None:
            nfree_path = (path[:-4] if path.endswith('.txt') else path) + '_nfree.txt'
        try:
            self.read_nfree(nfree_path)
        except OSError:
            pass

    def read_nfree(self, path):
        rows = []
        with open(path) as f:
            for line in f:
                if line.startswith('#'):
                    if 'Abar =' in line:
                        self.abar = float(line.split('Abar =')[1].split()[0])
                    continue
                rows.append(line)
        nt, nr = int(rows[0]), int(rows[1])
        self.nfree = np.array([r.split() for r in rows[2+nt+nr:2+nt+nr+nt]], float)

    def locate(self, temp, rho):
        """bracket (jj, ii) and weights (xj, xi) as OpacityTable::Locate, clamped"""
        lt, lr = np.log10(self.tg), np.log10(self.rg)
        dlt = (lt[-1] - lt[0])/(self.nt - 1)
        dlr = (lr[-1] - lr[0])/(self.nr - 1)
        xi = (np.log10(rho) - lr[0])/dlr
        ii = min(max(int(np.floor(xi)), 0), self.nr - 2)
        xi = min(max(xi - ii, 0.), 1.)
        jj = min(max(int(np.floor((np.log10(temp) - lt[0])/dlt)), 0), self.nt - 2)
        while jj < self.nt - 2 and self.tg[jj+1] < temp:
            jj += 1
        while jj > 0 and self.tg[jj] > temp:
            jj -= 1
        xj = np.log(temp/self.tg[jj])/np.log(self.tg[jj+1]/self.tg[jj])
        xj = min(max(xj, 0.), 1.)
        return jj, ii, xj, xi

    def _bilinear(self, arr, temp, rho, log=True):
        jj, ii, xj, xi = self.locate(temp, rho)
        c = np.array([arr[..., jj, ii], arr[..., jj+1, ii], arr[..., jj, ii+1],
                      arr[..., jj+1, ii+1]])
        wts = np.array([(1-xi)*(1-xj), (1-xi)*xj, xi*(1-xj), xi*xj])
        if log and np.all(c > 0.):
            return np.exp(np.tensordot(wts, np.log(c), axes=1))
        return np.tensordot(wts, c, axes=1)

    def planck(self, temp, rho):
        """Planck group means [cm^2/g] (ng)"""
        return self._bilinear(self.plan, temp, rho)

    def rosseland(self, temp, rho):
        return self._bilinear(self.ross, temp, rho)

    def rosseland_abs(self, temp, rho):
        if self.rossabs is None:
            return None
        return self._bilinear(self.rossabs, temp, rho)

    def nfree_at(self, temp, rho):
        return float(self._bilinear(self.nfree, temp, rho, log=False))

    def ne(self, temp, rho):
        """electron density with table_ne"""
        return rho*self.nfree_at(temp, rho)/(self.abar*MU)

    def split(self, temp, rho, s, w0=0.01, floor=0.003, closure='w'):
        """
        the two-band split of every group as OpacityTable::Split: w, kappa_h, kappa_l
        [cm^2/g] from the Planck mean, the Rosseland absorption (the block, else the
        Rosseland mean less s) and the scattering s per gram (scalar or per group)
        """
        kp = self.planck(temp, rho)
        kra = self.rosseland_abs(temp, rho)
        s = np.broadcast_to(np.asarray(s, float), kp.shape)
        if kra is None:
            kra = self.rosseland(temp, rho) - s
        ra = np.maximum(kra, floor*s)
        r, pt = ra + s, kp + s
        w = np.ones_like(kp)
        kh = kp.copy()
        kl = kp.copy()
        two = (r < pt*(1. - 1.e-9)) & (kp > 0.)
        with np.errstate(divide='ignore', invalid='ignore'):
            kh_g = np.where(two, kp*r/ra - s, kp)
            w_g = np.where(kh_g > 0., kp/kh_g, 1.)
        w[two], kh[two], kl[two] = w_g[two], kh_g[two], 0.
        if closure == 'w':
            b = pt - r*(1. - 2.*w0)
            d = np.maximum(b*b - 4.*w0*w0*r*pt, 0.)
            a = (b + np.sqrt(d))/(2.*w0)
            kl_w = (pt - w0*a)/(1. - w0) - s
            ok = two & (kl_w >= 0.)
            w[ok], kh[ok], kl[ok] = w0, a[ok] - s[ok], kl_w[ok]
        return w, kh, kl


def planck_nu(nu, temp):
    x = H*nu/(KB*temp)
    return 2.*H*nu**3/C**2/np.expm1(x)


def band_integral(f, elo, ehi, temp, n=401):
    """int f(nu, T) dnu over [elo, ehi] erg, trapezoid in ln nu"""
    lnu = np.linspace(np.log(elo/H), np.log(ehi/H), n)
    nu = np.exp(lnu)
    w = np.full(n, lnu[1] - lnu[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return float((f(nu, temp)*nu*w).sum())


def planck_band(elo, ehi, temp):
    """int B_nu dnu over the band"""
    return band_integral(planck_nu, elo, ehi, temp)


def photon_band(elo, ehi, temp):
    """int B_nu/(h nu) dnu over the band"""
    return band_integral(lambda nu, t: planck_nu(nu, t)/(H*nu), elo, ehi, temp)


def dplanck_dt_band(elo, ehi, temp):
    """int dB_nu/dT dnu over the band"""
    def dbdt(nu, t):
        x = H*nu/(KB*t)
        ex = np.exp(-x)
        return 2.*H*nu**3/C**2*x/t*ex/(-np.expm1(-x))**2
    return band_integral(dbdt, elo, ehi, temp)
