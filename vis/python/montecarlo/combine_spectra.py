#! /usr/bin/env python

"""
Combine spectrum files written by make_spectrum.py or by the code into one spectrum.

    combine_spectra.py [-m METHOD] -o OUT SPEC SPEC [SPEC ...]

A wrapper for athena_mc.add_spectra(): the inputs are folded together in the order
given and the result is written to OUT in the same format, so it can be plotted with
plot_spectrum.py or combined again.  Every input has to have the same bins, units and
polarization, and either all or none of them must carry errors.

Which method
------------
-m statistical (the default) sums the spectra.  It is for spectra that each hold a
share of the photons of one interval, as the lists the ranks of an MPI run write for
one output do; their integration times must agree, and the sum is the spectrum of all
the photons together:

    I = sum_i I_i            sigma = sqrt( sum_i sigma_i^2 )

-m time averages the spectra weighted by their integration times.  It is for spectra
that are each already a complete estimate, as the outputs of a run with nout > 1 are,
or two independent runs of the same problem; the result is the spectrum of the whole
time span, and the noise falls as the total time grows:

    I = sum_i dt_i I_i / sum_i dt_i    sigma = sqrt( sum_i (dt_i sigma_i)^2 ) / sum_i dt_i

Summing complete spectra would count the luminosity once per input, and averaging
shares of one interval would divide it by the number of ranks, so the choice matters;
the luminosity of each input and of the result are printed as a check.  In both cases
ntot, the number of photons run, is summed.

Examples
--------
The spectra make_spectrum.py wrote for each output of a run, averaged into one:

    combine_spectra.py -m time -o xrb.out1.spec xrb.out1.000*.spec

Two runs of the same problem, made separately, averaged by their integration times:

    combine_spectra.py -m time -o xrb.both.spec run1/xrb.out1.spec run2/xrb.out1.spec

Spectra made from the lists of single ranks of one output, summed into the spectrum of
that output:

    combine_spectra.py -o xrb.out1.00000.spec xrb.out1.proc*.00000.spec

make_spectrum.py does both combinations itself when given the list files, so this
script is for spectra that already exist.
"""

# python standard modules
import argparse
import os

# Athena++ modules
import athena_mc as athenamc


def main(args):
    """
    Read the input spectra, fold them together and write the result, using the options
    returned by parse_args().
    """

    combined = {}
    for i, file in enumerate(args.inputs):
        spectrum = athenamc.read_spectrum(file)
        if not args.quiet:
            print(f"  {file}: dt {spectrum['dt']:.6e}, ntot {spectrum['ntot']}, "
                  f"luminosity {athenamc.get_luminosity(spectrum):.6e}")
        try:
            combined = athenamc.add_spectra(combined, spectrum, method=args.method)
        except RuntimeError as err:
            hint = ""
            if args.method == 'statistical' and 'integration times' in str(err):
                hint = "\nspectra of different intervals are averaged with -m time"
            raise SystemExit(f"{file} cannot be combined with {args.inputs[0]}"
                             f"{'' if i == 1 else ' and those before it'}: {err}{hint}") \
                from None

    athenamc.write_spectrum(args.out, combined)
    if not args.quiet:
        print(f"{args.method} combination of {len(args.inputs)} spectra: "
              f"dt {combined['dt']:.6e}, ntot {combined['ntot']}, "
              f"luminosity {athenamc.get_luminosity(combined):.6e}")
        print(f"wrote {args.out}")


def parse_args(argv=None):
    """
    Parse and check command-line options; exits with a usage message on bad input
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+',
                        help='spectrum files to combine')
    parser.add_argument('-o', '--out', required=True,
                        help='combined output spectrum')
    parser.add_argument('-m', '--method', default='statistical',
                        choices=athenamc.ADD_SPECTRA_METHODS,
                        help='statistical sums shares of one interval, time averages '
                             'complete spectra weighted by integration time')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='print nothing but errors')

    args = parser.parse_args(argv)

    if len(args.inputs) < 2:
        parser.error("need at least two spectra to combine")
    missing = [f for f in args.inputs if not os.path.isfile(f)]
    if missing:
        parser.error(f"input file(s) not found: {', '.join(missing)}")

    # never write the output over an input
    inputs = {os.path.realpath(f) for f in args.inputs}
    if os.path.realpath(args.out) in inputs:
        parser.error(f"--out {args.out} would overwrite an input file")

    return args


if __name__ == '__main__':
    main(parse_args())
