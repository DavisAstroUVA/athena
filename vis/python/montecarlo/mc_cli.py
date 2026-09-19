"""
Shared utilities for generating spectra and images from photon lists

make_spectrum.py and make_image.py take the same input, the lists a run writes, named
<base>.proc<rank>.<output>.list, and treat it the same way: each list is streamed
through in chunks, the ranks of one output are summed, and the outputs are written
separately or averaged over time.  All of that lives here, so that a script supplies
only the function that bins one chunk of photons and the functions that add and write
its results, and the two cannot drift apart.
"""

# python standard modules
import collections
import glob
import os
import re
import sys

# Athena++ modules
import athena_mc as athenamc
from athena_mc import Photons

# chunks read between progress messages
PROGRESS_EVERY = 20

# Optional user module supplying --screen functions.  A screen.py is a per-dataset input
# kept next to the lists, so the working directory is searched as well as PYTHONPATH.
# Appended rather than prepended, so it cannot shadow an installed module.
sys.path.append(os.getcwd())
try:
    import screen
except ModuleNotFoundError:
    screen = None

# how the code names a photon list: <base>.proc<rank>.<output>.list
LIST_NAME = re.compile(r'^(?P<base>.+)\.proc(?P<rank>\d+)\.(?P<output>\d+)\.list$')

# the list files of one output of one run; output is None for a file not named like one
Output = collections.namedtuple('Output', ['base', 'output', 'files'])


def load_screen(name, script):
    """
    Return the function of the given name from the user's screen.py, or None when no
    screen was asked for.  Exits with a message saying what was looked for and where;
    script is the name of the calling script, for that message.
    """

    if name is None:
        return None
    if screen is None:
        raise SystemExit(
            f"--screen={name} needs a screen.py module providing a function "
            f"{name}(phots), and none was importable.\n"
            "Looked in the working directory, on PYTHONPATH, "
            f"and alongside {script}.\n"
            "It takes a Photons chunk and returns a boolean array that is True for "
            "the photons to leave out.")
    function = getattr(screen, name, None)
    if not callable(function):
        available = sorted(n for n in dir(screen)
                           if not n.startswith('_') and callable(getattr(screen, n)))
        raise SystemExit(
            f"screen.py ({getattr(screen, '__file__', 'location unknown')}) has no "
            f"function named {name!r}.\n"
            f"Defined there: {', '.join(available) if available else 'nothing callable'}")
    return function


def expand_files(patterns):
    """
    Expand any glob patterns the shell left unexpanded, keeping the order given; a
    pattern that matches nothing is kept as a name so that it is reported as missing.
    """

    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)) or [pattern])
    return files


def group_outputs(files):
    """
    Group list files by run and output number, ranks in order within each output and
    outputs in order within each run.
    """

    groups = {}
    for file in files:
        match = LIST_NAME.match(file)
        if match:
            key = (match['base'], match['output'])
        else:
            key = (os.path.splitext(file)[0], None)
        groups.setdefault(key, []).append(file)
    return [Output(base, output, sorted(names))
            for (base, output), names in sorted(groups.items(), key=lambda kv: (
                kv[0][0], kv[0][1] or ''))]


def output_names(outputs, outfile, combine, ext):
    """
    Names of the files to write, with extension ext: one per output, or a single one
    with combine.  An outfile given for several outputs has the output number put before
    its extension.
    """

    if combine:
        if outfile is not None:
            return [outfile]
        return [outputs[0].base + ext]

    names = []
    for out in outputs:
        if outfile is None:
            tag = '' if out.output is None else f".{out.output}"
            names.append(f"{out.base}{tag}{ext}")
        elif len(outputs) == 1:
            names.append(outfile)
        else:
            stem, oext = os.path.splitext(outfile)
            tag = os.path.basename(out.base) if out.output is None else out.output
            names.append(f"{stem}.{tag}{oext}")
    return names


def check_inputs(parser, args, combine_ext):
    """
    The checks on the input lists that make_spectrum.py and make_image.py share, done
    after parsing: expand globs, require every file to exist, group them into
    args.outputs, name the files to write in args.outnames, and refuse to write over an
    input.  parser.error() is used, so a failure exits with the usage message.
    """

    args.infile = expand_files(args.infile)
    missing = [f for f in args.infile if not os.path.isfile(f)]
    if missing:
        parser.error(f"input file(s) not found: {', '.join(missing)}")
    args.outputs = group_outputs(args.infile)

    # a combined result from several runs has no natural name
    bases = {out.base for out in args.outputs}
    if args.combine and args.outfile is None and len(bases) > 1:
        parser.error("--combine over lists from different runs needs --outfile")

    args.outnames = output_names(args.outputs, args.outfile, args.combine, combine_ext)
    inputs = {os.path.realpath(f) for f in args.infile}
    clash = [n for n in args.outnames if os.path.realpath(n) in inputs]
    if clash:
        parser.error(f"output file(s) {clash} would overwrite an input file")


def bin_list(infile, bin_chunk, add, screen_function=None, calclum=False):
    """
    Bin one list file, streamed in chunks so that the whole list is never in memory.
    bin_chunk(phots, mask) bins one chunk of photons, mask being None or True for the
    photons to leave out; add(a, b) folds two of its results, starting from {}.  Returns
    the result and the luminosity of the list, None unless calclum.
    """

    reader = athenamc.read_list_generator(infile)
    header = next(reader)['header']
    print(f"  reading {infile} ({header['length']} samples)")

    result = {}
    luminosity = 0.0 if calclum else None
    for nchunk, chunk in enumerate(reader):
        if nchunk > 0 and nchunk % PROGRESS_EVERY == 0:
            print(f"    {chunk['remaining']} samples remain")
        phlist = dict(header, list=chunk['chunk'], length=chunk['length'])
        if calclum:
            luminosity += athenamc.get_luminosity_list(phlist)

        phots = Photons(phlist)
        mask = None if screen_function is None else screen_function(phots)
        result = add(result, bin_chunk(phots, mask))
        if chunk['done']:
            break

    return result, luminosity


def bin_output(out, bin_chunk, add, screen_function=None, calclum=False):
    """
    Sum the ranks of one output; each rank holds a share of the same photons, so the
    sum is the result for all of them.  Returns it and the luminosity of the output,
    None unless calclum.
    """

    result = {}
    luminosity = 0.0 if calclum else None
    for infile in out.files:
        part, lum = bin_list(infile, bin_chunk, add, screen_function, calclum)
        result = add(result, part)
        if calclum:
            luminosity += lum
    return result, luminosity


def argv_from(positional, options):
    """
    Build a command line from values, so that a notebook or another script can use a
    script's parse_args() and its checks instead of a second interface.  positional is
    the list of positional values in order (a nested list or tuple is spread, for a
    file list); options maps option names without the leading dashes to values: None
    and False are left out, True gives a bare flag, a list or tuple gives the flag
    followed by each value, anything else the flag followed by its str().
    """

    argv = []
    for value in positional:
        if isinstance(value, (list, tuple)):
            argv.extend(str(v) for v in value)
        else:
            argv.append(str(value))
    for name, value in options.items():
        if value is None or value is False:
            continue
        flag = '--' + name
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in value)
        else:
            argv.extend([flag, str(value)])
    return argv


def bin_outputs(args, script, bin_chunk, add, write, screen_function=None):
    """
    The main loop of a binning script: one result per output written to args.outnames,
    or with args.combine all outputs averaged in time into args.outnames[0].  script
    is the caller's name, for messages; bin_chunk and add are as for bin_list, and
    write(filename, result) writes one result.  args.screen names the screen function
    in the user's screen.py unless one is passed in directly, as a notebook does, and
    args.calclum asks for the luminosity of each output to be printed.
    """

    if screen_function is None:
        screen_function = load_screen(args.screen, script)
    calclum = getattr(args, 'calclum', False)

    combined = {}
    lum_dt = 0.0
    total_dt = 0.0
    for i, out in enumerate(args.outputs):
        label = out.base if out.output is None else f"{out.base} output {out.output}"
        print(f"{label}: {len(out.files)} file(s)")
        result, luminosity = bin_output(out, bin_chunk, add, screen_function, calclum)
        if calclum:
            print(f"  luminosity: {luminosity:e}")
            lum_dt += luminosity*result['dt']
            total_dt += result['dt']

        if args.combine:
            # outputs are independent intervals, so their results are averaged in time
            combined = add(combined, result, method='time')
        else:
            write(args.outnames[i], result)
            print(f"  wrote {args.outnames[i]}")

    if args.combine:
        if calclum and len(args.outputs) > 1:
            print(f"combined luminosity: {lum_dt/total_dt:e}")
        write(args.outnames[0], combined)
        print(f"wrote {args.outnames[0]}")
