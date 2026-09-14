#! /usr/bin/env python

"""
Read in photon lists written by independent processes and join
into single file
"""

#SWD modify writing to file in the case of larger of memory cap

# python standard modules
import argparse
from os import system

# Athena++ modules
import athena_mc as athenamc

def list_match(list1, list2):
    """
    Check if headers of to lists match
    """

    if list1 is None:
        return False
    if list2 is None:
        return False
    if list1['npars'] != list2['npars']:
        return False
    if list1['polarized'] != list2['polarized']:
        return False
    return True

def join(filelist, outfile, skip, nproc):
    """
    Join a list of list files together into a single file
    """

    # First Read through and check headers match, get ntot
    firstlist = True
    ntot = 0
    length = 0
    dt = 0.
    for infile in filelist:
        if firstlist:
            phlist = athenamc.read_list(infile, data=False)
            firstlist = False
            ntot += phlist['ntot']
            length += phlist['length']
            dt += phlist['dt']
        else:
            addlist = athenamc.read_list(infile, data=False)
            if list_match(phlist, addlist):
                ntot += addlist['ntot']
                length += addlist['length']
                dt += phlist['dt']
            else:
                if addlist is None:
                    if skip:
                        print(infile+" not found, skipping.\n")
                    else:
                        raise RuntimeError(infile+" not found, aborting.\n")
                else:
                    raise RuntimeError("List headers do not match for "+infile+".\n")
    print(f"Final list contains {length:d} photons out of {ntot:d}"
          " initialized.")

    # If headers all match, read data and write output file as
    firstlist = True
    for infile in filelist:
        print("Reading: " + infile)
        phlist = athenamc.read_list(infile, data=True)
        if phlist is None:
            continue
        if firstlist:
            phlist['ntot'] = ntot
            phlist['dt'] = dt / float(nproc)
            firstlist = False
            athenamc.write_list(outfile, phlist, header=True, length=length)
        else:
            athenamc.write_list(outfile, phlist, header=False)

# Main function
def main(**kwargs):
    """
    Wrapper for running the join function above. Parameters for the joining
    of the lists are passed via kwargs and set from command line with argparse
    """
    nproc = kwargs['nproc']

    start = kwargs['start']
    end = kwargs['end']

    basename = kwargs['basename']

    if (start == end) or (not kwargs['multi_out']):
        # combine all outputs into single file
        if kwargs['outfile'] is None:
            outfile = basename+'.list'
        else:
            outfile = kwargs['outfile']

        fileliststring = ""
        filelist = []
        nout = end-start+1
        for j in range(start,end+1):
            for i in range(nproc):
                filename = f"{basename}.proc{i:d}.{j:05d}.list"
                filelist.append(filename)
                fileliststring += filename + " "

        join(filelist, outfile, kwargs['skip'], nproc)

        if kwargs['rm']:
            # Remove source files after write
            system("rm " + fileliststring)

    else:
        # Each output stage is its own list file
        for j in range(start,end+1):
            fileliststring = ""
            filelist = []
            for i in range(nproc):
                filename = f"{basename}.proc{i:d}.{j:05d}.list"
                filelist.append(filename)
                fileliststring += filename + " "

            outfile = f"{basename}.{j:05d}.list"
            join(filelist, outfile, kwargs['skip'], nproc)

            if kwargs['rm']:
                # Remove source files after write
                system("rm " + fileliststring)

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basename',
        help = 'basename of list files')
    parser.add_argument('nproc',
        type = int,
        help = 'number of processes')
    parser.add_argument('start',
        type = int,
        help = 'starting output number')
    parser.add_argument('end',
        type = int,
        help = 'ending output number')
    parser.add_argument('-skip',
        action = 'store_true',
        help = 'simply skip a list if not present')
    parser.add_argument('-multi_out',
        action = 'store_true',
        help = 'each outputs number is indvidual list file')
    parser.add_argument('-rm',
        action = 'store_true',
        help = 'delete origin files')
    parser.add_argument('--outfile',
        default = None,
        help = 'output filename for spectrum')

    args = parser.parse_args()
    main(**vars(args))
