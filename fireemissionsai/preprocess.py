"""
The preprocess.py module main script is designed to take a single argument,
the path to a directory containing NASA EarthData Global Fire Emissions
Database GFED4.1s_yyyy.hdf5 files.

If a valid directory path is passed to the utility it should validate any hdf
files in the directory and output feedback.

The preprocess module also acts as an importable source of valdation and a
means of streaming training entries and their target outputs from specified
valid hdf files.
"""

import h5py, json, itertools, re, os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

# -------------------------------------------------
# Utility functions class defined below.
# -------------------------------------------------
class Validator:

    @staticmethod
    def valid_hdf_file(f):
        extensions = "hdf","hdf4","hdf5","h4","h5", "he2", "he5"
        if os.path.isfile(f) and f.split(".")[-1].lower() in extensions:
            return True
        return False

    @staticmethod
    def valid_leaf_groups(group, month, hdf):
        groups_and_leaves = {
            "biosphere": ("BB", "NPP", "Rh"),
            "burned_area": ("burned_fraction",),
            "emissions": ("C", "DM")
        }
        valid = True
        for leaf in groups_and_leaves[group]:
            full_group = "{}/{:02d}/{}".format(group, month, leaf)
            if full_group not in hdf:
                valid = False
                print("Expected group '" + full_group + "' not in HDF file.")
        return valid


    @staticmethod
    def valid_hdf_structure(file_path):
        hdf = h5py.File(file_path, 'r')
        valid = True
        for group in "ancill/basis_regions", "lon", "lat":
            if group not in hdf:
                valid = False
                print("Expected group '" + group + "' not in HDF file '" + hdf.filename + "'")
        for group in "biosphere", "burned_area", "emissions":
            for month in range(1,13):
                full_group = "{}/{:02d}".format(group, month)
                if full_group not in hdf:
                    valid = False
                    print("Expected group '" + full_group + "' not in HDF file '" + hdf.filename + "'")
                else:
                    valid = valid and Validator.valid_leaf_groups(group, month, hdf)
        return valid


# -------------------------------------------------
# Script starts here.
# -------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("directory", help="Directory with GFED4.1s_yyyy HDF files")
    directory = parser.parse_args().directory

    print("Processing files in directory '" + directory + "'.")
    print("The following files adhered to the correct format.")
    found = False
    for f in sorted(os.listdir(directory)):
        full_path = directory + f
        if Validator.valid_hdf_file(full_path) and Validator.valid_hdf_structure(full_path):
            found = True
            print("  " + f)

    print("...")
    if not found:
        print("No valid HDF files found in that directory.")
