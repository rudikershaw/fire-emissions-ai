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
    """Used to validate hdf files to ensure they conform to the GFED format."""
    @staticmethod
    def valid_hdf_file(f):
        """
        Takes a string representation of a file and returns true if this
        file exists and has the correct extension.
        """
        extensions = "hdf","hdf4","hdf5","h4","h5", "he2", "he5"
        if os.path.isfile(f) and f.split(".")[-1].lower() in extensions:
            return True
        return False

    @staticmethod
    def valid_leaf_groups(group, month, hdf):
        """
        Checks that all expected groups within month specific groups exist
        and prints errors to the console if they cannot be found. Returns
        true if all expected groups are present, otherwise false.
        """
        groups_and_leaves = {
            "biosphere": ("BB", "NPP", "Rh"),
            "burned_area": ("burned_fraction",),
            "emissions": ("C", "DM")
        }
        valid = True
        expected_message = "Expected group '{}' not in HDF file '{}'"
        for leaf in groups_and_leaves[group]:
            full_group = "{}/{:02d}/{}".format(group, month, leaf)
            if full_group not in hdf:
                valid = False
                print(expected_message.format(full_group, hdf.filename))
        return valid


    @staticmethod
    def valid_hdf_structure(file_path):
        """
        Checks that all groups and group months exists in the file and
        prints errors to the console if expected groups cannot be found.
        Returns true if all expected groups are present, otherwise false.
        """
        hdf = h5py.File(file_path, 'r')
        valid = True
        expected_message = "Expected group '{}' not in HDF file '{}'"
        for group in "ancill/basis_regions", "lon", "lat":
            if group not in hdf:
                valid = False
                print(expected_message.format(group, hdf.filename))
        for group in "biosphere", "burned_area", "emissions":
            for month in range(1,13):
                full_group = "{}/{:02d}".format(group, month)
                if full_group not in hdf:
                    valid = False
                    print(expected_message.format(full_group, hdf.filename))
                else:
                    valid = valid and Validator.valid_leaf_groups(group, month, hdf)
        return valid


class EmissionsEntryStreamer:
    """
    Used to create a streamer object that parses groups of valid GFED files
    and outputs entries designed for training a recurrent neural net model.
    These entries consist of a 5x5 matrix of tuples containing lat long
    positions and their emissions data, as well as a singular target tuple.
    """
    # Current index for looping through lat long matrices.
    i, j = 0, 0
    # Max size of matrices dimensions.
    max_i, max_j = 0, 0
    # The index of the current file being processed
    f = 0

    def __init__ (self, files):
        """
        files should be a touple of h5py hdf file objects ending _yyyy.hdf5.
        This touple of files should only include files pre-validated by the
        preprocess.Validator
        """
        self.files = files
        self.max_i, self.max_j = files[0]["ancill/basis_regions"].shape

    def hasNext(self):
        """Checks whether there is another entry to parse."""
        more_in_file = (self.max_i - 1) < self.i and (self.max_j - 1) < self.j
        more_files = (len(self.files) - 1) < f
        return more_in_file or more_files


# -------------------------------------------------
# Script starts here.
# -------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("directory", help="Directory with GFED4.1s_yyyy HDF files")
    directory = parser.parse_args().directory

    print("Processing files in directory '" + directory + "'.")
    print("The following files adhere to the expected GFED4 format.")
    found = False
    for f in sorted(os.listdir(directory)):
        full_path = directory + f
        if Validator.valid_hdf_file(full_path) and Validator.valid_hdf_structure(full_path):
            found = True
            print("  " + f)

    print("...")
    if not found:
        print("No valid HDF GFED4 files found in that directory.")
