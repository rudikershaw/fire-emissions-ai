"""The preprocess.py module main script is designed to take a single argument,
the path to a directory containing NASA EarthData Global Fire Emissions
Database GFED4.1s_yyyy.hdf5 files.

If a valid directory path is passed to the utility it should validate any hdf
files in the directory and output feedback.

The preprocess module also acts as an importable source of valdation and a
means of streaming training entries and their target outputs from specified
valid hdf files.
"""

import re
import os
import json
import h5py
import pprint
import itertools
from argparse import ArgumentParser, RawTextHelpFormatter


# -------------------------------------------------
# Utility functions class defined below.
# -------------------------------------------------
class Validator:
    """Used to validate hdf files to ensure they conform to the GFED format."""

    @staticmethod
    def valid_hdf_file(f: str):
        """Returns true if this file exists and has the correct extension."""
        ic = re.IGNORECASE
        match = re.search('_(\d{4})\.(hdf$|hdf4$|hdf5$|h4$|h5$|he2$|he5$)', f, ic)
        if os.path.isfile(f) and match != None:
            return True
        return False

    @staticmethod
    def valid_leaf_groups(group: str, month: str, hdf: h5py.File):
        """Checks all expected groups within month specific groups exist.

        Function prints errors to the console if they cannot be found. Returns
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
    def valid_hdf_structure(file_path: str):
        """Checks all groups and group months exists in the file.

        Prints errors to the console if expected groups cannot be found.
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


class GFEDDataParser:
    """Used to create a streamer object that parses groups of valid GFED files.

    Outputs entries designed for training a recurrent neural net model.
    These entries consist of a 5x5 matrix of tuples containing lat long
    positions and their emissions data, as well as a singular target tuple.
    """

    def __init__ (self, files):
        """files should be a touple of h5py hdf file objects ending _yyyy.hdf5.

        This touple of files provided should only include files pre-validated
        by the preprocess.Validator methods.
        """
        self.files = files
        self.max_i, self.max_j = files[0]["ancill/basis_regions"].shape
        # The index of the current file being processed.
        self.f = 0
        # The current month being processed.
        self.month = 1
        # Current index for looping through lat long matrices.
        self.i, self.j = 0, 0

    def get_entry(self, i: int, j: int):
        """Gets an entry from position i,j in current file."""
        m = self.month
        hdf = self.files[self.f]
        return {
            "latitude" : hdf["lat"][i][j],
            "longitude" : hdf["lon"][i][j],
            "region" : hdf["ancill/basis_regions"][i][j],
            "BB" : hdf["biosphere/{:02d}/BB".format(m)][i][j],
            "NPP" : hdf["biosphere/{:02d}/NPP".format(m)][i][j],
            "Rh" : hdf["biosphere/{:02d}/Rh".format(m)][i][j],
            "C" : hdf["emissions/{:02d}/C".format(m)][i][j],
            "DM" : hdf["emissions/{:02d}/DM".format(m)][i][j],
            "burned" : hdf["burned_area/{:02d}/burned_fraction".format(m)][i][j]
        }

    def get_target(self, i: int, j: int):
        """Gets an entry from position i,j in file f+plus in files."""
        if self.has_next_month() or self.has_next_file():
            m = self.month + 1 if self.has_next_month() else 1
            hdf = self.files[self.f] if m > 1 else self.files[self.f + 1]
            return {
                "BB" : hdf["biosphere/{:02d}/BB".format(m)][i][j],
                "NPP" : hdf["biosphere/{:02d}/NPP".format(m)][i][j],
                "Rh" : hdf["biosphere/{:02d}/Rh".format(m)][i][j],
                "C" : hdf["emissions/{:02d}/C".format(m)][i][j],
                "DM" : hdf["emissions/{:02d}/DM".format(m)][i][j],
                "burned" : hdf["burned_area/{:02d}/burned_fraction".format(m)][i][j]
            }
        return {}

    def has_next_month(self):
        """Checks whether there is another month in the current file."""
        return self.month < 12

    def has_next_coordinate(self):
        """Checks whether there is another coordinate in the current file."""
        return self.i < (self.max_i - 1) or self.j < (self.max_j - 1)

    def has_next_file(self):
        """Checks whether there is another file after the current file."""
        return self.f < (len(self.files) - 1)

    def has_next(self):
        """Checks whether there is another entry to parse."""
        return self.has_next_month() or self.has_next_coordinate() or self.has_next_file()

    def next(self):
        """Parses and converts the next training example."""
        ic = re.IGNORECASE
        name = self.files[self.f].filename
        search = re.search('_(\d{4})\.(hdf$|hdf4$|hdf5$|h4$|h5$|he2$|he5$)', name, ic)
        training = {
            "year" : search.group(1),
            "month" : self.month,
            "entries" : []
        }
        for x, y in itertools.product(range(-2, 3), range(-2, 3)):
            ti, tj = self.i - x, self.j - y
            # Wrap matrix if values fall off bottom.
            if ti < 0:
                ti = self.max_i + ti
            if tj < 0:
                tj = self.max_j + tj
            # Use modulus to wrap matrix if values fall off top.
            if ti != 0:
                ti = (self.max_i - 1) % ti
            if tj != 0:
                tj = (self.max_j - 1) % tj

            training["entries"].append(self.get_entry(ti, tj))

        training["target"] = self.get_target(self.i, self.j)
        self.increment()
        return training

    def reset(self, month_b=True, i_b=False, j_b=False):
        "Resets to the lowest value of each month, i, or j when rolled over."
        if month_b:
            self.month = 1
        if i_b:
            self.i = 0
        if j_b:
            self.j = 0

    def increment(self):
        """Move month, position, or year file as appropriate."""
        if self.has_next_month():
            self.month += 1
            return
        if self.i < (self.max_i - 1):
            self.reset()
            self.i += 1
            return
        if self.j < (self.max_j - 1):
            self.reset(i_b=True)
            self.j += 1
            return
        if self.has_next_file():
            self.reset(i_b=True, j_b=True)
            self.f += 1
        return


# -------------------------------------------------
# Script starts here.
# -------------------------------------------------
def validate_and_parse(directory):
    """Validates the files in a directory for GFED format and parses them."""
    print("Processing files in directory '" + directory + "'.")
    print("The following files adhere to the expected GFED format.")
    files = []
    for fi in sorted(os.listdir(directory)):
        full_path = directory + fi
        if Validator.valid_hdf_file(full_path) and Validator.valid_hdf_structure(full_path):
            files.append(h5py.File(full_path, 'r'))
            print("  " + fi)

    print("...")
    if len(files) > 1:
        print("Directory contains valid GFED HDF files for training data.")
        parser = GFEDDataParser(files)
        pp = pprint.PrettyPrinter(indent=4)
        count = 0
        entry = {}
        while parser.has_next():
            entry = parser.next()
            count += 1
            print("Entries found: " + str(count), end="\r")

        print("Entries parsed: " + str(count))
        print("Printing example entry: \n")
        pp.pprint(entry)
    elif len(files) == 1:
        print("At least 2 valid HDF GFED files required but found 1.")
    else:
        print("No valid HDF GFED files found in that directory.")

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("directory", help="Directory with GFED4.1s_yyyy HDF files")
    validate_and_parse(parser.parse_args().directory)
