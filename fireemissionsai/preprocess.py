"""The preprocess.py module main script is designed to take a single argument,
the path to a directory containing NASA EarthData Global Fire Emissions
Database GFED4.1s_yyyy.hdf5 files.

If a valid directory path is passed to the utility it should validate any hdf
files in the directory and output feedback. Additionally an 'ouput' directory
will be created and training and validation .csv files will be created containing
example data.

The preprocess module also acts as an importable source of valdation and a
means of streaming training entries and their target outputs from specified
valid hdf files.
"""

import re
import os
import csv
from argparse import ArgumentParser, RawTextHelpFormatter

import h5py

class GFEDDataParser:
    """Used to create a streamer object that parses groups of valid GFED files.

    Outputs entries designed for training a recurrent neural net model.
    These entries consist of a 5x5 matrix of tuples containing lat long
    positions and their emissions data, as well as a singular target tuple.
    """

    def __init__(self, files):
        """files should be a touple of h5py hdf file objects ending _yyyy.hdf5.

        This touple of files provided should only include files pre-validated
        by the preprocess validator methods.
        """
        self.files = files
        self.max_i, self.max_j = files[0]["ancill/basis_regions"].shape
        # The index of the current file being processed.
        self.file_no = 0
        # The current month being processed.
        self.month = 1
        # Current index for looping through lat long matrices.
        self.i, self.j = 0, 0

    def current_file(self, next_file=False):
        """Gets the current hdf file or the following file if next=True."""
        return self.files[self.file_no + (1 if next_file else 0)]

    def get_entry(self, i: int, j: int):
        """Gets an entry from position i,j in current file."""
        hdf = self.current_file()
        name = hdf.filename
        regex_string = r'_(\d{4})\.(hdf$|hdf4$|hdf5$|h4$|h5$|he2$|he5$)'
        search = re.search(regex_string, name, re.IGNORECASE)
        return [
            search.group(1), # Year
            self.month,
            hdf["lat"][i][j],
            hdf["lon"][i][j],
            hdf["ancill/basis_regions"][i][j],
            hdf["biosphere/{:02d}/BB".format(self.month)][i][j],
            hdf["biosphere/{:02d}/NPP".format(self.month)][i][j],
            hdf["biosphere/{:02d}/Rh".format(self.month)][i][j],
            hdf["emissions/{:02d}/C".format(self.month)][i][j],
            hdf["emissions/{:02d}/DM".format(self.month)][i][j],
            hdf["burned_area/{:02d}/burned_fraction".format(self.month)][i][j]
        ]

    def get_target(self, i: int, j: int):
        """Gets an entry from position i,j in file f+plus in files."""
        if self.has_next_month() or self.has_next_file():
            target_month = self.month + 1 if self.has_next_month() else 1
            hdf = self.current_file() if target_month > 1 else self.current_file(next_file=True)
            return [
                hdf["biosphere/{:02d}/BB".format(target_month)][i][j],
                hdf["biosphere/{:02d}/NPP".format(target_month)][i][j],
                hdf["biosphere/{:02d}/Rh".format(target_month)][i][j],
                hdf["emissions/{:02d}/C".format(target_month)][i][j],
                hdf["emissions/{:02d}/DM".format(target_month)][i][j],
                hdf["burned_area/{:02d}/burned_fraction".format(target_month)][i][j]
            ]
        return []

    def has_next_month(self):
        """Checks whether there is another month in the current file."""
        return self.month < 12

    def has_next_coordinate(self):
        """Checks whether there is another coordinate in the current file."""
        return self.i < (self.max_i - 1) or self.j < (self.max_j - 1)

    def has_next_file(self):
        """Checks whether there is another file after the current file."""
        return self.file_no < (len(self.files) - 1)

    def has_next(self):
        """Checks whether there is another entry to parse."""
        return self.has_next_month() or self.has_next_coordinate() or self.has_next_file()

    def next(self):
        """Parses and converts the next training example."""
        training = self.get_entry(self.i, self.j)
        target = self.get_target(self.i, self.j)
        self.increment()
        return (training, target)

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
            self.file_no += 1
        return


def valid_hdf_file(file_path: str):
    """Returns true if this file exists and has the correct extension."""
    regex_string = r'_(\d{4})\.(hdf$|hdf4$|hdf5$|h4$|h5$|he2$|he5$)'
    match = re.search(regex_string, file_path, re.IGNORECASE)
    if os.path.isfile(file_path) and match != None:
        return True
    return False

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
        for month in range(1, 13):
            full_group = "{}/{:02d}".format(group, month)
            if full_group not in hdf:
                valid = False
                print(expected_message.format(full_group, hdf.filename))
            else:
                valid = valid and valid_leaf_groups(group, month, hdf)
    return valid

def valid_files(directory):
    """Collects and returns all valid files in the provided directory."""
    files = []
    for file_name in sorted(os.listdir(directory)):
        full_path = directory + file_name
        if valid_hdf_file(full_path) and valid_hdf_structure(full_path):
            files.append(h5py.File(full_path, 'r'))
            print("  " + file_name)
    return files

def validate_and_parse(directory, size):
    """Validates the files in a directory for GFED format and parses them."""
    print("Processing files in directory '" + directory + "'.")
    print("The following files adhere to the expected GFED format.")
    files = valid_files(directory)

    print("...")
    if len(files) > 1:
        print("Directory contains valid GFED HDF files for training data.")
        parser = GFEDDataParser(files)
        count = 0
        # Create csvs for features and targets for training and testing.
        if not os.path.isdir("output"):
            os.makedirs("output")
        with open("output/train-features.csv", "w") as csv_trainf, \
             open("output/train-targets.csv", "w") as csv_traint, \
             open("output/validation-features.csv", "w") as csv_valf, \
             open("output/validation-targets.csv", "w") as csv_valt, \
             open("output/test-features.csv", "w") as csv_testf, \
             open("output/test-targets.csv", "w") as csv_testt:
            trainf_writer, traint_writer = csv.writer(csv_trainf), csv.writer(csv_traint)
            valf_writer, valt_writer = csv.writer(csv_valf), csv.writer(csv_valt)
            testf_writer, testt_writer = csv.writer(csv_testf), csv.writer(csv_testt)
            dropped_last_entry = False
            while parser.has_next() and count < size:
                if parser.current_file()["ancill/basis_regions"][parser.i][parser.j] != 0:
                    features, targets = parser.next()
                    if dropped_last_entry or targets[len(targets) - 1] != 0:
                        dropped_last_entry = False
                        count += 1
                        if (count % 10) == 0:
                            valf_writer.writerow(features)
                            valt_writer.writerow(targets)
                        elif (count % 25) == 0:
                            testf_writer.writerow(features)
                            testt_writer.writerow(targets)
                        else:
                            trainf_writer.writerow(features)
                            traint_writer.writerow(targets)
                        print("Entries found: " + str(count), end="\r")
                    else:
                        accepted_last_entry = True
                else:
                    parser.increment()

        print("Example entries parsed: " + str(count))
        print("Example features and target values written to csvs in ouput directory.")
    elif len(files) == 1:
        print("At least 2 valid HDF GFED files required but found 1.")
    else:
        print("No valid HDF GFED files found in that directory.")

if __name__ == "__main__":
    PARSER = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    PARSER.add_argument("directory", help="Directory with GFED4.1s_yyyy HDF files")
    PARSER.add_argument("--size", type=int, help="No. of entries to extract", default=1000)
    validate_and_parse(PARSER.parse_args().directory, PARSER.parse_args().size)
