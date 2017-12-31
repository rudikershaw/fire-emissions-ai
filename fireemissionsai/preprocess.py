import sys
import h5py
from pathlib import Path

# -------------------------------------------------
# Utility functions class defined below.
# -------------------------------------------------
class Validator:
    def print_help_text():
        print("The processor.py utility is designed to take a single argument, the path")
        print("to a NASA EarthData Global Fire Emissions Database GFED4.1s_yyyy.hdf5 file.")
        print("Example - $ ./preprocess.py some/directory/GFED4.1s_2015.hdf5\n")

        print("If a valid file path is passed to the utility it should output individual JSO")
        print("files for each month, that contain data in the format required to train the")
        print("emissions predictor.\n")

        print("By default the new files will be ouput to the same directory that contains")
        print("the script. Alternatively, you can provide a second argument with a path to")
        print("another directory for the output files to be placed in.")


    def valid_hdf_file(path_string):
        valid_extensions = ("hdf","hdf4","hdf5","h4","h5", "he2", "he5")
        if path_string.split(".")[-1] in valid_extensions:
            if Path(path_string).is_file():
                return True
            else:
                print("\n'" + path_string + "' is not a valid file.\n")
                return False
        else:
            print("\nThe input file must be an HDF file with a correct extension.\n")
            return False


    def valid_arguments(arguements):
        args = len(arguements)
        if (args == 2 or args == 3) and arguements[1] != "--help":
            path_to_data = arguements[1]
            return Validator.valid_hdf_file(path_to_data)
        else:
            Validator.print_help_text()
            return False


    def valid_leaf_groups(group, month, hdf_file):
        groups_and_leaves = {
            "biosphere": ("BB", "NPP", "Rh"),
            "burned_area": ("burned_fraction",),
            "emissions": ("C", "DM")
        }
        valid = True
        for leaf in groups_and_leaves[group]:
            full_group = group + "/" + ("%02d" % month) + "/" + leaf
            if full_group not in hdf_file:
                valid = False
                print("Expected group '" + full_group + "' not in HDF file.")
        return valid


    def valid_hdf_structure(hdf_file):
        valid = True
        for group in ("ancill/basis_regions", "lon", "lat"):
            if group not in hdf_file:
                valid = False
                print("Expected group '" + group + "' not in HDF file.")
        for group in ("biosphere", "burned_area", "emissions"):
            for month in range(1,13):
                full_group = group + "/" + ("%02d" % month)
                if full_group not in hdf_file:
                    valid = False
                    print("Expected group '" + full_group + "' not in HDF file.")
                else:
                    valid = valid and Validator.valid_leaf_groups(group, month, hdf_file)
        return valid

# -------------------------------------------------
# Script starts here.
# -------------------------------------------------

if __name__ == "__main__":
    if not Validator.valid_arguments(sys.argv):
        sys.exit()

    filename = sys.argv[1]
    print("Processing - " + filename)
    hdf_file = h5py.File(filename, 'r')

    if not Validator.valid_hdf_structure(hdf_file):
        sys.exit()

    print("Basic structure of hdf file confirmed to conform to GFED4 format.")