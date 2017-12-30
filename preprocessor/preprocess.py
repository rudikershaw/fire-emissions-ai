import sys
from pathlib import Path

# -------------------------------------------------
# Functions defined below.
# -------------------------------------------------

def print_help_text():
    print("\nThe processor.py utility is designed to take a single argument, the path")
    print("to a NASA EarthData Global Fire Emissions Database GFED4.1s_yyyy.hdf5 file.")
    print("Example - $ ./preprocess.py some/directory/GFED4.1s_2015.hdf5\n")

    print("If a valid file path is passed to the utility it should output individual")
    print("files for each month, that contain data in the format required to train the")
    print("emissions predictor.\n")

    print("By default the new files will be ouput to the same directory that contains")
    print("the script. Alternatively, you can provide a second argument with a path to")
    print("another directory for the output files to be placed in.\n")

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
        return valid_hdf_file(path_to_data)
    else:
        print_help_text()
        return False

# -------------------------------------------------
# Script starts here.
# -------------------------------------------------

if __name__ == "__main__":
    if valid_arguments(sys.argv):
        print("\nProcessing file...")
    else:
        sys.exit()
