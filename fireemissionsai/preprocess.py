import io, sys, h5py, json, itertools, re
from pathlib import Path

# -------------------------------------------------
# Utility functions class defined below.
# -------------------------------------------------
class Validator:

    @staticmethod
    def print_help_text():
        help_text = """
            The processor.py utility is designed to take a single argument, the path
            to a NASA EarthData Global Fire Emissions Database GFED4.1s_yyyy.hdf5 file.
            Example - $ ./preprocess.py some/directory/GFED4.1s_2015.hdf5

            If a valid file path is passed to the utility it should output individual JSON
            files for each month, that contain data in the format required to train the
            emissions predictor.

            By default the new files will be output to the same directory that contains
            the script. Alternatively, you can provide a second argument with a path to
            another directory for the output files to be placed in.
        """
        print(re.sub(" +", " ", help_text))


    @staticmethod
    def valid_hdf_file(path_string):
        valid_extensions = "hdf","hdf4","hdf5","h4","h5", "he2", "he5"
        if path_string.split(".")[-1].lower() in valid_extensions:
            if Path(path_string).is_file():
                return True
            print("\n'" + path_string + "' is not a valid file.\n")
            return False
        print("\nThe input file must be an HDF file with a correct extension.\n")
        return False


    @staticmethod
    def valid_arguments(arguments):
        if len(arguments) in (2, 3) and arguments[1] != "--help":
            path_to_data = arguments[1]
            return Validator.valid_hdf_file(path_to_data)
        Validator.print_help_text()
        return False


    @staticmethod
    def valid_leaf_groups(group, month, hdf_file):
        groups_and_leaves = {
            "biosphere": ("BB", "NPP", "Rh"),
            "burned_area": ("burned_fraction",),
            "emissions": ("C", "DM")
        }
        valid = True
        for leaf in groups_and_leaves[group]:
            full_group = "{}/{:02d}/{}".format(group, month, leaf)
            if full_group not in hdf_file:
                valid = False
                print("Expected group '" + full_group + "' not in HDF file.")
        return valid


    @staticmethod
    def valid_hdf_structure(hdf_file):
        valid = True
        for group in "ancill/basis_regions", "lon", "lat":
            if group not in hdf_file:
                valid = False
                print("Expected group '" + group + "' not in HDF file.")
        for group in "biosphere", "burned_area", "emissions":
            for month in range(1,13):
                full_group = "{}/{:02d}".format(group, month)
                if full_group not in hdf_file:
                    valid = False
                    print("Expected group '" + full_group + "' not in HDF file.")
                else:
                    valid = valid and Validator.valid_leaf_groups(group, month, hdf_file)
        return valid


class GFEDtoJsonOutputWriter:

    def __init__(self, hdf_file):
        self.hdf = hdf_file
        self.regions = hdf_file["ancill/basis_regions"]
        self.latitude = hdf_file["lat"]
        self.longitude = hdf_file["lon"]
        self.matrix_shape = self.regions.shape


    def write(self, output_path="preprocess-output.json"):
        print("Writing to preprocess-output.json file. This might take a while...")
        with io.open(output_path, "w", encoding='utf-8') as output:
            output.write("[\n")
            first_entry = True
            x, y = self.matrix_shape[0], self.matrix_shape[1]
            for i, j in itertools.product(range(x), range(y)):
                if self.regions[i][j] != 0:
                    self.write_monthly_entry(i, j, output, first_entry)
                    first_entry = False
            output.write("\n]")


    def write_monthly_entry(self, i, j, output, first_entry):
        for month in range(1, 13):
            if not first_entry:
                output.write(",\n")

            entry = {
                "month" : str("{:02d}".format(month)),
                "latitude" : str(self.latitude[i][j]),
                "longitude" : str(self.longitude[i][j]),
                "region" : str(self.regions[i][j]),
                "BB" : str(self.hdf["biosphere/{:02d}/BB".format(month)][i][j]),
                "NPP" : str(self.hdf["biosphere/{:02d}/NPP".format(month)][i][j]),
                "Rh" : str(self.hdf["biosphere/{:02d}/Rh".format(month)][i][j]),
                "C" : str(self.hdf["emissions/{:02d}/C".format(month)][i][j]),
                "DM" : str(self.hdf["emissions/{:02d}/DM".format(month)][i][j]),
                "burned" : str(self.hdf["burned_area/{:02d}/burned_fraction".format(month)][i][j])
            }
            json.dump(entry, output)
            first_entry = False


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
    writer = GFEDtoJsonOutputWriter(hdf_file)
    writer.write()
