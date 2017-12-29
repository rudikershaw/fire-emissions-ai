import sys

args = len(sys.argv)
# The first arguement always contains the name of the file. Hence if args == 2.
if args == 2:
    print("Functionality currently unimplimented.")
else:
    print("The processor.py utility is designed to take a single argument, the path")
    print("to a NASA EarthData Global Fire Emissions Database GFED4.1s_yyyy.hdf5 file.")
    print("Example - $ ./preprocess.py some/directory/GFED4.1s_2015.hdf5")
