import h5py
import os.path
import tempfile
from unittest import TestCase
from fireemissionsai.preprocess import Validator

class TestValidator(TestCase):
    # Test the Validator.valid_arguments and valid_hdf_file functions.
    def test_valid_arguments(self):
        # It is not valid to provide a file that does not exist.
        self.assertFalse(Validator.valid_hdf_file("file_doesnt_exist.hdf5"))
        # It is valid to provide a hdf file that actually exists.
        self.assertTrue(Validator.valid_hdf_file("tests/resources/empty_2018.hdf5"))

    # Test the Validator.valid_hdf_file and valid_leaf_groups functions.
    def test_valid_hdf_structure(self):
        tmp_file = os.path.join(tempfile.gettempdir(), "new.hdf5")
        # Providing empty hdf5 file is not valid.
        self.assertFalse(Validator.valid_hdf_structure(tmp_file))
        # Providing GFED format hdf file is valid.
        self.assertTrue(Validator.valid_hdf_structure("tests/resources/min_2018.hdf5"))
