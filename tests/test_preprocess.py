import h5py
import os.path
import tempfile
from unittest import TestCase
from fireemissionsai.preprocess import Validator

class TestValidator(TestCase):
    # Test the Validator.valid_arguments and valid_hdf_file functions.
    def test_valid_arguments(self):
        # Providing no arguments is not a valid.
        self.assertFalse(Validator.valid_arguments([]))
        # --help is not valid to start up the process, show help text.
        self.assertFalse(Validator.valid_arguments(["filename", "--help"]))
        # Files that are not hdf files are not valid.
        self.assertFalse(Validator.valid_arguments(["filename", "test"]))
        # It is not valid to provide a file that does not exist.
        self.assertFalse(Validator.valid_arguments(["filename", "file_doesnt_exist.hdf5"]))
        # It is valid to provide a hdf file that actually exists.
        self.assertTrue(Validator.valid_arguments(["filename", "tests/resources/empty.hdf5"]))

    # Test the Validator.valid_hdf_file and valid_leaf_groups functions.
    def test_valid_hdf_structure(self):
        tmp_file = os.path.join(tempfile.gettempdir(), "new.hdf5")
        hdf5_file = h5py.File(tmp_file, "w")
        # Providing empty hdf5 file is not valid.
        self.assertFalse(Validator.valid_hdf_structure(hdf5_file))
