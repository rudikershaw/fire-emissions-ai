import h5py
import os.path
import tempfile
from unittest import TestCase
from fireemissionsai.preprocess import Validator, GFEDDataParser

class TestValidator(TestCase):
    """Test the preprocess.Validator."""

    def test_valid_arguments(self):
        """Test the Validator.valid_arguments and valid_hdf_file functions."""
        # It is not valid to provide a file that does not exist.
        self.assertFalse(Validator.valid_hdf_file("file_doesnt_exist.hdf5"))
        # It is valid to provide a hdf file that actually exists.
        self.assertTrue(Validator.valid_hdf_file("tests/resources/empty_2018.hdf5"))

    def test_valid_hdf_structure(self):
        """Test the Validator.valid_hdf_file and valid_leaf_groups functions."""
        tmp_file = os.path.join(tempfile.gettempdir(), "new.hdf5")
        # Providing empty hdf5 file is not valid.
        self.assertFalse(Validator.valid_hdf_structure(tmp_file))
        # Providing GFED format hdf file is valid.
        self.assertTrue(Validator.valid_hdf_structure("tests/resources/min_2018.hdf5"))


class TestParser(TestCase):
    """Test the preprocess.GFEDDataParser."""

    def test_incremement_and_has_nexts(self):
        """Test the basic construction, incremeneting, and has_next functions."""
        parser = GFEDDataParser([h5py.File('tests/resources/min_2018.hdf5', 'r')])
        self.assertEqual(parser.month, 1)
        self.assertEqual(parser.i, 0)
        self.assertTrue(parser.has_next())
        # Increment through all 12 months.
        for i in range(1,13):
            self.assertTrue(parser.has_next())
            self.assertEqual(parser.month, i)
            if i == 12:
                self.assertFalse(parser.has_next_month())
            else:
                self.assertTrue(parser.has_next_month())
            parser.increment()
        self.assertEqual(parser.month, 1)
        self.assertEqual(parser.i, 1)
        # Increment through all other i and their months to a new j
        for i in range(1, (9 * 12) + 1):
            self.assertTrue(parser.has_next())
            parser.increment()
        self.assertEqual(parser.j, 1)
        self.assertTrue(parser.has_next())
        self.assertFalse(parser.has_next_file())
        # Increment to the end of the file.
        for i in range(1, (10 * 12 * 9)):
            self.assertTrue(parser.has_next())
            parser.increment()
        self.assertFalse(parser.has_next_coordinate())
        self.assertFalse(parser.has_next())
