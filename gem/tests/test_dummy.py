import unittest

class DummyTests(unittest.TestCase):
    def test_two_plus_one_is_three(self):
        self.assertEqual(3, 2+1)
