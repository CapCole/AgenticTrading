"""Price cache naming. Run: python -m unittest test_data"""
import subprocess
import sys
import unittest
from pathlib import Path

from data import cache_key

HERE = Path(__file__).parent


class CacheKeyTest(unittest.TestCase):
    def test_same_request_gives_the_same_key_in_a_new_process(self):
        # Python's built-in hash() is randomised per process, so a key built on it
        # never matched a later run and every backtest re-downloaded its prices.
        code = "from data import cache_key; print(cache_key(['B.JO', 'A.JO'], '2024-01-01', None))"
        other = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True,
                               text=True, check=True).stdout.strip()
        self.assertEqual(other, cache_key(["A.JO", "B.JO"], "2024-01-01", None))

    def test_different_requests_get_different_keys(self):
        self.assertNotEqual(cache_key(["A.JO"], "2024-01-01", None),
                            cache_key(["A.JO"], "2023-01-01", None))


if __name__ == "__main__":
    unittest.main()
