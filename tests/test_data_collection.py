# tests/test_data_collection.py

import unittest
from src.data.data_collection import fetch_historical_token_data

class TestDataCollection(unittest.TestCase):
    def test_fetch_historical_token_data(self):
        data = fetch_historical_token_data()
        self.assertIsNotNone(data)
        self.assertFalse(data.empty)
        self.assertIn('token_name', data.columns)
        self.assertIn('price', data.columns)

if __name__ == '__main__':
    unittest.main()
