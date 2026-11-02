"""Small unit tests for the parsing helpers. Run with: python -m unittest discover -s tests"""
import unittest
from recommender import _parse_json_list, _names, _extract_director


class ParsingTests(unittest.TestCase):
    def test_parse_json_list(self):
        self.assertEqual(_parse_json_list('[{"name":"Drama"}]'), [{"name": "Drama"}])

    def test_parse_invalid(self):
        self.assertEqual(_parse_json_list("not-json"), [])

    def test_names(self):
        self.assertEqual(_names('[{"name":"Action"},{"name":"Drama"}]'), ["Action", "Drama"])

    def test_director(self):
        crew = '[{"job":"Producer","name":"A"},{"job":"Director","name":"B"}]'
        self.assertEqual(_extract_director(crew), "B")


if __name__ == "__main__":
    unittest.main()
