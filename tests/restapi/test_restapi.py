import unittest

from fastapi.testclient import TestClient

from azul_smart_string_filter.restapi.filter_strings import SearchResult, app

client = TestClient(app)


class TestRestAPI(unittest.TestCase):
    def test_endpoint(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_string_filter_good(self):
        file_format = "executable/windows/"
        list_of_strings = [
            SearchResult(string="string1", offset=1),
            SearchResult(string="string2", offset=2),
            SearchResult(string="dsfdf asfdsf", offset=3),
            SearchResult(string="string3", offset=4),
        ]
        # this is how metastore converts searchresult.strings to send to the string filter
        converted_results = [{"string": sr.string, "offset": sr.offset} for sr in list_of_strings]
        response = client.post("/v0/strings/?file_format=" + file_format, json=converted_results)
        self.assertEqual(response.status_code, 200)
        expected_response = [
            {"string": "string1", "offset": 1},
            {"string": "string2", "offset": 2},
            {"string": "string3", "offset": 4},
        ]

        self.assertEqual(response.json(), expected_response)

    def test_string_filter_bad(self):
        file_format = "executable/windows/"
        list_of_strings = ["string1", 1, "dsfdf asfdsf", "string3"]
        response = client.post("/v0/strings/?file_format=" + file_format, json=list_of_strings)
        self.assertEqual(response.status_code, 422)
