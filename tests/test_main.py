"""Test cases for plugin output."""

import os
import unittest

from azul_smart_string_filter.lib import SmartStringFilter


class TestAIModel(unittest.TestCase):
    def test_ai_model_output(self):
        # Threshold for how accuarate the model predicts good strings.
        accuracy_threshold = 0.9
        # number of good strings found in the text file by human.
        number_of_good_strings = 674

        # Read input data from file
        BASE_FILE_DIR = os.path.join(os.path.dirname(__file__), "data")
        file_path = os.path.join(BASE_FILE_DIR, "strings_list.txt")
        string_list = []
        with open(file_path, "r") as file:
            for line in file:
                string = line.strip()
                string_list.append(string)

        GSF = SmartStringFilter()
        # Call AI model
        output_data = GSF.find_legible_strings(string_list)
        good_string_counter = 0
        for is_good in output_data:
            if is_good:
                good_string_counter += 1
        # calculate the accuracy of the model.
        accuracy = 1 - (abs(number_of_good_strings - good_string_counter) / number_of_good_strings)
        # assert accuracy is greater than or equal to threshold.
        self.assertGreaterEqual(accuracy, accuracy_threshold)
