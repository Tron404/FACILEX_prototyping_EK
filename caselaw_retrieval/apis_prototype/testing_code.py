import json
import unittest

from facilex_platform_code_case_recommendation import semantic_search
from types import NoneType
from typing import TextIO

# made for custom output, to also show which test file was used
class result_text_custom_result(unittest.TextTestResult):
    def __init__(self, stream: TextIO, descriptions: bool, verbosity: int, test_name="") -> None:
        super().__init__(stream, descriptions, verbosity)
        self.test_name = test_name

    def startTest(self, test):
        super(unittest.TextTestResult, self).startTest(test)
        if self.showAll:
            self.stream.write(test.input_file)
            self.stream.write(" ... ")
            self.stream.write(self.getDescription(test))
            self.stream.write(" ... ")
            self.stream.flush()
            self._newline = False

# child class for custom text runner
class result_text_custom_runner(unittest.TextTestRunner):
    resultclass = result_text_custom_result

# main test case class
class input_test(unittest.TestCase):
    def __init__(self, methodName: str = "runTest", input_file="", testing_path="testing/") -> None:
        super().__init__(methodName)
        self.input_file = input_file
        self.testing_path = testing_path

    ############################
    # Auxiliary test functions #
    ############################

    def _test_input(self, result, expected_attribute_from_querry):
        if type(expected_attribute_from_querry) == list: # query has multiple CELEXs
            self._test_input_multiplecelex(result, expected_attribute_from_querry)
        elif type(expected_attribute_from_querry) == str: # query has one CELEX only
            self._test_input_1celex(result, expected_attribute_from_querry)
        elif type(expected_attribute_from_querry) == NoneType: # contains None or some CELEX if query has None - always ok
            self._test_input_nocelex(result)
        
    # contains same CELEX, if query has CELEX - check if equal
    def _test_input_1celex(self, result, expected_attribute_from_querry):
        self.assertGreater(len(result), 0, f"There are no results from the system! - {self.input_file}")
        self.assertLess(len(result), 6, f"There are more than 5 results from the system! - {self.input_file}")
        for result_case in result:
            self.assertEqual(expected_attribute_from_querry, result_case["euProvisions"], f"CELEX do not match - {self.input_file}")

    def _test_input_nocelex(self, result):
        self.assertEqual(len(result), 5, f"There are not 5 cases here! - {self.input_file}")

    def _test_input_multiplecelex(self, result, expected_attribute_from_querry):
        raise NotImplementedError()

    ############################
    #    Main test function    #
    ############################

    def _get_test_attribute(self, path):
        file = open(path, "r")
        attribute = json.load(file)["euProvisions"]
        file.close()
        return attribute

    def test_input_query(self):
        result, fail_safe = semantic_search(self.testing_path + self.input_file, return_fail_safe=True)
        with open(f"testing_data/expected/expected_{self.input_file}", "w") as output_file:
            output_file.write(result)

        expected_attribute_from_query = None
        if not fail_safe:
            expected_attribute_from_query = self._get_test_attribute(self.testing_path+self.input_file)

        result = json.loads(result)
        self._test_input(result, expected_attribute_from_query)