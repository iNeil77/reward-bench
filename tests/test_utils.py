# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import unittest

from rewardbench import save_results_locally


class SaveDataTest(unittest.TestCase):
    def test_save_locally(self):
        model_name = "fake/fake_model"
        fake_results = {
            "model": model_name,
            "model_type": "random",
            "alpacaeval-easy": 0.12345,
            "math-prm": 0.54321,
        }
        path = save_results_locally(fake_results, model_name, "eval-set/")

        self.assertEqual(path, "./results/eval-set/fake/fake_model.json")
        with open(path, "r") as f:
            output = json.load(f)

        self.assertAlmostEqual(output["alpacaeval-easy"], 0.12345, places=5)
        self.assertAlmostEqual(output["math-prm"], 0.54321, places=5)
        # accounts for weird json float conversion
