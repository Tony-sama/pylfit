#-----------------------
# @author: Tony Ribeiro
# @created: 2022/10/17
# @updated: 2023/12/27
#
# @desc: PyLFIT unit test script
#
#-----------------------

EXAMPLE_FOLDER = ".\\examples\\"

import os
import subprocess

example_files = [
    "api_acedia_example.py",
    "api_boolean_network_example.py",
    "api_explanation_score_example.py",
    "api_gula_and_pride_example.py",
    "api_multithread_example.py",
    "api_pride_completness.py",
    "api_probalizer_example.py",
    "api_synchronizer_example.py",
    "api_weighted_prediction_and_explanation_example.py",
    "sequences_learning/api_pride_heuristics.py"
    ]

for f in example_files:

    print("*******************************************")
    print(f)
    print("*******************************************")
    result = subprocess.run(["python", EXAMPLE_FOLDER+f], capture_output=True, text=True)
    print(result.stdout)
