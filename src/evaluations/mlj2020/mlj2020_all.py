#-----------------------
# @author: Tony Ribeiro
# @created: 2020/07/13
# @updated: 2020/07/13
#
# @desc: GULA/Synchronizer benchmarks evaluation script for MLJ 2020
#
#-----------------------

import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
sys.path.insert(0, 'src/evaluations/mlj2020')

import os

from utils import eprint

# 0: Constants
#--------------

run_tests = 10
use_15_cores = False

start_command = ""
end_command = ""

if use_15_cores:
    start_command = "nohup "
    end_command = " &"

# 1: Main
#------------
if __name__ == '__main__':


    eprint("Starting all experiement of MLJ 2020 paper")

    print("1) GULA scalability experiements")
    print()
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py GULA 0 13 13 "+str(run_tests)+" > tmp/output_run_time_gula_0_to_13.txt 2> /dev/null"+end_command)
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py GULA 15 15 15 "+str(run_tests)+" > tmp/output_run_time_gula_15.txt 2> /dev/null"+end_command)
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py GULA 18 18 15 "+str(run_tests)+" > tmp/output_run_time_gula_18.txt 2> /dev/null"+end_command)
    print()
    print("2) GULA accuracy experiements")

    print()
    print("2.1) Boolean network Benchmarks: X% train 100-X% test, all from train init state")
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py GULA 0 9 9 "+str(run_tests)+" accuracy > tmp/output_accuracy_0_to_9.txt 2> /dev/null"+end_command) #23 15")
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py GULA 10 10 10 "+str(run_tests)+" accuracy > tmp/output_accuracy_10.txt 2> /dev/null"+end_command)
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py GULA 12 12 12 "+str(run_tests)+" accuracy > tmp/output_accuracy_12.txt 2> /dev/null"+end_command)
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py GULA 13 13 13 "+str(run_tests)+" accuracy > tmp/output_accuracy_13.txt 2> /dev/null"+end_command)

    print()
    print("2.2) Boolean network Benchmarks: 80% train 20% test, random X% from train init state")
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py GULA 0 9 9 "+str(run_tests)+" accuracy random > tmp/output_accuracy_random_0_to_9.txt 2> /dev/null"+end_command)
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py GULA 10 10 10 "+str(run_tests)+" accuracy random > tmp/output_accuracy_random_10.txt 2> /dev/null"+end_command)
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py GULA 12 12 12 "+str(run_tests)+" accuracy random > tmp/output_accuracy_random_12.txt 2> /dev/null"+end_command)
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py GULA 13 13 13 "+str(run_tests)+" accuracy random > tmp/output_accuracy_random_13.txt 2> /dev/null"+end_command)
    print()

    print()
    print("3) Synchronizer experiements")
    print()
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py Synchronizer 0 7 7 "+str(run_tests)+" > tmp/output_run_time_synchronizer_0_to_7.txt 2> /dev/null"+end_command)
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py Synchronizer 9 10 10 "+str(run_tests)+" > tmp/output_run_time_synchronizer_9_to_10.txt 2> /dev/null"+end_command)
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py Synchronizer 12 12 12 "+str(run_tests)+" > tmp/output_run_time_synchronizer_12.txt 2> /dev/null"+end_command)
    os.system(start_command+"python3 -u src/evaluations/mlj2020/mlj2020_bn_benchmarks.py Synchronizer 13 13 13 "+str(run_tests)+" > tmp/output_run_time_synchronizer_13.txt 2> /dev/null"+end_command)
