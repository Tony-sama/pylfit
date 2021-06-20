#-----------------------
# @author: Tony Ribeiro
# @created: 2020/07/13
# @updated: 2021/06/21
#
# @desc: GULA/Synchronizer benchmarks evaluation script for MLJ 2020
# - Scalability
# - Accuracy
# - Expalanation
#
#-----------------------


import os

from pylfit.utils import eprint

# 0: Constants
#--------------

time_out = 1000
run_tests = 10
use_all_cores = False
debug = True

DEBUG_runs = True
GULA_accuracy = True
GULA_explanation = True
GULA_scalability = True
Synchronizer_scalability = True

start_command = ""
redirect_error = "2> /dev/null"
end_command = ""

if use_all_cores:
    start_command = "nohup "
    end_command = " &"

if debug:
    redirect_error = ""

# 1: Main
#------------
if __name__ == '__main__':

    current_directory = os.getcwd()
    tmp_directory = os.path.join(current_directory, r'tmp')
    if not os.path.exists(tmp_directory):
       os.makedirs(tmp_directory)

    eprint("Starting all experiement of MLJ 2020 paper")
    if DEBUG_runs:
        eprint("Debug runs")
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py gula 0 10 10 "+str(run_tests)+" scalability random_transitions "+str(1)+" > tmp/debug_bn_benchmarks_scalability_gula.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py brute-force 0 10 10 "+str(run_tests)+" scalability random_transitions "+str(1)+" > tmp/debug_bn_benchmarks_scalability_brute_force.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py synchronizer 0 10 10 "+str(run_tests)+" scalability random_transitions "+str(1)+" > tmp/debug_bn_benchmarks_scalability_synchronizer.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py gula 0 5 5 "+str(run_tests)+" accuracy random_transitions "+str(1)+" > tmp/debug_bn_benchmarks_accuracy_gula.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py gula 0 5 5 "+str(run_tests)+" explanation random_transitions "+str(1)+" > tmp/debug_bn_benchmarks_explanation_gula.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py baseline 0 5 5 "+str(run_tests)+" accuracy random_transitions "+str(1)+" > tmp/debug_bn_benchmarks_accuracy_baseline.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py baseline 0 5 5 "+str(run_tests)+" explanation random_transitions "+str(1)+" > tmp/debug_bn_benchmarks_explanation_baseline.csv"+redirect_error+end_command)


    if GULA_accuracy:
        print()
        print("2) GULA accuracy experiements")
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py gula 0 9 9 "+str(run_tests)+" accuracy random_transitions "+str(time_out)+" > tmp/bn_benchmarks_accuracy_gula_0_to_9.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py baseline 0 9 9 "+str(run_tests)+" accuracy random_transitions "+str(time_out)+" > tmp/bn_benchmarks_accuracy_baseline_0_to_9.csv"+redirect_error+end_command)
        #os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py gula 0 13 13 "+str(run_tests)+" accuracy random_transitions "+str(time_out)+" > tmp/bn_benchmarks_accuracy_gula_0_to_13.csv"+redirect_error+end_command)
        #os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py baseline 0 13 13 "+str(run_tests)+" accuracy random_transitions "+str(time_out)+" > tmp/bn_benchmarks_accuracy_baseline_0_to_13.csv"+redirect_error+end_command)

    if GULA_explanation:
        print()
        print("2) GULA explanation experiements")

        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py gula 0 9 9 "+str(run_tests)+" explanation random_transitions "+str(time_out)+" > tmp/bn_benchmarks_explanation_gula_0_to_9.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py baseline 0 9 9 "+str(run_tests)+" explanation random_transitions "+str(time_out)+" > tmp/bn_benchmarks_explanation_baseline_0_to_9.csv"+redirect_error+end_command)

    if GULA_scalability:
        print("1) GULA scalability experiements")
        print()
        print("1.1) Boolean network Benchmarks: X% train")
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py gula 0 13 13 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" > tmp/bn_benchmarks_scalability_gula_0_to_13.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py brute-force 0 13 13 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" > tmp/bn_benchmarks_scalability_brute_force_0_to_13.csv"+redirect_error+end_command)

        #os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py gula 14 16 16 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" > tmp/bn_benchmarks_scalability_gula_14_to_16.csv"+redirect_error+end_command)
        #os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py gula 17 18 16 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" > tmp/bn_benchmarks_scalability_gula_17_to_18.csv"+redirect_error+end_command)

        print("1.2) Random programs: evolving number of features")
        # TODO

        print("1.3) Random programs: evolving features domains")
        # TODO

    if Synchronizer_scalability:
        print()
        print("3) Synchronizer experiements")
        print()
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py synchronizer 0 7 7 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" > tmp/bn_benchmarks_scalability_synchronizer_0_to_7.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py synchronizer 8 10 10 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" > tmp/bn_benchmarks_scalability_synchronizer_8_to_10.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py synchronizer 11 12 12 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" > tmp/bn_benchmarks_scalability_synchronizer_11_to_12.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/mlj2020/mlj2020_bn_benchmarks.py synchronizer 13 13 13 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" > tmp/bn_benchmarks_scalability_synchronizer_13.csv"+redirect_error+end_command)
