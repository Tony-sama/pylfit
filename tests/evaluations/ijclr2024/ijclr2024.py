#-----------------------
# @author: Tony Ribeiro
# @created: 2024/06/18
# @updated: 2024/06/18
#
# @desc: GULA benchmarks evaluation script for IJCLR 2024
# - Scalability
#
#-----------------------


import os

from pylfit.utils import eprint

# 0: Constants
#--------------

time_out = 1000
max_unknowns = 0.5
run_tests = 10
use_all_cores = False
debug = True

DEBUG_runs = True
GULA_scalability = True

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

    eprint("Starting all experiement of IJCLR 2024 paper")
    if DEBUG_runs:
        eprint("Debug runs")
        os.system(start_command+"python3 -u evaluations/ijclr2024/ijclr2024_bn_benchmarks.py gula 5 10 10 "+str(2)+" scalability random_transitions "+str(1)+" "+str(0)+" > tmp/debug_bn_benchmarks_scalability_gula.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/ijclr2024/ijclr2024_bn_benchmarks.py brute-force 5 10 10 "+str(2)+" scalability random_transitions "+str(1)+" "+str(0)+" > tmp/debug_bn_benchmarks_scalability_brute_force.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/ijclr2024/ijclr2024_bn_benchmarks.py gula 5 10 10 "+str(2)+" scalability random_transitions "+str(1)+" "+str(max_unknowns)+" > tmp/debug_bn_benchmarks_scalability_gula.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/ijclr2024/ijclr2024_bn_benchmarks.py brute-force 5 10 10 "+str(2)+" scalability random_transitions "+str(1)+" "+str(max_unknowns)+" > tmp/debug_bn_benchmarks_scalability_brute_force.csv"+redirect_error+end_command)
      
    if GULA_scalability:
        print("1) GULA scalability experiements")
        print()
        print("1.1) Boolean network Benchmarks: X% train")
        os.system(start_command+"python3 -u evaluations/ijclr2024/ijclr2024_bn_benchmarks.py gula 5 10 10 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" "+str(0)+" > tmp/bn_benchmarks_scalability_gula_0_to_10_complete.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/ijclr2024/ijclr2024_bn_benchmarks.py brute-force 5 10 10 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" "+str(0)+" > tmp/bn_benchmarks_scalability_brute_force_0_to_10_complete.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/ijclr2024/ijclr2024_bn_benchmarks.py gula 5 10 10 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" "+str(max_unknowns)+" > tmp/bn_benchmarks_scalability_gula_0_to_10_partial.csv"+redirect_error+end_command)
        os.system(start_command+"python3 -u evaluations/ijclr2024/ijclr2024_bn_benchmarks.py brute-force 5 10 10 "+str(run_tests)+" scalability random_transitions "+str(time_out)+" "+str(max_unknowns)+" > tmp/bn_benchmarks_scalability_brute_force_0_to_10_partial.csv"+redirect_error+end_command)