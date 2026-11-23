"""
Compare the performance of Python loop implementations against NumPy
vectorisation.

This module provides benchmarking utilities that measure the execution timing
across multiple runs, compare equivalent implementations, and calculate the
difference in performance, such as speed-up.

To ensure fair comparison, Benchmarking should use the same inputs for both the
Python loop and NumPy vectorisation implementations and verify that both
produce equivalent results before runtime comparison.

Follows the benchmarking philosophies on:
 https://superfastpython.com/python-benchmarking-best-practices/

"""
import gc
import math
from time import perf_counter, time

import numpy as np

import random
import statistics

import numcompute_stream

from numcompute_stream import StreamStats, StreamQuantileP2



""" Helper functions for comparing benchmark results"""
# Looped quantile
def quantile_loop(data, q):
    """
    Calculates a quantile via python loop with linear interpolation

    Parameters

     data: numerical data
     q: quantile to be calculated

    Returns
     A float value for the given quantile

     Raises
    ValueError: If data is empty or q is less than 0 or more than one

    Time Complexity:
    O(n log n)

    Space Complexity:
    O(n)
    """
    if data is None:
        raise ValueError("Empty data!")

    elif q < 0 or q > 1:
        raise ValueError("q must be between 0 and 1!")
    sorted_data = np.sort(data)

    n = len(sorted_data)

    # Fraction index
    pos = q * (n - 1)

    # lower and upper indices
    lower = math.floor(pos)
    upper = math.ceil(pos)

    fraction = pos - lower

    if lower == upper:
        return sorted_data[lower]

    # performing linear interpolation to get final quantile
    return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])

def min_loop(data):
    """
    Calculates a minimum
     Parameters

     data: numerical data to be calculated on

    Returns
     A minimum value

    Raises
    ValueError: If data is empty

    Time Complexity:
    O(n)

    Space Complexity:
    O(n)
    """
    if data is None:
        raise ValueError("Empty data!")

    minimum = data[0]

    for n in data:
        if n < minimum:
            minimum = n

    return minimum


def max_loop(data):
    """
    Calculates a maximum
     Parameters

     data: numerical data to be calculated on

    Returns
     A maximum value

    Raises
    ValueError: If data is empty

    Time Complexity:
    O(n)

    Space Complexity:
    O(n)
    """
    if data is None:
        raise ValueError("Empty data!")

    max = data[0]

    for n in data:
        if n > max:
            max = n

    return max

def stan_dev_loop(data):
    """
    Calculates a maximum
    Parameters
     data: numerical data to be calculated on

    Returns
     A Standard Deviation value

    Raises
        ValueError: If data is empty

    Time Complexity:
    O(n)

    Space Complexity:
    O(n)
    """
    n = len(data)

    if data is None:
        raise ValueError("Empty data!")

    total = 0

    # Calculating mean
    for num in data:
        total = total + num

    mean = total / n

    # Calculating squared sum deviations from the mean
    sum_square_diff = 0
    for num in data:
        diff = num - mean
        sum_square_diff = sum_square_diff + (diff * diff)


    # Variance and then standard deviation
    variance = sum_square_diff / n

    return np.sqrt(variance)

"""Benchmarking"""

def benchit(func, *args, repeat=5, mode='standard'):
    """
        Ranks numerical data from 1 to n datapoints.
        Performs a benchmark on a given numcompute function.
        Compares runtime to equivalent Python loops.

        Parameters:
            func: input function to benchmark
            *args:
                optional arguments for a function if it requires them.

            function:
                The target function to be benchmarked

            repeat:
                How many executions are timed by the benchmark.
                Defaults to 5.

            mode:
                The type of benchmarking to occur:
                - demo: Demonstrates benchmarking utility by comparing the execution time, performance
                    and performance differences between a Numpy implementation of
                    numcompute functions vs a Python loop. 
                    For the most valuable insights, it benchmarks the most performance intensive numcompute
                    functions.
                - standard: standard benchmarking mode that displays execution time

        Returns:
            A printed statement describing execution timing and/or performance
            comparisons for the function and optionally compared to a numpy loop.

        Raises:
            ValueError:
                If a mode for benchmarking that does not exist
                is inputted.

                If in 'demo' mode functions that are being compared do not output the same error.

            TimeoutError:
                Function execution times out at a system level.

        Time Complexity:
           Same as algorithm being benchmarked. 

        Space Complexity:
            Same as algorithm being benchmarked
    """
    gc.disable() # Eliminates a potential source of benchmarking variation
                 # by disabling Python garbage collector.
 
    vectorized_benches = np.zeros(repeat)  # Initialise array to store repeated benchmark timings.

    if mode not in MODES: # Validating inputted mode
        raise ValueError(
        f"Unknown testing method '{mode}'. Please select from {MODES}."
        )

    match mode:
        case 'demo':
            if func not in DEMO_FUNCS: # Validating inputted function from tuple of demo functions
                raise ValueError(
                f"Unknown demo function '{mode}'. Please select from {DEMO_FUNCS}."
                )

            # Creates array to store Python loop implementation benches.
            pyloop_benches = np.copy(vectorized_benches) 

            match func:

                case 'quantile':
                    for i in range(repeat):
                        # Refresh inputs
                        test_arr = [random.randint(1, 50) for _ in range(100)]

                        ## VECTORISED
                        start_time = perf_counter()  # Start time, uses a high precision, non-adjustable clock for benchmarking.

                        quantile_stream = StreamQuantileP2(0.3)
                        quantile_stream.update(test_arr)
                        # Calling the function

                        end_time = perf_counter()  # end time

                        vectorized_benches[i] = round((end_time - start_time) * 1000,
                                                      6)  # Add result to the array and round to 6 decimal places

                        ## Python Loop
                        start_time = perf_counter()  # Start time, uses a high precision, non-adjustable clock for benchmarking.

                        quantile_loop(test_arr,0.3)  # Calling the function

                        end_time = perf_counter()  # end time

                        pyloop_benches[i] = round((end_time - start_time) * 1000,
                                                  6)  # Add result to the array and round to 6 decimal places

                case 'maximum':
                    for i in range(repeat):
                        # Refresh inputs
                        test_arr = [random.randint(1, 50) for _ in range(100)]

                        ## VECTORISED
                        start_time = perf_counter()  # Start time, uses a high precision, non-adjustable clock for benchmarking.

                        stream_max = StreamStats().max_stream(test_arr)# Calling the function

                        end_time = perf_counter()  # end time

                        vectorized_benches[i] = round((end_time - start_time) * 1000,
                                                      6)  # Add result to the array and round to 6 decimal places

                        ## Python Loop
                        start_time = perf_counter()  # Start time, uses a high precision, non-adjustable clock for benchmarking.

                        max_loop(test_arr)  # Calling the function

                        end_time = perf_counter()  # end time

                        pyloop_benches[i] = round((end_time - start_time) * 1000,
                                                  6)  # Add result to the array and round to 6 decimal places

                case 'minimum':


                    for i in range(repeat): 
                        #Refresh inputs
                        test_arr = [random.randint(1, 50) for _ in range(100)]

                        ## VECTORISED
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        stream_min = StreamStats().min_stream(test_arr) # Calling the function

                        end_time = perf_counter()   # end time

                        vectorized_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places


                        ## Python Loop
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        min_loop(test_arr) # Calling the function

                        end_time = perf_counter()   # end time

                        pyloop_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places  

                case 'stan_dev':

                    for i in range(repeat): 
                        #Refresh inputs
                        test_arr = [random.randint(1, 50) for _ in range(100)]

                        ## VECTORISED
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        stats = StreamStats().update_chunks(test_arr)
                        stream_stan_dev = stats.stan_dev()

                        end_time = perf_counter()   # end time

                        vectorized_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places


                        ## Python Loop
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        stan_dev_loop(test_arr) # Calling the function

                        end_time = perf_counter()   # end time

                        pyloop_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places  
                    
                case 'imputer':

                    for r in range(repeat): 
                        #Refresh inputs
                        test_arr = np.full((5,5), np.nan)
                        test_pyloop = test_arr

                        ## VECTORISED
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        numcompute_stream.SimpleImputer(fill_value=-1).partial_fit(test_arr).transform(test_arr)# Calling the function

                        end_time = perf_counter()   # end time

                        vectorized_benches[r] = round((end_time - start_time)* 1000, 6)  # Add result to the array and round to 6 decimal places


                        ## Python Loop
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        for i in range(len(test_pyloop)):
                            for j in range(len(test_pyloop[i])):
                                test_pyloop[i][j] = -1   # Performing transform with a loop
                        end_time = perf_counter()   # end time

                        pyloop_benches[i] = round((end_time - start_time)* 1000, 6)   # Add result to the array and round to 6 decimal places  

            # Performance table
            print(f'Table of performance (in ms):\n')
            print(f'Pyloop implementation:\n{pyloop_benches}\nVectorization implementation:\n{vectorized_benches}')

            # Print results of comparisons
            print(f'\nAverage python loop implementation execution time: {np.average(pyloop_benches):.6f} ms') # Mean execution time for python loops
            print(f'Average vectorization execution time: {np.average(vectorized_benches):.6f} ms') # Mean execution time

            pyloop_avg_time = np.average(pyloop_benches)
            vectorized_avg_time = np.average(vectorized_benches)

            # Calculates average performance difference from loop to vectorization
            perf_percentage = (vectorized_avg_time - pyloop_avg_time) / pyloop_avg_time * 100

            if(perf_percentage > 0):
                 print(f'\nVectorization performance impact is {perf_percentage:.4f}% slow down')
            else:
                 print(f'\nVectorization performance impact is {abs(perf_percentage):.4f}% speed up')

        case 'standard':

            for i in range(repeat):  

                start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                func(*args)                 # Calling the function

                end_time = perf_counter()   # end time

                vectorized_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places
            
            print(f'\nMilisecond execution results for {repeat} repeats (in ms):\n', vectorized_benches) 

            print(f'\nAverage execution time: {np.average(vectorized_benches):.6f} ms') # Mean execution time

MODES = ("demo","standard")
DEMO_FUNCS = ("quantile", "maximum", "minimum", "stan_dev", "imputer")
for func in DEMO_FUNCS:
    benchit(func, mode='demo', repeat=10)