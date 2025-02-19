import numpy as np
import time
import matplotlib.pyplot as plt

# Flatten list
# Using for loop
def flatten_list_for(input_list):
    result = []
    for sublist in input_list:
        for item in sublist:
            result.append(item)
    return result
    
# Using python list comprehension
def flatten_list_list(nested_list: list):
    return [item for sublist in nested_list for item in sublist]
    
# Using extend method
def flatten_list_extend(nested_list: list):
    result = []
    for sublist in nested_list:
        result.extend(sublist)
    return result

# for autograde
def flatten_list(nested_list: list):
    return [item for sublist in nested_list for item in sublist]

# Count characters
# Using dictionary comprehension
def char_count_dict(input_str: str):
    return {char: input_str.count(char) for char in set(input_str)}

# Using for loop with builtin methods
def char_count_for(input_str: str):
    char_count = {}
    for char in input_str:
        char_count[char] = char_count.get(char, 0) + 1
    return char_count

# Using naive for loop
def char_count_naive_for(input_str: str):
    char_count = {}
    for char in input_str:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    return char_count

# for autograde
def char_count(input_str: str):
    return char_count_dict(input_str)
    
    
# Experiment
def naive_generate_large_nested_list(size):
    return [[i] for i in range(size)]

def generate_large_nested_list(size):
    nested_list = []   
    sublist_size = 1000
    for i in range(0, size, sublist_size):
        sublist = list(range(i, min(i + sublist_size, size)))
        nested_list.append(sublist)
    return nested_list



if __name__ == "__main__":
    sizes = [10**3, 10**4, 10**5, 10**6, 10**7]
    # 测试 flatten_list
    # timing_results_list = {}
    # timing_results_extend = {}
    # timing_results_for = {}
    
    # for size in sizes:
    #     nested_list = generate_large_nested_list(size)
    #     start_time = time.time()
    #     result = flatten_list_list(nested_list)
    #     time_taken = time.time() - start_time
    #     timing_results_list[size] = time_taken
    #     start_time = time.time()
    #     result = flatten_list_extend(nested_list)
    #     time_taken = time.time() - start_time
    #     timing_results_extend[size] = time_taken
    #     start_time = time.time()
    #     result = flatten_list_for(nested_list)
    #     time_taken = time.time() - start_time
    #     timing_results_for[size] = time_taken

    # # Report
    # print("\nTiming Report:")
    # print("list_comprehensions:")
    # for size, time_taken in timing_results_list.items():
    #     print(f"Input size: {size}, Time taken: {time_taken:.6f} seconds")
    # print("list_extend:")
    # for size, time_taken in timing_results_extend.items():
    #     print(f"Input size: {size}, Time taken: {time_taken:.6f} seconds")
    # print("for_loop:")
    # for size, time_taken in timing_results_for.items():
    #     print(f"Input size: {size}, Time taken: {time_taken:.6f} seconds")
   
    # # 将三种算法的时间复杂度进行对比,三条曲线画在一张图上
    # # plt.plot(list(timing_results_list.keys()), list(timing_results_list.values()), label='list_comprehensions')
    # # plt.plot(list(timing_results_extend.keys()), list(timing_results_extend.values()), label='list_extend')
    # # plt.plot(list(timing_results_for.keys()), list(timing_results_for.values()), label='for_loop')
    # # plot log
    # plt.plot(range(3,8), list(timing_results_list.values()), label='list_comprehensions')
    # plt.plot(range(3,8), list(timing_results_extend.values()), label='list_extend')
    # plt.plot(range(3,8), list(timing_results_for.values()), label='for_loop')
    # plt.xlabel('Log size')
    # plt.ylabel('Time taken (seconds)')
    # plt.title('Time taken to flatten large nested lists')
    # plt.legend()
    # plt.show()
    
    # 测试 char_count
    # load txt data
    with open('text.txt', 'r') as file:
        data = file.read()
    
    # get [10**3, 10**4, 10**5, 10**6, 10**7] size of slices from data
    slices = [data[:i] for i in sizes]
    timing_results_dict = {}
    timing_results_for = {}
    timing_results_naive_for = {}
    
    for slice , size in zip(slices,sizes):
        start_time = time.time()
        result = char_count_dict(slice)
        time_taken = time.time() - start_time
        timing_results_dict[size] = time_taken
        start_time = time.time()
        result = char_count_for(slice)
        time_taken = time.time() - start_time
        timing_results_for[size] = time_taken
        start_time = time.time()
        result = char_count_naive_for(slice)
        time_taken = time.time() - start_time
        timing_results_naive_for[size] = time_taken
        print(result)
    # Report
    print("\nTiming Report:")
    print("dict_comprehensions:")
    for size, time_taken in timing_results_dict.items():
        print(f"Input size: {size}, Time taken: {time_taken:.6f} seconds")
    print("for_loop:")
    for size, time_taken in timing_results_for.items():
        print(f"Input size: {size}, Time taken: {time_taken:.6f} seconds")
    print("naive_for_loop:")
    for size, time_taken in timing_results_naive_for.items():
        print(f"Input size: {size}, Time taken: {time_taken:.6f} seconds")
        
    # log plot
    plt.plot(list(timing_results_dict.keys()), list(timing_results_dict.values()), label='dict_comprehensions')
    plt.plot(list(timing_results_for.keys()), list(timing_results_for.values()), label='for_loop')
    plt.plot(list(timing_results_naive_for.keys()), list(timing_results_naive_for.values()), label='naive_for_loop')
    # plt.plot(range(3,8), list(timing_results_dict.values()), label='dict_comprehensions')
    # plt.plot(range(3,8), list(timing_results_for.values()), label='for_loop')
    # plt.plot(range(3,8), list(timing_results_naive_for.values()), label='naive_for_loop')
    plt.xlabel('Input size')
    plt.ylabel('Time taken (seconds)')
    plt.title('Time taken to count characters in large strings')
    plt.legend()
    plt.show()
    