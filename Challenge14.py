from ast import literal_eval
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time

def get_input_data():
    with open('data/Number_Sequences.txt') as f:
        data = f.read()
    return data

def get_answer():
    start = 0
    end = 0
    number = 1
    for line in get_input_data().splitlines():
        l = literal_eval(line)
        l = np.array(sorted(l))
        try:
            min_idx, max_idx, signal, double_idx = get_signal2(l)
            start += l[min_idx]
            end += l[max_idx]
            signal = l[min_idx:max_idx + 1]
            (missing,) = set(range(signal.min(), signal.max())) - set(signal)
            print(f'No. {number}. start={l[min_idx]} end={l[max_idx]} missing={missing} double={l[double_idx+min_idx]}')
        except Exception as e:
            print(f'***** {sorted(l)} no signal found')
            print(e)
            continue
        finally:
            number += 1
    return start + end

def get_signal2(s_arr):
    s_diff = np.diff(s_arr, append=s_arr[-1])
    max_seg = max(find_segments_with_conditions(s_diff), key=lambda x: len(x[2]))
    return max_seg

# def find_segments_with_conditions(arr):
#     segments = []
#     n = len(arr)
#     for i in range(n):
#         for j in range(i, n):
#             segment = arr[i : j + 1]
#             count_zero = np.count_nonzero(segment == 0)
#             count_one = np.count_nonzero(segment == 1)
#             count_two = np.count_nonzero(segment == 2)
#             count_not = np.count_nonzero((segment < 0) | (segment > 2))
#             if count_zero == 1 and count_two == 1 and count_one >= 1 and count_not == 0:
#                 segments.append([i, j+1, segment, (segment == 0).argmax()])
#     return segments


def find_segments_with_conditions(arr):
    segments = []
    n = len(arr)
    arr = np.asarray(arr)
    for length in range(1, n + 1):
        windows = sliding_window_view(arr, window_shape=length)

        count_zero = (windows == 0).sum(axis=1)
        count_one = (windows == 1).sum(axis=1)
        count_two = (windows == 2).sum(axis=1)
        count_not = ((windows < 0) | (windows > 2)).sum(axis=1)

        mask = (count_zero == 1) & (count_two == 1) & (count_one >= 1) & (count_not == 0)
        valid_indices = np.where(mask)[0]

        for idx in valid_indices:
            i = idx
            j = idx + length
            segment = windows[idx]
            zero_pos = np.argmax(segment == 0)
            segments.append([i, j, segment.copy(), zero_pos])

    return segments


if __name__ == '__main__':
    start_time = time.perf_counter()
    print(get_answer())
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.6f} seconds")