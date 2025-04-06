import numpy as np

import numpy as np

# 1D Array
arr = np.array([10, 20, 30, 40, 50])

# 2D Array (Matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Array of zeros
zeros = np.zeros((2, 3))   # 2 rows, 3 columns

# Array of ones
ones = np.ones((3, 2))     # 3 rows, 2 columns

# Range of numbers
r = np.arange(0, 10, 2)    # 0, 2, 4, 6, 8

# Random numbers
rand = np.random.rand(3, 4)  # 3x4 matrix of random numbers between 0-1

# arry operations
# print(arr + 5) 
# print(arr * 2) 
# print(arr ** 2)

# # array slicing
# print(arr[0]) 
# print(arr[1:3])
# print(arr[:2]) 
# print(arr[-2:])

# functions
# print(np.sum(arr))
# print(np.mean(arr))
# print(np.max(arr))
# print(np.min(arr))
# print(np.std(arr))

# speed faceoff
import numpy as np
import time

# Create BIG data
size = 10_000_000
python_list = list(range(size))
numpy_array = np.arange(size)

# Timing pure Python list (adding 5 to each item)
start = time.time()
python_list = [x + 5 for x in python_list]
end = time.time()
print(f"Pure Python took: {end - start:.5f} seconds")

# Timing NumPy array (adding 5 to each item)
start = time.time()
numpy_array = numpy_array + 5
end = time.time()
print(f"NumPy took: {end - start:.5f} seconds")
