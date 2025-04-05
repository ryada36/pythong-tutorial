__all__ = ["random_choice"]

from random import *

def random_choice(seq):
    if len(seq) == 0:
        raise ValueError("Cannot choose from an empty sequence")
    return choice(seq)

def __not_implemented():
    print("This function is not implemented yet")

print(random_choice([1, 2, 3, 4, 5]))
__not_implemented()
