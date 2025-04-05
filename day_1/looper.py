def looper(range_limit:int):
    for i in range(range_limit):
        print(i)
    print("Loop finished")
looper(5)


def looper_with_reduced_range(range_limit: int) -> int:
    result = 0
    for i in range(0, range_limit):
        result += i
    print(f"The sum of even numbers up to {range_limit} is: {result}")

looper_with_reduced_range(10)