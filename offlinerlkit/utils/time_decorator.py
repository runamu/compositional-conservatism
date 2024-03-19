from time import time

# time decortaor: check how much time does func take

accumulated_times = {}

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"\n{func.__name__} func ran in {round(end_time - start_time, 2)} seconds \n")
        return result
    return wrapper


def print_accumulated_times():

    for func_name, acc_time in accumulated_times.items():
        print(f"Total accumulated time for {func_name}: {round(acc_time, 2)} seconds")


def timer_decorator_cumulative(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()

        elapsed_time = end_time - start_time

        if func.__name__ in accumulated_times:
            accumulated_times[func.__name__] += elapsed_time
        else:
            accumulated_times[func.__name__] = elapsed_time

        return result
    return wrapper
