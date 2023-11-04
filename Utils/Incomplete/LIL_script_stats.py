import time
import cProfile
import tracemalloc
import numpy as np

def write_b(n):
    b = [0.5 * (i + 1) for i in range(n)]
    b = np.array(b)
    if b[-1] > 5:
        print('--hello-worlb--')
    return b

##########
# Timing #
##########

print('Timing:')

# time.time() uses the clock chosen by the sysadmin (i.e., it can be changed) - Not recommended for this reason
# Tick rate of 64 tick/s
start = time.time()
b = write_b(1000000)
end = time.time()
print('time.time(): {}'.format(end - start))

# time.perf_counter() measures system-wide high-res time elapse, and includes sleeps
# Tick rate of ~2mil tick/s
start2 = time.perf_counter()
b = write_b(1000000)
end2 = time.perf_counter()
print('time.perf_counter(): {}'.format(end2 - start2))

# time.process_time() measures process-wide high-res time elapse, not including sleeps
# Tick rate of 10mil tick/s
start3 = time.process_time()
b = write_b(1000000)
end3 = time.process_time()
print('time.process_time(): {}'.format(end3 - start3))

#############################
# Tracing Memory Allocation #
#############################

print('\n#####\nTrace Memory:')

tracemalloc.start()

written_b = write_b(1000000)

snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:10]:
    print(stat)

######################
# Function Profiling #
######################

print('\n#####\nFunction Profiling:')

cProfile.run('write_b(1000000)')
