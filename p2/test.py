
import time
from multiprocess import Pool, Process

import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
print(2)
if __name__ == "__main__":
    p = Process(target=lambda: print(1), args=[])
    p.start()
    p1 = Process(target=lambda: print(1), args=[])
    p1.start()
    p.join()
    p1.join()
    # pool = Pool()
    # results = [pool.apply(lambda x:x * 3, args=(a,)) for a in range(10)]
    # pool.close()
    # print(results)
print(3)
