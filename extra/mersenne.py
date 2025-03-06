
from math import floor
from time import sleep

MOD = 127
MAX = 2**100

# OK -> up to infinity
def f128(x):
    return x & 127

def f127(x):
    a = floor(x / 128)
    b = x % 128
    return a + b

for i in range(0, MAX):
    a = f127(i)
    b = i % MOD
    if(a != b):
        print("error", i, a, b)
        sleep(1)
        # break 

print("done")