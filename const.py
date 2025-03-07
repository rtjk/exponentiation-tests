from math import log2, ceil, floor

# binary to decimal
def btd(a):
    return int(a, 2)

# low 4 bits
maxd = btd("1111")
for x in range(maxd+1):
    y = (16**x) %509
    print(f"{x:5} {y:5}")

print()

# high 3 bits
maxd = btd("111")
for x in range(maxd+1):
    y = (16**(x*16)) %509
    print(f"{x:5} {y:5}")