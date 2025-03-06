from math import log2, ceil, floor

for x in range(1, 128):
    y = bin(x)[2:]
    print(f"{x:3} {y:>7}")