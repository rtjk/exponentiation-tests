from math import log2, ceil, floor

# for x in range(1, 128):
#     y = bin(x)[2:]
#     print(f"{x:3} {y:>7}")

# binary to decimal
def btd(a):
    return int(a, 2)

# decimal to binary, pad to 7 bits
def dtb(a):
    return format(a, '07b')[2:]

x = 1
y = dtb(x)

# print(x)
# print(y)

z = (16**61) % 509 
w = ( (16**(3*16)) * (16**13) ) % 509
print(z)
print(w)
print(z == w)