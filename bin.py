from math import log2, ceil, floor

x = 30

y = bin(x)[2:]

print(x)
print(y)
print(len(y), 'bit')

# 16                      10000      5 bit
# 126                   1111110      7 bit
# 127                   1111111      7 bit
# 128                  10000000      8 bit
# 508                 111111100      9 bit
# 509                 111111101      9 bit
# 126^2          11111000000100     14 bit
# 508^2      111111000000010000     18 bit

# 256 /  8          32
# 256 / 16          16
# 256 / 32           8
# 256 / 64           4