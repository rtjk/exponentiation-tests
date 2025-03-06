
#include <stdio.h>
#include <stdint.h>

#define TABLE ((uint64_t) (0x0140201008040201))

#define RTV(x) ( (uint8_t) (TABLE >> (8*(uint64_t)(x))) )

// RTV(x) = 7**x

int main()
{
    uint8_t x = 6;
    uint8_t y = RTV(x);
    printf("%d", y);
    return 0;
}
