#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>

void print_m256i(__m256i var) {
    unsigned char vals[32];
    _mm256_storeu_si256((__m256i*)vals, var);
    printf("[ ");
    for (int i = 0; i < 32; i++) {
        printf("%02X ", vals[i]);
    }
    printf("]\n");
}

int main() {

    // set vs setr

    ///////////////////////////////////////////////////////////////////////

    __m256i a = _mm256_set_epi16(
        0x0001,0x0020,0x0008,0x0002,  0x0040,0x0010,0x0004,0x0001,
        0x0001,0x0020,0x0008,0x0002,  0x0040,0x0010,0x0004,0x0001
    );

    __m256i tmp = _mm256_srli_si256(a, 7);
    a = _mm256_or_si256(a,tmp);

    ///////////////////////////////////////////////////////////////////////

    uint8_t bx = 3;

    __m256i b = _mm256_set1_epi8(bx);

    b = _mm256_setr_epi8(
        0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    );

    __m256i result = _mm256_shuffle_epi8(a, b);

    // Print the result
    printf("Original a:   ");
    print_m256i(a);
    printf("Shuffle mask: ");
    print_m256i(b);
    printf("Shuffled:     ");
    print_m256i(result);

    return 0;
}

/*
rm -f rtv.o; gcc -o rtv.o rtv.c -mavx2; ./rtv.o
*/