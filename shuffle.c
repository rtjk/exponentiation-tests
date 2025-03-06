#include <immintrin.h>
#include <stdio.h>

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

    
    __m256i a = _mm256_setr_epi8(
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
        0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87, 0x98, 0xA9, 0xBA, 0xCB, 0xDC, 0xED, 0xFE, 0x0F
    );

    __m256i b = _mm256_setr_epi8(
        3, 2, 1, 0,  7, 6, 5, 4,  11, 10, 9, 8,  15, 14, 13, 12,
        3, 2, 1, 0,  7, 6, 5, 4,  11, 10, 9, 8,  15, 14, 13, 12
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
rm -f shuffle.o; gcc -o shuffle.o shuffle.c -mavx2; ./shuffle.o
*/
