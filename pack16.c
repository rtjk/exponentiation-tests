#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>

void print_m256i_8bit(__m256i var) {
    unsigned char vals[32];
    _mm256_storeu_si256((__m256i*)vals, var);
    printf("[ ");
    for (int i = 0; i < 32; i++) {
        printf("%02X ", vals[i]);
    }
    printf("]\n");
}

void print_m256i_16bit(__m256i var) {
    unsigned short vals[16];  // 16 values, 2 bytes each (16 bits)
    _mm256_storeu_si256((__m256i*)vals, var);  // Store 256 bits (16 values) into the vals array
    printf("[ ");
    for (int i = 0; i < 16; i++) {
        printf("%04X ", vals[i]);  // Print each value as a 4-digit hexadecimal number (16 bits)
    }
    printf("]\n");
}

void print_m256i_32bit(__m256i var) {
    unsigned int vals[8];  // 8 values, 4 bytes each (32 bits)
    _mm256_storeu_si256((__m256i*)vals, var);  // Store 256 bits (8 values) into the vals array
    printf("[ ");
    for (int i = 0; i < 8; i++) {
        printf("%08X ", vals[i]);  // Print each value as an 8-digit hexadecimal number (32 bits)
    }
    printf("]\n");
}


int main() {

    
    __m256i a = _mm256_setr_epi32(
        1, 2, 3, 4, 0xA, 0xB, 0xC, 0xD
    );

    __m256i b = _mm256_setr_epi32(
        5, 6, 7, 8, 0xE, 0xF, 0, 0
    );

    __m256i c;

    c = _mm256_packs_epi32(a, b);
    // a = 1,2,3,4,A,B,C,D
    // b = 5,6,7,8,E,F,G,H
    // c = 1,2,3,4,5,6,7,8,A,B,C,D,E,F,G,H

    // Print the result
    printf("a:   ");
    print_m256i_32bit(a);
    printf("b:   ");
    print_m256i_32bit(b);
    printf("c:   ");
    print_m256i_16bit(c);

    return 0;
}

/*
rm -f pack16.o; gcc -o pack16.o pack16.c -mavx2; ./pack16.o
*/
