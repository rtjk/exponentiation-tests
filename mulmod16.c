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


/******************************************************************************/

__m256i reduce_avx2_32(__m256i a){
    int b_shift = 18; // ceil(log2(509))*2
    int b_mul = (((uint64_t)1U << b_shift) / 509);
    /* r = a - ((B_MUL * a) >> B_SHIFT) * P) */
    __m256i b_mul_32 = _mm256_set1_epi32(b_mul);
    __m256i p_32 = _mm256_set1_epi32(509);
    __m256i r = _mm256_mullo_epi32(a, b_mul_32);
            r = _mm256_srli_epi32(r, b_shift);
            r = _mm256_mullo_epi32(r, p_32);
            r = _mm256_sub_epi32(a, r);
    /* r = min(r, r - P) */
    __m256i rs= _mm256_sub_epi32(r, p_32);
            r = _mm256_min_epu32(r, rs);
    return r;
}

void mm256_split(__m256i vec, __m128i *low, __m128i *high) {
    *low  = _mm256_extracti128_si256(vec, 0);
    *high = _mm256_extracti128_si256(vec, 1);
}

__m256i mm256_shuffle_epi16_A(__m256i a, __m256i b) {
    __m256i x1 = _mm256_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
    __m256i x2 = _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    b = _mm256_adds_epu16(b, b);
    b = _mm256_shuffle_epi8(b, x1);
    b = _mm256_adds_epu8(b, x2);
    b = _mm256_shuffle_epi8(a, b);
    return b;
}

/******************************************************************************/

/* ******** */

/******************************************************************************/

// BROKEN
__m256i mm256_mulmod509_epu16(__m256i a, __m256i b) {
    
    __m256i mask0   = _mm256_setr_epi16(0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF);
    __m256i mask1   = _mm256_setr_epi16(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);

    // entend a and b to 32 bits
    __m256i a0      = _mm256_and_si256(a, mask0);
            a0      = _mm256_bsrli_epi128(a0, 2);
    __m256i a1      = _mm256_and_si256(a, mask1);
    __m256i b0      = _mm256_and_si256(b, mask0);
            b0      = _mm256_bsrli_epi128(b0, 2);
    __m256i b1      = _mm256_and_si256(b, mask1);

    // multiply modulo 509
    __m256i m0      = _mm256_mullo_epi32(a0, b0);
            // m0      = reduce_avx2_32(m0);
            m0      = _mm256_bslli_epi128(m0, 2);
    __m256i m1      = _mm256_mullo_epi32(a1, b1);
            // m1      = reduce_avx2_32(m1);

    /// print m1
    printf("m0:   ");
    print_m256i_32bit(m0);

    // reassemble
    __m256i r       = _mm256_or_si256(m0, m1);
    
    return r;
}

/******************************************************************************/

int main() {

    
    __m256i a = _mm256_setr_epi16(
        3, 2, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 509
    );

    __m256i b = _mm256_setr_epi16(
        1, 2, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 2
    );

    __m256i result = mm256_mulmod509_epu16(a, b);

    // Print the result
    printf("a:   ");
    print_m256i_16bit(a);
    printf("b:   ");
    print_m256i_16bit(b);
    printf("c:   ");
    print_m256i_16bit(result);

    return 0;
}

/*
rm -f mulmod16.o; gcc -o mulmod16.o mulmod16.c -mavx2; ./mulmod16.o
*/
