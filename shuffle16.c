#include <immintrin.h>
#include <stdio.h>

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

/******************************************************************************/

/* shuffle sixteen 16-bit integers packed into a 256-bit vector:
   shuffle(a[], b[]) returns c[] where c[i]=a[b[i]] */

/******************************************************************************/

// OK
// MaxMath/Runtime/XSE Core/Shuffles/Variable Permute.cs
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

// OK
__m256i mm256_shuffle_epi16_B(__m256i a, __m256i b) {
    __m256i add_mask = _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    b = _mm256_adds_epu16(b, b);
    b = _mm256_adds_epu16(b, _mm256_bslli_epi128(b, 1));
    b = _mm256_adds_epu8(b, add_mask);
    b = _mm256_shuffle_epi8(a, b);
    return b;
}

/******************************************************************************/

// add vs adds: check Throughput (CPI)

/******************************************************************************/

int main() {

    
    __m256i a = _mm256_setr_epi16(
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7
    );

    __m256i b = _mm256_setr_epi16(
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7
    );

    __m256i result = mm256_shuffle_epi16_B(a, b);

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
rm -f shuffle16.o; gcc -o shuffle16.o shuffle16.c -mavx2; ./shuffle16.o
*/
