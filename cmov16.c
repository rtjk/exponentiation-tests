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

/* for each 16-bit integer packed into a 256-bit vector, select one of two 
   values based on a boolean condition, without using if-else statements:
   cmov(cond[], true_val[], false_val[]) returns r[] where r[i]=true_val[i]
   if cond[i]==1, and r[i]=false_val[i] if cond[i]==0 */

/******************************************************************************/

// OK
__m256i mm256_cmov_epu16_A(__m256i c, __m256i t, __m256i f) {
    __m256i zeros  = _mm256_setzero_si256();
    __m256i cmask  = _mm256_sub_epi16(zeros, c);
    __m256i cmaskn = _mm256_xor_si256(cmask, _mm256_set1_epi16(0xFFFF));
    __m256i tval   = _mm256_and_si256(cmask, t);
    __m256i fval   = _mm256_and_si256(cmaskn, f);
    __m256i r      = _mm256_or_si256(tval, fval);
    return r;
}

/******************************************************************************/

// try min or max to speed up (like in reduce_avx2_32)

// try the sign extension to expand the zero or one to 16 bits

/******************************************************************************/

int main() {

    
    __m256i a = _mm256_setr_epi16(
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3
    );

    __m256i b = _mm256_setr_epi16(
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7
    );

    __m256i c = _mm256_setr_epi16(
        1, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0
    );

    __m256i result = mm256_cmov_epu16_A(c, a, b);

    // Print the result
    printf("a:   ");
    print_m256i_16bit(a);
    printf("b:   ");
    print_m256i_16bit(b);
    printf("c:   ");
    print_m256i_16bit(c);
    printf("res: ");
    print_m256i_16bit(result);

    return 0;
}

/*
rm -f cmov16.o; gcc -o cmov16.o cmov16.c -mavx2; ./cmov16.o
*/
