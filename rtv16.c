#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

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

void print_m256i_16bit_asint(__m256i var) {
    unsigned short vals[16];  // 16 values, 2 bytes each (16 bits)
    _mm256_storeu_si256((__m256i*)vals, var);  // Store 256 bits (16 values) into the vals array
    printf("[ ");
    for (int i = 0; i < 16; i++) {
        printf("%d ", vals[i]);  // Print each value as a 4-digit hexadecimal number (16 bits)
    }
    printf("]\n");
}

long long mod_exp(long long base, long long exp, long long mod) {
    long long result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {  // If exponent is odd, multiply result by base
            result = (result * base) % mod;
        }
        base = (base * base) % mod;  // Square the base
        exp /= 2;
    }
    return result;
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

__m256i mm256_shuffle_epi16_A(__m256i a, __m256i b) {
    __m256i x1 = _mm256_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
    __m256i x2 = _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    b = _mm256_adds_epu16(b, b);
    b = _mm256_shuffle_epi8(b, x1);
    b = _mm256_adds_epu8(b, x2);
    b = _mm256_shuffle_epi8(a, b);
    return b;
}

__m256i mm256_cmov_epu16_A(__m256i c, __m256i t, __m256i f) {
    __m256i zeros  = _mm256_setzero_si256();
    __m256i cmask  = _mm256_sub_epi16(zeros, c);
    __m256i cmaskn = _mm256_xor_si256(cmask, _mm256_set1_epi16(0xFFFF));
    __m256i tval   = _mm256_and_si256(cmask, t);
    __m256i fval   = _mm256_and_si256(cmaskn, f);
    __m256i r      = _mm256_or_si256(tval, fval);
    return r;
}

__m256i mm256_mulmod509_epu16_B(__m256i a, __m256i b) {
    /* multiply */
    __m256i l = _mm256_mullo_epi16(a, b);
    __m256i h = _mm256_mulhi_epu16(a, b);
    /* unpack 16-bit to 32-bit */
    __m256i u0 = _mm256_unpacklo_epi16(l, h);
    __m256i u1 = _mm256_unpackhi_epi16(l, h);
    /* reduce */
    u0 = reduce_avx2_32(u0);
    u1 = reduce_avx2_32(u1);
    /* pack 32-bit to 16-bit */
    __m256i r = _mm256_packs_epi32(u0, u1);
    return r;
}

/******************************************************************************/

/* for each 16-bit integer x packed into a 256-bit vector, with x in [1, 127],
   compute: (16**x) mod 509 */

/******************************************************************************/

// OK
__m256i mm256_exp16mod509_epu16(__m256i a) {
    
    __m256i h3 = _mm256_srli_epi16(a, 4);

    __m256i pre_h3 = _mm256_setr_epi16(
        1,302,93,91,505,319,137,145,
        1,302,93,91,505,319,137,145);
    __m256i h3_shu = mm256_shuffle_epi16_A(pre_h3, h3);

    // printf("h3_s:");
    // print_m256i_16bit_asint(h3_shu);
    
    __m256i mask_l4 = _mm256_set1_epi16(0x0F);
    __m256i l4 = _mm256_and_si256(a, mask_l4);

    __m256i mask_l4_bit4 = _mm256_set1_epi16(0b1000);
    __m256i l4_bit4 = _mm256_and_si256(a, mask_l4_bit4);
            l4_bit4 = _mm256_srli_epi16(l4_bit4, 3);
    // _mm256_bsrli_epi128 vs _mm256_srli_epi16

    // printf("l4b4:");
    // print_m256i_16bit_asint(l4_bit4);

    __m256i l4_sub8 = _mm256_sub_epi16(l4, _mm256_set1_epi16(8));

    __m256i pre_l4_0 = _mm256_setr_epi16(
        1,16,256,24,384,36,67,54,
        1,16,256,24,384,36,67,54);
    __m256i l4_shu_0 = mm256_shuffle_epi16_A(pre_l4_0, l4);

    // printf("l4s0:");
    // print_m256i_16bit_asint(l4_shu_0);

    __m256i pre_l4_1 = _mm256_setr_epi16(
        355,81,278,376,417,55,371,337,
        355,81,278,376,417,55,371,337);
    __m256i l4_shu_1 = mm256_shuffle_epi16_A(pre_l4_1, l4_sub8);

    // printf("l4s1:");
    // print_m256i_16bit_asint(l4_shu_1);

    __m256i l4_shu = mm256_cmov_epu16_A(l4_bit4, l4_shu_1, l4_shu_0);

    // printf("l4_s:");
    // print_m256i_16bit_asint(l4_shu);

    __m256i r = mm256_mulmod509_epu16_B(h3_shu, l4_shu);

    return r;

}

/******************************************************************************/

int main() {

    __m256i a = _mm256_setr_epi16(
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16
    );

    a = _mm256_set1_epi16(8);

    __m256i result = mm256_exp16mod509_epu16(a);

    // Print the result
    printf("a:   ");
    print_m256i_16bit_asint(a);
    printf("r:   ");
    print_m256i_16bit_asint(result);

    ////////////////////////////////////////////////////////////////////////////
    for(int i=1; i<=127; i++){
        printf(".");
        a = _mm256_set1_epi16(i);
        __m256i r = mm256_exp16mod509_epu16(a);
        uint16_t arr[16];
        _mm256_storeu_si256((__m256i*)arr, r);
        uint64_t true_r = mod_exp(16, i, 509);
        for(int j=0; j<16; j++){
            if(arr[j] != true_r){
                printf("\nError: %d %d %d\n", i, arr[j], true_r);
                exit(1);
            }
        }
    }
    printf("\nOK\n");
    ////////////////////////////////////////////////////////////////////////////

    return 0;
}

/*
rm -f rtv16.o; gcc -o rtv16.o rtv16.c -mavx2 -lm; ./rtv16.o
*/
