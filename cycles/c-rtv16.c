#include "cpucycles.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>

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

uint16_t exp16mod509(uint16_t exponent) {
    uint32_t result = 1;
    for (int i = 0; i < exponent; i++) {
        result = (result * 16) % 509;
    }
    return (uint16_t)result;
}

/******************************************************************************/

static inline __m256i reduce_avx2_32(__m256i a){
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

static inline __m256i mm256_shuffle_epi16_A(__m256i a, __m256i b) {
    __m256i x1 = _mm256_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
    __m256i x2 = _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    b = _mm256_adds_epu16(b, b);
    b = _mm256_shuffle_epi8(b, x1);
    b = _mm256_adds_epu8(b, x2);
    b = _mm256_shuffle_epi8(a, b);
    return b;
}

static inline __m256i mm256_cmov_epu16_A(__m256i c, __m256i t, __m256i f) {
    __m256i zeros  = _mm256_setzero_si256();
    __m256i cmask  = _mm256_sub_epi16(zeros, c);
    __m256i cmaskn = _mm256_xor_si256(cmask, _mm256_set1_epi16(0xFFFF));
    __m256i tval   = _mm256_and_si256(cmask, t);
    __m256i fval   = _mm256_and_si256(cmaskn, f);
    __m256i r      = _mm256_or_si256(tval, fval);
    return r;
}

static inline __m256i mm256_mulmod509_epu16_B(__m256i a, __m256i b) {
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

#define N (106)
#define EPI16_PER_REG 16
#define ROUND_UP(amount, round_amt) ( ((amount+round_amt-1)/round_amt)*round_amt )

/******************************************************************************/

static inline __m256i mm256_exp16mod509_epu16(__m256i a) {
    
    __m256i h3 = _mm256_srli_epi16(a, 4);

    __m256i pre_h3 = _mm256_setr_epi16(
        1,302,93,91,505,319,137,145,
        1,302,93,91,505,319,137,145);
    __m256i h3_shu = mm256_shuffle_epi16_A(pre_h3, h3);
    
    __m256i mask_l4 = _mm256_set1_epi16(0x0F);
    __m256i l4 = _mm256_and_si256(a, mask_l4);

    __m256i mask_l4_bit4 = _mm256_set1_epi16(0b1000);
    __m256i l4_bit4 = _mm256_and_si256(a, mask_l4_bit4);
            l4_bit4 = _mm256_srli_epi16(l4_bit4, 3);
    // _mm256_bsrli_epi128 vs _mm256_srli_epi16

    __m256i l4_sub8 = _mm256_sub_epi16(l4, _mm256_set1_epi16(8));

    __m256i pre_l4_0 = _mm256_setr_epi16(
        1,16,256,24,384,36,67,54,
        1,16,256,24,384,36,67,54);
    __m256i l4_shu_0 = mm256_shuffle_epi16_A(pre_l4_0, l4);

    __m256i pre_l4_1 = _mm256_setr_epi16(
        355,81,278,376,417,55,371,337,
        355,81,278,376,417,55,371,337);
    __m256i l4_shu_1 = mm256_shuffle_epi16_A(pre_l4_1, l4_sub8);

    __m256i l4_shu = mm256_cmov_epu16_A(l4_bit4, l4_shu_1, l4_shu_0);

    __m256i r = mm256_mulmod509_epu16_B(h3_shu, l4_shu);

    return r;

}

static inline void exp16mod509_x16_par(uint16_t *r, uint8_t *a) {
    
    // a: convert from uint8 to uint16, expand, align
    alignas(32) uint16_t a_align[ROUND_UP(N,EPI16_PER_REG)];
    for (int i = 0; i < N; i++) {
        a_align[i] = a[i];
    }

    // r: expand, align
    alignas(32) uint16_t r_align[ROUND_UP(N,EPI16_PER_REG)];

    for(int i = 0; i < ROUND_UP(N,EPI16_PER_REG)/EPI16_PER_REG; i++ ){
        __m256i a_256 =  _mm256_load_si256( (__m256i const *) &a_align[i*EPI16_PER_REG] );
        __m256i r_256 = mm256_exp16mod509_epu16(a_256);
        _mm256_store_si256 ((__m256i *) &r_align[i*EPI16_PER_REG], r_256);
    }
    memcpy(r,r_align,N);
}

/******************************************************************************/

#define FP_ELEM uint16_t
#define P (509)
#define FPRED_SINGLE(x) (((x) - (((uint64_t)(x) * 2160140723) >> 40) * P))
#define FP_ELEM_CMOV(BIT,TRUE_V,FALSE_V)  ( (((FP_ELEM)0 - (BIT)) & (TRUE_V)) | (~((FP_ELEM)0 - (BIT)) & (FALSE_V)) )
#define RESTR_G_GEN_1  ((FP_ELEM) 16)
#define RESTR_G_GEN_2  ((FP_ELEM) 256)
#define RESTR_G_GEN_4  ((FP_ELEM) 384)
#define RESTR_G_GEN_8  ((FP_ELEM) 355)
#define RESTR_G_GEN_16 ((FP_ELEM) 302)
#define RESTR_G_GEN_32 ((FP_ELEM) 93)
#define RESTR_G_GEN_64 ((FP_ELEM) 505)
static inline FP_ELEM RESTR_TO_VAL(FP_ELEM x){
    uint32_t res1, res2, res3, res4;
    res1 = ( FP_ELEM_CMOV(((x >> 0) &1),RESTR_G_GEN_1 ,1)) *
           ( FP_ELEM_CMOV(((x >> 1) &1),RESTR_G_GEN_2 ,1)) ;
    res2 = ( FP_ELEM_CMOV(((x >> 2) &1),RESTR_G_GEN_4 ,1)) *
           ( FP_ELEM_CMOV(((x >> 3) &1),RESTR_G_GEN_8 ,1)) ;
    res3 = ( FP_ELEM_CMOV(((x >> 4) &1),RESTR_G_GEN_16,1)) *
           ( FP_ELEM_CMOV(((x >> 5) &1),RESTR_G_GEN_32,1)) ;
    res4 =   FP_ELEM_CMOV(((x >> 6) &1),RESTR_G_GEN_64,1);

    /* Two intermediate reductions necessary:
     *     RESTR_G_GEN_1*RESTR_G_GEN_2*RESTR_G_GEN_4*RESTR_G_GEN_8    < 2^32
     *     RESTR_G_GEN_16*RESTR_G_GEN_32*RESTR_G_GEN_64               < 2^32 */
    return FPRED_SINGLE( FPRED_SINGLE(res1 * res2) * FPRED_SINGLE(res3 * res4) );
}

static inline void exp16mod509_x16_ser(uint16_t *r, uint8_t *a) {
    for (int i = 0; i < 16; i++) {
        r[i] = FPRED_SINGLE(RESTR_TO_VAL(a[i]));
    }
}


/******************************************************************************/

static inline void rand_arr_16xN_mod509(uint8_t *a) {
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 509;
    }
}

/******************************************************************************/

#define TESTS 10000000
/*
#define SERIAL 1
#define TESTS 100
*/

int main() {

    srand(time(0));
    srand(0);

    long long count_1;
    long long count_2;
    long long sum = 0;

    uint8_t a[N];
    uint16_t r[N];

    uint64_t throwaway = 0;

    for(long long test=0; test<TESTS; test++){

        rand_arr_16xN_mod509(a);

        count_1 = cpucycles();
        ////////////////////////////////////////////////////////////////////////////
        #if SERIAL
            exp16mod509_x16_ser(r, a);
        #else
            exp16mod509_x16_par(r, a);
        #endif
        ////////////////////////////////////////////////////////////////////////////
        count_2 = cpucycles();
        sum += count_2 - count_1;

        // throwaway
        for(int i=0; i<16; i++){
            throwaway += r[i];
            throwaway %= 10000;
        }

    }
    printf("[%lu]Cycles: %lld\n", throwaway, sum/TESTS);

    return throwaway;
}

// 16x16 (a single 256-bit register) 
// par 27
// ser 200

// 16x106 (7 256-bit registers)
// par 296
// ser 303

/*
rm -f c-rtv16.o; gcc -o c-rtv16.o c-rtv16.c -march=native -O3 -lcpucycles; taskset --cpu-list 0 ./c-rtv16.o
rm -f c-rtv16.o; gcc -o c-rtv16.o c-rtv16.c -march=native -O3 -lcpucycles; echo "COMPILED!"
taskset --cpu-list 0 ./c-rtv16.o
-g3 -fsanitize=address
*/
