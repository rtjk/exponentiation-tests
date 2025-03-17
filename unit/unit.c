#include <assert.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/****************************** RSDPG parameters ******************************/

/*
#define N (55)
#define N (79)
*/
#define N (106)

#if (1)
#define P (509)
#define EPI16_PER_REG 16
#define ROUND_UP(amount, round_amt) ( ((amount+round_amt-1)/round_amt)*round_amt )
#define FP_ELEM uint16_t
#define FZ_ELEM uint8_t
#define FP_DOUBLEPREC uint32_t
#define FPRED_SINGLE(x) ((x) % P)
#define FP_ELEM_CMOV(BIT,TRUE_V,FALSE_V)  ( (((FP_ELEM)0 - (BIT)) & (TRUE_V)) | (~((FP_ELEM)0 - (BIT)) & (FALSE_V)) )
#define RESTR_G_GEN_1  ((FP_ELEM) 16)
#define RESTR_G_GEN_2  ((FP_ELEM) 256)
#define RESTR_G_GEN_4  ((FP_ELEM) 384)
#define RESTR_G_GEN_8  ((FP_ELEM) 355)
#define RESTR_G_GEN_16 ((FP_ELEM) 302)
#define RESTR_G_GEN_32 ((FP_ELEM) 93)
#define RESTR_G_GEN_64 ((FP_ELEM) 505)
static inline FP_ELEM RESTR_TO_VAL(FP_ELEM x) {
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
#endif

/**************************** reference functions *****************************/

void convert_restr_vec_to_fp(FP_ELEM res[N], const FZ_ELEM in[N]) {
    for(int j = 0; j < N; j++) {
        res[j] = RESTR_TO_VAL(in[j]);
    }
}

void fp_vec_by_restr_vec_scaled(FP_ELEM res[N], const FZ_ELEM e[N], const FP_ELEM chall_1, const FP_ELEM u_prime[N]) {
    for(int i = 0; i < N; i++) {
        res[i] = FPRED_SINGLE( (FP_DOUBLEPREC) u_prime[i] + (FP_DOUBLEPREC) RESTR_TO_VAL(e[i]) * (FP_DOUBLEPREC) chall_1) ;
    }
}

/*************************** avx2 helper functions ****************************/

/* reduce modulo 509 eigth 32-bit integers packed into a 256-bit vector, using Barrett's method
 * each 32-bit integer sould be in the range [0, 508*508] i.e. the result of a mul in FP
 * however, the function actually works for integers in the wider range [0, 8339743] */
static inline __m256i mm256_mod509_epu32(__m256i a) {
    int b_shift = 18; // ceil(log2(509))*2
    int b_mul = (((uint64_t)1U << b_shift) / P);
    /* r = a - ((B_MUL * a) >> B_SHIFT) * P) */
    __m256i b_mul_32 = _mm256_set1_epi32(b_mul);
    __m256i p_32 = _mm256_set1_epi32(P);
    __m256i r = _mm256_mullo_epi32(a, b_mul_32);
            r = _mm256_srli_epi32(r, b_shift);
            r = _mm256_mullo_epi32(r, p_32);
            r = _mm256_sub_epi32(a, r);
    /* r = min(r, r - P) */
    __m256i rs= _mm256_sub_epi32(r, p_32);
            r = _mm256_min_epu32(r, rs);
    return r;
}

/* reduce modulo 509 sixteen 16-bit integers packed into a 256-bit vector
 * each 32-bit integer sould be in the range [0, 508*2] */
static inline __m256i mm256_mod509_epu16(__m256i a) {
    /* r = min(r, r - P) */
    __m256i p_256 = _mm256_set1_epi16(509);
    __m256i as= _mm256_sub_epi16(a, p_256);
    return _mm256_min_epu16(a, as);
}

/* shuffle sixteen 16-bit integers packed into a 256-bit vector:
 * shuffle(a[], b[]) returns c[] where c[i]=a[b[i]] */
static inline __m256i mm256_shuffle_epi16(__m256i a, __m256i b) {
    __m256i x1 = _mm256_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
    __m256i x2 = _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    b = _mm256_adds_epu16(b, b);
    b = _mm256_shuffle_epi8(b, x1);
    b = _mm256_adds_epu8(b, x2);
    b = _mm256_shuffle_epi8(a, b);
    return b;
}

/* for each 16-bit integer packed into a 256-bit vector, select one of two 
 * values based on a boolean condition, without using if-else statements:
 * cmov(cond[], true_val[], false_val[]) returns r[] where r[i]=true_val[i]
 * if cond[i]==1, and r[i]=false_val[i] if cond[i]==0 */
static inline __m256i mm256_cmov_epu16(__m256i c, __m256i t, __m256i f) {
    __m256i zeros  = _mm256_setzero_si256();
    __m256i cmask  = _mm256_sub_epi16(zeros, c);
    __m256i cmaskn = _mm256_xor_si256(cmask, _mm256_set1_epi16(-1));
    __m256i tval   = _mm256_and_si256(cmask, t);
    __m256i fval   = _mm256_and_si256(cmaskn, f);
    __m256i r      = _mm256_or_si256(tval, fval);
    return r;
}

/* multiply 16-bit integers packed into 256-bit vectors and reduce the result
 * modulo 509: mulmod509(a[], b[]) returns c[] where c[i]=(a[i]*b[i])%509 */
static inline __m256i mm256_mulmod509_epu16(__m256i a, __m256i b) {
    /* multiply */
    __m256i l = _mm256_mullo_epi16(a, b);
    __m256i h = _mm256_mulhi_epu16(a, b);
    /* unpack 16-bit to 32-bit */
    __m256i u0 = _mm256_unpacklo_epi16(l, h);
    __m256i u1 = _mm256_unpackhi_epi16(l, h);
    /* reduce */
    u0 = mm256_mod509_epu32(u0);
    u1 = mm256_mod509_epu32(u1);
    /* pack 32-bit to 16-bit */
    __m256i r = _mm256_packs_epi32(u0, u1);
    return r;
}

/* for each 16-bit integer x packed into a 256-bit vector, with x in [1, 127],
 * compute: (16^x) mod 509 */
static inline __m256i mm256_exp16mod509_epu16(__m256i a) {
    /* high 3 bits */
    __m256i h3 = _mm256_srli_epi16(a, 4);
    __m256i pre_h3 = _mm256_setr_epi16(
        1,302,93,91,505,319,137,145,
        1,302,93,91,505,319,137,145);
    __m256i h3_shu = mm256_shuffle_epi16(pre_h3, h3);
    /* low 4 bits */
    __m256i mask_l4 = _mm256_set1_epi16(0x0F); //0b1111
    __m256i l4 = _mm256_and_si256(a, mask_l4);
    __m256i mask_l4_bit4 = _mm256_set1_epi16(0x8); //0b1000
    __m256i l4_bit4 = _mm256_and_si256(a, mask_l4_bit4);
            l4_bit4 = _mm256_srli_epi16(l4_bit4, 3);
    __m256i l4_sub8 = _mm256_sub_epi16(l4, _mm256_set1_epi16(8));
    __m256i pre_l4_0 = _mm256_setr_epi16(
        1,16,256,24,384,36,67,54,
        1,16,256,24,384,36,67,54);
    __m256i l4_shu_0 = mm256_shuffle_epi16(pre_l4_0, l4);
    __m256i pre_l4_1 = _mm256_setr_epi16(
        355,81,278,376,417,55,371,337,
        355,81,278,376,417,55,371,337);
    __m256i l4_shu_1 = mm256_shuffle_epi16(pre_l4_1, l4_sub8);
    __m256i l4_shu = mm256_cmov_epu16(l4_bit4, l4_shu_1, l4_shu_0);
    /* multiply */
    __m256i r = mm256_mulmod509_epu16(h3_shu, l4_shu);
    return r;
}

//////////////////// fp_vec_by_fp_matrix

/**************************** optimized functions *****************************/

void fp_vec_by_restr_vec_scaled_PAR(FP_ELEM r[N], const FZ_ELEM vr[N], const FP_ELEM el, const FP_ELEM vn[N]) {

    /* r: expand, align */
    alignas(32) FP_ELEM r_x[ROUND_UP(N,EPI16_PER_REG)];
    /* vr: convert from uint8 to uint16, expand, align */
    alignas(32) FP_ELEM vr_x[ROUND_UP(N,EPI16_PER_REG)];
    for (int i = 0; i < N; i++) {
        vr_x[i] = vr[i];
    }
    /* el: convert to m256i */
    __m256i el_256 = _mm256_set1_epi16(el);
    /* vn: expand, align */
    alignas(32) FP_ELEM vn_x[ROUND_UP(N,EPI16_PER_REG)];
    memcpy(vn_x, vn, N*sizeof(FP_ELEM));

    /* r = vn + RTV(vr) * el */
    for(int i = 0; i < ROUND_UP(N,EPI16_PER_REG)/EPI16_PER_REG; i++ ) {
        __m256i vn_256 = _mm256_load_si256( (__m256i const *) &vn_x[i*EPI16_PER_REG] );
        __m256i vr_256 = _mm256_load_si256( (__m256i const *) &vr_x[i*EPI16_PER_REG] );
        __m256i r_256;
        r_256 = mm256_exp16mod509_epu16(vr_256);
        r_256 = mm256_mulmod509_epu16(r_256, el_256);
        r_256 = _mm256_add_epi16(r_256, vn_256);
        r_256 = mm256_mod509_epu16(r_256);
        _mm256_store_si256 ((__m256i *) &r_x[i*EPI16_PER_REG], r_256);
    }
    memcpy(r, r_x, N*sizeof(FP_ELEM));
}

void convert_restr_vec_to_fp_PAR(FP_ELEM r[N], const FZ_ELEM vr[N]) {
    /* r: expand, align */
    alignas(32) FP_ELEM r_x[ROUND_UP(N,EPI16_PER_REG)];
    /* vr: convert from uint8 to uint16, expand, align */
    alignas(32) FP_ELEM vr_x[ROUND_UP(N,EPI16_PER_REG)];
    for (int i = 0; i < N; i++) {
        vr_x[i] = vr[i];
    }

    for(int i = 0; i < ROUND_UP(N,EPI16_PER_REG)/EPI16_PER_REG; i++ ) {
        __m256i vr_256 = _mm256_load_si256( (__m256i const *) &vr_x[i*EPI16_PER_REG] );
        __m256i r_256 = mm256_exp16mod509_epu16(vr_256);
        _mm256_store_si256 ((__m256i *) &r_x[i*EPI16_PER_REG], r_256);
    }
    memcpy(r, r_x, N*sizeof(FP_ELEM));
}


/****************************** testing utility *******************************/

uint16_t exp16mod509(uint16_t exponent) {
    uint32_t result = 1;
    for (int i = 0; i < exponent; i++) {
        result = (result * 16) % 509;
    }
    return (uint16_t)result;
}

void rand_arr_16xN_mod509(uint16_t *a) {
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 509;
    }
}

void rand_arr_8xN_1to127(uint8_t *a) {
    // instead of 0-126, use 1-127
    for (int i = 0; i < N; i++) {
        a[i] = (rand() % 127) + 1;
    }
}

/***************************** testing functions ******************************/

void TEST_mm256_mod509_epu32() {
    printf("TEST_mm256_mod509_epu32\n");
    for(uint32_t i=0; i<=(508*508); i++) {
        __m256i a = _mm256_set1_epi32(i);
        __m256i r = mm256_mod509_epu32(a);
        uint32_t arr[8];
        _mm256_storeu_si256((__m256i*)arr, r);
        for(int j=0; j<8; j++) {
            if(arr[j] != (i % 509)) {
                printf("Error: %d %d %d\n", i, arr[j], i % 509);
                exit(1);
            }
        }
    }
}

void TEST_mm256_mod509_epu16() {
    printf("TEST_mm256_mod509_epu16\n");
    for(uint16_t i=0; i<=(508*2); i++) {
        __m256i a = _mm256_set1_epi16(i);
        __m256i r = mm256_mod509_epu16(a);
        uint16_t arr[16];
        _mm256_storeu_si256((__m256i*)arr, r);
        for(int j=0; j<16; j++) {
            if(arr[j] != (i % 509)) {
                printf("Error: %d %d %d\n", i, arr[j], i % 509);
                exit(1);
            }
        }
    }
}

void TEST_mm256_shuffle_epi16() {}

void TEST_mm256_cmov_epu16() {}

void TEST_mm256_mulmod509_epu16() {}

void TEST_mm256_exp16mod509_epu16() {
    printf("TEST_mm256_exp16mod509_epu16\n");
    for(int i=1; i<=127; i++) {
        __m256i a = _mm256_set1_epi16(i);
        __m256i r = mm256_exp16mod509_epu16(a);
        uint16_t arr[16];
        _mm256_storeu_si256((__m256i*)arr, r);
        uint64_t true_r = exp16mod509(i);
        for(int j=0; j<16; j++) {
            if(arr[j] != true_r) {
                printf("Error: %d %d %d\n", i, arr[j], true_r);
                exit(1);
            }
        }
    }
}

void TEST_fp_vec_by_restr_vec_scaled_PAR(uint64_t random_tests) {
    printf("TEST_fp_vec_by_restr_vec_scaled_PAR [%lld]\n", random_tests);
    uint16_t r_ser[N];
    uint16_t r_par[N];
    uint8_t vr[N];
    uint16_t vn[N];
    uint16_t el; 
    for(int i = 0; i < random_tests; i++) {
        rand_arr_8xN_1to127(vr);
        rand_arr_16xN_mod509(vn);
        el = rand() % 509;
        fp_vec_by_restr_vec_scaled(r_ser, vr, el, vn);
        fp_vec_by_restr_vec_scaled_PAR(r_par, vr, el, vn);
        for(int j=0; j<N; j++){
            if(r_ser[j] != r_par[j]){
                printf("Error: \n i=%d \n j=%d \n vr[j]=%d \n vn[j]=%d \n el=%d \n r_ser[j]=%d \n r_par[j]=%d\n", i, j, vr[j], vn[j], el, r_ser[j], r_par[j]);
                exit(1);
            }
        }
    
    }
}

void TEST_convert_restr_vec_to_fp_PAR() {}

/******************************************************************************/

#define TESTS 10000

int main() {

    srand(time(0));
    srand(0);

    TEST_mm256_mod509_epu32();
    TEST_mm256_mod509_epu16();
    TEST_mm256_shuffle_epi16();
    TEST_mm256_cmov_epu16();
    TEST_mm256_mulmod509_epu16();
    TEST_mm256_exp16mod509_epu16();
    TEST_fp_vec_by_restr_vec_scaled_PAR(TESTS);
    TEST_convert_restr_vec_to_fp_PAR();

    printf("\nOK\n");

    return 0;
}

/*
rm -f unit.o; gcc -o unit.o unit.c -march=native -g3 -fsanitize=address; ./unit.o
*/

