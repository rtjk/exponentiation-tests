/*** unit tests for the AVX2-optimized arithmetic functions in CROSS RSDPG ****/

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
#define RSDPG1
#define RSDPG3
*/
#define RSDPG5

#if (1)
#define P (509)
#if defined(RSDPG1)
    #define N (55)
    #define K (36)
#elif defined(RSDPG3)
    #define N (79)
    #define K (48)
#elif defined(RSDPG5)
    #define N (106)
    #define K (69)
#endif
#define EPI8_PER_REG 32
#define EPI16_PER_REG 16
#define EPI32_PER_REG 8
#define ROUND_UP(amount, round_amt) ( ((amount+round_amt-1)/round_amt)*round_amt )
#define FP_ELEM uint16_t
#define FZ_ELEM uint8_t
#define FP_DOUBLEPREC uint32_t
#define FPRED_SINGLE(x) ((x) % P)
#define FPRED_DOUBLE(x) (FPRED_SINGLE(x))
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

void fp_vec_by_fp_matrix(FP_ELEM res[N-K], FP_ELEM e[N], FP_ELEM V_tr[K][N-K]){
    memcpy(res,e+K,(N-K)*sizeof(FP_ELEM));
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N-K; j++){
            res[j] = FPRED_DOUBLE( (FP_DOUBLEPREC) res[j] + (FP_DOUBLEPREC) e[i] * (FP_DOUBLEPREC) V_tr[i][j]);
        }
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
 * shuffle(a[], b[]) returns c[] where c[i]=a[b[i]] 
 * operates within 128-bit lanes, so b[i] must be in the range [0,7] */
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

/**************************** optimized functions *****************************/

void fp_vec_by_restr_vec_scaled_PAR(FP_ELEM res[N], const FZ_ELEM e[N], const FP_ELEM chall_1, const FP_ELEM u_prime[N]) {

    /* res: expand, align */
    alignas(32) FP_ELEM res_x[ROUND_UP(N,EPI16_PER_REG)];
    /* e: convert from uint8 to uint16, expand, align */
    alignas(32) FP_ELEM e_x[ROUND_UP(N,EPI16_PER_REG)];
    for (int i = 0; i < N; i++) {
        e_x[i] = e[i];
    }
    /* chall_1: convert to m256i */
    __m256i chall_1_256 = _mm256_set1_epi16(chall_1);
    /* u_prime: expand, align */
    alignas(32) FP_ELEM u_prime_x[ROUND_UP(N,EPI16_PER_REG)];
    memcpy(u_prime_x, u_prime, N*sizeof(FP_ELEM));

    /* res = u_prime + RTV(e) * chall_1 */
    for(int i = 0; i < ROUND_UP(N,EPI16_PER_REG)/EPI16_PER_REG; i++ ){
        __m256i u_prime_256 = _mm256_load_si256( (__m256i const *) &u_prime_x[i*EPI16_PER_REG] );
        __m256i e_256 = _mm256_load_si256( (__m256i const *) &e_x[i*EPI16_PER_REG] );
        __m256i r_256;
        r_256 = mm256_exp16mod509_epu16(e_256);
        r_256 = mm256_mulmod509_epu16(r_256, chall_1_256);
        r_256 = _mm256_add_epi16(r_256, u_prime_256);
        r_256 = mm256_mod509_epu16(r_256);
        _mm256_store_si256 ((__m256i *) &res_x[i*EPI16_PER_REG], r_256);
    }
    memcpy(res, res_x, N*sizeof(FP_ELEM));
}

void convert_restr_vec_to_fp_PAR(FP_ELEM res[N], const FZ_ELEM in[N]){
    /* res: expand, align */
    alignas(32) FP_ELEM res_x[ROUND_UP(N,EPI16_PER_REG)];
    /* in: convert from uint8 to uint16, expand, align */
    alignas(32) FP_ELEM in_x[ROUND_UP(N,EPI16_PER_REG)];
    for (int i = 0; i < N; i++) {
        in_x[i] = in[i];
    }

    for(int i = 0; i < ROUND_UP(N,EPI16_PER_REG)/EPI16_PER_REG; i++ ){
        __m256i in_256 = _mm256_load_si256( (__m256i const *) &in_x[i*EPI16_PER_REG] );
        __m256i res_256 = mm256_exp16mod509_epu16(in_256);
        _mm256_store_si256 ((__m256i *) &res_x[i*EPI16_PER_REG], res_256);
    }
    memcpy(res, res_x, N*sizeof(FP_ELEM));
}

void fp_vec_by_fp_matrix_PAR(FP_ELEM res[N-K], FP_ELEM e[N], FP_DOUBLEPREC V_tr[K][ROUND_UP(N-K,EPI32_PER_REG)]){

    alignas(EPI8_PER_REG) FP_DOUBLEPREC res_dprec[ROUND_UP(N-K,EPI32_PER_REG)] = {0};
    for(int i=0; i< N-K;i++) {
        res_dprec[i]=e[K+i];
    }

    for(int i = 0; i < K; i++){
        __m256i e_coeff = _mm256_set1_epi32(e[i]);
        for(int j = 0; j < ROUND_UP(N-K,EPI32_PER_REG)/EPI32_PER_REG; j++){
            __m256i res_w = _mm256_load_si256((__m256i const *) &res_dprec[j*EPI32_PER_REG] );
            __m256i V_tr_slice = _mm256_lddqu_si256((__m256i const *) &V_tr[i][j*EPI32_PER_REG] );
            __m256i a  = _mm256_mullo_epi32(e_coeff, V_tr_slice);
            /* add to result */
            res_w = _mm256_add_epi32(res_w, a);
            /* - the previous sum is performed K times with K <= 69
             * - adding each time a value "a" in the range [0,(P-1)*(P-1)] 
             * - the reduction function mm256_mod509_epu32(x) works for x < 8339743
             * therefore 3 reductions are enough */
            if(i == K/3 || i == (K/3)*2 || i == K-1){
                res_w  = mm256_mod509_epu32(res_w);
            }
            /* store back */
            _mm256_store_si256 ((__m256i *) &res_dprec[j*EPI32_PER_REG], res_w);
        }
    }
    for(int i=0; i< N-K;i++) {
        res[i] = res_dprec[i];
    }
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

void rand_arr_16x16_mod509(uint16_t *a) {
    for (int i = 0; i < 16; i++) {
        a[i] = rand() % 509;
    }
}

void rand_arr_16xL_modM(uint16_t *a, uint64_t length, uint16_t mod) {
    for (int i = 0; i < length; i++) {
        a[i] = rand() % mod;
    }
}

void rand_arr_32xL_modM(uint32_t *a, uint64_t length, uint32_t mod) {
    for (int i = 0; i < length; i++) {
        a[i] = rand() % mod;
    }
}

/************************ iterative testing functions *************************/

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

void TEST_mm256_mulmod509_epu16() {
    printf("TEST_mm256_mulmod509_epu16\n");
    for(uint16_t i=0; i<509; i++) {
        for(uint16_t j=0; j<509; j++){
            __m256i a = _mm256_set1_epi16(i);
            __m256i b = _mm256_set1_epi16(j);
            __m256i r = mm256_mulmod509_epu16(a, b);
            uint16_t arr[16];
            _mm256_storeu_si256((__m256i*)arr, r);
            for(int k=0; k<16; k++) {
                if(arr[k] != (i*j) % 509) {
                    printf("Error: %d %d %d %d\n", i, j, arr[k], (i*j) % 509);
                    exit(1);
                }
            }
        }
    }
}

void TEST_mm256_exp16mod509_epu16() {
    printf("TEST_mm256_exp16mod509_epu16\n");
    for(uint16_t i=1; i<=127; i++) {
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

/************************** random testing functions **************************/

void TEST_RAND_mm256_mod509_epu32(uint64_t random_tests) {
    printf("TEST_RAND_mm256_mod509_epu32 [%lld]\n", random_tests);
    uint32_t a[8];
    uint32_t res_par[8];
    for(uint64_t i=0; i< random_tests; i++) {
        rand_arr_32xL_modM(a, 8, 508*508 +1);
        __m256i a_256 = _mm256_loadu_si256((__m256i*)a);
        __m256i r_256 = mm256_mod509_epu32(a_256);
        _mm256_storeu_si256((__m256i*)res_par, r_256);
        for(int j=0; j<8; j++) {
            if(res_par[j] != (a[j] % 509)) {
                printf("Error: \n i=%d \n j=%d \n a[j]=%d \n res_ser[j]=%d \n res_par[j]=%d\n", i, j, a[j], a[j] % 509, res_par[j]);
                exit(1);
            }
        }
    }
}

void TEST_RAND_mm256_mod509_epu16(uint64_t random_tests) {
    printf("TEST_RAND_mm256_mod509_epu16 [%lld]\n", random_tests);
    uint16_t a[16];
    uint16_t res_par[16];
    for(uint64_t i=0; i< random_tests; i++) {
        rand_arr_16xL_modM(a, 16, 508*2 +1);
        __m256i a_256 = _mm256_loadu_si256((__m256i*)a);
        __m256i r_256 = mm256_mod509_epu16(a_256);
        _mm256_storeu_si256((__m256i*)res_par, r_256);
        for(int j=0; j<16; j++) {
            if(res_par[j] != (a[j] % 509)) {
                printf("Error: \n i=%d \n j=%d \n a[j]=%d \n res_ser[j]=%d \n res_par[j]=%d\n", i, j, a[j], a[j] % 509, res_par[j]);
                exit(1);
            }
        }
    }
}

void TEST_RAND_mm256_mulmod509_epu16(uint64_t random_tests) {
    printf("TEST_RAND_mm256_mulmod509_epu16 [%lld]\n", random_tests);
    uint16_t a[16];
    uint16_t b[16];
    uint16_t res_par[16];
    for(uint64_t i=0; i< random_tests; i++) {
        rand_arr_16xL_modM(a, 16, 509);
        rand_arr_16xL_modM(b, 16, 509);
        __m256i a_256 = _mm256_loadu_si256((__m256i*)a);
        __m256i b_256 = _mm256_loadu_si256((__m256i*)b);
        __m256i r_256 = mm256_mulmod509_epu16(a_256, b_256);
        _mm256_storeu_si256((__m256i*)res_par, r_256);
        for(int j=0; j<16; j++) {
            if(res_par[j] != (a[j]*b[j]) % 509) {
                printf("Error: \n i=%d \n j=%d \n a[j]=%d \n b[j]=%d \n res_ser[j]=%d \n res_par[j]=%d\n", i, j, a[j], b[j], (a[j]*b[j]) % 509, res_par[j]);
                exit(1);
            }
        }
    }
}

void TEST_RAND_mm256_exp16mod509_epu16(uint64_t random_tests) {
    printf("TEST_RAND_mm256_exp16mod509_epu16 [%lld]\n", random_tests);
    uint16_t a[16];
    uint16_t res_par[16];
    for(uint64_t i=0; i< random_tests; i++) {
        rand_arr_16xL_modM(a, 16, 127);
        __m256i a_256 = _mm256_loadu_si256((__m256i*)a);
        __m256i r_256 = mm256_exp16mod509_epu16(a_256);
        _mm256_storeu_si256((__m256i*)res_par, r_256);
        for(int j=0; j<16; j++) {
            if(res_par[j] != exp16mod509(a[j])) {
                printf("Error: \n i=%d \n j=%d \n a[j]=%d \n res_ser[j]=%d \n res_par[j]=%d\n", i, j, a[j], exp16mod509(a[j]), res_par[j]);
                exit(1);
            }
        }
    }
}

void TEST_RAND_mm256_shuffle_epi16(uint64_t random_tests) {
    printf("TEST_RAND_mm256_shuffle_epi16 [%lld]\n", random_tests);
    uint16_t a[16];
    uint16_t b[16];
    uint16_t res_par[16];
    for(uint64_t i = 0; i < random_tests; i++) {
        rand_arr_16xL_modM(a, 16, UINT16_MAX);
        rand_arr_16xL_modM(b, 16, 8);
        __m256i a_256 = _mm256_loadu_si256((__m256i*)a);
        __m256i b_256 = _mm256_loadu_si256((__m256i*)b);
        __m256i res_256 = mm256_shuffle_epi16(a_256, b_256);
        _mm256_storeu_si256((__m256i*)res_par, res_256);
        // first 128-bit lane
        for(int j=0; j<=7; j++) {
            uint16_t res_ser = a[b[j]];
            if(res_par[j] != res_ser) {
                printf("Error: \n i=%d \n j=%d \n a[j]=%d \n b[j]=%d \n res_ser[j]=%d \n res_par[j]=%d\n", i, j, a[j], b[j], res_par[j]);
                exit(1);
            }
        }
        // second 128-bit lane
        for(int j=8; j<16; j++) {
            uint16_t res_ser = a[b[j]+8];
            if(res_par[j] != res_ser) {
                printf("Error: \n i=%d \n j=%d \n a[j]=%d \n b[j]=%d \n res_ser[j]=%d \n res_par[j]=%d\n", i, j, a[j], b[j], res_par[j]);
                exit(1);
            }
        }
    }
}

void TEST_RAND_mm256_cmov_epu16(uint64_t random_tests) {
    printf("TEST_RAND_mm256_cmov_epu16 [%lld]\n", random_tests);
    uint16_t tv[16];
    uint16_t fv[16];
    uint16_t cond[16];
    uint16_t res_par[16];
    for(uint64_t i = 0; i < random_tests; i++) {
        rand_arr_16xL_modM(tv, 16, UINT16_MAX);
        rand_arr_16xL_modM(fv, 16, UINT16_MAX);
        rand_arr_16xL_modM(cond, 16, 2);
        __m256i tv_256 = _mm256_loadu_si256((__m256i*)tv);
        __m256i fv_256 = _mm256_loadu_si256((__m256i*)fv);
        __m256i cond_256 = _mm256_loadu_si256((__m256i*)cond);
        __m256i res_256 = mm256_cmov_epu16(cond_256, tv_256, fv_256);
        _mm256_storeu_si256((__m256i*)res_par, res_256);
        for(int j=0; j<16; j++) {
            uint16_t res_ser = FP_ELEM_CMOV(cond[j], tv[j], fv[j]);
            assert( res_ser == (cond[j] ? tv[j] : fv[j]) );
            if(res_par[j] != res_ser) {
                printf("Error: \n i=%d \n j=%d \n tv[j]=%d \n fv[j]=%d \n cond[j]=%d \n res_ser[j]=%d \n res_par[j]=%d\n", i, j, tv[j], fv[j], cond[j], res_ser, res_par[j]);
                exit(1);
            }
        }
    }

}

void TEST_RAND_fp_vec_by_restr_vec_scaled_PAR(uint64_t random_tests) {
    printf("TEST_RAND_fp_vec_by_restr_vec_scaled_PAR [%lld]\n", random_tests);
    uint16_t r_ser[N];
    uint16_t r_par[N];
    uint8_t vr[N];
    uint16_t vn[N];
    uint16_t el; 
    for(uint64_t i = 0; i < random_tests; i++) {
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

void TEST_RAND_convert_restr_vec_to_fp_PAR(uint64_t random_tests) {
    printf("TEST_RAND_convert_restr_vec_to_fp_PAR [%lld]\n", random_tests);
    uint16_t res_ser[N];
    uint16_t res_par[N];
    uint8_t exponents[N];
    for(uint64_t i = 0; i < random_tests; i++) {
        rand_arr_8xN_1to127(exponents);
        convert_restr_vec_to_fp(res_ser, exponents);
        convert_restr_vec_to_fp_PAR(res_par, exponents);
        for(int j=0; j<N; j++){
            if(res_ser[j] != res_par[j]){
                printf("Error: \n i=%d \n j=%d \n exponents[j]=%d \n res_ser[j]=%d \n res_par[j]=%d\n", i, j, exponents[j], res_ser[j], res_par[j]);
                exit(1);
            }
        }
    }
}

void TEST_RAND_fp_vec_by_fp_matrix_PAR(uint64_t random_tests) {
    printf("TEST_RAND_fp_vec_by_fp_matrix_PAR [%lld]\n", random_tests);
    // void fp_vec_by_fp_matrix    (FP_ELEM res[N-K], FP_ELEM e[N], FP_ELEM       V_tr[K][N-K]){
    // void fp_vec_by_fp_matrix_PAR(FP_ELEM res[N-K], FP_ELEM e[N], FP_DOUBLEPREC V_tr[K][ROUND_UP(N-K,EPI32_PER_REG)]){
    uint16_t res_ser[N-K];
    uint16_t res_par[N-K];
    uint16_t vec[N];
    uint16_t mat_ser[K][N-K];
    uint32_t mat_par[K][ROUND_UP(N-K,EPI32_PER_REG)];
    for(uint64_t i = 0; i < random_tests; i++) {
        rand_arr_16xN_mod509(vec);
        for(int row=0; row<K; row++) {
            rand_arr_16xL_modM(mat_ser[row], N-K, 509);
        }
        for(int row = 0; row < K; row++){
            for (int col = 0; col < N-K; col++){
                mat_par[row][col] = mat_ser[row][col];
            }
        }
        fp_vec_by_fp_matrix(res_ser, vec, mat_ser);
        fp_vec_by_fp_matrix_PAR(res_par, vec, mat_par);
        for(int j=0; j<N-K; j++){
            if(res_ser[j] != res_par[j]){
                printf("Error: \n i=%d \n j=%d \n vec[j]=%d \n res_ser[j]=%d \n res_par[j]=%d\n", i, j, vec[j], res_ser[j], res_par[j]);
                exit(1);
            }
        }
    }
}

/******************************************************************************/

/*
                                    ITEERATIVE      RANDOM
    mm256_mod509_epu32              x               x
    mm256_mod509_epu16              x               x
    mm256_shuffle_epi16                             x
    mm256_cmov_epu16                                x
    mm256_mulmod509_epu16           x               x
    mm256_exp16mod509_epu16         x               x
    fp_vec_by_restr_vec_scaled                      x
    fp_vec_by_fp_matrix                             x
    convert_restr_vec_to_fp                         x
*/

/******************************************************************************/

#define TESTS 100000

int main() {

    srand(time(0)); // non reproducible
    srand(0);       // reproducible

    TEST_mm256_mod509_epu32();
    TEST_mm256_mod509_epu16();
    TEST_mm256_mulmod509_epu16();
    TEST_mm256_exp16mod509_epu16();
    
    printf("\n");
    
    TEST_RAND_mm256_mod509_epu32(TESTS);
    TEST_RAND_mm256_mod509_epu16(TESTS);
    TEST_RAND_mm256_mulmod509_epu16(TESTS);
    TEST_RAND_mm256_exp16mod509_epu16(TESTS);
    TEST_RAND_mm256_shuffle_epi16(TESTS);
    TEST_RAND_mm256_cmov_epu16(TESTS);
    TEST_RAND_convert_restr_vec_to_fp_PAR(TESTS);
    TEST_RAND_fp_vec_by_restr_vec_scaled_PAR(TESTS);
    TEST_RAND_fp_vec_by_fp_matrix_PAR(TESTS);

    printf("\nOK\n");

    return 0;
}

/*
rm -f RSDPG_avx2_arith_test.o; gcc -o RSDPG_avx2_arith_test.o RSDPG_avx2_arith_test.c -march=native -g3 -fsanitize=address; ./RSDPG_avx2_arith_test.o
*/

