#include <assert.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

__m256i mm256_mod509_epu32(__m256i a){
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

__m256i mm256_mod509_epu16(__m256i a){
    /* r = min(r, r - P) */
    __m256i p_256 = _mm256_set1_epi16(509);
    __m256i as= _mm256_sub_epi16(a, p_256);
    return _mm256_min_epu16(a, as);
    // works only up to 508*2 !!!
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
    u0 = mm256_mod509_epu32(u0);
    u1 = mm256_mod509_epu32(u1);
    /* pack 32-bit to 16-bit */
    __m256i r = _mm256_packs_epi32(u0, u1);
    return r;
}

__m256i mm256_exp16mod509_epu16(__m256i a) {
    
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


/******************************************************************************/

#define N (106)
#define EPI16_PER_REG 16
#define ROUND_UP(amount, round_amt) ( ((amount+round_amt-1)/round_amt)*round_amt )
#define FP_ELEM uint16_t
#define FZ_ELEM uint8_t
#define FP_DOUBLEPREC uint32_t
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

/******************************************************************************/

void fp_vec_by_restr_vec_scaled_ser(FP_ELEM res[N],
                                const FZ_ELEM e[N],
                                const FP_ELEM chall_1,
                                const FP_ELEM u_prime[N]){
    for(int i = 0; i < N; i++){
        res[i] = FPRED_SINGLE( (FP_DOUBLEPREC) u_prime[i] + (FP_DOUBLEPREC) RESTR_TO_VAL(e[i]) * (FP_DOUBLEPREC) chall_1) ;
    }
}

/******************************************************************************/

// OK
void fp_vec_by_restr_vec_scaled_par(uint16_t r[N], const uint8_t vr[N], const uint16_t el, const uint16_t vn[N]) {

    // r: expand, align
    alignas(32) uint16_t r_x[ROUND_UP(N,EPI16_PER_REG)];
    // vr: convert from uint8 to uint16, expand, align
    alignas(32) uint16_t vr_x[ROUND_UP(N,EPI16_PER_REG)];
    for (int i = 0; i < N; i++) {
        vr_x[i] = vr[i];
    }
    // el: convert to m256i
    __m256i el_256 = _mm256_set1_epi16(el);
    // vn: expand, align
    alignas(32) uint16_t vn_x[ROUND_UP(N,EPI16_PER_REG)];
    memcpy(vn_x, vn, N*sizeof(uint16_t));

    // r = vn + RTV(vr) * el
    for(int i = 0; i < ROUND_UP(N,EPI16_PER_REG)/EPI16_PER_REG; i++ ){
        
        __m256i vn_256 = _mm256_load_si256( (__m256i const *) &vn_x[i*EPI16_PER_REG] );
        __m256i vr_256 = _mm256_load_si256( (__m256i const *) &vr_x[i*EPI16_PER_REG] );
        
        __m256i r_256;
        r_256 = mm256_exp16mod509_epu16(vr_256);
        r_256 = mm256_mulmod509_epu16_B(r_256, el_256);
        r_256 = _mm256_add_epi16(r_256, vn_256);
        r_256 = mm256_mod509_epu16(r_256);

        _mm256_store_si256 ((__m256i *) &r_x[i*EPI16_PER_REG], r_256);
    }
    memcpy(r, r_x, N*sizeof(uint16_t));
}

/******************************************************************************/

static inline void rand_arr_16xN_mod509(uint16_t *a) {
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 509;
    }
}


////////////////////////////// +1 ???
static inline void rand_arr_8xN_mod127(uint8_t *a) {
    for (int i = 0; i < N; i++) {
        a[i] = (rand() % 128) + 1;
    }
}

/******************************************************************************/

#define TESTS 10000

int main() {

    uint16_t r_ser[N];
    uint16_t r_par[N];
    uint8_t vr[N];
    uint16_t vn[N];
    uint16_t el; 
    
    srand(time(0));
    srand(0);
    
    for(int i = 0; i < TESTS; i++) {
        
        rand_arr_8xN_mod127(vr);
        rand_arr_16xN_mod509(vn);
        el = rand() % 509;
        
        // memset(vr, 127, N*sizeof(uint8_t));
        // memset(vn, 0, N*sizeof(uint16_t));
        // el = 2;
        
        fp_vec_by_restr_vec_scaled_ser(r_ser, vr, el, vn);
        fp_vec_by_restr_vec_scaled_par(r_par, vr, el, vn);

        for(int j=0; j<N; j++){
            if(r_ser[j] != r_par[j]){
                printf("Error: \n i=%d \n j=%d \n vr[j]=%d \n vn[j]=%d \n el=%d \n r_ser[j]=%d \n r_par[j]=%d\n", i, j, vr[j], vn[j], el, r_ser[j], r_par[j]);
                exit(1);
            }
        }
    
    }

    printf("\nOK\n");

    return 0;
}

/*
rm -f vbv16.o; gcc -o vbv16.o vbv16.c -mavx2 -lm -fsanitize=address; ./vbv16.o
*/

