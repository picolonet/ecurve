#ifndef CUDA_MPMA_HAND_CU
#define CUDA_MPMA_HAND_CU

#include "primitives.cu"
#include "subwarp.cu"

/*
 * A "hand" encapsulates our notion of a fixed collection of digits in
 * a multiprecision number.
 *
 * A hand is stored in a warp register, and so implicitly consumes 32
 * times the size of the register.  The interface can support
 * different underlying representations however, including a "full
 * utilisation always handle carries" implementation, a "avoid carries
 * with nail bits" implementation, as well as more exotic
 * implementations using floating-point registers.
 *
 * A limb is an array of digits stored in memory. Chunks of digits are
 * loaded into hands in order for arithmetic to be performed on
 * numbers. So a limb can be thought of as a sequence of hands.
 *
 * The interface to hands and the interface to limbs is (largely) the
 * same. The interface for limbs is defined in terms of the one for
 * hands.
 */

// A hand implementation
//
// Normally a hand_implementation will not be templatised by digit
// type, but in this case it works with uint32_t and uint64_t.
template< typename digit_tp, int WIDTH = WARPSIZE >
class full_hand {
    typedef subwarp<WIDTH> subwarp;

public:
    static constexpr int SLOT_WIDTH = WIDTH;
    static constexpr int NSLOTS = WARPSIZE / WIDTH;

    typedef digit_tp digit;
    static constexpr int DIGIT_BYTES = sizeof(digit);
    static constexpr int DIGIT_BITS = DIGIT_BYTES * 8;
    static constexpr int DIGITS_PER_HAND = WARPSIZE;
    static constexpr int HAND_BYTES = DIGITS_PER_HAND * DIGIT_BYTES;
    static constexpr int HAND_BITS = DIGITS_PER_HAND * DIGIT_BITS;
    static constexpr int FIXNUM_BYTES = SLOT_WIDTH * DIGIT_BYTES;

    //  +=  +  -  <<  >>  &=
    //  <  >

    // multiply-by-zero-or-one

    static __device__
    int
    add_cy(digit r[], const digit a[], const digit b[]) {
        int cy;
        int L = subwarp::laneIdx();

        r[L] = a[L] + b[L];
        cy = r[L] < a[L];

        return resolve_carries(r[L], cy);
    }

    /*
     * r = lo_half(a * b) with subwarp size width.
     *
     * The "lo_half" is the product modulo 2^(???), i.e. the same size as
     * the inputs.
     */
    static __device__
    void
    mullo(digit &r, digit a, digit b) {
        // TODO: This should be smaller, probably uint16_t (smallest
        // possible for addition).  Strangely, the naive translation to
        // the smaller size broke; to investigate.
        digit cy = 0;

        r = 0;
        for (int i = WIDTH - 1; i >= 0; --i) {
            digit aa = subwarp::shfl(a, i);

            // TODO: See if using umad.wide improves this.
            umad_hi_cc(r, cy, aa, b, r);
            r = subwarp::shfl_up0(r, 1);
            cy = subwarp::shfl_up0(cy, 1);
            umad_lo_cc(r, cy, aa, b, r);
        }
        cy = subwarp::shfl_up0(cy, 1);
        (void) add_cy(r, r, cy);
    }

private:
    static constexpr digit DIGIT_MAX = ~(digit)0;

    static __device__
    int
    resolve_carries(digit &r, int cy) {
        int L = subwarp::laneIdx();
        uint32_t allcarries, p, g;
        int cy_hi;

        g = subwarp::ballot(cy);                  // carry generate
        p = subwarp::ballot(r == DIGIT_MAX);      // carry propagate
        allcarries = (p | g) + g;                 // propagate all carries
        cy_hi = allcarries < g;                   // detect final overflow
        allcarries = (allcarries ^ p) | (g << 1); // get effective carries
        r += (allcarries >> L) & 1;

        // return highest carry
        return cy_hi;
    }
};

#endif
