use crate::poly::{
    monomial::{FastMonomial, Monomial, MonomialOrd},
    polynomial::{FastPolynomial, Polynomial},
};
use ark_ff::{Fp, FpConfig, PrimeField};
use flint_sys::{
    deps::mp_limb_signed_t,
    flint::{flint_get_num_threads, flint_set_num_threads},
    nmod_poly::{
        nmod_poly_clear, nmod_poly_evaluate_nmod, nmod_poly_get_coeff_ui, nmod_poly_init,
        nmod_poly_length, nmod_poly_set_coeff_ui, nmod_poly_struct,
    },
    nmod_poly_factor::{
        nmod_poly_factor_clear, nmod_poly_factor_init, nmod_poly_factor_struct,
        nmod_poly_factor_with_kaltofen_shoup,
    },
};
use std::mem::MaybeUninit;

pub fn set_flint_num_threads(num: i32) {
    unsafe {
        flint_set_num_threads(num);
    }
}

pub fn get_flint_num_threads() -> i32 {
    unsafe { flint_get_num_threads() }
}

pub fn factor<P: FpConfig<1>, const NM: usize, O: MonomialOrd>(
    input: &FastPolynomial<Fp<P, 1>, FastMonomial<NM, O>>,
) -> Vec<FastPolynomial<Fp<P, 1>, FastMonomial<NM, O>>> {
    unsafe {
        let mut poly = MaybeUninit::<nmod_poly_struct>::uninit();
        nmod_poly_init(poly.as_mut_ptr(), P::MODULUS.as_ref()[0]);
        let mut poly = poly.assume_init();

        let mut factors = MaybeUninit::<nmod_poly_factor_struct>::uninit();
        nmod_poly_factor_init(factors.as_mut_ptr());
        let mut factors = factors.assume_init();

        let mut i: u16 = 0;
        for (c, x) in input.terms() {
            for j in i..x.degree() {
                nmod_poly_set_coeff_ui(&mut poly, j as mp_limb_signed_t, 0);
            }
            nmod_poly_set_coeff_ui(
                &mut poly,
                x.degree() as mp_limb_signed_t,
                c.into_bigint().as_ref()[0],
            );
            i = x.degree() + 1;
        }

        nmod_poly_factor_with_kaltofen_shoup(&mut factors, &mut poly);

        let var = input
            .leading_monomial()
            .unwrap()
            .iter()
            .position(|&x| x > 0)
            .unwrap();
        let mut result = Vec::new();
        for i in 0..factors.num as usize {
            let p = factors.p.add(i);
            result.push(FastPolynomial::new(
                input.num_of_vars,
                &(0..nmod_poly_length(p) as usize)
                    .map(|j| {
                        (
                            Fp::from(nmod_poly_get_coeff_ui(p, j as mp_limb_signed_t)),
                            FastMonomial::new(&[(var, j as u16)]),
                        )
                    })
                    .collect::<Vec<(Fp<P, 1>, FastMonomial<NM, O>)>>(),
            ));
        }

        nmod_poly_clear(&mut poly);
        nmod_poly_factor_clear(&mut factors);

        result.sort_unstable_by(|a, b| {
            a.leading_monomial()
                .unwrap()
                .cmp(&b.leading_monomial().unwrap())
        });

        result
    }
}

pub fn evaluate<P: FpConfig<1>, const NM: usize, O: MonomialOrd>(
    input: &[(Fp<P, 1>, FastMonomial<NM, O>)],
    point: Fp<P, 1>,
) -> Fp<P, 1> {
    unsafe {
        let mut poly = MaybeUninit::<nmod_poly_struct>::uninit();
        nmod_poly_init(poly.as_mut_ptr(), P::MODULUS.as_ref()[0]);
        let mut poly = poly.assume_init();

        let mut i: u16 = 0;
        for (c, x) in input {
            for j in i..x.degree() {
                nmod_poly_set_coeff_ui(&mut poly, j as mp_limb_signed_t, 0);
            }
            nmod_poly_set_coeff_ui(
                &mut poly,
                x.degree() as mp_limb_signed_t,
                c.into_bigint().as_ref()[0],
            );
            i = x.degree() + 1;
        }

        let result = nmod_poly_evaluate_nmod(&mut poly, point.into_bigint().as_ref()[0]);

        nmod_poly_clear(&mut poly);

        Fp::from(result)
    }
}

#[cfg(test)]
#[allow(clippy::all)]
mod tests {
    use super::*;
    use crate::{
        poly::{
            monomial::{FastMonomial, Monomial},
            polynomial::{FastPolynomial, Polynomial},
        },
        LexPolynomial, GF,
    };

    #[test]
    fn test_factor() {
        let input: LexPolynomial<1> = FastPolynomial::new(
            4,
            &[
                (GF::from(1), FastMonomial::new(&[(1, 4)])),
                (GF::from(-1 as i64), FastMonomial::one()),
            ],
        );

        let mut result = factor(&input);
        result.sort_unstable_by(|a, b| {
            a.terms()
                .first()
                .unwrap()
                .0
                .cmp(&b.terms.first().unwrap().0)
        });
        let want = vec![
            FastPolynomial::new(
                4,
                &[
                    (GF::from(1), FastMonomial::new(&[(1, 1)])),
                    (GF::from(1), FastMonomial::one()),
                ],
            ),
            FastPolynomial::new(
                4,
                &[
                    (GF::from(1), FastMonomial::new(&[(1, 1)])),
                    (GF::from(2296021864060584341u64), FastMonomial::one()),
                ],
            ),
            FastPolynomial::new(
                4,
                &[
                    (GF::from(1), FastMonomial::new(&[(1, 1)])),
                    (GF::from(16150722209648967216u64), FastMonomial::one()),
                ],
            ),
            FastPolynomial::new(
                4,
                &[
                    (GF::from(1), FastMonomial::new(&[(1, 1)])),
                    (GF::from(18446744073709551556u64), FastMonomial::one()),
                ],
            ),
        ];
        assert_eq!(result, want);
    }

    #[test]
    fn test_evaluate() {
        let poly: LexPolynomial<1> = LexPolynomial::new(
            1,
            &vec![
                // x^7 + 9*x^6 + x^2 + 3*x + 1
                (1.into(), FastMonomial::new(&[(0, 7)])),
                (9.into(), FastMonomial::new(&[(0, 6)])),
                (1.into(), FastMonomial::new(&[(0, 2)])),
                (3.into(), FastMonomial::new(&[(0, 1)])),
                (1.into(), FastMonomial::new(&[(0, 0)])),
            ],
        );
        let result = evaluate(poly.terms(), 2.into());
        assert_eq!(result, 715.into());
        let result = evaluate(poly.terms(), 100.into());
        assert_eq!(result, 109000000010301i64.into());

        let poly: LexPolynomial<1> = LexPolynomial::new(
            1,
            &vec![
                // x^7 + 9*x^6 + x^2
                (1.into(), FastMonomial::new(&[(0, 7)])),
                (9.into(), FastMonomial::new(&[(0, 6)])),
                (1.into(), FastMonomial::new(&[(0, 2)])),
            ],
        );
        let result = evaluate(poly.terms(), 3.into());
        assert_eq!(result, 8757.into());
        let result = evaluate(poly.terms(), 100.into());
        assert_eq!(result, 109000000010000i64.into());

        let poly: LexPolynomial<1> = LexPolynomial::new(
            1,
            &vec![
                // x^7 + 9*x^6 + x^2 + 3*x + 1
                (1.into(), FastMonomial::new(&[(0, 7)])),
                (9.into(), FastMonomial::new(&[(0, 6)])),
                (9.into(), FastMonomial::new(&[(0, 5)])),
                (9.into(), FastMonomial::new(&[(0, 4)])),
                (9.into(), FastMonomial::new(&[(0, 3)])),
                (1.into(), FastMonomial::new(&[(0, 2)])),
                (3.into(), FastMonomial::new(&[(0, 1)])),
                (1.into(), FastMonomial::new(&[(0, 0)])),
            ],
        );
        let result = evaluate(poly.terms(), 7.into());
        assert_eq!(result, 2058414.into());
        let result = evaluate(poly.terms(), 19.into());
        assert_eq!(result, 1340804598.into());
    }
}
