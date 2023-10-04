#![feature(portable_simd)]
#![feature(option_as_slice)]
#![feature(arc_unwrap_or_clone)]
#![allow(non_snake_case)]
#![allow(clippy::new_ret_no_self)]
#![allow(clippy::needless_return)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::upper_case_acronyms)]

extern crate core;

pub mod factor;
pub mod fglm;
pub mod groebner;
pub mod gvw;
pub mod poly;

use crate::poly::{monomial::*, polynomial::*};
use ark_ff::fields::{Fp64, MontBackend, MontConfig};
use std::cmp;

#[derive(Debug, Clone)]
pub struct Entry<L, R>(pub L, pub R);
impl<L: PartialEq, R> PartialEq for Entry<L, R> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<L: Eq, R> Eq for Entry<L, R> {}
impl<L: PartialOrd, R> PartialOrd for Entry<L, R> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<L: Ord, R> Ord for Entry<L, R> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

unsafe impl<L: Sync, R: Sync> Sync for Entry<L, R> {}

#[derive(MontConfig)]
#[modulus = "18446744073709551557"]
#[generator = "2"]
pub struct FqConfig18446744073709551557;

pub type GF = Fp64<MontBackend<FqConfig18446744073709551557, 1>>;

pub type LexPolynomial<const N: usize> = FastPolynomial<GF, FastMonomial<N, LexOrder>>;
pub type DegRevLexPolynomial<const N: usize> = FastPolynomial<GF, FastMonomial<N, DegRevLexOrder>>;

#[cfg(test)]
#[allow(clippy::all)]
mod tests {
    use super::*;
    use ark_ff::{One, Zero};

    #[test]
    fn test_div_mod_polys() {
        let f: LexPolynomial<1> = FastPolynomial::new(
            2,
            &vec![
                ((1).into(), FastMonomial::new(&vec![(0, 5)])),
                ((1).into(), FastMonomial::new(&vec![(0, 1)])),
            ],
        );

        let polys: Vec<LexPolynomial<1>> = vec![
            FastPolynomial::new(
                2,
                &vec![
                    ((1).into(), FastMonomial::new(&vec![(0, 2)])),
                    ((-1).into(), FastMonomial::new(&vec![(1, 3)])),
                ],
            ),
            FastPolynomial::new(
                2,
                &vec![
                    ((1).into(), FastMonomial::new(&vec![(0, 1), (1, 2)])),
                    ((1).into(), FastMonomial::new(&vec![(0, 1)])),
                ],
            ),
        ];

        let (_, r) = f.div_mod_polys(&polys);

        assert!(r.is_zero());
    }

    #[test]
    fn test_lex_order() {
        let m1: FastMonomial<1, LexOrder> = FastMonomial::new(&vec![
            (0, 3),
            (1, 0),
            (2, 2),
            (3, 0),
            (4, 1),
            (5, 2),
            (6, 0),
            (7, 1),
        ]);
        let m18: FastMonomial<1, LexOrder> = FastMonomial::new(&vec![
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 0),
            (7, 2),
        ]);
        assert!(m1 > m18);
        let test: FastPolynomial<GF, FastMonomial<1, LexOrder>> = FastPolynomial::new(
            8,
            &vec![
                (
                    GF::from(1),
                    FastMonomial::new(&vec![
                        (0, 3),
                        (1, 0),
                        (2, 2),
                        (3, 0),
                        (4, 1),
                        (5, 2),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(2),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 1),
                        (2, 1),
                        (3, 0),
                        (4, 0),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(3),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 0),
                        (2, 3),
                        (3, 0),
                        (4, 1),
                        (5, 1),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(4),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 0),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 0),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(5),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 3),
                        (2, 2),
                        (3, 1),
                        (4, 0),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(6),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 1),
                        (2, 1),
                        (3, 2),
                        (4, 1),
                        (5, 1),
                        (6, 0),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(7),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 0),
                        (2, 2),
                        (3, 1),
                        (4, 0),
                        (5, 1),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(8),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 1),
                        (2, 0),
                        (3, 0),
                        (4, 2),
                        (5, 1),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(9),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 0),
                        (2, 1),
                        (3, 1),
                        (4, 0),
                        (5, 1),
                        (6, 2),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(10),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 1),
                        (2, 0),
                        (3, 1),
                        (4, 1),
                        (5, 0),
                        (6, 0),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(11),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 0),
                        (2, 1),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(12),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 1),
                        (3, 1),
                        (4, 0),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(13),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 1),
                        (2, 2),
                        (3, 0),
                        (4, 2),
                        (5, 1),
                        (6, 0),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(14),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 0),
                        (2, 1),
                        (3, 2),
                        (4, 1),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(15),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 2),
                        (2, 0),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(16),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 1),
                        (2, 2),
                        (3, 0),
                        (4, 0),
                        (5, 2),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(17),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 0),
                        (2, 1),
                        (3, 1),
                        (4, 2),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(18),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 0),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 0),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(19),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 1),
                        (2, 1),
                        (3, 0),
                        (4, 1),
                        (5, 2),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(20),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 0),
                        (2, 2),
                        (3, 1),
                        (4, 0),
                        (5, 0),
                        (6, 2),
                        (7, 1),
                    ]),
                ),
            ],
        );
        let want: Vec<u64> = vec![
            1, 8, 3, 11, 12, 18, 4, 16, 6, 2, 20, 14, 9, 5, 15, 13, 19, 10, 7, 17,
        ];
        assert_eq!(
            test.terms()
                .iter()
                .rev()
                .map(|(coeff, _)| *coeff)
                .collect::<Vec<GF>>(),
            want.iter().map(|x| GF::from(*x)).collect::<Vec<GF>>()
        );

        let test: FastPolynomial<GF, FastMonomial<1, LexOrder>> = FastPolynomial::new(
            8,
            &vec![
                (
                    GF::from(1),
                    FastMonomial::new(&vec![
                        (0, 3),
                        (1, 2),
                        (2, 1),
                        (3, 0),
                        (4, 2),
                        (5, 1),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(2),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 1),
                        (2, 3),
                        (3, 0),
                        (4, 1),
                        (5, 2),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(3),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 3),
                        (2, 0),
                        (3, 2),
                        (4, 0),
                        (5, 1),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(4),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 2),
                        (2, 2),
                        (3, 1),
                        (4, 0),
                        (5, 0),
                        (6, 1),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(5),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 0),
                        (2, 1),
                        (3, 3),
                        (4, 1),
                        (5, 0),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(6),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 1),
                        (2, 0),
                        (3, 2),
                        (4, 2),
                        (5, 1),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(7),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 0),
                        (2, 2),
                        (3, 1),
                        (4, 1),
                        (5, 2),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(8),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 1),
                        (3, 0),
                        (4, 0),
                        (5, 1),
                        (6, 2),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(9),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 1),
                        (2, 0),
                        (3, 3),
                        (4, 2),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(10),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 3),
                        (2, 2),
                        (3, 0),
                        (4, 1),
                        (5, 1),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
            ],
        );
        let want: Vec<u64> = vec![1, 2, 9, 5, 3, 8, 6, 10, 4, 7];
        assert_eq!(
            test.terms()
                .iter()
                .rev()
                .map(|(coeff, _)| *coeff)
                .collect::<Vec<GF>>(),
            want.iter().map(|x| GF::from(*x)).collect::<Vec<GF>>()
        );

        let test: FastPolynomial<GF, FastMonomial<1, LexOrder>> = FastPolynomial::new(
            8,
            &vec![
                (
                    GF::from(3),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 3),
                        (2, 1),
                        (3, 0),
                        (4, 1),
                        (5, 2),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(5),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 0),
                        (3, 3),
                        (4, 0),
                        (5, 1),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(7),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 1),
                        (2, 2),
                        (3, 1),
                        (4, 0),
                        (5, 0),
                        (6, 1),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(9),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 0),
                        (2, 1),
                        (3, 2),
                        (4, 1),
                        (5, 0),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(11),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 1),
                        (2, 0),
                        (3, 1),
                        (4, 2),
                        (5, 1),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(13),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 0),
                        (2, 2),
                        (3, 0),
                        (4, 1),
                        (5, 2),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(15),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 1),
                        (3, 0),
                        (4, 0),
                        (5, 1),
                        (6, 2),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(17),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 3),
                        (2, 2),
                        (3, 0),
                        (4, 1),
                        (5, 1),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(19),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 1),
                        (2, 0),
                        (3, 2),
                        (4, 2),
                        (5, 1),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(21),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 0),
                        (2, 2),
                        (3, 1),
                        (4, 1),
                        (5, 2),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
            ],
        );
        let want: Vec<u64> = vec![3, 13, 15, 5, 19, 9, 17, 7, 11, 21];
        assert_eq!(
            test.terms()
                .iter()
                .rev()
                .map(|(coeff, _)| *coeff)
                .collect::<Vec<GF>>(),
            want.iter().map(|x| GF::from(*x)).collect::<Vec<GF>>()
        );
    }

    #[test]
    fn test_deg_rev_lex_order_0() {
        let f1: FastMonomial<1, DegRevLexOrder> = FastMonomial::new(&vec![
            (0, 3),
            (1, 0),
            (2, 2),
            (3, 0),
            (4, 1),
            (5, 2),
            (6, 0),
            (7, 1),
        ]);
        let f3: FastMonomial<1, DegRevLexOrder> = FastMonomial::new(&vec![
            (0, 2),
            (1, 0),
            (2, 3),
            (3, 0),
            (4, 1),
            (5, 1),
            (6, 2),
            (7, 0),
        ]);
        assert!(f3 > f1);
        let test: FastPolynomial<GF, FastMonomial<1, DegRevLexOrder>> = FastPolynomial::new(
            8,
            &vec![
                (
                    GF::from(1),
                    FastMonomial::new(&vec![
                        (0, 3),
                        (1, 0),
                        (2, 2),
                        (3, 0),
                        (4, 1),
                        (5, 2),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(2),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 1),
                        (2, 1),
                        (3, 0),
                        (4, 0),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(3),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 0),
                        (2, 3),
                        (3, 0),
                        (4, 1),
                        (5, 1),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(4),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 0),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 0),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(5),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 3),
                        (2, 2),
                        (3, 1),
                        (4, 0),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(6),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 1),
                        (2, 1),
                        (3, 2),
                        (4, 1),
                        (5, 1),
                        (6, 0),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(7),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 0),
                        (2, 2),
                        (3, 1),
                        (4, 0),
                        (5, 1),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(8),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 1),
                        (2, 0),
                        (3, 0),
                        (4, 2),
                        (5, 1),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(9),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 0),
                        (2, 1),
                        (3, 1),
                        (4, 0),
                        (5, 1),
                        (6, 2),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(10),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 1),
                        (2, 0),
                        (3, 1),
                        (4, 1),
                        (5, 0),
                        (6, 0),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(11),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 0),
                        (2, 1),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(12),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 1),
                        (3, 1),
                        (4, 0),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(13),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 1),
                        (2, 2),
                        (3, 0),
                        (4, 2),
                        (5, 1),
                        (6, 0),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(14),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 0),
                        (2, 1),
                        (3, 2),
                        (4, 1),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(15),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 2),
                        (2, 0),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(16),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 1),
                        (2, 2),
                        (3, 0),
                        (4, 0),
                        (5, 2),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(17),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 0),
                        (2, 1),
                        (3, 1),
                        (4, 2),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(18),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 0),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 0),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(19),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 1),
                        (2, 1),
                        (3, 0),
                        (4, 1),
                        (5, 2),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(20),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 0),
                        (2, 2),
                        (3, 1),
                        (4, 0),
                        (5, 0),
                        (6, 2),
                        (7, 1),
                    ]),
                ),
            ],
        );
        let want: Vec<u64> = vec![
            3, 1, 5, 16, 18, 6, 11, 8, 12, 14, 20, 9, 4, 13, 15, 19, 17, 7, 2, 10,
        ];
        assert_eq!(
            test.terms()
                .iter()
                .rev()
                .map(|(coeff, _)| *coeff)
                .collect::<Vec<GF>>(),
            want.iter().map(|x| GF::from(*x)).collect::<Vec<GF>>()
        );

        let test: FastPolynomial<GF, FastMonomial<1, DegRevLexOrder>> = FastPolynomial::new(
            8,
            &vec![
                (
                    GF::from(1),
                    FastMonomial::new(&vec![
                        (0, 3),
                        (1, 2),
                        (2, 1),
                        (3, 0),
                        (4, 2),
                        (5, 1),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(2),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 1),
                        (2, 3),
                        (3, 0),
                        (4, 1),
                        (5, 2),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(3),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 3),
                        (2, 0),
                        (3, 2),
                        (4, 0),
                        (5, 1),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(4),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 2),
                        (2, 2),
                        (3, 1),
                        (4, 0),
                        (5, 0),
                        (6, 1),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(5),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 0),
                        (2, 1),
                        (3, 3),
                        (4, 1),
                        (5, 0),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(6),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 1),
                        (2, 0),
                        (3, 2),
                        (4, 2),
                        (5, 1),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(7),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 0),
                        (2, 2),
                        (3, 1),
                        (4, 1),
                        (5, 2),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(8),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 1),
                        (3, 0),
                        (4, 0),
                        (5, 1),
                        (6, 2),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(9),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 1),
                        (2, 0),
                        (3, 3),
                        (4, 2),
                        (5, 0),
                        (6, 1),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(10),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 3),
                        (2, 2),
                        (3, 0),
                        (4, 1),
                        (5, 1),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
            ],
        );
        let want: Vec<u64> = vec![1, 2, 9, 5, 3, 10, 8, 6, 4, 7];
        assert_eq!(
            test.terms()
                .iter()
                .rev()
                .map(|(coeff, _)| *coeff)
                .collect::<Vec<GF>>(),
            want.iter().map(|x| GF::from(*x)).collect::<Vec<GF>>()
        );

        let test: FastPolynomial<GF, FastMonomial<1, DegRevLexOrder>> = FastPolynomial::new(
            8,
            &vec![
                (
                    GF::from(3),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 3),
                        (2, 1),
                        (3, 0),
                        (4, 1),
                        (5, 2),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(5),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 0),
                        (3, 3),
                        (4, 0),
                        (5, 1),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(7),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 1),
                        (2, 2),
                        (3, 1),
                        (4, 0),
                        (5, 0),
                        (6, 1),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(9),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 0),
                        (2, 1),
                        (3, 2),
                        (4, 1),
                        (5, 0),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(11),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 1),
                        (2, 0),
                        (3, 1),
                        (4, 2),
                        (5, 1),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(13),
                    FastMonomial::new(&vec![
                        (0, 2),
                        (1, 0),
                        (2, 2),
                        (3, 0),
                        (4, 1),
                        (5, 2),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(15),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 2),
                        (2, 1),
                        (3, 0),
                        (4, 0),
                        (5, 1),
                        (6, 2),
                        (7, 2),
                    ]),
                ),
                (
                    GF::from(17),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 3),
                        (2, 2),
                        (3, 0),
                        (4, 1),
                        (5, 1),
                        (6, 2),
                        (7, 0),
                    ]),
                ),
                (
                    GF::from(19),
                    FastMonomial::new(&vec![
                        (0, 1),
                        (1, 1),
                        (2, 0),
                        (3, 2),
                        (4, 2),
                        (5, 1),
                        (6, 0),
                        (7, 1),
                    ]),
                ),
                (
                    GF::from(21),
                    FastMonomial::new(&vec![
                        (0, 0),
                        (1, 0),
                        (2, 2),
                        (3, 1),
                        (4, 1),
                        (5, 2),
                        (6, 1),
                        (7, 0),
                    ]),
                ),
            ],
        );
        let want: Vec<u64> = vec![3, 5, 17, 15, 13, 19, 21, 9, 7, 11];
        assert_eq!(
            test.terms()
                .iter()
                .rev()
                .map(|(coeff, _)| *coeff)
                .collect::<Vec<GF>>(),
            want.iter().map(|x| GF::from(*x)).collect::<Vec<GF>>()
        );

        let test: FastPolynomial<GF, FastMonomial<1, DegRevLexOrder>> = FastPolynomial::new(
            8,
            &vec![
                (GF::from(3), FastMonomial::new(&vec![(0, 2)])),
                (GF::from(5), FastMonomial::new(&vec![(0, 2), (1, 1)])),
            ],
        );
        let want: Vec<u64> = vec![5, 3];
        assert_eq!(
            test.terms()
                .iter()
                .rev()
                .map(|(coeff, _)| *coeff)
                .collect::<Vec<GF>>(),
            want.iter().map(|x| GF::from(*x)).collect::<Vec<GF>>()
        );
    }

    #[test]
    fn test_mul_sparse_polynomial_by_sparse_term() {
        let term = FastMonomial::new(&vec![(0, 2)]);
        let polynomial: FastPolynomial<GF, FastMonomial<1, DegRevLexOrder>> = FastPolynomial::new(
            1,
            &vec![
                (1.into(), FastMonomial::new(&vec![(0, 1)])),
                (2.into(), FastMonomial::new(&vec![(0, 3)])),
            ],
        );

        let result = polynomial * &(GF::one(), term);
        let expected: FastPolynomial<GF, FastMonomial<1, DegRevLexOrder>> = FastPolynomial::new(
            1,
            &vec![
                (1.into(), FastMonomial::new(&vec![(0, 3)])),
                (2.into(), FastMonomial::new(&vec![(0, 5)])),
            ],
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_sparse_polynomial_by_field_element() {
        let field_element: GF = 2.into();
        let polynomial: FastPolynomial<GF, FastMonomial<1, DegRevLexOrder>> = FastPolynomial::new(
            1,
            &vec![
                (1.into(), FastMonomial::new(&vec![(0, 1)])),
                (2.into(), FastMonomial::new(&vec![(0, 3)])),
            ],
        );

        let result = polynomial * &(field_element, FastMonomial::one());
        let expected: FastPolynomial<GF, FastMonomial<1, DegRevLexOrder>> = FastPolynomial::new(
            1,
            &vec![
                (2.into(), FastMonomial::new(&vec![(0, 1)])),
                (4.into(), FastMonomial::new(&vec![(0, 3)])),
            ],
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_s_polynomial() {
        let poly1: FastPolynomial<GF, FastMonomial<1, DegRevLexOrder>> = FastPolynomial::new(
            1,
            &vec![
                (1.into(), FastMonomial::new(&vec![(0, 1)])),
                (2.into(), FastMonomial::new(&vec![(0, 3)])),
            ],
        );

        let poly2: FastPolynomial<GF, FastMonomial<1, DegRevLexOrder>> = FastPolynomial::new(
            1,
            &vec![
                (3.into(), FastMonomial::new(&vec![(0, 2)])),
                (4.into(), FastMonomial::new(&vec![(0, 4)])),
            ],
        );

        let result = poly1.s_polynomial(&poly2);
        let expected: FastPolynomial<GF, FastMonomial<1, DegRevLexOrder>> =
            FastPolynomial::new(1, &vec![((-2).into(), FastMonomial::new(&vec![(0, 2)]))]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_div_mod_polys_multivariate() {
        let f: LexPolynomial<1> = FastPolynomial::new(
            3,
            &vec![
                (1.into(), FastMonomial::new(&vec![(0, 2), (1, 0), (2, 0)])),
                (1.into(), FastMonomial::new(&vec![(0, 0), (1, 1), (2, 0)])),
            ],
        );

        let gs = vec![
            FastPolynomial::new(
                3,
                &vec![(1.into(), FastMonomial::new(&vec![(0, 1), (1, 0), (2, 0)]))],
            ),
            FastPolynomial::new(
                3,
                &vec![(1.into(), FastMonomial::new(&vec![(0, 0), (1, 1), (2, 0)]))],
            ),
            FastPolynomial::new(
                3,
                &vec![(1.into(), FastMonomial::new(&vec![(0, 0), (1, 0), (2, 1)]))],
            ),
        ];
        let (qs, r) = f.div_mod_polys(&gs);

        let expected_qs = vec![
            FastPolynomial::new(
                3,
                &vec![(1.into(), FastMonomial::new(&vec![(0, 1), (1, 0), (2, 0)]))],
            ),
            FastPolynomial::new(
                3,
                &vec![(1.into(), FastMonomial::new(&vec![(0, 0), (1, 0), (2, 0)]))],
            ),
            FastPolynomial::new(3, &vec![]),
        ];
        let expected_r = FastPolynomial::new(3, &vec![]);

        assert_eq!(qs, expected_qs);
        assert_eq!(r, expected_r);
    }
}
