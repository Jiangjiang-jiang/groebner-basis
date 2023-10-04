use crate::{
    groebner::reduce_groebner_basis,
    poly::{monomial::*, polynomial::*},
    Entry,
};
use ark_ff::{Field, Zero};
use std::{borrow::Borrow, cmp, collections::BinaryHeap};
use chrono::Local;

/// The sugar selection strategy for critical pairs
#[inline]
pub fn sugar<F: Field, O: MonomialOrd>(
    f: &FastPolynomial<F, FastMonomial<O>>,
    g: &FastPolynomial<F, FastMonomial<O>>,
) -> cmp::Reverse<usize> {
    let lm_f = f
        .leading_monomial()
        .unwrap_or_else(FastMonomial::<O>::default);
    let lm_g = g
        .leading_monomial()
        .unwrap_or_else(FastMonomial::<O>::default);
    let lcm_fg = lm_f.lcm(&lm_g);
    let total_f = f.degree() - lm_f.degree();
    let total_g = g.degree() - lm_g.degree();
    cmp::Reverse((total_f.max(total_g) + lcm_fg.degree()) as usize)
}

/// Buchberger algorithm with sugar strategy, coprimarity and syzygy criterion.
#[inline]
pub fn buchberger<F: Field, O: MonomialOrd>(ideal: &mut Vec<FastPolynomial<F, FastMonomial<O>>>) {
    println!(
        "{} Buchberger start",
        Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
    );
    buchberger_with(sugar, ideal);
    println!(
        "{} Buchberger end",
        Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
    );
    reduce_groebner_basis(ideal, true);
}

/// Buchberger algorithm with coprimarity and syzygy criterion,
/// which accepts selection strategy as a weighting function.
/// This function processes critical pairs in heavier-first manner.
pub fn buchberger_with<F: Field, O: MonomialOrd, W: Ord + Sync, WF>(
    calc_weight: WF,
    ideal: &mut Vec<FastPolynomial<F, FastMonomial<O>>>,
) where
    WF: Fn(&FastPolynomial<F, FastMonomial<O>>, &FastPolynomial<F, FastMonomial<O>>) -> W + Copy,
{
    let mut pairs = BinaryHeap::new();
    for i in 0..ideal.len() {
        for j in 0..i {
            // Registering ciritcal pairs, with a weight for selection strategy
            pairs.push(Entry(calc_weight(&ideal[i], &ideal[j]), (i, j)))
        }
    }
    let mut n = ideal.len();
    while let Some(Entry(_, (i, j))) = pairs.pop() {
        let (lt_f, lt_g) = (
            ideal[i].leading_monomial().unwrap(),
            ideal[j].leading_monomial().unwrap(),
        );
        let lcm_fg = lt_f.lcm(&lt_g);

        // Primarity check
        if lcm_fg == lt_f * &lt_g {
            continue;
        }

        // Syzygy test
        if ideal.iter().enumerate().any(|(k, g_k)| {
            let (b1, s1) = (i.max(k), i.min(k));
            let (b2, s2) = (j.max(k), j.min(k));
            k != i
                && k != j
                && (g_k.leading_monomial().unwrap() / &lcm_fg).is_some()
                && pairs
                    .iter()
                    .all(|Entry(_, (b, s))| (*b, *s) != (b1, s1) && (*b, *s) != (b2, s2))
        }) {
            continue;
        }

        // S-test
        let (_, s) = ideal[i]
            .s_polynomial(ideal[j].borrow())
            .div_mod_polys(ideal);
        if !s.is_zero() {
            ideal.push(s);
            for k in 0..n {
                pairs.push(Entry(calc_weight(&ideal[n], &ideal[k]), (n, k)));
            }
            n += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{groebner::is_groebner_basis, DegRevLexPolynomial, LexPolynomial};
    use ark_ff::{Fp, FpConfig, PrimeField, Zero};

    fn print_ideal<P: FpConfig<1>, O: MonomialOrd>(
        input_polynomials: &Vec<FastPolynomial<Fp<P, 1>, FastMonomial<O>>>,
    ) {
        let polys = input_polynomials
            .iter()
            .map(|poly| {
                poly.terms()
                    .rev()
                    .filter(|(c, _)| !c.is_zero())
                    .map(|(coeff, term)| {
                        if term.is_constant() {
                            format!("{}", coeff.into_bigint().0.as_slice()[0])
                        } else {
                            format!(
                                "{}{}",
                                coeff.into_bigint().0.as_slice()[0],
                                term.iter()
                                    .enumerate()
                                    .map(|(v, e)| {
                                        if e == 1 {
                                            format!("*x_{}", v)
                                        } else if e > 1 {
                                            format!("*x_{}^{}", v, e)
                                        } else {
                                            String::new()
                                        }
                                    })
                                    .reduce(|a, b| a + &b)
                                    .unwrap()
                            )
                        }
                    })
                    .reduce(|a, b| a + " + " + &b)
                    .unwrap()
            })
            .reduce(|a, b| a + "\n" + &b)
            .unwrap_or("".parse().unwrap());

        println!("Constructed polynomial: \n{}", polys);
    }

    #[test]
    fn test_div_mod_polys_paper_case() {
        let f: LexPolynomial = FastPolynomial::new(
            2,
            &vec![
                ((1).into(), FastMonomial::new(&vec![(0, 5)])),
                ((1).into(), FastMonomial::new(&vec![(0, 1)])),
            ],
        );

        let gs: Vec<LexPolynomial> = vec![
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

        let (_, h) = f.div_mod_polys(&gs);

        assert!(h.is_zero());
    }

    #[test]
    fn test_div_mod_polys_wiki_case() {
        let f: LexPolynomial = FastPolynomial::new(
            2,
            &vec![
                ((2).into(), FastMonomial::new(&vec![(0, 3)])),
                ((-1).into(), FastMonomial::new(&vec![(0, 2), (1, 1)])),
                ((1).into(), FastMonomial::new(&vec![(1, 3)])),
                ((3).into(), FastMonomial::new(&vec![(1, 1)])),
            ],
        );

        let gs: Vec<LexPolynomial> = vec![
            FastPolynomial::new(
                2,
                &vec![
                    ((1).into(), FastMonomial::new(&vec![(0, 2)])),
                    ((1).into(), FastMonomial::new(&vec![(1, 2)])),
                    ((-1).into(), FastMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                2,
                &vec![
                    ((1).into(), FastMonomial::new(&vec![(0, 1), (1, 1)])),
                    ((-2).into(), FastMonomial::new(&vec![])),
                ],
            ),
        ];

        let (_, h) = f.div_mod_polys(&gs);

        let possible1: LexPolynomial = FastPolynomial::new(
            2,
            &vec![
                ((1).into(), FastMonomial::new(&vec![(1, 3)])),
                ((-1).into(), FastMonomial::new(&vec![(1, 1)])),
            ],
        );

        let possible2: LexPolynomial = FastPolynomial::new(
            2,
            &vec![
                ((2).into(), FastMonomial::new(&vec![(0, 1)])),
                ((2).into(), FastMonomial::new(&vec![(1, 3)])),
                ((-2).into(), FastMonomial::new(&vec![(1, 1)])),
            ],
        );

        assert!(h == possible1 || h == possible2);
    }

    #[test]
    fn test_buchberger_given_case_1() {
        let mut ideal: Vec<DegRevLexPolynomial> = vec![
            FastPolynomial::new(
                3,
                &vec![
                    ((1).into(), FastMonomial::new(&vec![(1, 3)])),
                    ((1).into(), FastMonomial::new(&vec![(0, 2)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    ((1).into(), FastMonomial::new(&vec![(0, 2), (1, 1)])),
                    ((1).into(), FastMonomial::new(&vec![(0, 2)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    ((1).into(), FastMonomial::new(&vec![(0, 3)])),
                    ((-1).into(), FastMonomial::new(&vec![(0, 2)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    ((1).into(), FastMonomial::new(&vec![(2, 4)])),
                    ((-1).into(), FastMonomial::new(&vec![(0, 2)])),
                    ((-1).into(), FastMonomial::new(&vec![(1, 1)])),
                ],
            ),
        ];

        buchberger(&mut ideal);
        assert!(is_groebner_basis(&ideal));
        print_ideal(&ideal);
    }

    #[test]
    fn test_buchberger_given_case_2() {
        let mut ideal: Vec<DegRevLexPolynomial> = vec![
            FastPolynomial::new(
                8,
                &vec![
                    ((-1).into(), FastMonomial::new(&vec![(0, 1)])),
                    (
                        (9511602413006487524u64).into(),
                        FastMonomial::new(&vec![(2, 1)]),
                    ),
                    (
                        (720575940379279356u64).into(),
                        FastMonomial::new(&vec![(3, 1)]),
                    ),
                    (
                        (8214565720323784678u64).into(),
                        FastMonomial::new(&vec![(4, 1)]),
                    ),
                    ((15774967588931459619u64).into(), FastMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((-1).into(), FastMonomial::new(&vec![(1, 1)])),
                    (
                        (10376293541461622753u64).into(),
                        FastMonomial::new(&vec![(2, 1)]),
                    ),
                    (
                        (5188146770730811374u64).into(),
                        FastMonomial::new(&vec![(3, 1)]),
                    ),
                    (
                        (2882303761517117431u64).into(),
                        FastMonomial::new(&vec![(4, 1)]),
                    ),
                    ((17132433913710081944u64).into(), FastMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    (
                        (4611686018427387891u64).into(),
                        FastMonomial::new(&vec![(2, 1)]),
                    ),
                    (
                        (6917529027641081833u64).into(),
                        FastMonomial::new(&vec![(3, 1)]),
                    ),
                    (
                        (6917529027641081834u64).into(),
                        FastMonomial::new(&vec![(4, 1)]),
                    ),
                    ((2075905444283933401u64).into(), FastMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((8).into(), FastMonomial::new(&vec![(2, 3)])),
                    (
                        (18446744073709551543u64).into(),
                        FastMonomial::new(&vec![(3, 3)]),
                    ),
                    ((7).into(), FastMonomial::new(&vec![(4, 3)])),
                    (
                        (12754686725922488265u64).into(),
                        FastMonomial::new(&vec![(5, 3)]),
                    ),
                    (
                        (885273986127101979u64).into(),
                        FastMonomial::new(&vec![(5, 2), (6, 1)]),
                    ),
                    (
                        (15982377876049625016u64).into(),
                        FastMonomial::new(&vec![(5, 1), (6, 2)]),
                    ),
                    (
                        (18406871384039882698u64).into(),
                        FastMonomial::new(&vec![(6, 3)]),
                    ),
                    (
                        (8188001519396716513u64).into(),
                        FastMonomial::new(&vec![(5, 2), (7, 1)]),
                    ),
                    (
                        (18051588390779879373u64).into(),
                        FastMonomial::new(&vec![(5, 1), (6, 1), (7, 1)]),
                    ),
                    (
                        (14895699747993026510u64).into(),
                        FastMonomial::new(&vec![(6, 2), (7, 1)]),
                    ),
                    (
                        (10147823821302792159u64).into(),
                        FastMonomial::new(&vec![(5, 1), (7, 2)]),
                    ),
                    (
                        (9498592991426641890u64).into(),
                        FastMonomial::new(&vec![(6, 1), (7, 2)]),
                    ),
                    (
                        (1869547999219154938u64).into(),
                        FastMonomial::new(&vec![(7, 3)]),
                    ),
                    (
                        (6547817342896512130u64).into(),
                        FastMonomial::new(&vec![(5, 2)]),
                    ),
                    (
                        (9279799793654434575u64).into(),
                        FastMonomial::new(&vec![(5, 1), (6, 1)]),
                    ),
                    (
                        (1363756090648335788u64).into(),
                        FastMonomial::new(&vec![(6, 2)]),
                    ),
                    (
                        (10404383306642543815u64).into(),
                        FastMonomial::new(&vec![(5, 1), (7, 1)]),
                    ),
                    (
                        (1940303722204995108u64).into(),
                        FastMonomial::new(&vec![(6, 1), (7, 1)]),
                    ),
                    (
                        (6984199018677488094u64).into(),
                        FastMonomial::new(&vec![(7, 2)]),
                    ),
                    (
                        (14719088602915975115u64).into(),
                        FastMonomial::new(&vec![(5, 1)]),
                    ),
                    (
                        (17366754088523144755u64).into(),
                        FastMonomial::new(&vec![(6, 1)]),
                    ),
                    (
                        (6719810797959261327u64).into(),
                        FastMonomial::new(&vec![(7, 1)]),
                    ),
                    ((3806417358522808896u64).into(), FastMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((56).into(), FastMonomial::new(&vec![(2, 3)])),
                    (
                        (18446744073709551467u64).into(),
                        FastMonomial::new(&vec![(3, 3)]),
                    ),
                    ((35).into(), FastMonomial::new(&vec![(4, 3)])),
                    (
                        (16424627841020198849u64).into(),
                        FastMonomial::new(&vec![(5, 3)]),
                    ),
                    (
                        (14440792205163495398u64).into(),
                        FastMonomial::new(&vec![(5, 2), (6, 1)]),
                    ),
                    (
                        (1257630195943210991u64).into(),
                        FastMonomial::new(&vec![(5, 1), (6, 2)]),
                    ),
                    (
                        (3683381545235644407u64).into(),
                        FastMonomial::new(&vec![(6, 3)]),
                    ),
                    (
                        (3082713944935104499u64).into(),
                        FastMonomial::new(&vec![(5, 2), (7, 1)]),
                    ),
                    (
                        (6577507255774609391u64).into(),
                        FastMonomial::new(&vec![(5, 1), (6, 1), (7, 1)]),
                    ),
                    (
                        (13038483871191007189u64).into(),
                        FastMonomial::new(&vec![(6, 2), (7, 1)]),
                    ),
                    (
                        (5842294616606375917u64).into(),
                        FastMonomial::new(&vec![(5, 1), (7, 2)]),
                    ),
                    (
                        (16008607825441849293u64).into(),
                        FastMonomial::new(&vec![(6, 1), (7, 2)]),
                    ),
                    (
                        (11877681067236261850u64).into(),
                        FastMonomial::new(&vec![(7, 3)]),
                    ),
                    (
                        (2315200782863393558u64).into(),
                        FastMonomial::new(&vec![(5, 2)]),
                    ),
                    (
                        (15470057352885188411u64).into(),
                        FastMonomial::new(&vec![(5, 1), (6, 1)]),
                    ),
                    (
                        (7544915043732670853u64).into(),
                        FastMonomial::new(&vec![(6, 2)]),
                    ),
                    (
                        (4152388971314589023u64).into(),
                        FastMonomial::new(&vec![(5, 1), (7, 1)]),
                    ),
                    (
                        (14459726586885204931u64).into(),
                        FastMonomial::new(&vec![(6, 1), (7, 1)]),
                    ),
                    (
                        (1129956652251207029u64).into(),
                        FastMonomial::new(&vec![(7, 2)]),
                    ),
                    (
                        (13362083559747081110u64).into(),
                        FastMonomial::new(&vec![(5, 1)]),
                    ),
                    (
                        (9856833213872142272u64).into(),
                        FastMonomial::new(&vec![(6, 1)]),
                    ),
                    (
                        (1336208355974708111u64).into(),
                        FastMonomial::new(&vec![(7, 1)]),
                    ),
                    ((8675376255439944221u64).into(), FastMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((280).into(), FastMonomial::new(&vec![(2, 3)])),
                    (
                        (18446744073709551123u64).into(),
                        FastMonomial::new(&vec![(3, 3)]),
                    ),
                    ((155).into(), FastMonomial::new(&vec![(4, 3)])),
                    (
                        (12393906174523604947u64).into(),
                        FastMonomial::new(&vec![(5, 3)]),
                    ),
                    (
                        (9079256848778919915u64).into(),
                        FastMonomial::new(&vec![(5, 2), (6, 1)]),
                    ),
                    (
                        (4683743612465315821u64).into(),
                        FastMonomial::new(&vec![(5, 1), (6, 2)]),
                    ),
                    (
                        (14591662792680406994u64).into(),
                        FastMonomial::new(&vec![(6, 3)]),
                    ),
                    (
                        (17149707381026848712u64).into(),
                        FastMonomial::new(&vec![(5, 2), (7, 1)]),
                    ),
                    (
                        (1297036692682702845u64).into(),
                        FastMonomial::new(&vec![(5, 1), (6, 1), (7, 1)]),
                    ),
                    (
                        (4287426845256712178u64).into(),
                        FastMonomial::new(&vec![(6, 2), (7, 1)]),
                    ),
                    (
                        (1224979098644774908u64).into(),
                        FastMonomial::new(&vec![(5, 1), (7, 2)]),
                    ),
                    (
                        (17834254524387164103u64).into(),
                        FastMonomial::new(&vec![(6, 1), (7, 2)]),
                    ),
                    (
                        (9691746398101307361u64).into(),
                        FastMonomial::new(&vec![(7, 3)]),
                    ),
                    (
                        (13414881433324831873u64).into(),
                        FastMonomial::new(&vec![(5, 2)]),
                    ),
                    (
                        (5031862640384719684u64).into(),
                        FastMonomial::new(&vec![(5, 1), (6, 1)]),
                    ),
                    (
                        (17188778413613371636u64).into(),
                        FastMonomial::new(&vec![(6, 2)]),
                    ),
                    (
                        (1916411633332118839u64).into(),
                        FastMonomial::new(&vec![(5, 1), (7, 1)]),
                    ),
                    (
                        (8265166220188716359u64).into(),
                        FastMonomial::new(&vec![(6, 1), (7, 1)]),
                    ),
                    (
                        (13903501327901167912u64).into(),
                        FastMonomial::new(&vec![(7, 2)]),
                    ),
                    (
                        (3596302261513828044u64).into(),
                        FastMonomial::new(&vec![(5, 1)]),
                    ),
                    (
                        (16648592942952637535u64).into(),
                        FastMonomial::new(&vec![(6, 1)]),
                    ),
                    (
                        (13433124499900667401u64).into(),
                        FastMonomial::new(&vec![(7, 1)]),
                    ),
                    ((11749052441434960042u64).into(), FastMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((280).into(), FastMonomial::new(&vec![(5, 3)])),
                    (
                        (18446744073709551123u64).into(),
                        FastMonomial::new(&vec![(6, 3)]),
                    ),
                    ((155).into(), FastMonomial::new(&vec![(7, 3)])),
                    ((277610931875235913u64).into(), FastMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((56).into(), FastMonomial::new(&vec![(5, 3)])),
                    (
                        (18446744073709551467u64).into(),
                        FastMonomial::new(&vec![(6, 3)]),
                    ),
                    ((35).into(), FastMonomial::new(&vec![(7, 3)])),
                    ((2084698901943657590u64).into(), FastMonomial::new(&vec![])),
                ],
            ),
        ];

        buchberger(&mut ideal);
        assert!(is_groebner_basis(&ideal));
        print_ideal(&ideal);

        // ideal.push(FastPolynomial::new(
        //     8,
        //     &vec![
        //         ((1).into(), SimdMonomial::new(&vec![(2, 1)])),
        //         ((1).into(), SimdMonomial::new(&vec![(3, 1)])),
        //     ],
        // ));
        //
        // buchberger(&mut ideal);
        //
        // let want: Vec<DegRevLexPolynomial> = vec![FastPolynomial::new(
        //     8,
        //     &vec![((1).into(), SimdMonomial::new(&vec![]))],
        // )];
        // assert_eq!(want, ideal);
    }

    #[test]
    fn test_buchberger_given_case_3() {
        type MPolynomial = DegRevLexPolynomial;
        let mut ideal: Vec<MPolynomial> = vec![
            FastPolynomial::new(
                3,
                &vec![
                    // f1 = x1^2 + x1*x2 - x3
                    ((1).into(), FastMonomial::new(&vec![(0, 2)])),
                    ((1).into(), FastMonomial::new(&vec![(0, 1), (1, 1)])),
                    ((-1).into(), FastMonomial::new(&vec![(2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // f2 = x1*x3 - x2^2
                    ((1).into(), FastMonomial::new(&vec![(0, 1), (2, 1)])),
                    ((-1).into(), FastMonomial::new(&vec![(1, 2)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // f3 = x1x2^2 - x2x3
                    ((1).into(), FastMonomial::new(&vec![(0, 1), (1, 2)])),
                    ((-1).into(), FastMonomial::new(&vec![(1, 1), (2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // f4 = x2^2*x3 - x1^3
                    ((1).into(), FastMonomial::new(&vec![(1, 2), (2, 1)])),
                    ((-1).into(), FastMonomial::new(&vec![(0, 3)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // f5 = x2^3 - x1^2*x3
                    ((1).into(), FastMonomial::new(&vec![(1, 3)])),
                    ((-1).into(), FastMonomial::new(&vec![(0, 2), (2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // f6 = x1^3*x3 - x3^3
                    ((1).into(), FastMonomial::new(&vec![(0, 3), (2, 1)])),
                    ((-1).into(), FastMonomial::new(&vec![(2, 3)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // f7 = x1^2x2 - x2^2x3
                    ((1).into(), FastMonomial::new(&vec![(0, 2), (1, 1)])),
                    ((-1).into(), FastMonomial::new(&vec![(1, 2), (2, 1)])),
                ],
            ),
        ];

        buchberger(&mut ideal);
        assert!(is_groebner_basis(&ideal));
        print_ideal(&ideal);
    }
}
