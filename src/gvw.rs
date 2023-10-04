use crate::{
    groebner::reduce_groebner_basis,
    poly::{
        monomial::{FastMonomial, Monomial, MonomialOrd},
        polynomial::{FastPolynomial, Polynomial},
    },
};
use ark_ff::{Field, Zero};
use chrono::Local;
use hashbrown::HashSet;
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    collections::BTreeSet,
    ops::{Div, Mul, MulAssign},
    sync::Arc,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Signature<const N: usize, O: MonomialOrd> {
    monomial: FastMonomial<N, O>,
    position: usize,
    lm: FastMonomial<N, O>,
}

impl<const N: usize, O: MonomialOrd> Signature<N, O> {
    fn new<F: Field>(
        monomial: FastMonomial<N, O>,
        position: usize,
        poly: &FastPolynomial<F, FastMonomial<N, O>>,
    ) -> Self {
        Self {
            lm: poly.leading_monomial().unwrap() * &monomial,
            monomial,
            position,
        }
    }
}

impl<const N: usize, O: MonomialOrd> PartialOrd for Signature<N, O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Self::cmp(self, other))
    }
}

impl<const N: usize, O: MonomialOrd> Ord for Signature<N, O> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.lm.cmp(&other.lm).then_with(|| {
            other
                .position
                .cmp(&self.position)
                .then_with(|| self.monomial.cmp(&other.monomial))
        })
    }
}

impl<'a, const N: usize, O: MonomialOrd> MulAssign<&'a FastMonomial<N, O>> for Signature<N, O> {
    fn mul_assign(&mut self, rhs: &'a FastMonomial<N, O>) {
        self.monomial *= rhs;
        self.lm *= rhs;
    }
}

impl<'a, 'b, const N: usize, O: MonomialOrd> Mul<&'a FastMonomial<N, O>> for &'b Signature<N, O> {
    type Output = Signature<N, O>;

    fn mul(self, rhs: &'a FastMonomial<N, O>) -> Self::Output {
        let mut result = self.clone();
        result.mul_assign(rhs);
        result
    }
}

impl<'a, 'b, const N: usize, O: MonomialOrd> Div<&'a Signature<N, O>> for &'b Signature<N, O> {
    type Output = Option<FastMonomial<N, O>>;

    fn div(self, rhs: &'a Signature<N, O>) -> Self::Output {
        if self.position == rhs.position {
            &self.monomial / &rhs.monomial
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MSignature<const N: usize, F: Field, O: MonomialOrd> {
    signature: Signature<N, O>,
    polynomial: Arc<FastPolynomial<F, FastMonomial<N, O>>>,
}

impl<const N: usize, F: Field, O: MonomialOrd> PartialOrd for MSignature<N, F, O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Self::cmp(self, other))
    }
}

impl<const N: usize, F: Field, O: MonomialOrd> Ord for MSignature<N, F, O> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.signature.cmp(&other.signature)
    }
}

impl<'a, const N: usize, F: Field, O: MonomialOrd> MulAssign<&'a FastMonomial<N, O>>
    for MSignature<N, F, O>
{
    fn mul_assign(&mut self, rhs: &'a FastMonomial<N, O>) {
        self.signature *= rhs;
        self.polynomial = Arc::new(self.polynomial.as_ref() * rhs);
    }
}

impl<'a, 'b, const N: usize, F: Field, O: MonomialOrd> Mul<&'a FastMonomial<N, O>>
    for &'b MSignature<N, F, O>
{
    type Output = MSignature<N, F, O>;

    fn mul(self, rhs: &'a FastMonomial<N, O>) -> Self::Output {
        let mut result = self.clone();
        result.mul_assign(rhs);
        result
    }
}

/// GVW
/// paper: http://www.math.clemson.edu/~sgao/papers/gvw_R130704.pdf
pub struct GVW<const N: usize, F: Field, O: MonomialOrd> {
    pub ideal: Vec<FastPolynomial<F, FastMonomial<N, O>>>,
    pub n: usize,
}

impl<const N: usize, F: Field, O: MonomialOrd> GVW<N, F, O> {
    fn make_jpair(
        uf: &MSignature<N, F, O>,
        vg: &MSignature<N, F, O>,
    ) -> Option<(FastMonomial<N, O>, MSignature<N, F, O>)> {
        let (lmf, lmg) = (
            uf.polynomial.leading_monomial().unwrap(),
            vg.polynomial.leading_monomial().unwrap(),
        );
        let t = lmf.lcm(&lmg);
        let (tf, tg) = ((&t / &lmf).unwrap(), (&t / &lmg).unwrap());
        let uftf = &uf.signature * &tf;
        let vgtg = &vg.signature * &tg;
        match uftf.cmp(&vgtg) {
            Ordering::Less => Some((tg, vg.clone())),
            Ordering::Equal => None,
            Ordering::Greater => Some((tf, uf.clone())),
        }
    }

    fn is_divisible(
        t: &FastMonomial<N, O>,
        xaeif: &MSignature<N, F, O>,
        H: &HashSet<Signature<N, O>>,
    ) -> bool {
        H.par_iter()
            .any(|h| (&(&xaeif.signature * t) / h).is_some())
    }

    fn is_covered(uf: &MSignature<N, F, O>, G: &[MSignature<N, F, O>]) -> bool {
        G.par_iter().any(|vg| {
            (&uf.signature / &vg.signature)
                .filter(|t| {
                    vg.polynomial.leading_monomial().unwrap() * t
                        < uf.polynomial.leading_monomial().unwrap()
                })
                .is_some()
        })
    }

    fn regular_top_reduce(
        mut reminder: MSignature<N, F, O>,
        G: &[MSignature<N, F, O>],
    ) -> MSignature<N, F, O> {
        let mut reminder_polynomial = Arc::unwrap_or_clone(reminder.polynomial);
        while !reminder_polynomial.is_zero() {
            if let Some((t, vg)) = G.par_iter().find_map_first(|vg| {
                let lmf = reminder_polynomial.leading_monomial().unwrap();
                let lmu = &reminder.signature;
                let lmg = vg.polynomial.leading_monomial().unwrap();
                let lmv = &vg.signature;
                (lmf / &lmg).filter(|t| lmu > &(lmv * t)).map(|t| (t, vg))
            }) {
                let c = reminder_polynomial.leading_coefficient().unwrap()
                    / vg.polynomial.leading_coefficient().unwrap();
                reminder_polynomial -= &(vg.polynomial.as_ref() * &(c, t));
            } else {
                break;
            }
        }

        reminder.polynomial = Arc::new(reminder_polynomial);
        reminder
    }

    pub fn new(
        input: &[FastPolynomial<F, FastMonomial<N, O>>],
    ) -> Vec<FastPolynomial<F, FastMonomial<N, O>>> {
        let mut this = Self {
            ideal: input.to_vec(),
            n: input.par_iter().map(|f| f.num_of_vars).max().unwrap(),
        };

        println!(
            "{} GVW start ideal.len(): {}",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            this.ideal.len()
        );

        let mut H: HashSet<Signature<N, O>> =
            HashSet::from_par_iter(this.ideal.par_iter().enumerate().flat_map(|(i, ii)| {
                this.ideal[..i].par_iter().enumerate().map(move |(j, jj)| {
                    let fjei = Signature::new(jj.leading_monomial().unwrap(), i, ii);
                    let fiej = Signature::new(ii.leading_monomial().unwrap(), j, jj);
                    Signature::max(fjei, fiej)
                })
            }));
        let mut G: Vec<MSignature<N, F, O>> =
            Vec::from_par_iter(this.ideal.par_iter().enumerate().map(|(i, f)| {
                let g = &this.ideal;
                MSignature {
                    signature: Signature::new(FastMonomial::one(), i, &g[i]),
                    polynomial: Arc::new(f.clone()),
                }
            }));
        G.par_sort_unstable_by(|a, b| {
            a.polynomial
                .leading_monomial()
                .unwrap()
                .cmp(&b.polynomial.leading_monomial().unwrap())
        });
        let mut jpair_set: BTreeSet<(FastMonomial<N, O>, MSignature<N, F, O>)> =
            BTreeSet::from_par_iter(G.par_iter().enumerate().flat_map(|(i, gi)| {
                G[..i]
                    .par_iter()
                    .filter_map(|gj| Self::make_jpair(gi, gj))
                    .filter(|(t, xaeif)| {
                        !Self::is_divisible(t, xaeif, &H) && !Self::is_covered(xaeif, &G)
                    })
            }));

        let mut counter: isize = 0;
        while let Some((t, xaeif)) = jpair_set.pop_first() {
            if Self::is_divisible(&t, &xaeif, &H) || Self::is_covered(&xaeif, &G) {
                continue;
            }

            counter += 1;
            if counter == 100 {
                counter = 0;
                println!(
                    "{} GVM H.len(): {}, G.len(): {}, jpair_set.len(): {} after 100 iterations",
                    Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                    H.len(),
                    G.len(),
                    jpair_set.len()
                );
            }

            let uf = Self::regular_top_reduce(&xaeif * &t, &G);
            if uf.polynomial.is_zero() {
                H.insert(uf.signature);
            } else {
                for (h, pair) in G
                    .par_iter()
                    .filter_map(|vg| {
                        let left = &uf.signature * &vg.polynomial.leading_monomial().unwrap();
                        let right = &vg.signature * &uf.polynomial.leading_monomial().unwrap();
                        left.ne(&right).then_some((
                            Signature::max(left, right),
                            Self::make_jpair(vg, &uf).filter(|(tt, xa)| {
                                !Self::is_divisible(tt, xa, &H) && !Self::is_covered(xa, &G)
                            }),
                        ))
                    })
                    .collect::<Vec<(
                        Signature<N, O>,
                        Option<(FastMonomial<N, O>, MSignature<N, F, O>)>,
                    )>>()
                {
                    H.insert(h);
                    if let Some(p) = pair {
                        jpair_set.insert(p);
                    }
                }
                let uf_lm = uf.polynomial.leading_monomial().unwrap();
                match G.binary_search_by(|a| a.polynomial.leading_monomial().unwrap().cmp(&uf_lm)) {
                    Ok(i) | Err(i) => G.insert(i, uf),
                }
            }
        }

        this.ideal = G
            .into_par_iter()
            .map(|v| Arc::unwrap_or_clone(v.polynomial))
            .collect();
        println!(
            "{} Produce ideal.len(): {}",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            this.ideal.len()
        );

        reduce_groebner_basis(&mut this.ideal);
        this.ideal
    }
}

#[cfg(test)]
#[allow(clippy::all)]
mod tests {
    use crate::{
        groebner::is_groebner_basis,
        gvw::GVW,
        poly::{
            monomial::{FastMonomial, Monomial, MonomialOrd},
            polynomial::{FastPolynomial, Polynomial},
        },
        DegRevLexPolynomial,
    };
    use ark_ff::{Fp, FpConfig, PrimeField, Zero};

    fn print_ideal<const N: usize, P: FpConfig<1>, O: MonomialOrd>(
        input_polynomials: &Vec<FastPolynomial<Fp<P, 1>, FastMonomial<N, O>>>,
    ) {
        let polys = input_polynomials
            .iter()
            .map(|poly| {
                poly.terms()
                    .iter()
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
                                    .map(|(v, &e)| {
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
    fn test_GVW_given_case_1() {
        let mut ideal: Vec<DegRevLexPolynomial<1>> = vec![
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

        ideal = GVW::new(&ideal);
        assert!(is_groebner_basis(&ideal));
        print_ideal(&ideal);
    }

    #[test]
    fn test_GVW_given_case_2() {
        let mut ideal: Vec<DegRevLexPolynomial<1>> = vec![
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

        ideal = GVW::new(&ideal);
        assert!(is_groebner_basis(&ideal));
        print_ideal(&ideal);
    }

    #[test]
    fn test_GVW_given_case_3() {
        type MPolynomial = DegRevLexPolynomial<1>;
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
                    // GVW = x2^3 - x1^2*x3
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

        ideal = GVW::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));
    }

    #[test]
    fn test_GVW_given_case_4() {
        type MPolynomial = DegRevLexPolynomial<1>;
        let mut ideal: Vec<MPolynomial> = vec![
            FastPolynomial::new(
                4,
                &vec![
                    // x^2*y - z^2*t
                    ((1).into(), FastMonomial::new(&vec![(0, 2), (1, 1)])),
                    ((-1).into(), FastMonomial::new(&vec![(2, 2), (3, 1)])),
                ],
            ),
            FastPolynomial::new(
                4,
                &vec![
                    // x*z^2 - y^2*t
                    ((1).into(), FastMonomial::new(&vec![(0, 1), (2, 2)])),
                    ((-1).into(), FastMonomial::new(&vec![(1, 2), (3, 1)])),
                ],
            ),
            FastPolynomial::new(
                4,
                &vec![
                    // y*z^3 - x^2*t^2
                    ((1).into(), FastMonomial::new(&vec![(1, 1), (2, 3)])),
                    ((-1).into(), FastMonomial::new(&vec![(0, 2), (3, 2)])),
                ],
            ),
        ];

        ideal = GVW::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));
    }

    #[test]
    fn test_GVW_given_case_5() {
        type MPolynomial = DegRevLexPolynomial<1>;
        let mut ideal: Vec<MPolynomial> = vec![
            FastPolynomial::new(
                3,
                &vec![
                    // x*y*z - 1
                    ((1).into(), FastMonomial::new(&vec![(0, 1), (1, 1), (2, 1)])),
                    ((-1).into(), FastMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // x*y - z
                    ((1).into(), FastMonomial::new(&vec![(0, 1), (1, 1)])),
                    ((-1).into(), FastMonomial::new(&vec![(2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // y*z - y
                    ((1).into(), FastMonomial::new(&vec![(1, 1), (2, 1)])),
                    ((-1).into(), FastMonomial::new(&vec![(1, 1)])),
                ],
            ),
        ];

        ideal = GVW::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));
    }

    #[test]
    fn test_GVW_given_case_6() {
        type MPolynomial = DegRevLexPolynomial<1>;
        let mut ideal: Vec<MPolynomial> = vec![
            FastPolynomial::new(
                3,
                &vec![
                    // x*y + z
                    ((1).into(), FastMonomial::new(&vec![(0, 1), (1, 1)])),
                    ((1).into(), FastMonomial::new(&vec![(2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // x^2
                    ((1).into(), FastMonomial::new(&vec![(0, 2)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // z
                    ((1).into(), FastMonomial::new(&vec![(2, 1)])),
                ],
            ),
        ];

        ideal = GVW::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));
    }

    #[test]
    fn test_GVW_given_case_7() {
        type MPolynomial = DegRevLexPolynomial<1>;
        let mut ideal: Vec<MPolynomial> = vec![
            FastPolynomial::new(
                3,
                &vec![
                    // x*y*z
                    ((1).into(), FastMonomial::new(&vec![(0, 1), (1, 1), (2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // z^4 + y^2 + x
                    ((1).into(), FastMonomial::new(&vec![(0, 1)])),
                    ((1).into(), FastMonomial::new(&vec![(1, 2)])),
                    ((1).into(), FastMonomial::new(&vec![(2, 4)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // y^5 + z^5
                    ((1).into(), FastMonomial::new(&vec![(2, 5)])),
                    ((1).into(), FastMonomial::new(&vec![(1, 5)])),
                ],
            ),
        ];

        ideal = GVW::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));
    }
}
