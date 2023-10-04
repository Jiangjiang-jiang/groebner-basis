use crate::poly::{
    monomial::{FastMonomial, Monomial, MonomialOrd},
    polynomial::{FastPolynomial, Polynomial},
};
use ark_ff::Field;
use chrono::Local;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    collections::BTreeSet,
    fmt::{Debug, Formatter},
    iter,
    marker::PhantomData,
};

pub fn canonical_basis<const N: usize, O: MonomialOrd>(
    num_vars: usize,
    max_degree: u16,
    mut leading_monomials: Vec<FastMonomial<N, O>>,
) -> HashSet<FastMonomial<N, O>> {
    leading_monomials.par_sort_unstable();

    let mut result = HashSet::new();
    let mut stack = vec![FastMonomial::<N, O>::one()];

    for var in 0..num_vars {
        stack = stack
            .par_iter()
            .flat_map(|curr| {
                (0..(max_degree - curr.degree()))
                    .into_par_iter()
                    .filter_map(|degree| {
                        let monomial = FastMonomial::new(&[(var, degree)]) * &curr.clone();
                        leading_monomials
                            .iter()
                            .all(|lt| (&monomial / lt).is_none())
                            .then_some(monomial)
                    })
            })
            .collect();
        result.par_extend(stack.par_iter().cloned());
    }

    println!(
        "{} Produce canonical basis size: {}",
        Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        result.len()
    );

    result
}

struct LinearDependency<F: Field> {
    inner: Vec<Vec<F>>, // nD x D
    expr: Vec<Vec<F>>,
    lookup: Vec<isize>,
}

impl<F: Field> LinearDependency<F> {
    fn new(capacity: usize) -> Self {
        let mut lookup = Vec::with_capacity(capacity);
        (0..capacity).for_each(|_| {
            lookup.push(-1);
        });
        Self {
            inner: Vec::with_capacity(capacity),
            expr: Vec::with_capacity(capacity),
            lookup,
        }
    }

    fn solve(&self, mut reminder: Vec<F>) -> Result<Vec<F>, (Vec<F>, Vec<F>)> {
        let mut combination: Vec<F> = Vec::with_capacity(self.expr.len() + 1);
        (0..self.expr.len()).for_each(|_| {
            combination.push(F::zero());
        });

        for index in 0..reminder.len() {
            if reminder[index].is_zero() {
                continue;
            }

            if self.lookup[index] >= 0 {
                let i = self.lookup[index] as usize;
                let inner_i = &self.inner[i];
                let expr_i = &self.expr[i];

                let coeff = reminder[index] / inner_i[0];
                reminder[index..]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(j, f)| {
                        *f -= coeff * inner_i[j];
                    });
                combination
                    .par_iter_mut()
                    .take(expr_i.len())
                    .enumerate()
                    .for_each(|(j, f)| {
                        *f -= coeff * expr_i[j];
                    });
            } else {
                return Err((reminder, combination));
            }
        }

        return Ok(combination);
    }

    fn update(&mut self, (mut reminder, mut combination): (Vec<F>, Vec<F>)) {
        let index = reminder.iter().position(|x| !x.is_zero()).unwrap();
        self.lookup[index] = self.inner.len() as isize;
        reminder.drain(0..index);
        reminder.shrink_to_fit();
        self.inner.push(reminder);
        combination.push(F::one());
        self.expr.push(combination);

        if self.inner.len() % 100 == 0 {
            println!(
                "{} LinearDependency inner matrix size: {}x{}",
                Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                self.inner.len(),
                self.inner[0].len(),
            );
        }
    }
}

struct OrdWrapper<const N: usize, OF: MonomialOrd, OT: MonomialOrd>(
    FastMonomial<N, OF>,
    usize,
    PhantomData<OT>,
);

impl<const N: usize, OF: MonomialOrd, OT: MonomialOrd> Eq for OrdWrapper<N, OF, OT> {}

impl<const N: usize, OF: MonomialOrd, OT: MonomialOrd> PartialEq<Self> for OrdWrapper<N, OF, OT> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<const N: usize, OF: MonomialOrd, OT: MonomialOrd> PartialOrd<Self> for OrdWrapper<N, OF, OT> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize, OF: MonomialOrd, OT: MonomialOrd> Ord for OrdWrapper<N, OF, OT> {
    fn cmp(&self, other: &Self) -> Ordering {
        OT::compare(&self.0, &other.0)
    }
}

impl<const N: usize, OF: MonomialOrd, OT: MonomialOrd> Debug for OrdWrapper<N, OF, OT> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

pub struct FGLM<const N: usize, F: Field, OF: MonomialOrd, OT: MonomialOrd> {
    pub old_basis: Vec<FastPolynomial<F, FastMonomial<N, OF>>>,
    nf_cache: HashMap<FastMonomial<N, OF>, Vec<(F, FastMonomial<N, OF>)>>,
    _marker: PhantomData<OT>,
}

impl<const N: usize, F: Field, OF: MonomialOrd, OT: MonomialOrd> FGLM<N, F, OF, OT> {
    pub fn new(
        input: &[FastPolynomial<F, FastMonomial<N, OF>>],
    ) -> Vec<FastPolynomial<F, FastMonomial<N, OT>>> {
        Self {
            old_basis: input.to_vec(),
            nf_cache: HashMap::new(),
            _marker: PhantomData,
        }
        .new_basis()
    }

    fn normal_form(
        &mut self,
        monom: &FastMonomial<N, OF>,
        nf: &HashMap<FastMonomial<N, OF>, Vec<(F, FastMonomial<N, OF>)>>,
        xi: usize,
    ) -> Vec<(F, FastMonomial<N, OF>)> {
        if let Some(result) = nf.get(monom).or_else(|| self.nf_cache.get(monom)) {
            result.clone()
        } else {
            let u = (monom / xi).unwrap();
            let auk = nf.get(&u).or_else(|| self.nf_cache.get(&u)).unwrap();
            let mut result = auk
                .par_iter()
                .map(|(a, uk)| {
                    let xiuk = uk * xi;
                    nf.get(&xiuk)
                        .or_else(|| self.nf_cache.get(&xiuk))
                        .unwrap()
                        .par_iter()
                        .map(|(b, c)| (*a * b, c.clone()))
                        .collect()
                })
                .reduce(
                    Vec::new,
                    |a: Vec<(F, FastMonomial<N, OF>)>, b: Vec<(F, FastMonomial<N, OF>)>| {
                        let mut result = Vec::new();
                        let mut cur_iter = a.into_iter().peekable();
                        let mut other_iter = b.into_iter().peekable();
                        loop {
                            let which = match (cur_iter.peek(), other_iter.peek()) {
                                (Some(cur), Some(other)) => Some((cur.1).cmp(&other.1)),
                                (Some(_), None) => Some(Ordering::Less),
                                (None, Some(_)) => Some(Ordering::Greater),
                                (None, None) => None,
                            };
                            let smallest = match which {
                                Some(Ordering::Less) => cur_iter.next().unwrap(),
                                Some(Ordering::Equal) => {
                                    let other = other_iter.next().unwrap();
                                    let cur = cur_iter.next().unwrap();
                                    (cur.0 + other.0, cur.1)
                                },
                                Some(Ordering::Greater) => other_iter.next().unwrap(),
                                None => break,
                            };
                            if !smallest.0.is_zero() {
                                result.push(smallest);
                            }
                        }
                        result
                    },
                );
            result.shrink_to_fit();
            self.nf_cache.insert(monom.clone(), result.clone());
            result
        }
    }

    pub fn new_basis(&mut self) -> Vec<FastPolynomial<F, FastMonomial<N, OT>>> {
        let (mut nf, mut s1) = self.multiplication_matrix();
        nf.shrink_to_fit();
        nf.values_mut().for_each(|x| x.shrink_to_fit());
        s1.shrink_to_fit();
        let n = self.old_basis[0].num_of_vars;
        let mut results: Vec<FastPolynomial<F, FastMonomial<N, OT>>> = Vec::new();
        let mut next = BTreeSet::from_iter(iter::once(OrdWrapper(
            FastMonomial::<N, OF>::one(),
            0,
            PhantomData::<OT>,
        )));
        let mut mbasis: Vec<FastMonomial<N, OT>> = Vec::with_capacity(s1.len());
        let mut staircase: Vec<FastMonomial<N, OF>> = Vec::new();
        let mut linear = LinearDependency::<F>::new(s1.len());

        while let Some(OrdWrapper(monom, xi, _)) = next.pop_first() {
            if staircase.par_iter().all(|s| (&monom / s).is_none()) {
                let reminder = self.normal_form(&monom, &nf, xi);
                let mut reminder_peekable = reminder.into_iter().peekable();
                let reminder: Vec<F> = s1
                    .iter()
                    .map(|m| {
                        if let Some((_, r)) = reminder_peekable.peek() {
                            if r == m {
                                return reminder_peekable.next().unwrap().0;
                            }
                        }

                        F::zero()
                    })
                    .collect();

                match linear.solve(reminder) {
                    Ok(combination) => {
                        let pol = FastPolynomial::new(
                            n,
                            &combination
                                .into_par_iter()
                                .enumerate()
                                .filter(|(_, c)| !c.is_zero())
                                .map(|(i, c)| (c, mbasis[i].clone()))
                                .chain(rayon::iter::once((
                                    F::one(),
                                    monom.clone().transform_order(),
                                )))
                                .collect::<Vec<(F, FastMonomial<N, OT>)>>(),
                        );
                        results.push(pol);
                        staircase.push(monom);
                    },
                    Err(reduced) => {
                        next.par_extend(
                            (0..n)
                                .into_par_iter()
                                .map(|i| OrdWrapper(&monom * i, i, PhantomData::<OT>)),
                        );
                        linear.update(reduced);
                        mbasis.push(monom.transform_order());
                    },
                }
            }
        }

        results.par_sort_unstable_by(|a, b| {
            b.leading_monomial()
                .unwrap()
                .cmp(&a.leading_monomial().unwrap())
        });
        results
    }

    pub fn multiplication_matrix(
        &self,
    ) -> (
        HashMap<FastMonomial<N, OF>, Vec<(F, FastMonomial<N, OF>)>>,
        Vec<FastMonomial<N, OF>>,
    ) {
        let n = self.old_basis[0].num_of_vars;
        let leading_monomials: HashSet<FastMonomial<N, OF>> = self
            .old_basis
            .par_iter()
            .map(|p| p.leading_monomial().unwrap())
            .collect();
        let S = canonical_basis(
            n,
            leading_monomials
                .par_iter()
                .cloned()
                .reduce(FastMonomial::one, |a, b| a.lcm(&b))
                .degree(),
            leading_monomials.iter().cloned().collect(),
        );
        let mut F: Vec<FastMonomial<N, OF>> = S
            .par_iter()
            .flat_map(|u| {
                (0..n).into_par_iter().filter_map(|i| {
                    let xiu = u.clone() * i;
                    if !S.contains(&xiu) && !leading_monomials.contains(&xiu) {
                        Some(xiu)
                    } else {
                        None
                    }
                })
            })
            .collect();
        let mut NN: HashMap<FastMonomial<N, OF>, Vec<(F, FastMonomial<N, OF>)>> =
            HashMap::from_par_iter(
                S.par_iter()
                    .map(|u| (u.clone(), vec![(F::one(), u.clone())])),
            );
        NN.par_extend(self.old_basis.par_iter().map(|p| {
            (
                p.leading_monomial().unwrap(),
                p.trailing_terms()
                    .par_iter()
                    .map(|(c, m)| (c.neg(), m.clone()))
                    .collect(),
            )
        }));

        F.par_sort_unstable();
        for u in F {
            let (i, uu) = (0..n)
                .into_par_iter()
                .find_map_any(|i| {
                    (&u / i)
                        .filter(|r| NN.contains_key(r) && !S.contains(r))
                        .map(|r| (i, r))
                })
                .unwrap();
            NN.insert(
                u,
                NN.get(&uu)
                    .unwrap()
                    .par_iter()
                    .map(|(a, uuu)| {
                        NN.get(&(uuu * i))
                            .unwrap()
                            .par_iter()
                            .map(|(c, m)| (*a * c, m.clone()))
                            .collect()
                    })
                    .reduce(
                        Vec::new,
                        |a: Vec<(F, FastMonomial<N, OF>)>, b: Vec<(F, FastMonomial<N, OF>)>| {
                            let mut result = Vec::new();
                            let mut cur_iter = a.into_iter().peekable();
                            let mut other_iter = b.into_iter().peekable();
                            loop {
                                let which = match (cur_iter.peek(), other_iter.peek()) {
                                    (Some(cur), Some(other)) => Some((cur.1).cmp(&other.1)),
                                    (Some(_), None) => Some(Ordering::Less),
                                    (None, Some(_)) => Some(Ordering::Greater),
                                    (None, None) => None,
                                };
                                let smallest = match which {
                                    Some(Ordering::Less) => cur_iter.next().unwrap(),
                                    Some(Ordering::Equal) => {
                                        let other = other_iter.next().unwrap();
                                        let cur = cur_iter.next().unwrap();
                                        (cur.0 + other.0, cur.1)
                                    },
                                    Some(Ordering::Greater) => other_iter.next().unwrap(),
                                    None => break,
                                };
                                if !smallest.0.is_zero() {
                                    result.push(smallest);
                                }
                            }
                            result
                        },
                    ),
            );
        }

        let mut S = Vec::from_iter(S.into_iter());
        S.par_sort_unstable();
        println!(
            "{} Compute normal form size: {}",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            NN.len()
        );
        (NN, S)
    }
}

#[cfg(test)]
#[allow(clippy::all)]
mod tests {
    use crate::{
        fglm::{canonical_basis, LinearDependency, FGLM},
        groebner::is_groebner_basis,
        gvw::GVW,
        poly::{
            monomial::{FastMonomial, LexOrder, Monomial, MonomialOrd},
            polynomial::{FastPolynomial, Polynomial},
        },
        DegRevLexPolynomial, LexPolynomial, GF,
    };
    use ark_ff::{Fp, FpConfig, PrimeField, Zero};
    use chrono::Local;
    use hashbrown::HashMap;
    use rayon::prelude::*;
    use std::{marker::PhantomData, ops::Neg};

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
    fn test_canonical_basis1() {
        let result = canonical_basis::<1, LexOrder>(
            2,
            3,
            vec![FastMonomial::new(&[(0, 2)]), FastMonomial::new(&[(1, 1)])],
        );
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_canonical_basis2() {
        let ideal: Vec<DegRevLexPolynomial<1>> = vec![
            // y^2 + 9*y + 2*x +6
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&vec![(1, 2)])),
                    (GF::from(9), FastMonomial::new(&vec![(1, 1)])),
                    (GF::from(2), FastMonomial::new(&vec![(0, 1)])),
                    (GF::from(6), FastMonomial::one()),
                ],
            ),
            // x^2 + 2*y + 9
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&vec![(0, 2)])),
                    (GF::from(2), FastMonomial::new(&vec![(1, 1)])),
                    (GF::from(9), FastMonomial::one()),
                ],
            ),
            // z + 9
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&vec![(2, 1)])),
                    (GF::from(9), FastMonomial::one()),
                ],
            ),
        ];

        let leading_monomials: Vec<_> = ideal
            .iter()
            .map(|p| p.leading_monomial().unwrap())
            .collect();
        let max_degree = leading_monomials
            .par_iter()
            .cloned()
            .reduce(FastMonomial::one, |a, b| a.lcm(&b))
            .degree();
        let result = canonical_basis(3, max_degree, leading_monomials);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_multiplication_matrix() {
        let ideal: Vec<DegRevLexPolynomial<1>> = vec![
            // y^2 + 9*y + 2*x +6
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&vec![(1, 2)])),
                    (GF::from(9), FastMonomial::new(&vec![(1, 1)])),
                    (GF::from(2), FastMonomial::new(&vec![(0, 1)])),
                    (GF::from(6), FastMonomial::one()),
                ],
            ),
            // x^2 + 2*y + 9
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&vec![(0, 2)])),
                    (GF::from(2), FastMonomial::new(&vec![(1, 1)])),
                    (GF::from(9), FastMonomial::one()),
                ],
            ),
            // z + 9
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&vec![(2, 1)])),
                    (GF::from(9), FastMonomial::one()),
                ],
            ),
        ];

        let fglm = FGLM {
            old_basis: ideal.clone(),
            nf_cache: HashMap::new(),
            _marker: PhantomData::<LexOrder>,
        };
        let (nf, _) = fglm.multiplication_matrix();
        nf.into_par_iter().all(|(m, reminder)| {
            reminder.eq(&DegRevLexPolynomial::new(3, &vec![(GF::from(1), m)])
                .div_mod_polys(&ideal)
                .1
                .terms)
        });
    }

    #[test]
    fn test_fglm() {
        let ideal: Vec<DegRevLexPolynomial<1>> = vec![
            // y^2 + 9*y + 2*x +6
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&vec![(1, 2)])),
                    (GF::from(9), FastMonomial::new(&vec![(1, 1)])),
                    (GF::from(2), FastMonomial::new(&vec![(0, 1)])),
                    (GF::from(6), FastMonomial::one()),
                ],
            ),
            // x^2 + 2*y + 9
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&vec![(0, 2)])),
                    (GF::from(2), FastMonomial::new(&vec![(1, 1)])),
                    (GF::from(9), FastMonomial::one()),
                ],
            ),
            // z + 9
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&vec![(2, 1)])),
                    (GF::from(9), FastMonomial::one()),
                ],
            ),
        ];

        let result: Vec<LexPolynomial<1>> = FGLM::new(&ideal);

        let want: Vec<LexPolynomial<1>> = vec![
            // 1*x_0 + 9223372036854775779*x_1^2 + 9223372036854775783*x_1 + 3
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&[(0, 1)])),
                    (
                        GF::from(9223372036854775779u64),
                        FastMonomial::new(&[(1, 2)]),
                    ),
                    (
                        GF::from(9223372036854775783u64),
                        FastMonomial::new(&[(1, 1)]),
                    ),
                    (GF::from(3), FastMonomial::one()),
                ],
            ),
            // 1*x_1^4 + 18*x_1^3 + 93*x_1^2 + 116*x_1 + 72
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&[(1, 4)])),
                    (GF::from(18), FastMonomial::new(&[(1, 3)])),
                    (GF::from(93), FastMonomial::new(&[(1, 2)])),
                    (GF::from(116), FastMonomial::new(&[(1, 1)])),
                    (GF::from(72), FastMonomial::one()),
                ],
            ),
            // 1*x_2 + 9
            FastPolynomial::new(
                3,
                &vec![
                    (GF::from(1), FastMonomial::new(&[(2, 1)])),
                    (GF::from(9), FastMonomial::one()),
                ],
            ),
        ];

        assert_eq!(result, want);
    }

    #[test]
    fn test_fglm2() {
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

        println!(
            "{} fglm start",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
        );
        let result: Vec<LexPolynomial<1>> = FGLM::new(&ideal);
        println!(
            "{} fglm end",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
        );
        print_ideal(&result);
        assert_eq!(8, result.len());
        assert_eq!(result[0].terms()[0].0, GF::from(15173136768223169137u64));
        assert_eq!(result[1].terms()[0].0, GF::from(2222020229172752966u64));
        assert_eq!(result[2].terms()[0].0, GF::from(9600559040574426672u64));
        assert_eq!(result[3].terms()[0].0, GF::from(8509429470013082800u64));
        assert_eq!(result[4].terms()[0].0, GF::from(15552399571159279628u64));
        assert_eq!(result[5].terms()[0].0, GF::from(4713528656291954913u64));
        assert_eq!(result[6].terms()[0].0, GF::from(9536593990729476427u64));
        assert_eq!(result[7].terms()[0].0, GF::from(1363273397881249216u64));
        assert_eq!(
            result[0].terms()[result[0].terms().len() - 2].0,
            GF::from(15023880628536987373u64)
        );
        assert_eq!(
            result[1].terms()[result[1].terms().len() - 2].0,
            GF::from(11847206246306325334u64)
        );
        assert_eq!(
            result[2].terms()[result[2].terms().len() - 2].0,
            GF::from(9670389801135998979u64)
        );
        assert_eq!(
            result[3].terms()[result[3].terms().len() - 2].0,
            GF::from(14896425988726183069u64)
        );
        assert_eq!(
            result[4].terms()[result[4].terms().len() - 2].0,
            GF::from(5783012852598398891u64)
        );
        assert_eq!(
            result[5].terms()[result[5].terms().len() - 2].0,
            GF::from(17385693665674487526u64)
        );
        assert_eq!(
            result[6].terms()[result[6].terms().len() - 2].0,
            GF::from(17933356555528874765u64)
        );
        assert_eq!(
            result[7].terms()[result[7].terms().len() - 2].0,
            GF::from(7820664615100453659u64)
        );
        assert!(is_groebner_basis(&result));
    }

    #[test]
    fn test_fglm3() {
        let mut ideal: Vec<DegRevLexPolynomial<1>> = vec![
            FastPolynomial::new(
                4,
                &vec![
                    (
                        (18446744073709551123u64).into(),
                        FastMonomial::new(&vec![(3, 3)]),
                    ),
                    (
                        (14741170423712439482u64).into(),
                        FastMonomial::new(&vec![(3, 2)]),
                    ),
                    ((280).into(), FastMonomial::new(&vec![(1, 1)])),
                    ((155).into(), FastMonomial::new(&vec![(2, 1)])),
                    (
                        (6938193394142292513u64).into(),
                        FastMonomial::new(&vec![(3, 1)]),
                    ),
                    ((11495118231455885229u64).into(), FastMonomial::one()),
                ],
            ),
            FastPolynomial::new(
                4,
                &vec![
                    (
                        (9540029153929078255u64).into(),
                        FastMonomial::new(&vec![(0, 3)]),
                    ),
                    (
                        (9511602413006487524u64).into(),
                        FastMonomial::new(&vec![(1, 3)]),
                    ),
                    (
                        (8214565720323784678u64).into(),
                        FastMonomial::new(&vec![(2, 3)]),
                    ),
                    (
                        (3984360272876093435u64).into(),
                        FastMonomial::new(&vec![(0, 2)]),
                    ),
                    (
                        (13874580259866134005u64).into(),
                        FastMonomial::new(&vec![(0, 1)]),
                    ),
                    (
                        (720575940379279356u64).into(),
                        FastMonomial::new(&vec![(3, 1)]),
                    ),
                    ((9801537791525413721u64).into(), FastMonomial::one()),
                ],
            ),
            FastPolynomial::new(
                4,
                &vec![
                    (
                        (2022116232689352708u64).into(),
                        FastMonomial::new(&vec![(1, 9)]),
                    ),
                    (
                        (15364030128774447058u64).into(),
                        FastMonomial::new(&vec![(1, 6), (2, 3)]),
                    ),
                    (
                        (12604449457103175640u64).into(),
                        FastMonomial::new(&vec![(1, 3), (2, 6)]),
                    ),
                    (
                        (6569063006473289707u64).into(),
                        FastMonomial::new(&vec![(2, 9)]),
                    ),
                    (
                        (4005951868546056159u64).into(),
                        FastMonomial::new(&vec![(1, 6), (3, 1)]),
                    ),
                    (
                        (11869236817934942166u64).into(),
                        FastMonomial::new(&vec![(1, 3), (2, 3), (3, 1)]),
                    ),
                    (
                        (2438136248267702264u64).into(),
                        FastMonomial::new(&vec![(2, 6), (3, 1)]),
                    ),
                    (
                        (13795448932593041637u64).into(),
                        FastMonomial::new(&vec![(1, 6)]),
                    ),
                    (
                        (17516485045486249573u64).into(),
                        FastMonomial::new(&vec![(1, 3), (2, 3)]),
                    ),
                    (
                        (11021533492814565835u64).into(),
                        FastMonomial::new(&vec![(2, 6)]),
                    ),
                    (
                        (17189113877766340566u64).into(),
                        FastMonomial::new(&vec![(1, 3), (3, 2)]),
                    ),
                    (
                        (5408260202518544368u64).into(),
                        FastMonomial::new(&vec![(2, 3), (3, 2)]),
                    ),
                    (
                        (709738303232783738u64).into(),
                        FastMonomial::new(&vec![(1, 3), (3, 1)]),
                    ),
                    (
                        (11139020274549009308u64).into(),
                        FastMonomial::new(&vec![(2, 3), (3, 1)]),
                    ),
                    (
                        (12266479594948129885u64).into(),
                        FastMonomial::new(&vec![(1, 3)]),
                    ),
                    (
                        (10450019996349588767u64).into(),
                        FastMonomial::new(&vec![(2, 3)]),
                    ),
                    (
                        (14763362528473907150u64).into(),
                        FastMonomial::new(&vec![(3, 3)]),
                    ),
                    (
                        (3724743561184366275u64).into(),
                        FastMonomial::new(&vec![(3, 2)]),
                    ),
                    (
                        (17744319919474170262u64).into(),
                        FastMonomial::new(&vec![(0, 1)]),
                    ),
                    (
                        (6608276318305135583u64).into(),
                        FastMonomial::new(&vec![(3, 1)]),
                    ),
                    ((17093922729234360527u64).into(), FastMonomial::one()),
                ],
            ),
            FastPolynomial::new(
                4,
                &vec![
                    (
                        (10912412344460258544u64).into(),
                        FastMonomial::new(&vec![(0, 3)]),
                    ),
                    (
                        (4611686018427387891u64).into(),
                        FastMonomial::new(&vec![(1, 3)]),
                    ),
                    (
                        (6917529027641081834u64).into(),
                        FastMonomial::new(&vec![(2, 3)]),
                    ),
                    (
                        (9464579679177512115u64).into(),
                        FastMonomial::new(&vec![(0, 2)]),
                    ),
                    (
                        (3432701790560393188u64).into(),
                        FastMonomial::new(&vec![(0, 1)]),
                    ),
                    (
                        (6917529027641081833u64).into(),
                        FastMonomial::new(&vec![(3, 1)]),
                    ),
                    ((7312314097538643946u64).into(), FastMonomial::one()),
                ],
            ),
        ];

        ideal = GVW::new(&ideal);
        assert!(is_groebner_basis(&ideal));

        println!(
            "{} fglm start",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
        );
        let result: Vec<LexPolynomial<1>> = FGLM::new(&ideal);
        println!(
            "{} fglm end",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
        );
        print_ideal(&result);
        assert!(is_groebner_basis(&result));
    }

    #[test]
    fn test_multiplication_matrix2() {
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

        let fglm = FGLM {
            old_basis: ideal.clone(),
            nf_cache: HashMap::new(),
            _marker: PhantomData::<LexOrder>,
        };
        println!(
            "{} fglm matphi start",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
        );
        let (nf, _) = fglm.multiplication_matrix();
        println!(
            "{} fglm matphi end",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
        );
        nf.into_par_iter().all(|(m, reminder)| {
            reminder.eq(&DegRevLexPolynomial::new(8, &vec![(GF::from(1), m)])
                .div_mod_polys(&ideal)
                .1
                .terms)
        });
    }

    #[test]
    fn test_LinearDependency() {
        let mut linear = LinearDependency::<GF>::new(2);
        let err = linear.solve(vec![GF::from(2), GF::from(1)]).unwrap_err();
        linear.update(err);
        let err = linear.solve(vec![GF::from(1), GF::from(3)]).unwrap_err();
        linear.update(err);
        let b = vec![GF::from(4), GF::from(7)];
        assert_eq!(
            linear
                .solve(b)
                .unwrap()
                .into_iter()
                .map(GF::neg)
                .collect::<Vec<GF>>(),
            vec![GF::from(1), GF::from(2)]
        );

        let mut linear = LinearDependency::<GF>::new(2);
        let err = linear.solve(vec![GF::from(2), GF::from(1)]).unwrap_err();
        linear.update(err);
        assert_eq!(
            linear.solve(vec![GF::from(4), GF::from(2)]).unwrap(),
            vec![GF::from(-2)]
        );
        let b = vec![GF::from(4), GF::from(3)];
        assert!(linear.solve(b).is_err());
    }
}
