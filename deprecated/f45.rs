use crate::{
    groebner::{dehomogenize, homogenize, is_homogeneous_ideal, reduce_groebner_basis},
    poly::{
        monomial::Monomial,
        polynomial::{FastPolynomial, Polynomial},
    },
    Entry,
};
use ark_ff::{Field, Zero};
use chrono::Local;
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    ops::Mul,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature<M>
where
    M: Monomial,
{
    monomial: M,
    position: usize,
}

impl<M: Monomial> PartialOrd for Signature<M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Self::cmp(self, other))
    }
}

impl<M: Monomial> Ord for Signature<M> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.position
            .cmp(&other.position)
            .then_with(|| self.monomial.cmp(&other.monomial))
    }
}

impl<'a, 'b, M: Monomial> Mul<&'a M> for &'b Signature<M> {
    type Output = Signature<M>;

    fn mul(self, rhs: &'a M) -> Self::Output {
        let mut result = self.clone();
        result.monomial *= rhs;
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CriticalPair<M>
where
    M: Monomial,
{
    t: M,
    u: M,
    k: usize,
    v: M,
    l: usize,
}

impl<M: Monomial> PartialOrd for CriticalPair<M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Self::cmp(self, other))
    }
}

impl<M: Monomial> Ord for CriticalPair<M> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.t.degree().cmp(&other.t.degree())
    }
}

pub struct F45<F: Field, M: Monomial> {
    pub ideal: Vec<FastPolynomial<F, M>>,
    pub L: Vec<Entry<Signature<M>, FastPolynomial<F, M>>>,
    pub rules: Vec<Vec<(M, usize)>>,
}

impl<F: Field, M: Monomial> F45<F, M> {
    pub fn new(input: &[FastPolynomial<F, M>]) -> Vec<FastPolynomial<F, M>> {
        let mut this = Self {
            ideal: input.to_vec(),
            L: vec![],
            rules: vec![],
        };

        let need_homogeneous = !is_homogeneous_ideal(&this.ideal);
        if need_homogeneous {
            homogenize(&mut this.ideal);
        }

        {
            // F45 main function
            this.ideal.par_sort_unstable_by(|a, b| {
                a.degree().cmp(&b.degree()).then_with(|| {
                    a.leading_monomial()
                        .unwrap()
                        .cmp(&b.leading_monomial().unwrap())
                })
            });

            let mut P: HashSet<CriticalPair<M>> = HashSet::new();
            let mut G: Vec<Vec<usize>> = Vec::new();
            let _ideal: Vec<(usize, FastPolynomial<F, M>)> =
                this.ideal.par_iter().cloned().enumerate().collect();
            for (i, f) in _ideal.into_iter() {
                this.L.push(Entry(
                    Signature {
                        monomial: M::one(),
                        position: i,
                    },
                    f.to_owned()
                        * &(
                            f.leading_coefficient().unwrap().inverse().unwrap(),
                            M::one(),
                        ),
                ));
                this.rules.push(Vec::new());
                this.add_rule(
                    &Signature {
                        monomial: M::one(),
                        position: i,
                    },
                    i,
                );
                G.iter().flatten().for_each(|&g| {
                    if let Some(cp) = this.critical_pair(i, g, &G) {
                        P.insert(cp);
                    }
                });
                G.push(vec![i]);
            }

            while !P.is_empty() {
                let d = P.par_iter().map(|cp| cp.t.degree()).min().unwrap();
                let Pd: Vec<CriticalPair<M>> = P.drain_filter(|cp| cp.t.degree() == d).collect();
                println!(
                    "{} Iteration start d = {}, Pd.len() = {}, P.len() = {}",
                    Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                    d,
                    Pd.len(),
                    P.len()
                );
                let S = this.s_polynomials(&Pd);
                let Stilde = this.reduction(&S, &G);
                for h in Stilde {
                    G.iter().flatten().for_each(|&g| {
                        if let Some(cp) = this.critical_pair(h, g, &G) {
                            P.insert(cp);
                        }
                    });
                    G[this.sig(h).position].push(h);
                    // if G.par_iter().flatten().count() > 100 && this.terminate(&P, &G) {
                    //     break;
                    // }
                }
                println!(
                    "{} Iteration end d = {}, Pd.len() = {}, P.len() = {}",
                    Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                    d,
                    Pd.len(),
                    P.len()
                );
            }

            this.ideal = G.par_iter().flatten().map(|&f| this.poly(f)).collect();
        }

        if need_homogeneous {
            dehomogenize(&mut this.ideal);
        }

        println!(
            "{} Ideal size = {}",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            this.ideal.len()
        );
        reduce_groebner_basis(&mut this.ideal, true);
        println!(
            "{} Ideal size = {}",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            this.ideal.len()
        );
        this.ideal
    }

    pub fn terminate(&self, P: &HashSet<CriticalPair<M>>, G: &Vec<Vec<usize>>) -> bool {
        let mut I = G
            .par_iter()
            .flat_map(|v| v.into_par_iter().map(|f| self.poly(*f)))
            .collect();
        reduce_groebner_basis(&mut I, false);
        P.par_iter().all(|cp| {
            (self.is_rewritable(&cp.u, cp.k) || self.is_rewritable(&cp.v, cp.l))
                || self
                    .poly(cp.k)
                    .s_polynomial(&self.poly(cp.l))
                    .div_mod_polys(&I)
                    .1
                    .is_zero()
        })
    }

    pub fn poly(&self, i: usize) -> FastPolynomial<F, M> {
        self.L[i].1.clone()
    }

    pub fn sig(&self, i: usize) -> Signature<M> {
        self.L[i].0.clone()
    }

    pub fn add_rule(&mut self, s: &Signature<M>, k: usize) {
        let monomial = s.monomial.clone();
        let position = s.position;
        let want = match self.rules[position]
            .binary_search_by(|x| x.0.cmp(&monomial).then(Ordering::Less))
        {
            Ok(i) => i,
            Err(i) => i,
        };
        self.rules[position].insert(want, (monomial, k));
    }

    #[inline]
    pub fn find_rewriting(&self, u: &M, k: usize) -> usize {
        let Signature {
            monomial: mk,
            position: v,
        } = self.sig(k);
        let tmp = mk * u;
        self.rules[v]
            .par_iter()
            .find_map_last(|(mj, j)| (tmp.to_owned() / mj).map(|_| *j))
            .unwrap_or(k)
    }

    pub fn is_rewritable(&self, u: &M, k: usize) -> bool {
        k != self.find_rewriting(u, k)
    }

    fn is_top_reducible(&self, t: &M, ideal: &[Vec<usize>]) -> bool {
        ideal
            .par_iter()
            .flatten()
            .any(|&g| (t.clone() / &self.poly(g).leading_monomial().unwrap()).is_some())
    }

    pub fn critical_pair(&self, k: usize, l: usize, G: &[Vec<usize>]) -> Option<CriticalPair<M>> {
        let (tk, tl) = (
            self.poly(k).leading_monomial().unwrap(),
            self.poly(l).leading_monomial().unwrap(),
        );
        let t = tk.lcm(&tl);
        let (uk, ul) = ((t.clone() / &tk).unwrap(), (t.clone() / &tl).unwrap());
        let (
            Signature {
                monomial: mk,
                position: ek,
            },
            Signature {
                monomial: ml,
                position: el,
            },
        ) = (self.sig(k), self.sig(l));
        let (ukmk, ulml) = (uk.to_owned() * &mk, ul.to_owned() * &ml);

        if ek == el && ukmk == ulml {
            return None;
        }

        if self.is_top_reducible(&ukmk, &G[..ek]) || self.is_top_reducible(&ulml, &G[..el]) {
            return None;
        }

        if self.is_rewritable(&uk, k) || self.is_rewritable(&ul, l) {
            return None;
        }

        if &self.sig(k) * &uk < &self.sig(l) * &ul {
            Some(CriticalPair {
                t,
                u: ul,
                k: l,
                v: uk,
                l: k,
            })
        } else {
            Some(CriticalPair {
                t,
                u: uk,
                k,
                v: ul,
                l,
            })
        }
    }

    /// Can not be called concurrently
    pub fn s_polynomials(&mut self, P: &[CriticalPair<M>]) -> Vec<Signature<M>> {
        let mut P: Vec<Signature<M>> = P
            .par_iter()
            .filter(|cp| !self.is_rewritable(&cp.u, cp.k) && !self.is_rewritable(&cp.v, cp.l))
            .map(|cp| Signature {
                monomial: cp.u.clone(),
                position: cp.k,
            })
            .collect();

        P.par_sort_unstable();
        P.dedup();
        P
    }

    pub fn reduction(&mut self, S: &[Signature<M>], G: &[Vec<usize>]) -> Vec<usize> {
        let F: Vec<Signature<M>> = self.symbolic_preprocessing(S, G);
        let Ft: Vec<FastPolynomial<F, M>> = self.gauss_elimination(&F);

        let mut ret = Vec::new();
        for (k, p) in Ft.iter().enumerate() {
            let (u, i) = (F[k].monomial.clone(), F[k].position);
            if p.leading_monomial() == (self.poly(i) * &(F::one(), u.to_owned())).leading_monomial()
            {
                continue;
            }

            let sigma = &self.sig(i) * &u;
            self.L.push(Entry(sigma.to_owned(), p.to_owned()));
            self.add_rule(&sigma, self.L.len() - 1);
            if !p.is_zero() {
                ret.push(self.L.len() - 1);
            }
        }

        ret
    }

    pub fn find_reductor(
        &self,
        m: &M,
        sig_m: &Signature<M>,
        G: &[Vec<usize>],
        _F: &[Signature<M>],
    ) -> Option<Signature<M>> {
        G.par_iter().flatten().find_map_any(|&k| {
            if let Some(u) = m.clone() / &self.poly(k).leading_monomial().unwrap() {
                // let uk = Signature{ monomial: u, position: k, };
                // if F.contains(&uk) {
                //     return None;
                // }
                if self.is_top_reducible(&(self.sig(k).monomial * &u), &G[..self.sig(k).position]) {
                    return None;
                }
                if self.is_rewritable(&u, k) {
                    return None;
                }
                if (&self.sig(k) * &u).eq(sig_m) {
                    return None;
                }
                Some(Signature {
                    monomial: u,
                    position: k,
                })
            } else {
                None
            }
        })
    }

    fn symbolic_preprocessing(&self, S: &[Signature<M>], G: &[Vec<usize>]) -> Vec<Signature<M>> {
        let mut F = S.to_vec();
        let mut done: HashSet<M> = HashSet::new();
        let M: Vec<FastPolynomial<F, M>> = F
            .par_iter()
            .map(|s| self.poly(s.position) * &(F::one(), s.monomial.to_owned()))
            .collect();
        let mut M: HashSet<M> = HashSet::from_par_iter(
            M.par_iter()
                .flat_map(|f| f.terms().par_iter().map(|(_, m)| m).cloned()),
        );

        while M != done {
            let m = M.par_iter().filter(|m| !done.contains(m)).max().unwrap();
            done.insert(m.clone());

            let ms = F
                .par_iter()
                .filter(|k| {
                    (self.poly(k.position) * &(F::one(), k.monomial.to_owned()))
                        .terms()
                        .par_iter()
                        .any(|(_, mm)| mm == m)
                })
                .min()
                .unwrap();

            if let Some(sig) = self.find_reductor(m, ms, G, &F) {
                F.push(sig.clone());
                M.par_extend(
                    (self.poly(sig.position) * &(F::one(), sig.monomial.to_owned()))
                        .terms()
                        .par_iter()
                        .map(|(_, x)| x)
                        .cloned(),
                );
            }
        }

        F.par_sort_unstable();
        F
    }

    fn gauss_elimination(&self, F1: &[Signature<M>]) -> Vec<FastPolynomial<F, M>> {
        if !F1.is_empty() {
            let F: Vec<FastPolynomial<F, M>> = F1
                .par_iter()
                .map(|s| self.poly(s.position) * &(F::one(), s.monomial.to_owned()))
                .collect();
            let mut monomials: Vec<M> = F
                .par_iter()
                .flat_map(|f| f.terms().par_iter().map(|(_, m)| m))
                .cloned()
                .collect();
            monomials.par_sort_unstable_by(|a, b| b.cmp(a));
            monomials.dedup();
            let monomials_map: HashMap<M, usize> = HashMap::from_par_iter(
                monomials
                    .par_iter()
                    .enumerate()
                    .map(|(i, m)| (m.clone(), i)),
            );

            let (num_rows, num_cols) = (F.len(), monomials.len());

            let mut matrix: Vec<Vec<(usize, F)>> = F
                .par_iter()
                .map(|f| {
                    f.terms()
                        .iter()
                        .rev()
                        .map(|(c, m)| (*monomials_map.get(m).unwrap(), *c))
                        .collect()
                })
                .collect();

            println!(
                "{} GE start matrix {}x{}, density: {:.2}%",
                Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                num_rows,
                num_cols,
                (matrix.par_iter().map(|x| x.len()).sum::<usize>() as f64)
                    / ((num_rows * num_cols) as f64)
            );

            (0..num_cols).for_each(|c| {
                if let Some(r) = (0..num_rows)
                    .into_par_iter()
                    .find_first(|&r| !matrix[r].is_empty() && matrix[r][0].0 == c)
                {
                    let inverse = matrix[r][0].1.inverse().unwrap();
                    matrix[r].par_iter_mut().for_each(|v| v.1 *= inverse);
                    let base = matrix[r].clone();
                    matrix
                        .par_iter_mut()
                        .enumerate()
                        .filter(|(i, _)| r < *i)
                        .for_each(|(_, row)| {
                            if let Ok(j) = row.binary_search_by(|(index, _)| index.cmp(&c)) {
                                let mut result: Vec<(usize, F)> = Vec::new();
                                let mut row_iter = row.iter().peekable();
                                let mut base_iter = base.iter().peekable();
                                loop {
                                    let which = match (row_iter.peek(), base_iter.peek()) {
                                        (Some(cur), Some(other)) => Some((cur.0).cmp(&other.0)),
                                        (Some(_), None) => Some(Ordering::Less),
                                        (None, Some(_)) => Some(Ordering::Greater),
                                        (None, None) => None,
                                    };
                                    let smallest = match which {
                                        Some(Ordering::Less) => *row_iter.next().unwrap(),
                                        Some(Ordering::Equal) => {
                                            let cur = row_iter.next().unwrap();
                                            let other = base_iter.next().unwrap();
                                            (cur.0, cur.1 - row[j].1 * other.1)
                                        },
                                        Some(Ordering::Greater) => {
                                            let other = base_iter.next().unwrap();
                                            (other.0, -(row[j].1 * other.1))
                                        },
                                        None => break,
                                    };
                                    result.push(smallest);
                                }
                                result.retain(|(_, f)| !f.is_zero());
                                *row = result;
                            }
                        });
                }
            });

            println!(
                "{} GE end matrix {}x{}",
                Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                num_rows,
                num_cols
            );

            let monomials_map: HashMap<usize, M> =
                HashMap::from_par_iter(monomials_map.into_par_iter().map(|(m, i)| (i, m)));

            matrix
                .into_par_iter()
                .map(|row| FastPolynomial {
                    num_of_vars: F[0].num_of_vars,
                    terms: row
                        .into_par_iter()
                        .map(|(index, coefficient)| {
                            (coefficient, monomials_map.get(&index).unwrap().clone())
                        })
                        .rev()
                        .collect(),
                })
                .collect()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        f45::F45,
        groebner::is_groebner_basis,
        poly::{
            monomial::{Monomial, MonomialOrd, SimdMonomial},
            polynomial::{FastPolynomial, Polynomial},
        },
        DegRevLexPolynomial,
    };
    use ark_ff::{Fp, FpConfig, PrimeField, Zero};

    fn print_ideal<P: FpConfig<1>, O: MonomialOrd>(
        input_polynomials: &Vec<FastPolynomial<Fp<P, 1>, SimdMonomial<O>>>,
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
    fn test_F45_given_case_1() {
        let mut ideal: Vec<DegRevLexPolynomial> = vec![
            FastPolynomial::new(
                3,
                &vec![
                    ((1).into(), SimdMonomial::new(&vec![(1, 3)])),
                    ((1).into(), SimdMonomial::new(&vec![(0, 2)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    ((1).into(), SimdMonomial::new(&vec![(0, 2), (1, 1)])),
                    ((1).into(), SimdMonomial::new(&vec![(0, 2)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    ((1).into(), SimdMonomial::new(&vec![(0, 3)])),
                    ((-1).into(), SimdMonomial::new(&vec![(0, 2)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    ((1).into(), SimdMonomial::new(&vec![(2, 4)])),
                    ((-1).into(), SimdMonomial::new(&vec![(0, 2)])),
                    ((-1).into(), SimdMonomial::new(&vec![(1, 1)])),
                ],
            ),
        ];

        ideal = F45::new(&ideal);
        assert!(is_groebner_basis(&ideal));
    }

    #[test]
    fn test_F45_given_case_2() {
        let mut ideal: Vec<DegRevLexPolynomial> = vec![
            FastPolynomial::new(
                8,
                &vec![
                    ((-1).into(), SimdMonomial::new(&vec![(0, 1)])),
                    (
                        (9511602413006487524u64).into(),
                        SimdMonomial::new(&vec![(2, 1)]),
                    ),
                    (
                        (720575940379279356u64).into(),
                        SimdMonomial::new(&vec![(3, 1)]),
                    ),
                    (
                        (8214565720323784678u64).into(),
                        SimdMonomial::new(&vec![(4, 1)]),
                    ),
                    ((15774967588931459619u64).into(), SimdMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((-1).into(), SimdMonomial::new(&vec![(1, 1)])),
                    (
                        (10376293541461622753u64).into(),
                        SimdMonomial::new(&vec![(2, 1)]),
                    ),
                    (
                        (5188146770730811374u64).into(),
                        SimdMonomial::new(&vec![(3, 1)]),
                    ),
                    (
                        (2882303761517117431u64).into(),
                        SimdMonomial::new(&vec![(4, 1)]),
                    ),
                    ((17132433913710081944u64).into(), SimdMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    (
                        (4611686018427387891u64).into(),
                        SimdMonomial::new(&vec![(2, 1)]),
                    ),
                    (
                        (6917529027641081833u64).into(),
                        SimdMonomial::new(&vec![(3, 1)]),
                    ),
                    (
                        (6917529027641081834u64).into(),
                        SimdMonomial::new(&vec![(4, 1)]),
                    ),
                    ((2075905444283933401u64).into(), SimdMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((8).into(), SimdMonomial::new(&vec![(2, 3)])),
                    (
                        (18446744073709551543u64).into(),
                        SimdMonomial::new(&vec![(3, 3)]),
                    ),
                    ((7).into(), SimdMonomial::new(&vec![(4, 3)])),
                    (
                        (12754686725922488265u64).into(),
                        SimdMonomial::new(&vec![(5, 3)]),
                    ),
                    (
                        (885273986127101979u64).into(),
                        SimdMonomial::new(&vec![(5, 2), (6, 1)]),
                    ),
                    (
                        (15982377876049625016u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (6, 2)]),
                    ),
                    (
                        (18406871384039882698u64).into(),
                        SimdMonomial::new(&vec![(6, 3)]),
                    ),
                    (
                        (8188001519396716513u64).into(),
                        SimdMonomial::new(&vec![(5, 2), (7, 1)]),
                    ),
                    (
                        (18051588390779879373u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (6, 1), (7, 1)]),
                    ),
                    (
                        (14895699747993026510u64).into(),
                        SimdMonomial::new(&vec![(6, 2), (7, 1)]),
                    ),
                    (
                        (10147823821302792159u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (7, 2)]),
                    ),
                    (
                        (9498592991426641890u64).into(),
                        SimdMonomial::new(&vec![(6, 1), (7, 2)]),
                    ),
                    (
                        (1869547999219154938u64).into(),
                        SimdMonomial::new(&vec![(7, 3)]),
                    ),
                    (
                        (6547817342896512130u64).into(),
                        SimdMonomial::new(&vec![(5, 2)]),
                    ),
                    (
                        (9279799793654434575u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (6, 1)]),
                    ),
                    (
                        (1363756090648335788u64).into(),
                        SimdMonomial::new(&vec![(6, 2)]),
                    ),
                    (
                        (10404383306642543815u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (7, 1)]),
                    ),
                    (
                        (1940303722204995108u64).into(),
                        SimdMonomial::new(&vec![(6, 1), (7, 1)]),
                    ),
                    (
                        (6984199018677488094u64).into(),
                        SimdMonomial::new(&vec![(7, 2)]),
                    ),
                    (
                        (14719088602915975115u64).into(),
                        SimdMonomial::new(&vec![(5, 1)]),
                    ),
                    (
                        (17366754088523144755u64).into(),
                        SimdMonomial::new(&vec![(6, 1)]),
                    ),
                    (
                        (6719810797959261327u64).into(),
                        SimdMonomial::new(&vec![(7, 1)]),
                    ),
                    ((3806417358522808896u64).into(), SimdMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((56).into(), SimdMonomial::new(&vec![(2, 3)])),
                    (
                        (18446744073709551467u64).into(),
                        SimdMonomial::new(&vec![(3, 3)]),
                    ),
                    ((35).into(), SimdMonomial::new(&vec![(4, 3)])),
                    (
                        (16424627841020198849u64).into(),
                        SimdMonomial::new(&vec![(5, 3)]),
                    ),
                    (
                        (14440792205163495398u64).into(),
                        SimdMonomial::new(&vec![(5, 2), (6, 1)]),
                    ),
                    (
                        (1257630195943210991u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (6, 2)]),
                    ),
                    (
                        (3683381545235644407u64).into(),
                        SimdMonomial::new(&vec![(6, 3)]),
                    ),
                    (
                        (3082713944935104499u64).into(),
                        SimdMonomial::new(&vec![(5, 2), (7, 1)]),
                    ),
                    (
                        (6577507255774609391u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (6, 1), (7, 1)]),
                    ),
                    (
                        (13038483871191007189u64).into(),
                        SimdMonomial::new(&vec![(6, 2), (7, 1)]),
                    ),
                    (
                        (5842294616606375917u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (7, 2)]),
                    ),
                    (
                        (16008607825441849293u64).into(),
                        SimdMonomial::new(&vec![(6, 1), (7, 2)]),
                    ),
                    (
                        (11877681067236261850u64).into(),
                        SimdMonomial::new(&vec![(7, 3)]),
                    ),
                    (
                        (2315200782863393558u64).into(),
                        SimdMonomial::new(&vec![(5, 2)]),
                    ),
                    (
                        (15470057352885188411u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (6, 1)]),
                    ),
                    (
                        (7544915043732670853u64).into(),
                        SimdMonomial::new(&vec![(6, 2)]),
                    ),
                    (
                        (4152388971314589023u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (7, 1)]),
                    ),
                    (
                        (14459726586885204931u64).into(),
                        SimdMonomial::new(&vec![(6, 1), (7, 1)]),
                    ),
                    (
                        (1129956652251207029u64).into(),
                        SimdMonomial::new(&vec![(7, 2)]),
                    ),
                    (
                        (13362083559747081110u64).into(),
                        SimdMonomial::new(&vec![(5, 1)]),
                    ),
                    (
                        (9856833213872142272u64).into(),
                        SimdMonomial::new(&vec![(6, 1)]),
                    ),
                    (
                        (1336208355974708111u64).into(),
                        SimdMonomial::new(&vec![(7, 1)]),
                    ),
                    ((8675376255439944221u64).into(), SimdMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((280).into(), SimdMonomial::new(&vec![(2, 3)])),
                    (
                        (18446744073709551123u64).into(),
                        SimdMonomial::new(&vec![(3, 3)]),
                    ),
                    ((155).into(), SimdMonomial::new(&vec![(4, 3)])),
                    (
                        (12393906174523604947u64).into(),
                        SimdMonomial::new(&vec![(5, 3)]),
                    ),
                    (
                        (9079256848778919915u64).into(),
                        SimdMonomial::new(&vec![(5, 2), (6, 1)]),
                    ),
                    (
                        (4683743612465315821u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (6, 2)]),
                    ),
                    (
                        (14591662792680406994u64).into(),
                        SimdMonomial::new(&vec![(6, 3)]),
                    ),
                    (
                        (17149707381026848712u64).into(),
                        SimdMonomial::new(&vec![(5, 2), (7, 1)]),
                    ),
                    (
                        (1297036692682702845u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (6, 1), (7, 1)]),
                    ),
                    (
                        (4287426845256712178u64).into(),
                        SimdMonomial::new(&vec![(6, 2), (7, 1)]),
                    ),
                    (
                        (1224979098644774908u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (7, 2)]),
                    ),
                    (
                        (17834254524387164103u64).into(),
                        SimdMonomial::new(&vec![(6, 1), (7, 2)]),
                    ),
                    (
                        (9691746398101307361u64).into(),
                        SimdMonomial::new(&vec![(7, 3)]),
                    ),
                    (
                        (13414881433324831873u64).into(),
                        SimdMonomial::new(&vec![(5, 2)]),
                    ),
                    (
                        (5031862640384719684u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (6, 1)]),
                    ),
                    (
                        (17188778413613371636u64).into(),
                        SimdMonomial::new(&vec![(6, 2)]),
                    ),
                    (
                        (1916411633332118839u64).into(),
                        SimdMonomial::new(&vec![(5, 1), (7, 1)]),
                    ),
                    (
                        (8265166220188716359u64).into(),
                        SimdMonomial::new(&vec![(6, 1), (7, 1)]),
                    ),
                    (
                        (13903501327901167912u64).into(),
                        SimdMonomial::new(&vec![(7, 2)]),
                    ),
                    (
                        (3596302261513828044u64).into(),
                        SimdMonomial::new(&vec![(5, 1)]),
                    ),
                    (
                        (16648592942952637535u64).into(),
                        SimdMonomial::new(&vec![(6, 1)]),
                    ),
                    (
                        (13433124499900667401u64).into(),
                        SimdMonomial::new(&vec![(7, 1)]),
                    ),
                    ((11749052441434960042u64).into(), SimdMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((280).into(), SimdMonomial::new(&vec![(5, 3)])),
                    (
                        (18446744073709551123u64).into(),
                        SimdMonomial::new(&vec![(6, 3)]),
                    ),
                    ((155).into(), SimdMonomial::new(&vec![(7, 3)])),
                    ((277610931875235913u64).into(), SimdMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                8,
                &vec![
                    ((56).into(), SimdMonomial::new(&vec![(5, 3)])),
                    (
                        (18446744073709551467u64).into(),
                        SimdMonomial::new(&vec![(6, 3)]),
                    ),
                    ((35).into(), SimdMonomial::new(&vec![(7, 3)])),
                    ((2084698901943657590u64).into(), SimdMonomial::new(&vec![])),
                ],
            ),
        ];

        ideal = F45::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));

        ideal.push(FastPolynomial::new(
            8,
            &vec![
                ((1).into(), SimdMonomial::new(&vec![(2, 1)])),
                ((1).into(), SimdMonomial::new(&vec![(3, 1)])),
            ],
        ));

        ideal = F45::new(&ideal);

        let want: Vec<DegRevLexPolynomial> = vec![FastPolynomial::new(
            8,
            &vec![((1).into(), SimdMonomial::new(&vec![]))],
        )];
        assert_eq!(want, ideal);
    }

    #[test]
    fn test_F45_given_case_3() {
        type MPolynomial = DegRevLexPolynomial;
        let mut ideal: Vec<MPolynomial> = vec![
            FastPolynomial::new(
                3,
                &vec![
                    // f1 = x1^2 + x1*x2 - x3
                    ((1).into(), SimdMonomial::new(&vec![(0, 2)])),
                    ((1).into(), SimdMonomial::new(&vec![(0, 1), (1, 1)])),
                    ((-1).into(), SimdMonomial::new(&vec![(2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // f2 = x1*x3 - x2^2
                    ((1).into(), SimdMonomial::new(&vec![(0, 1), (2, 1)])),
                    ((-1).into(), SimdMonomial::new(&vec![(1, 2)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // f3 = x1x2^2 - x2x3
                    ((1).into(), SimdMonomial::new(&vec![(0, 1), (1, 2)])),
                    ((-1).into(), SimdMonomial::new(&vec![(1, 1), (2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // f4 = x2^2*x3 - x1^3
                    ((1).into(), SimdMonomial::new(&vec![(1, 2), (2, 1)])),
                    ((-1).into(), SimdMonomial::new(&vec![(0, 3)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // F45 = x2^3 - x1^2*x3
                    ((1).into(), SimdMonomial::new(&vec![(1, 3)])),
                    ((-1).into(), SimdMonomial::new(&vec![(0, 2), (2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // f6 = x1^3*x3 - x3^3
                    ((1).into(), SimdMonomial::new(&vec![(0, 3), (2, 1)])),
                    ((-1).into(), SimdMonomial::new(&vec![(2, 3)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // f7 = x1^2x2 - x2^2x3
                    ((1).into(), SimdMonomial::new(&vec![(0, 2), (1, 1)])),
                    ((-1).into(), SimdMonomial::new(&vec![(1, 2), (2, 1)])),
                ],
            ),
        ];

        ideal = F45::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));
    }

    #[test]
    fn test_F45_given_case_4() {
        type MPolynomial = DegRevLexPolynomial;
        let mut ideal: Vec<MPolynomial> = vec![
            FastPolynomial::new(
                4,
                &vec![
                    // x^2*y - z^2*t
                    ((1).into(), SimdMonomial::new(&vec![(0, 2), (1, 1)])),
                    ((-1).into(), SimdMonomial::new(&vec![(2, 2), (3, 1)])),
                ],
            ),
            FastPolynomial::new(
                4,
                &vec![
                    // x*z^2 - y^2*t
                    ((1).into(), SimdMonomial::new(&vec![(0, 1), (2, 2)])),
                    ((-1).into(), SimdMonomial::new(&vec![(1, 2), (3, 1)])),
                ],
            ),
            FastPolynomial::new(
                4,
                &vec![
                    // y*z^3 - x^2*t^2
                    ((1).into(), SimdMonomial::new(&vec![(1, 1), (2, 3)])),
                    ((-1).into(), SimdMonomial::new(&vec![(0, 2), (3, 2)])),
                ],
            ),
        ];

        ideal = F45::new(&ideal);
        assert!(is_groebner_basis(&ideal));
        print_ideal(&ideal);
    }
}
