use crate::{
    groebner::{dehomogenize, homogenize, is_homogeneous_ideal, reduce_groebner_basis},
    poly::{
        monomial::Monomial,
        polynomial::{FastPolynomial, Polynomial},
    },
    Entry,
};
use ark_ff::{Field, Zero};
use ark_std::iterable::Iterable;
use rayon::prelude::*;
use std::{
    borrow::BorrowMut,
    cmp,
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, VecDeque},
    iter,
    ops::Mul,
};

#[derive(Debug, Clone)]
pub struct Signature<M>
where
    M: Monomial,
{
    position: usize,
    monomial: M,
}

impl<M: Monomial> PartialEq for Signature<M> {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position && self.monomial == other.monomial
    }
}

impl<M: Monomial> Eq for Signature<M> {}

impl<M: Monomial> PartialOrd for Signature<M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Self::cmp(self, other))
    }
}

impl<M: Monomial> Ord for Signature<M> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .position
            .cmp(&self.position)
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriticalPair<M>
where
    M: Monomial,
{
    lcm: M,
    u1: M,
    r1: usize,
    u2: M,
    r2: usize,
}

impl<M: Monomial> PartialOrd for CriticalPair<M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Self::cmp(self, other))
    }
}

impl<M: Monomial> Ord for CriticalPair<M> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.lcm.degree().cmp(&other.lcm.degree())
    }
}

#[inline]
fn f_ord_wrt_signature<M: Monomial>(sig: Signature<M>) -> cmp::Reverse<Signature<M>> {
    cmp::Reverse(sig)
}

#[inline]
fn critical_pair_ord_wrt_degree<M: Monomial>(cp: &CriticalPair<M>) -> cmp::Reverse<M> {
    cmp::Reverse(cp.lcm.clone())
}

pub struct F5<F: Field, M: Monomial> {
    pub ideal: Vec<FastPolynomial<F, M>>,
    pub N: usize,
    pub Rs: HashMap<usize, Entry<Signature<M>, FastPolynomial<F, M>>>,
    pub Gs: HashMap<usize, Vec<usize>>,
    pub rules: HashMap<usize, VecDeque<(M, usize)>>,
}

impl<F: Field, M: Monomial> F5<F, M> {
    pub fn new(input: &[FastPolynomial<F, M>]) -> Vec<FastPolynomial<F, M>> {
        let mut this = Self {
            ideal: iter::once(FastPolynomial::zero())
                .chain(input.iter().cloned())
                .collect(),
            N: input.len(),
            Rs: HashMap::new(),
            Gs: HashMap::new(),
            rules: HashMap::new(),
        };

        let need_homogeneous = !is_homogeneous_ideal(&this.ideal);
        if need_homogeneous {
            homogenize(&mut this.ideal);
        }

        {
            // F5 main function
            let m = this.N;
            (1..=m).into_iter().for_each(|i| {
                this.rules.insert(i, VecDeque::new());
            });
            this.Rs.insert(
                m,
                Entry(
                    Signature {
                        position: m,
                        monomial: M::one(),
                    },
                    this.ideal[m].clone(),
                ),
            );
            this.Gs.insert(m, vec![m]);
            for i in (1..=(m - 1)).rev() {
                println!("Start iteration {}", i);
                this.AlgorithmF5(i);

                if this
                    .Gs
                    .get(&i)
                    .unwrap()
                    .par_iter()
                    .any(|&index| this.poly_r(index).is_zero())
                {
                    return vec![FastPolynomial::new(
                        this.ideal[0].num_of_vars(),
                        &[(F::one(), M::one())],
                    )];
                }
            }
            this.ideal = this
                .Gs
                .get(&1)
                .unwrap()
                .par_iter()
                .map(|&i| this.poly_r(i).clone())
                .collect();
        }

        if need_homogeneous {
            dehomogenize(&mut this.ideal);
        }
        reduce_groebner_basis(&mut this.ideal, true);

        this.ideal
    }

    pub fn poly_r(&self, i: usize) -> &FastPolynomial<F, M> {
        &self.Rs.get(&i).unwrap().1
    }

    pub fn S_r(&self, i: usize) -> &Signature<M> {
        &self.Rs.get(&i).unwrap().0
    }

    pub fn index_r(&self, i: usize) -> usize {
        self.Rs.get(&i).unwrap().0.position
    }

    pub fn normal_form(&self, f: &FastPolynomial<F, M>, i: usize) -> FastPolynomial<F, M> {
        f.terms()
            .par_iter()
            .map(|term| {
                let mut remainder = FastPolynomial::new(f.num_of_vars(), &[term.clone()]);

                while !remainder.is_zero() {
                    let basis = self.Gs.get(&i).unwrap().par_iter().map(|&r| self.poly_r(r));
                    if let Some(delta) = basis.find_map_any(|divisor| {
                        let (r_coefficient, r_monomial) = remainder.leading_term().unwrap();
                        let (g_coefficient, g_monomial) = divisor.leading_term().unwrap();

                        (r_monomial / &g_monomial).map(|t_monomial| {
                            divisor.clone() * &(r_coefficient / g_coefficient, t_monomial)
                        })
                    }) {
                        remainder -= &delta;
                    } else {
                        break;
                    }
                }

                remainder
            })
            .reduce(FastPolynomial::zero, |a, b| a + b)
    }

    pub fn add_rule(&mut self, k: usize) {
        let i = self.index_r(k);
        let new_rule = (self.S_r(k).monomial.clone(), k);
        self.rules
            .entry(i)
            .or_insert_with(VecDeque::new)
            .push_front(new_rule);
    }

    #[inline]
    pub fn rewritten(&self, u: &M, k: usize) -> (M, usize) {
        let t = &self.S_r(k).monomial;
        let i = self.index_r(k);
        let ut = u.clone() * t;

        self.rules
            .get(&i)
            .unwrap()
            .par_iter()
            .find_map_first(|(t_i, k_i)| (ut.to_owned() / t_i).map(|quotient| (quotient, *k_i)))
            .unwrap_or_else(|| (u.clone(), k))
    }

    pub fn is_rewritten(&self, u: &M, k: usize) -> bool {
        let (_, l) = self.rewritten(u, k);
        l != k
    }

    pub fn is_rewritten_optimized(&self, u: &M, k: usize) -> bool {
        let t = &self.S_r(k).monomial;
        let i = self.index_r(k);
        let ut = u.clone() * t;

        k != self
            .rules
            .get(&i)
            .unwrap()
            .par_iter()
            .find_map_first(|(t_i, k_i)| (ut.to_owned() / t_i).map(|_| (*k_i)))
            .unwrap_or(k)
    }

    pub fn AlgorithmF5(&mut self, i: usize) {
        self.Rs.insert(
            i,
            Entry(
                Signature {
                    position: i,
                    monomial: M::one(),
                },
                self.ideal[i].clone(),
            ),
        );
        self.Gs.insert(
            i,
            iter::once(i)
                .chain(self.Gs.get(&(i + 1)).unwrap().iter().copied())
                .collect(),
        );

        let mut P: BinaryHeap<Entry<cmp::Reverse<M>, CriticalPair<M>>> = BinaryHeap::from_par_iter(
            self.Gs
                .get(&(i + 1))
                .unwrap()
                .par_iter()
                .map(|&r| self.critical_pair(i, r, i))
                .filter_map(|e| e)
                .map(|cp| Entry(critical_pair_ord_wrt_degree(&cp), cp)),
        );

        while !P.is_empty() {
            let d = P.peek().unwrap().1.lcm.degree();
            let mut P_d = Vec::new();
            while let Some(Entry(_, cp)) = P.peek() {
                if cp.lcm.degree() == d {
                    let Entry(_, cp) = P.pop().unwrap();
                    P_d.push(cp);
                }
            }

            let F = self.s_polynomials(&P_d);
            let R_d = self.reduction(&F, i);
            for r in R_d {
                P.par_extend(
                    self.Gs
                        .get(&i)
                        .unwrap()
                        .par_iter()
                        .map(|&p| self.critical_pair(r, p, i))
                        .filter_map(|e| e)
                        .map(|cp| Entry(critical_pair_ord_wrt_degree(&cp), cp)),
                );
                self.Gs.get_mut(&i).unwrap().push(r);
            }
        }
    }

    pub fn critical_pair(&self, r1: usize, r2: usize, k: usize) -> Option<CriticalPair<M>> {
        let (S_r1, S_r2) = (self.S_r(r1), self.S_r(r2));
        if S_r1.lt(S_r2) {
            return self.critical_pair(r2, r1, k);
        }

        if self.index_r(r1) > k {
            return None;
        }

        let (p1, p2) = (self.poly_r(r1), self.poly_r(r2));
        let t = p1
            .leading_monomial()
            .unwrap()
            .lcm(&p2.leading_monomial().unwrap());
        let u1 = (t.clone() / &p1.leading_monomial().unwrap()).unwrap();
        let u2 = (t.clone() / &p2.leading_monomial().unwrap()).unwrap();
        let u1t1 =
            FastPolynomial::new(p1.num_of_vars(), &[(F::one(), S_r1.monomial.clone() * &u1)]);
        if self.normal_form(&u1t1, k + 1).ne(&u1t1) {
            return None;
        }
        let u2t2 =
            FastPolynomial::new(p2.num_of_vars(), &[(F::one(), S_r2.monomial.clone() * &u2)]);
        if self.index_r(r2) == k && self.normal_form(&u2t2, k + 1).ne(&u2t2) {
            return None;
        }

        Some(CriticalPair {
            lcm: t,
            u1,
            r1,
            u2,
            r2,
        })
    }

    /// Can not be called concurrently
    pub fn s_polynomials(&mut self, P: &[CriticalPair<M>]) -> Vec<usize> {
        let mut F: Vec<usize> = P
            .par_iter()
            .filter(|cp| {
                !self.is_rewritten_optimized(&cp.u1, cp.r1)
                    && !self.is_rewritten_optimized(&cp.u2, cp.r2)
            })
            .cloned()
            .collect::<Vec<CriticalPair<M>>>()
            .iter()
            .map(|cp| {
                self.N += 1;
                self.Rs.insert(
                    self.N,
                    Entry(
                        self.S_r(cp.r1) * &cp.u1,
                        (self.poly_r(cp.r1).clone() * &(F::one(), cp.u1.clone()))
                            - &(self.poly_r(cp.r2).clone() * &(F::one(), cp.u2.clone())),
                    ),
                );
                self.add_rule(self.N);
                self.N
            })
            .collect();

        F.sort_unstable_by(|&a, &b| self.S_r(a).cmp(self.S_r(b)));
        F
    }

    pub fn reduction(&mut self, todo: &[usize], k: usize) -> Vec<usize> {
        let mut todo: BinaryHeap<Entry<cmp::Reverse<Signature<M>>, usize>> =
            BinaryHeap::from_par_iter(
                todo.par_iter()
                    .map(|&i| Entry(f_ord_wrt_signature(self.S_r(i).clone()), i)),
            );
        let mut done = Vec::new();
        let mut G = self.Gs.get(&k).unwrap().clone();

        while let Some(Entry(_, h)) = todo.pop() {
            *self.Rs.get_mut(&h).unwrap().1.borrow_mut() = self.normal_form(self.poly_r(h), k + 1);
            let (h_, todo_) = self.top_reduction(h, &G, k);
            done.extend(&h_);
            G.extend(&h_);
            todo.par_extend(
                todo_
                    .par_iter()
                    .map(|&i| Entry(f_ord_wrt_signature(self.S_r(i).clone()), i)),
            );
        }

        done
    }

    pub fn is_reducible(&self, i_0: usize, G: &[usize], k: usize) -> Option<usize> {
        G.par_iter()
            .find_any(|&&i_j| {
                let t_j = self.S_r(i_j).monomial.clone();

                if let Some(u) = self.poly_r(i_0).leading_monomial().unwrap()
                    / &self.poly_r(i_j).leading_monomial().unwrap()
                {
                    let ut_j = FastPolynomial::new(0, &[(F::one(), u.clone() * &t_j)]);

                    self.normal_form(&ut_j, k + 1) == ut_j
                        && !self.is_rewritten_optimized(&u, i_j)
                        && (self.S_r(i_j) * &u).ne(self.S_r(i_0))
                } else {
                    false
                }
            })
            .copied()
    }

    pub fn top_reduction(&mut self, k0: usize, G: &[usize], k: usize) -> (Vec<usize>, Vec<usize>) {
        if self.poly_r(k0).is_zero() {
            println!("the system is not a regular sequence");
            return (Vec::new(), Vec::new());
        }

        if let Some(r_) = self.is_reducible(k0, G, k) {
            let u = (self.poly_r(k0).leading_monomial().unwrap()
                / &self.poly_r(r_).leading_monomial().unwrap())
                .unwrap();
            if (self.S_r(r_) * &u).lt(self.S_r(k0)) {
                let tmp = self.poly_r(r_).clone() * &(F::one(), u);
                *self.Rs.get_mut(&k0).unwrap().1.borrow_mut() -= &tmp;

                (Vec::new(), vec![k0])
            } else {
                self.N += 1;
                self.Rs.insert(
                    self.N,
                    Entry(
                        self.S_r(r_) * &u,
                        (self.poly_r(r_).clone() * &(F::one(), u.clone())) - self.poly_r(k0),
                    ),
                );
                self.add_rule(self.N);

                (Vec::new(), vec![self.N, k0])
            }
        } else {
            let f = self.poly_r(k0);
            *self.Rs.get_mut(&k0).unwrap().1.borrow_mut() = f.clone()
                * &(
                    f.leading_coefficient().unwrap().inverse().unwrap(),
                    M::one(),
                );

            (vec![k0], Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        f5::F5,
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
    fn test_f5_given_case_1() {
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

        ideal = F5::new(&ideal);
        assert!(is_groebner_basis(&ideal));
    }

    #[test]
    fn test_f5_given_case_2() {
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

        ideal = F5::new(&ideal);
        assert!(is_groebner_basis(&ideal));

        ideal.push(FastPolynomial::new(
            8,
            &vec![
                ((1).into(), SimdMonomial::new(&vec![(2, 1)])),
                ((1).into(), SimdMonomial::new(&vec![(3, 1)])),
            ],
        ));

        ideal = F5::new(&ideal);

        let want: Vec<DegRevLexPolynomial> = vec![FastPolynomial::new(
            8,
            &vec![((1).into(), SimdMonomial::new(&vec![]))],
        )];
        assert_eq!(want, ideal);
    }

    #[test]
    fn test_f5_given_case_3() {
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
                    // f5 = x2^3 - x1^2*x3
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

        ideal = F5::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));
    }
}
