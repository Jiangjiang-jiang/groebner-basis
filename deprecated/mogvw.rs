use crate::{
    groebner::reduce_groebner_basis,
    poly::{
        monomial::Monomial,
        polynomial::{FastPolynomial, Polynomial},
    },
};
use ark_ff::{Field, Zero};
use chrono::Local;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    ops::{Div, Mul},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Signature<M>
where
    M: Monomial,
{
    monomial: M,
    position: usize,
}

impl<M: Monomial> Signature<M> {
    fn e(i: usize) -> Self {
        Self {
            monomial: M::one(),
            position: i,
        }
    }
}

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

impl<'a, 'b, M: Monomial> Div<&'a Signature<M>> for &'b Signature<M> {
    type Output = Option<M>;

    fn div(self, rhs: &'a Signature<M>) -> Self::Output {
        if self.position == rhs.position {
            return self.monomial.to_owned() / &rhs.monomial;
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MSignature<F, M>
where
    F: Field,
    M: Monomial,
{
    signature: Signature<M>,
    polynomial: FastPolynomial<F, M>,
}

impl<F: Field, M: Monomial> PartialOrd for MSignature<F, M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Self::cmp(self, other))
    }
}

impl<F: Field, M: Monomial> Ord for MSignature<F, M> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.signature.cmp(&other.signature)
    }
}

impl<'a, 'b, F: Field, M: Monomial> Mul<&'a M> for &'b MSignature<F, M> {
    type Output = MSignature<F, M>;

    fn mul(self, rhs: &'a M) -> Self::Output {
        let mut result = self.clone();
        result.signature = &result.signature * rhs;
        result.polynomial *= &(F::one(), rhs.to_owned());
        result
    }
}

/// mo-GVW
/// Paper "A Monomial-Oriented GVW for Computing Groebner Bases"
/// https://arxiv.org/pdf/1410.0105v1.pdf
pub struct moGVW<F: Field, M: Monomial> {
    pub ideal: Vec<FastPolynomial<F, M>>,
    pub n: usize,
    pub lifted: HashSet<M>,
    pub G: HashMap<M, MSignature<F, M>>,
}

impl<F: Field, M: Monomial> moGVW<F, M> {
    pub fn new(input: &[FastPolynomial<F, M>]) -> Vec<FastPolynomial<F, M>> {
        let mut this = Self {
            ideal: input.to_vec(),
            n: input.par_iter().map(|f| f.num_of_vars).max().unwrap(),
            lifted: HashSet::new(),
            G: HashMap::new(),
        };

        reduce_groebner_basis(&mut this.ideal, false);

        {
            // moGVW main function

            this.G
                .par_extend(this.ideal.par_iter().enumerate().map(|(i, f)| {
                    (
                        f.leading_monomial().unwrap(),
                        MSignature {
                            signature: Signature::e(i),
                            polynomial: f.clone(),
                        },
                    )
                }));

            let mut liftdeg = this
                .G
                .par_iter()
                .filter(|&(m, uf)| Self::is_primitive(m, uf))
                .map(|(m, _)| m.degree())
                .max()
                .unwrap();
            let mut mindeg = this
                .G
                .par_iter()
                .filter(|x| !this.is_lifted(x.0))
                .map(|(m, _)| m.degree())
                .min()
                .unwrap();

            loop {
                while mindeg <= liftdeg {
                    let todo: HashMap<M, MSignature<F, M>> = this
                        .G
                        .par_iter()
                        .filter(|&(m, _)| m.degree() == mindeg && !this.is_lifted(m))
                        .map(|(a, b)| (a.clone(), b.clone()))
                        .collect();

                    println!(
                        "{} mindeg: {}, liftdeg: {}, todo.len(): {} begin.",
                        Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                        mindeg,
                        liftdeg,
                        todo.len()
                    );

                    let mut H: HashSet<MSignature<F, M>> = this.lift(todo);
                    this.append(&mut H);
                    let P: Vec<MSignature<F, M>> = this.eliminate(H);
                    this.update(P);

                    println!(
                        "{} mindeg: {}, liftdeg: {} end.",
                        Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                        mindeg,
                        liftdeg
                    );

                    liftdeg = this
                        .G
                        .par_iter()
                        .filter(|&(m, uf)| Self::is_primitive(m, uf))
                        .map(|(m, _)| m.degree())
                        .max()
                        .unwrap();
                    mindeg = this
                        .G
                        .par_iter()
                        .filter(|x| !this.is_lifted(x.0))
                        .map(|(m, _)| m.degree())
                        .min()
                        .unwrap();
                }

                let maxcpdeg = this
                    .G
                    .par_iter()
                    .filter(|(a, b)| Self::is_primitive(a, b))
                    .flat_map(|(lmf, _)| {
                        this.G
                            .par_iter()
                            .filter(|(a, b)| Self::is_primitive(a, b))
                            .map(|(lmg, _)| lmg.lcm(lmf).degree())
                    })
                    .max()
                    .unwrap();
                if maxcpdeg > liftdeg + 1 {
                    liftdeg = maxcpdeg - 1;
                } else {
                    break;
                }
            }

            this.ideal = this
                .G
                .into_par_iter()
                .filter(|(m, uf)| Self::is_primitive(m, uf))
                .map(|(_, v)| v.polynomial)
                .collect();
            println!(
                "{} Produce ideal.len(): {}",
                Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                this.ideal.len()
            );
        }

        reduce_groebner_basis(&mut this.ideal, true);
        this.ideal
    }

    pub fn is_primitive(m: &M, uf: &MSignature<F, M>) -> bool {
        !m.is_constant() && *m == uf.polynomial.leading_monomial().unwrap()
    }

    pub fn is_lifted(&self, x: &M) -> bool {
        self.lifted.contains(x)
    }

    pub fn set_lifted(&mut self, x: M) {
        self.lifted.insert(x);
    }

    pub fn lcm_criterion_rejected(&self, m: &M, uf: &MSignature<F, M>) -> bool {
        if let Some(vg) = self.G.get(m) {
            let lmf = uf.polynomial.leading_monomial().unwrap();
            let lmg = vg.polynomial.leading_monomial().unwrap();

            if *m == lmf.lcm(&lmg) {
                false
            } else {
                let left = &uf.signature * &(m.to_owned() / &lmf).unwrap();
                let right = &vg.signature * &(m.to_owned() / &lmg).unwrap();
                left > right
            }
        } else {
            false
        }
    }

    pub fn syzygy_criterion_rejected(&self, m: &M, uf: &MSignature<F, M>) -> bool {
        let t_f = (m.to_owned() / &uf.polynomial.leading_monomial().unwrap()).unwrap();

        if let Some(vg) = self.G.get(&(uf.signature.monomial.clone() * &t_f)) {
            uf.signature.position > vg.signature.position
        } else {
            false
        }
    }

    pub fn rewritten_criterion_rejected(&self, _m: &M, uf: &MSignature<F, M>) -> bool {
        self.G
            .par_iter()
            .find_any(|&(_, vg)| {
                if let Some(t) = &uf.signature / &vg.signature {
                    t * &vg.polynomial.leading_monomial().unwrap()
                        < uf.polynomial.leading_monomial().unwrap()
                } else {
                    false
                }
            })
            .is_some()
    }

    pub fn lift(&mut self, todo: HashMap<M, MSignature<F, M>>) -> HashSet<MSignature<F, M>> {
        let mut H: HashSet<MSignature<F, M>> = HashSet::new();

        for (m, uf) in todo {
            for xi in (0..self.n).map(|i| M::new(&[(i, 1)])) {
                let xim = xi * &m;
                if let Some(vg) = self.G.get(&xim).cloned() {
                    let tf = (xim.to_owned() / &uf.polynomial.leading_monomial().unwrap()).unwrap();
                    let tg = (xim.to_owned() / &vg.polynomial.leading_monomial().unwrap()).unwrap();
                    let tf_lmu = &uf.signature * &tf;
                    let tg_lmv = &vg.signature * &tg;
                    if tf_lmu > tg_lmv
                        && !self.lcm_criterion_rejected(&xim, &uf)
                        && !self.syzygy_criterion_rejected(&xim, &uf)
                        && !self.rewritten_criterion_rejected(&xim, &uf)
                    {
                        H.insert(uf.mul(&tf));
                        H.insert(vg.mul(&tg));
                    }
                    if tf_lmu < tg_lmv {
                        self.G.insert(xim.clone(), uf.clone());
                        if !self.lcm_criterion_rejected(&xim, &vg)
                            && !self.syzygy_criterion_rejected(&xim, &vg)
                            && !self.rewritten_criterion_rejected(&xim, &vg)
                        {
                            H.insert(uf.mul(&tf));
                            H.insert(vg.mul(&tg));
                        }
                    }
                } else {
                    self.G.insert(xim, uf.clone());
                }
            }
            self.set_lifted(m);
        }

        H
    }

    fn append(&mut self, H: &mut HashSet<MSignature<F, M>>) {
        let mut done: HashSet<M> = HashSet::from_par_iter(
            H.par_iter()
                .map(|wh| wh.polynomial.leading_monomial().unwrap()),
        );

        while let Some(m) = H
            .par_iter()
            .flat_map(|h| h.polynomial.trailing_terms().par_iter().map(|(_, m)| m))
            .find_any(|m| !done.contains(*m))
        {
            done.insert(m.clone());

            if let Some(vg) = self.G.get(m) {
                H.insert(vg * &(m.clone() / &vg.polynomial.leading_monomial().unwrap()).unwrap());
            }
        }
    }

    fn eliminate(&self, H: HashSet<MSignature<F, M>>) -> Vec<MSignature<F, M>> {
        let mut P: Vec<MSignature<F, M>> = H.into_par_iter().collect();
        P.par_sort_unstable();
        let mut monomials: Vec<M> = P
            .par_iter()
            .flat_map(|p| p.polynomial.terms().par_iter().map(|(_, m)| m.clone()))
            .collect();
        monomials.par_sort_unstable_by(|a, b| b.cmp(a));
        monomials.dedup();

        let (num_rows, num_cols) = (P.len(), monomials.len());

        println!(
            "{} matrix elimination size: {}x{} begin. density: {:.2}%",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            num_rows,
            num_cols,
            ((P.par_iter()
                .map(|p| p.polynomial.terms().len())
                .sum::<usize>() as f64) * 100.0)
                / ((num_rows * num_cols) as f64)
        );

        (0..num_cols).for_each(|c| {
            if let Some(r) = (0..num_rows).into_par_iter().find_first(|&r| {
                !P[r].polynomial.is_zero()
                    && P[r].polynomial.leading_monomial().unwrap() == monomials[c]
            }) {
                let inverse = P[r]
                    .polynomial
                    .leading_coefficient()
                    .unwrap()
                    .inverse()
                    .unwrap();
                P[r].polynomial *= &(inverse, M::one());
                let base = P[r].clone();
                P.par_iter_mut()
                    .enumerate()
                    .filter(|(i, row)| {
                        r < *i
                            && !row.polynomial.is_zero()
                            && row.polynomial.leading_monomial().unwrap() == monomials[c]
                    })
                    .for_each(|(_, row)| {
                        let inverse = row
                            .polynomial
                            .leading_coefficient()
                            .unwrap()
                            .inverse()
                            .unwrap();
                        row.polynomial *= &(inverse, M::one());
                        row.polynomial -= &base.polynomial;
                        row.signature =
                            Signature::max(row.signature.to_owned(), base.signature.to_owned());
                    });
            }
        });

        println!(
            "{} matrix elimination size: {}x{} end.",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            num_rows,
            num_cols
        );

        P
    }

    fn update(&mut self, P: Vec<MSignature<F, M>>) {
        for wh in P.into_iter().filter(|wh| !wh.polynomial.is_zero()) {
            let lmh = wh.polynomial.leading_monomial().unwrap();
            if let Some(vg) = self.G.get(&lmh) {
                if &vg.signature
                    * &(lmh.to_owned() / &vg.polynomial.leading_monomial().unwrap()).unwrap()
                    > wh.signature
                {
                    self.G.insert(lmh, wh);
                }
            } else {
                self.G.insert(lmh, wh);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        groebner::is_groebner_basis,
        mogvw::moGVW,
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
    fn test_GVW_given_case_1() {
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

        ideal = moGVW::new(&ideal);
        assert!(is_groebner_basis(&ideal));
        print_ideal(&ideal);
    }

    #[test]
    fn test_GVW_given_case_2() {
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

        ideal = moGVW::new(&ideal);
        assert!(is_groebner_basis(&ideal));
        print_ideal(&ideal);
    }

    #[test]
    fn test_GVW_given_case_3() {
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
                    // GVW = x2^3 - x1^2*x3
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

        ideal = moGVW::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));
    }

    #[test]
    fn test_GVW_given_case_4() {
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

        ideal = moGVW::new(&ideal);
        assert!(is_groebner_basis(&ideal));
        print_ideal(&ideal);
    }

    #[test]
    fn test_GVW_given_case_5() {
        type MPolynomial = DegRevLexPolynomial;
        let mut ideal: Vec<MPolynomial> = vec![
            FastPolynomial::new(
                3,
                &vec![
                    // x*y*z - 1
                    ((1).into(), SimdMonomial::new(&vec![(0, 1), (1, 1), (2, 1)])),
                    ((-1).into(), SimdMonomial::new(&vec![])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // x*y - z
                    ((1).into(), SimdMonomial::new(&vec![(0, 1), (1, 1)])),
                    ((-1).into(), SimdMonomial::new(&vec![(2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // y*z - y
                    ((1).into(), SimdMonomial::new(&vec![(1, 1), (2, 1)])),
                    ((-1).into(), SimdMonomial::new(&vec![(1, 1)])),
                ],
            ),
        ];

        ideal = moGVW::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));
    }

    #[test]
    fn test_GVW_given_case_6() {
        type MPolynomial = DegRevLexPolynomial;
        let mut ideal: Vec<MPolynomial> = vec![
            FastPolynomial::new(
                3,
                &vec![
                    // x*y + z
                    ((1).into(), SimdMonomial::new(&vec![(0, 1), (1, 1)])),
                    ((1).into(), SimdMonomial::new(&vec![(2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // x^2
                    ((1).into(), SimdMonomial::new(&vec![(0, 2)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // z
                    ((1).into(), SimdMonomial::new(&vec![(2, 1)])),
                ],
            ),
        ];

        ideal = moGVW::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));
    }

    #[test]
    fn test_GVW_given_case_7() {
        type MPolynomial = DegRevLexPolynomial;
        let mut ideal: Vec<MPolynomial> = vec![
            FastPolynomial::new(
                3,
                &vec![
                    // x*y*z
                    ((1).into(), SimdMonomial::new(&vec![(0, 1), (1, 1), (2, 1)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // z^4 + y^2 + x
                    ((1).into(), SimdMonomial::new(&vec![(0, 1)])),
                    ((1).into(), SimdMonomial::new(&vec![(1, 2)])),
                    ((1).into(), SimdMonomial::new(&vec![(2, 4)])),
                ],
            ),
            FastPolynomial::new(
                3,
                &vec![
                    // y^5 + z^5
                    ((1).into(), SimdMonomial::new(&vec![(2, 5)])),
                    ((1).into(), SimdMonomial::new(&vec![(1, 5)])),
                ],
            ),
        ];

        ideal = moGVW::new(&ideal);
        print_ideal(&ideal);
        assert!(is_groebner_basis(&ideal));
    }
}
