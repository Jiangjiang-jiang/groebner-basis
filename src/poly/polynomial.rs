use ark_ff::{Field, Zero};
use derivative::Derivative;
use std::{
    cmp::Ordering,
    fmt,
    fmt::Debug,
    hash::Hash,
    iter,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    vec::Vec,
};

use crate::poly::monomial::Monomial;

pub trait Polynomial<F: Field, M: Monomial>:
    Sized
    + Clone
    + Debug
    + Hash
    + PartialEq
    + Eq
    + Add
    + Neg
    + Zero
    + Send
    + Sync
    + for<'a> AddAssign<&'a Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a (F, M)>
    + for<'a> Add<&'a (F, M), Output = Self>
    + for<'a> SubAssign<&'a (F, M)>
    + for<'a> Sub<&'a (F, M), Output = Self>
    + for<'a> MulAssign<&'a (F, M)>
    + for<'a> Mul<&'a (F, M), Output = Self>
    + for<'a> MulAssign<&'a M>
    + for<'a> Mul<&'a M, Output = Self>
{
    fn new(num_of_vars: usize, terms: &[(F, M)]) -> Self;

    fn terms(&self) -> &[(F, M)];

    fn degree(&self) -> u16;

    fn num_of_vars(&self) -> usize;

    fn leading_term(&self) -> Option<(F, M)>;

    fn trailing_terms(&self) -> &[(F, M)];

    fn leading_monomial(&self) -> Option<M> {
        self.leading_term().map(|(_, term)| term)
    }

    fn leading_coefficient(&self) -> Option<F> {
        self.leading_term().map(|(coeff, _)| coeff)
    }

    fn s_polynomial(&self, other: &Self) -> Self;

    fn div_mod_polys(&self, gs: &[Self]) -> (Vec<Self>, Self);
}

/// Stores a sparse multivariate polynomial in coefficient form.
#[derive(Derivative)]
#[derivative(Clone, PartialEq, Eq, Hash, Default)]
pub struct FastPolynomial<F: Field, M: Monomial> {
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(crate) num_of_vars: usize,
    pub(crate) terms: Vec<(F, M)>,
}

impl<F: Field, M: Monomial> Add for FastPolynomial<F, M> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl<F: Field, M: Monomial> Polynomial<F, M> for FastPolynomial<F, M> {
    /// Input terms con not contains (F, M), which F is zero.
    fn new(num: usize, terms: &[(F, M)]) -> Self {
        let mut terms = terms.to_vec();
        terms.sort_unstable_by(|(_, m1), (_, m2)| m1.cmp(m2));
        let mut terms_dedup: Vec<(F, M)> = Vec::new();
        for term in terms {
            if let Some(prev) = terms_dedup.last_mut() {
                if prev.1 == term.1 {
                    prev.0 += term.0;
                    if prev.0.is_zero() {
                        terms_dedup.pop();
                    }
                    continue;
                }
            };
            terms_dedup.push(term);
        }

        Self {
            num_of_vars: num,
            terms: terms_dedup,
        }
    }

    fn terms(&self) -> &[(F, M)] {
        self.terms.as_slice()
    }

    fn degree(&self) -> u16 {
        self.terms.last().map(|(_, t)| t.degree()).unwrap_or(0)
    }

    fn num_of_vars(&self) -> usize {
        self.num_of_vars
    }

    fn leading_term(&self) -> Option<(F, M)> {
        self.terms.last().cloned()
    }

    fn trailing_terms(&self) -> &[(F, M)] {
        if self.terms.len() > 1 {
            &self.terms[..self.terms.len() - 1]
        } else {
            &[]
        }
    }

    fn s_polynomial(&self, other: &Self) -> Self {
        let (coeff_self, lm_self) = self.leading_term().unwrap();
        let (coeff_other, lm_other) = other.leading_term().unwrap();

        // Compute the least common multiple of the leading monomials
        let lcm = lm_self.lcm(&lm_other);

        let t_self = (lcm.to_owned() / &lm_self).unwrap();
        let t_other = (lcm / &lm_other).unwrap();

        // Multiply and subtract the polynomials
        let mut result = self * &(coeff_other, t_self);
        result -= &(other * &(coeff_self, t_other));
        result
    }

    fn div_mod_polys(&self, gs: &[Self]) -> (Vec<Self>, Self) {
        let mut qs: Vec<Self> = iter::repeat(Self::zero()).take(gs.len()).collect();
        let mut remainder = self.clone();

        while !remainder.is_zero() {
            let mut division_occurred = false;

            for (i, divisor) in gs.iter().enumerate() {
                let (r_coefficient, r_monomial) = remainder.leading_term().unwrap();
                let (g_coefficient, g_monomial) = divisor.leading_term().unwrap();

                if let (t_coefficient, Some(t_monomial)) =
                    (r_coefficient / g_coefficient, r_monomial / &g_monomial)
                {
                    remainder -= &(divisor * &(t_coefficient, t_monomial.to_owned()));
                    qs[i] += &(t_coefficient, t_monomial);
                    division_occurred = true;
                    break;
                }
            }

            if !division_occurred {
                break;
            }
        }

        (qs, remainder)
    }
}

impl<F: Field, M: Monomial> Neg for FastPolynomial<F, M> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self::Output {
        for coeff in &mut self.terms {
            (coeff).0 = -coeff.0;
        }
        self
    }
}

impl<F: Field, M: Monomial> Zero for FastPolynomial<F, M> {
    /// Returns the zero polynomial.
    fn zero() -> Self {
        Self {
            num_of_vars: 0,
            terms: Vec::new(),
        }
    }

    /// Checks if the given polynomial is zero.
    fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }
}

impl<F: Field, M: Monomial> Debug for FastPolynomial<F, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        for (coeff, term) in self.terms.iter().filter(|(c, _)| !c.is_zero()) {
            if term.is_constant() {
                write!(f, "\n{:?}", coeff)?;
            } else {
                write!(f, "\n{:?} {:?}", coeff, term)?;
            }
        }
        Ok(())
    }
}

impl<'a, F: Field, M: Monomial> AddAssign<&'a Self> for FastPolynomial<F, M> {
    fn add_assign(&mut self, other: &'a Self) {
        self.num_of_vars = self.num_of_vars.max(other.num_of_vars);
        let (mut i, mut ii): (usize, usize) = (0, self.terms.len());
        let (mut j, jj): (usize, usize) = (0, other.terms.len());

        let mut append = Vec::new();
        loop {
            let which = match (
                i.lt(&ii).then(|| &self.terms[i]),
                j.lt(&jj).then(|| &other.terms[j]),
            ) {
                (Some((_, cur)), Some((_, other))) => Some(cur.cmp(other)),
                (Some(_), None) => Some(Ordering::Less),
                (None, Some(_)) => Some(Ordering::Greater),
                (None, None) => None,
            };

            match which {
                Some(Ordering::Less) => i += 1,
                Some(Ordering::Equal) => {
                    let (cur, _) = &mut self.terms[i];
                    let (other, _) = &other.terms[j];
                    *cur += *other;
                    if cur.is_zero() {
                        self.terms.remove(i);
                        ii -= 1;
                    } else {
                        i += 1;
                    }
                    j += 1;
                },
                Some(Ordering::Greater) => {
                    append.push((i, other.terms[j].clone()));
                    j += 1;
                },
                None => break,
            };
        }
        append.into_iter().rev().for_each(|(index, value)| {
            self.terms.insert(index, value);
        });
    }
}

impl<'a, F: Field, M: Monomial> Add<&'a Self> for FastPolynomial<F, M> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        let mut result = self;
        result.add_assign(rhs);
        result
    }
}

impl<'a, 'b, F: Field, M: Monomial> Add<&'a FastPolynomial<F, M>> for &'b FastPolynomial<F, M> {
    type Output = FastPolynomial<F, M>;

    fn add(self, rhs: &'a FastPolynomial<F, M>) -> Self::Output {
        let mut result = self.clone();
        result.add_assign(rhs);
        result
    }
}

impl<'a, F: Field, M: Monomial> SubAssign<&'a Self> for FastPolynomial<F, M> {
    fn sub_assign(&mut self, other: &'a Self) {
        self.num_of_vars = self.num_of_vars.max(other.num_of_vars);
        let (mut i, mut ii): (usize, usize) = (0, self.terms.len());
        let (mut j, jj): (usize, usize) = (0, other.terms.len());

        let mut append = Vec::new();
        loop {
            let which = match (
                i.lt(&ii).then(|| &self.terms[i]),
                j.lt(&jj).then(|| &other.terms[j]),
            ) {
                (Some((_, cur)), Some((_, other))) => Some(cur.cmp(other)),
                (Some(_), None) => Some(Ordering::Less),
                (None, Some(_)) => Some(Ordering::Greater),
                (None, None) => None,
            };

            match which {
                Some(Ordering::Less) => i += 1,
                Some(Ordering::Equal) => {
                    let (cur, _) = &mut self.terms[i];
                    let (other, _) = &other.terms[j];
                    *cur -= *other;
                    if cur.is_zero() {
                        self.terms.remove(i);
                        ii -= 1;
                    } else {
                        i += 1;
                    }
                    j += 1;
                },
                Some(Ordering::Greater) => {
                    append.push((i, (other.terms[j].0.neg(), other.terms[j].1.clone())));
                    j += 1;
                },
                None => break,
            };
        }
        append.into_iter().rev().for_each(|(index, value)| {
            self.terms.insert(index, value);
        });
    }
}

impl<'a, F: Field, M: Monomial> Sub<&'a Self> for FastPolynomial<F, M> {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        let mut result = self;
        result.sub_assign(rhs);
        result
    }
}

impl<'a, 'b, F: Field, M: Monomial> Sub<&'a FastPolynomial<F, M>> for &'b FastPolynomial<F, M> {
    type Output = FastPolynomial<F, M>;

    fn sub(self, rhs: &'a FastPolynomial<F, M>) -> Self::Output {
        let mut result = self.clone();
        result.sub_assign(rhs);
        result
    }
}

impl<'a, F: Field, M: Monomial> AddAssign<&'a (F, M)> for FastPolynomial<F, M> {
    fn add_assign(&mut self, rhs: &'a (F, M)) {
        if !rhs.0.is_zero() {
            match self.terms.binary_search_by(|(_, m)| m.cmp(&rhs.1)) {
                Ok(i) => {
                    self.terms[i].0 += rhs.0;
                    if self.terms[i].0.is_zero() {
                        self.terms.remove(i);
                    }
                },
                Err(i) => self.terms.insert(i, rhs.clone()),
            }
        }
    }
}

impl<'a, F: Field, M: Monomial> Add<&'a (F, M)> for FastPolynomial<F, M> {
    type Output = Self;

    fn add(self, rhs: &'a (F, M)) -> Self::Output {
        let mut result = self;
        result.add_assign(rhs);
        result
    }
}

impl<'a, 'b, F: Field, M: Monomial> Add<&'a (F, M)> for &'b FastPolynomial<F, M> {
    type Output = FastPolynomial<F, M>;

    fn add(self, rhs: &'a (F, M)) -> Self::Output {
        let mut result = self.clone();
        result.add_assign(rhs);
        result
    }
}

impl<'a, F: Field, M: Monomial> SubAssign<&'a (F, M)> for FastPolynomial<F, M> {
    fn sub_assign(&mut self, rhs: &'a (F, M)) {
        if !rhs.0.is_zero() {
            match self.terms.binary_search_by(|(_, m)| m.cmp(&rhs.1)) {
                Ok(i) => {
                    self.terms[i].0 -= rhs.0;
                    if self.terms[i].0.is_zero() {
                        self.terms.remove(i);
                    }
                },
                Err(i) => self.terms.insert(i, (rhs.0.neg(), rhs.1.clone())),
            }
        }
    }
}

impl<'a, F: Field, M: Monomial> Sub<&'a (F, M)> for FastPolynomial<F, M> {
    type Output = Self;

    fn sub(self, rhs: &'a (F, M)) -> Self::Output {
        let mut result = self;
        result.sub_assign(rhs);
        result
    }
}

impl<'a, 'b, F: Field, M: Monomial> Sub<&'a (F, M)> for &'b FastPolynomial<F, M> {
    type Output = FastPolynomial<F, M>;

    fn sub(self, rhs: &'a (F, M)) -> Self::Output {
        let mut result = self.clone();
        result.sub_assign(rhs);
        result
    }
}

impl<'a, F: Field, M: Monomial> MulAssign<&'a (F, M)> for FastPolynomial<F, M> {
    fn mul_assign(&mut self, rhs: &'a (F, M)) {
        if rhs.0.is_zero() {
            self.terms = Vec::new();
        } else {
            self.terms.iter_mut().for_each(|(c, m)| {
                *c *= rhs.0;
                *m *= &rhs.1;
            });
        }
    }
}

impl<'a, F: Field, M: Monomial> Mul<&'a (F, M)> for FastPolynomial<F, M> {
    type Output = Self;

    fn mul(self, rhs: &'a (F, M)) -> Self::Output {
        let mut result = self;
        result.mul_assign(rhs);
        result
    }
}

impl<'a, 'b, F: Field, M: Monomial> Mul<&'a (F, M)> for &'b FastPolynomial<F, M> {
    type Output = FastPolynomial<F, M>;

    fn mul(self, rhs: &'a (F, M)) -> Self::Output {
        let mut result = self.clone();
        result.mul_assign(rhs);
        result
    }
}

impl<'a, F: Field, M: Monomial> MulAssign<&'a M> for FastPolynomial<F, M> {
    fn mul_assign(&mut self, rhs: &'a M) {
        self.terms.iter_mut().for_each(|(_, m)| {
            *m *= rhs;
        });
    }
}

impl<'a, F: Field, M: Monomial> Mul<&'a M> for FastPolynomial<F, M> {
    type Output = Self;

    fn mul(self, rhs: &'a M) -> Self::Output {
        let mut result = self;
        result.mul_assign(rhs);
        result
    }
}

impl<'a, 'b, F: Field, M: Monomial> Mul<&'a M> for &'b FastPolynomial<F, M> {
    type Output = FastPolynomial<F, M>;

    fn mul(self, rhs: &'a M) -> Self::Output {
        let mut result = self.clone();
        result.mul_assign(rhs);
        result
    }
}

#[cfg(test)]
#[allow(clippy::all)]
mod tests {
    use ark_ff::UniformRand;
    use ark_std::test_rng;
    use ark_test_curves::bls12_381::Fr;
    use rand::Rng;

    use crate::poly::monomial::{DegRevLexOrder, FastMonomial, LexOrder, MonomialOrd};

    use super::*;

    /// Generate random `l`-variate polynomial of maximum individual degree `d`
    fn rand_poly<R: Rng, O: MonomialOrd>(
        l: usize,
        d: u16,
        rng: &mut R,
    ) -> FastPolynomial<Fr, FastMonomial<3, O>> {
        let mut random_terms = Vec::new();
        let num_terms = rng.gen_range(1..1000);
        // For each term, randomly select up to `l` variables with degree
        // in [1,d] and random coefficient
        random_terms.push((Fr::rand(rng), FastMonomial::new(&vec![])));
        for _ in 1..num_terms {
            let term: Vec<(usize, u16)> = (0..l)
                .map(|i| {
                    if rng.gen_bool(0.5) {
                        Some((i, rng.gen_range(1..(d + 1))))
                    } else {
                        None
                    }
                })
                .flatten()
                .collect();
            let coeff = Fr::rand(rng);
            random_terms.push((coeff, FastMonomial::new(&term)));
        }
        FastPolynomial::new(l, &random_terms)
    }

    fn rand_monomial<R: Rng, O: MonomialOrd>(l: usize, d: u16, rng: &mut R) -> FastMonomial<3, O> {
        let term: Vec<(usize, u16)> = (0..l)
            .map(|i| {
                if rng.gen_bool(0.5) {
                    Some((i, rng.gen_range(1..(d + 1))))
                } else {
                    None
                }
            })
            .flatten()
            .collect();
        FastMonomial::new(&term)
    }

    #[test]
    fn add_polynomials() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for b_var_count in 1..20 {
                let p1 = rand_poly::<_, LexOrder>(a_var_count, max_degree, rng);
                let p2 = rand_poly::<_, LexOrder>(b_var_count, max_degree, rng);
                let res1 = p1.to_owned() + &p2;
                let res2 = p2.to_owned() + &p1;
                assert_eq!(res1, res2);
                assert!((res2 - &res1).is_zero());
            }
        }
    }

    #[test]
    fn sub_polynomials() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for b_var_count in 1..20 {
                let p1 = rand_poly::<_, LexOrder>(a_var_count, max_degree, rng);
                let p2 = rand_poly::<_, LexOrder>(b_var_count, max_degree, rng);
                let res1 = p1.to_owned() - &p2;
                let res2 = p2.to_owned() - &p1;
                assert_eq!(res1, -res2);
                assert_eq!(res1.to_owned() + &p2, p1);
            }
        }
    }

    /// Generate random `l`-variate polynomial of maximum individual degree `d`
    fn rand_poly2<R: Rng>(
        l: usize,
        d: u16,
        rng: &mut R,
    ) -> FastPolynomial<Fr, FastMonomial<3, LexOrder>> {
        let mut random_terms = Vec::new();
        let num_terms = rng.gen_range(1..1000);
        // For each term, randomly select up to `l` variables with degree
        // in [1,d] and random coefficient
        random_terms.push((Fr::rand(rng), FastMonomial::new(&vec![])));
        for _ in 1..num_terms {
            let term: Vec<(usize, u16)> = (0..l)
                .map(|i| {
                    if rng.gen_bool(0.5) {
                        Some((i, rng.gen_range(1..(d + 1))))
                    } else {
                        None
                    }
                })
                .flatten()
                .collect();
            let coeff = Fr::rand(rng);
            random_terms.push((coeff, FastMonomial::new(&term)));
        }
        FastPolynomial::new(l, &random_terms)
    }

    fn rand_monomial2<R: Rng>(l: usize, d: u16, rng: &mut R) -> FastMonomial<3, LexOrder> {
        let term: Vec<(usize, u16)> = (0..l)
            .map(|i| {
                if rng.gen_bool(0.5) {
                    Some((i, rng.gen_range(1..(d + 1))))
                } else {
                    None
                }
            })
            .flatten()
            .collect();
        FastMonomial::new(&term)
    }

    #[test]
    fn add_polynomials2() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for b_var_count in 1..20 {
                let p1 = rand_poly2(a_var_count, max_degree, rng);
                let p2 = rand_poly2(b_var_count, max_degree, rng);
                let res1 = p1.to_owned() + &p2;
                let res2 = p2.to_owned() + &p1;
                assert_eq!(res1, res2);
                assert!((res2 - &res1).is_zero());
            }
        }
    }

    #[test]
    fn sub_polynomials2() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for b_var_count in 1..20 {
                let p1 = rand_poly2(a_var_count, max_degree, rng);
                let p2 = rand_poly2(b_var_count, max_degree, rng);
                let res1 = p1.to_owned() - &p2;
                let res2 = p2.to_owned() - &p1;
                assert_eq!(res1, -res2);
                assert_eq!(res1.to_owned() + &p2, p1);
            }
        }
    }

    #[test]
    fn add_tuple() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for _ in 1..20 {
                let p = rand_poly::<_, DegRevLexOrder>(a_var_count, max_degree, rng);
                let t = (Fr::rand(rng), rand_monomial(a_var_count, max_degree, rng));
                let res1 = &p + &t;
                let res2 = FastPolynomial::new(p.num_of_vars, &[t.to_owned()]) + &p;
                assert_eq!(res1, res2);
                assert_eq!(res1 - &t, p);
            }
        }
    }

    #[test]
    fn sub_tuple() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for _ in 1..20 {
                let p = rand_poly::<_, DegRevLexOrder>(a_var_count, max_degree, rng);
                let t = (Fr::rand(rng), rand_monomial(a_var_count, max_degree, rng));
                let res1 = &p - &t;
                let res2 = FastPolynomial::new(p.num_of_vars, &[t.to_owned()]) - &p;
                assert_eq!(&res1 + &t, p);
                assert_eq!(res1, -res2);
            }
        }
    }

    #[test]
    fn mul_tuple() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for _ in 1..20 {
                let p = rand_poly::<_, DegRevLexOrder>(a_var_count, max_degree, rng);
                let t = (Fr::rand(rng), rand_monomial(a_var_count, max_degree, rng));
                let res1 = &p * &t;
                let res2 = p
                    .terms
                    .iter()
                    .map(|tt| FastPolynomial::new(p.num_of_vars, &[tt.to_owned()]) * &t)
                    .reduce(|a, b| a + &b)
                    .unwrap();
                assert_eq!(res1, res2);
            }
        }
    }

    #[test]
    fn mul_monomial() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for _ in 1..20 {
                let p = rand_poly::<_, DegRevLexOrder>(a_var_count, max_degree, rng);
                let t = rand_monomial(a_var_count, max_degree, rng);
                let res1 = &p * &t;
                let res2 = p
                    .terms
                    .iter()
                    .map(|tt| FastPolynomial::new(p.num_of_vars, &[tt.to_owned()]) * &t)
                    .reduce(|a, b| a + &b)
                    .unwrap();
                assert_eq!(res1, res2);
            }
        }
    }

    #[test]
    fn add_tuple2() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for _ in 1..20 {
                let p = rand_poly2(a_var_count, max_degree, rng);
                let t = (Fr::rand(rng), rand_monomial2(a_var_count, max_degree, rng));
                let res1 = &p + &t;
                let res2 = FastPolynomial::new(p.num_of_vars, &[t.to_owned()]) + &p;
                assert_eq!(res1, res2);
                assert_eq!(res1 - &t, p);
            }
        }
    }

    #[test]
    fn sub_tuple2() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for _ in 1..20 {
                let p = rand_poly2(a_var_count, max_degree, rng);
                let t = (Fr::rand(rng), rand_monomial2(a_var_count, max_degree, rng));
                let res1 = &p - &t;
                let res2 = FastPolynomial::new(p.num_of_vars, &[t.to_owned()]) - &p;
                assert_eq!(&res1 + &t, p);
                assert_eq!(res1, -res2);
            }
        }
    }

    #[test]
    fn mul_tuple2() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for _ in 1..20 {
                let p = rand_poly2(a_var_count, max_degree, rng);
                let t = (Fr::rand(rng), rand_monomial2(a_var_count, max_degree, rng));
                let res1 = &p * &t;
                let res2 = p
                    .terms
                    .iter()
                    .map(|tt| FastPolynomial::new(p.num_of_vars, &[tt.to_owned()]) * &t)
                    .reduce(|a, b| a + &b)
                    .unwrap();
                assert_eq!(res1, res2);
            }
        }
    }

    #[test]
    fn mul_monomial2() {
        let rng = &mut test_rng();
        let max_degree = 10;
        for a_var_count in 1..20 {
            for _ in 1..20 {
                let p = rand_poly2(a_var_count, max_degree, rng);
                let t = rand_monomial2(a_var_count, max_degree, rng);
                let res1 = &p * &t;
                let res2 = p
                    .terms
                    .iter()
                    .map(|tt| FastPolynomial::new(p.num_of_vars, &[tt.to_owned()]) * &t)
                    .reduce(|a, b| a + &b)
                    .unwrap();
                assert_eq!(res1, res2);
            }
        }
    }
}
