use once_cell::sync::Lazy;
use std::{
    cmp::Ordering,
    fmt::{Debug, Error, Formatter},
    hash::Hash,
    iter,
    marker::PhantomData,
    ops::{Div, Mul, MulAssign},
    simd::{Simd, SimdOrd, SimdPartialOrd, SimdUint, ToBitMask},
    vec::Vec,
};

pub trait Monomial:
    Clone
    + PartialOrd
    + Ord
    + PartialEq
    + Eq
    + Hash
    + Default
    + Debug
    + Send
    + Sync
    + for<'a> MulAssign<&'a Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Option<Self>>
    + for<'a> MulAssign<usize>
    + for<'a> Mul<usize, Output = Self>
    + for<'a> Div<usize, Output = Option<Self>>
{
    fn new(monomial: &[(usize, u16)]) -> Self;

    fn degree(&self) -> u16;

    fn is_constant(&self) -> bool;

    fn lcm(&self, other: &Self) -> Self;

    fn gcd(&self, other: &Self) -> Self;

    fn one() -> Self;

    fn iter(&self) -> Box<dyn DoubleEndedIterator<Item = &'_ u16> + '_ + Sync + Send>;

    fn compare_lex_order(lhs: &Self, rhs: &Self) -> Ordering;

    fn compare_deg_rev_lex_order(lhs: &Self, rhs: &Self) -> Ordering;
}

pub trait MonomialOrd: Clone + PartialEq + Eq + Hash + Default + Send + Sync {
    fn compare<T: Monomial>(lhs: &T, rhs: &T) -> Ordering;
}

#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct LexOrder {}
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct DegRevLexOrder {}

impl MonomialOrd for LexOrder {
    fn compare<T: Monomial>(lhs: &T, rhs: &T) -> Ordering {
        T::compare_lex_order(lhs, rhs)
    }
}

impl MonomialOrd for DegRevLexOrder {
    fn compare<T: Monomial>(lhs: &T, rhs: &T) -> Ordering {
        T::compare_deg_rev_lex_order(lhs, rhs)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct FastMonomial<const N: usize, O: MonomialOrd>([Simd<u16, 8>; N], PhantomData<O>);

static SIMD_ZERO: Lazy<Simd<u16, 8>> = Lazy::new(|| Simd::splat(0));

impl<const N: usize, O: MonomialOrd> FastMonomial<N, O> {
    pub fn transform_order<OT: MonomialOrd>(self) -> FastMonomial<N, OT> {
        FastMonomial(self.0, PhantomData)
    }
}

impl<const N: usize, O: MonomialOrd> Default for FastMonomial<N, O> {
    fn default() -> Self {
        FastMonomial([*SIMD_ZERO; N], PhantomData)
    }
}

impl<const N: usize, O: MonomialOrd> PartialOrd for FastMonomial<N, O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(O::compare(self, other))
    }
}

impl<const N: usize, O: MonomialOrd> Ord for FastMonomial<N, O> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<const N: usize, O: MonomialOrd> Monomial for FastMonomial<N, O> {
    fn new(term: &[(usize, u16)]) -> Self {
        let mut term: Vec<(usize, u16)> =
            term.iter().filter(|(_, pow)| *pow != 0).copied().collect();
        if term.is_empty() {
            return Self::default();
        }
        term.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        let mut term_dedup: Vec<(usize, u16)> = vec![(0, 0)];
        for (var, pow) in term {
            if let Some(prev) = term_dedup.last_mut() {
                if prev.0 == var {
                    prev.1 += pow;
                    continue;
                } else {
                    for index in (prev.0 + 1)..var {
                        term_dedup.push((index, 0));
                    }
                }
            }
            term_dedup.push((var, pow));
        }
        let mut term: Vec<u16> = term_dedup.into_iter().map(|(_, pow)| pow).collect();
        term.extend(iter::repeat(0).take((term.len() + 7) / 8 * 8 - term.len()));
        let mut term_inner = [*SIMD_ZERO; N];
        term.chunks(8)
            .map(Simd::from_slice)
            .take(N)
            .enumerate()
            .for_each(|(i, v)| {
                term_inner[i] = v;
            });
        Self(term_inner, PhantomData)
    }

    fn degree(&self) -> u16 {
        self.0
            .iter()
            .copied()
            .reduce(|a, b| a + b)
            .map(|s| s.reduce_sum())
            .unwrap_or(0)
    }

    fn is_constant(&self) -> bool {
        self.0.iter().all(|x| SIMD_ZERO.eq(x))
    }

    fn one() -> Self {
        Self::default()
    }

    fn iter(&self) -> Box<dyn DoubleEndedIterator<Item = &'_ u16> + '_ + Sync + Send> {
        Box::new(self.0.iter().flat_map(|s| s.as_array().iter()))
    }

    fn compare_lex_order(lhs: &Self, rhs: &Self) -> Ordering {
        lhs.0
            .iter()
            .zip(rhs.0.iter())
            .find_map(|(&lhs_exp, &rhs_exp)| {
                let lr = lhs_exp.simd_gt(rhs_exp).to_bitmask().trailing_zeros();
                let rl = rhs_exp.simd_gt(lhs_exp).to_bitmask().trailing_zeros();
                let ord = rl.cmp(&lr);
                ord.is_ne().then_some(ord)
            })
            .unwrap_or(Ordering::Equal)
    }

    fn compare_deg_rev_lex_order(lhs: &Self, rhs: &Self) -> Ordering {
        lhs.degree().cmp(&rhs.degree()).then_with(|| {
            lhs.0
                .iter()
                .rev()
                .zip(rhs.0.iter().rev())
                .find_map(|(&lhs_exp, &rhs_exp)| {
                    let lr = lhs_exp.simd_gt(rhs_exp).to_bitmask().leading_zeros();
                    let rl = rhs_exp.simd_gt(lhs_exp).to_bitmask().leading_zeros();
                    let ord = rl.cmp(&lr);
                    ord.is_ne().then_some(ord)
                })
                .unwrap_or(Ordering::Equal)
                .reverse()
        })
    }

    fn lcm(&self, other: &Self) -> Self {
        let mut terms = [*SIMD_ZERO; N];
        self.0
            .iter()
            .zip(other.0.iter())
            .enumerate()
            .for_each(|(i, (a, b))| terms[i] = a.simd_max(*b));

        Self(terms, PhantomData)
    }

    fn gcd(&self, other: &Self) -> Self {
        let mut terms = [*SIMD_ZERO; N];
        self.0
            .iter()
            .zip(other.0.iter())
            .enumerate()
            .for_each(|(i, (a, b))| terms[i] = a.simd_min(*b));

        Self(terms, PhantomData)
    }
}

impl<const N: usize, O: MonomialOrd> Debug for FastMonomial<N, O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        for variable in self.iter().enumerate() {
            match variable.1.cmp(&1) {
                Ordering::Less => {},
                Ordering::Equal => write!(f, " * x_{}", variable.0)?,
                Ordering::Greater => write!(f, " * x_{}^{}", variable.0, variable.1)?,
            }
        }
        Ok(())
    }
}

impl<'a, const N: usize, O: MonomialOrd> MulAssign<&'a FastMonomial<N, O>> for FastMonomial<N, O> {
    fn mul_assign(&mut self, other: &'a FastMonomial<N, O>) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(lhs, rhs)| *lhs += rhs);
    }
}

impl<'a, const N: usize, O: MonomialOrd> Mul<&'a FastMonomial<N, O>> for FastMonomial<N, O> {
    type Output = Self;

    fn mul(self, other: &'a FastMonomial<N, O>) -> Self::Output {
        let mut result = self;
        result.mul_assign(other);
        result
    }
}

impl<'a, 'b, const N: usize, O: MonomialOrd> Mul<&'a FastMonomial<N, O>>
    for &'b FastMonomial<N, O>
{
    type Output = FastMonomial<N, O>;

    fn mul(self, other: &'a FastMonomial<N, O>) -> Self::Output {
        let mut result = self.clone();
        result.mul_assign(other);
        result
    }
}

impl<'a, const N: usize, O: MonomialOrd> Div<&'a FastMonomial<N, O>> for FastMonomial<N, O> {
    type Output = Option<Self>;

    fn div(self, other: &'a FastMonomial<N, O>) -> Self::Output {
        (&self).div(other)
    }
}

impl<'a, 'b, const N: usize, O: MonomialOrd> Div<&'a FastMonomial<N, O>>
    for &'b FastMonomial<N, O>
{
    type Output = Option<FastMonomial<N, O>>;

    fn div(self, other: &'a FastMonomial<N, O>) -> Self::Output {
        if self
            .0
            .iter()
            .zip(other.0.iter())
            .all(|(lhs, &rhs)| lhs.simd_ge(rhs).all())
        {
            let mut terms: [Simd<u16, 8>; N] = [*SIMD_ZERO; N];
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(lhs, rhs)| lhs - rhs)
                .enumerate()
                .for_each(|(i, v)| {
                    terms[i] = v;
                });

            Some(FastMonomial(terms, PhantomData))
        } else {
            None
        }
    }
}

impl<const N: usize, O: MonomialOrd> MulAssign<usize> for FastMonomial<N, O> {
    fn mul_assign(&mut self, other: usize) {
        let page: usize = other / 8;
        let position: usize = other % 8;
        let mut acc: [u16; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
        acc[position] = 1;
        self.0[page] += Simd::from_array(acc);
    }
}

impl<const N: usize, O: MonomialOrd> Mul<usize> for FastMonomial<N, O> {
    type Output = Self;

    fn mul(self, other: usize) -> Self::Output {
        let mut result = self;
        result.mul_assign(other);
        result
    }
}

impl<'b, const N: usize, O: MonomialOrd> Mul<usize> for &'b FastMonomial<N, O> {
    type Output = FastMonomial<N, O>;

    fn mul(self, other: usize) -> Self::Output {
        let mut result = self.clone();
        result.mul_assign(other);
        result
    }
}

impl<const N: usize, O: MonomialOrd> Div<usize> for FastMonomial<N, O> {
    type Output = Option<Self>;

    fn div(self, other: usize) -> Self::Output {
        (&self).div(other)
    }
}

impl<'b, const N: usize, O: MonomialOrd> Div<usize> for &'b FastMonomial<N, O> {
    type Output = Option<FastMonomial<N, O>>;

    fn div(self, other: usize) -> Self::Output {
        let page: usize = other / 8;
        let position: usize = other % 8;
        let mut acc: [u16; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
        acc[position] = 1;
        let acc = Simd::from_array(acc);
        if self.0[page].simd_ge(acc).all() {
            let mut div_term = self.0;
            div_term[page] -= acc;
            Some(FastMonomial(div_term, PhantomData))
        } else {
            None
        }
    }
}

#[cfg(test)]
#[allow(clippy::all)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_div_var() {
        type M = FastMonomial<3, LexOrder>;
        let term1 = M::new(&vec![(0, 1), (8, 1), (16, 1)]);
        let expected = M::new(&vec![(8, 1), (16, 1)]);
        assert_eq!(&term1 / 0, Some(expected));
        let term1 = M::new(&vec![(0, 1), (8, 1), (16, 1)]);
        let expected = M::new(&vec![(0, 1), (16, 1)]);
        assert_eq!(&term1 / 8, Some(expected));
        let term1 = M::new(&vec![(0, 1), (8, 1), (16, 1)]);
        let expected = M::new(&vec![(0, 1), (8, 1)]);
        assert_eq!(&term1 / 16, Some(expected));
        let term1 = M::new(&vec![(0, 3), (8, 3), (16, 3)]);
        let expected = M::new(&vec![(0, 2), (8, 3), (16, 3)]);
        assert_eq!(&term1 / 0, Some(expected));
        let term1 = M::new(&vec![(0, 3), (8, 3), (16, 3)]);
        let expected = M::new(&vec![(0, 3), (8, 2), (16, 3)]);
        assert_eq!(&term1 / 8, Some(expected));
        let term1 = M::new(&vec![(0, 3), (8, 3), (16, 3)]);
        let expected = M::new(&vec![(0, 3), (8, 3), (16, 2)]);
        assert_eq!(&term1 / 16, Some(expected));
    }

    #[test]
    fn test_simd_mul_var() {
        type M = FastMonomial<3, LexOrder>;
        let term1 = M::new(&vec![(8, 1), (16, 1)]);
        let expected = M::new(&vec![(0, 1), (8, 1), (16, 1)]);
        assert_eq!(&term1 * 0, expected);
        let term1 = M::new(&vec![(0, 1), (16, 1)]);
        let expected = M::new(&vec![(0, 1), (8, 1), (16, 1)]);
        assert_eq!(&term1 * 8, expected);
        let term1 = M::new(&vec![(0, 1), (8, 1)]);
        let expected = M::new(&vec![(0, 1), (8, 1), (16, 1)]);
        assert_eq!(&term1 * 16, expected);
        let term1 = M::new(&vec![(0, 2), (8, 3), (16, 3)]);
        let expected = M::new(&vec![(0, 3), (8, 3), (16, 3)]);
        assert_eq!(&term1 * 0, expected);
        let term1 = M::new(&vec![(0, 3), (8, 2), (16, 3)]);
        let expected = M::new(&vec![(0, 3), (8, 3), (16, 3)]);
        assert_eq!(&term1 * 8, expected);
        let term1 = M::new(&vec![(0, 3), (8, 3), (16, 2)]);
        let expected = M::new(&vec![(0, 3), (8, 3), (16, 3)]);
        assert_eq!(&term1 * 16, expected);
    }

    #[test]
    fn test_simd_div() {
        type M = FastMonomial<3, LexOrder>;
        let term1 = M::new(&vec![(0, 1), (8, 1), (16, 1)]);
        let term2 = M::new(&vec![(16, 1)]);
        let expected = M::new(&vec![(0, 1), (8, 1)]);
        assert_eq!(term1 / &term2, Some(expected));
    }

    #[test]
    fn test_lcm_empty_terms2() {
        type M = FastMonomial<3, LexOrder>;
        let term1 = M::new(&vec![]);
        let term2 = M::new(&vec![]);
        let expected = M::new(&vec![]);
        assert_eq!(term1.lcm(&term2), expected);
    }

    #[test]
    fn test_lcm_one_empty_term2() {
        type M = FastMonomial<2, LexOrder>;
        let term1 = M::new(&vec![(1, 2), (2, 3)]);
        let term2 = M::new(&vec![]);
        let expected = M::new(&vec![(1, 2), (2, 3)]);
        assert_eq!(term1.lcm(&term2), expected);
    }

    #[test]
    fn test_lcm_no_common_vars2() {
        type M = FastMonomial<1, LexOrder>;
        let term1 = M::new(&vec![(1, 2), (2, 3)]);
        let term2 = M::new(&vec![(3, 1), (4, 2)]);
        let expected = M::new(&vec![(1, 2), (2, 3), (3, 1), (4, 2)]);
        assert_eq!(term1.lcm(&term2), expected);
    }

    #[test]
    fn test_lcm_common_vars2() {
        type M = FastMonomial<1, LexOrder>;
        let term1 = M::new(&vec![(1, 2), (2, 3)]);
        let term2 = M::new(&vec![(1, 1), (2, 5), (3, 2)]);
        let expected = M::new(&vec![(1, 2), (2, 5), (3, 2)]);
        assert_eq!(term1.lcm(&term2), expected);
    }

    #[test]
    fn test_lcm_all_common_vars2() {
        type M = FastMonomial<1, LexOrder>;
        let term1 = M::new(&vec![(1, 2), (2, 3)]);
        let term2 = M::new(&vec![(1, 4), (2, 5)]);
        let expected = M::new(&vec![(1, 4), (2, 5)]);
        assert_eq!(term1.lcm(&term2), expected);
    }

    #[test]
    fn test_mul2() {
        type M = FastMonomial<1, LexOrder>;
        let term1 = M::new(&vec![(1, 2), (2, 3)]);
        let term2 = M::new(&vec![(2, 1), (3, 4)]);
        let result = M::new(&vec![(1, 2), (2, 4), (3, 4)]);
        assert_eq!(term1 * &term2, result);
    }

    #[test]
    fn test_div2() {
        type M = FastMonomial<1, LexOrder>;
        let term1 = M::new(&vec![(1, 5), (2, 4), (3, 6)]);
        let term2 = M::new(&vec![(1, 2), (2, 3)]);
        let expected = M::new(&vec![(1, 3), (2, 1), (3, 6)]);
        let Some(result) = term1 / &term2 else { unreachable!() };
        assert_eq!(result, expected);
    }

    #[test]
    fn test_div_incompatible_terms2() {
        type M = FastMonomial<1, LexOrder>;
        let term1 = M::new(&vec![(1, 2), (2, 3)]);
        let term2 = M::new(&vec![(2, 1), (3, 4)]);
        assert_eq!(term1 / &term2, None);
    }
}
