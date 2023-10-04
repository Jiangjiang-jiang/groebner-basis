use crate::poly::{
    monomial::Monomial,
    polynomial::{FastPolynomial, Polynomial},
};
use ark_ff::{Field, Zero};
use chrono::Local;
use rayon::prelude::*;

/// Check whether give ideal is groebner basis
pub fn is_groebner_basis<F: Field, M: Monomial, P: Polynomial<F, M>>(ideal: &[P]) -> bool {
    ideal.par_iter().enumerate().all(|(i, f)| {
        ideal.par_iter().enumerate().all(|(j, g)| {
            if i == j {
                true
            } else {
                let (_, remainder) = f.s_polynomial(g).div_mod_polys(ideal);
                remainder.is_zero()
            }
        })
    })
}

/// Reduce groebner basis to minimal
pub fn reduce_groebner_basis<F: Field, M: Monomial>(ideal: &mut Vec<FastPolynomial<F, M>>) {
    println!(
        "{} Interreduce start with ideal.len(): {}",
        Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        ideal.len()
    );
    ideal.par_sort_unstable_by(|f, g| {
        f.leading_monomial()
            .unwrap()
            .cmp(&g.leading_monomial().unwrap())
    });
    ideal.dedup_by(|f, g| {
        f.leading_monomial()
            .unwrap()
            .eq(&g.leading_monomial().unwrap())
    });

    loop {
        println!(
            "{} Interreduce start with ideal.len(): {}",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            ideal.len()
        );
        let mut changed = false;

        for i in 0..ideal.len() {
            let mut reminder = ideal[i].clone();

            while !reminder.is_zero() {
                if let Some(delta) = reminder.terms().par_iter().find_map_last(|fm| {
                    ideal.iter().enumerate().find_map(|(j, divisor)| {
                        if i != j && !divisor.is_zero() {
                            let (r_coefficient, r_monomial) = fm.clone();
                            let (g_coefficient, g_monomial) = divisor.leading_term().unwrap();

                            (r_monomial / &g_monomial).map(|t_monomial| {
                                divisor * &(r_coefficient / g_coefficient, t_monomial)
                            })
                        } else {
                            None
                        }
                    })
                }) {
                    reminder -= &delta;
                    changed = true;
                } else {
                    break;
                }
            }

            ideal[i] = reminder;
        }

        // If polynomial is not zero and is a constant, it's 1.
        // Abandon this case, generator 1 mean no solution to this system.
        if ideal
            .par_iter()
            .any(|f| !f.is_zero() && f.leading_monomial().unwrap().is_constant())
        {
            *ideal = vec![FastPolynomial::new(
                ideal[0].num_of_vars(),
                &[(F::one(), M::one())],
            )];
            return;
        }

        ideal.retain(|poly| !poly.is_zero());

        println!(
            "{} Interreduce end with ideal.len(): {}",
            Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            ideal.len()
        );

        if !changed {
            break;
        }
    }

    ideal.par_iter_mut().for_each(|f| {
        let coefficient_inv = f.leading_coefficient().unwrap().inverse().unwrap();
        *f *= &(coefficient_inv, M::one());
    });
}
