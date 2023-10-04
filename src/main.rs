extern crate core;

use ark_ff::{Fp, FpConfig, PrimeField, Zero};
use chrono::Local;
use clap::{arg, Parser};
use fancy_regex::Regex;
use groebner_basis::{
    factor::{evaluate, factor, get_flint_num_threads, set_flint_num_threads},
    fglm::FGLM,
    gvw::GVW,
    poly::{
        monomial::{DegRevLexOrder, FastMonomial, Monomial, MonomialOrd},
        polynomial::{FastPolynomial, Polynomial},
    },
    DegRevLexPolynomial, LexPolynomial, GF,
};
use rayon::{current_num_threads, prelude::*};
use std::{
    cmp::Ordering,
    collections::HashMap,
    error::Error,
    fs::File,
    io,
    io::{BufRead, BufReader, Seek, Write},
    ops::Neg,
    path::Path,
    time::Instant,
};

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
    #[arg(short, long)]
    input: String,

    #[arg(short, long)]
    output: String,
}

fn split_polynomial(text: &str) -> Vec<(&str, i128)> {
    let re = Regex::new(r"(?=[+-])").unwrap();
    let mut result = Vec::new();
    let mut prev_match_end = 0;

    for mat in re.captures_iter(text).flatten() {
        let start = mat.get(0).unwrap().start();
        let end = mat.get(0).unwrap().end();

        // Add the substring from the end of the previous match to the start of the current match
        let part = text[prev_match_end..start].trim();
        if !part.is_empty() {
            result.push(part);
        }

        prev_match_end = end;
    }

    // Add the remaining part of the text after the last match
    let remaining_part = text[prev_match_end..].trim();
    if !remaining_part.is_empty() {
        result.push(remaining_part);
    }

    result
        .iter()
        .map(|t| {
            if t.starts_with('-') {
                (t.strip_prefix('-').unwrap().trim(), -1)
            } else {
                (t.strip_prefix('+').unwrap_or(t).trim(), 1)
            }
        })
        .collect()
}

fn parse_vars(reader: &mut BufReader<File>) -> io::Result<HashMap<String, usize>> {
    let mut vars: HashMap<String, usize> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.starts_with("Defining") {
            vars = line
                .strip_prefix("Defining")
                .unwrap()
                .split(',')
                .map(|s| s.trim().to_owned())
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();
            break;
        }
    }

    Ok(vars)
}

fn parse_polynomials<const N: usize>(
    reader: &mut BufReader<File>,
    vars: HashMap<String, usize>,
) -> io::Result<(HashMap<usize, String>, Vec<DegRevLexPolynomial<N>>)> {
    let mut polynomials: Vec<DegRevLexPolynomial<N>> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let mut line = line.trim();

        if line.starts_with('[') && line.ends_with(']') {
            line = &line[1..line.len() - 1];

            polynomials = line
                .split(',')
                .map(|s| s.trim())
                .map(|poly| {
                    DegRevLexPolynomial::new(
                        vars.len(),
                        &split_polynomial(poly)
                            .into_iter()
                            .map(|(monomial, sign)| {
                                let term_parts: Vec<&str> =
                                    monomial.split('*').map(|s| s.trim()).collect();
                                let (coeff_part, term_part) =
                                    match term_parts[0].chars().next().unwrap().is_ascii_digit() {
                                        true => (term_parts[0], &term_parts[1..]),
                                        false => ("1", &term_parts[..]),
                                    };
                                (
                                    GF::from(coeff_part.parse::<i128>().unwrap() * sign), // Coefficient
                                    FastMonomial::new(
                                        &term_part
                                            .iter()
                                            .map(|&term| {
                                                let mut term_split = term.split('^');
                                                match (term_split.next(), term_split.next()) {
                                                    (Some(var), Some(power)) => (
                                                        *vars.get(var).unwrap(),
                                                        power.parse::<u16>().unwrap(),
                                                    ),
                                                    (Some(var), None) => {
                                                        (*vars.get(var).unwrap(), 1)
                                                    }
                                                    _ => panic!("Invalid input"),
                                                }
                                            })
                                            .collect::<Vec<(usize, u16)>>(),
                                    ),
                                )
                            })
                            .collect::<Vec<(GF, FastMonomial<N, DegRevLexOrder>)>>(),
                    )
                })
                .collect();
            continue;
        }

        if !line.is_empty() {
            panic!();
        }
    }

    Ok((vars.into_iter().map(|(v, i)| (i, v)).collect(), polynomials))
}

fn ideal_tostring<const NM: usize, const NF: usize, P: FpConfig<NF>, O: MonomialOrd>(
    var_maps: &HashMap<usize, String>,
    input_polynomials: &[FastPolynomial<Fp<P, NF>, FastMonomial<NM, O>>],
) -> String {
    input_polynomials
        .iter()
        .map(|poly| {
            poly.terms()
                .iter()
                .rev()
                .filter(|(c, _)| !c.is_zero())
                .map(|(coeff, term)| {
                    if term.is_constant() {
                        format!("{}", coeff.into_bigint())
                    } else {
                        format!(
                            "{}{}",
                            coeff.into_bigint(),
                            term.iter()
                                .enumerate()
                                .map(|(v, &e)| {
                                    match e.cmp(&1) {
                                        Ordering::Less => String::new(),
                                        Ordering::Equal => {
                                            format!("*{}", var_maps.get(&v).unwrap())
                                        },
                                        Ordering::Greater => {
                                            format!("*{}^{}", var_maps.get(&v).unwrap(), e)
                                        },
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
        .unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    set_flint_num_threads(current_num_threads() as i32);

    let path = Path::new(&args.input);
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let var_maps = parse_vars(&mut reader)?;

    match var_maps.len() {
        0 => unreachable!("No variable!"),
        1..=8 => main_inner::<1>(&mut reader, &args.output, var_maps)?,
        9..=16 => main_inner::<2>(&mut reader, &args.output, var_maps)?,
        17..=24 => main_inner::<3>(&mut reader, &args.output, var_maps)?,
        25..=32 => main_inner::<4>(&mut reader, &args.output, var_maps)?,
        33..=40 => main_inner::<5>(&mut reader, &args.output, var_maps)?,
        41..=48 => main_inner::<6>(&mut reader, &args.output, var_maps)?,
        49..=56 => main_inner::<7>(&mut reader, &args.output, var_maps)?,
        57..=64 => main_inner::<8>(&mut reader, &args.output, var_maps)?,
        _ => unreachable!("Number of variables exceeds limitation!"),
    }

    Ok(())
}

fn main_inner<const N: usize>(
    reader: &mut BufReader<File>,
    output: &str,
    var_maps: HashMap<String, usize>,
) -> Result<(), Box<dyn Error>> {
    let (var_maps, mut input_polynomials) = parse_polynomials::<N>(reader, var_maps)?;

    println!(
        "{} Computation start",
        Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
    );
    let start_time = Instant::now();

    input_polynomials = GVW::new(&input_polynomials);

    let mut output = File::create(Path::new(output))?;
    output.write_all(ideal_tostring(&var_maps, &input_polynomials).as_bytes())?;
    output.flush()?;

    let input_polynomials: Vec<LexPolynomial<N>> = FGLM::new(&input_polynomials);

    let elapsed_time = start_time.elapsed();
    let elapsed_secs = elapsed_time.as_secs() as f64 + elapsed_time.subsec_nanos() as f64 * 1e-9;
    println!(
        "{} Computation end, total time {:.3} seconds",
        Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        elapsed_secs
    );

    output.seek(io::SeekFrom::Start(0))?;
    output.set_len(0)?; // Truncate the file
    output.write_all(ideal_tostring(&var_maps, &input_polynomials).as_bytes())?;
    output.flush()?;

    println!(
        "{} Factorization start, thread number: {}",
        Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        get_flint_num_threads()
    );
    let factored_poly = factor(input_polynomials.last().unwrap());
    output.write_all(b"\nFactorization: \n")?;
    output.write_all(ideal_tostring(&var_maps, &factored_poly).as_bytes())?;
    output.flush()?;

    let last_var_roots: Vec<_> = factored_poly
        .iter()
        .filter(|p| {
            p.terms().len() == 2
                && p.terms()
                    .iter()
                    .all(|(_, m)| m.is_constant() || m.degree() == p.degree())
        })
        .map(|p| {
            p.terms()
                .iter()
                .find_map(|(f, m)| m.is_constant().then_some(f.neg()))
                .unwrap()
        })
        .collect();

    for last_var in last_var_roots {
        let vars: Vec<_> = input_polynomials
            .par_iter()
            .rev()
            .skip(1)
            .map(|p| {
                (
                    p.leading_monomial()
                        .unwrap()
                        .iter()
                        .position(|&d| d > 0)
                        .unwrap(),
                    evaluate(p.trailing_terms(), last_var).neg(),
                )
            })
            .collect();
        output.write_all(b"\nRoots: \n")?;
        for (var, root) in vars.into_iter().rev() {
            output.write_all(
                format!(
                    "{} = K({})\n",
                    var_maps.get(&var).unwrap(),
                    root.into_bigint()
                )
                .as_bytes(),
            )?;
        }
        output.write_all(
            format!(
                "{} = K({})\n",
                var_maps.get(var_maps.keys().max().unwrap()).unwrap(),
                last_var.into_bigint()
            )
            .as_bytes(),
        )?;
        output.flush()?;
    }

    Ok(())
}
