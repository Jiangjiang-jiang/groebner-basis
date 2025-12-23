# Gröbner Basis Calculator

A high-performance Rust implementation of Gröbner basis computation with multiple algorithms, FGLM conversion, and polynomial factorization support.

## Features

- **Multiple Algorithm Implementations**: 
  - **GVW (Gao-Volny-Wang)**: Signature-based Gröbner basis computation (primary algorithm)
  - **FGLM**: Gröbner basis conversion between monomial orderings
  - **Polynomial Factorization**: Efficient factorization using FLINT library
  - **Deprecated Algorithms** (for research reference): Buchberger, F5, F4.5, mo-GVW

- **High-Performance Optimizations**:
  - SIMD-optimized monomial operations
  - Parallel computation using Rayon
  - Efficient sparse polynomial representation
  - Arc smart pointers to avoid unnecessary polynomial copies

- **Flexible Configuration**:
  - Support for up to 64 variables
  - Multiple monomial orderings (Lex, DegRevLex)
  - Finite field GF(p) where p = 18446744073709551557

## Project Structure

```
src/
├── lib.rs              # Core library definitions and type aliases
├── main.rs             # Command-line interface
├── poly/
│   ├── monomial.rs     # SIMD-optimized monomial implementation
│   ├── polynomial.rs   # Sparse polynomial operations
│   └── mod.rs          # Module definitions
├── gvw.rs              # GVW algorithm implementation (primary)
├── fglm.rs             # FGLM algorithm implementation
├── groebner.rs         # Common Gröbner basis utilities
├── factor.rs           # Polynomial factorization (FLINT bindings)
└── deprecated/         # Historical algorithm implementations
    ├── buchberger.rs   # Classic Buchberger algorithm
    ├── f5.rs           # F5 algorithm
    ├── f45.rs          # F4.5 algorithm
    └── mogvw.rs        # Monomial-oriented GVW
```

## Requirements

- **Rust**: Nightly version (required for SIMD features)
- **FLINT**: For polynomial factorization
- **System**: CPU with AVX2 support (for SIMD optimization)

## Installation

```bash
# Install FLINT library (Ubuntu/Debian)
sudo apt-get install libflint-dev

# Or on macOS
brew install flint

# Build the project
cargo +nightly build --release
```

## Usage

### Command-Line Tool

```bash
cargo +nightly run --release -- -i input.txt -o output.txt
```

### Input Format

The input file should contain:
1. Variable definition line: `Defining x_0, x_1, x_2, ...`
2. Polynomial ideal: `[poly1, poly2, ...]`

Example:
```
Defining x_0, x_1, x_2
[1*x_1^3 + 1*x_0^2, 1*x_0^2*x_1 + 1*x_0^2, 1*x_0^3 + -1*x_0^2, 1*x_2^4 + -1*x_0^2 + -1*x_1]
```

### Output Format

The output file contains:
1. **Gröbner basis in DegRevLex order** (intermediate result)
2. **Gröbner basis in Lex order** (final result)
3. **Factorization**: Factorization of the last polynomial
4. **Solutions (roots)**: If computable

Example output:
```
1*x_0 + ...
1*x_1^4 + ...
1*x_2 + 9

Factorization:
1*x_2 + 9

Roots:
x_0 = K(...)
x_1 = K(...)
x_2 = K(...)
```

## Algorithm Details

### GVW Algorithm (Primary)

Signature-based Gröbner basis computation:

**Core Idea**:
- Associate each polynomial with a signature `(u, i)`
- Use signatures to avoid redundant S-polynomial computations
- Three main criteria: divisibility, coverage, rewritability

**Advantages**:
- Avoids many useless S-polynomial computations
- Parallel processing of critical pairs
- Memory efficient (using Arc for sharing)

**Reference**: [GVW Paper](http://www.math.clemson.edu/~sgao/papers/gvw_R130704.pdf)

### FGLM Algorithm

Converts Gröbner basis from DegRevLex to Lex ordering:

**Steps**:
1. **Build multiplication matrices**: Compute standard basis
2. **Normal form computation**: Using normal form algorithm
3. **Linear dependency detection**: Construct basis in new ordering

**Key Points**:
- Applicable to zero-dimensional ideals
- Complexity depends on quotient ring dimension, not polynomial count

### Polynomial Factorization

Uses FLINT library's Kaltofen-Shoup algorithm:
- Efficient factorization over finite fields
- Multi-threaded support
- C bindings via `flint-sys`

## Algorithm Evolution History

The project explored multiple Gröbner basis algorithms (see `deprecated/`):

### 1. Buchberger Algorithm
- **Classic algorithm**, proposed in 1965
- Uses sugar strategy for critical pair selection
- Implements coprimality and syzygy criteria
- **Drawback**: Generates many redundant computations

### 2. F5 Algorithm
- Proposed by Faugère in 2002
- First signature-based algorithm
- Uses signatures to avoid S-polynomials reducing to zero
- **Issue**: Complex implementation, high overhead for rule checking

### 3. F4.5 Algorithm
- Hybrid of F4 and F5
- Uses matrix elimination
- Symbolic preprocessing + Gaussian elimination
- **Issue**: High matrix density, large memory consumption

### 4. mo-GVW Algorithm
- Monomial-oriented GVW
- Organizes computation by monomial degree
- **Issue**: Complex degree lifting strategy

### 5. GVW Algorithm (Final Choice)
- Cleaner signature implementation
- Efficient criteria checking
- Parallel-friendly data structures
- **Chosen for**: Best balance of performance and maintainability

## Code Examples

### Rust API

```rust
use groebner_basis::{GVW, FGLM, DegRevLexPolynomial, LexPolynomial};
use groebner_basis::poly::{monomial::*, polynomial::*};

// Define polynomials
let ideal: Vec<DegRevLexPolynomial<1>> = vec![
    FastPolynomial::new(3, &vec![
        (1.into(), FastMonomial::new(&vec![(0, 2)])),
        (1.into(), FastMonomial::new(&vec![(0, 1), (1, 1)])),
        ((-1).into(), FastMonomial::new(&vec![(2, 1)])),
    ]),
    // ...
];

// Compute Gröbner basis
let gb = GVW::new(&ideal);

// Convert to Lex ordering
let lex_gb: Vec<LexPolynomial<1>> = FGLM::new(&gb);

// Factorize
use groebner_basis::factor::factor;
let factors = factor(lex_gb.last().unwrap());
```

## Performance Considerations

### Variable Count and Type Parameters

The constant generic parameter `N` controls monomial storage (8 variables per SIMD vector):
```rust
N=1  =>  1-8   variables
N=2  =>  9-16  variables
N=3  =>  17-24 variables
N=4  =>  25-32 variables
...
N=8  =>  57-64 variables
```

### Parallelization

- **GVW**: Parallel processing of critical pairs
- **FGLM**: Parallelized matrix operations
- **Factorization**: Control via `set_flint_num_threads()`

### Memory Optimization

- Use `Arc<FastPolynomial>` to avoid large polynomial copies
- SIMD vectors for compact monomial storage
- Sparse representation avoids storing zero coefficients

## Testing

```bash
# Run all tests
cargo +nightly test

# Run specific module tests
cargo +nightly test --lib gvw
cargo +nightly test --lib fglm

# Run individual test
cargo +nightly test test_GVW_given_case_1
```

Test coverage:
- ✓ Monomial operations (multiplication, division, LCM, GCD)
- ✓ Polynomial operations (add, subtract, multiply, division with remainder)
- ✓ Correctness of different monomial orderings
- ✓ GVW algorithm verification
- ✓ FGLM conversion correctness
- ✓ Polynomial factorization

## Technical Highlights

### 1. SIMD Optimization
```rust
// Using Rust portable SIMD
use std::simd::{Simd, SimdOrd, SimdUint};

pub struct FastMonomial<const N: usize, O: MonomialOrd>(
    [Simd<u16, 8>; N],  // Pack 8 u16s into one SIMD vector
    PhantomData<O>
);
```

### 2. Zero-Copy Polynomials
```rust
pub struct MSignature<F: Field, M: Monomial> {
    signature: Signature<M>,
    polynomial: Arc<FastPolynomial<F, M>>,  // Shared ownership
}
```

### 3. Parallel Critical Pair Processing
```rust
let jpair_set: BTreeSet<_> = BTreeSet::from_par_iter(
    G.par_iter()
        .enumerate()
        .flat_map(|(i, gi)| {
            G[..i].par_iter()
                .filter_map(|gj| Self::make_jpair(gi, gj))
                .filter(|(t, xaeif)| /* criteria */)
        })
);
```
