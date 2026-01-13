# Lava: A Gröbner Basis Solver in Rust

**Lava** is a high-performance open-source tool for computing Gröbner Bases of multivariate polynomial systems. It is designed specifically for the algebraic analysis of Arithmetic-oriented Primitives.

This project implements the **GVW Algorithm** for basis computation and the **FGLM Algorithm** for basis conversion, aiming to solve high-degree polynomial systems over large finite fields efficiently.

## Motivation

When solving algebraic systems over finite fields, the bottlenecks of general-purpose mathematical software directly motivated the development of Lava:

- **SageMath** threw errors and failed to execute calculations when handling the specific large finite fields required for large finite fields (e.g. GF(p) where p = 18446744073709551557).
- While **Magma** is closed-source and expensive, the web-based trial free version could not complete the solving process within the time limits for the scale of certain computations.

Thus, we chose to implement the core algorithms from scratch using Rust to achieve better memory control and execution efficiency.

## Features

Lava includes implementations for two core algorithms:

1. **GVW Algorithm (Gröbner Basis Computation)**
   - Unlike the Buchberger and F5 algorithms, the GVW algorithm possesses a rigorous proof of termination.
   - Filters Critical Pairs using the Syzygy Criterion and Rewriting Criterion, reducing redundant computations.
2. **FGLM Algorithm (Basis Conversion)**
   - Used to convert a computed GrevLex (Greedy Reverse Lexicographic) Gröbner Basis into a Lex (Lexicographic) basis for finding root for univariate polynomials.

## Installation & Usage

This project requires a nightly rust version and CPU with AVX2 for portable SIMD support. Finite field arithmetic relies on the [Arkworks](https://github.com/arkworks-rs) `ff` library.

### Installation

```bash
sudo apt-get install libflint-dev libmpfr-dev libgmp-dev m4

git clone https://github.com/Jiangjiang-jiang/groebner-basis.git
cd groebner-basis

cargo +nightly build --release
```

### Test

```bash
# Run all tests
cargo +nightly test

# Run specific module tests
cargo +nightly test --lib gvw
cargo +nightly test --lib fglm

# Run individual test
cargo +nightly test test_GVW_given_case_1
```

### Usage

Run `./groebner-basis --help`:

```
Usage: groebner-basis --input <INPUT> --output <OUTPUT>

Options:
  -i, --input <INPUT>    
  -o, --output <OUTPUT>  
  -h, --help             Print help
  -V, --version          Print version
```

The input file should contain:

1. Variable definition line: `Defining x_0, x_1, x_2, ...`
2. Polynomial array: `[poly1, poly2, ...]`

Example `input.txt`:

```text
Defining x_0, x_1, x_2
[1*x_1^3 + 1*x_0^2, 1*x_0^2*x_1 + 1*x_0^2, 1*x_0^3 + -1*x_0^2, 1*x_2^4 + -1*x_0^2 + -1*x_1]
```

The output file contains:

1. Gröbner basis in DegRevLex order (intermediate result)
2. Gröbner basis in Lex order (final result)
3. Factorization: Factorization of the last polynomial
4. Solutions (roots): If computable

Example output:

```text
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

Build & Run:

```bash
cargo +nightly run --release -- -i input.txt -o output.txt
```

## Implementation Details

To outperform general-purpose mathematical software, Lava incorporates extensive optimizations in underlying data structures and parallel computing:

### 1. Monomial & Polynomial Optimization

- **SIMD Acceleration**: Monomial operations are rewritten using 128-bit SIMD (SSE4 instruction set). Tests show an 82%-254% speedup in LCM (Least Common Multiple) calculations and a ~195% speedup in Degree calculations.
- **Sparse Representation**: Polynomials are stored using a sparse structure with dynamic arrays, sorted in ascending monomial order. This ensures that addition, subtraction, and multiplication operations have a linear time complexity of $O(n)$.

### 2. GVW Algorithm Optimization

- **Data Structures**: A `BTreeSet` is used to maintain the Pair set, ensuring insertion and retrieval of the minimal element have a complexity of $O(\log n)$.
- **Copy On Write**: Smart pointers are used extensively during new polynomial generation to avoid deep copies. Copying only occurs during Top-reduction if modifications are necessary, reducing peak memory usage.
- **Parallel Processing**: The generation of new elements is parallelized to leverage multi-core CPUs.

### 3. FGLM Algorithm & Linear Algebra

- **Custom Parallel LU Decomposition**: Due to the lack of high-performance matrix libraries supporting specific finite field in rust, a custom parallel Partial Pivot LU decomposition algorithm is implemented to replace Gaussian elimination.
- **Underdetermined/Overdetermined Systems**: The algorithm is adapted to handle non-square matrices.
- **Caching Mechanism**: Previous LU decomposition results are reused when the matrix is not updated, reducing the solving complexity from $O(d^3)$ to $O(d^2)$.

## References

Aly A, Ashur T, Ben-Sasson E, et al. Design of symmetric-key primitives for advanced cryptographic protocols[J]. IACR Transactions on Symmetric Cryptology, 2020: 1-45. 

Szepieniec A, Ashur T, Dhooghe S. Rescue-prime: a standard specification (sok)[J]. Cryptology ePrint Archive, 2020. 

J.C. Faugère, P. Gianni, D. Lazard, and T. Mora. 1993. Efficient Computation of Zerodimensional Gröbner Bases by Change of Ordering. J. Symb. Comput. 16, 4 (Oct. 1993), 329– 344. https://doi.org/10.1006/jsco.1993.1051 

Yao Sun, Zhenyu Huang, Dingkang Wang, and Dongdai Lin. 2016. An improvement over the GVW algorithm for inhomogeneous polynomial systems. Finite Fields Appl. 41, C (September 2016), 174–192. https://doi.org/10.1016/j.ffa.2016.06.002
