# Compiler CAS

The optimizer in `crates/laddu-compile/src/cas/` searches equivalent typed expressions. The input and output are ordinary `ExprGraph`s; rule authors never move graph IDs or metadata. `cas.rs` owns the `egg` graph and budgets, `pattern.rs` owns matching and guards, `builder.rs` owns checked candidate construction, and `extract.rs` selects a shared runtime DAG.

## Add a local identity

Put the equation in `cas/rules.rs` inside `Theory::standard`:

```rust
cas_rules! {
    x + 0 => x;
    sin(x)^2 + cos(x)^2 => 1;
    x*a + x*b <=> x*(a+b);
    exp(x) * exp(y) <=> exp(x + y);
    x^2 - y^2 <=> (x - y) * (x + y);
    norm_sqr(exp(I * phase) * wave) => norm_sqr(wave) if real(phase);
}
```

`=>` adds one direction; `<=>` retains both. The left side binds symbols. The right side and guards may only use bound symbols, checked when the macro compiles. Use `I` for the imaginary unit. `+` and `*` are scalar associative and commutative operators: patterns match terms in any order, including a subset of a larger sum or product. Ordered tensor operations use `matmul`, `matvec`, `dot`, and `solve`. The matcher checks scalar/tensor shapes before emitting a candidate. Other guards are `nonzero`, `scalar`, `zero`, and `identity`; combine guards with `&&` or `||`.

Growth directions, including distribution from `x*(a+b)` to `x*a+x*b`, run once for each matching source expression. A product containing multiple additive factors is not expanded automatically, since the combinations quickly exhaust the search budget. Shrinking directions can run in every search round. The search still obeys the global node, memory, and time budgets.

Put a test beside the rule that checks an equivalent candidate is discoverable. When several equivalent graphs have similar warm cost, do not require a particular spelling from extraction. Test values and gradients through the compiled model as well.

| Transformation | Place to add it |
| --- | --- |
| Scalar, complex, trig, and exponential identities | `cas/rules.rs` |
| Coefficient collection, factoring, power multisets, and constant folding | `cas/algebra.rs` with `ProductView` |
| Coefficient-aware Euler and even half-angle alternatives | `cas/trig.rs` |
| Indexed vector and matrix formulas | `cas/tensor.rs` with `contraction` |
| Matching, guards, and type checks shared by every equation | `cas/pattern.rs` and `cas/builder.rs` |

## Add an analyzed transformation

Use `CandidateBuilder` in `algebra.rs`, `trig.rs`, or `tensor.rs` when a rule needs to inspect an arbitrary number of terms, compute a common factor, or emit several alternatives. For example, `ProductView::of(cas, term)` yields a coefficient and a multiset of factors with integer powers; collection and factorization share this view. Build with `sum`, `product`, `unary`, `component`, or `matrix_element`, then call `emit_equivalent(original, candidate)`. Every builder method checks types or dimensions. Keep graph traversal, provenance, and cost decisions in the shared engine.

Indexed tensor equations belong in `tensor.rs`. The shared `contraction` builder implements `sum_k left(k) * right(k)`; dot products, matrix-vector products, and matrix products supply typed index projections. The same formulas drive selected-element projection and full expansion. Expansion is bounded by shape so the e-graph stays within its memory budget.

Tensor–scalar `+`, `-`, and `*` work in either operand order; tensor / scalar is elementwise. The expression layer keeps a tensor–scalar syntax node until graph construction, then lowers its elements into the existing runtime graph format. `tensor.rs` recognizes a shared scalar factor across a vector or matrix constructor and offers an equivalent contraction with the factor lifted out. `CandidateBuilder::scale` constructs the resulting tensor candidate with checked dimensions.

## Search and selection

`OptimizationBudget` controls rounds, e-nodes, estimated memory, search time, and solver time. Rule families rotate their first position each round. A budget stop still extracts a valid graph. Extraction uses a deterministic greedy tree selection by default. Set `solver_seconds` to a positive value to opt into exact extraction. That time is shared between execution extraction and normalization extraction within one model compilation. `good_lp` with `microlp` then chooses one node per active equivalence class with DAG sharing and acyclicity constraints. Execution extraction solves five integer objectives in order: repeated event work, parameter-only work, dataset preparation, compile-time work, then node count. Greedy extraction is also the fallback when exact extraction cannot complete. `CompiledModel::optimization_diagnostics()` reports search and extraction outcomes. The normalization candidate and execution graph come from the same equivalence search, then use separate objectives. Internal normalization basis models skip normalization extraction because their normalization plan is disabled.

In Python, `Model(expr)` uses greedy extraction. Use `Model(expr, exact_solver_seconds=5.0)` to opt into the exact solver. Recompilation methods (`projection`, `with_parameters`, and `Model.from_json`) each accept the same keyword; their default is greedy regardless of how the original model was compiled. Inspect `model.optimization_diagnostics` for `execution_exact`, `normalization_exact`, and fallback reasons. The timeout controls extraction during compilation, so it belongs on `Model` and its recompilation methods rather than `Execution`.
