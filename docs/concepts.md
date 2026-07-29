# Core concepts

## Channels and events

A {py:class}`laddu.Channel` is a directed reaction graph. Edges represent initial, intermediate, or final particles; vertices represent production and decay steps. An event is a row of named four-vectors and scalar values. laddu uses

$$p^\mu=(E,p_x,p_y,p_z), \qquad p^2=E^2-\lvert\mathbf p\rvert^2.$$

Channel helpers create symbolic invariants and angles from those named columns, preventing model code from depending on column positions.

## Expressions and models

An {py:class}`laddu.Expr` is an immutable symbolic graph. Constants, event variables, complex arithmetic, parameters, angular functions, and line shapes compose naturally. A {py:class}`laddu.Model` wraps an intensity expression $I(\Omega;\boldsymbol\theta)\geq0$ and owns the resulting parameter layout.

Tag meaningful subexpressions before forming the final intensity. Tags let projections retain selected coherent components without rebuilding a second model.

## Data, generated MC, and accepted MC

Keep these samples conceptually distinct:

- **Data** enter the logarithmic term of the likelihood.
- **Generated MC** describe phase space before selection and are useful for efficiencies and projections.
- **Accepted MC** have passed the same reconstruction and selection as data and normalize the fitted intensity.

For a normalized unbinned fit, laddu evaluates an accepted-MC approximation to

$$\mathcal N(\boldsymbol\theta)=\int \epsilon(\Omega) I(\Omega;\boldsymbol\theta)\,d\Phi(\Omega).$$

## Compilation and execution

Model construction is cheap and symbolic. Preparation compiles the graph for a dataset schema and an {py:class}`laddu.Execution` configuration. Reuse a prepared {py:class}`laddu.Likelihood` inside the optimizer: do not reconstruct it on every objective call.

Forward automatic differentiation is usually effective for modest parameter counts; reverse mode can be preferable when a scalar objective has many free parameters. Measure on the real analysis because model shape, sample size, and hardware all matter.
