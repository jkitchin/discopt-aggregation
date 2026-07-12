# Integer aggregation (opt-in)

By default only scalar **continuous** variables are eliminated, so the default
behaviour is MINLP-safe and byte-identical to the continuous-only pipeline. Pass
`integer_aggregation=True` to also eliminate a scalar **integer/binary** variable
— but only when its integrality is *implied*.

```python
solve(m, method="lm", integer_aggregation=True)   # collapse an integer chain
```

This is the standard MIP-presolve condition (Achterberg et al. 2020): an integer
variable `y` may be eliminated via `coeff*y + rest = 0` only when

- the defining equality is **fully affine** (no nonlinear term),
- the **pivot is ±1**,
- **every other participating variable is integer/binary**, and
- **every other coefficient and the constant term are integral**.

An all-integer stock balance qualifies:

```python
# s[t] = s[t-1] + q[t] - d[t]  — all integer, unit pivot, integral data
m.subject_to(s[t] - (s[t-1] + q[t] - d[t]) == 0)
```

## What abstains

The flag never eliminates a variable it cannot prove integral:

- a **continuous** variable in the definition (breaks implied integrality),
- a **non-unit pivot** — `y = 2*i` can define `y` but cannot eliminate `i`,
- a **fractional** coefficient or constant (including a variable-free nonlinear
  term such as `sin(2)`),
- a **nonlinear** body.

## Recovery and duals

Recovery of an eliminated integer returns an **exactly integral** value; a
non-integral result raises rather than silently rounding a wrong answer. Dual
recovery reports `available=False` over eliminated integers (KKT multipliers are
not defined for branching variables).

## When it helps

Use it for MINLPs with integer accounting/counting chains, where fewer integers
means a smaller branching space. It does nothing for models whose integers appear
only in **inequalities** (e.g. on/off binaries `x <= cap*b`) — those are not
defined by an equality, so they are never candidates.

!!! note "Default-off, proven no-op"
    `integer_aggregation` defaults to `False`. With the flag off, integer and
    binary variables are never aggregation candidates, and a test suite pins that
    the continuous-only results are unchanged.
