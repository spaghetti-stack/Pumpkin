use pumpkin_lib::basic_types::variables::AffineView;
use pumpkin_lib::basic_types::variables::IntVar;
use pumpkin_lib::basic_types::DomainId;
use pumpkin_lib::basic_types::Literal;
use pumpkin_lib::constraints::ConstraintsExt;
use pumpkin_lib::engine::ConstraintSatisfactionSolver;

pub(crate) fn int_lin_le_reif(
    solver: &mut ConstraintSatisfactionSolver,
    terms: Box<[AffineView<DomainId>]>,
    rhs: i32,
    reif: Literal,
) {
    solver.int_lin_le_reif(terms.clone(), rhs, reif);
    solver.int_lin_le_reif(
        terms.iter().map(|term| term.scaled(-1)).collect::<Vec<_>>(),
        -rhs - 1,
        !reif,
    );
}

pub(crate) fn int_lin_eq_reif(
    solver: &mut ConstraintSatisfactionSolver,
    terms: Box<[AffineView<DomainId>]>,
    rhs: i32,
    reif: Literal,
) {
    int_lin_le_reif(solver, terms.clone(), rhs, reif);

    let negated = terms.iter().map(|var| var.scaled(-1)).collect::<Box<[_]>>();
    int_lin_le_reif(solver, negated, -rhs, !reif);
}

pub(crate) fn int_le_reif(
    solver: &mut ConstraintSatisfactionSolver,
    a: DomainId,
    b: DomainId,
    reif: Literal,
) {
    int_lin_le_reif(solver, vec![a.scaled(1), b.scaled(-1)].into(), 0, reif)
}

pub(crate) fn int_lt_reif(
    solver: &mut ConstraintSatisfactionSolver,
    a: DomainId,
    b: DomainId,
    reif: Literal,
) {
    int_lin_le_reif(solver, vec![a.scaled(1), b.scaled(-1)].into(), -1, reif)
}

pub(crate) fn int_eq_reif(
    solver: &mut ConstraintSatisfactionSolver,
    a: DomainId,
    b: DomainId,
    reif: Literal,
) {
    int_lin_eq_reif(solver, vec![a.scaled(1), b.scaled(-1)].into(), 0, reif);
}

pub(crate) fn array_bool_or(
    solver: &mut ConstraintSatisfactionSolver,
    clause: impl Into<Vec<Literal>>,
    reif: Literal,
) {
    let mut clause = clause.into();

    // \/clause -> r
    clause.iter().for_each(|&literal| {
        let _ = solver.add_permanent_clause(vec![!literal, reif]);
    });

    // r -> \/clause
    clause.insert(0, !reif);
    let _ = solver.add_permanent_clause(clause);
}

pub(crate) fn int_ne_reif(
    solver: &mut ConstraintSatisfactionSolver,
    a: DomainId,
    b: DomainId,
    reif: Literal,
) {
    solver.int_ne_reif(a, b, reif);
    solver.int_eq_reif(a, b, !reif);
}

pub(crate) fn int_lin_ne_reif(
    solver: &mut ConstraintSatisfactionSolver,
    terms: Box<[AffineView<DomainId>]>,
    rhs: i32,
    reif: Literal,
) {
    solver.int_lin_ne_reif(terms.clone(), rhs, reif);
    solver.int_lin_eq_reif(terms, rhs, !reif)
}