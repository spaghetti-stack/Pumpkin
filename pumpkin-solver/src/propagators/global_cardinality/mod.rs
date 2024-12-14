use crate::{
    engine::propagation::ReadDomains,
    predicate,
    predicates::{Predicate, PropositionalConjunction},
    variables::IntegerVariable,
};

pub(crate) mod ford_fulkerson_lower_bounds;
pub(crate) mod gcc_lower_upper;
pub(crate) mod gcc_lower_upper_2;
pub(crate) mod simple_gcc_lower_upper;

#[derive(Clone, Debug, Copy)]
pub struct Values {
    pub value: i32,
    pub omin: u32,
    pub omax: u32,
}

/// Check if, in all variables with a fixed assignment `value` occurs at least `min` and at most `max` times.
fn vars_satisfy_value<Variable: IntegerVariable>(
    vars: &[Variable],
    value: i32,
    min: u32,
    max: u32,
    context: &crate::engine::propagation::PropagationContextMut,
) -> bool {
    let occurences: u32 = vars
        .iter()
        .filter(|v| context.is_fixed(*v) && context.upper_bound(*v) == value)
        .count() as u32;

    occurences >= min && occurences <= max
}

/// Return the most specific explanation possible: A conjunction describing the current domain of all given variables.
fn conjunction_all_vars<'a, I, Variable>(
    context: &crate::engine::propagation::PropagationContextMut,
    vars: I,
) -> PropositionalConjunction
where
    I: IntoIterator<Item = &'a Variable>,
    Variable: IntegerVariable + 'a,
{
    let res: Vec<Predicate> = vars
        .into_iter()
        .flat_map(|var| {
            [
                predicate!(var >= context.lower_bound(var)),
                predicate!(var <= context.upper_bound(var)),
            ]
        })
        .collect();
    res.into()
}
