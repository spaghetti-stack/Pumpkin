use std::convert::Into;

use petgraph::{dot::Dot, Graph};

use crate::{
    propagators::global_cardinality::{
        ford_fulkerson_lower_bounds::{ford_fulkerson, BoundedCapacity},
        gcc_lower_upper::GCCLowerUpper,
        gcc_lower_upper_2::GCCLowerUpper2,
        simple_gcc_lower_upper::SimpleGCCLowerUpper,
    },
    variables::IntegerVariable,
};

use super::Constraint;

pub use crate::propagators::global_cardinality::Values;

pub fn global_cardinality_lower_upper<Variable: IntegerVariable + 'static>(
    variables: impl IntoIterator<Item = Variable>,
    values: impl IntoIterator<Item = Values>,
) -> impl Constraint {
    GCCLowerUpper::new(
        variables.into_iter().collect(),
        values.into_iter().collect(),
    )
}
