use crate::{
    propagators::global_cardinality::{
        gcc_lower_upper_2::GCCLowerUpper2, simple_gcc_lower_upper::SimpleGCCLowerUpper,
    },
    variables::IntegerVariable,
};

use super::Constraint;

pub use crate::propagators::global_cardinality::Values;

pub fn global_cardinality_lower_upper<Variable: IntegerVariable + 'static>(
    variables: impl IntoIterator<Item = Variable>,
    values: impl IntoIterator<Item = Values>,
) -> impl Constraint {
    GCCLowerUpper2::new(
        variables.into_iter().collect(),
        values.into_iter().collect(),
    )
}
