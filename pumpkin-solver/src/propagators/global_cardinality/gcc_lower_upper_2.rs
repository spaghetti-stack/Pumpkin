use crate::{
    basic_types::Inconsistency,
    conjunction,
    engine::{
        propagation::{Propagator, ReadDomains},
        reason::Reason,
        EmptyDomain,
    },
    predicate,
    predicates::{Predicate, PropositionalConjunction},
    variables::{IntegerVariable, Literal},
};

use super::Values;

#[derive(Clone, Debug)]
pub(crate) struct GCCLowerUpper2<Variable> {
    variables: Box<[Variable]>,
    values: Box<[Values]>,
}

impl<Variable: IntegerVariable> GCCLowerUpper2<Variable> {
    pub(crate) fn new(variables: Box<[Variable]>, values: Box<[Values]>) -> Self {
        Self { variables, values }
    }
}

/// Check if, in all variables with a fixed assignment `value` occurs at least `min` and at most `max` times.
fn vars_satisfy_value<Variable: IntegerVariable>(
    vars: &[Variable],
    value: i32,
    min: i32,
    max: i32,
    context: &crate::engine::propagation::PropagationContextMut,
) -> bool {
    let occurences: i32 = vars
        .iter()
        .filter(|v| context.is_fixed(*v) && context.upper_bound(*v) == value)
        .count() as i32;

    occurences >= min && occurences <= max
}

fn min_count<Variable: IntegerVariable>(
    vars: &[Variable],
    value: i32,
    context: &crate::engine::propagation::PropagationContextMut,
) -> i32 {
    let occurences: i32 = vars
        .iter()
        .filter(|v| context.is_fixed(*v) && context.upper_bound(*v) == value)
        .count() as i32;

    occurences
}

fn max_count<Variable: IntegerVariable>(
    vars: &[Variable],
    value: i32,
    context: &crate::engine::propagation::PropagationContextMut,
) -> i32 {
    let occurences: i32 = vars.iter().filter(|v| context.contains(*v, value)).count() as i32;

    occurences
}

impl<Variable: IntegerVariable + 'static> Propagator for GCCLowerUpper2<Variable> {
    fn name(&self) -> &str {
        "Global Cardinality Low Up 2"
    }

    fn debug_propagate_from_scratch(
        &self,
        mut context: crate::engine::propagation::PropagationContextMut,
    ) -> crate::basic_types::PropagationStatusCP {
        self.variables.iter().for_each(|v| {
            println!(
                "called. u: {:?}, l: {:?}",
                context.upper_bound(v),
                context.lower_bound(v)
            );
        });
        println!();

        let x1 = self.variables.get(0).unwrap();
        let x2 = self.variables.get(1).unwrap();
        let x3 = self.variables.get(2).unwrap();
        let x4 = self.variables.get(3).unwrap();

        self.values.iter().try_for_each(|value| {
            let min = min_count(&self.variables, value.value, &context);
            let max = max_count(&self.variables, value.value, &context);
            println!("v: {:?}, min_count: {:?}, max_count: {:?}", value, min, max);

            // If this is false, there is definitely no solution
            if min > value.omax || max < value.omin {
                // Constraint violation
                return Err(Inconsistency::Conflict(conjunction!()));
            }

            self.variables.iter().try_for_each(|var| {
                println!(
                    "var: u {:?}, l: {:?}",
                    context.upper_bound(var),
                    context.lower_bound(var)
                );
                if context.contains(var, value.value) {
                    // If assigning value $v$ to variable $x$ causes the min_count to be greater than the upper bound this would make problem inconsistent.
                    // Therefore we must remove $v$ from $D(x)$
                    if !context.is_fixed(var)
                        && min_count(&self.variables, value.value, &context) + 1 > value.omax
                    {
                        println!("  Removing val = {}", value.value);
                        context.remove(var, value.value, conjunction!())?;
                    }
                    //If not assigning variable $x$ to this value $v$ would make the max_count lower than the lower bound,
                    //then problem becomes inconsistent. Therefore  $D(x)=v$.
                    else if max_count(&self.variables, value.value, &context) - 1 < value.omin {
                        println!("  Setting val = {}", value.value);
                        context.set_lower_bound(var, value.value, conjunction!())?;
                        context.set_upper_bound(var, value.value, conjunction!())?;
                    }
                }
                Ok(())
            })
        })
    }

    fn initialise_at_root(
        &mut self,
        _: &mut crate::engine::propagation::PropagatorInitialisationContext,
    ) -> Result<(), PropositionalConjunction> {
        Ok(())
    }
}
