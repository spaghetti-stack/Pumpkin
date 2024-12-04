use crate::{
    conjunction,
    engine::{
        propagation::{Propagator, ReadDomains},
        reason::Reason,
    },
    predicate,
    variables::IntegerVariable,
};

#[derive(Clone, Debug, Copy)]
pub struct Values {
    pub value: i32,
    pub omin: i32,
    pub omax: i32,
}

#[derive(Clone, Debug)]
pub(crate) struct SimpleGCCLowerUpper<Variable> {
    variables: Box<[Variable]>,
    values: Box<[Values]>,
}

impl<Variable: IntegerVariable> SimpleGCCLowerUpper<Variable> {
    pub(crate) fn new(variables: Box<[Variable]>, values: Box<[Values]>) -> Self {
        Self { variables, values }
    }
}

impl<Variable: IntegerVariable> Propagator for SimpleGCCLowerUpper<Variable> {
    fn name(&self) -> &str {
        "Global Cardinality Low Up"
    }

    fn debug_propagate_from_scratch(
        &self,
        mut context: crate::engine::propagation::PropagationContextMut,
    ) -> crate::basic_types::PropagationStatusCP {
        println!("called. {:?}, {:?} \n", context, self.values);
        for value in self.values.clone() {
            let (assigned, others): (Vec<_>, Vec<_>) = self.variables.iter().partition(|var| {
                context.is_fixed(*var) && context.lower_bound(*var) == value.value
            });

            if assigned.len() as i32 >= value.omax {
                others.iter().try_for_each(|v| {
                    let vv = *v;
                    context.remove(vv, value.value, conjunction!([vv != value.value]))
                })?;
            }
        }

        Ok(())
    }

    fn initialise_at_root(
        &mut self,
        _: &mut crate::engine::propagation::PropagatorInitialisationContext,
    ) -> Result<(), crate::predicates::PropositionalConjunction> {
        Ok(())
    }
}
