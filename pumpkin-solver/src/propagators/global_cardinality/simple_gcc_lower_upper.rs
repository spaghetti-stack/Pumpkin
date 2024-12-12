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
pub(crate) struct SimpleGCCLowerUpper<Variable> {
    variables: Box<[Variable]>,
    values: Box<[Values]>,
}

impl<Variable: IntegerVariable> SimpleGCCLowerUpper<Variable> {
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

impl<Variable: IntegerVariable + 'static> Propagator for SimpleGCCLowerUpper<Variable> {
    fn name(&self) -> &str {
        "Global Cardinality Low Up"
    }

    //     fn debug_propagate_from_scratch(
    //     &self,
    //     mut context: crate::engine::propagation::PropagationContextMut,
    // ) -> crate::basic_types::PropagationStatusCP {
    //     println!("called. {:?}, {:?} \n", context, self.values);
    //     for value in self.values.clone() {
    //         let (assigned, others): (Vec<_>, Vec<_>) = self.variables.iter().partition(|var| {
    //             context.is_fixed(*var) && context.lower_bound(*var) == value.value
    //         });

    //         if assigned.len() as i32 >= value.omax {
    //             others.iter().try_for_each(|v| {
    //                 let vv = *v;
    //                 context.remove(vv, value.value, conjunction!([vv != value.value]))
    //             })?;
    //         }
    //     }

    //     Ok(())
    // }

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

        // Wait until the search fixes all values, and then check if the assignment satisfies the constraint of the propagator.
        if self.variables.iter().all(|var| context.is_fixed(var))
            && !self.values.iter().all(|value| {
                vars_satisfy_value(
                    &self.variables,
                    value.value,
                    value.omin,
                    value.omax,
                    &context,
                )
            })
        {
            // If the assignment satisfied the GCC, just return OK(()). This means a valid solution is found, or not all variables have been assigned.
            // If not, need to use an explanation so that the solver knows this assignment is not valid?
            //TODO: Implement the explanation for an error.

            println!("this!!1");
            return Err(Inconsistency::Conflict(conjunction!(
                [x1 == context.lower_bound(x1)]
                    & [x2 == context.lower_bound(x2)]
                    & [x3 == context.lower_bound(x3)]
                    & [x4 == context.lower_bound(x4)]
            )));
            //context.remove(x1, 1, conjunction!())?;
            // let r = context.remove(
            //     x1,
            //     context.lower_bound(x1),
            //     //PropositionalConjunction::new(vec![Predicate::False]),
            //     //conjunction!([x1 =! 1] & [x2 == 1] & [x3 == 1] & [x4 == 1]),
            //     conjunction!(),
            // )?;
            //return Err(Inconsistency::EmptyDomain);

            //println!("{:?}", r);
            //r?
            // return Err(conjunction!(
            //     [x1 == context.lower_bound(x1)]
            //         & [x2 == context.lower_bound(x2)]
            //         & [x3 == context.lower_bound(x3)]
            //         & [x4 == context.lower_bound(x4)]
            // )
            // .into());
        }

        Ok(())
    }

    fn initialise_at_root(
        &mut self,
        _: &mut crate::engine::propagation::PropagatorInitialisationContext,
    ) -> Result<(), PropositionalConjunction> {
        Ok(())
    }
}
