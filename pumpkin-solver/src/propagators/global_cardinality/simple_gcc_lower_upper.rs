use log::debug;

use crate::{
    basic_types::Inconsistency,
    engine::{
        propagation::{LocalId, Propagator, ReadDomains},
        DomainEvents,
    },
    predicates::{Predicate, PropositionalConjunction},
    propagators::global_cardinality::*,
    variables::IntegerVariable,
};

use super::Values;

// local ids of array vars are shifted by ID_X_OFFSET
const ID_X_OFFSET: u32 = 2;

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

/// Bruteforce implementation of the Global Cardinality Constraint (Lower-Upper)
impl<Variable: IntegerVariable + 'static> Propagator for SimpleGCCLowerUpper<Variable> {
    fn name(&self) -> &str {
        "Simple Global Cardinality Low Up"
    }

    fn debug_propagate_from_scratch(
        &self,
        context: crate::engine::propagation::PropagationContextMut,
    ) -> crate::basic_types::PropagationStatusCP {
        self.variables.iter().for_each(|v| {
            debug!(
                "called. u: {:?}, l: {:?}",
                context.upper_bound(v),
                context.lower_bound(v)
            );
        });

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

            debug!("all values fixed");
            return Err(Inconsistency::Conflict(conjunction_all_vars(
                &context,
                &self.variables,
            )));
        }

        Ok(())
    }

    fn initialise_at_root(
        &mut self,
        context: &mut crate::engine::propagation::PropagatorInitialisationContext,
    ) -> Result<(), PropositionalConjunction> {
        // Register all variables to domain change events.
        self.variables.iter().enumerate().for_each(|(i, x_i)| {
            let _ = context.register(
                x_i.clone(),
                DomainEvents::ANY_INT,
                LocalId::from(i as u32 + ID_X_OFFSET),
            );
        });

        // Register for backtrack events if needed with:
        //context.register_for_backtrack_events(var, domain_events, local_id);
        Ok(())
    }

    fn propagate(
        &mut self,
        context: crate::engine::propagation::PropagationContextMut,
    ) -> crate::basic_types::PropagationStatusCP {
        self.debug_propagate_from_scratch(context)
    }

    fn notify(
        &mut self,
        _context: crate::engine::propagation::PropagationContext,
        _local_id: LocalId,
        _event: crate::engine::opaque_domain_event::OpaqueDomainEvent,
    ) -> crate::engine::propagation::EnqueueDecision {
        debug!("notify");
        crate::engine::propagation::EnqueueDecision::Enqueue
    }

    fn notify_backtrack(
        &mut self,
        _context: crate::engine::propagation::PropagationContext,
        _local_id: LocalId,
        _event: crate::engine::opaque_domain_event::OpaqueDomainEvent,
    ) {
        debug!("notify backtrack");
    }

    fn synchronise(&mut self, _context: crate::engine::propagation::PropagationContext) {}

    fn priority(&self) -> u32 {
        // setting an arbitrary priority by default
        3
    }

    fn detect_inconsistency(
        &self,
        _context: crate::engine::propagation::PropagationContext,
    ) -> Option<PropositionalConjunction> {
        None
    }

    fn lazy_explanation(
        &mut self,
        _code: u64,
        _context: crate::engine::propagation::ExplanationContext,
    ) -> &[Predicate] {
        std::panic!(
            "{}",
            format!(
                "Propagator {} does not support lazy explanations.",
                self.name()
            )
        );
    }

    fn log_statistics(&self, _statistic_logger: crate::statistics::StatisticLogger) {}
}
