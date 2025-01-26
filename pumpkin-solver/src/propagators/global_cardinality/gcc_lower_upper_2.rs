use log::debug;

use crate::{
    basic_types::Inconsistency, create_statistics_struct, engine::{
        propagation::{LocalId, Propagator, ReadDomains}, DomainEvents
    }, predicate, predicates::PropositionalConjunction, propagators::global_cardinality::{max_count, min_count}, variables::IntegerVariable
};

use super::Values;
// local ids of array vars are shifted by ID_X_OFFSET
const ID_X_OFFSET: u32 = 2;

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

impl<Variable: IntegerVariable + 'static> Propagator for GCCLowerUpper2<Variable> {
    fn name(&self) -> &str {
        "Global Cardinality Low Up 2"
    }

    fn debug_propagate_from_scratch(
        &self,
        mut context: crate::engine::propagation::PropagationContextMut,
    ) -> crate::basic_types::PropagationStatusCP {
        self.variables.iter().for_each(|v| {
            debug!(
                "called. u: {:?}, l: {:?}",
                context.upper_bound(v),
                context.lower_bound(v)
            );
        });

        self.values.iter().try_for_each(|value| {
            let min = min_count(&self.variables, value.value, &context);
            let max = max_count(&self.variables, value.value, &context);
            debug!("v: {:?}, min_count: {:?}, max_count: {:?}", value, min, max);

            // If this is false, there is definitely no solution
            if min > value.omax {
                // Constraint violation
                debug!("Inconsistency detected: min: {:?}, max: {:?}, value: {:?}", min, max, value);
                return Err(Inconsistency::Conflict(self.variables.iter().filter(|v| context.is_fixed(*v) && context.upper_bound(*v) == value.value)
                    .map(|v| predicate!(v == value.value))
                    .collect()));
            }

            if max < value.omin {
                // Constraint violation
                debug!("Inconsistency detected: min: {:?}, max: {:?}, value: {:?}", min, max, value);
                return Err(Inconsistency::Conflict(self.variables.iter().filter(|v| !context.contains(*v, value.value))
                    .map(|v| predicate!(v != value.value))
                    .collect()));
            }

            self.variables.iter().try_for_each(|var| {
                debug!(
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
                        debug!("  Removing val = {}", value.value);
                        let reason: PropositionalConjunction = self.variables.iter().filter(|v| context.is_fixed(*v) && context.upper_bound(*v) == value.value)
                        .map(|v| predicate!(v == value.value))
                        .collect();
                        context.remove(
                            var,
                            value.value,
                            reason,
                        )?;
                    }
                    //If not assigning variable $x$ to this value $v$ would make the max_count lower than the lower bound,
                    //then problem becomes inconsistent. Therefore  $D(x)=v$.
                    else if max_count(&self.variables, value.value, &context) - 1 < value.omin {
                        debug!("  Setting val = {}", value.value);
                        let reason: PropositionalConjunction = self.variables.iter().filter(|v| !context.contains(*v, value.value))
                        .map(|v| predicate!(v != value.value))
                        .collect();
                        context.set_lower_bound(
                            var,
                            value.value,
                            reason.clone(),
                        )?;

                        context.set_upper_bound(
                            var,
                            value.value,
                            reason,
                        )?;
                    }
                }
                Ok(())
            })
        })
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

    fn log_statistics(&self, statistic_logger: crate::statistics::StatisticLogger) {
        create_statistics_struct!(Statistics { test: u32});

        let statistics = Statistics { test: 0 };

        statistic_logger.log_statistic(format!("{:?}", statistics));
    }

    fn priority(&self) -> u32 {
        3
    }

}