use log::{debug, warn};

use crate::{
    basic_types::Inconsistency,
    engine::{
        propagation::{LocalId, Propagator, ReadDomains},
        DomainEvents,
    },
    predicates::PropositionalConjunction,
    propagators::global_cardinality::{conjunction_all_vars, max_count, min_count},
    variables::IntegerVariable,
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
            if min > value.omax || max < value.omin {
                // Constraint violation
                warn!("Inconsistency detected: min: {:?}, max: {:?}, value: {:?}", min, max, value);
                return Err(Inconsistency::Conflict(conjunction_all_vars(
                    &context,
                    &self.variables,
                )));
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
                        warn!("  Removing val = {}", value.value);
                        context.remove(
                            var,
                            value.value,
                            conjunction_all_vars(&context, &self.variables),
                        )?;
                    }
                    //If not assigning variable $x$ to this value $v$ would make the max_count lower than the lower bound,
                    //then problem becomes inconsistent. Therefore  $D(x)=v$.
                    else if max_count(&self.variables, value.value, &context) - 1 < value.omin {
                        warn!("  Setting val = {}", value.value);
                        context.set_lower_bound(
                            var,
                            value.value,
                            conjunction_all_vars(&context, &self.variables)
                            //conjunction_all_vars(
                                //&context,
                                //self.variables
                                //    .iter()
                                //    .filter(|v| context.contains(*v, value.value)),
                            //),
                        )?;

                        context.set_upper_bound(
                            var,
                            value.value,
                            conjunction_all_vars(&context, &self.variables)
                            //conjunction_all_vars(
                                //&context,
                                //self.variables.iter().filter(|v| {
                                //    context.is_fixed(*v) && context.contains(*v, value.value)
                                //}),
                            //),
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
}
