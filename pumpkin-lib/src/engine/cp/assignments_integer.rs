use crate::{
    basic_types::{DomainId, IntegerVariableGeneratorIterator, Predicate},
    pumpkin_assert_moderate, pumpkin_assert_simple,
};

use super::{event_sink::EventSink, DomainEvent, PropagatorId, PropagatorVarId};

#[derive(Clone, Default)]
pub struct AssignmentsInteger {
    state: AssignmentsIntegerInternalState,
    current_decision_level: u32,
    trail_delimiter: Vec<u32>, //[i] is the position where the i-th decision level ends (exclusive) on the trail
    trail: Vec<ConstraintProgrammingTrailEntry>,
    domains: Vec<IntegerDomainExplicit>, //[domain_id.id][j] indicates if value j is in the domain of the integer variable

    events: EventSink,
}

impl AssignmentsInteger {
    pub fn increase_decision_level(&mut self) {
        self.current_decision_level += 1;
        self.trail_delimiter.push(self.trail.len() as u32);
    }

    pub fn get_decision_level(&self) -> u32 {
        self.current_decision_level
    }

    pub fn num_domains(&self) -> u32 {
        self.domains.len() as u32
    }

    pub fn get_domains(&self) -> IntegerVariableGeneratorIterator {
        IntegerVariableGeneratorIterator::new(0, self.num_domains())
    }

    pub fn num_trail_entries(&self) -> usize {
        self.trail.len()
    }

    pub fn get_trail_entry(&self, index: usize) -> ConstraintProgrammingTrailEntry {
        self.trail[index]
    }

    pub fn get_last_entry_on_trail(&self) -> ConstraintProgrammingTrailEntry {
        *self.trail.last().unwrap()
    }

    pub fn get_last_predicates_on_trail(&self, num_predicates: usize) -> Vec<Predicate> {
        //perhaps this could be done with an iteration without needing to copy
        self.trail[(self.num_trail_entries() - num_predicates)..self.num_trail_entries()]
            .iter()
            .map(|e| e.predicate)
            .collect::<Vec<Predicate>>()
    }

    pub fn get_last_entries_on_trail(
        &self,
        num_predicates: usize,
    ) -> Vec<ConstraintProgrammingTrailEntry> {
        //perhaps this could be done with an iteration without needing to copy
        self.trail[(self.num_trail_entries() - num_predicates)..self.num_trail_entries()].to_vec()
    }

    //registers the domain of a new integer variable
    //note that this is an internal method that does _not_ allocate additional information necessary for the solver apart from the domain
    //when creating a new integer variable, use create_new_domain_id in the ConstraintSatisfactionSolver
    pub fn grow(&mut self, lower_bound: i32, upper_bound: i32) -> DomainId {
        self.domains
            .push(IntegerDomainExplicit::new(lower_bound, upper_bound));

        self.events.grow();

        DomainId {
            id: self.num_domains() - 1,
        }
    }

    //todo explain that it can return None
    pub fn get_propagator_id_on_trail(&self, index_on_trail: usize) -> Option<PropagatorId> {
        self.trail[index_on_trail]
            .propagator_reason
            .map(|entry| entry.propagator)
    }

    pub fn drain_domain_events(&mut self) -> impl Iterator<Item = (DomainEvent, DomainId)> + '_ {
        self.events.drain()
    }
}

//methods for getting info about the domains
impl AssignmentsInteger {
    pub fn get_lower_bound(&self, domain_id: DomainId) -> i32 {
        self.domains[domain_id].lower_bound
    }

    pub fn get_upper_bound(&self, domain_id: DomainId) -> i32 {
        self.domains[domain_id].upper_bound
    }

    pub fn get_assigned_value(&self, domain_id: DomainId) -> i32 {
        pumpkin_assert_simple!(self.is_domain_assigned(domain_id));
        self.domains[domain_id].lower_bound
    }

    pub fn get_lower_bound_predicate(&self, domain_id: DomainId) -> Predicate {
        Predicate::LowerBound {
            domain_id,
            lower_bound: self.get_lower_bound(domain_id),
        }
    }

    pub fn get_upper_bound_predicate(&self, domain_id: DomainId) -> Predicate {
        let upper_bound = self.get_upper_bound(domain_id);
        Predicate::UpperBound {
            domain_id,
            upper_bound,
        }
    }

    pub fn get_lower_bound_predicates<'a, I: Iterator<Item = &'a DomainId>>(
        &self,
        domain_ids: I,
    ) -> Vec<Predicate> {
        domain_ids
            .map(|i| self.get_lower_bound_predicate(*i))
            .collect()
    }

    pub fn get_upper_bound_predicates<'a, I: Iterator<Item = &'a DomainId>>(
        &self,
        domain_ids: I,
    ) -> Vec<Predicate> {
        domain_ids
            .map(|i| self.get_upper_bound_predicate(*i))
            .collect()
    }

    pub fn get_bound_predicates<'a, I: Iterator<Item = &'a DomainId>>(
        &self,
        domain_ids: I,
    ) -> Vec<Predicate> {
        domain_ids
            .flat_map(|domain_id| {
                [
                    self.get_lower_bound_predicate(*domain_id),
                    self.get_upper_bound_predicate(*domain_id),
                ]
            })
            .collect()
    }

    pub fn is_value_in_domain(&self, domain_id: DomainId, value: i32) -> bool {
        //recall that the data structure is lazy
        //  so we first need to check whether the value falls within the bounds,
        //  and only then check the is_value_in_domain vector
        self.get_lower_bound(domain_id) <= value
            && value <= self.get_upper_bound(domain_id)
            && self.domains[domain_id].is_value_in_domain[value as usize]
    }

    pub fn is_domain_assigned(&self, domain_id: DomainId) -> bool {
        self.get_lower_bound(domain_id) == self.get_upper_bound(domain_id)
    }

    pub fn is_domain_assigned_to_value(&self, domain_id: DomainId, value: i32) -> bool {
        self.is_domain_assigned(domain_id) && self.get_lower_bound(domain_id) == value
    }
}

//methods to change the domains
impl AssignmentsInteger {
    pub fn tighten_lower_bound(
        &mut self,
        domain_id: DomainId,
        new_lower_bound: i32,
        propagator_reason: Option<PropagatorVarId>,
    ) -> DomainOperationOutcome {
        pumpkin_assert_simple!(
            self.state.is_ok(),
            "Cannot tighten lower bound if state is not ok."
        );
        pumpkin_assert_moderate!(new_lower_bound > self.get_lower_bound(domain_id));

        let predicate = Predicate::LowerBound {
            domain_id,
            lower_bound: new_lower_bound,
        };

        if new_lower_bound > self.get_upper_bound(domain_id) {
            self.state = AssignmentsIntegerInternalState::Conflict;
            return DomainOperationOutcome::Failure;
        }

        let old_lower_bound = self.get_lower_bound(domain_id);
        let old_upper_bound = self.get_upper_bound(domain_id);

        self.trail.push(ConstraintProgrammingTrailEntry {
            predicate,
            old_lower_bound,
            old_upper_bound,
            propagator_reason,
        });

        self.domains[domain_id].lower_bound = new_lower_bound;

        if old_lower_bound < new_lower_bound {
            self.events
                .event_occurred(DomainEvent::LowerBound, domain_id);
        }

        if self.is_domain_assigned(domain_id) {
            self.events.event_occurred(DomainEvent::Assign, domain_id);
        }

        DomainOperationOutcome::Success
    }

    pub fn tighten_upper_bound(
        &mut self,
        domain_id: DomainId,
        new_upper_bound: i32,
        propagator_reason: Option<PropagatorVarId>,
    ) -> DomainOperationOutcome {
        pumpkin_assert_simple!(
            self.state.is_ok(),
            "Cannot tighten upper if state is not ok."
        );
        pumpkin_assert_moderate!(new_upper_bound < self.get_upper_bound(domain_id));

        let predicate = Predicate::UpperBound {
            domain_id,
            upper_bound: new_upper_bound,
        };

        if new_upper_bound < self.get_lower_bound(domain_id) {
            self.state = AssignmentsIntegerInternalState::Conflict;
            return DomainOperationOutcome::Failure;
        }

        let old_lower_bound = self.get_lower_bound(domain_id);
        let old_upper_bound = self.get_upper_bound(domain_id);

        self.trail.push(ConstraintProgrammingTrailEntry {
            predicate,
            old_lower_bound,
            old_upper_bound,
            propagator_reason,
        });

        self.domains[domain_id].upper_bound = new_upper_bound;

        if old_upper_bound > new_upper_bound {
            self.events
                .event_occurred(DomainEvent::UpperBound, domain_id);
        }

        if self.is_domain_assigned(domain_id) {
            self.events.event_occurred(DomainEvent::Assign, domain_id);
        }

        DomainOperationOutcome::Success
    }

    pub fn make_assignment(
        &mut self,
        domain_id: DomainId,
        assigned_value: i32,
        propagator_reason: Option<PropagatorVarId>,
    ) -> DomainOperationOutcome {
        pumpkin_assert_simple!(
            self.state.is_ok(),
            "Cannot make assignment if state is not ok."
        );
        pumpkin_assert_moderate!(!self.is_domain_assigned_to_value(domain_id, assigned_value));

        if !self.is_value_in_domain(domain_id, assigned_value) {
            self.state = AssignmentsIntegerInternalState::Conflict;
            return DomainOperationOutcome::Failure;
        }

        //only tighten the lower bound if needed
        if self.get_lower_bound(domain_id) < assigned_value {
            self.tighten_lower_bound(domain_id, assigned_value, propagator_reason);
        }

        //only tighten the uper bound if needed
        if self.get_upper_bound(domain_id) > assigned_value {
            self.tighten_upper_bound(domain_id, assigned_value, propagator_reason);
        }

        DomainOperationOutcome::Success
    }

    pub fn remove_value_from_domain(
        &mut self,
        domain_id: DomainId,
        removed_value_from_domain: i32,
        propagator_reason: Option<PropagatorVarId>,
    ) -> DomainOperationOutcome {
        let predicate = Predicate::NotEqual {
            domain_id,
            not_equal_constant: removed_value_from_domain,
        };

        if !self.is_value_in_domain(domain_id, removed_value_from_domain) {
            self.state = AssignmentsIntegerInternalState::Conflict;
            return DomainOperationOutcome::Failure;
        }

        let old_lower_bound = self.get_lower_bound(domain_id);
        let old_upper_bound = self.get_upper_bound(domain_id);

        self.trail.push(ConstraintProgrammingTrailEntry {
            predicate,
            old_lower_bound,
            old_upper_bound,
            propagator_reason,
        });

        let domain = &mut self.domains[domain_id];

        domain.is_value_in_domain[removed_value_from_domain as usize] = false;

        //adjust the lower bound
        if old_lower_bound == removed_value_from_domain {
            //set the lower bound to the next value
            //  note that the lower bound might increase by more than one, if the values greater than 'not_equal_constant' are also not in the domain
            while domain.is_value_in_domain[domain.lower_bound as usize] {
                domain.lower_bound += 1;

                pumpkin_assert_moderate!(domain.debug_bounds_check());
            }

            self.events
                .event_occurred(DomainEvent::LowerBound, domain_id);
        }
        //adjust the upper bound
        if old_upper_bound == removed_value_from_domain {
            //set the upper bound to the next value
            //  note that the upper bound might increase by more than one, if the values lower than 'not_equal_constant' are also not in the domain
            while domain.is_value_in_domain[domain.upper_bound as usize] {
                domain.upper_bound -= 1;

                pumpkin_assert_moderate!(domain.debug_bounds_check());
            }

            self.events
                .event_occurred(DomainEvent::UpperBound, domain_id);
        }

        self.events.event_occurred(DomainEvent::Any, domain_id);

        DomainOperationOutcome::Success
    }

    //changes the domains according to the predicate
    //  in case the predicate is already true, no changes happen
    //  however in case the predicate would lead to inconsistent domains, e.g., decreasing the upper bound past the lower bound
    //      pumpkin asserts will make the program crash
    pub fn apply_predicate(
        &mut self,
        predicate: &Predicate,
        propagator_reason: Option<PropagatorVarId>,
    ) -> DomainOperationOutcome {
        pumpkin_assert_simple!(
            self.state.is_ok(),
            "Cannot apply predicate after getting into a bad state."
        );

        if self.does_predicate_hold(predicate) {
            return DomainOperationOutcome::Success;
        }

        match *predicate {
            Predicate::LowerBound {
                domain_id,
                lower_bound,
            } => self.tighten_lower_bound(domain_id, lower_bound, propagator_reason),
            Predicate::UpperBound {
                domain_id,
                upper_bound,
            } => self.tighten_upper_bound(domain_id, upper_bound, propagator_reason),
            Predicate::NotEqual {
                domain_id,
                not_equal_constant,
            } => self.remove_value_from_domain(domain_id, not_equal_constant, propagator_reason),
            Predicate::Equal {
                domain_id,
                equality_constant,
            } => self.make_assignment(domain_id, equality_constant, propagator_reason),
        }
    }

    pub fn does_predicate_hold(&self, predicate: &Predicate) -> bool {
        pumpkin_assert_simple!(
            self.state.is_ok(),
            "Cannot evaluate predicate after getting into a bad state."
        );

        match *predicate {
            Predicate::LowerBound {
                domain_id,
                lower_bound,
            } => self.get_lower_bound(domain_id) >= lower_bound,
            Predicate::UpperBound {
                domain_id,
                upper_bound,
            } => self.get_upper_bound(domain_id) <= upper_bound,
            Predicate::NotEqual {
                domain_id,
                not_equal_constant,
            } => !self.is_value_in_domain(domain_id, not_equal_constant),
            Predicate::Equal {
                domain_id,
                equality_constant,
            } => self.is_domain_assigned_to_value(domain_id, equality_constant),
        }
    }

    pub fn undo_trail(&mut self, num_trail_entries_to_remove: usize) {
        pumpkin_assert_simple!(num_trail_entries_to_remove <= self.trail.len());

        for _i in 0..num_trail_entries_to_remove {
            pumpkin_assert_moderate!(
                !self.trail.last().unwrap().predicate.is_equality_predicate(),
                "For now we do not expect equality predicates on the trail, since currently equality predicates are split into lower and upper bound predicates."
            );

            let popped_entry = self.trail.pop().unwrap();
            let domain_id = popped_entry.predicate.get_domain();

            if let Predicate::NotEqual {
                domain_id: _,
                not_equal_constant,
            } = popped_entry.predicate
            {
                self.domains[domain_id].is_value_in_domain[not_equal_constant as usize] = true;
            }

            self.domains[domain_id].lower_bound = popped_entry.old_lower_bound;
            self.domains[domain_id].upper_bound = popped_entry.old_upper_bound;

            pumpkin_assert_moderate!(self.domains[domain_id].debug_bounds_check());
        }
    }

    pub fn synchronise(&mut self, new_decision_level: u32) {
        pumpkin_assert_simple!(new_decision_level < self.current_decision_level);

        let num_trail_entries_to_remove =
            self.trail.len() - self.trail_delimiter[new_decision_level as usize] as usize;

        self.undo_trail(num_trail_entries_to_remove);
        self.current_decision_level = new_decision_level;
        self.trail_delimiter.truncate(new_decision_level as usize);

        if self.is_conflict() {
            self.restore_state_to_ok();
        }
    }

    pub fn is_conflict(&self) -> bool {
        self.state.is_conflict()
    }

    pub fn restore_state_to_ok(&mut self) {
        pumpkin_assert_simple!(self.is_conflict());
        self.state = AssignmentsIntegerInternalState::Ok;
    }
}

#[derive(Clone, Copy)]
pub struct ConstraintProgrammingTrailEntry {
    pub predicate: Predicate,
    pub old_lower_bound: i32, //explicitly store the bound before the predicate was applied so that it is easier later on to update the bounds when backtracking
    pub old_upper_bound: i32,
    pub propagator_reason: Option<PropagatorVarId>, //stores the id of the propagator that made the assignment, only makes sense if a propagation took place, e.g., does _not_ make sense in the case of a decision or if the update was due to synchronisation from the propositional trail
}

#[derive(Clone)]
struct IntegerDomainExplicit {
    lower_bound: i32, //note that even though we only support nonnegative domains, i32 are used over u32 for simplicity
    upper_bound: i32,
    is_value_in_domain: Vec<bool>,
}

impl IntegerDomainExplicit {
    pub fn new(lower_bound: i32, upper_bound: i32) -> IntegerDomainExplicit {
        pumpkin_assert_simple!(
            !lower_bound.is_negative() && !upper_bound.is_negative(),
            "Currently we only support nonnegative domains."
        );
        pumpkin_assert_simple!(
            upper_bound.is_positive(),
            "In principle we could allocate integers with upper bound zero,
            but for now we consider this an error."
        );

        let mut is_value_in_domain = vec![true; (upper_bound + 1) as usize];

        //values outside of the domain need to be marked as false
        for value in is_value_in_domain.iter_mut().take(lower_bound as usize) {
            *value = false;
        }

        IntegerDomainExplicit {
            lower_bound,
            upper_bound,
            is_value_in_domain,
        }
    }

    fn debug_bounds_check(&self) -> bool {
        self.lower_bound <= self.upper_bound
            && (self.lower_bound as usize) < self.is_value_in_domain.len()
            && (self.upper_bound as usize) < self.is_value_in_domain.len()
            && self.is_value_in_domain[self.lower_bound as usize] //the lower and upper bound value should at least be in the is_value_in_domain
            && self.is_value_in_domain[self.upper_bound as usize]
    }
}

#[derive(Clone, Default)]
enum AssignmentsIntegerInternalState {
    #[default]
    Ok,
    Conflict,
}

impl AssignmentsIntegerInternalState {
    pub fn is_ok(&self) -> bool {
        matches!(*self, AssignmentsIntegerInternalState::Ok)
    }

    pub fn is_conflict(&self) -> bool {
        matches!(*self, AssignmentsIntegerInternalState::Conflict)
    }
}

#[derive(PartialEq, Eq)]
pub enum DomainOperationOutcome {
    Success,
    Failure,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_bound_change_lower_bound_event() {
        let mut assignment = AssignmentsInteger::default();
        let d1 = assignment.grow(1, 5);

        assignment.tighten_lower_bound(d1, 2, None);

        let events = assignment.drain_domain_events().collect::<Vec<_>>();
        assert_eq!(events.len(), 2);

        assert_contains_events(&events, d1, [DomainEvent::LowerBound, DomainEvent::Any]);
    }

    #[test]
    fn upper_bound_change_triggers_upper_bound_event() {
        let mut assignment = AssignmentsInteger::default();
        let d1 = assignment.grow(1, 5);

        assignment.tighten_upper_bound(d1, 2, None);

        let events = assignment.drain_domain_events().collect::<Vec<_>>();
        assert_eq!(events.len(), 2);
        assert_contains_events(&events, d1, [DomainEvent::UpperBound, DomainEvent::Any]);
    }

    #[test]
    fn bounds_change_can_also_trigger_assign_event() {
        let mut assignment = AssignmentsInteger::default();

        let d1 = assignment.grow(1, 5);
        let d2 = assignment.grow(1, 5);

        assignment.tighten_lower_bound(d1, 5, None);
        assignment.tighten_upper_bound(d2, 1, None);

        let events = assignment.drain_domain_events().collect::<Vec<_>>();
        assert_eq!(events.len(), 6);

        assert_contains_events(
            &events,
            d1,
            [
                DomainEvent::LowerBound,
                DomainEvent::Any,
                DomainEvent::Assign,
            ],
        );
        assert_contains_events(
            &events,
            d2,
            [
                DomainEvent::UpperBound,
                DomainEvent::Any,
                DomainEvent::Assign,
            ],
        );
    }

    #[test]
    fn making_assignment_triggers_appropriate_events() {
        let mut assignment = AssignmentsInteger::default();

        let d1 = assignment.grow(1, 5);
        let d2 = assignment.grow(1, 5);
        let d3 = assignment.grow(1, 5);

        assignment.make_assignment(d1, 1, None);
        assignment.make_assignment(d2, 5, None);
        assignment.make_assignment(d3, 3, None);

        let events = assignment.drain_domain_events().collect::<Vec<_>>();
        assert_eq!(events.len(), 10);

        assert_contains_events(
            &events,
            d1,
            [
                DomainEvent::Assign,
                DomainEvent::UpperBound,
                DomainEvent::Any,
            ],
        );
        assert_contains_events(
            &events,
            d2,
            [
                DomainEvent::Assign,
                DomainEvent::LowerBound,
                DomainEvent::Any,
            ],
        );
        assert_contains_events(
            &events,
            d3,
            [
                DomainEvent::Assign,
                DomainEvent::LowerBound,
                DomainEvent::UpperBound,
                DomainEvent::Any,
            ],
        );
    }

    #[test]
    fn removal_triggers_any_event() {
        let mut assignment = AssignmentsInteger::default();
        let d1 = assignment.grow(1, 5);

        assignment.remove_value_from_domain(d1, 2, None);

        let events = assignment.drain_domain_events().collect::<Vec<_>>();
        assert_eq!(events.len(), 1);
        assert!(events.contains(&(DomainEvent::Any, d1)));
    }

    fn assert_contains_events(
        slice: &[(DomainEvent, DomainId)],
        domain: DomainId,
        required_events: impl AsRef<[DomainEvent]>,
    ) {
        for event in required_events.as_ref() {
            assert!(slice.contains(&(*event, domain)));
        }
    }
}
