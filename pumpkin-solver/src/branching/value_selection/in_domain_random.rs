use crate::branching::SelectionContext;
use crate::branching::ValueSelector;
use crate::engine::predicates::predicate::Predicate;
use crate::engine::variables::DomainId;
use crate::engine::variables::Literal;
use crate::engine::variables::PropositionalVariable;
use crate::predicate;

/// A [`ValueSelector`] which assigns to a random value in the domain.
#[derive(Debug, Clone, Copy)]
pub struct InDomainRandom;

impl ValueSelector<DomainId> for InDomainRandom {
    fn select_value(
        &mut self,
        context: &mut SelectionContext,
        decision_variable: DomainId,
    ) -> Predicate {
        let values_in_domain = (context.lower_bound(decision_variable)
            ..=context.upper_bound(decision_variable))
            .filter(|bound| context.contains(decision_variable, *bound))
            .collect::<Vec<_>>();
        let random_index = context
            .random()
            .generate_usize_in_range(0..values_in_domain.len());
        predicate!(decision_variable == values_in_domain[random_index])
    }

    fn is_restart_pointless(&mut self) -> bool {
        false
    }
}

impl ValueSelector<PropositionalVariable> for InDomainRandom {
    fn select_value(
        &mut self,
        context: &mut SelectionContext,
        decision_variable: PropositionalVariable,
    ) -> Predicate {
        Literal::new(decision_variable, context.random().generate_bool(0.5)).into()
    }

    fn is_restart_pointless(&mut self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::basic_types::tests::TestRandom;
    use crate::branching::InDomainRandom;
    use crate::branching::SelectionContext;
    use crate::branching::ValueSelector;
    use crate::predicate;

    #[test]
    fn test_returns_correct_literal() {
        let (assignments_integer, assignments_propositional) =
            SelectionContext::create_for_testing(1, 0, Some(vec![(0, 10)]));
        let mut test_random = TestRandom {
            usizes: vec![3],
            bools: vec![],
        };
        let mut context = SelectionContext::new(
            &assignments_integer,
            &assignments_propositional,
            &mut test_random,
        );
        let domain_ids = context.get_domains().collect::<Vec<_>>();

        let mut selector = InDomainRandom;

        let selected_predicate = selector.select_value(&mut context, domain_ids[0]);

        assert_eq!(selected_predicate, predicate!(domain_ids[0] == 3))
    }
}