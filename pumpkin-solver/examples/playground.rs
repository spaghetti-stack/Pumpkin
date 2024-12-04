use std::borrow::{Borrow, BorrowMut};

use pumpkin_solver::{
    constraints::{self, global_cardinality_lower_upper::Values},
    results::{solution_iterator::IteratedSolution, ProblemSolution, SatisfactionResult},
    termination::Indefinite,
    Solver,
};

fn main() {
    // We create the solver with default options
    let mut solver = Solver::default();

    // We create 3 variables with domains within the range [0, 10]
    let x11 = solver.new_bounded_integer(1, 2);
    let x12 = solver.new_bounded_integer(1, 2);
    let x21 = solver.new_bounded_integer(1, 2);
    let x22 = solver.new_bounded_integer(1, 2);

    let vars = vec![x11, x12, x21, x22];

    // We create the constraint:
    // - x + y + z = 17
    let _ = solver
        .add_constraint(
            constraints::global_cardinality_lower_upper::global_cardinality_lower_upper(
                vars.clone(),
                vec![Values {
                    value: 1,
                    omin: 1,
                    omax: 1,
                }],
            ),
        )
        .post();

    // We create a termination condition which allows the solver to run indefinitely
    let mut termination = Indefinite;
    // And we create a search strategy (in this case, simply the default)
    let mut brancher = solver.default_brancher_over_all_propositional_variables();

    while let IteratedSolution::Solution(solution) = solver
        .get_solution_iterator(&mut brancher, &mut termination)
        .next_solution()
    {
        for var in vars.clone() {
            println!("{:?}: {:?}", var, solution.get_integer_value(var));
        }
        println!("");
    }
}
