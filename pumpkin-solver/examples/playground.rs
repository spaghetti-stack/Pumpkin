
use convert_case::Case;
use log::LevelFilter;
use pumpkin_solver::{
    constraints::{self, global_cardinality_lower_upper::{GccMethod, Values}}, results::{solution_iterator::IteratedSolution, ProblemSolution}, statistics::configure_statistic_logging, termination::Indefinite, Solver
};

fn main() {
    // We create the solver with default options
    let mut solver = Solver::default();

    configure_statistic_logging("",  None,  Some(Case::Camel),  None );


    env_logger::Builder::new()
        .filter_level(LevelFilter::Trace)
        .target(env_logger::Target::Stdout)
        .init();

    // We create 3 variables with domains within the range [0, 10]
    let x1 = solver.new_bounded_integer(1, 2);
    let x2 = solver.new_bounded_integer(1, 2);
    let x3 = solver.new_bounded_integer(1, 2);
    let x4 = solver.new_bounded_integer(1, 2);
    let x5 = solver.new_bounded_integer(1, 3);
    let x6 = solver.new_bounded_integer(2, 5);
    let x7 = solver.new_sparse_integer(vec![3, 5]);

    let vars = vec![x1, x2, x3, x4, x5, x6, x7];

    // We create the constraint:
    let _ = solver
        .add_constraint(
            constraints::global_cardinality_lower_upper::global_cardinality_lower_upper(
                vars.clone(),
                vec![
                    Values {
                        value: 1,
                        omin: 1,
                        omax: 2,
                    },
                    Values {
                        value: 2,
                        omin: 1,
                        omax: 2,
                    },
                    Values {
                        value: 3,
                        omin: 1,
                        omax: 1,
                    },
                    Values {
                        value: 4,
                        omin: 0,
                        omax: 2,
                    },
                    Values {
                        value: 5,
                        omin: 0,
                        omax: 2,
                    },
                ],
            GccMethod::ReginArcConsistent),
        )
        .post();

    // We create a termination condition which allows the solver to run indefinitely
    let mut termination = Indefinite;
    // And we create a search strategy (in this case, simply the default)
    let mut brancher = solver.default_brancher();

    // while let IteratedSolution::Solution(solution) = solver
    if let IteratedSolution::Solution(solution) = solver
        .get_solution_iterator(&mut brancher, &mut termination)
        .next_solution()
    {
        println!("Solution found");
        for var in vars.clone() {
            println!("{:?}: {:?}", var, solution.get_integer_value(var));
        }
        println!("");

        solver.log_statistics();
    }
    //{

    //}
}
