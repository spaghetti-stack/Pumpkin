use std::borrow::{Borrow, BorrowMut};

use log::LevelFilter;
use petgraph::{algo::ford_fulkerson, dot::Dot, visit::IntoEdges, Graph};
use pumpkin_solver::{
    constraints::{self, global_cardinality_lower_upper::Values},
    results::{solution_iterator::IteratedSolution, ProblemSolution, SatisfactionResult},
    termination::Indefinite,
    Solver,
};

fn main() {
    let mut graph = Graph::<&str, u32>::new();
    let source = graph.add_node("source");
    let _ = graph.add_node("a");
    let _ = graph.add_node("b");
    let _ = graph.add_node("c");
    let _ = graph.add_node("d");
    let _ = graph.add_node("e");
    let _ = graph.add_node("f");
    let _ = graph.add_node("g");

    let _ = graph.add_node("m");
    let _ = graph.add_node("a");
    let _ = graph.add_node("n");
    let _ = graph.add_node("h");
    let _ = graph.add_node("r");

    let sink = graph.add_node("sink");

    let superource = graph.add_node("supersource"); //14
    let supersink = graph.add_node("supersink"); //15

    graph.extend_with_edges([(8, 1, 1), (8, 2, 1), (8, 3, 1), (8, 4, 1), (8, 5, 1)]);
    graph.extend_with_edges([
        (9, 1, 1),
        (9, 2, 1),
        (9, 3, 1),
        (9, 4, 1),
        (9, 5, 1),
        (9, 6, 1),
    ]);

    graph.extend_with_edges([(10, 5, 1), (10, 6, 1), (10, 7, 1)]);

    graph.extend_with_edges([(11, 6, 1)]);

    graph.extend_with_edges([(12, 6, 1), (12, 7, 1)]);

    graph.extend_with_edges([
        (1, 13, 1),
        (2, 13, 1),
        (3, 13, 1),
        (4, 13, 1),
        (5, 13, 1),
        (6, 13, 1),
        (7, 13, 1),
    ]);

    graph.extend_with_edges([(0, 8, 1), (0, 9, 1), (0, 10, 0), (0, 11, 2), (0, 12, 2)]);

    graph.extend_with_edges([(0, 15, 3)]);

    graph.extend_with_edges([(14, 8, 1), (14, 9, 1), (14, 10, 1)]);

    graph.extend_with_edges([(13, 0, 1000)]);

    println!("graph: {:?}", graph);

    println!("{}", Dot::new(&graph));

    let (max_flow, edge_flows) = ford_fulkerson(&graph, superource, supersink);
    //println!("{:?}", edge_flows);

    //edge_flows.iter().enumerate().for_each(|(i, e)| {
    //    println!("{}: {}", i, e);
    //});

    let mut result = graph.clone();

    result
        .edge_weights_mut()
        .zip(edge_flows)
        .for_each(|(w, f)| *w = f);

    println!("{}", Dot::new(&result));

    assert_eq!(23, max_flow);

    // We create the solver with default options
    let mut solver = Solver::default();

    env_logger::Builder::new()
        .filter_level(LevelFilter::Trace)
        .target(env_logger::Target::Stdout)
        .init();

    // We create 3 variables with domains within the range [0, 10]
    let x11 = solver.new_bounded_integer(1, 4);
    let x12 = solver.new_bounded_integer(1, 2);
    let x21 = solver.new_bounded_integer(1, 5);
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
    }
    //{

    //}
}
