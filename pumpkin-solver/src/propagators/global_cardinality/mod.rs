#[allow(unused)]
use ford_fulkerson_lower_bounds::BoundedCapacity;
use petgraph::{graph::{DiGraph, EdgeReference, NodeIndex}, visit::EdgeRef};
use rand::Rng;

use crate::{
    basic_types::HashMap, engine::propagation::ReadDomains, predicates::{Predicate, PropositionalConjunction}, variables::IntegerVariable
};

pub(crate) mod ford_fulkerson_lower_bounds;
pub(crate) mod gcc_lower_upper;
pub(crate) mod gcc_lower_upper_2;
pub(crate) mod simple_gcc_lower_upper;

#[derive(Clone, Debug, Copy)]
pub struct Values {
    pub value: i32,
    pub omin: u32,
    pub omax: u32,
}


/// Check if, in all variables with a fixed assignment `value` occurs at least `min` and at most `max` times.
fn vars_satisfy_value<Variable: IntegerVariable>(
    vars: &[Variable],
    value: i32,
    min: u32,
    max: u32,
    context: &crate::engine::propagation::PropagationContextMut,
) -> bool {
    let occurences: u32 = vars
        .iter()
        .filter(|v| context.is_fixed(*v) && context.upper_bound(*v) == value)
        .count() as u32;

    occurences >= min && occurences <= max
}

/// Return the most specific explanation possible: A conjunction describing the current domain of all given variables.
fn conjunction_all_vars<'a, I, Variable>(
    context: &crate::engine::propagation::PropagationContextMut,
    vars: I,
) -> PropositionalConjunction
where
    I: IntoIterator<Item = &'a Variable>,
    Variable: IntegerVariable + 'a,
{
    conjunction_all_vars_vec(context, vars).into()
}

fn conjunction_all_vars_vec<'a, I, Variable>(
    context: &crate::engine::propagation::PropagationContextMut,
    vars: I,
) -> Vec<Predicate>
where
    I: IntoIterator<Item = &'a Variable>,
    Variable: IntegerVariable + 'a,
{
    let res: Vec<Predicate> = vars
        .into_iter()
        .flat_map(|var| {
            //[
            //    predicate!(var >= context.lower_bound(var)),
            //    predicate!(var <= context.upper_bound(var)),
            //]
            var.describe_domain(&context.assignments)
        })
        .collect();
    res.into()
}

fn min_count<Variable: IntegerVariable>(
    vars: &[Variable],
    value: i32,
    context: &crate::engine::propagation::PropagationContextMut,
) -> u32 {
    let occurences = vars
        .iter()
        .filter(|v| context.is_fixed(*v) && context.upper_bound(*v) == value)
        .count() as u32;

    occurences
}

fn max_count<Variable: IntegerVariable>(
    vars: &[Variable],
    value: i32,
    context: &crate::engine::propagation::PropagationContextMut,
) -> u32 {
    let occurences = vars.iter().filter(|v| context.contains(*v, value)).count() as u32;

    occurences
}

#[cfg(debug_assertions)]
/// Helper method to convert a graph to a dot string.
/// The graph is optionally colored according to the strongly connected components.
/// If supplied, the variable nodes are placed in the same rank, as well as the value nodes.
/// If supplied, the inconsistent edges are drawn in red and dotted.
fn graph_to_dot<N: ToString, E: std::fmt::Display>(graph: &DiGraph<N, E>, scc: &[Vec<NodeIndex>], variable_nodes: &Vec<NodeIndex>, value_nodes: &Vec<NodeIndex>, inconsistent_edges: &Vec<EdgeReference<BoundedCapacity>>) -> String {
    let mut dot_string = String::new();
    dot_string.push_str("digraph {\n");

    let mut node_colors = HashMap::new();
    let mut rng = rand::thread_rng();
    for component in scc.iter() {
        let color = format!("#{:06x}", rng.gen_range(0..=0xFFFFFF));
        for &node in component {
            let _ = node_colors.insert(node, color.clone());
        }
    }

    if !variable_nodes.is_empty() {
        dot_string.push_str("  { rank=same; ");
        for node in variable_nodes {
            dot_string.push_str(&format!("{}; ", node.index()));
        }
        dot_string.push_str("}\n");
    }

    if !value_nodes.is_empty() {
        dot_string.push_str("  { rank=same; ");
        for node in value_nodes {
            dot_string.push_str(&format!("{}; ", node.index()));
        }
        dot_string.push_str("}\n");
    }

    for node in graph.node_indices() {
        let node_name = format!("{}", node.index()); // n0, n1, n2, ...
        let label = graph[node].to_string();
        let default_color = "black".to_string();
        let color = node_colors.get(&node).unwrap_or(&default_color);
        dot_string.push_str(&format!("  {} [label=\"{}\", color=\"{}\"];\n", node_name, label, color));
    }

    for edge in graph.edge_indices() {
        let (source, target) = graph.edge_endpoints(edge).unwrap();
        let edge_r = graph.edge_references().find(|e| e.id() == edge).unwrap();
        let source_name = format!("{}", source.index());
        let target_name = format!("{}", target.index());
        let edge_style = if inconsistent_edges.iter().any(|e| e.id() == edge) {
            ",color=\"red\", style=\"dotted\""
        } else {
            ""
        };
        dot_string.push_str(&format!("  {} -> {} [label=\"{}\"{}];\n", source_name, target_name, edge_r.weight(), edge_style));
    }

    dot_string.push_str("}\n");
    dot_string
}