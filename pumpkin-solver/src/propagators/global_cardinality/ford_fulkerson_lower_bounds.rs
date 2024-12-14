use std::{
    collections::VecDeque,
    fmt::Display,
    ops::{Add, Sub},
};

use num::zero;
use petgraph::{algo::PositiveMeasure, visit::EdgeRef, Direction};

use {
    petgraph::data::DataMap,
    petgraph::visit::{
        EdgeCount, EdgeIndexable, IntoEdges, IntoEdgesDirected, NodeCount, NodeIndexable, VisitMap,
        Visitable,
    },
};

#[derive(Default, Debug, Clone, Copy)]
pub(crate) struct BoundedCapacity {
    pub capacity: u32,
    pub lower_bound: u32,
    pub flow_display: u32,
}

impl From<(u32, u32)> for BoundedCapacity {
    fn from(tuple: (u32, u32)) -> Self {
        let (lower_bound, capacity) = tuple;
        assert!(capacity >= lower_bound, "Capacity must be >= lower bound");
        BoundedCapacity::new(lower_bound, capacity)
    }
}

impl Display for BoundedCapacity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{}),f={}",
            self.lower_bound, self.capacity, self.flow_display
        )
    }
}

impl BoundedCapacity {
    pub fn new(lower_bound: u32, capacity: u32) -> Self {
        Self {
            capacity,
            lower_bound,
            flow_display: 0,
        }
    }
}

impl Sub for BoundedCapacity {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        BoundedCapacity::new(0, self.capacity - other.capacity)
    }
}

impl Add for BoundedCapacity {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        BoundedCapacity::new(0, self.capacity + other.capacity)
    }
}

impl PositiveMeasure for BoundedCapacity {
    fn zero() -> Self {
        BoundedCapacity {
            capacity: zero(),
            lower_bound: zero(),
            flow_display: 0,
        }
    }

    fn max() -> Self {
        BoundedCapacity {
            capacity: <u32 as PositiveMeasure>::max(),
            lower_bound: 0,
            flow_display: 0,
        }
    }
}

impl std::cmp::PartialEq for BoundedCapacity {
    fn eq(&self, other: &Self) -> bool {
        self.capacity == other.capacity
    }
}

impl std::cmp::PartialOrd for BoundedCapacity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.capacity.partial_cmp(&other.capacity)
    }
}

fn residual_capacity<N>(
    network: N,
    edge: N::EdgeRef,
    vertex: N::NodeId,
    flow: N::EdgeWeight,
) -> N::EdgeWeight
where
    N: IntoEdgesDirected<EdgeWeight = BoundedCapacity> + petgraph::visit::NodeIndexable,
    N::EdgeWeight: Sub<Output = N::EdgeWeight> + PositiveMeasure,
{
    if vertex == edge.source() {
        // backward edge
        BoundedCapacity {
            capacity: flow.capacity - edge.weight().lower_bound,
            lower_bound: 0,
            flow_display: 0,
        }
    } else if vertex == edge.target() {
        // forward edge
        return *edge.weight() - flow;
    } else {
        let end_point = NodeIndexable::to_index(&network, vertex);
        panic!("Illegal endpoint {}", end_point);
    }
}

/// Gets the other endpoint of graph edge, if any, otherwise panics.
fn other_endpoint<N>(network: N, edge: N::EdgeRef, vertex: N::NodeId) -> N::NodeId
where
    N: NodeIndexable + IntoEdges,
{
    if vertex == edge.source() {
        edge.target()
    } else if vertex == edge.target() {
        edge.source()
    } else {
        let end_point = NodeIndexable::to_index(&network, vertex);
        panic!("Illegal endpoint {}", end_point);
    }
}

/// Tells whether there is an augmented path in the graph
fn has_augmented_path<N>(
    network: N,
    source: N::NodeId,
    destination: N::NodeId,
    edge_to: &mut [Option<N::EdgeRef>],
    flows: &[N::EdgeWeight],
) -> bool
where
    N: IntoEdgesDirected<EdgeWeight = BoundedCapacity>
        + NodeCount
        + IntoEdgesDirected
        + NodeIndexable
        + EdgeIndexable
        + Visitable,
    N::EdgeWeight: Sub<Output = N::EdgeWeight> + PositiveMeasure,
{
    let mut visited = network.visit_map();
    let mut queue = VecDeque::new();
    visited.visit(source);
    queue.push_back(source);

    while let Some(vertex) = queue.pop_front() {
        let out_edges = network.edges_directed(vertex, Direction::Outgoing);
        let in_edges = network.edges_directed(vertex, Direction::Incoming);
        for edge in out_edges.chain(in_edges) {
            let next = other_endpoint(&network, edge, vertex);
            let edge_index: usize = EdgeIndexable::to_index(&network, edge.id());
            let residual_cap = residual_capacity(&network, edge, next, flows[edge_index]);
            if !visited.is_visited(&next) && (residual_cap > N::EdgeWeight::zero()) {
                visited.visit(next);
                edge_to[NodeIndexable::to_index(&network, next)] = Some(edge);
                if destination == next {
                    return true;
                }
                queue.push_back(next);
            }
        }
    }
    false
}

fn adjust_residual_flow<N>(
    network: N,
    edge: N::EdgeRef,
    vertex: N::NodeId,
    flow: N::EdgeWeight,
    delta: N::EdgeWeight,
) -> N::EdgeWeight
where
    N: IntoEdgesDirected<EdgeWeight = BoundedCapacity> + NodeIndexable,
    N::EdgeWeight: Sub<Output = N::EdgeWeight> + PositiveMeasure,
{
    if vertex == edge.source() {
        // backward edge
        flow - delta
    } else if vertex == edge.target() {
        // forward edge
        flow + delta
    } else {
        let end_point = NodeIndexable::to_index(&network, vertex);
        panic!("Illegal endpoint {}", end_point);
    }
}

/// \[Modified\] Ford-Fulkerson algorithm
///
/// Adapted from [petgraph](https://docs.rs/petgraph/latest/petgraph/algo/ford_fulkerson/fn.ford_fulkerson.html) and modified to work with lower bounds on edges.
///
/// If it terminates, it returns the maximum flow and also the computed edge flows.
///
/// [ff]: https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm
///
/// # Example
/// ```rust
/// use petgraph::Graph;
/// use petgraph::algo::ford_fulkerson;
/// // Example from CLRS book
/// let mut graph = Graph::<u8, u8>::new();
/// let source = graph.add_node(0);
/// let _ = graph.add_node(1);
/// let _ = graph.add_node(2);
/// let _ = graph.add_node(3);
/// let _ = graph.add_node(4);
/// let destination = graph.add_node(5);
/// graph.extend_with_edges(&[
///    (0, 1, 16),
///    (0, 2, 13),
///    (1, 2, 10),
///    (1, 3, 12),
///    (2, 1, 4),
///    (2, 4, 14),
///    (3, 2, 9),
///    (3, 5, 20),
///    (4, 3, 7),
///    (4, 5, 4),
/// ]);
/// let (max_flow, _) = ford_fulkerson(&graph, source, destination);
/// assert_eq!(23, max_flow);
/// ```
pub fn ford_fulkerson<N>(
    network: N,
    source: N::NodeId,
    destination: N::NodeId,
    mut flows_initial: Vec<BoundedCapacity>,
    mut max_flow: BoundedCapacity,
) -> (N::EdgeWeight, Vec<N::EdgeWeight>)
where
    N: NodeCount
        + EdgeCount
        + IntoEdgesDirected
        + EdgeIndexable
        + NodeIndexable
        + DataMap
        + Visitable,
    N: IntoEdgesDirected<EdgeWeight = BoundedCapacity>,
    N::NodeId: PartialEq + Clone + Copy,
{
    let mut edge_to = vec![None; network.node_count()];
    let mut flows = vec![N::EdgeWeight::zero(); network.edge_count()];
    flows[..flows_initial.len()].copy_from_slice(&flows_initial);

    //let mut flows = vec![N::EdgeWeight::zero(); network.edge_count()];
    //let mut max_flow = N::EdgeWeight::zero();
    while has_augmented_path(&network, source, destination, &mut edge_to, &flows) {
        let mut path_flow = N::EdgeWeight::max();

        // Find the bottleneck capacity of the path
        let mut vertex = destination;
        let mut vertex_index = NodeIndexable::to_index(&network, vertex);
        while let Some(edge) = edge_to[vertex_index] {
            let edge_index = EdgeIndexable::to_index(&network, edge.id());
            let residual_capacity = residual_capacity(&network, edge, vertex, flows[edge_index]);
            // Minimum between the current path flow and the residual capacity.
            path_flow = if path_flow > residual_capacity {
                residual_capacity
            } else {
                path_flow
            };
            vertex = other_endpoint(&network, edge, vertex);
            vertex_index = NodeIndexable::to_index(&network, vertex);
        }

        // Update the flow of each edge along the path
        let mut vertex = destination;
        let mut vertex_index = NodeIndexable::to_index(&network, vertex);
        while let Some(edge) = edge_to[vertex_index] {
            let edge_index = EdgeIndexable::to_index(&network, edge.id());
            flows[edge_index] =
                adjust_residual_flow(&network, edge, vertex, flows[edge_index], path_flow);
            vertex = other_endpoint(&network, edge, vertex);
            vertex_index = NodeIndexable::to_index(&network, vertex);
        }
        max_flow = max_flow + path_flow;
    }
    (max_flow, flows)
}
