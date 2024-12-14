use std::{borrow::Borrow, cell::RefCell, collections::HashMap};

use petgraph::{
    dot::Dot,
    graph::{self, DiGraph, NodeIndex},
    prelude::EdgeIndex,
    visit::EdgeRef,
    Graph,
};

use crate::{
    basic_types::Inconsistency,
    conjunction,
    engine::{
        propagation::{LocalId, Propagator, ReadDomains},
        reason::Reason,
        DomainEvents, EmptyDomain,
    },
    predicate,
    predicates::{Predicate, PropositionalConjunction},
    propagators::global_cardinality::{
        conjunction_all_vars, ford_fulkerson_lower_bounds::ford_fulkerson,
    },
    variables::{IntegerVariable, Literal},
};

use super::{ford_fulkerson_lower_bounds::BoundedCapacity, Values};
// local ids of array vars are shifted by ID_X_OFFSET
const ID_X_OFFSET: u32 = 2;

#[derive(Clone, Debug)]
struct GraphData {
    graph: Graph<String, BoundedCapacity>,
    source: NodeIndex,
    sink: NodeIndex,
    variables_nodes: Vec<NodeIndex>,
    values_nodes: Vec<NodeIndex>,
    intermediate_edges: Vec<EdgeIndex>,
}

#[derive(Clone, Debug)]
pub(crate) struct GCCLowerUpper<Variable> {
    variables: Box<[Variable]>,
    values: Box<[Values]>,
    graph_data: GraphData,
}

impl<Variable: IntegerVariable> GCCLowerUpper<Variable> {
    fn construct_graph(
        &self,
        context: &crate::engine::propagation::PropagatorInitialisationContext,
    ) -> GraphData {
        let mut graph = RefCell::new(Graph::<String, BoundedCapacity>::new());

        let source = graph.borrow_mut().add_node("s".to_owned());

        let sink = graph.borrow_mut().add_node("t".to_owned());

        let variables_nodes: Vec<NodeIndex> = self
            .variables
            .iter()
            .enumerate()
            .map(|(i, _)| {
                graph
                    .borrow_mut()
                    .add_node(format!("v{}", i + 1).to_owned())
            })
            .collect();

        let values_nodes: Vec<NodeIndex> = self
            .values
            .iter()
            .map(|v| graph.borrow_mut().add_node(v.value.to_string()))
            .collect();

        // Add from vals to vars if the var has that val in its domain
        let intermediate_edges: Vec<_> = values_nodes
            .iter()
            .zip(&self.values.clone())
            .flat_map(|(ival, val)| {
                variables_nodes
                    .iter()
                    .zip(self.variables.clone())
                    .filter(|(_, var)| context.contains(var, val.value))
                    .map(|(ivar, _)| graph.borrow_mut().add_edge(*ival, *ivar, (0, 1).into()))
            })
            .collect();

        // Add from vars to t
        variables_nodes.iter().for_each(|i| {
            let _ = graph.borrow_mut().add_edge(*i, sink, (0, 1).into());
        });

        // Add from t to s with inf capacity
        let _ = graph
            .borrow_mut()
            .add_edge(sink, source, (0, u32::MAX).into());

        //println!("graph: {:?}", graph);

        //println!("{}", Dot::new(&graph.borrow().clone()));

        GraphData {
            graph: graph.into_inner(),
            source,
            sink,
            variables_nodes,
            values_nodes,
            intermediate_edges,
        }
    }

    fn update_value_edges_feasible_flow(&self, graph_data: &mut GraphData) {
        graph_data
            .values_nodes
            .iter()
            .zip(self.values.clone())
            .for_each(|(i, v)| {
                if v.omin == 0 {
                    if let Some(rem) = graph_data.graph.find_edge(graph_data.source, *i) {
                        let _ = graph_data.graph.remove_edge(rem);
                    }
                } else {
                    let _ = graph_data
                        .graph
                        .update_edge(graph_data.source, *i, (0, v.omin).into());
                }
            });
    }

    fn update_value_edges_max_flow(&self, graph_data: &mut GraphData) {
        // Add from s to vals usig omax and omin
        graph_data
            .values_nodes
            .iter()
            .zip(self.values.clone())
            .for_each(|(i, v)| {
                let _ =
                    graph_data
                        .graph
                        .update_edge(graph_data.source, *i, (v.omin, v.omax).into());
            });
    }

    fn update_graph(
        &self,
        graph_data: &mut GraphData,
        context: &crate::engine::propagation::PropagationContextMut,
    ) {
        let mut intermediate_edges = std::mem::take(&mut graph_data.intermediate_edges);
        for edge in intermediate_edges {
            let _ = graph_data.graph.remove_edge(edge);
        }

        let mut intermediate_edges = Vec::new();
        for (ival, val) in graph_data.values_nodes.iter().zip(&self.values) {
            for (ivar, var) in graph_data.variables_nodes.iter().zip(&self.variables) {
                if context.contains(var, val.value) {
                    intermediate_edges.push(graph_data.graph.add_edge(*ival, *ivar, (0, 1).into()));
                }
            }
        }

        graph_data.intermediate_edges = intermediate_edges;
    }
}

impl<Variable: IntegerVariable> GCCLowerUpper<Variable> {
    pub(crate) fn new(variables: Box<[Variable]>, values: Box<[Values]>) -> Self {
        Self {
            variables,
            values,
            graph_data: GraphData {
                graph: Graph::new(),
                source: NodeIndex::default(),
                sink: NodeIndex::default(),
                variables_nodes: vec![],
                values_nodes: vec![],
                intermediate_edges: vec![],
            },
        }
    }
}

impl<Variable: IntegerVariable + 'static> Propagator for GCCLowerUpper<Variable> {
    fn name(&self) -> &str {
        "Global Cardinality Low Up"
    }

    fn debug_propagate_from_scratch(
        &self,
        mut context: crate::engine::propagation::PropagationContextMut,
    ) -> crate::basic_types::PropagationStatusCP {
        self.variables.iter().for_each(|v| {
            println!(
                "called. u: {:?}, l: {:?}",
                context.upper_bound(v),
                context.lower_bound(v)
            );
        });
        println!();

        //let graph_data = self.construct_graph(&context);

        let mut graph_data = self.graph_data.clone();
        self.update_graph(&mut graph_data, &context);

        self.update_value_edges_feasible_flow(&mut graph_data);

        // Find feasible flow
        let (max_flow, edge_flows) =
            petgraph::algo::ford_fulkerson(&graph_data.graph, graph_data.source, graph_data.sink);

        // Update the flows so they appear on the graph when displayed
        graph_data
            .graph
            .edge_weights_mut()
            .zip(&edge_flows)
            .for_each(|(edge, flow): (&mut BoundedCapacity, &BoundedCapacity)| {
                edge.flow_display = flow.capacity;
            });

        println!("{}", Dot::new(&graph_data.graph));

        println!("Feasible flow: {:?}", max_flow);

        self.update_value_edges_max_flow(&mut graph_data);

        let (max_flow, edge_flows) = ford_fulkerson(
            &graph_data.graph,
            graph_data.source,
            graph_data.sink,
            edge_flows,
            max_flow,
        );

        // Update the flows so they appear on the graph when displayed
        graph_data
            .graph
            .edge_weights_mut()
            .zip(&edge_flows)
            .for_each(|(edge, flow): (&mut BoundedCapacity, &BoundedCapacity)| {
                edge.flow_display = flow.capacity;
            });

        println!("{}", Dot::new(&graph_data.graph));

        // Find maximum flow with lower bounds

        println!("Max flow: {:?}", max_flow);

        // //edge_flows.iter().enumerate().for_each(|(i, e)| {
        // //    println!("{}: {}", i, e);
        // //});

        // let mut result = graph.clone();

        // println!("{}", Dot::new(&result));

        // Find the residual graph
        let mut residual_graph = DiGraph::new();

        // Add the same nodes to the residual graph
        let nodes: Vec<_> = graph_data
            .graph
            .node_indices()
            .map(|idx| residual_graph.add_node(graph_data.graph[idx].clone()))
            .collect();

        assert!(graph_data.graph.node_count() == residual_graph.node_count());

        // Iterate over edges to calculate residual capacities
        for edge in graph_data.graph.edge_references() {
            let src = edge.source();
            let dst = edge.target();
            let BoundedCapacity {
                capacity,
                lower_bound,
                flow_display,
            } = edge.weight();

            // Add forward edge with residual capacity
            if capacity > flow_display {
                let _ = residual_graph.add_edge(
                    nodes[src.index()],
                    nodes[dst.index()],
                    capacity - flow_display,
                );
            }

            // Add reverse edge with residual capacity (equal to flow)
            if *flow_display > *lower_bound {
                let _ =
                    residual_graph.add_edge(nodes[dst.index()], nodes[src.index()], *flow_display);
            }
        }

        println!("{}", Dot::new(&residual_graph));

        let scc = petgraph::algo::tarjan_scc(&residual_graph);

        let mut node_to_scc = HashMap::new();
        for (scc_index, scc) in scc.iter().enumerate() {
            for &node in scc {
                let _ = node_to_scc.insert(node, scc_index);
            }
        }

        // Example: Check if two nodes are in different SCCs
        let are_different =
            |node1: NodeIndex, node2: NodeIndex| node_to_scc.get(&node1) != node_to_scc.get(&node2);

        let mut inconsistent_edges = Vec::new();

        let edges_ref: Vec<_> = graph_data.graph.edge_references().collect();
        for edge in graph_data.intermediate_edges {
            let curr_edge = edges_ref[edge.index()];
            let ivar = curr_edge.target();
            let ival = curr_edge.source();

            if curr_edge.weight().flow_display == 0 && are_different(ivar, ival) {
                inconsistent_edges.push(curr_edge);

                let var_index = graph_data
                    .variables_nodes
                    .iter()
                    .position(|vn| *vn == ivar)
                    .unwrap();

                let val_index = graph_data
                    .values_nodes
                    .iter()
                    .position(|vn| *vn == ival)
                    .unwrap();

                context.remove(
                    &self.variables[var_index],
                    self.values[val_index].value,
                    conjunction_all_vars(&context, &self.variables),
                );

                println!(
                    "Removed: x{} = {}",
                    var_index + 1,
                    self.values[val_index].value
                )
            }
        }

        println!("Inconsistent edges: {:?}", inconsistent_edges);

        //panic!("Test");

        Ok(())
    }

    fn initialise_at_root(
        &mut self,
        context: &mut crate::engine::propagation::PropagatorInitialisationContext,
    ) -> Result<(), PropositionalConjunction> {
        println!("initialize root");

        // Register all variables to domain change events.
        self.variables.iter().enumerate().for_each(|(i, x_i)| {
            let _ = context.register(
                x_i.clone(),
                DomainEvents::ANY_INT,
                LocalId::from(i as u32 + ID_X_OFFSET),
            );
        });

        self.graph_data = self.construct_graph(&context);

        // Register for backtrack events if needed with:
        //context.register_for_backtrack_events(var, domain_events, local_id);
        Ok(())
    }

    fn notify(
        &mut self,
        _context: crate::engine::propagation::PropagationContext,
        _local_id: crate::engine::propagation::LocalId,
        _event: crate::engine::opaque_domain_event::OpaqueDomainEvent,
    ) -> crate::engine::propagation::EnqueueDecision {
        println!("notify");
        crate::engine::propagation::EnqueueDecision::Enqueue
    }

    fn notify_backtrack(
        &mut self,
        _context: crate::engine::propagation::PropagationContext,
        _local_id: crate::engine::propagation::LocalId,
        _event: crate::engine::opaque_domain_event::OpaqueDomainEvent,
    ) {
        println!("notify backtrack");
    }
}
