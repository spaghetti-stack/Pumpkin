use std::cell::RefCell;

use fnv::{FnvBuildHasher, FnvHashMap};
use log::{debug, warn};
use petgraph::{
    algo::has_path_connecting, graph::{DiGraph, NodeIndex}, prelude::EdgeIndex, visit::EdgeRef, Graph
};

use crate::{
    basic_types::{HashSet, Inconsistency}, create_statistics_struct, engine::{
        propagation::{LocalId, Propagator, ReadDomains},
        DomainEvents,
    }, predicate, predicates::PropositionalConjunction, propagators::global_cardinality::*, variables::IntegerVariable
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
    intermediate_edges: HashSet<EdgeIndex>,
    initial_intermediate_edges: Vec<(NodeIndex, NodeIndex)>,
    node_index_to_variable_index: FnvHashMap<NodeIndex, usize>,
    node_index_to_value_index: FnvHashMap<NodeIndex, usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct GCCLowerUpper<Variable> {
    variables: Box<[Variable]>,
    values: Box<[Values]>,
    graph_data: GraphData,
}

impl<Variable: IntegerVariable> GCCLowerUpper<Variable> {
    fn edge_joins_sccs(variable_index: NodeIndex, value_index: NodeIndex, residual_graph: &Graph<String, u32>, edge_source: NodeIndex, edge_target: NodeIndex) -> bool {
        
        // Clone the original graph
        let mut cloned_graph = residual_graph.clone();
        
        // Add the new edge to the cloned graph
        let _ = cloned_graph.add_edge(edge_source, edge_target, 1);
        
        // Check if there is a path from `source` to `target` in the cloned graph
        has_path_connecting(&cloned_graph, variable_index, value_index, None)
    }
}

impl<Variable: IntegerVariable> GCCLowerUpper<Variable> {
    fn construct_graph(
        &self,
        context: &crate::engine::propagation::PropagatorInitialisationContext,
    ) -> GraphData {
        let graph = RefCell::new(Graph::<String, BoundedCapacity>::new());

        let source = graph.borrow_mut().add_node("s".to_owned());

        let sink = graph.borrow_mut().add_node("t".to_owned());

        let variables_nodes: Vec<NodeIndex> = self
            .variables
            .iter()
            .enumerate()
            .map(|(i, _)| {
                graph
                    .borrow_mut()
                    .add_node(format!("x{}", i + 1).to_owned())
            })
            .collect();

        let values_nodes: Vec<NodeIndex> = self
            .values
            .iter()
            .map(|v| graph.borrow_mut().add_node(v.value.to_string()))
            .collect();
        

        // Add from vals to vars if the var has that val in its domain
        let intermediate_edges: HashSet<_> = values_nodes
            .iter()
            .zip(&self.values.clone())
            .flat_map(|(ival, val)| {
                variables_nodes
                    .iter()
                    .zip(self.variables.clone())
                    .filter(|(_, var)| context.contains(var, val.value))
                    .map(|(ivar, _)| {
                        let e = graph.borrow_mut().add_edge(*ival, *ivar, (0, 1).into());
                        //debug!("add egde: {:?} -> {:?}: {:?}", *ival, *ivar, e);
                        e
                    })
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

        //debug!("graph: {:?}", graph);

        //debug!("{}", Dot::new(&graph.borrow().clone()));

        let node_index_to_value_index: HashMap<NodeIndex, usize> = values_nodes.iter().enumerate().map(|(i, node)| (*node, i)).collect();
        let node_index_to_variable_index: HashMap<NodeIndex, usize> = variables_nodes.iter().enumerate().map(|(i, node)| (*node, i)).collect();

        GraphData {
            graph: graph.into_inner(),
            source,
            sink,
            variables_nodes,
            values_nodes,
            intermediate_edges,
            initial_intermediate_edges: vec![],
            node_index_to_value_index,
            node_index_to_variable_index,
        }
    }

    fn update_value_edges_feasible_flow(&mut self) {
        self.graph_data
            .values_nodes
            .iter()
            //.zip(self.values.clone())
            .for_each(|i,| {
                let v = self.values[*self.graph_data.node_index_to_value_index.get(i).unwrap()];
                if v.omin == 0 {
                    //if let Some(rem) = self.graph_data.graph.find_edge(self.graph_data.source, *i) {
                        //let _ = self.graph_data.graph.remove_edge(rem);
                        
                    let _ = self.graph_data.graph.update_edge(self.graph_data.source, *i, (0, 0).into());
                    //}
                } else {
                    let _ = self.graph_data
                        .graph
                        .update_edge(self.graph_data.source, *i, (0, v.omin).into());
                }
            });
    }

    fn update_value_edges_max_flow(&mut self) {
        // Add from s to vals usig omax and omin
        self.graph_data
            .values_nodes
            .iter()
            .zip(self.values.clone())
            .for_each(|(i, v)| {
                let _ =
                    self.graph_data
                        .graph
                        .update_edge(self.graph_data.source, *i, (v.omin, v.omax).into());
            });
    }

    fn update_graph(
        &mut self,
        context: &crate::engine::propagation::PropagationContextMut,
    ) {

        let intermediate_edges = std::mem::take(&mut self.graph_data.intermediate_edges);
        
        // Remove the specified edges using retain_edges
        // Using remove_edge shifts the indices, causing potential bugs.
        let edge_set: std::collections::HashSet<EdgeIndex> = intermediate_edges.into_iter().collect();
        self.graph_data.graph.retain_edges(|_, edge_index| !edge_set.contains(&edge_index));

        let mut intermediate_edges = Vec::new();
        for (ival, val) in self.graph_data.values_nodes.iter().zip(&self.values) {
            for (ivar, var) in self.graph_data.variables_nodes.iter().zip(&self.variables) {
                if context.contains(var, val.value) {
                    intermediate_edges.push(self.graph_data.graph.add_edge(*ival, *ivar, (0, 1).into()));
                }
            }
        }

        self.graph_data.intermediate_edges = intermediate_edges.into_iter().collect();
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
                intermediate_edges: HashSet::with_hasher(FnvBuildHasher::default()),
                initial_intermediate_edges: vec![],
                node_index_to_variable_index: FnvHashMap::default(),
                node_index_to_value_index: FnvHashMap::default(),
            },
        }
    }
}

impl<Variable: IntegerVariable + 'static> Propagator for GCCLowerUpper<Variable> {
    fn name(&self) -> &str {
        "Global Cardinality Low Up"
    }

    fn propagate(&mut self, mut context: crate::engine::propagation::PropagationContextMut) -> crate::basic_types::PropagationStatusCP {
        
        #[cfg(debug_assertions)]
        {
            self.variables.iter().for_each(|v| {
                debug!(
                    "var: u: {:?}, l: {:?}",
                    context.upper_bound(v),
                    context.lower_bound(v)
                );
            });
    
            self.values.iter().for_each(|v| {
                debug!(
                    "value: v: {:?}, omin: {:?}, omax: {:?}",
                    v.value, v.omin, v.omax
                );
            });
        }

         self.values.iter().try_for_each(|value| {
            let min = min_count(&self.variables, value.value, &context);
            let max = max_count(&self.variables, value.value, &context);
            debug!("v: {:?}, min_count: {:?}, max_count: {:?}", value, min, max);

            // If this is false, there is definitely no solution
            if min > value.omax || max < value.omin {
                debug!("Inconsistency: {:?}", value);
                // Constraint violation
                return Err(Inconsistency::Conflict(conjunction_all_vars(
                    &context,
                    &self.variables,
                )));
            }
            Ok(())
        })?; 

        //let self.graph_data = self.construct_graph(&context);
                                //conjunction_all_vars(&context, &self.variables),


        self.update_graph(&context);

        self.update_value_edges_feasible_flow();

        // Find feasible flow
        let (max_flow, edge_flows) =
            petgraph::algo::ford_fulkerson(&self.graph_data.graph, self.graph_data.source, self.graph_data.sink);

        // Update the flows so they appear on the graph when displayed
        self.graph_data
            .graph
            .edge_weights_mut()
            .zip(&edge_flows)
            .for_each(|(edge, flow): (&mut BoundedCapacity, &BoundedCapacity)| {
                edge.flow_display = flow.capacity;
            });


        #[cfg(debug_assertions)]
        {
            let dot = graph_to_dot(&self.graph_data.graph, &vec![], &self.graph_data.variables_nodes, &self.graph_data.values_nodes, &vec![]);
            debug!("feasible flow: {}", dot);
        }

        // If feasible flow less than sum of lower bounds, then no solution exists
        let sum_lower_bounds: u32 = self
            .values
            .iter()
            .map(|v| v.omin)
            .sum::<u32>();

        debug!("Feasible flow: {:?}, Sum lower bounds: {:?}", max_flow, sum_lower_bounds);

        if max_flow.capacity < sum_lower_bounds {
            debug!("Inconsistency: flow {:?}, sum lower bounds: {:?}", max_flow, sum_lower_bounds);
              return Err(Inconsistency::Conflict(conjunction_all_vars(
                &context,
                &self.variables,
            )));
        }

        self.update_value_edges_max_flow();

        let (max_flow, edge_flows) = ford_fulkerson_lower_bounds::ford_fulkerson(
            &self.graph_data.graph,
            self.graph_data.source,
            self.graph_data.sink,
            edge_flows,
            max_flow,
        );

        // Update the flows so they appear on the graph when displayed
        self.graph_data
            .graph
            .edge_weights_mut()
            .zip(&edge_flows)
            .for_each(|(edge, flow): (&mut BoundedCapacity, &BoundedCapacity)| {
                edge.flow_display = flow.capacity;
            });


        #[cfg(debug_assertions)]
        {
            let dot = graph_to_dot(&self.graph_data.graph, &vec![], &self.graph_data.variables_nodes, &self.graph_data.values_nodes, &vec![]);
            debug!("max flow {}", dot);
        }

        // Find maximum flow with lower bounds

        debug!("Max flow: {:?}", max_flow);

        // //edge_flows.iter().enumerate().for_each(|(i, e)| {
        // //    debug!("{}: {}", i, e);
        // //});

        // let mut result = graph.clone();

        // debug!("{}", Dot::new(&result));

        // Find the residual graph
        let mut residual_graph = DiGraph::new();

        // Add the same nodes to the residual graph
        let nodes: Vec<_> = self.graph_data
            .graph
            .node_indices()
            .map(|idx| residual_graph.add_node(self.graph_data.graph[idx].clone()))
            .collect();

        assert!(self.graph_data.graph.node_count() == residual_graph.node_count());

        // Iterate over edges to calculate residual capacities
        for edge in self.graph_data.graph.edge_references() {
            let src = edge.source();
            let dst = edge.target();
            let BoundedCapacity {
                capacity,
                lower_bound,
                flow_display,
            } = edge.weight();

            // Add forward edge with residual capacity
            //if capacity > flow_display {
            if flow_display < capacity {
                let _ = residual_graph.add_edge(
                    nodes[src.index()],
                    nodes[dst.index()],
                    capacity - flow_display,
                );
            }

            // Add reverse edge with residual capacity (equal to flow)
            if *flow_display > *lower_bound {
                let _ =
                    residual_graph.add_edge(nodes[dst.index()], nodes[src.index()], *flow_display - *lower_bound);
                    //residual_graph.add_edge(nodes[dst.index()], nodes[src.index()], *flow_display);
            }
        }


        if residual_graph.contains_edge(self.graph_data.source, self.graph_data.sink) {
            warn!("Residual graph contains edge from source to sink. Regin doesn't specify if this is allowed, Katsirelos et al. 2011 does not allow this.");
            //assert!(false);
        }

        let scc = petgraph::algo::tarjan_scc(&residual_graph);


        let mut node_to_scc = HashMap::new();
        for (scc_index, scc) in scc.iter().enumerate() {
            for &node in scc {
                let _ = node_to_scc.insert(node, scc_index);
            }
        }

        debug!("scc: {:?}", scc);

        // Check if two nodes are in different SCCs
        let are_different =
            |node1: NodeIndex, node2: NodeIndex| node_to_scc.get(&node1) != node_to_scc.get(&node2);

        let mut inconsistent_edges = Vec::new();

        let edges_ref: Vec<_> = self.graph_data.graph.edge_references().collect();
        for edge in &self.graph_data.intermediate_edges {
            let curr_edge = edges_ref[edge.index()];
            let ivar = curr_edge.target();
            let ival = curr_edge.source();

            let var_index = self.graph_data.node_index_to_variable_index[&ivar];
            /*let var_index = self.graph_data
                .variables_nodes
                .iter()
                .position(|vn| *vn == ivar)
                .unwrap();
            */

            let val_index = self.graph_data.node_index_to_value_index[&ival];

            /* 
            let val_index = self.graph_data
                .values_nodes
                .iter()
                .position(|vn| *vn == ival)
                .unwrap();
            */

            if curr_edge.weight().flow_display == 0 && are_different(ivar, ival) {
                inconsistent_edges.push(curr_edge);

                /* let mut expl = Vec::new();
                let mut expl2 = Vec::new();
                self.graph_data.variables_nodes.iter().zip(self.variables.clone()).enumerate().for_each( |(ic, (vari_c,var_c)) | {
                    
                    self.graph_data.values_nodes.iter().zip(self.values.clone()).for_each(|(vali_c, val_c)| {

                        let var_index_c = self.graph_data
                        .variables_nodes
                        .iter()
                        .position(|vn| *vn == *vari_c)
                        .unwrap();
        
                        let val_index_c = self.graph_data
                            .values_nodes
                            .iter()
                            .position(|vn| *vn == *vali_c)
                            .unwrap();

                        if *vari_c != ivar && !are_different(*vari_c, ivar) && are_different(ivar, *vali_c) {
                            expl.push((vari_c.clone(), vali_c));
                            expl2.push(predicate!( self.variables[val_index_c] != self.values[val_index_c].value ));
                        }
                    });
                }); */



                /* warn!("expl: {:?}", expl);
                let expl2: PropositionalConjunction = expl2.into();
                warn!("expl2: {:?}", expl2);
                warn!("conj all vars: {:?}", conjunction_all_vars(&context, &self.variables));
                */
                let naive_expl: Vec<Predicate> = conjunction_all_vars_vec(&context, &self.variables);
                let mut expl2: Vec<Predicate> = vec![];

                // Avoid lenghty explanation computation if the naive explanation only contains one predicate
                if naive_expl.len() > 1 {
                    
                for (i, j) in self.graph_data.initial_intermediate_edges.iter() {

                    let var_index = self.graph_data.node_index_to_variable_index[&j];

                    let val_index = self.graph_data.node_index_to_value_index[&i];

                    let var = &self.variables[var_index];
                    let val = self.values[val_index].value;
                    
                    if !context.contains(var, val ) && Self::edge_joins_sccs(ivar, ival, &residual_graph, *i, *j) {

                        expl2.push(predicate!(  var != val ));

                    }
                };

                }else {
                    expl2 = naive_expl.clone();
                }

                
                debug!(
                    "Removed: x{} = {}. expl_pred: {:?}, expl2: {:?}",
                    var_index + 1,
                    self.values[val_index].value,
                    naive_expl,
                    expl2
                );

                

                context.remove(
                    &self.variables[var_index],
                    self.values[val_index].value,
                    //naive_expl.into::<PropositionalConjunction>(),
                    Into::<PropositionalConjunction>::into(expl2),
                )?;
            } else {
                debug!(
                    "Kept: x{} = {}",
                    var_index + 1,
                    self.values[val_index].value
                );
            }
        }

        #[cfg(debug_assertions)]
        {
            //let dot = Dot::new(&residual_graph);
            let dot = graph_to_dot(&residual_graph, &scc, self.graph_data.variables_nodes.as_ref(), self.graph_data.values_nodes.as_ref(), &inconsistent_edges);
            debug!("residual graph: {}", dot);
        }

        debug!("Inconsistent edges: {:?}", inconsistent_edges);

       // panic!("Test");

       Ok(())

    }

    fn debug_propagate_from_scratch(
        &self,
        _context: crate::engine::propagation::PropagationContextMut,
    ) -> crate::basic_types::PropagationStatusCP {
        Ok(())
    }

    fn initialise_at_root(
        &mut self,
        context: &mut crate::engine::propagation::PropagatorInitialisationContext,
    ) -> Result<(), PropositionalConjunction> {
        debug!("initialize root");

        // Register all variables to domain change events.
        self.variables.iter().enumerate().for_each(|(i, x_i)| {
            let _ = context.register(
                x_i.clone(),
                DomainEvents::ANY_INT,
                LocalId::from(i as u32 + ID_X_OFFSET),
            );
        });

        self.graph_data = self.construct_graph(context);

        // Needed for creating explanations
        self.graph_data.initial_intermediate_edges = self.graph_data.intermediate_edges.iter().map(|e| {
            let edge = self.graph_data.graph.edge_references().nth(e.index()).unwrap();
            (edge.source(), edge.target())
        }).collect();
        

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

#[cfg(test)]
mod tests {
    use crate::{engine::test_solver::TestSolver, propagators::global_cardinality::Values};

    use super::GCCLowerUpper;


    #[test]
    fn test_propagation(){
        let _ = env_logger::builder().is_test(true).try_init();
        let mut solver = TestSolver::default();

        let x_a = solver.new_variable(1, 2);
        let x_b = solver.new_variable(1, 2);
        let x_c = solver.new_variable(1, 4);
        let x_d = solver.new_variable(1, 4);

        let propagator = solver.new_propagator(GCCLowerUpper::new(vec![x_a, x_b, x_c, x_d].into(), vec![
            Values {
                value: 1,
                omin: 0,
                omax: 1,
            },
            Values {
                value: 2,
                omin: 0,
                omax: 1,
            },
            Values {
                value: 3,
                omin: 0,
                omax: 1,
            },
            Values {
                value: 4,
                omin: 0,
                omax: 1,
            }
        ].into())).expect("No empty domains");

        //let r = solver.propagate(propagator);

        assert!(solver.propagate_until_fixed_point(propagator).is_ok());

        assert!(!solver.contains(x_c, 1));
        assert!(!solver.contains(x_c, 2));
        assert!(!solver.contains(x_d, 1));
        assert!(!solver.contains(x_d, 2));

        assert!(solver.contains(x_d, 3));
        assert!(solver.contains(x_d, 4));
        assert!(solver.contains(x_c, 3));
        assert!(solver.contains(x_c, 4));
        assert!(solver.contains(x_a, 1));
        assert!(solver.contains(x_a, 2));
        assert!(solver.contains(x_b, 1));
        assert!(solver.contains(x_b, 2));
    }

    #[test]
    fn test_propagation_2(){
        let _ = env_logger::builder().is_test(true).try_init();
        let mut solver = TestSolver::default();

        let x_1 = solver.new_variable(1, 2);
        let x_2 = solver.new_variable(2, 3);
        let x_3 = solver.new_variable(2, 4);

        let propagator = solver.new_propagator(GCCLowerUpper::new(vec![x_1, x_2, x_3].into(), vec![
            Values {
                value: 1,
                omin: 0,
                omax: 1,
            },
            Values {
                value: 2,
                omin: 0,
                omax: 1,
            },
            Values {
                value: 3,
                omin: 0,
                omax: 1,
            },
            Values {
                value: 4,
                omin: 0,
                omax: 1,
            }
        ].into())).expect("No empty domains");

        solver.remove(x_3, 4);

        //let r = solver.propagate(propagator);

        assert!(solver.propagate_until_fixed_point(propagator).is_ok());
    }
}