use crate::bag::*;
use bimap::BiMap;
use log::{debug, info};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::{Directed, Graph};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::fs::{read_to_string, File};
use std::hash::Hash;
use std::io::Write;
use std::num::ParseIntError;
use std::str::FromStr;
use std::time::Instant;

use self::limit::Limits;

mod limit;
mod test;

/// Objective index for travel time (must be 0).
const OBJECTIVE_TIME_IDX: usize = 0;
/// Objective index for monetary cost (must be 1).
const OBJECTIVE_COST_IDX: usize = 1;

/// Log queue size every this many iterations.
const QUEUE_LOG_INTERVAL: usize = 1_000;

type UpdateLabelFunc = fn(&Label<usize>, &Label<usize>, u64) -> Label<usize>;

/// Multi-Label Constrained (MLC) shortest-path algorithm.
///
/// Implements a multi-objective variant of Dijkstra that finds all Pareto-optimal routes
/// from an origin to N destinations. Labels are pruned via dominance: a new label reaching
/// a node is discarded if an existing label at that node weakly dominates it. An optional
/// limit/early-stopping mechanism further prunes labels whose (time, cost) objectives
/// exceed known bounds derived from constrained nodes.
pub struct MLC<'a> {
    // problem state
    graph: &'a Graph<Vec<u8>, WeightsTuple, Directed>,
    update_label_func: Option<UpdateLabelFunc>,

    // config
    node_map: Option<BiMap<String, usize>>,
    debug: bool,
    disable_paths: bool,
    enable_limit: bool,
    accuracy: Option<u64>,

    // helper variables
    objective_count: usize,
    auxiliary_count: usize,

    // internal state
    bags: Bags<usize>,
    queue: BinaryHeap<Label<usize>>,
    limits: Limits<u8>,
}

pub type Bags<T> = HashMap<T, Bag<T>>;

#[derive(Debug)]
pub enum MLCError {
    StartNodeNotFound(String),
    NodeMapNotSet,
    UnknownNodeId(usize),
    EmptyStartingQueue,
}

impl fmt::Display for MLCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MLCError::StartNodeNotFound(start_node) => {
                write!(f, "Start node not found: {}", start_node)
            }
            MLCError::NodeMapNotSet => write!(f, "Node map not set"),
            MLCError::UnknownNodeId(node_id) => write!(f, "Unknown node id: {}", node_id),
            MLCError::EmptyStartingQueue => write!(
                f,
                "Starting queue is empty. Specify either a start node or a starting queue."
            ),
        }
    }
}

impl Error for MLCError {}

impl MLC<'_> {
    pub fn new(g: &Graph<Vec<u8>, WeightsTuple, Directed>) -> Result<MLC, Box<dyn Error>> {
        if g.edge_count() == 0 {
            return Err("Graph has no edges".into());
        }

        let sample_edge_weight = g.edge_references().next().unwrap().weight();
        let objective_count = sample_edge_weight.objectives.len();
        let auxiliary_count = sample_edge_weight.auxiliary.len();

        for node in g.node_indices() {
            for edge in g.edges(node) {
                if objective_count != edge.weight().objectives.len() {
                    return Err("Graph has inconsistent edge weights".into());
                }
                if auxiliary_count != edge.weight().auxiliary.len() {
                    return Err("Graph has inconsistent hidden edge weights".into());
                }
            }
        }

        let mut limits = Limits::new();
        let categories = g
            .node_indices()
            .map(|node| g.node_weight(node).unwrap())
            .flatten()
            .collect::<HashSet<_>>();
        for category in categories {
            limits.add_category(category.clone());
        }

        Ok(MLC {
            graph: g,
            bags: HashMap::new(),
            queue: BinaryHeap::new(),
            objective_count,
            node_map: None,
            disable_paths: false,
            auxiliary_count,
            update_label_func: None,
            accuracy: None,
            debug: false,
            limits,
            enable_limit: false,
        })
    }

    pub fn set_update_label_func(
        &mut self,
        update_label_func: fn(&Label<usize>, &Label<usize>, u64) -> Label<usize>,
    ) {
        self.update_label_func = Some(update_label_func);
    }

    pub fn set_accuracy(&mut self, accuracy: u64) {
        self.accuracy = Some(accuracy);
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    pub fn set_node_map(&mut self, node_map: BiMap<String, usize>) {
        self.node_map = Some(node_map);
    }

    pub fn set_disable_paths(&mut self, disable_paths: bool) {
        self.disable_paths = disable_paths;
    }

    pub fn set_enable_limit(&mut self, enable_limit: bool) {
        self.enable_limit = enable_limit;
    }

    /// Sets the starting bags and derives the starting queue from them.
    ///
    /// The bags should be in a consistent state, meaning that the labels in the bags should
    /// not dominate each other.
    ///
    /// # Arguments
    ///
    /// * `bags` - A HashMap of bags, where the key is the node id and the value is the bag.
    pub fn set_bags(&mut self, bags: Bags<usize>) {
        assert!(!bags.is_empty());
        self.bags = bags;
        let mut label_node_tuples = vec![];
        for bag in self.bags.values() {
            for label in &bag.labels {
                let node_weight = self
                    .graph
                    .node_weight(NodeIndex::new(label.node_id))
                    .unwrap();
                if self.enable_limit && node_weight.len() > 0 {
                    label_node_tuples.push((label.clone(), node_weight));
                }

                assert_eq!(label.objectives.len(), self.objective_count);
                assert_eq!(
                    label.auxiliary.len(),
                    self.auxiliary_count,
                    "new label length: {} != {}: old label length",
                    label.auxiliary.len(),
                    self.auxiliary_count
                );

                self.queue.push(label.clone());
            }
        }
        for (label, node_weight) in label_node_tuples {
            self.update_limits(&label, node_weight);
        }
    }

    /// Constructs a start label for `node` with the given initial objective values.
    fn make_start_label(&self, node: usize, initial_objectives: Vec<u64>) -> Label<usize> {
        let path = if self.disable_paths {
            vec![]
        } else {
            vec![node]
        };
        Label {
            objectives: initial_objectives,
            auxiliary: vec![0; self.auxiliary_count],
            path,
            node_id: node,
        }
    }

    pub fn set_start_node(&mut self, start_node: usize) {
        let start_label =
            self.make_start_label(start_node, vec![0; self.objective_count]);
        self.queue.push(start_label.clone());
        self.bags
            .insert(start_node, Bag::new_start_bag(start_label));
    }

    pub fn set_start_node_with_time(&mut self, start_node: usize, time: usize) {
        let mut initial_objectives = vec![0; self.objective_count];
        initial_objectives[OBJECTIVE_TIME_IDX] = time.try_into().unwrap();
        let start_label = self.make_start_label(start_node, initial_objectives);
        self.queue.push(start_label.clone());
        self.bags
            .insert(start_node, Bag::new_start_bag(start_label));
    }

    pub fn set_external_start_node(&mut self, start_node: String) -> Result<(), MLCError> {
        let start_node = self
            .node_map
            .as_ref()
            .ok_or(MLCError::NodeMapNotSet)?
            .get_by_left(&start_node)
            .ok_or(MLCError::StartNodeNotFound(start_node))?;
        self.set_start_node(*start_node);
        Ok(())
    }

    /// Run the MLC algorithm on the graph, starting at the given node.
    /// The node id is expected to be the integer node id.
    ///
    /// Labels are popped from a min-heap ordered by objectives. A label is skipped (stale
    /// label pruning) if it is no longer present in its node's bag — this happens when a
    /// better label arrived after it was enqueued, evicting it via dominance. This pruning
    /// gives ~20% speedup without affecting correctness.
    ///
    /// # Returns
    /// * `Bags<usize>` - The bags of each node.
    pub fn run(&mut self) -> Result<&Bags<usize>, MLCError> {
        debug!("mlc config: {:?}", self);

        let mut counter = 0;
        let mut time = Instant::now();

        let mut n_limit_exceeded = 0;

        if self.enable_limit && !self.limits.is_initialized() {
            panic!("Limits must be initialized before running the algorithm");
        } else {
            debug!("Limits initialized: {:?}", self.limits);
        }
        if self.accuracy.is_none() {
            debug!("Assuming accuracy of 1");
            self.accuracy = Some(1);
        }

        while let Some(label) = self.queue.pop() {
            if self.enable_limit && self.exceeds_limit(&label) {
                n_limit_exceeded += 1;
                continue;
            }

            let node_id = label.node_id;

            // check if this label is still in the bag of its node, if not, we can skip it
            // to speed up the algorithm (~20%)
            if !self
                .bags
                .get(&node_id)
                .ok_or(MLCError::UnknownNodeId(node_id))?
                .labels
                .contains(&label)
            {
                continue;
            }

            for edge in self.graph.edges(NodeIndex::new(node_id)) {
                let old_label = label.clone();
                let mut new_label = label.new_along(&edge, self.disable_paths);
                if let Some(update_label_func) = self.update_label_func {
                    new_label = update_label_func(&old_label, &new_label, self.accuracy.unwrap());
                }
                let target_bag = self
                    .bags
                    .entry(edge.target().index())
                    .or_insert_with(Bag::new_empty);
                if target_bag.add_if_necessary(new_label.clone()) {
                    let target_node_values = self
                        .graph
                        .node_weight(edge.target())
                        .ok_or(MLCError::UnknownNodeId(edge.target().index()))?;
                    if self.enable_limit && target_node_values.len() > 0 {
                        self.update_limits(&new_label, target_node_values);
                    }
                    self.queue.push(new_label);
                }
            }

            counter += 1;
            // print queue size every QUEUE_LOG_INTERVAL iterations
            // if debug is enabled, write labels to csv every 10 seconds
            if counter % QUEUE_LOG_INTERVAL == 0 {
                debug!("queue size: {}", self.queue.len());
                if self.debug {
                    let duration = time.elapsed();
                    if duration.as_secs() > 10 {
                        info!("writing labels to csv");
                        write_bags(&self.translate_bags(&self.bags), "data/labels.csv").unwrap();
                        time = Instant::now();
                    }
                }
            }
        }

        if self.enable_limit && n_limit_exceeded > 0 {
            debug!(
                "{} labels were discarded because they exceeded the limit",
                n_limit_exceeded
            );
            debug!("limits: {:?}", self.limits);
        }

        // remove labels that exceeded the limit
        for (_, bag) in self.bags.iter_mut() {
            bag.labels.retain(|l| {
                if self.enable_limit {
                    let cost = l.objectives[OBJECTIVE_COST_IDX];
                    let time = l.objectives[OBJECTIVE_TIME_IDX];
                    if let Some(max_time) = self.limits.get_max_time_for_cost(cost) {
                        return max_time >= time;
                    }
                }
                return true;
            });
        }

        Ok(&self.bags)
    }

    fn translate_bags(&self, bags: &Bags<usize>) -> Bags<String> {
        let node_map = self
            .node_map
            .as_ref()
            .expect("node_map must be passed when calling translate_bags");
        let mut translated_bags: Bags<String> = HashMap::new();
        for (node_id, bag) in bags {
            let translated_node_id = node_map.get_by_right(node_id).unwrap();
            let translated_bag = Bag {
                labels: bag
                    .labels
                    .iter()
                    .map(|label| Label {
                        node_id: translated_node_id.clone(),
                        path: label
                            .path
                            .iter()
                            .map(|n| node_map.get_by_right(n).unwrap().to_string())
                            .collect(),
                        objectives: label.objectives.clone(),
                        auxiliary: label.auxiliary.clone(),
                    })
                    .collect(),
            };
            translated_bags.insert(translated_node_id.to_string(), translated_bag);
        }
        translated_bags
    }

    #[allow(dead_code)]
    fn write_node_map_as_csv(&self, filename: &str) {
        let node_map = self
            .node_map
            .as_ref()
            .expect("node_map must be passed when calling write_node_map_as_csv");

        let mut file = File::create(filename).unwrap();
        writeln!(file, "node_id,mlc_node_id").unwrap();
        for (key, value) in node_map {
            writeln!(file, "{},{}", key, value).unwrap();
        }
    }

    /// Returns true if the label's (time, cost) objectives exceed the known limit.
    ///
    /// Assumes exactly 2 objectives: time at index `OBJECTIVE_TIME_IDX` and cost at
    /// index `OBJECTIVE_COST_IDX`. Panics if the label has a different number of objectives.
    fn exceeds_limit(&mut self, label: &Label<usize>) -> bool {
        let objectives = &label.objectives;
        if objectives.len() != 2 {
            panic!(
                "exceeds_limit requires exactly 2 objectives (time, cost), got {}",
                objectives.len()
            );
        }
        let cost = objectives[OBJECTIVE_COST_IDX];
        let time = objectives[OBJECTIVE_TIME_IDX];
        self.limits.is_limit_exceeded(cost, time)
    }

    fn update_limits(&mut self, label: &Label<usize>, node_values: &Vec<u8>) {
        for value in node_values.iter() {
            let category = value;
            let cost = label.objectives[OBJECTIVE_COST_IDX];
            let time = label.objectives[OBJECTIVE_TIME_IDX];
            self.limits.update_limit(*category, cost, time);
        }
    }
}

impl fmt::Debug for MLC<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MLC")
            .field("debug", &self.debug)
            .field("disable_paths", &self.disable_paths)
            .field("enable_limit", &self.enable_limit)
            .field(
                "update_label_func_defined",
                &self.update_label_func.is_some(),
            )
            .field("limits_defined", &self.limits)
            .field("objective_count", &self.objective_count)
            .field("auxiliary_count", &self.auxiliary_count)
            .finish()
    }
}

#[derive(Debug)]
struct LabelEntry {
    node_id: NodeId,
    path: Vec<NodeId>,
    values: Vec<Weight>,
}

impl FromStr for LabelEntry {
    type Err = ParseIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // node_id|path_node1,path_node2,...|value1,value2,...
        let mut parts = s.split('|');

        let node_id = parts.next().unwrap().parse::<NodeId>()?;
        let path = parts
            .next()
            .unwrap()
            .split(',')
            .map(|s| s.parse::<NodeId>())
            .collect::<Result<Vec<NodeId>, _>>()?;
        let values = parts
            .next()
            .unwrap()
            .split(',')
            .map(|s| s.parse::<Weight>())
            .collect::<Result<Vec<Weight>, _>>()?;
        Ok(LabelEntry {
            node_id,
            path,
            values,
        })
    }
}

pub fn read_bags(path: &str) -> Result<Bags<usize>, Box<dyn Error>> {
    let mut bags: Bags<usize> = HashMap::new();
    for line in read_to_string(path)?.lines().skip(1) {
        let label_entry: LabelEntry = line.parse()?;
        let label = Label {
            objectives: label_entry.values.clone(),
            auxiliary: vec![],
            path: label_entry.path.clone(),
            node_id: label_entry.node_id,
        };
        let bag = bags
            .entry(label_entry.node_id)
            .or_insert_with(Bag::new_empty);
        bag.add_if_necessary(label);
    }
    Ok(bags)
}

pub fn write_bags<T: Eq + Hash + Display>(
    bags: &Bags<T>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut file = std::fs::File::create(path)?;
    let header = "node_id|path|weights\n";
    file.write_all(header.as_bytes())?;

    for bag in bags.values() {
        for label in bag.labels.iter() {
            let mut values = label.objectives.clone();
            values.extend(label.auxiliary.clone());
            let line = format!(
                "{}|{}|{}\n",
                label.node_id,
                label
                    .path
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<String>>()
                    .join(","),
                values
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<String>>()
                    .join(",")
            );

            file.write_all(line.as_bytes())?;
        }
    }
    Ok(())
}
