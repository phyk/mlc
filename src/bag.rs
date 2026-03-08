mod test;

use petgraph::graph::EdgeReference;
use petgraph::visit::EdgeRef;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::Hash;

pub type Weight = u64;
pub type NodeId = usize;
pub type UntranslatedNodeId = String;

#[derive(Debug, Clone)]
pub struct Weights(pub Vec<Weight>);

impl From<Vec<u64>> for Weights {
    fn from(v: Vec<u64>) -> Self {
        Weights(v)
    }
}

/// An edge weight tuple containing both Pareto objectives and auxiliary tracked values.
///
/// - `objectives`: the values used for dominance comparisons (e.g., travel time at index 0,
///   monetary cost at index 1)
/// - `auxiliary`: additional values tracked per-label (e.g., distance) but excluded from
///   dominance checks
#[derive(Debug, Clone)]
pub struct WeightsTuple {
    pub objectives: Vec<Weight>,
    pub auxiliary: Vec<Weight>,
}

/// A Pareto label representing a route from the origin to `node_id`.
///
/// `objectives` and `auxiliary` are accumulated sums of edge weights along the path.
/// Only `objectives` are used for dominance comparisons; `auxiliary` values are tracked
/// but ignored when determining whether one label dominates another.
#[derive(Debug, Clone)]
pub struct Label<T> {
    pub objectives: Vec<u64>,
    pub auxiliary: Vec<u64>,
    pub path: Vec<T>,
    pub node_id: T,
}

impl Label<NodeId> {
    pub fn new_along(
        &self,
        edge: &EdgeReference<WeightsTuple>,
        disable_path: bool,
    ) -> Label<NodeId> {
        let weight = edge.weight();
        let objectives = self
            .objectives
            .iter()
            .zip(weight.objectives.iter())
            .map(|(a, b)| a + b)
            .collect();
        let auxiliary = self
            .auxiliary
            .iter()
            .zip(weight.auxiliary.iter())
            .map(|(a, b)| a + b)
            .collect();

        let mut path = if disable_path {
            vec![]
        } else {
            self.path.clone()
        };
        let target_node_id = edge.target().index();
        if !disable_path {
            path.push(target_node_id);
        }
        Label {
            objectives,
            path,
            node_id: target_node_id,
            auxiliary,
        }
    }

    // returns true if the label weakly dominates the other label
    // this is the case if it either strictly dominates the other label
    // or if it is equal to the other label
    fn weakly_dominates(&self, other: &Label<NodeId>) -> bool {
        self.objectives
            .iter()
            .zip(other.objectives.iter())
            .all(|(a, b)| a <= b)
    }
}

impl<T> Ord for Label<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // lexicographical order, but the smaller the better
        // we need a "min-heap" as queue
        match self.objectives.cmp(&other.objectives) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => Ordering::Equal,
        }
    }
}

impl<T> PartialOrd for Label<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // lexicographical order, but the smaller the better
        // we need a "min-heap" as queue
        match self.objectives.cmp(&other.objectives) {
            Ordering::Less => Some(Ordering::Greater),
            Ordering::Greater => Some(Ordering::Less),
            Ordering::Equal => Some(Ordering::Equal),
        }
    }
}

impl<T> PartialEq for Label<T> {
    fn eq(&self, other: &Self) -> bool {
        self.objectives == other.objectives
    }
}

impl<T> Eq for Label<T> {}

impl<T> Hash for Label<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.objectives.hash(state);
    }
}

/// Holds the Pareto-optimal (non-dominated) set of labels for a single node.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Bag<T: Eq + Hash> {
    pub labels: HashSet<Label<T>>,
}

impl Bag<NodeId> {
    pub fn new_start_bag(start_label: Label<NodeId>) -> Bag<NodeId> {
        let mut labels = HashSet::new();
        labels.insert(start_label);
        Bag { labels }
    }

    pub fn new_empty() -> Bag<NodeId> {
        Bag {
            labels: HashSet::new(),
        }
    }

    pub fn add_if_necessary(&mut self, label: Label<NodeId>) -> bool {
        if self.content_dominates(&label) {
            return false;
        }
        self.remove_dominated_by(&label);
        self.labels.insert(label);
        true
    }

    pub fn content_dominates(&self, label: &Label<NodeId>) -> bool {
        for l in &self.labels {
            if l.weakly_dominates(label) {
                return true;
            }
        }
        false
    }

    fn remove_dominated_by(&mut self, label: &Label<NodeId>) {
        self.labels.retain(|l| !label.weakly_dominates(l));
    }
}
