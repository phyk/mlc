mod test;

use petgraph::graph::EdgeReference;
use petgraph::visit::EdgeRef;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::Hash;
use std::ops::Add;

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
    pub objective: Objective,
    pub auxiliary: Vec<u64>,
    pub path: Vec<T>,
    pub node_id: T,
}

#[derive(Debug, Clone)]
pub struct Objective {
    pub time: u64,
    pub cost: u64,
}

impl Add for Objective {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Objective {
            time: self.time + rhs.time,
            cost: self.cost + rhs.cost,
        }
    }
}

impl PartialEq for Objective {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.cost == other.cost
    }
}

impl Eq for Objective {}

impl PartialOrd for Objective {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Objective {
    fn cmp(&self, other: &Self) -> Ordering {
        let ordering = self.time.cmp(&other.time);
        let second_ordering = self.cost.cmp(&other.cost);
        if (ordering.is_gt() & second_ordering.is_ge())
            || (second_ordering.is_gt() & ordering.is_ge())
        {
            Ordering::Greater
        } else if ordering.is_eq() & second_ordering.is_eq() {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    }
}

impl Hash for Objective {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.time.hash(state);
        self.cost.hash(state);
    }
}

impl Objective {
    pub fn new(time: u64, cost: u64) -> Objective {
        Objective { time, cost }
    }
}

impl Label<NodeId> {
    pub fn new_along(
        &self,
        edge: &EdgeReference<WeightsTuple>,
        disable_path: bool,
        update_label_func: &Box<dyn Fn(&Label<usize>, &WeightsTuple) -> Objective>,
    ) -> Label<NodeId> {
        let weight = edge.weight();
        let objectives = update_label_func(self, weight);
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
            objective: objectives,
            path,
            node_id: target_node_id,
            auxiliary,
        }
    }

    // returns true if the label weakly dominates the other label
    // this is the case if it either strictly dominates the other label
    // or if it is equal to the other label
    fn weakly_dominates(&self, other: &Label<NodeId>) -> bool {
        self.objective <= other.objective
    }
}

impl<T> Ord for Label<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // lexicographical order, but the smaller the better
        // we need a "min-heap" as queue
        match self.objective.cmp(&other.objective) {
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
        match self.objective.cmp(&other.objective) {
            Ordering::Less => Some(Ordering::Greater),
            Ordering::Greater => Some(Ordering::Less),
            Ordering::Equal => Some(Ordering::Equal),
        }
    }
}

impl<T> PartialEq for Label<T> {
    fn eq(&self, other: &Self) -> bool {
        self.objective == other.objective
    }
}

impl<T> Eq for Label<T> {}

impl<T> Hash for Label<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.objective.hash(state);
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
