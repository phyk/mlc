mod test;

use petgraph::graph::EdgeReference;
use petgraph::visit::EdgeRef;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::Hash;
use std::ops::Add;
use std::rc::Rc;

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
    pub distance_mm: u64,
}

impl WeightsTuple {
    fn new(distance_mm: u64) -> WeightsTuple {
        WeightsTuple { distance_mm }
    }
}

/// A single hop in a label's traversal history. Each node points back at the
/// previous node via `parent`, so labels sharing a path prefix share the same
/// `PathNode` allocations — extending a path is an `Rc::clone` instead of a
/// `Vec` copy + push.
#[derive(Debug)]
pub struct PathNode<T> {
    pub node_id: T,
    pub parent: Option<Rc<PathNode<T>>>,
}

/// A label's traversal history. `None` ⇔ "no path recorded" (used when
/// `disable_paths` is set). Walking `parent` links from the head yields nodes
/// in reverse (terminal → start); `path_to_vec` materialises them start → end.
pub type Path<T> = Option<Rc<PathNode<T>>>;

/// Build a Path from a Vec ordered start→end. The returned head's `node_id`
/// corresponds to `v.last()`. Used for the few callers (debug `read_bags`,
/// tests) that still produce paths as Vecs.
pub fn path_from_vec<T>(v: Vec<T>) -> Path<T> {
    let mut cur: Path<T> = None;
    for node_id in v {
        cur = Some(Rc::new(PathNode {
            node_id,
            parent: cur,
        }));
    }
    cur
}

/// Materialise a Path into a Vec ordered start→end. Walks the linked list and
/// reverses. O(path_len) — used at serialisation time only.
pub fn path_to_vec<T: Clone>(path: &Path<T>) -> Vec<T> {
    let mut out = Vec::new();
    let mut cur = path.as_deref();
    while let Some(p) = cur {
        out.push(p.node_id.clone());
        cur = p.parent.as_deref();
    }
    out.reverse();
    out
}

/// Extend a path by one node, sharing the prior suffix via `Rc::clone`.
pub fn path_extend<T>(parent: &Path<T>, node_id: T) -> Path<T> {
    Some(Rc::new(PathNode {
        node_id,
        parent: parent.clone(),
    }))
}

/// A Pareto label representing a route from the origin to `node_id`.
///
/// `objectives` and `auxiliary` are accumulated sums of edge weights along the path.
/// Only `objectives` are used for dominance comparisons; `auxiliary` values are tracked
/// but ignored when determining whether one label dominates another.
#[derive(Debug, Clone)]
pub struct Label<T> {
    pub objective: Objective,
    pub auxiliary: Auxiliary,
    pub path: Path<T>,
    pub node_id: T,
}

#[derive(Debug, Clone)]
pub struct Objective {
    pub time: u64,
    pub cost: u64,
}

/// The transport mode used on the most recent edge of a label's path.
/// Tracked in `Auxiliary` so mode-transition costs (e.g., car switch_time)
/// can fire only on actual transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LastRoutedMode {
    Walking,
    Cycling,
    Car,
    SharedBike,
    SharedScooter,
}

#[derive(Debug, Clone)]
pub struct Auxiliary {
    pub dist_walk: u64,
    pub dist_bike: u64,
    pub dist_car: u64,
    pub last_routed_mode: Option<LastRoutedMode>,
}

impl Add for Auxiliary {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Auxiliary {
            dist_bike: self.dist_bike + rhs.dist_bike,
            dist_car: self.dist_car + rhs.dist_car,
            dist_walk: self.dist_walk + rhs.dist_walk,
            last_routed_mode: rhs.last_routed_mode.or(self.last_routed_mode),
        }
    }
}

impl Auxiliary {
    pub fn new(
        dist_walk: u64,
        dist_bike: u64,
        dist_car: u64,
        last_routed_mode: Option<LastRoutedMode>,
    ) -> Auxiliary {
        Auxiliary {
            dist_walk,
            dist_bike,
            dist_car,
            last_routed_mode,
        }
    }

    pub fn new_empty() -> Auxiliary {
        Auxiliary {
            dist_walk: 0,
            dist_bike: 0,
            dist_car: 0,
            last_routed_mode: None,
        }
    }
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
        self.time.cmp(&other.time).then(self.cost.cmp(&other.cost))
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
        update_label_func: impl Fn(&Label<usize>, &WeightsTuple) -> (Objective, Auxiliary),
    ) -> Label<NodeId> {
        let weight = edge.weight();
        let (objective, auxiliary) = update_label_func(self, weight);

        let target_node_id = edge.target().index();
        let path = if disable_path {
            None
        } else {
            path_extend(&self.path, target_node_id)
        };
        Label {
            objective,
            path,
            node_id: target_node_id,
            auxiliary,
        }
    }

    // returns true if the label weakly dominates the other label
    // this is the case if it either strictly dominates the other label
    // or if it is equal to the other label
    fn weakly_dominates(&self, other: &Label<NodeId>) -> bool {
        self.objective.time <= other.objective.time
            && self.objective.cost <= other.objective.cost
    }
}

impl<T> Ord for Label<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
       self.objective.cmp(&other.objective)
    }
}

impl<T> PartialOrd for Label<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
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
