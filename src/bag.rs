mod test;

use petgraph::graph::EdgeReference;
use petgraph::visit::EdgeRef;
use std::cmp::Ordering;
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
/// `objective` is summed along the path and used for dominance comparisons.
/// `auxiliary` is an opaque value of type `A` — the MLC algorithm only stores
/// and forwards it (via the `update_label_func` closure); callers attach
/// whatever per-label metadata their domain requires.
#[derive(Debug, Clone)]
pub struct Label<T, A> {
    pub objective: Objective,
    pub auxiliary: A,
    pub path: Path<T>,
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

impl<A> Label<NodeId, A> {
    pub fn new_along(
        &self,
        edge: &EdgeReference<WeightsTuple>,
        disable_path: bool,
        update_label_func: impl Fn(&Label<usize, A>, &WeightsTuple) -> (Objective, A),
    ) -> Label<NodeId, A> {
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
    fn weakly_dominates(&self, other: &Label<NodeId, A>) -> bool {
        self.objective.time <= other.objective.time
            && self.objective.cost <= other.objective.cost
    }
}

impl<T, A> Ord for Label<T, A> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.objective.cmp(&other.objective)
    }
}

impl<T, A> PartialOrd for Label<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, A> PartialEq for Label<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.objective == other.objective
    }
}

impl<T, A> Eq for Label<T, A> {}

impl<T, A> Hash for Label<T, A> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.objective.hash(state);
    }
}

/// Holds the Pareto-optimal (non-dominated) set of labels for a single node.
///
/// `labels` is a plain `Vec` (not a `HashSet`): Pareto fronts are tiny —
/// usually a handful of labels — so contiguous-memory linear scans beat
/// SipHash + bucket chasing. `add_if_necessary` guarantees no two labels in
/// the Vec are weakly comparable, so duplicates by objective never appear.
#[derive(Debug, Clone)]
pub struct Bag<T, A> {
    pub labels: Vec<Label<T, A>>,
}

impl<T, A> PartialEq for Bag<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.labels == other.labels
    }
}

impl<T, A> Eq for Bag<T, A> {}

impl<A> Bag<NodeId, A> {
    pub fn new_start_bag(start_label: Label<NodeId, A>) -> Bag<NodeId, A> {
        Bag {
            labels: vec![start_label],
        }
    }

    pub fn new_empty() -> Bag<NodeId, A> {
        Bag { labels: Vec::new() }
    }

    pub fn add_if_necessary(&mut self, label: Label<NodeId, A>) -> bool {
        if self.content_dominates(&label) {
            return false;
        }
        self.remove_dominated_by(&label);
        self.labels.push(label);
        true
    }

    pub fn content_dominates(&self, label: &Label<NodeId, A>) -> bool {
        self.labels.iter().any(|l| l.weakly_dominates(label))
    }

    fn remove_dominated_by(&mut self, label: &Label<NodeId, A>) {
        self.labels.retain(|l| !label.weakly_dominates(l));
    }
}
