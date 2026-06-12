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
///
/// `edge_tag: M` describes the edge that led *into* `node_id`. Callers with no
/// per-edge metadata leave `M = ()`; richer callers (e.g. MCR's per-hop
/// transport mode) supply a custom `M` type and produce values from each
/// `update_label_func` invocation. The start node's `edge_tag` is `M::default()`
/// (no incoming edge).
#[derive(Debug)]
pub struct PathNode<T, M = ()> {
    pub node_id: T,
    pub edge_tag: M,
    pub parent: Option<Rc<PathNode<T, M>>>,
}

/// A label's traversal history. `None` ⇔ "no path recorded" (used when
/// `disable_paths` is set). Walking `parent` links from the head yields nodes
/// in reverse (terminal → start); `path_to_vec` materialises them start → end.
pub type Path<T, M = ()> = Option<Rc<PathNode<T, M>>>;

/// Build a Path from a Vec ordered start→end. Each node gets `M::default()` as
/// its edge tag — used by callers that don't track per-edge metadata (tests,
/// CSV `read_bags`).
pub fn path_from_vec<T, M: Default>(v: Vec<T>) -> Path<T, M> {
    let mut cur: Path<T, M> = None;
    for node_id in v {
        cur = Some(Rc::new(PathNode {
            node_id,
            edge_tag: M::default(),
            parent: cur,
        }));
    }
    cur
}

/// Build a Path from a Vec of `(node, edge_tag)` pairs ordered start→end. The
/// first entry's tag should be `M::default()` (no incoming edge into the start).
pub fn path_from_vec_tagged<T, M>(v: Vec<(T, M)>) -> Path<T, M> {
    let mut cur: Path<T, M> = None;
    for (node_id, edge_tag) in v {
        cur = Some(Rc::new(PathNode {
            node_id,
            edge_tag,
            parent: cur,
        }));
    }
    cur
}

/// Materialise a Path into a Vec ordered start→end. Walks the linked list and
/// reverses. O(path_len) — used at serialisation time only.
pub fn path_to_vec<T: Clone, M>(path: &Path<T, M>) -> Vec<T> {
    let mut out = Vec::new();
    let mut cur = path.as_deref();
    while let Some(p) = cur {
        out.push(p.node_id.clone());
        cur = p.parent.as_deref();
    }
    out.reverse();
    out
}

/// Like `path_to_vec`, but also yields each node's `edge_tag`. The first
/// element's tag is the start node's (typically `M::default()`); subsequent
/// elements' tags describe the edge that brought the label *into* that node.
pub fn path_to_vec_tagged<T: Clone, M: Clone>(path: &Path<T, M>) -> Vec<(T, M)> {
    let mut out = Vec::new();
    let mut cur = path.as_deref();
    while let Some(p) = cur {
        out.push((p.node_id.clone(), p.edge_tag.clone()));
        cur = p.parent.as_deref();
    }
    out.reverse();
    out
}

/// Extend a path by one node, sharing the prior suffix via `Rc::clone`. The
/// supplied `edge_tag` describes the edge from the parent into `node_id`.
pub fn path_extend<T, M>(parent: &Path<T, M>, node_id: T, edge_tag: M) -> Path<T, M> {
    Some(Rc::new(PathNode {
        node_id,
        edge_tag,
        parent: parent.clone(),
    }))
}

/// Tie-break ordering on the auxiliary field, used by `Bag` to suppress
/// dominance between equal-objective labels that differ on a non-objective
/// dimension the domain treats as load-bearing.
///
/// The default implementation returns `true` for every pair, meaning the
/// auxiliary is *not* tie-broken — equal-objective labels dominate each
/// other as they do under standard 2-D Pareto. Callers with a richer aux
/// (e.g. MCR's `Auxiliary::pt_shift_secs`) override this method to encode
/// the rule "an aux state strictly less flexible than another's must not
/// dominate it at equal time and cost." Concretely a label `L1` weakly
/// dominates `L2` iff
///     `L1.time ≤ L2.time ∧ L1.cost ≤ L2.cost ∧ L1.aux.at_least_as_flexible(&L2.aux)`.
///
/// Implementations must be reflexive (`a.at_least_as_flexible(&a) == true`).
/// They need not be antisymmetric — two aux values can each be at-least-as-
/// flexible-as the other, in which case the bag keeps both at equal
/// objectives (the standard Pareto-equivalent case).
pub trait AuxFlex {
    fn at_least_as_flexible(&self, _other: &Self) -> bool {
        true
    }
}

impl AuxFlex for () {}

/// A Pareto label representing a route from the origin to `node_id`.
///
/// `objective` is summed along the path and used for dominance comparisons.
/// `auxiliary` is an opaque value of type `A` — the MLC algorithm only stores
/// and forwards it (via the `update_label_func` closure); callers attach
/// whatever per-label metadata their domain requires.
///
/// `M` is the per-edge tag stored on each `PathNode`; defaults to `()` for
/// callers that don't track per-hop metadata.
#[derive(Debug, Clone)]
pub struct Label<T, A, M = ()> {
    pub objective: Objective,
    pub auxiliary: A,
    pub path: Path<T, M>,
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

impl<A, M> Label<NodeId, A, M> {
    pub fn new_along(
        &self,
        edge: &EdgeReference<WeightsTuple>,
        disable_path: bool,
        update_label_func: impl Fn(&Label<usize, A, M>, &WeightsTuple) -> (Objective, A, M),
    ) -> Label<NodeId, A, M> {
        let weight = edge.weight();
        let (objective, auxiliary, edge_tag) = update_label_func(self, weight);

        let target_node_id = edge.target().index();
        let path = if disable_path {
            None
        } else {
            path_extend(&self.path, target_node_id, edge_tag)
        };
        Label {
            objective,
            path,
            node_id: target_node_id,
            auxiliary,
        }
    }

}

impl<A: AuxFlex, M> Label<NodeId, A, M> {
    /// Standard weak 2-D Pareto dominance on `(time, cost)`, augmented with
    /// the `AuxFlex` tie-break: `self` does *not* dominate `other` if its
    /// aux is strictly less flexible than `other`'s. With the default
    /// `AuxFlex` impl (always returns `true`) this reduces to the plain
    /// 2-D rule.
    fn weakly_dominates(&self, other: &Label<NodeId, A, M>) -> bool {
        self.objective.time <= other.objective.time
            && self.objective.cost <= other.objective.cost
            && self.auxiliary.at_least_as_flexible(&other.auxiliary)
    }
}

impl<T, A, M> Ord for Label<T, A, M> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.objective.cmp(&other.objective)
    }
}

impl<T, A, M> PartialOrd for Label<T, A, M> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, A, M> PartialEq for Label<T, A, M> {
    fn eq(&self, other: &Self) -> bool {
        self.objective == other.objective
    }
}

impl<T, A, M> Eq for Label<T, A, M> {}

impl<T, A, M> Hash for Label<T, A, M> {
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
pub struct Bag<T, A, M = ()> {
    pub labels: Vec<Label<T, A, M>>,
}

impl<T, A, M> PartialEq for Bag<T, A, M> {
    fn eq(&self, other: &Self) -> bool {
        self.labels == other.labels
    }
}

impl<T, A, M> Eq for Bag<T, A, M> {}

impl<A, M> Bag<NodeId, A, M> {
    pub fn new_start_bag(start_label: Label<NodeId, A, M>) -> Bag<NodeId, A, M> {
        Bag {
            labels: vec![start_label],
        }
    }

    pub fn new_empty() -> Bag<NodeId, A, M> {
        Bag { labels: Vec::new() }
    }
}

impl<A: AuxFlex, M> Bag<NodeId, A, M> {
    pub fn add_if_necessary(&mut self, label: Label<NodeId, A, M>) -> bool {
        if self.content_dominates(&label) {
            return false;
        }
        self.remove_dominated_by(&label);
        self.labels.push(label);
        true
    }

    /// Time-only variant of `add_if_necessary`. Dominance ignores `cost` and the
    /// `AuxFlex` tie-break: `self` dominates `other` iff `self.time <= other.time`.
    /// Reject the incoming label if any existing label has `time <= new.time`,
    /// and evict any existing label whose `time >= new.time`. At equal times
    /// the first arrival wins. Result: each node holds at most one
    /// time-minimum label (multiple coexist only if they tie on time *and*
    /// the previous occupant was kept).
    pub fn add_if_necessary_time_only(&mut self, label: Label<NodeId, A, M>) -> bool {
        if self.content_dominates_time_only(&label) {
            return false;
        }
        self.remove_dominated_by_time_only(&label);
        self.labels.push(label);
        true
    }

    pub fn content_dominates(&self, label: &Label<NodeId, A, M>) -> bool {
        self.labels.iter().any(|l| l.weakly_dominates(label))
    }

    fn content_dominates_time_only(&self, label: &Label<NodeId, A, M>) -> bool {
        self.labels
            .iter()
            .any(|l| l.objective.time <= label.objective.time)
    }

    fn remove_dominated_by(&mut self, label: &Label<NodeId, A, M>) {
        self.labels.retain(|l| !label.weakly_dominates(l));
    }

    fn remove_dominated_by_time_only(&mut self, label: &Label<NodeId, A, M>) {
        self.labels
            .retain(|l| label.objective.time > l.objective.time);
    }
}
