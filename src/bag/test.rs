#[cfg(test)]
mod tests {
    use super::super::*;
    use petgraph::{Directed, Graph};
    use std::collections::BinaryHeap;

    #[test]
    fn test_label_ordering() {
        let label_small = Label {
            objective: Objective::new(1, 10),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0]),
            node_id: 0,
        };
        let label_large = Label {
            objective: Objective::new(5, 2),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0]),
            node_id: 0,
        };

        let mut heap = BinaryHeap::new();
        heap.push(label_large);
        heap.push(label_small);

        // min-heap: label that was added first stays first
        // this behavior occurs because the labels do not dominate each other
        let first = heap.pop().unwrap();
        assert_eq!(first.objective.time, 5);
        let second = heap.pop().unwrap();
        assert_eq!(second.objective.time, 1);
    }

    #[test]
    fn test_label_equality_ignores_path() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let label_a = Label {
            objective: Objective::new(5, 3),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0, 1, 2]),
            node_id: 2,
        };
        let label_b = Label {
            objective: Objective::new(5, 3),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![9, 8]),
            node_id: 99,
        };

        assert_eq!(label_a, label_b);

        let hash_a = {
            let mut h = DefaultHasher::new();
            label_a.hash(&mut h);
            h.finish()
        };
        let hash_b = {
            let mut h = DefaultHasher::new();
            label_b.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn test_label_new_along() {
        let mut g = Graph::<Vec<u8>, WeightsTuple, Directed>::new();
        let n0 = g.add_node(vec![]);
        let n1 = g.add_node(vec![]);
        g.add_edge(n0, n1, WeightsTuple { distance_mm: 50 });

        let start_label = Label {
            objective: Objective::new(5, 0),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0]),
            node_id: 0,
        };

        let closure = |label: &Label<usize>, weight: &WeightsTuple| {
            (
                Objective::new(
                    label.objective.time + weight.distance_mm / 5,
                    label.objective.cost + 1,
                ),
                label.auxiliary.clone(),
            )
        };
        // with disable_path=false
        let edge = g.edges(n0).next().unwrap();
        let new_label = start_label.new_along(&edge, false, closure);
        assert_eq!(new_label.objective, Objective::new(15, 1));
        assert_eq!(path_to_vec(&new_label.path), vec![0, 1]);
        assert_eq!(new_label.node_id, 1);

        // with disable_path=true
        let edge = g.edges(n0).next().unwrap();
        let new_label_no_path = start_label.new_along(&edge, true, closure);
        assert_eq!(new_label_no_path.objective, Objective::new(15, 1));
        assert!(new_label_no_path.path.is_none());
    }

    #[test]
    fn test_weakly_dominates() {
        let label1 = Label {
            objective: Objective::new(1, 2),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0, 1, 2]),
            node_id: 2,
        };
        let label2 = Label {
            objective: Objective::new(1, 2),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0, 1, 2]),
            node_id: 2,
        };
        let label3 = Label {
            objective: Objective::new(2, 3),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0, 1, 2]),
            node_id: 2,
        };
        let label4 = Label {
            objective: Objective::new(1, 3),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0, 1, 2]),
            node_id: 2,
        };
        let label_bug_1 = Label {
            objective: Objective::new(350274, 0),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0]),
            node_id: 1,
        };
        let label_bug_2 = Label {
            objective: Objective::new(357024742, 0),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0]),
            node_id: 1,
        };

        assert!(label1.weakly_dominates(&label2));
        assert!(label2.weakly_dominates(&label1));
        assert!(label1.weakly_dominates(&label3));
        assert!(!label3.weakly_dominates(&label1));
        assert!(label1.weakly_dominates(&label4));
        assert!(!label4.weakly_dominates(&label1));
        assert!(label_bug_1.weakly_dominates(&label_bug_2));
    }

    #[test]
    fn test_bag_add_if_necessary() {
        let mut bag = Bag::new_empty();
        let label1 = Label {
            objective: Objective::new(1, 2),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0, 1, 2]),
            node_id: 2,
        };
        let label2 = Label {
            objective: Objective::new(2, 3),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0, 1, 2]),
            node_id: 2,
        };

        assert!(bag.add_if_necessary(label1.clone()));
        assert_eq!(bag.labels.len(), 1);

        assert!(!bag.add_if_necessary(label2.clone()));
        assert_eq!(bag.labels.len(), 1);

        let label3 = Label {
            objective: Objective::new(2, 1),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0, 1, 2]),
            node_id: 2,
        };
        assert!(bag.add_if_necessary(label3.clone()));
        assert_eq!(bag.labels.len(), 2);

        assert!(bag.content_dominates(&label1));
        assert!(bag.content_dominates(&label2)); // weakly
        assert!(bag.content_dominates(&label3)); // weakly
    }

    #[test]
    fn test_bag_remove_dominated_by() {
        let mut bag = Bag::new_empty();
        let label1 = Label {
            objective: Objective::new(1, 5),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0, 1, 2]),
            node_id: 2,
        };
        let label2 = Label {
            objective: Objective::new(4, 2),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0, 1, 2]),
            node_id: 2,
        };
        let label3 = Label {
            objective: Objective::new(0, 0),
            auxiliary: Auxiliary::new_empty(),
            path: path_from_vec(vec![0, 1, 2]),
            node_id: 2,
        };

        bag.labels.push(label1);
        bag.labels.push(label2);
        assert_eq!(bag.labels.len(), 2);

        bag.remove_dominated_by(&label3);
        assert_eq!(bag.labels.len(), 0);
    }
}
