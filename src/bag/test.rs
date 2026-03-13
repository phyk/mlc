#[cfg(test)]
mod tests {
    use super::super::*;
    use std::collections::BinaryHeap;
    use petgraph::{Directed, Graph};

    #[test]
    fn test_label_ordering() {
        let label_small = Label {
            objectives: vec![1, 10],
            auxiliary: vec![],
            path: vec![0],
            node_id: 0,
        };
        let label_large = Label {
            objectives: vec![5, 2],
            auxiliary: vec![],
            path: vec![0],
            node_id: 0,
        };

        let mut heap = BinaryHeap::new();
        heap.push(label_large);
        heap.push(label_small);

        // min-heap: label with smaller objectives[0] pops first
        let first = heap.pop().unwrap();
        assert_eq!(first.objectives[0], 1);
        let second = heap.pop().unwrap();
        assert_eq!(second.objectives[0], 5);
    }

    #[test]
    fn test_label_equality_ignores_path() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let label_a = Label {
            objectives: vec![5, 3],
            auxiliary: vec![],
            path: vec![0, 1, 2],
            node_id: 2,
        };
        let label_b = Label {
            objectives: vec![5, 3],
            auxiliary: vec![],
            path: vec![9, 8],
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
        g.add_edge(
            n0,
            n1,
            WeightsTuple {
                objectives: vec![10, 1],
                auxiliary: vec![],
            },
        );

        let start_label = Label {
            objectives: vec![5, 0],
            auxiliary: vec![],
            path: vec![0],
            node_id: 0,
        };

        // with disable_path=false
        let edge = g.edges(n0).next().unwrap();
        let new_label = start_label.new_along(&edge, false);
        assert_eq!(new_label.objectives, vec![15, 1]);
        assert_eq!(new_label.path, vec![0, 1]);
        assert_eq!(new_label.node_id, 1);

        // with disable_path=true
        let edge = g.edges(n0).next().unwrap();
        let new_label_no_path = start_label.new_along(&edge, true);
        assert_eq!(new_label_no_path.objectives, vec![15, 1]);
        assert_eq!(new_label_no_path.path, vec![]);
    }

    #[test]
    fn test_weakly_dominates() {
        let label1 = Label {
            objectives: vec![1, 2, 3],
            auxiliary: vec![],
            path: vec![0, 1, 2],
            node_id: 2,
        };
        let label2 = Label {
            objectives: vec![1, 2, 3],
            auxiliary: vec![],
            path: vec![0, 1, 2],
            node_id: 2,
        };
        let label3 = Label {
            objectives: vec![2, 3, 4],
            auxiliary: vec![],
            path: vec![0, 1, 2],
            node_id: 2,
        };
        let label4 = Label {
            objectives: vec![1, 2, 4],
            auxiliary: vec![],
            path: vec![0, 1, 2],
            node_id: 2,
        };
        let label_bug_1 = Label {
            objectives: vec![1852375, 0],
            auxiliary: vec![],
            path: vec![0],
            node_id: 1,
        };
        let label_bug_2 = Label {
            objectives: vec![2003938, 0],
            auxiliary: vec![],
            path: vec![0],
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
            objectives: vec![1, 2, 3],
            auxiliary: vec![],
            path: vec![0, 1, 2],
            node_id: 2,
        };
        let label2 = Label {
            objectives: vec![2, 3, 4],
            auxiliary: vec![],
            path: vec![0, 1, 2],
            node_id: 2,
        };

        assert!(bag.add_if_necessary(label1.clone()));
        assert_eq!(bag.labels.len(), 1);

        assert!(!bag.add_if_necessary(label2.clone()));
        assert_eq!(bag.labels.len(), 1);

        let label3 = Label {
            objectives: vec![0, 1, 6],
            auxiliary: vec![],
            path: vec![0, 1, 2],
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
            objectives: vec![1, 2, 5],
            auxiliary: vec![],
            path: vec![0, 1, 2],
            node_id: 2,
        };
        let label2 = Label {
            objectives: vec![2, 3, 4],
            auxiliary: vec![],
            path: vec![0, 1, 2],
            node_id: 2,
        };
        let label3 = Label {
            objectives: vec![0, 0, 0],
            auxiliary: vec![],
            path: vec![0, 1, 2],
            node_id: 2,
        };

        bag.labels.insert(label1);
        bag.labels.insert(label2);
        assert_eq!(bag.labels.len(), 2);

        bag.remove_dominated_by(&label3);
        assert_eq!(bag.labels.len(), 0);
    }
}
