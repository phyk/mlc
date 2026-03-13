#[cfg(test)]
mod tests {
    use crate::mlc;
    use crate::read;
    use std::collections::HashMap;
    use crate::bag::{Bag, Label, WeightsTuple};
    use petgraph::graph::NodeIndex;
    use petgraph::{Directed, Graph};

    #[test]
    fn test_run_mlc() {
        let g = read::read_graph_with_int_ids("testdata/edges.csv").unwrap();

        let mut mlc = mlc::MLC::new(&g).unwrap();
        mlc.set_start_node(0);
        let bags = mlc.run().unwrap();
        let expected_result = mlc::read_bags("testdata/results.csv").unwrap();
        assert!(bags == &expected_result);
    }

    fn read_node_categories(path: &str) -> HashMap<usize, Vec<u8>> {
        let mut map = HashMap::new();
        for line in std::fs::read_to_string(path).unwrap().lines().skip(1) {
            let mut parts = line.splitn(2, ',');
            let node_id: usize = parts.next().unwrap().parse().unwrap();
            let cats: Vec<u8> = parts.next().unwrap()
                .split(';').map(|s| s.parse().unwrap()).collect();
            map.insert(node_id, cats);
        }
        map
    }

    fn build_brugge_walking_graph() -> Graph<Vec<u8>, WeightsTuple, Directed> {
        let mut g = read::read_graph_with_int_ids(
            "testdata/brugge_walking_edges.csv"
        ).unwrap();
        let categories = read_node_categories(
            "testdata/brugge_walking_node_categories.csv"
        );
        for node in g.node_indices() {
            if let Some(cats) = categories.get(&node.index()) {
                *g.node_weight_mut(node).unwrap() = cats.clone();
            }
        }
        g
    }

    fn path_is_valid(label: &Label<usize>, g: &Graph<Vec<u8>, WeightsTuple, Directed>) -> bool {
        // path contains the full route including the destination node,
        // so consecutive pairs are the edges that must exist in the graph.
        label.path.windows(2).all(|w| {
            g.contains_edge(NodeIndex::new(w[0]), NodeIndex::new(w[1]))
        })
    }

    #[test]
    fn test_set_start_node_with_time() {
        let g = read::read_graph_with_int_ids("testdata/edges.csv").unwrap();
        let mut m = mlc::MLC::new(&g).unwrap();
        m.set_start_node_with_time(0, 100);
        let bags = m.run().unwrap();

        // Start node bag must contain a label with time == 100
        let start_bag = bags.get(&0).expect("start node bag missing");
        assert!(
            start_bag.labels.iter().any(|l| l.objectives[0] == 100),
            "start bag must have a label with objectives[0] == 100"
        );

        // Every label at node 1 must have time >= 100 (offset propagates)
        let bag1 = bags.get(&1).expect("node 1 bag missing");
        for label in &bag1.labels {
            assert!(
                label.objectives[0] >= 100,
                "node 1 label has objectives[0] = {} < 100",
                label.objectives[0]
            );
        }
    }

    #[test]
    fn test_set_external_start_node() {
        let g = read::read_graph_with_int_ids("testdata/edges.csv").unwrap();
        let mut m = mlc::MLC::new(&g).unwrap();
        let node_map = bimap::BiMap::from_iter(vec![
            ("0".to_string(), 0usize),
            ("1".to_string(), 1usize),
            ("2".to_string(), 2usize),
            ("3".to_string(), 3usize),
            ("4".to_string(), 4usize),
        ]);
        m.set_node_map(node_map);
        m.set_external_start_node("0".to_string()).unwrap();
        let bags = m.run().unwrap();
        assert!(bags.contains_key(&0), "result bags must contain internal node 0");
    }

    #[test]
    fn test_set_external_start_node_error_no_map() {
        let g = read::read_graph_with_int_ids("testdata/edges.csv").unwrap();
        let mut m = mlc::MLC::new(&g).unwrap();
        let result = m.set_external_start_node("0".to_string());
        assert!(
            matches!(result, Err(mlc::MLCError::NodeMapNotSet)),
            "expected NodeMapNotSet error"
        );
    }

    #[test]
    fn test_set_external_start_node_error_not_found() {
        let g = read::read_graph_with_int_ids("testdata/edges.csv").unwrap();
        let mut m = mlc::MLC::new(&g).unwrap();
        let node_map = bimap::BiMap::from_iter(vec![("0".to_string(), 0usize)]);
        m.set_node_map(node_map);
        let result = m.set_external_start_node("99999".to_string());
        assert!(
            matches!(result, Err(mlc::MLCError::StartNodeNotFound(_))),
            "expected StartNodeNotFound error"
        );
    }

    #[test]
    fn test_set_disable_paths() {
        let g = read::read_graph_with_int_ids("testdata/edges.csv").unwrap();
        let mut m = mlc::MLC::new(&g).unwrap();
        m.set_disable_paths(true);
        m.set_start_node(0);
        let bags = m.run().unwrap();
        for bag in bags.values() {
            for label in &bag.labels {
                assert!(
                    label.path.is_empty(),
                    "expected empty path when disable_paths=true, got {:?}",
                    label.path
                );
            }
        }
    }

    #[test]
    fn test_write_bags_read_bags_roundtrip() {
        let g = read::read_graph_with_int_ids("testdata/edges.csv").unwrap();
        let mut m = mlc::MLC::new(&g).unwrap();
        m.set_start_node(0);
        let bags = m.run().unwrap().clone();

        let tmp_path = "/tmp/test_mlc_roundtrip.csv";
        mlc::write_bags(&bags, tmp_path).unwrap();
        let read_back = mlc::read_bags(tmp_path).unwrap();

        assert_eq!(bags, read_back);
    }

    #[test]
    fn test_mlc_new_empty_graph_error() {
        use petgraph::{Directed, Graph};
        let mut g = Graph::<Vec<u8>, WeightsTuple, Directed>::new();
        g.add_node(vec![]);
        g.add_node(vec![]);
        // no edges added
        assert!(mlc::MLC::new(&g).is_err());
    }

    #[test]
    fn test_set_update_label_func() {
        let g = read::read_graph_with_int_ids("testdata/edges.csv").unwrap();
        let mut m = mlc::MLC::new(&g).unwrap();
        m.set_update_label_func(|_old, new_label, _accuracy| {
            let mut updated = new_label.clone();
            updated.objectives[0] += 1000;
            updated
        });
        m.set_start_node(0);
        let bags = m.run().unwrap();

        // node 4 is reachable via 4 edges; each step adds 1000 to objectives[0]
        let bag4 = bags.get(&4).expect("node 4 bag must exist");
        for label in &bag4.labels {
            assert!(
                label.objectives[0] >= 4000,
                "node 4 label has objectives[0] = {} < 4000",
                label.objectives[0]
            );
        }
    }

    #[test]
    fn test_enable_limit_requires_categories() {
        // edges.csv nodes have no categories → limits not initialised → panic
        let result = std::panic::catch_unwind(|| {
            let g = read::read_graph_with_int_ids("testdata/edges.csv").unwrap();
            let mut m = mlc::MLC::new(&g).unwrap();
            m.set_enable_limit(true);
            m.set_start_node(0);
            let _ = m.run();
        });
        assert!(result.is_err(), "expected a panic when limits are not initialised");
    }

    #[test]
    fn test_enable_limit_with_categorised_graph() {
        let g = build_brugge_walking_graph();
        let mut m = mlc::MLC::new(&g).unwrap();
        m.set_enable_limit(true);
        m.set_start_node(0);
        let bags = m.run().unwrap();

        assert!(!bags.is_empty(), "bags must be non-empty");
        assert!(bags.contains_key(&0), "start node must be in bags");

        // Validate paths for the first 50 bags
        let mut checked = 0;
        for bag in bags.values().take(50) {
            for label in &bag.labels {
                assert!(
                    path_is_valid(label, &g),
                    "invalid path {:?} → {}",
                    label.path,
                    label.node_id
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no labels to validate");
    }

    #[test]
    fn test_brugge_walking_with_limits() {
        let g = build_brugge_walking_graph();

        // ── 1. Pre-initialised bags (set_bags) ────────────────────────────────
        // Simulate two origins that were already reached (e.g. from a prior pass)
        let mut start_bags: mlc::Bags<usize> = HashMap::new();
        for &start_node in &[0_usize, 100_usize] {
            let label = Label {
                objectives: vec![0, 0],
                auxiliary:  vec![],
                path:        vec![start_node],
                node_id:     start_node,
            };
            start_bags.insert(start_node, Bag::new_start_bag(label));
        }

        // ── 2. Run WITH limits ────────────────────────────────────────────────
        let bags_limited = {
            let mut m = mlc::MLC::new(&g).unwrap();
            m.set_enable_limit(true);
            m.set_bags(start_bags.clone());
            m.run().unwrap().clone()
        };

        // ── 3. Run WITHOUT limits (baseline) ─────────────────────────────────
        let bags_unlimited = {
            let mut m = mlc::MLC::new(&g).unwrap();
            m.set_bags(start_bags);
            m.run().unwrap().clone()
        };

        // ── 4. Basic non-empty assertions ────────────────────────────────────
        assert!(!bags_limited.is_empty());
        assert!(bags_limited.contains_key(&0));
        assert!(bags_limited.contains_key(&100));

        // ── 5. Limit system reduces or preserves label count ─────────────────
        let mut limit_reduced_at_least_one = false;
        for (node_id, bag) in &bags_limited {
            let unlimited_count = bags_unlimited
                .get(node_id).map(|b| b.labels.len()).unwrap_or(0);
            assert!(bag.labels.len() <= unlimited_count,
                "node {}: limited={} > unlimited={}", node_id, bag.labels.len(), unlimited_count);
            if bag.labels.len() < unlimited_count {
                limit_reduced_at_least_one = true;
            }
        }
        assert!(limit_reduced_at_least_one,
            "limit system should prune at least one label in the Brugge graph");

        // ── 6. Path validity ─────────────────────────────────────────────────
        let mut checked = 0;
        for bag in bags_limited.values().take(50) {
            for label in &bag.labels {
                assert!(path_is_valid(label, &g),
                    "invalid path {:?} → {}", label.path, label.node_id);
                assert!(
                    label.path.first().map(|&n| n == 0 || n == 100).unwrap_or(true),
                    "path starts at unexpected node: {:?}", label.path
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no labels to check paths for");

        // ── 7. POI category nodes are reachable ──────────────────────────────
        let poi_in_results = bags_limited.keys().any(|nid| {
            g.node_weight(NodeIndex::new(*nid))
                .map(|w| !w.is_empty()).unwrap_or(false)
        });
        assert!(poi_in_results, "no POI category nodes found in result bags");
    }
}
