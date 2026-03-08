#[cfg(test)]
mod tests {
    use crate::mlc;
    use crate::read;
    use std::collections::HashMap;
    use crate::bag::{Bag, Label};
    use petgraph::graph::NodeIndex;
    use petgraph::{Directed, Graph};
    use crate::bag::WeightsTuple;

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
