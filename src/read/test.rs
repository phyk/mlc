#[cfg(test)]
mod tests {

    use bimap::BiMap;

    use crate::read;

    #[test]
    fn test_read_graph() {
        let (_, node_map) =
            read::read_graph_and_reset_ids("testdata/edges_high_index.csv").unwrap();
        let expected_node_map = BiMap::from_iter(vec![
            ("10".to_string(), 0),
            ("11".to_string(), 1),
            ("20".to_string(), 2),
            ("22".to_string(), 3),
        ]);
        assert_eq!(node_map, expected_node_map);
    }

    #[test]
    fn test_read_graph_with_int_ids_structure() {
        let g = read::read_graph_with_int_ids("testdata/edges.csv").unwrap();
        // edges.csv: 4 directed pairs, 2 parallel edges each = 8 edges total
        assert_eq!(g.edge_count(), 8);
        // nodes 0-4
        assert_eq!(g.node_count(), 5);
        // each edge has 2 objectives (e.g. "(0;1)")
        let first_edge = g.edge_references().next().unwrap();
        assert_eq!(first_edge.weight().objectives.len(), 2);
    }

    #[test]
    fn test_read_graph_and_reset_ids_structure() {
        let (g, node_map) =
            read::read_graph_and_reset_ids("testdata/edges_high_index.csv").unwrap();
        // edges_high_index.csv: 10→11 and 20→22 (2 edges, 4 distinct nodes)
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.node_count(), 4);
        // node_map must contain exactly the 4 original IDs mapped to 0-based indices
        assert_eq!(node_map.len(), 4);
        assert!(node_map.contains_left("10"));
        assert!(node_map.contains_left("11"));
        assert!(node_map.contains_left("20"));
        assert!(node_map.contains_left("22"));
        // indices are in 0..4
        for i in 0..4 {
            assert!(node_map.contains_right(&i));
        }
    }

    #[test]
    fn test_read_graph_invalid_path() {
        let result = read::read_graph_with_int_ids("testdata/nonexistent.csv");
        assert!(result.is_err());
    }
}
