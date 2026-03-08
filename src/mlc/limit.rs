mod test;
use std::collections::HashMap;

/// Routing constraint limits grouped by category.
///
/// Each "category" is a node-weight value (e.g., a zone or fare class) that labels nodes
/// on the graph. For each category, the limit system tracks the Pareto-optimal set of
/// (cost, time) pairs observed at constrained nodes, enabling early stopping of the MLC
/// search when a label has already exceeded every known limit for every category.
///
/// The limit system assumes exactly 2 objectives: time (index 0) and cost (index 1).
#[derive(Debug)]
pub struct Limits<T: std::cmp::Eq + std::hash::Hash + std::marker::Copy> {
    pub limits: HashMap<T, Vec<Limit>>,
    /// Cache mapping cost → max-time limit to avoid repeated `determine_limit` calls.
    limit_cache: HashMap<u64, u64>,
}

/// A single (cost, time) constraint point within a category's Pareto frontier.
///
/// A label exceeds a limit when its cost is ≥ `cost` and its time is ≥ `time`.
#[derive(Debug, PartialEq, Eq)]
pub struct Limit {
    pub cost: u64,
    pub time: u64,
}

impl<T: std::cmp::Eq + std::hash::Hash + std::marker::Copy> Limits<T> {
    pub fn new() -> Limits<T> {
        Limits {
            limits: HashMap::new(),
            limit_cache: HashMap::new(),
        }
    }

    pub fn add_category(&mut self, category: T) {
        self.limits.insert(category, Vec::new());
        self.update_limit(category, u64::max_value(), u64::max_value());
    }

    pub fn is_initialized(&self) -> bool {
        // limits must contain at least category and each category must have at least one limit
        self.limits.len() > 0 && self.limits.values().all(|v| v.len() > 0)
    }

    pub fn update_limit(&mut self, category: T, cost: u64, time: u64) -> bool {
        let limit = Limit { cost, time };
        let limits = self.limits.get_mut(&category).unwrap();
        // check if any limit dominates the new limit
        for l in limits.iter() {
            if l.cost <= limit.cost && l.time <= limit.time {
                return false;
            }
        }
        // retain all limits that are not dominated by the new limit
        limits.retain(|l| l.cost < limit.cost || l.time < limit.time);

        limits.push(limit);

        self.limit_cache.clear();

        return true;
    }

    /// Returns the maximum allowed time for a given cost, or `None` if no limit applies.
    ///
    /// Used by the post-processing step in `run()` to discard labels whose time exceeds
    /// the tightest limit across all categories for their cost bucket.
    pub fn get_max_time_for_cost(&mut self, cost: u64) -> Option<u64> {
        if let Some(&limit) = self.limit_cache.get(&cost) {
            return Some(limit);
        }
        let limit = self.determine_limit(cost);
        self.limit_cache.insert(cost, limit);
        Some(limit)
    }

    /// `is_limit_exceeded` returns true if each category has a limit that dominates the given cost and time
    pub fn is_limit_exceeded(&mut self, cost: u64, time: u64) -> bool {
        if let Some(&limit) = self.limit_cache.get(&cost) {
            return limit <= time;
        }
        let limit = self.determine_limit(cost);
        self.limit_cache.insert(cost, limit);
        return limit <= time;
    }

    fn determine_limit(&mut self, cost: u64) -> u64 {
        let mut min_limits = Vec::new();
        for limits in self.limits.values() {
            let mut min_limit = u64::max_value();
            for limit in limits.iter() {
                if limit.cost <= cost {
                    min_limit = std::cmp::min(min_limit, limit.time);
                }
            }
            min_limits.push(min_limit);
        }
        return min_limits.iter().max().unwrap().clone();
    }
}
