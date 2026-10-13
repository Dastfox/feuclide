//! Several takes of one sound, picked at random without repeating.

use std::time::{SystemTime, UNIX_EPOCH};

use fk_math::rand::Seeded;

/// Picks one of several takes of a sound at random, never the one picked last time: footsteps,
/// breaths, knocks that do not sound like a loop.
#[derive(Clone, Debug)]
pub struct Takes {
    last: Option<usize>,
    random: Seeded,
}

impl Default for Takes {
    fn default() -> Self {
        Self::new()
    }
}

impl Takes {
    /// Seeded from the clock, so every run hears its own order.
    pub fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos() as u64);
        Self::seeded(nanos)
    }

    /// Seeded with `seed`, the same order every time.
    pub fn seeded(seed: u64) -> Self {
        Self {
            last: None,
            random: Seeded::new(seed),
        }
    }

    /// An index in `0..count`, other than the last one picked unless `count` is 1.
    ///
    /// # Panics
    ///
    /// If `count` is 0.
    pub fn pick(&mut self, count: usize) -> usize {
        assert!(count > 0, "no takes to pick from");
        let choices = match self.last {
            Some(last) if last < count && count > 1 => count - 1,
            _ => count,
        };
        let mut pick = (self.random.next_u64() % choices as u64) as usize;
        if let Some(last) = self.last
            && choices < count
            && pick >= last
        {
            pick += 1;
        }
        self.last = Some(pick);
        pick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_the_same_take_twice_and_both_others_come_up() {
        let mut takes = Takes::seeded(42);
        let mut last = takes.pick(3);
        let mut after = [[0; 3]; 3];
        for _ in 0..600 {
            let pick = takes.pick(3);
            assert_ne!(pick, last);
            after[last][pick] += 1;
            last = pick;
        }
        for (from, counts) in after.iter().enumerate() {
            for (to, &count) in counts.iter().enumerate() {
                if from != to {
                    assert!(count > 50, "{from} → {to} only {count} times");
                }
            }
        }
        assert_eq!(Takes::seeded(7).pick(1), 0);
    }
}
