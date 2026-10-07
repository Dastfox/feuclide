use bevy_ecs::component::Component;

/// One generator of a deck group, or its inverse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Letter {
    /// Index of the generator.
    pub generator: u16,
    /// Whether this is the generator's inverse.
    pub inverse: bool,
}

impl Letter {
    /// The generator `generator`.
    pub fn new(generator: u16) -> Self {
        Self {
            generator,
            inverse: false,
        }
    }

    /// The inverse letter.
    pub fn inverse(self) -> Self {
        Self {
            inverse: !self.inverse,
            ..self
        }
    }
}

/// Which copy of the fundamental domain a pose is expressed in, as a word in the deck group's
/// generators.
///
/// In a quotient space (a torus, a portal-linked place) a pose is stored relative to a
/// representative chart; crossing a face of the fundamental domain applies a deck
/// transformation and appends its letter here. Words are kept freely reduced: a letter next to
/// its inverse cancels. The quotient's own relations (`fk-quotient`, milestone E2) reduce
/// further.
///
/// As a component it is the chart of an entity relative to its parent. Propagation composes it
/// the same way as poses, `world(child) = world(parent) · local(child)`, into
/// [`GlobalPose::chart`](crate::GlobalPose::chart). Entities without one are in their parent's
/// chart.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ChartTag(Vec<Letter>);

impl ChartTag {
    /// The empty word: the representative chart.
    pub fn identity() -> Self {
        Self::default()
    }

    /// The freely reduced word of `letters`.
    pub fn from_letters(letters: impl IntoIterator<Item = Letter>) -> Self {
        let mut tag = Self::identity();
        for letter in letters {
            tag.push(letter);
        }
        tag
    }

    /// The letters of the word, applied left to right.
    pub fn letters(&self) -> &[Letter] {
        &self.0
    }

    /// Whether this is the representative chart.
    pub fn is_identity(&self) -> bool {
        self.0.is_empty()
    }

    /// Appends a letter, cancelling it against the last one if they are inverses.
    pub fn push(&mut self, letter: Letter) {
        if self.0.last() == Some(&letter.inverse()) {
            self.0.pop();
        } else {
            self.0.push(letter);
        }
    }

    /// `self · rhs`.
    pub fn then(&self, rhs: &Self) -> Self {
        let mut tag = self.clone();
        for &letter in &rhs.0 {
            tag.push(letter);
        }
        tag
    }

    /// The inverse word.
    pub fn inverse(&self) -> Self {
        Self(self.0.iter().rev().map(|letter| letter.inverse()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_reduce_freely() {
        let (a, b) = (Letter::new(0), Letter::new(1));
        let word = ChartTag::from_letters([a, b, b.inverse(), a]);
        assert_eq!(word.letters(), &[a, a]);
        assert!(word.then(&word.inverse()).is_identity());
        let mixed = ChartTag::from_letters([a, b]);
        assert_eq!(mixed.inverse().letters(), &[b.inverse(), a.inverse()]);
        assert!(mixed.inverse().then(&mixed).is_identity());
    }
}
