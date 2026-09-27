//! Core state and statistics for the native Boruta implementation.

use thiserror::Error;

/// The current, irreversible decision for a feature from the original input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureDecision {
    Tentative,
    Confirmed,
    Rejected,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum LedgerError {
    #[error("feature id {feature_id} is outside the original feature range 0..{feature_count}")]
    InvalidFeatureId {
        feature_id: usize,
        feature_count: usize,
    },
    #[error("feature {feature_id} already has a final decision")]
    AlreadyDecided { feature_id: usize },
    #[error("a feature decision must be Confirmed or Rejected")]
    TentativeIsNotFinal,
}

/// Stores decisions by original feature id, independent of the active training view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionLedger {
    decisions: Vec<FeatureDecision>,
}

impl DecisionLedger {
    pub fn new(feature_count: usize) -> Self {
        Self {
            decisions: vec![FeatureDecision::Tentative; feature_count],
        }
    }

    pub fn decisions(&self) -> &[FeatureDecision] {
        &self.decisions
    }

    pub fn decide(
        &mut self,
        feature_id: usize,
        decision: FeatureDecision,
    ) -> Result<(), LedgerError> {
        let feature_count = self.decisions.len();
        let current = self
            .decisions
            .get_mut(feature_id)
            .ok_or(LedgerError::InvalidFeatureId {
                feature_id,
                feature_count,
            })?;

        if decision == FeatureDecision::Tentative {
            return Err(LedgerError::TentativeIsNotFinal);
        }
        if *current != FeatureDecision::Tentative {
            return Err(LedgerError::AlreadyDecided { feature_id });
        }
        *current = decision;
        Ok(())
    }
}
