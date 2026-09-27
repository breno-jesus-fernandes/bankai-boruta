use boruta_core::{DecisionLedger, FeatureDecision};

#[test]
fn preserves_original_feature_ids_and_initially_marks_all_tentative() {
    let ledger = DecisionLedger::new(4);

    assert_eq!(ledger.decisions(), &[FeatureDecision::Tentative; 4]);
}

#[test]
fn confirms_and_rejects_only_the_requested_original_feature() {
    let mut ledger = DecisionLedger::new(3);

    ledger.decide(2, FeatureDecision::Confirmed).unwrap();
    ledger.decide(0, FeatureDecision::Rejected).unwrap();

    assert_eq!(
        ledger.decisions(),
        &[
            FeatureDecision::Rejected,
            FeatureDecision::Tentative,
            FeatureDecision::Confirmed,
        ]
    );
}

#[test]
fn refuses_to_change_a_final_decision() {
    let mut ledger = DecisionLedger::new(1);
    ledger.decide(0, FeatureDecision::Confirmed).unwrap();

    assert!(ledger.decide(0, FeatureDecision::Rejected).is_err());
    assert_eq!(ledger.decisions(), &[FeatureDecision::Confirmed]);
}

#[test]
fn rejects_feature_ids_outside_the_original_input() {
    let mut ledger = DecisionLedger::new(2);

    assert!(ledger.decide(2, FeatureDecision::Confirmed).is_err());
}

#[test]
fn tentative_cannot_be_used_to_reset_a_final_decision() {
    let mut ledger = DecisionLedger::new(1);
    ledger.decide(0, FeatureDecision::Confirmed).unwrap();

    assert!(ledger.decide(0, FeatureDecision::Tentative).is_err());
    assert_eq!(ledger.decisions(), &[FeatureDecision::Confirmed]);
}

#[test]
fn empty_input_has_no_decisions() {
    assert!(DecisionLedger::new(0).decisions().is_empty());
}
