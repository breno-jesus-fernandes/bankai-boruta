use boruta_core::{FeatureDecision, binomial_lower_tail, binomial_upper_tail, decide_from_hits};

#[test]
fn binomial_tails_are_inclusive_and_symmetric() {
    assert_eq!(binomial_upper_tail(0, 2), 1.0);
    assert_eq!(binomial_upper_tail(1, 2), 0.75);
    assert_eq!(binomial_upper_tail(2, 2), 0.25);
    assert_eq!(binomial_lower_tail(0, 2), 0.25);
    assert_eq!(binomial_lower_tail(1, 2), 0.75);
    assert_eq!(binomial_lower_tail(2, 2), 1.0);
    assert_eq!(binomial_upper_tail(3, 2), 0.0);
}

#[test]
fn binomial_extremes_remain_stable_for_large_tree_counts() {
    let extreme = binomial_upper_tail(500, 500);
    assert!((extreme - 3.054936363499605e-151).abs() / extreme < 1e-12);
    assert_eq!(binomial_upper_tail(2_000, 2_000), 0.0);
    assert_eq!(binomial_lower_tail(0, 2_000), 0.0);
}

#[test]
fn decisions_apply_strict_bonferroni_against_original_feature_count() {
    assert_eq!(
        decide_from_hits(20, 20, 1, 0.05),
        FeatureDecision::Confirmed
    );
    assert_eq!(decide_from_hits(0, 20, 1, 0.05), FeatureDecision::Rejected);
    assert_eq!(
        decide_from_hits(8, 8, 100, 0.05),
        FeatureDecision::Tentative
    );
    assert_eq!(
        decide_from_hits(10, 20, 1, 0.05),
        FeatureDecision::Tentative
    );
    assert_eq!(decide_from_hits(2, 2, 1, 0.25), FeatureDecision::Tentative);
    assert_eq!(decide_from_hits(0, 2, 1, 0.25), FeatureDecision::Tentative);
}
