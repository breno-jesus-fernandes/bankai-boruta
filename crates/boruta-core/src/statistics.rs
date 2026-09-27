use crate::FeatureDecision;

/// Inclusive upper tail for `Binomial(trials, 0.5)`, computed in log space at
/// the tail boundary and summed by recurrence to avoid factorial overflow.
pub fn binomial_upper_tail(hits: usize, trials: usize) -> f64 {
    if hits == 0 {
        return 1.0;
    }
    if hits > trials {
        return 0.0;
    }
    if hits <= trials / 2 {
        let boundary = hits - 1;
        return (1.0 - lower_probability(boundary, trials)).clamp(0.0, 1.0);
    }

    let mut probability = binomial_probability(hits, trials);
    let mut tail = probability;
    for value in hits..trials {
        probability *= (trials - value) as f64 / (value + 1) as f64;
        tail += probability;
    }
    tail.clamp(0.0, 1.0)
}

/// Inclusive lower tail for `Binomial(trials, 0.5)`.
pub fn binomial_lower_tail(hits: usize, trials: usize) -> f64 {
    if hits >= trials {
        return 1.0;
    }
    binomial_upper_tail(trials - hits, trials)
}

/// Apply the reference sequential decision for one feature. The correction
/// denominator is always the original feature count, including finalized IDs.
pub fn decide_from_hits(
    hits: usize,
    trials: usize,
    original_feature_count: usize,
    alpha: f64,
) -> FeatureDecision {
    if trials == 0
        || hits > trials
        || original_feature_count == 0
        || !alpha.is_finite()
        || !(0.0..1.0).contains(&alpha)
    {
        return FeatureDecision::Tentative;
    }
    let upper = binomial_upper_tail(hits, trials) * original_feature_count as f64;
    if upper < alpha {
        return FeatureDecision::Confirmed;
    }
    let lower = binomial_lower_tail(hits, trials) * original_feature_count as f64;
    if lower < alpha {
        FeatureDecision::Rejected
    } else {
        FeatureDecision::Tentative
    }
}

pub(crate) fn decision_table(
    trials: usize,
    original_feature_count: usize,
    alpha: f64,
) -> Vec<FeatureDecision> {
    let mut probabilities = vec![0.0; trials + 1];
    let mode = trials / 2;
    probabilities[mode] = binomial_probability(mode, trials);
    for hits in mode..trials {
        probabilities[hits + 1] = probabilities[hits] * (trials - hits) as f64 / (hits + 1) as f64;
    }
    for hits in (1..=mode).rev() {
        probabilities[hits - 1] = probabilities[hits] * hits as f64 / (trials - hits + 1) as f64;
    }

    let mut lower = vec![0.0; trials + 1];
    let mut running = 0.0;
    for (hits, probability) in probabilities.iter().enumerate() {
        running += probability;
        lower[hits] = running;
    }
    let mut upper = vec![0.0; trials + 1];
    running = 0.0;
    for hits in (0..=trials).rev() {
        running += probabilities[hits];
        upper[hits] = running;
    }
    (0..=trials)
        .map(|hits| {
            let upper_adjusted = upper[hits] * (original_feature_count as f64);
            if upper_adjusted < alpha {
                FeatureDecision::Confirmed
            } else if lower[hits] * (original_feature_count as f64) < alpha {
                FeatureDecision::Rejected
            } else {
                FeatureDecision::Tentative
            }
        })
        .collect()
}

fn lower_probability(hits: usize, trials: usize) -> f64 {
    if hits == 0 {
        return binomial_probability(0, trials);
    }
    let mut probability = binomial_probability(hits, trials);
    let mut total = probability;
    for value in (1..=hits).rev() {
        probability *= value as f64 / (trials - value + 1) as f64;
        total += probability;
    }
    total
}

fn binomial_probability(hits: usize, trials: usize) -> f64 {
    let terms = hits.min(trials - hits);
    let log_choose = (1..=terms)
        .map(|i| ((trials - terms + i) as f64 / i as f64).ln())
        .sum::<f64>();
    (log_choose - trials as f64 * std::f64::consts::LN_2).exp()
}

#[cfg(test)]
mod tests {
    use super::{decide_from_hits, decision_table};

    #[test]
    fn precomputed_decisions_match_reference_tails_for_every_hit_count() {
        for trials in 1..=128 {
            for original_features in [1, 3, 64] {
                let table = decision_table(trials, original_features, 0.01);
                for (hits, actual) in table.into_iter().enumerate() {
                    assert_eq!(
                        actual,
                        decide_from_hits(hits, trials, original_features, 0.01),
                        "trials={trials}, hits={hits}, features={original_features}"
                    );
                }
            }
        }
    }
}
