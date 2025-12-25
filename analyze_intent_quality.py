"""
Intent Quality Analysis Script

Analyzes the quality and characteristics of intent labels to understand
why Flow Adapter might be causing overfitting.

Usage:
    python analyze_intent_quality.py --data_path nyc_2m_jan_feb_with_intents.parquet
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

def analyze_intent_distribution(df):
    """Analyze intent label distribution"""
    print("\n" + "=" * 80)
    print("INTENT DISTRIBUTION ANALYSIS")
    print("=" * 80)

    intent_counts = df['intent_category'].value_counts().sort_index()
    total = len(df)

    print("\nIntent Category Counts:")
    for intent, count in intent_counts.items():
        pct = count / total * 100
        print(f"  Intent {intent}: {count:,} ({pct:.2f}%)")

    # Check balance
    max_count = intent_counts.max()
    min_count = intent_counts.min()
    imbalance_ratio = max_count / min_count

    print(f"\nBalance Analysis:")
    print(f"  Max samples: {max_count:,}")
    print(f"  Min samples: {min_count:,}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")

    if imbalance_ratio > 5:
        print("  ⚠️  WARNING: Severe class imbalance detected!")
    elif imbalance_ratio > 2:
        print("  ⚠️  CAUTION: Moderate class imbalance")
    else:
        print("  ✓ Classes are relatively balanced")

    return intent_counts


def analyze_intent_characteristics(df):
    """Analyze physical characteristics of each intent"""
    print("\n" + "=" * 80)
    print("INTENT CHARACTERISTICS")
    print("=" * 80)

    # Calculate trip distance and duration
    df['distance_km'] = np.sqrt(
        (df['dropoff_latitude'] - df['pickup_latitude'])**2 +
        (df['dropoff_longitude'] - df['pickup_longitude'])**2
    ) * 111  # Approximate km

    df['duration_min'] = df['trip_duration'] / 60

    print("\nCharacteristics by Intent:")
    print(f"{'Intent':<8} {'Count':<10} {'Dist(km)':<12} {'Duration(min)':<15} {'Passengers':<12}")
    print("-" * 80)

    for intent in sorted(df['intent_category'].unique()):
        intent_df = df[df['intent_category'] == intent]

        count = len(intent_df)
        dist_mean = intent_df['distance_km'].mean()
        dist_std = intent_df['distance_km'].std()
        dur_mean = intent_df['duration_min'].mean()
        dur_std = intent_df['duration_min'].std()
        pass_mean = intent_df['passenger_count'].mean()

        print(f"{intent:<8} {count:<10,} "
              f"{dist_mean:>5.2f}±{dist_std:<4.2f} "
              f"{dur_mean:>6.1f}±{dur_std:<5.1f} "
              f"{pass_mean:>5.2f}")

    # Test for separability
    print("\n" + "=" * 80)
    print("INTENT SEPARABILITY TEST")
    print("=" * 80)

    # ANOVA test for distance
    from scipy import stats
    groups = [df[df['intent_category'] == i]['distance_km'].dropna()
              for i in sorted(df['intent_category'].unique())]
    f_stat, p_value = stats.f_oneway(*groups)

    print(f"\nANOVA test for distance across intents:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")

    if p_value < 0.001:
        print("  ✓ Intents are SIGNIFICANTLY different in distance")
    elif p_value < 0.05:
        print("  ⚠️  Intents show some difference in distance")
    else:
        print("  ❌ Intents are NOT distinguishable by distance")

    return df


def analyze_temporal_patterns(df):
    """Analyze temporal patterns of intents"""
    print("\n" + "=" * 80)
    print("TEMPORAL PATTERNS")
    print("=" * 80)

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

    # Intent by hour
    print("\nIntent Distribution by Hour of Day:")
    print(f"{'Hour':<6}", end='')
    for intent in sorted(df['intent_category'].unique()):
        print(f"Int{intent}%  ", end='')
    print()
    print("-" * 60)

    for hour in range(0, 24, 4):
        hour_df = df[(df['hour'] >= hour) & (df['hour'] < hour + 4)]
        total = len(hour_df)

        print(f"{hour:02d}-{hour+3:02d}  ", end='')
        for intent in sorted(df['intent_category'].unique()):
            count = len(hour_df[hour_df['intent_category'] == intent])
            pct = count / total * 100 if total > 0 else 0
            print(f"{pct:>5.1f}  ", end='')
        print()

    # Check if temporal patterns are distinctive
    print("\nTemporal Distinctiveness:")
    temporal_variance = []
    for intent in sorted(df['intent_category'].unique()):
        intent_df = df[df['intent_category'] == intent]
        hour_dist = intent_df['hour'].value_counts(normalize=True).sort_index()
        variance = hour_dist.var()
        temporal_variance.append(variance)
        print(f"  Intent {intent} hour variance: {variance:.4f}")

    avg_variance = np.mean(temporal_variance)
    if avg_variance > 0.005:
        print(f"  ✓ Strong temporal patterns (avg variance: {avg_variance:.4f})")
    else:
        print(f"  ⚠️  Weak temporal patterns (avg variance: {avg_variance:.4f})")


def analyze_spatial_patterns(df):
    """Analyze spatial patterns of intents"""
    print("\n" + "=" * 80)
    print("SPATIAL PATTERNS")
    print("=" * 80)

    print("\nPickup Location Spread by Intent:")
    print(f"{'Intent':<8} {'Lat Std':<10} {'Lon Std':<10} {'Spatial Var':<12}")
    print("-" * 50)

    for intent in sorted(df['intent_category'].unique()):
        intent_df = df[df['intent_category'] == intent]
        lat_std = intent_df['pickup_latitude'].std()
        lon_std = intent_df['pickup_longitude'].std()
        spatial_var = lat_std * lon_std  # Proxy for spatial spread

        print(f"{intent:<8} {lat_std:<10.6f} {lon_std:<10.6f} {spatial_var:<12.8f}")

    # Check if intents have distinct spatial patterns
    print("\nSpatial Clustering Test:")
    overall_lat_std = df['pickup_latitude'].std()
    overall_lon_std = df['pickup_longitude'].std()

    print(f"  Overall spatial std: lat={overall_lat_std:.6f}, lon={overall_lon_std:.6f}")

    clustered_count = 0
    for intent in sorted(df['intent_category'].unique()):
        intent_df = df[df['intent_category'] == intent]
        lat_std = intent_df['pickup_latitude'].std()
        lon_std = intent_df['pickup_longitude'].std()

        if lat_std < overall_lat_std * 0.8 and lon_std < overall_lon_std * 0.8:
            clustered_count += 1

    if clustered_count >= len(df['intent_category'].unique()) / 2:
        print(f"  ✓ {clustered_count} intents show spatial clustering")
    else:
        print(f"  ⚠️  Only {clustered_count} intents show spatial clustering")


def analyze_prediction_relevance(df):
    """Analyze if intents are relevant for flow prediction"""
    print("\n" + "=" * 80)
    print("PREDICTION RELEVANCE ANALYSIS")
    print("=" * 80)

    # Compute grid (same as in dataset)
    grid_size = 0.01
    lat_min, lon_min = 40.5, -74.5

    df['pickup_grid_x'] = ((df['pickup_longitude'] - lon_min) / grid_size).astype(int)
    df['pickup_grid_y'] = ((df['pickup_latitude'] - lat_min) / grid_size).astype(int)
    df['pickup_grid'] = df['pickup_grid_y'] * 1000 + df['pickup_grid_x']

    # Sample 5000 trips for efficiency
    sample_df = df.sample(min(5000, len(df)), random_state=42)

    print("\nFor sampled trips, checking if same intent → similar destination:")

    for intent in sorted(sample_df['intent_category'].unique()):
        intent_df = sample_df[sample_df['intent_category'] == intent]

        if len(intent_df) < 10:
            continue

        # Check destination diversity
        unique_destinations = intent_df['pickup_grid'].nunique()
        total_trips = len(intent_df)
        diversity = unique_destinations / total_trips

        print(f"  Intent {intent}: {total_trips} trips, "
              f"{unique_destinations} unique grids, "
              f"diversity={diversity:.2f}")

    print("\nInterpretation:")
    print("  - Low diversity (< 0.3): Intent strongly constrains destination → GOOD for prediction")
    print("  - High diversity (> 0.7): Intent provides little spatial info → BAD for prediction")


def check_label_quality(df):
    """Check for obvious labeling issues"""
    print("\n" + "=" * 80)
    print("LABEL QUALITY CHECK")
    print("=" * 80)

    # Check for impossible values
    issues = []

    if df['intent_category'].min() < 0 or df['intent_category'].max() >= 6:
        issues.append(f"Invalid intent values: range [{df['intent_category'].min()}, {df['intent_category'].max()}]")

    if df['intent_category'].isna().any():
        na_count = df['intent_category'].isna().sum()
        issues.append(f"Missing intent labels: {na_count:,} ({na_count/len(df)*100:.2f}%)")

    # Check if intent column looks like it's just random noise
    expected_entropy = np.log(6)  # Max entropy for 6 classes
    intent_probs = df['intent_category'].value_counts(normalize=True)
    actual_entropy = -(intent_probs * np.log(intent_probs)).sum()

    print(f"\nLabel Entropy Analysis:")
    print(f"  Actual entropy: {actual_entropy:.4f}")
    print(f"  Max entropy (uniform): {expected_entropy:.4f}")
    print(f"  Normalized entropy: {actual_entropy/expected_entropy:.4f}")

    if actual_entropy / expected_entropy > 0.95:
        issues.append("⚠️  Intent distribution is nearly uniform (might be random)")

    if len(issues) == 0:
        print("\n✓ No obvious labeling issues detected")
    else:
        print("\n⚠️  Potential Issues:")
        for issue in issues:
            print(f"  - {issue}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='nyc_2m_jan_feb_with_intents.parquet')
    parser.add_argument('--sample_size', type=int, default=100000,
                       help='Sample size for faster analysis')
    args = parser.parse_args()

    print("=" * 80)
    print("INTENT QUALITY ANALYSIS")
    print("=" * 80)
    print(f"\nData: {args.data_path}")

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(args.data_path)
    print(f"  Total trips: {len(df):,}")

    # Sample if too large
    if len(df) > args.sample_size:
        print(f"  Sampling {args.sample_size:,} trips for analysis...")
        df = df.sample(args.sample_size, random_state=42)

    # Run analyses
    intent_counts = analyze_intent_distribution(df)
    df = analyze_intent_characteristics(df)
    analyze_temporal_patterns(df)
    analyze_spatial_patterns(df)
    analyze_prediction_relevance(df)
    issues = check_label_quality(df)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    # Compute overall quality score
    quality_scores = []

    # Balance score
    max_count = intent_counts.max()
    min_count = intent_counts.min()
    imbalance_ratio = max_count / min_count
    balance_score = max(0, 1 - (imbalance_ratio - 1) / 10)
    quality_scores.append(('Balance', balance_score))

    print(f"\nQuality Scores:")
    print(f"  Balance: {balance_score:.2f}/1.0")

    overall_score = np.mean([score for _, score in quality_scores])
    print(f"\n  Overall Quality: {overall_score:.2f}/1.0")

    print("\nRecommendations:")
    if overall_score < 0.5:
        print("  ❌ Intent labels appear to be low quality")
        print("  → Consider re-labeling with DeepSeek or using unsupervised approach")
    elif overall_score < 0.7:
        print("  ⚠️  Intent labels have moderate quality")
        print("  → Try both current labels and re-labeling, compare results")
    else:
        print("  ✓ Intent labels appear reasonable")
        print("  → Problem might be in model architecture, not data")

    if len(issues) > 0:
        print(f"\n  Found {len(issues)} potential issues - see details above")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
