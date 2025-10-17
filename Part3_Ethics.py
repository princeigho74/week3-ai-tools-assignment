"""
AI Tools Assignment - Part 3: Ethics & Optimization
Bias Analysis, Mitigation Strategies, and Debugging Challenges

Author: AI Assignment
Date: 2025
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PART 3: ETHICS & OPTIMIZATION - BIAS ANALYSIS AND DEBUGGING")
print("=" * 80)

# ============================================================================
# Section 1: Ethical Considerations - Bias Analysis in MNIST
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 1: ETHICAL CONSIDERATIONS - BIAS ANALYSIS IN MNIST")
print("=" * 80)

print("""
IDENTIFIED POTENTIAL BIASES IN MNIST CLASSIFIER:

1. DISTRIBUTION BIAS
   ├─ Issue: MNIST dataset may have class imbalance where some digits (0-9)
   │         are underrepresented compared to others
   ├─ Impact: Model performs worse on underrepresented digits
   └─ Scenario: If digit '8' has fewer samples, it gets lower accuracy

2. REPRESENTATION BIAS
   ├─ Issue: MNIST contains 28x28 handwritten digits from limited sources
   │         (mostly US postal service data)
   ├─ Impact: Model may not generalize to handwriting styles from other
   │         countries or age groups
   └─ Scenario: Elderly handwriting or non-Latin scripts show degraded performance

3. ANNOTATION BIAS
   ├─ Issue: If digit labels were determined by human annotators, subjective
   │         interpretations could introduce errors
   ├─ Impact: Model learns from potentially mislabeled examples
   └─ Scenario: Ambiguous digits (like '0' vs 'O') may be labeled inconsistently

4. GENDER AND DEMOGRAPHIC BIAS
   ├─ Issue: Training on specific demographic groups' handwriting
   ├─ Impact: Performance varies across demographic characteristics
   └─ Scenario: Model performs differently for handwriting of different age groups

5. HISTORICAL BIAS
   ├─ Issue: MNIST is from the 1990s; handwriting habits have changed
   ├─ Impact: Model may not recognize modern handwriting styles
   └─ Scenario: Digital pen inputs or modern writing styles underperform

═════════════════════════════════════════════════════════════════════════════════

MITIGATION STRATEGIES:

1. DATA-LEVEL MITIGATIONS
   ✓ Class Balancing: Use stratified sampling, oversampling, or SMOTE to ensure
     equal representation of all digit classes (0-9)
   ✓ Data Augmentation: Apply rotations, scaling, and elastic deformations to
     create diverse training examples
   ✓ Diversify Sources: Train on multiple handwriting datasets (EMNIST, IAM,
     Chars74K) from different countries and demographics
   ✓ Fairness Datasets: Use balanced datasets like "Balanced MNIST" with equal
     samples per class

2. ALGORITHMIC MITIGATIONS
   ✓ Fairness Constraints: Use TensorFlow Fairness Indicators to monitor
     performance across subgroups
   ✓ Threshold Adjustment: Calibrate decision thresholds differently for
     underrepresented classes
   ✓ Ensemble Methods: Combine multiple models trained on different subsets
     to reduce bias from any single model
   ✓ Adversarial Debiasing: Train a debiasing network to remove biased features

3. MONITORING AND EVALUATION
   ✓ Per-Class Metrics: Report accuracy, precision, recall separately for
     each digit (0-9)
   ✓ Fairness Audits: Conduct regular bias audits across demographic groups
   ✓ Error Analysis: Investigate which digit classes have highest error rates
   ✓ User Testing: Test on real-world data from diverse populations

4. SPACY NLP BIAS MITIGATION (Amazon Reviews)
   ├─ Language Bias: Different languages/accents may be misclassified
   ├─ Cultural Bias: Sentiment may vary by culture (e.g., "cheap" is negative
   │                 in developed countries but positive in developing ones)
   ├─ Review Source Bias: Professional vs consumer reviews have different patterns
   ├─ Mitigation: Use multilingual models, diversify training data, audit on
   │             different demographic review sources

5. PRACTICAL IMPLEMENTATION WITH TENSORFLOW FAIRNESS
""")
