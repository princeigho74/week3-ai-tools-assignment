"""
AI Tools Assignment - Part 2: Task 3
NLP with spaCy: Named Entity Recognition and Sentiment Analysis

Author: AI Assignment
Date: 2025
"""

import spacy
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TASK 3: NLP WITH SPACY - NER AND SENTIMENT ANALYSIS")
print("=" * 70)

# ============================================================================
# Step 1: Load spaCy Model
# ============================================================================
print("\n[Step 1] Loading spaCy NLP Model...")
try:
    # Load the English model (ensure it's installed: python -m spacy download en_core_web_sm)
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy model loaded successfully!")
except OSError:
    print("Note: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    print("Using basic tokenizer instead...")
    nlp = spacy.blank("en")

# ============================================================================
# Step 2: Sample Amazon Product Reviews
# ============================================================================
print("\n[Step 2] Loading Sample Amazon Product Reviews...")

# Sample reviews dataset (in real scenario, load from Kaggle)
reviews = [
    "The iPhone 14 Pro Max is an incredible smartphone with amazing camera quality. Highly recommended!",
    "Samsung Galaxy S23 has excellent display but the battery life is disappointing.",
    "I love my MacBook Pro for its performance. Apple really outdid themselves.",
    "The Amazon Echo Dot works great for smart home automation. Very satisfied with this purchase.",
    "Google Pixel 7 has poor build quality and stopped working after two weeks. Avoid!",
    "The Sony WH-1000XM5 headphones have exceptional noise cancellation and sound quality.",
    "Microsoft Surface Laptop 5 is overpriced and has too many software issues.",
    "Canon EOS R5 is perfect for professional photography. Worth every penny!",
    "The Dell XPS 13 is compact and powerful, but the keyboard feels cheap.",
    "LG OLED TV provides stunning picture quality. Best investment for my living room."
]

print(f"Number of reviews: {len(reviews)}")
print("\nSample reviews:")
for i, review in enumerate(reviews[:3]):
    print(f"  {i+1}. {review}")

# ============================================================================
# Step 3: Named Entity Recognition (NER)
# ============================================================================
print("\n" + "=" * 70)
print("NAMED ENTITY RECOGNITION (NER) EXTRACTION")
print("=" * 70)

# Dictionary to store extracted entities
entities_data = {
    'Review': [],
    'Entity': [],
    'Entity Type': [],
    'Label': []
}

all_products = []
all_brands = []

print("\n[Step 3] Performing NER on Reviews...\n")

for idx, review in enumerate(reviews):
    # Process the review with spaCy
    doc = nlp(review)
    
    print(f"Review {idx + 1}: {review[:60]}...")
    
    # Extract named entities
    entities_found = False
    for ent in doc.ents:
        # Look for product names (often PRODUCT label in trained models)
        # Brands are typically ORG or GPE entities
        if ent.label_ in ['PRODUCT', 'ORG', 'GPE', 'PERSON']:
            entities_data['Review'].append(idx + 1)
            entities_data['Entity'].append(ent.text)
            entities_data['Entity Type'].append(ent.label_)
            
            # Classify as product or brand
            if 'iPhone' in ent.text or 'Galaxy' in ent.text or 'MacBook' in ent.text or \
               'Echo' in ent.text or 'Pixel' in ent.text or 'WH-' in ent.text or \
               'Surface' in ent.text or 'XPS' in ent.text or 'OLED' in ent.text or \
               'EOS' in ent.text or any(device in ent.text for device in ['Phone', 'Laptop', 'TV', 'Headphones']):
                entities_data['Label'].append('PRODUCT')
                all_products.append(ent.text)
            else:
                entities_data['Label'].append('BRAND')
                all_brands.append(ent.text)
            
            print(f"  ├─ Entity: '{ent.text}' | Type: {ent.label_} | Label: {entities_data['Label'][-1]}")
            entities_found = True
    
    # Rule-based extraction for product names not caught by NER
    tokens = [token.text for token in doc]
    for token in tokens:
        if any(keyword in token for keyword in ['iPhone', 'Galaxy', 'MacBook', 'Echo', 'Pixel', 
                                                   'Headphones', 'Surface', 'XPS', 'OLED', 'EOS']):
            if token not in [e['Entity'] for e in [dict(zip(entities_data.keys(), 
                            [entities_data[k][i] for k in entities_data.keys()])) 
                            for i in range(len(entities_data['Review']))]]:
                entities_data['Review'].append(idx + 1)
                entities_data['Entity'].append(token)
                entities_data['Entity Type'].append('PRODUCT')
                entities_data['Label'].append('PRODUCT')
                all_products.append(token)
                print(f"  └─ Entity (Rule-based): '{token}' | Type: PRODUCT | Label: PRODUCT")
                entities_found = True
    
    if not entities_found:
        print(f"  └─ No named entities found")
    print()

# Create DataFrame for NER results
df_ner = pd.DataFrame(entities_data)

print("\nNER RESULTS SUMMARY:")
print("=" * 70)
print(df_ner.to_string(index=False))

print("\n\nEntity Statistics:")
print(f"Total entities extracted: {len(df_ner)}")
print(f"Unique products: {len(set(all_products))}")
print(f"Unique brands: {len(set(all_brands))}")

if all_products:
    print(f"\nMost common products:")
    for product, count in Counter(all_products).most_common(5):
        print(f"  - {product}: {count} mentions")

# ============================================================================
# Step 4: Sentiment Analysis (Rule-Based Approach)
# ============================================================================
print("\n" + "=" * 70)
print("SENTIMENT ANALYSIS (RULE-BASED APPROACH)")
print("=" * 70)

# Define sentiment lexicons
positive_words = {
    'excellent', 'amazing', 'incredible', 'great', 'love', 'perfect', 'awesome', 
    'outstanding', 'wonderful', 'fantastic', 'best', 'superior', 'exceptional',
    'highly', 'recommended', 'good', 'beautiful', 'stunning', 'worth', 'satisfied'
}

negative_words = {
    'poor', 'bad', 'disappointing', 'terrible', 'hate', 'worst', 'awful',
    'horrible', 'useless', 'cheap', 'issue', 'problem', 'avoid', 'stopped',
    'overpriced', 'broken', 'failed', 'waste', 'regret', 'expensive'
}

# Negation words that flip sentiment
negation_words = {'no', 'not', 'never', 'neither', 'nobody', 'nothing', 'cannot', "can't"}

print("\n[Step 4] Analyzing Sentiment of Reviews...\n")

sentiment_results = {
    'Review Number': [],
    'Review Text': [],
    'Sentiment': [],
    'Positive Words': [],
    'Negative Words': [],
    'Confidence': []
}

for idx, review in enumerate(reviews):
    doc = nlp(review.lower())
    
    # Extract words and check sentiment
    pos_count = 0
    neg_count = 0
    positive_found = []
    negative_found = []
    
    tokens = [token.text for token in doc]
    
    for i, token in enumerate(tokens):
        # Check for negation context (word before the sentiment word)
        negated = False
        if i > 0 and tokens[i-1] in negation_words:
            negated = True
        
        if token in positive_words:
            if negated:
                neg_count += 1
                negative_found.append(token)
            else:
                pos_count += 1
                positive_found.append(token)
        
        elif token in negative_words:
            if negated:
                pos_count += 1
                positive_found.append(f"not {token}")
            else:
                neg_count += 1
                negative_found.append(token)
    
    # Determine sentiment
    if pos_count > neg_count:
        sentiment = "POSITIVE"
        confidence = (pos_count - neg_count) / max(pos_count + neg_count, 1)
    elif neg_count > pos_count:
        sentiment = "NEGATIVE"
        confidence = (neg_count - pos_count) / max(pos_count + neg_count, 1)
    else:
        sentiment = "NEUTRAL"
        confidence = 0.0 if pos_count == 0 else 1.0
    
    sentiment_results['Review Number'].append(idx + 1)
    sentiment_results['Review Text'].append(review[:50] + "...")
    sentiment_results['Sentiment'].append(sentiment)
    sentiment_results['Positive Words'].append(', '.join(positive_found) if positive_found else 'None')
    sentiment_results['Negative Words'].append(', '.join(negative_found) if negative_found else 'None')
    sentiment_results['Confidence'].append(f"{confidence:.2f}")
    
    # Print details
    print(f"Review {idx + 1}: {sentiment}")
    print(f"  Text: {review[:60]}...")
    print(f"  Positive words: {positive_found if positive_found else 'None'}")
    print(f"  Negative words: {negative_found if negative_found else 'None'}")
    print(f"  Confidence: {confidence:.2f}\n")

# Create DataFrame for sentiment results
df_sentiment = pd.DataFrame(sentiment_results)

print("\nSENTIMENT ANALYSIS SUMMARY:")
print("=" * 70)
print(df_sentiment.to_string(index=False))

# Sentiment distribution
print("\n\nSentiment Distribution:")
sentiment_counts = df_sentiment['Sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df_sentiment)) * 100
    print(f"  {sentiment}: {count} ({percentage:.1f}%)")

# ============================================================================
# Step 5: Combined Analysis
# ============================================================================
print("\n" + "=" * 70)
print("COMBINED NER + SENTIMENT ANALYSIS")
print("=" * 70)

# Merge NER and Sentiment results
combined_results = []
for review_idx, entities_in_review in df_ner.groupby('Review'):
    sentiment_row = df_sentiment[df_sentiment['Review Number'] == review_idx].iloc[0]
    
    for _, entity_row in entities_in_review.iterrows():
        combined_results.append({
            'Review #': review_idx,
            'Entity': entity_row['Entity'],
            'Type': entity_row['Label'],
            'Sentiment': sentiment_row['Sentiment'],
            'Text Preview': reviews[review_idx - 1][:45]
        })

df_combined = pd.DataFrame(combined_results)

print("\nProduct-Sentiment Associations:")
print("=" * 70)
if len(df_combined) > 0:
    print(df_combined[['Entity', 'Type', 'Sentiment', 'Text Preview']].drop_duplicates().to_string(index=False))
else:
    print("No entities found in reviews.")

# Product sentiment summary
if len(df_combined) > 0:
    print("\n\nProduct Sentiment Summary:")
    product_sentiment = df_combined.groupby('Entity')['Sentiment'].apply(
        lambda x: f"POSITIVE ({(x=='POSITIVE').sum()}), NEGATIVE ({(x=='NEGATIVE').sum()}), NEUTRAL ({(x=='NEUTRAL').sum()})"
    )
    for product, sentiment_dist in product_sentiment.items():
        print(f"  {product}: {sentiment_dist}")

print("\n" + "=" * 70)
print("PART 2 TASK 3 COMPLETED SUCCESSFULLY!")
print("=" * 70)
