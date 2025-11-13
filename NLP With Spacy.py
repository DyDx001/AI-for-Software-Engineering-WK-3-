import spacy

def run_nlp_project():
    print("--- Task 3: NLP on Amazon Reviews (NER & Sentiment) ---")

    # Load English language model
    # Ensure you ran: python -m spacy download en_core_web_sm
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Error: spaCy model 'en_core_web_sm' not found.")
        print("Please run: python -m spacy download en_core_web_sm")
        return

    # Sample Amazon-style Reviews
    reviews = [
        "I absolutely love my new Sony WH-1000XM4 headphones! The noise cancellation is amazing.",
        "The battery life on this iPhone 14 is terrible. I regret buying it.",
        "Shipping was fast, but the screen of the Samsung Galaxy Tab arrived cracked. Very disappointed.",
        "Great value for money. The Kindle Paperwhite is the best e-reader I have used.",
        "It's okay. The Logitech mouse works, but the scroll wheel feels cheap."
    ]

    print(f"Processing {len(reviews)} reviews...\n")

    for i, text in enumerate(reviews):
        print(f"Review #{i+1}: \"{text}\"")
        doc = nlp(text)

        # --- Goal 1: Named Entity Recognition (NER) ---
        print("  > Extracted Entities:")
        entities_found = False
        for ent in doc.ents:
            # We look for ORG (Companies), PRODUCT (Objects), or GPE (Locations)
            if ent.label_ in ["ORG", "PRODUCT", "GPE", "PERSON"]:
                print(f"    - {ent.text} ({ent.label_})")
                entities_found = True
        if not entities_found:
            print("    - None found.")

        # --- Goal 2: Rule-Based Sentiment Analysis ---
        sentiment, score = rule_based_sentiment(doc)
        print(f"  > Sentiment: {sentiment} (Score: {score})")
        print("-" * 50)

def rule_based_sentiment(doc):
    """
    A simple rule-based sentiment analyzer.
    It calculates a score based on a predefined list of positive/negative words.
    """
    # Simple lexicon for demonstration
    positive_words = {"love", "amazing", "great", "best", "fast", "value", "good", "excellent"}
    negative_words = {"terrible", "regret", "cracked", "disappointed", "cheap", "bad", "slow", "hate"}
    
    score = 0
    
    # Iterate through tokens to calculate score
    for token in doc:
        # Lemmatization helps match "loved" to "love"
        lemma = token.lemma_.lower()
        
        if lemma in positive_words:
            score += 1
        elif lemma in negative_words:
            score -= 1
            
    # Determine label
    if score > 0:
        label = "POSITIVE"
    elif score < 0:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
        
    return label, score

if __name__ == "__main__":
    run_nlp_project()
