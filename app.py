import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import TextPreprocessor
from model_training import PhishingClassifier
from evaluate import ModelEvaluator


def main():
    print("Loading data...")
    try:
        df = pd.read_csv("sms_spam.csv", encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        print("Dataset not found. Generating fallback dummy dataset...")
        df = pd.DataFrame({
            'label': [
                'ham', 'spam', 'ham', 'spam', 'ham', 'spam',
                'ham', 'spam', 'ham', 'spam', 'ham', 'spam',
                'ham', 'spam', 'ham', 'spam', 'ham', 'spam'
            ],
            'message': [
                'Hi, how are you?',
                'Claim your FREE lottery ticket now! Call 08000930705 now!',
                'Meeting at 10 AM tomorrow in Room 101.',
                'URGENT: Account compromised, click here to verify immediately!',
                'Can you send me those notes from class?',
                'Congrats! You won a $1000 gift card! Claim reward now!',
                'Are you free for lunch today?',
                'WINNER! Reply WIN to claim your cash prize today!',
                'I am heading to the library now, meet you there.',
                'Security Alert: Unauthorized access detected. Reset password.',
                'Let me know when the meeting starts.',
                'Get pre-approved for a loan now with zero down payment!',
                'Can we catch up on the project this evening?',
                'Your bank account will be suspended unless you verify details now.',
                'The homework assignment has been uploaded to portal.',
                'Free entry to win cash prizes! Text WIN to 87121 now.',
                'Great presentation today, good job team.',
                'Unclaimed prize waiting for you! Click link to claim.'
            ]
        })

    print("Cleaning text and splitting data...")
    preprocessor = TextPreprocessor()
    df['clean_message'] = df['message'].apply(preprocessor.clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_message'], df['label'], test_size=0.25, random_state=42
    )

    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Multinomial Naive Bayes model...")
    classifier = PhishingClassifier()
    classifier.train(X_train_vec, y_train)

    print("Evaluating model...")
    predictions = classifier.predict(X_test_vec)
    metrics = ModelEvaluator.evaluate(y_test, predictions, pos_label='spam')

    print("\n--- MODEL METRICS ---")
    print(f"Accuracy:  {metrics['Accuracy']:.2%}")
    print(f"Precision: {metrics['Precision']:.2%}")
    print(f"Recall:    {metrics['Recall']:.2%}")
    print(f"F1-Score:  {metrics['F1-Score']:.2%}")
    print("\nConfusion Matrix (Format: TN, FP | FN, TP):")
    print(metrics['Confusion Matrix'])

    print("\n--- LIVE PREDICTION TEST ---")
    test_messages = [
        "Hey, are we still studying applied numerical methods tonight?",
        "URGENT: Your university account password has expired. Click here to verify."
    ]

    for msg in test_messages:
        clean_msg = preprocessor.clean_text(msg)
        vec_msg = vectorizer.transform([clean_msg])

        prediction = classifier.predict(vec_msg)[0]
        proba = classifier.predict_proba(vec_msg)[0]

        status = "[MALICIOUS - SPAM]" if prediction == 'spam' else "[SAFE - HAM]"
        print(f"\nInput: \"{msg}\"")
        print(f"Classification: {status}")
        print(f"Confidence Score: {max(proba):.2%}")


if __name__ == "__main__":
    main()