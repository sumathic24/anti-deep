def train_text_model():
    print("Training Upgraded NLP Model...")
    
    # Improved AI Text (more variety)
    ai_texts = [
        "The implementation of scalable architectures enables efficient handling of large datasets.",
        "Furthermore, the integration of intelligent systems enhances decision-making capabilities.",
        "This approach ensures optimal utilization of computational resources.",
        "Adaptive models improve system robustness over time.",
        "Automated pipelines significantly reduce manual intervention.",
        "These methodologies contribute to sustainable innovation.",
        "This framework provides a comprehensive solution for modern challenges.",
        "Data-driven insights enable strategic advantages.",
        "The convergence of multiple technologies creates an efficient ecosystem.",
        "This structured approach guarantees consistent performance improvements.",
        "This system is designed to improve efficiency across multiple domains.",
        "The methodology focuses on scalability and long-term sustainability.",
        "Advanced algorithms are used to enhance performance and accuracy.",
        "The model leverages optimization techniques for better results.",
        "This solution integrates seamlessly with existing infrastructure."
    ]
    
    # Improved Human Text (casual + semi-formal mix)
    human_texts = [
        "Phishing is one of the major threats affecting businesses worldwide in current times. Organizations and customers face the hazards arising out of phishing attacks because of anonymous access to vulnerable details. Such attacks often result in substantial financial losses. Thus, there is a need for effective intrusion detection techniques to identify and possibly nullify the effects of phishing. Classifying phishing and non-phishing web content is a critical task in information security protocols, and full-proof mechanisms have yet to be implemented in practice. The purpose of the current study is to present an ensemble machine learning model for classifying phishing websites..",
        "Yesterday I spent 3 hours debugging this issue.",
        "This feels way too complicated, maybe there's an easier way.",
        "I don’t think this approach is practical in real life.",
        "Can someone explain this in simple terms?",
        "Bro this is confusing af",
        "lol I have no idea what’s going on",
        "I think I messed something up again",
        "Let’s try a different approach tomorrow",
        "This worked earlier but now it's broken",
        "The results were not as expected in this experiment.",
        "We observed some inconsistencies during testing.",
        "This method requires further evaluation.",
        "Initial findings suggest potential improvements are needed.",
        "The outcome was slightly different than predicted."
    ]
    
    # Create dataset
    texts = (ai_texts * 100) + (human_texts * 100)
    labels = ([1] * len(ai_texts * 100)) + ([0] * len(human_texts * 100))
    
    from sklearn.utils import shuffle
    texts, labels = shuffle(texts, labels, random_state=42)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    
    # Improved pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2000,
            stop_words='english'
        )),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(texts, labels)
    
    import os, pickle
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "models", "saved")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(os.path.join(MODEL_DIR, "fast_text_model.pkl"), "wb") as f:
        pickle.dump(pipeline, f)
    
    print("NLP Model trained and saved!")