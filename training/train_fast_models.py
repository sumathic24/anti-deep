import os
import pickle
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "models", "saved"))
os.makedirs(MODEL_DIR, exist_ok=True)


def train_text_model():
    print("Training Upgraded Fast NLP AI/Human Detector...")

    ai_texts = [
        # Technical & Architecture
        "The implementation of scalable architectures enables efficient handling of large datasets.",
        "Furthermore, the integration of intelligent systems enhances decision-making capabilities.",
        "This approach ensures optimal utilization of computational resources across distributed clusters.",
        "Adaptive models improve system robustness over time through continuous optimization loops.",
        "Automated pipelines significantly reduce manual intervention, accelerating release cadence.",
        "These methodologies contribute to sustainable innovation across modern enterprise software.",
        "This framework provides a comprehensive solution for modern challenges in throughput and latency.",
        "Data-driven insights enable strategic advantages across cross-functional product organizations.",
        "The convergence of multiple technologies creates an efficient and highly interoperable ecosystem.",
        "This structured approach guarantees consistent performance improvements across diverse benchmarks.",
        "This system is designed to improve efficiency across multiple domains through modular orchestration.",
        "The methodology focuses on scalability, long-term sustainability, and fault-tolerant architecture.",
        "Advanced algorithms are utilized to enhance predictive performance and reduce computational overhead.",
        "The model leverages optimization techniques to achieve superior generalization on unseen distributions.",
        "This solution integrates seamlessly with existing cloud-native microservice infrastructure.",
        "In modern distributed environments, container orchestration facilitates seamless service availability.",
        "Leveraging generative modeling unlocks unprecedented capabilities in automated content synthesis.",
        "The overarching paradigm focuses on holistic data governance and regulatory compliance.",
        "Cognitive architectures simulate human reasoning through multi-layered hierarchical representations.",
        "By synthesizing real-time telemetry, the platform dynamically mitigates potential performance bottlenecks.",
        "In conclusion, the proposed methodology yields notable empirical improvements across standard evaluation metrics.",
        "To foster meaningful synergy, organizations must align core strategic pillars with continuous feedback mechanisms.",
        "It is imperative to establish clear benchmarks that evaluate throughput, resilience, and operational fidelity.",
        "Natural language processing pipelines utilize self-attention mechanisms to capture rich contextual dependencies.",
        "The resulting feature embeddings provide dense mathematical representations suitable for downstream classification.",
        "Strategic technological roadmaps dictate prioritizing low-latency data pipelines and decoupled state storage.",
        "Implementing deterministic state machines ensures predictable transitions across asynchronous worker nodes.",
        "This architectural blueprint balances high availability with transactional consistency in distributed databases.",
        "Empirical validation underscores the superiority of cross-attention modules over legacy recurrent mechanisms.",
        "Comprehensive evaluations reveal that parameter pruning retains model accuracy while minimizing inference latency."
    ]

    human_texts = [
        # Daily life, casual, personal
        "Yesterday I spent 3 hours debugging this annoying issue and it turned out to be a missing semicolon.",
        "This feels way too complicated, maybe there's a simpler and more straightforward way to do it.",
        "I don't think this approach is practical in real life, especially on a tight budget.",
        "Can someone please explain this to me in simple terms? I'm completely lost.",
        "Bro this is confusing as hell, honestly I have no idea what is going on here.",
        "I think I messed something up again while trying to configure my local environment.",
        "Let's try a different approach tomorrow morning when we have fresh eyes on the problem.",
        "This worked completely fine earlier today, but now it's throwing an unexpected error.",
        "The results were not as expected in this experiment, probably due to temperature fluctuations in the room.",
        "We observed some weird inconsistencies during testing when multiple users connected at once.",
        "This method definitely requires further testing before we roll it out to anyone else.",
        "Initial findings suggest potential improvements are needed in the UI layout.",
        "The outcome was slightly different than what we originally predicted during our team meeting.",
        "Hey, are we still meeting up at the cafe around 5 PM or did your schedule change?",
        "My daughter fell off her bike and scraped her elbow, so we spent the evening in urgent care.",
        "Honestly, the latest software update completely killed the battery life on my phone.",
        "I baked fresh sourdough bread this morning and the crust turned out super crispy.",
        "Has anyone figured out why the Wi-Fi in the main office keeps disconnecting every twenty minutes?",
        "I went to three different grocery stores looking for that specific hot sauce and none of them had it.",
        "Thanks a lot for sending that file over so quickly, I really appreciate your help on this.",
        "I'm feeling pretty exhausted after that five-mile hike up the mountain trail this weekend.",
        "Can you believe how crazy the weather has been lately? It went from sunny to hailing in ten minutes.",
        "Just checked the mail and the package from last week finally arrived undamaged.",
        "I left my car keys on the kitchen counter again, almost made me late for my dentist appointment.",
        "The cat knocked over my water glass right onto my keyboard while I was typing.",
        # Human academic & informative
        "Phishing is one of the major threats affecting businesses worldwide in current times.",
        "The Roman Republic was governed by elected magistrates, a senate of elders, and popular assemblies.",
        "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
        "The Industrial Revolution was the transition to new manufacturing processes from about 1760 to 1840.",
        "Cellular respiration is a set of metabolic reactions that convert nutrients into adenosine triphosphate."
    ]

    # Create balanced dataset with multiple augmentations
    texts = (ai_texts * 20) + (human_texts * 20)
    labels = ([1] * len(ai_texts * 20)) + ([0] * len(human_texts * 20))
    texts, labels = shuffle(texts, labels, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3500,
            sublinear_tf=True,
            stop_words='english'
        )),
        ('clf', LogisticRegression(max_iter=1000, C=2.0))
    ])

    pipeline.fit(texts, labels)
    preds = pipeline.predict(texts)
    acc = accuracy_score(labels, preds)
    print(f"Fast NLP Model Train Accuracy: {acc * 100:.2f}%")

    save_path = os.path.join(MODEL_DIR, "fast_text_model.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved: {save_path}")


def train_fakenews_model():
    print("\nTraining Upgraded Fast Fake News / Misinformation Detector...")

    credible_texts = [
        "The Bureau of Labor Statistics reported that nonfarm payroll employment increased by 175,000 in April.",
        "According to peer-reviewed research published in Nature, solar flares follow an 11-year cyclical activity pattern.",
        "The Federal Reserve maintained interest rates at their current target range following their two-day policy meeting.",
        "Local municipal authorities approved funding for repairing the water filtration plant following state environmental audits.",
        "The European Space Agency confirmed the successful launch of its Earth observation satellite from the spaceport in Kourou.",
        "The World Health Organization released updated epidemiological guidance regarding seasonal influenza vaccination programs.",
        "Quarterly earnings reports filed with the SEC indicated a modest three percent revenue growth for the manufacturing sector.",
        "Civil engineers completed structural inspections of the suspension bridge and concluded that no immediate repairs are required.",
        "The United Nations General Assembly convened an annual plenary session to discuss international humanitarian aid allocation.",
        "Meteorologists from the National Weather Service forecasted widespread moderate precipitation across the Midwest this weekend."
    ]

    fakenews_texts = [
        "SHOCKING! Doctors are furious because this one secret household vegetable cures all illnesses overnight!",
        "THEY DON'T WANT YOU TO KNOW: Government whistleblowers reveal hidden mind-control chips hidden inside tap water!",
        "EXPOSED: Mainstream media is hiding the incredible truth about secret miraculous free energy generators!",
        "BREAKING: Secret leaked documents prove secret alien reptilian cabal controls global financial banking system!",
        "You won't believe what celebrities are hiding in their basements to stop aging instantly!",
        "BOMBSHELL TRUTH: Secret miracle cure banned by corrupt pharmaceutical corporations to keep you sick!",
        "URGENT: Forward this immediately to everyone you love before the authorities delete this secret video forever!",
        "They are lying to you! Proof that giant ancient pyramids were wireless power stations for galactic spaceships!",
        "WARNING: Secret chemical spraying program exposed by brave anonymous insider who vanished mysteriously!",
        "SHOCKING CONSPIRACY: The moon is completely fake and was built in a Hollywood studio in 1968!"
    ]

    texts = (credible_texts * 30) + (fakenews_texts * 30)
    labels = ([0] * len(credible_texts * 30)) + ([1] * len(fakenews_texts * 30))
    texts, labels = shuffle(texts, labels, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(max_iter=1000, C=3.0))
    ])

    pipeline.fit(texts, labels)
    preds = pipeline.predict(texts)
    acc = accuracy_score(labels, preds)
    print(f"Fake News Model Train Accuracy: {acc * 100:.2f}%")

    save_path = os.path.join(MODEL_DIR, "fast_fakenews_model.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    train_text_model()
    train_fakenews_model()
