import os
import sys
import torch
import pickle

sys.path.insert(0, os.path.abspath("backend"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from services.text_service import TextAnalysisService

svc = TextAnalysisService()
with open("backend/models/saved/fast_text_model.pkl", "rb") as f:
    fast_text = pickle.load(f)

test_samples = [
    # Human samples
    ("Human 1 (Casual)", "Hey guys, what time does the grocery store close tonight? I need to pick up some bread.", "Human Written"),
    ("Human 2 (Personal)", "I spent the whole afternoon fixing the flat tire on my bicycle. The valve was completely stripped.", "Human Written"),
    ("Human 3 (Story)", "My grandmother used to make the most incredible apple pies every Thanksgiving with homemade crust.", "Human Written"),
    ("Human 4 (Review)", "Just finished watching the season finale. Honestly, the ending made zero sense to me after all the buildup.", "Human Written"),
    ("Human 5 (History)", "The Roman Republic was governed by elected magistrates, a senate of elders, and popular assemblies of citizens.", "Human Written"),
    ("Human 6 (Bio)", "In biology, cellular respiration is a set of metabolic reactions that take place in the cells of organisms.", "Human Written"),
    ("Human 7 (Daily)", "Yesterday the rain started early in the morning and by noon the gutters on the north side were overflowing.", "Human Written"),

    # AI samples
    ("AI 1 (Tech)", "Artificial intelligence represents a transformative paradigm in contemporary technological development.", "AI Generated"),
    ("AI 2 (Cloud)", "The implementation of scalable architectures enables efficient handling of large datasets across distributed environments.", "AI Generated"),
    ("AI 3 (Corporate)", "Furthermore, the integration of intelligent systems enhances decision-making capabilities while mitigating redundancies.", "AI Generated"),
    ("AI 4 (Architecture)", "This framework provides a comprehensive solution for modern challenges by optimizing throughput and resource allocation.", "AI Generated"),
    ("AI 5 (Data)", "By leveraging advanced neural network architectures, organizations can unlock deeper insights from unstructured streams.", "AI Generated"),
    ("AI 6 (Strategy)", "In today dynamic digital ecosystem, strategic alignment between engineering methodologies and business objectives is paramount.", "AI Generated"),
    ("AI 7 (DevOps)", "Automated pipelines significantly reduce manual intervention, thereby accelerating continuous integration and delivery.", "AI Generated"),
]

header = f"{'Sample':<22} | {'Expected':<13} | {'Raw Desklib':<11} | {'Calibrated':<10} | {'FastModel':<9} | {'Ensemble':<8} | {'Pred':<13} | {'OK?'}"
print(header)
print("-" * len(header))

correct = 0
for name, text, expected in test_samples:
    inputs = svc.desklib_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        raw_logit = svc.desklib_model(**inputs)["logits"].item()
    
    raw_p = torch.sigmoid(torch.tensor(raw_logit)).item()
    # Calibrated Desklib probability centering around decision threshold 3.0
    cal_p = torch.sigmoid(torch.tensor((raw_logit - 3.0) * 1.2)).item()
    
    # Fast text model probability
    fast_p = float(fast_text.predict_proba([text])[0][1])
    
    # Ensemble (70% Calibrated Desklib + 30% Fast Model)
    ensemble_p = 0.70 * cal_p + 0.30 * fast_p
    pred = "AI Generated" if ensemble_p > 0.5 else "Human Written"
    is_ok = (pred == expected)
    if is_ok:
        correct += 1
    
    print(f"{name:<22} | {expected:<13} | {raw_p:<11.4f} | {cal_p:<10.4f} | {fast_p:<9.4f} | {ensemble_p:<8.4f} | {pred:<13} | {'YES' if is_ok else 'NO'}")

print("-" * len(header))
print(f"Accuracy: {correct}/{len(test_samples)} ({correct/len(test_samples)*100:.1f}%)")

