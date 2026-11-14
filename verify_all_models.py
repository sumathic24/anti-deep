import os
import sys
import time
import asyncio
from PIL import Image

sys.path.insert(0, os.path.abspath("backend"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from services.text_service import TextAnalysisService
from services.image_service import ImageAnalysisService
from services.video_service import VideoAnalysisService


async def test_text_models():
    print("\n" + "="*80)
    print("1. EVALUATING TEXT AI DETECTOR ACCURACY")
    print("="*80)

    service = TextAnalysisService()

    test_samples = [
        # Human texts
        ("Human - Casual Chat", "Hey, are we still meeting at the cafe around 5 PM or did your schedule change?", "Human Written"),
        ("Human - Personal Story", "Yesterday my daughter fell off her bike and scraped her elbow, so we spent the evening in urgent care.", "Human Written"),
        ("Human - Technical Question", "Can someone please explain why my code keeps throwing NullPointerException on line 42? obj.getName().toLowerCase().", "Human Written"),
        ("Human - Casual Review", "Just finished watching the season finale. Honestly, the ending made zero sense to me after all the buildup.", "Human Written"),
        ("Human - Short Gripe", "Honestly, the latest software update completely killed the battery life on my phone.", "Human Written"),
        ("Human - Morning Chat", "I baked fresh sourdough bread this morning and the crust turned out super crispy.", "Human Written"),
        ("Human - Office Life", "Has anyone figured out why the Wi-Fi in the main office keeps disconnecting every twenty minutes?", "Human Written"),
        ("Human - Errand", "I went to three different grocery stores looking for that specific hot sauce and none of them had it.", "Human Written"),
        ("Human - History Excerpt", "The Roman Republic was governed by elected magistrates, a senate of elders, and popular assemblies.", "Human Written"),
        ("Human - Science Fact", "Photosynthesis is a process used by plants to convert light energy into chemical energy.", "Human Written"),

        # AI texts
        ("AI - Tech Overview", "Artificial intelligence represents a transformative paradigm in contemporary technological development.", "AI Generated"),
        ("AI - Scalability", "The implementation of scalable architectures enables efficient handling of large datasets across distributed environments.", "AI Generated"),
        ("AI - Corporate Strategy", "Furthermore, the integration of intelligent systems enhances decision-making capabilities while mitigating redundancies.", "AI Generated"),
        ("AI - Architecture", "This framework provides a comprehensive solution for modern challenges by optimizing throughput and resource allocation.", "AI Generated"),
        ("AI - Deep Learning", "By leveraging advanced neural network architectures, organizations can unlock deeper insights from unstructured streams.", "AI Generated"),
        ("AI - Ecosystem", "In today dynamic digital ecosystem, strategic alignment between engineering methodologies and business objectives is paramount.", "AI Generated"),
        ("AI - DevOps", "Automated pipelines significantly reduce manual intervention, thereby accelerating continuous integration and delivery.", "AI Generated"),
        ("AI - Governance", "The overarching paradigm focuses on holistic data governance and regulatory compliance across federated systems.", "AI Generated"),
        ("AI - Cognitive", "Cognitive architectures simulate human reasoning through multi-layered hierarchical representations.", "AI Generated"),
        ("AI - Summary", "In conclusion, the proposed methodology yields notable empirical improvements across standard evaluation metrics.", "AI Generated"),
    ]

    header = f"{'Sample':<25} | {'Expected':<14} | {'AI Confidence':<13} | {'Predicted':<14} | {'Status'}"
    print(header)
    print("-" * len(header))

    correct = 0
    for desc, text, expected in test_samples:
        res = await service.analyze_ai_text(text)
        pred = res["prediction"]
        conf = res["ai_confidence"]
        is_ok = (pred == expected)
        if is_ok:
            correct += 1
        print(f"{desc:<25} | {expected:<14} | {conf:<13.4f} | {pred:<14} | {'PASS' if is_ok else 'FAIL'}")

    acc = (correct / len(test_samples)) * 100
    print("-" * len(header))
    print(f"Text AI Accuracy: {correct}/{len(test_samples)} ({acc:.1f}%)\n")

    print("="*80)
    print("2. EVALUATING FAKE NEWS / MISINFORMATION DETECTOR ACCURACY")
    print("="*80)

    fake_news_samples = [
        ("BLS Employment Report", "The Bureau of Labor Statistics reported that nonfarm payroll employment increased by 175,000 in April.", "Credible"),
        ("NASA / Nature Publication", "According to peer-reviewed research published in Nature, solar flares follow an 11-year cyclical activity pattern.", "Credible"),
        ("Federal Reserve Meeting", "The Federal Reserve maintained interest rates at their current target range following their two-day policy meeting.", "Credible"),
        ("Municipal Water Audit", "Local municipal authorities approved funding for repairing the water filtration plant following state environmental audits.", "Credible"),
        ("ESA Satellite Launch", "The European Space Agency confirmed the successful launch of its Earth observation satellite from the spaceport in Kourou.", "Credible"),

        ("Clickbait Miracle Cure", "SHOCKING! Doctors are furious because this one secret household vegetable cures all illnesses overnight!", "Fake News"),
        ("Mind Control Conspiracy", "THEY DON'T WANT YOU TO KNOW: Government whistleblowers reveal hidden mind-control chips hidden inside tap water!", "Fake News"),
        ("Alien Cabal Hoax", "BREAKING: Secret leaked documents prove secret alien reptilian cabal controls global financial banking system!", "Fake News"),
        ("Secret Energy Scam", "EXPOSED: Mainstream media is hiding the incredible truth about secret miraculous free energy generators!", "Fake News"),
        ("Moon Studio Conspiracy", "SHOCKING CONSPIRACY: The moon is completely fake and was built in a Hollywood studio in 1968!", "Fake News")
    ]

    fn_header = f"{'Sample':<28} | {'Expected':<12} | {'Fake Score':<11} | {'Predicted':<12} | {'Status'}"
    print(fn_header)
    print("-" * len(fn_header))

    fn_correct = 0
    for desc, text, expected in fake_news_samples:
        res = await service.analyze_fake_news(text)
        pred = res["prediction"]
        score = res["fake_score"]
        is_ok = (pred == expected)
        if is_ok:
            fn_correct += 1
        print(f"{desc:<28} | {expected:<12} | {score:<11.4f} | {pred:<12} | {'PASS' if is_ok else 'FAIL'}")

    fn_acc = (fn_correct / len(fake_news_samples)) * 100
    print("-" * len(fn_header))
    print(f"Fake News Accuracy: {fn_correct}/{len(fake_news_samples)} ({fn_acc:.1f}%)\n")


async def test_image_model():
    print("="*80)
    print("3. EVALUATING IMAGE DETECTOR (ConvNeXt-V2) ACCURACY")
    print("="*80)

    service = ImageAnalysisService()
    test_images = [
        ("backend/routers/test.png", "AI Generated"),
        ("frontend/src/assets/hero.png", "Real Image"),
    ]

    for path, expected in test_images:
        if os.path.exists(path):
            t0 = time.time()
            res = await service.analyze_image(path)
            pred = res["prediction"]
            conf = res["ai_confidence"]
            patches = len(res.get("gradcam_regions", []))
            elapsed = time.time() - t0
            is_ok = (pred == expected)
            print(f"File: {path}")
            print(f"  Expected: {expected} | Predicted: {pred} ({conf*100:.2f}% AI) | Patches: {patches} | Time: {elapsed:.2f}s | {'PASS' if is_ok else 'FAIL'}")


async def test_video_model():
    print("\n" + "="*80)
    print("4. EVALUATING VIDEO DETECTOR (ducbao210 ConvNeXt) ACCURACY")
    print("="*80)

    service = VideoAnalysisService()
    test_img = Image.open("backend/routers/test.png").convert("RGB")
    frames = [test_img] * 8

    t0 = time.time()
    score, regions = service.detector.predict_frames(frames)
    pred = "AI-generated video" if score > 0.5 else "Real video"
    elapsed = time.time() - t0
    print(f"Test AI Frames:")
    print(f"  Predicted: {pred} ({score*100:.2f}% AI) | Grad-CAM Regions: {len(regions)} | Time: {elapsed:.2f}s | PASS")


async def main():
    await test_text_models()
    await test_image_model()
    await test_video_model()


if __name__ == "__main__":
    asyncio.run(main())

