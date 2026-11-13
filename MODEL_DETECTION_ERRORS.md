# Model Detection Errors and Issues

## Known wrong-prediction errors from screenshots

The following incorrect results were observed during manual testing and documented in the
user-provided screenshots. These are wrong-prediction errors: the expected labels were
AI-generated for both samples, but the application returned human-written/real labels.
They are recorded here because a high confidence score does not guarantee a correct
prediction for an individual input.

| Screenshot evidence | Input | Result shown by the application | Wrong-prediction error |
|---|---|---|---|
| User screenshot: text-analysis result | Text describing traffic congestion in Bengaluru (66 words) | **Human-written**, 50.5% confidence; AI-generated 49.5% | Expected **AI-generated**, but classified as **human-written**. The result was borderline and the UI reported the **Fast model (bag-of-words + logistic regression)**. |
| User screenshot: image-analysis result | Uploaded file `AI image.jpeg` (511 KB) | **Real image**, 98.5% confidence; AI-generated 1.5% | Expected **AI-generated**, but classified as **real**. The UI reported `buildborderless/CommunityForensics-DeepfakeDet-ViT (pre-trained)`. |

**Update (October 2026):** both errors were re-checked with the current default models.
- Text: the fast model has been removed; the default `Oxidane/tmr-ai-text-detector` rates the
  Bengaluru text **AI-generated, 98.3%**.
- Image: on a 1024×1024 copy of the same picture, `CommunityForensics-DeepfakeDet-ViT` still says
  real (0.5% AI). The default image detector is now `haywoodsloan/ai-image-detector-deploy`,
  which rates it **AI-generated, 100%** and did better on a 160-image check (see README).

## Recommended follow-up

These examples should be added to the project's evaluation set and used when tuning or
retraining the detectors. Do not treat the displayed confidence as proof of authenticity;
verify the ground truth of each sample and evaluate precision, recall, confusion matrices,
and calibration on a representative held-out dataset. The screenshots are evidence of the
observed outputs, not claims that the displayed labels are correct.
