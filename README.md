<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=220&section=header&text=AI-Powered%20Visual%20Defect%20Detection&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Real-Time%20Manufacturing%20Quality%20Inspection%20with%20Deep%20Learning&descAlignY=55&descSize=18" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=1000&color=8A63D2&center=true&vCenter=true&width=750&lines=Detecting+scratches%2C+dents+%26+cracks+in+real+time...;Powered+by+MobileNetV2+%2B+Two-Phase+Transfer+Learning...;FastAPI+backend+%7C+Streamlit+demo+UI...;Explainable+AI+with+Grad-CAM+%2B+Anomaly+Detection..." alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#-license)

[![Stars](https://img.shields.io/github/stars/thrinathpolanki/defect-detection?style=social)](https://github.com/thrinathpolanki)
[![Forks](https://img.shields.io/github/forks/thrinathpolanki/defect-detection?style=social)](https://github.com/thrinathpolanki)

<br/>

![Profile Views](https://komarev.com/ghpvc/?username=thrinathpolanki&repo=defect-detection&color=8A63D2&style=for-the-badge&label=REPO+VIEWS)

</div>

---

## 📌 What Is This Project?

**AI-Powered Visual Defect Detection** is a full-stack computer vision system that automatically inspects photos of manufactured products and classifies each one as **✅ GOOD** or **❌ DEFECTIVE** — the same job a human quality-control inspector does on a factory line, done instantly by a trained neural network.

It's built as a **complete, end-to-end ML pipeline**, not just a notebook experiment:

```
📸 Image → 🧹 Preprocessing/Augmentation → 🧠 Two-Phase CNN Training → 🛡️ Anomaly Guardrail → 📊 Evaluation → 🚀 Real-Time Deployment
```

<div align="center">
<img src="https://media.giphy.com/media/qgQUggAC3Pfv687qPC/giphy.gif" width="55%" alt="AI scanning animation"/>
</div>

---

## 💡 Why Is This Useful?

Manual visual inspection on a production line is:
- **Slow** — a human can only check so many parts per minute.
- **Inconsistent** — fatigue, distraction, and subjective judgment cause missed defects.
- **Expensive to scale** — more output requires more inspectors.

This system solves that by giving a factory line an **instant, consistent, tireless inspector**:

| Benefit | Impact |
|---|---|
| ⚡ **Real-time decisions** | Predictions in milliseconds via a REST API |
| 🎯 **Consistent accuracy** | No fatigue, no subjectivity — same standard every time |
| 🛡️ **Fail-safe by design** | Unrecognized/anomalous inputs are flagged as defective, not silently passed |
| 🔍 **Explainable results** | Grad-CAM heatmaps show *exactly* where the model saw a defect |
| 📈 **Measurable quality** | Precision/recall/ROC-AUC reporting, not just a gut feeling |
| 🧩 **Reusable pipeline** | Swap in any product's images — no architecture changes needed |

---

## 🛠️ Technology Stack

<div align="center">

| Layer | Technology | Purpose |
|---|---|---|
| 🐍 **Language** | Python 3.10+ | Core implementation language |
| 🧠 **Deep Learning** | TensorFlow / Keras | Model building & training |
| 🏗️ **Architecture** | MobileNetV2 (Two-Phase Transfer Learning) | Fast, lightweight, accurate CNN backbone |
| 🖼️ **Image Handling** | Pillow, OpenCV | Image loading, resizing, feature extraction, heatmap rendering |
| 📊 **Evaluation** | scikit-learn, Matplotlib, Seaborn | Precision/recall/F1/ROC-AUC + visual reports |
| ⚙️ **Backend API** | FastAPI + Uvicorn | Real-time inference REST endpoint |
| 🎨 **Demo UI** | Streamlit | Interactive web dashboard for live testing |
| 🔬 **Explainability** | Grad-CAM (custom) | Visual heatmap of the model's decision reasoning |
| 🛡️ **Safety Layer** | Custom statistical anomaly detector | Flags out-of-distribution inputs (wrong object, unrelated image) as defective |

</div>

---

## 📁 Project Structure

```
defect-detection/
├── data/
│   ├── train/
│   │   ├── good/            # 🟢 normal product images
│   │   └── defective/       # 🔴 defective product images
│   └── test/
│       ├── good/
│       └── defective/
├── src/
│   ├── generate_synthetic_data.py     # creates a demo dataset instantly
│   ├── data_preprocessing.py          # augmentation & data loading
│   ├── model.py                       # CNN architecture (MobileNetV2, two-phase)
│   ├── train.py                       # two-phase training loop
│   ├── compute_reference_stats.py     # builds the anomaly-detection fingerprint
│   ├── evaluate.py                    # precision/recall/confusion matrix
│   └── utils.py                       # plotting + Grad-CAM + anomaly scoring
├── app/
│   ├── api.py                         # FastAPI real-time inference API
│   └── streamlit_app.py               # interactive demo UI
├── models/                            # trained model + reports saved here
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started on Your Local Machine (Windows / PowerShell)

Every command below includes **why** you're running it — not just what to type.

### Step 1 — Clone / pull the project
```powershell
git clone https://github.com/thrinathpolanki/defect-detection.git
cd defect-detection
```
> 📥 This downloads the project code to your machine and moves your terminal into the project folder, so every command afterward runs in the right place.

### Step 2 — Create a virtual environment
```powershell
python -m venv venv
```
> 🧪 A virtual environment is an **isolated Python installation** just for this project. It keeps this project's libraries (TensorFlow, FastAPI, etc.) separate from other Python projects on your machine, avoiding version conflicts.

### Step 3 — Activate the virtual environment
```powershell
venv\Scripts\activate
```
> 🔑 This switches your terminal to use the isolated environment you just created. You'll see `(venv)` appear at the start of your PowerShell prompt, confirming it's active. **You must do this every time** you open a new terminal to work on this project.

> ⚠️ If PowerShell blocks this with an execution-policy error, run this once (as your normal user, not admin):
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> This allows locally-created scripts (like the venv activator) to run, while still blocking untrusted downloaded scripts — a safe, standard developer setting.

### Step 4 — Install all dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```
> 📦 The first command updates `pip` itself (the tool that installs packages) to avoid outdated-installer bugs. The second reads `requirements.txt` and installs every library this project needs (TensorFlow, FastAPI, Streamlit, etc.) in one shot, with versions chosen to work together.

### Step 5 — Generate a demo dataset
```powershell
python src\generate_synthetic_data.py
```
> 🖼️ This generates 500 synthetic product images (good + defective) so you can run the **entire pipeline immediately**, without needing to source and label a real dataset first. It populates `data/train/` and `data/test/`.

### Step 6 — Train the model (two-phase training)
```powershell
python src\train.py
```
> 🧠 This is where the actual learning happens, in **two phases**:
> - **Phase 1 (warm-up):** the pretrained MobileNetV2 base is completely frozen, and only the new classification head trains. This lets the head learn to use the pretrained features properly *before* anything is allowed to disturb them.
> - **Phase 2 (fine-tuning):** the later base layers are unfrozen and trained further with a much smaller learning rate, gently adapting the pretrained features to your specific product.
>
> Skipping straight to fine-tuning (unfreezing from the first epoch) is a common transfer-learning mistake that causes the model to *look* like it's learning (training accuracy climbs) while actually failing to generalize to anything outside what it memorized — this two-phase approach avoids that. The best model (by validation loss, across both phases) is saved to `models/best_model.keras`.

### Step 7 — Build the anomaly-detection safety net
```powershell
python src\compute_reference_stats.py
```
> 🛡️ A CNN can only ever choose between the classes it was trained on. Shown something completely unrelated (a logo, a random photo), it has no way to say "I don't recognize this" — it's forced to guess. This script scans every "good" training image and records its typical color/texture "fingerprint," then **automatically calibrates a detection threshold from your own data** (rather than a hardcoded guess) so genuine good images always pass, while truly unrelated images get flagged as **defective**. This is the same "when in doubt, flag it" philosophy real QA systems use — a missed real defect is far costlier than an unnecessary flag.

### Step 8 — Evaluate the model
```powershell
python src\evaluate.py
```
> 📊 This runs the trained model against the held-out test set (images it never saw during training) and reports **precision**, **recall**, **F1-score**, and **ROC-AUC** — the numbers that prove how well it actually works. Results are saved to `models/evaluation_report.txt` plus confusion-matrix and ROC-curve plots.

### Step 9 — Launch the interactive demo
```powershell
streamlit run app\streamlit_app.py --server.enableXsrfProtection false
```
> 🎨 This starts a local web app (opens automatically at `http://localhost:8501`) where you can upload a product photo and instantly see the prediction, confidence score, anomaly score, and a Grad-CAM heatmap. The `--server.enableXsrfProtection false` flag avoids a known Streamlit file-upload issue on some local setups.

### Step 10 (optional) — Launch the REST API
```powershell
uvicorn app.api:app --reload --port 8000
```
> 🌐 This starts the FastAPI backend, simulating how the model would be called by a real inspection system on a production line. Visit `http://127.0.0.1:8000/docs` for an interactive Swagger page to test the `/predict` endpoint directly. `--reload` auto-restarts the server whenever you edit the code, which is convenient during development.

<div align="center">
<img src="https://raw.githubusercontent.com/catppuccin/catppuccin/main/assets/footers/gray0_ctp_on_line.svg"/>
</div>

---

## ✅ Testing It's Working

| Check | Command | Expect |
|---|---|---|
| Model architecture builds | `python src\model.py` | Full Keras model summary, no errors |
| Model file exists after training | — | `models\best_model.keras` present |
| Reference fingerprint exists | — | `models\reference_stats.json` present |
| Metrics are reasonable | `python src\evaluate.py` | Precision & recall generally > 0.90 on synthetic data, for BOTH classes |
| Real good images pass | Upload a `data/train/good` or `data/test/good` image in Streamlit | Shows ✅ GOOD |
| Real defective images are caught | Upload a `data/test/defective` image in Streamlit | Shows ❌ DEFECTIVE |
| Unrelated images are rejected | Upload a logo or random unrelated photo | Shows ❌ DEFECTIVE with an anomaly-score note |
| API responds | `curl -X POST "http://127.0.0.1:8000/predict" -F "file=@data/test/defective/defective_0000.png"` | JSON with `predicted_class`, `confidence`, `is_anomalous`, `ood_score` |

---

## 📸 Using Your Own Real Dataset

1. Collect real product photos.
2. Sort them into `data/train/good/`, `data/train/defective/`, `data/test/good/`, `data/test/defective/`.
3. Re-run Steps 6–8 above — no code changes required. The anomaly-detection threshold (Step 7) will automatically recalibrate to how visually varied *your* product photos naturally are.

Public datasets to practice on: **MVTec AD**, Kaggle's **"Casting Product Image Data for Quality Inspection"**, Kaggle's **"Surface Crack Detection"**.

---

## 🎓 How It Works (Interview-Ready Explanation)

- **Transfer learning:** MobileNetV2 was pretrained on 1.4M general images; we reuse its learned visual features instead of training from zero, which needs far less data.
- **Two-phase training:** Phase 1 trains only the new classification head with the pretrained base frozen, so the head learns to use existing features without corrupting them via noisy gradients from its own random initialization. Phase 2 unfreezes the later base layers and fine-tunes with a much smaller learning rate, gently adapting features to the specific product. Skipping straight to fine-tuning is a common mistake that can look like learning (rising training accuracy) while actually destroying generalization.
- **Augmentation on train only:** Random rotation/flip/zoom/brightness during training builds robustness to camera angle and lighting; test data stays clean so metrics reflect real-world performance.
- **Precision vs. recall:** Precision = "of what we flagged as defective, how much really was?" Recall = "of all real defects, how many did we catch?" Both are tracked because in QA, missing a real defect is usually costlier than a false alarm.
- **Grad-CAM:** Highlights which pixels most influenced the decision, turning a black-box prediction into something a human inspector can verify and trust.
- **Anomaly / out-of-distribution guardrail:** A CNN can only choose between the classes it was trained on — shown a completely unrelated image, it has no way to say "I don't recognize this" and will force a guess. A second, independent check compares each input's basic visual statistics (average color, color variation, edge/texture density) against a fingerprint built from known-good training images, with a threshold **calibrated automatically from that same data** rather than a fixed guess. If an image deviates too far, it's flagged as defective regardless of the CNN's own prediction — deliberately biased toward caution, since a missed real defect is costlier than an unnecessary flag.

---

## 🔮 Roadmap / Future Improvements

- [ ] Multi-class defect typing (scratch vs. dent vs. crack, not just binary)
- [ ] Docker containerization for one-command deployment
- [ ] Edge deployment via TensorFlow Lite / ONNX
- [ ] Live webcam feed inference
- [ ] Model versioning & experiment tracking (MLflow / Weights & Biases)
- [ ] Covariance-aware (Mahalanobis) anomaly scoring instead of the current diagonal approximation

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

## 👤 Author

<div align="center">

### **Polanki Thrinath**

[![GitHub](https://img.shields.io/badge/GitHub-thrinathpolanki-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/thrinathpolanki)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-thrinathpolanki-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/thrinathpolanki)
[![Email](https://img.shields.io/badge/Email-polankithrinath%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:polankithrinath@gmail.com)

<br/>

*If this project helped you, consider giving it a ⭐ on GitHub!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=120&section=footer" width="100%"/>

</div>
