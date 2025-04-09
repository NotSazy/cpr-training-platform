# 🫀 CPR Training Platform

A web-based platform that uses AI and biomechanics to help assess and improve CPR training quality. Built with React and enhanced by computer vision techniques, this tool enables practitioners and researchers to upload videos, analyze body posture, and get feedback on CPR performance metrics like compression depth, force alignment, and fatigue potential.

---

## 📌 Features

- 📤 Upload CPR training video (OpenPose-supported)
- 📈 Visual & numerical feedback on compression quality
- 🧍 User-specific calibration (gender, height, weight)
- 🎥 Camera calibration (automatic/manual)
- ⚕️ Posture assessment using body joint angles
- 🧠 Biomechanics-based analysis for fatigue detection
- 📊 Dashboard-ready architecture (coming soon)

---

## 🚀 Tech Stack

- **Frontend:** React, Tailwind CSS, shadcn/ui
- **Computer Vision:** OpenPose integration (CSV/JSON input)
- **Data Processing:** Inverse dynamics and joint angle estimation
- **Hosting:** GitHub Pages / Codespaces

---

## 🛠️ How It Works

1. **Upload Video** – Frontend accepts training footage
2. **Preprocessing** – OpenPose extracts keypoints and stores them in CSV
3. **Biomechanical Analysis** – Our custom script estimates joint angles and forces
4. **Result Generation** – Output is visualized with professional styling
5. **Feedback** – Practitioner gets real-time metrics to improve posture and reduce fatigue

---

## 📥 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/NotSazy/cpr-training-platform.git
cd cpr-training-platform

# Install dependencies and start dev server
npm install
npm start
