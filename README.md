# ⚡ DreamShaper Interactive Workbench

**A real-time generative prototyping tool exploring the trade-offs between inference latency and high-fidelity image synthesis.**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/rukmininazre/dreamshaper-workbench)
*(Click above to try the live demo running on CPU)*

---

## 💡 Project Overview
Built to mirror the commercial workflows of tools like **Adobe Firefly**, this project demonstrates how **Latent Consistency Models (LCM)** can be leveraged to create sub-second, interactive creative tools. 

By distilling the standard Stable Diffusion inference process from 50 steps down to just 6, this workbench achieves a **10x reduction in latency** (1.2s on T4 GPU) without sacrificing the high-frequency texture details required for professional design assets.

### Key Features
* **⚡ Sub-Second Inference:** Optimized pipeline achieving ~1.2s latency on consumer GPUs using LCM-LoRA.
* **🎨 Multi-Style Comparison Engine:** Parallelized batch generation that allows users to instantly visualize a single concept across 5 distinct aesthetic styles (Cinematic, 3D Render, Anime, etc.) for rapid creative iteration.
* **🛠️ Full-Stack Interactive UI:** A responsive **Gradio** frontend featuring real-time performance metrics, reproducible seed controls, and a "Sketch-to-Image" canvas powered by **ControlNet**.

---

## 📸 Performance & Visuals

### 1. The "Design Workbench" (Style Comparison)
*Demonstrating the ability to rapidly explore aesthetic variations of a single concept—critical for speeding up the ideation phase in professional workflows.*

![Floating Island Style Grid](dreamshaperall.jpg)
*Prompt: "Magical floating island in a light blue sky..." | Time: 7.6s (Total for 5 images)*

### 2. High-Fidelity Texture Generation
*Proving that speed does not compromise quality. The model successfully renders complex textures (frosting, light reflection) even at low step counts.*

![Single Cupcake Render](dreamshaper-single.jpg)
*Prompt: "A cupcake with lemon frosting" | Time: 1.28s | Steps: 6*

---

## 🛠️ Tech Stack & Engineering Decisions

* **Core Frameworks:** Python, PyTorch, Hugging Face Diffusers
* **Models:** * **Checkpoint:** `Lykon/dreamshaper-8` (Selected for superior photorealism over base SD 1.5)
    * **Speed Optimization:** `latent-consistency/lcm-lora-sdv1-5` (Enables 4-8 step inference)
    * **Control:** `lllyasviel/control_v11p_sd15_scribble` (Ensures strict geometric adherence to user input)
* **Interface:** Gradio (Custom CSS for responsive grid layouts)
* **Deployment:** Hugging Face Spaces (CPU/GPU agnostic architecture)

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/rukmininazre/dreamshaper-workbench.git](https://github.com/rukmininazre/dreamshaper-workbench.git)
    cd dreamshaper-workbench
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python app.py
    ```
    *Note: The application defaults to CUDA if available. To run on CPU, ensuring `torch_dtype=torch.float32` is handled automatically.*

