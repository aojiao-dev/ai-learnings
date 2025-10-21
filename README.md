# 🚀 ML & LLM Study Plan

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?logo=tensorflow&logoColor=white)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

This repository documents my personal learning journey as I transition from a **traditional search engineer** to an **AI/ML engineer**.  
It includes study notes, code snippets, and mini-projects that build toward developing and fine-tuning domain-specific LLMs.

---

## 🧩 Learning Milestones

### 1. Review ML Fundamentals
Andrew Ng’s *Machine Learning Specialization* is my starting point.  
It covers essential concepts like:
- Linear & logistic regression  
- Loss functions and gradient descent  
- Supervised vs. unsupervised learning  

It also introduces more advanced topics such as **neural networks**, **decision trees**, **clustering**, **recommender systems**, and **reinforcement learning**.

---

### 2. Review DL Fundamentals
Andrew Ng’s *Deep Learning Specialization* focuses on:
- **Recurrent Neural Networks (RNNs)**
- **Convolutional Neural Networks (CNNs)**
- **Sequence models and Transformers** (which modern LLMs are based on)

---

### 3. Cross-check Understanding
To reinforce what I’ve learned, I’ll watch and follow along with other experts who explain neural networks and LLMs in concise, intuitive ways.  
If a topic doesn’t make sense, it means I need to revisit the fundamentals.

**Resources:**
- 🧠 3Blue1Brown – [Deep Learning](https://www.youtube.com/watch?v=aircAruvnKk)  
- 💻 Andrej Karpathy – [Zero to Hero](https://karpathy.ai/zero-to-hero.html)  
- 🧩 OpenAI Blog – [Spinning Up](https://spinningup.openai.com/en/latest/user/introduction.html)  
- 📘 Brandon Rohrer – [Transformers from Scratch](https://brandonrohrer.com/transformers.html)

---

### 4. Hands-on Coding
The goal is to **build from scratch** for deeper intuition — not just use pre-built frameworks.

#### Milestones:
- **NN from Scratch:** Build simple MLP, RNN, LSTM, and GRU models for text generation.
- **LLM from Scratch:** Build a minimal transformer-based model and pre-train it on small text datasets (e.g., Common Crawl subsets).  
  > Inspired by Stanford’s CS336, but without going all the way down to tokenizer training.
- **LLM Post-training:**  
  - Fine-tune an open-source GPT-OSS model via **SFT** to act as a shopping Q&A agent with tool-use abilities.  
  - Apply **RLHF** (via DPO or PPO) to align the model’s tone and helpfulness to user preferences.  
  - Build an **E2E demo**: GPT-4 handles query classification and routes shopping-related questions to the fine-tuned model.

#### Out of Scope
We will not implement the following architectures:
- **CNNs:** use convolutional filters, mainly for image processing, not text generation.
- **Autoencoders:** compress data into compact latent representations.
- **GNNs:** model relationships in graph structures, used in recommendation and social networks.
- **GANs:** generative models with a competing generator and discriminator, for images/audios.
- **Diffusion Models:** power modern multimodal generation (e.g., Google’s NanoBanana).
- **Decision Tree:** a supervised algorithm can handle binary, multi-class, and regression tasks.
- **Random Forest:** many decision trees trained on subset of data and then vote.
- **XGBoost:** many decision trees but each tree tries to correct errors of previou sone.
- **Unsupervised Learnings:** clustering (k-means), anomaly detection, etc.
- **Recommendation:** collaborative filtering, nn based approach (search search, ranking, re-ranking).
---

## 🧰 Useful Commands

```bash
uv init
uv add tensorflow
uv add torch
uv add torchvision
uv add matplotlib
