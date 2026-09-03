<src="https://github.com/user-attachments/assets/c237c6a4-7d28-4196-99de-818d1b45f1a4" />


# ♻️ 삼성교육 16기 AI Challenge: 재활용품 이미지 기반 질의응답 모델 개발

> **"이미지로 묻고, AI로 답하다 ㅡ 이미지 기반 질의응답 모델 개발 AI 챌린지"**
> 삼성교육 16기 1회차 AI 챌린지 참가 프로젝트입니다.

## 🏆 프로젝트 성과

* **최종 순위:** 전체 954명 중 **22위** 달성
* **최종 정확도(Accuracy):** **0.93732**
* **참가자:** 대전_6반_이창엽
* ## 🏆 프로젝트 성과
* **최종 순위:** 전체 954명 중 **22위** 달성
* **최종 정확도(Accuracy):** **0.93732**
* **참가자:** 대전_6반_이창엽

<br>

<div align="center">
  <img src="src="https://github.com/user-attachments/assets/f76feb86-1ae2-4f24-8d87-a5cfdb371cfe" alt="최종 순위 및 점수" width="100%">

  <br><br>
  <"src="https://github.com/user-attachments/assets/9c93fdb5-ed3c-4729-8281-ede4c7f2e15b"  alt="전체 참가 인원" width="80%">
  <img src="https://github.com/user-attachments/assets/7ffac96c-bc48-47b5-95e5-018d25d8766b" alt="전체 참가 인원" width="80%">
</div>

## 📖 대회 개요

이미지 속 세상을 '읽고, 이해하고, 답하는' 인공지능인 **VQA(Visual Question Answering)** 모델은 인간의 시각적 사고에 가장 가까운 형태의 AI 기술입니다. 본 프로젝트는 재활용품 이미지 데이터를 활용하여 VQA 모델을 개발하고, 주어진 이미지와 자연어 질문을 바탕으로 4지선다(a, b, c, d) 중 알맞은 정답을 추론하는 시스템을 구축했습니다.



## 🤖 모델 선정

본 대회에서는 탁월한 한국어 이해 능력과 이미지 인식 성능을 갖춘 **Qwen-VL 시리즈**를 핵심 모델로 선정했습니다.

* **Qwen2.5-VL-7B-Instruct:** 로컬 GPU(RTX 5060 Ti) 환경에서 LoRA 파인튜닝을 적용하기 가장 적합한 크기와 성능의 밸런스를 갖춘 주력 모델로 활용했습니다.
* **Qwen3.5-27B:** 압도적인 제로샷 성능을 활용하여, 파인튜닝된 7B 모델이 헷갈려하는 고난도 문항을 방어하는 '전문가(Expert) 모델'로 앙상블에 활용했습니다.

---

## 🛠️ 기술 스택 및 라이브러리

* **Language:** Python
* **Framework:** PyTorch (v2.11.0, CUDA 12.8)
* **AI/ML Libraries:**
* `transformers` (HuggingFace 최신 아키텍처 로드)
* `peft` (LoRA 파인튜닝)
* `accelerate`, `bitsandbytes` (메모리 최적화 및 양자화)
* `qwen-vl-utils` (이미지 프로세싱)


* **Data Processing:** `pandas`, `numpy`, `Pillow`, `tqdm`
* **Hardware:** Windows Desktop, NVIDIA RTX 5060 Ti

---

## 🚀 성능 향상을 위한 핵심 전략

* **프롬프트 엔지니어링 (Prompt Engineering):** 모델이 질문과 선지를 명확히 구분할 수 있도록 프롬프트를 완전한 한국어 지시문으로 구조화하고, 출력 형식을 강제하는 강력한 System Prompt를 적용했습니다.
* **메모리 최적화 파인튜닝 (PEFT/LoRA):** 제한된 VRAM 환경에서 7B 모델을 학습시키기 위해 LoRA 기법을 적용했습니다. 타겟 모듈만 효율적으로 학습시키고, `torch.bfloat16` 데이터 타입과 Gradient Checkpointing으로 OOM을 방지했습니다.
* **추론 최적화 (Test Time Augmentation, TTA):** 객관식 선지 순서에 따른 모델의 편향을 줄이기 위해, 선지 순서를 4번 순환하여 예측 확률을 구한 뒤 평균을 내는 방식을 적용했습니다.
* **고도화된 앙상블 (Smart Routing Ensemble):**
* **Soft Voting:** 다양한 조건으로 학습된 모델들의 로짓(Logits)을 Softmax로 변환 후 최적의 가중치로 결합했습니다.
* **Confidence-Gated Routing:** 베이스 앙상블 모델의 확신도가 `0.70` 이하로 낮을 때, 대형 모델(27B)이 `0.85` 이상의 강한 확신을 보인다면 정답을 대형 모델의 예측값으로 교체하는 자체 스마트 라우팅 로직을 구현하여 정확도를 극대화했습니다.



---

## ⚙️ 실행 방법

### 1. 환경 설정

`requirements.txt`를 기반으로 필수 패키지를 설치합니다.

```bash
pip install -r requirements.txt

```

### 2. 프로젝트 파이프라인

제공된 주피터 노트북 파일(`baseline_desktop_final.ipynb`)을 순차적으로 실행합니다.

1. **데이터 준비 및 모델 초기화:** 데이터셋 로드 및 `Qwen2.5-VL-7B-Instruct` 준비
2. **모델 학습 (Fine-Tuning):** LoRA 기반 파라미터 효율적 튜닝 진행
3. **검증 및 추론 (Inference):** TTA를 적용하여 단일 모델 결과 `.npy` 및 `submission.csv` 추출
4. **스마트 라우팅 앙상블 (Ensemble):** 다중 모델 확률값 Soft Voting 및 확신도 기반 라우팅 적용하여 최종 제출 파일 생성
