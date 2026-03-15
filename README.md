# LLM 미세 튜닝, 핵심만 빠르게!

<p align="center">
  <img src="https://tensorflow.blog/wp-content/uploads/2026/03/cover_llmebafb8ec84b8ed8a9ceb8b9ded95b5ec8baceba78cebb9a0eba5b4eab28c.jpg" width="400"/>
  <br>
  <strong>
    <a href="https://product.kyobobook.co.kr/detail/S000219441762">교보문고</a> | 
    <a href="https://www.yes24.com/product/goods/182295904">Yes24</a> | 
    <a href="https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=388054972">알라딘</a>
  </strong>
</p>

## 설정

- [FAQ](https://github.com/rickiepark/fine-tuning-llm/blob/main/FAQ.md)
- [부록 A - GPU 포드 설정하기](https://github.com/rickiepark/fine-tuning-llm/blob/main/AppendixA.md)

## 구글 코랩

코랩을 사용하면 **깃허브에서 노트북을 쉽게 불러와서** 구글이 제공하는 **GPU**로 실행할 수 있습니다. 이를 위해서는 본인 구글 계정에 로그인되어 있어야 합니다.

아래 링크를 사용해 이미 각 장의 노트북을 실행할 수 있습니다.

- [0장 - 대규모 언어 모델 미세 튜닝 레시피](https://colab.research.google.com/github/rickiepark/fine-tuning-llm/blob/main/Chapter0.ipynb)
- [1장 - 대규모 언어 모델](https://colab.research.google.com/github/rickiepark/fine-tuning-llm/blob/main/Chapter1.ipynb)
- [2장 - 양자화된 모델 로드하기](https://colab.research.google.com/github/rickiepark/fine-tuning-llm/blob/main/Chapter2.ipynb)
- [3장 - LoRA](https://colab.research.google.com/github/rickiepark/fine-tuning-llm/blob/main/Chapter3.ipynb)
- [4장 - 데이터셋 포매팅](https://colab.research.google.com/github/rickiepark/fine-tuning-llm/blob/main/Chapter4.ipynb)
- [5장 - `SFTTrainer`로 미세 튜닝하기](https://colab.research.google.com/github/rickiepark/fine-tuning-llm/blob/main/Chapter5.ipynb)
- [6장 - 로컬에 배포하기](https://colab.research.google.com/github/rickiepark/fine-tuning-llm/blob/main/Chapter6.ipynb)
- [부록 B - 데이터 타입의 내부 표현](https://colab.research.google.com/github/rickiepark/fine-tuning-llm/blob/main/AppendixB.ipynb)


## 서문

이 글을 읽고 있다면 대규모 언어 모델이 이제 거의 모든 곳에 있다는 말을 할 필요가 없겠죠? 

2022년 11월 ChatGPT가 출시된 이후로 이 분야의 발전 속도가 빨라져 따라가는 것이 거의 불가능하게 느껴집니다. 매일 새로운 기술, 새로운 모델, 또는 획기적인 발표가 있습니다. 분명히 흥미로운 시기이지만 동시에 압도적이고, 지치며, 때로는 좌절감을 느끼게 할 수도 있습니다.

"_이걸 어디서부터 배워야 하지?_"라는 질문이 생기는게 너무 당연하며, 혼자서 답을 찾기 어려운 질문입니다. 이 질문에 대한 답변으로 이 책을 썼습니다. 이 책은 안정성이 입증되었고 가까운 미래에도 미세 튜닝 과정의 핵심으로 남을 수 있는 개념들에 초점을 맞춥니다. 양자화, LoRA, 그리고 템플릿 포맷팅입니다.

이러한 개념들을 마스터하는 것은 현재 상황을 이해하는 데 중요하며, 앞으로의 발전을 따라갈 수 있는 능력도 갖추게 해줄 것입니다. 이 개념들은 언어 모델뿐만 아니라 다양한 대규모 모델을 훈련하거나 미세 튜닝하는 데에도 유용할 수 있습니다. 즉, 모든 데이터 과학자의 툴킷에 필요한 도구들입니다.

****
**이 책은 중급 수준의 책이므로, 그 내용을 최대한 활용하기 위해서는 확실하게 기초를 알고 있어야 합니다. 만약 트랜스포머, 어텐션, Adam 옵티마이저, 토큰, 임베딩, GPUs 익숙하지 않다면, 제가 쓴 초보자를 위한 도서인 'Deep Learning with PyTorch Step-by-Step'을 추천합니다.**
****

이 책은 허깅 페이스 생태계를 기반으로 합니다. 그 이유는 언어 모델이든 아니든 허깅 페이스가 딥러닝 모델 작업을 위한 사실상의 표준이기 때문입니다. 이 책에서 소개하는 개념(양자화, 어댑터, 템플릿)이 이 생태계 안에서 깔끔하게 구현되고 통합되어 있기 때문에 비교적 사용하기 쉽습니다. 하지만 효과적으로 설정하는 방법과 이런 설정이 실제로 내부에서 무엇을 하는지 이해해야 합니다. 이런 정보를 찾는 것은 쉽지 않습니다. 특히 GPU에서 LLM을 미세 튜닝할 때 이러한 기술들이 실제로 어떻게 함께 작동하는지 설명하는 포괄적인 자료가 부족합니다. 이 책으로 이 간극을 메우려고 합니다.

원래 이 책의 제목은 "퀵 가이드"(short guide)였지만 범위가 너무 넓어져서 결국 "핸즈온 가이드"(hands-on guide)로 제목을 바꿨습니다. 이 책은 많은 내용을 다루고 있으며 여러 분의 학습 여정에 도움이 되기를 진심으로 바랍니다. 책에는 재미있는 예제, 지어낸 인용구, 영화 대사들이 많이 들어 있습니다. 무엇보다도 학습은 즐거워야 한다고 믿기 때문입니다.

새로운 것을 배우고, 직접 시도해보고, 그것이 잘 작동하는 것을 지켜보는 것보다 더 멋진 일은 없지 않나요? 바로 여러분이 0장 "LLM 미세 튜닝 레시피"에서 할 일입니다. 이 장에서 LLM 미세 튜닝의 전체 과정(양자화, LoRA, 데이터셋 포맷팅, 훈련, 모델 쿼리)을 안내합니다. 다음으로 한 발 물러나서 1장에서 언어 모델, 트랜스포머, 어텐션 메커니즘, 그리고 다양한 유형의 미세 튜닝에 대해 간략히 논의할 것입니다.

다음 장인 2장부터 6장까지는 0장의 각 절에 해당하는 내용이 이어집니다. 2장 "양자화된 모델 로드하기"에서는 8비트 및 4비트 양자화와 BitsAndBytes 설정에 대해 더 자세히 논의할 것입니다. 3장 "LoRA"에서는 LoRA 어댑터의 역할과 사용법을 탐구합니다. PEFT 패키지를 사용하여 구성하는 방법과 훈련 중 수치적 안정성을 향상시키기 위해 (양자화된) 베이스 모델을 준비하는 방법을 설명합니다. 그런 다음 4장 "데이터셋 포맷팅"에서는 데이터 포맷팅, 채팅 템플릿, 그리고 토크나이저, 패딩, 패킹, 데이터 콜레이터의 역할에 초점을 맞출 것입니다.

그다음 5장 "SFTTrainer로 미세 튜닝 하기"에서는 개인용 GPU의 성능을 최대화하고 모델을 효과적으로 미세 튜닝하기 위한 다양한 설정을 탐색합니다. 또한 다른 어텐션 메커니즘 구현인 플래시 어텐션과 파이토치의 SDPA에 대해 논의하고, 속도와 메모리 요구 사항을 비교합니다. 6장 "로컬에 배포하기"는 엔지니어링에 초점을 맞춘 장입니다. 이 장은 미세 튜닝된 모델을 GGUF 형식으로 변환하는 세부 사항과 대안을 다룹니다. 또한 Ollama 또는 llama.cpp를 사용하여 모델을 서빙하는 방법을 설명합니다.

모든 학습 여정에는 어려움과 함정이 있습니다. 이 책에서 등장하는 경고와 예외가 이에 해댱합니다. 마지막 장인 -1장 "문제 해결"은 사용자가 겪을 수 있는 일반적인 오류를 이해하고 해결하는 데 도움이 되는 설명을 제공합니다.

마지막으로 두 개의 부록이 있습니다. 첫 번째는 "부록 A GPU 포드 설정"으로 클라우드 서비스(runpod.io)를 사용하여 GPU 기반 주피터 노트북을 실행하기 위한 단계별 튜토리얼을 제공합니다. 두 번째는 "부록 B 데이터 타입의 내부 표현"으로 각 데이터 타입의 장단점을 더 자세히 이해하고 싶은 독자들을 위해 정수와 부동소수점 숫자가 비트로 내부적으로 어떻게 표현되는지에 대해 소개합니다.

이게 전부입니다! 즐거운 여행되세요!
