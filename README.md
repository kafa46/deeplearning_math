# 소프트웨어 꼰대 강의 - 딥러닝 수학 <a id='top'></a>

<p>
    안녕하세요? 소프트웨어 꼰대강의를 운영하고 있는 노기섭 교수입니다.
    이 저장소는 '딥러닝 수학'과 관련된 자료를 제공하고 있습니다. 딥러닝을 공부하다 보면 피해갈 수 없는 장벽을 만나게 됩니다. 바로 수학과 관련된 지식입니다. 개인적으로 딥러닝을 공부하면서 누군가 쉽게 풀어서 설명하는 자료가 있으면 좋겠다는 생각을 종종 했습니다. 제가 이해한 방식을 기반으로 딥러닝 관련된 수학 지식을 제공하고자 합니다.
</p>

꼰대 교수님 홈페이지: 바로가기 [(click me)](https://prof.acin.kr/)

궁금한 사항, 오류 등이 있으면 [kafa46@cju.ac.kr](mailto:kafa46@cju.ac.kr)로 연락주시면 감사하겠습니다.


## 딥러닝 수학 시리즈 구성
- [1장. 확률( Probability)](#prob)
- [2장. 선형대수 (Linear Algebra)](#linear)
- [3장. 미분 (Differentiation)](#diff)
- [Bonus. 집합론 (곱집합)](#set)
<hr>

### Chapter 1. Probability <a id='prob'></a>

|분야|주제|Youtube|Codes|PPT|
|---|---|---|---|---|
|확률|01. Random Variables (확률변수)... 이게 뭔가요?|[click](https://youtu.be/iTxTGBOhzCA)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/01_random_variables.pdf)|
|확률|02. Expected Value (기댓값)... 이게 뭔가요?|[click](https://youtu.be/nvHyIScyQxs)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/02.%20expected_value.pdf)|
|확률|03. 딥러닝과 확률 분포의 관계는 무엇인가요?|[click](https://youtu.be/qpbIKg21mvI)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/03_probability_distribution_and_deeplearning.pdf)|
|확률|04. 파라미터와 딥러닝|[click](https://youtu.be/PobNLp279-w)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/04_parameters_in_deep_learning.pdf)|
|확률|05. odds, logit, sigmoid, and softmax 개념설명|[click](https://youtu.be/V0uyiu6X4Zs)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/05_odds_logit_sigmoid_and_softmax.pdf)|
|확률|06. 샘플링 표현에 대한 이해와 몬테 카를로 근사|[click](https://youtu.be/nw_tVBCw0Z8)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/06_sampling_representation_and_monte_carlo_approximation.pdf)|
|확률|07. Stochastic Approach (확률적 접근법)|[click](https://youtu.be/LBT41oKsHWg)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/07_stochastic.pdf)|
|확률|08. Stochastic 실습 및 분석|[click](https://youtu.be/k6xog4ZNnT0)|[codes](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/07_stochastic.py)|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/07_stochastic.pdf)|
|확률|09. 제발 Stochastic Process가 뭔지 설명 좀 해주세요!|[click](https://youtu.be/HtJ-q8tc5qQ)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/07_stochastic.pdf)|
|확률|10. Stochastic Gradient Descent에 왜 "Stochastic"라는 단어가 붙은 건가?|[click](https://youtu.be/DEQhCJ0nav4)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/07_stochastic.pdf)|
|확률|11. Bayesian vs. Frequentist|[click](https://youtu.be/Kmw1pCsAqfM)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/08_01_Bayesian_vs_frequentist.pdf)|
|확률|12. Bayes theorem_예제 풀이(당구공 굴리기)|[click](https://youtu.be/xdMor6957E0)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/08_02_Bayes_theorem_example_and_simulation.pdf)|
|확률|13. Bayes theorem_예제 시뮬레이션(파이썬)|[click](https://youtu.be/7nyj0DvUluI)|[codes](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/08_02_Bayes_theorem_simulation.py)|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/08_02_Bayes_theorem_example_and_simulation.pdf)|
|확률|14. Bayes theorem 딥러닝에 적용하여 해석하기|[click](https://youtu.be/YvWqPQhliaI)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/08_03_Bayes_theorem_applying_into_deeplearning.pdf)|
|확률|15. Maximum Likelihood Estimation (MLE) 완벽히 파헤치기 (deep dive)!|[click](https://youtu.be/vzNRLY_hLlM)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/09_Maximum_Likelihood_Estimation_(MLE).pdf)|
|확률|16. Maximum A Posterior (MAP) 완벽히 파헤치기 (deep dive)!|[click](https://youtu.be/H342QehYSqo)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/10_Maximum_A_Posterior_(MAP).pdf)|
|확률|17. Bayesian Neural Networks 깊은 이해 (Bayesian Inference)|[click](https://youtu.be/126JfX_kJTU)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/01_probability_theory/11_bayesian_neural_networks.pdf)|

[맨위로 이동](#top)
<hr>

### Chapter 2. Linear Algebra <a id='linear'></a>

|분야|주제|Youtube|Codes|PPT|
|---|---|---|---|---|
|선형대수|01. 딥러닝에서의 선형대수! 오리엔테이션 (동기부여)|[click](https://youtu.be/Si2QxZEz8Po)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/01_orientation_and_motivations.pdf)|
|선형대수|02. Matrix(행렬) 역사, 정의,  표기법, 용어 정리|[click](https://youtu.be/ToWPEh1neCY)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/02_intro_definition_notation.pdf)|
|선형대수|03. Matrix(행렬) 기본연산(행렬의 덧셈과 곱셈), 가우스-조던 소거법|[click](https://youtu.be/hj3PBuvW5TA)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/03_matrix_operation.pdf)|
|선형대수|04. Determinant(행렬식), Inverse Matrix(역행렬) 깊게 이해하기|[click](https://youtu.be/rfeEx1saFVE)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/04_determinant_and_inverse_matrix.pdf)|
|선형대수|05. 벡터 소개 (역사, 정의 및 표현법, 종류, 기저, Norm)|[click](https://youtu.be/MKzejgqrW6Q)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/05_vector_introduction.pdf)|
|선형대수|06. 벡터의 덧셈/뺄셈, 스칼라배|[click](https://youtu.be/S4B4CKURlEs)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/06_vector_addtion_scalar_mulitplication.pdf)|
|선형대수|07. 벡터의 곱셈 (내적, Cosine Similarity, Cross곱)|[click](https://youtu.be/VophYxpve0k)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/07_vector_product.pdf)|
|선형대수|08. 벡터 공간 (vector space) 및  부분 공간 (vector subspace)|[click](https://youtu.be/6EjSnqXGwHQ)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/08_vector_vector_space.pdf)|
|선형대수|09. 선형결합(linear combination) 및 생성(span)|[click](https://youtu.be/HTXay7LuSlY)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/09_vector_linear_combination_and_span.pdf)|
|선형대수|10. 선형변환(liear transformation), 벡터와 행렬의 만남|[click](https://youtu.be/dlFRj45ckXE)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/10_vector_join_vector_and_matrix_linear_transformation.pdf)|
|선형대수|11. 행렬을 이용한 선형변환|[click](https://youtu.be/1E02Md0o-Vc)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/11_vector_transformation_with_matrix.pdf)|
|선형대수|12. 행렬과 행렬을 곱한다는 의미 (선형변환의 합성)|[click](https://youtu.be/EXMWzuZHbfo)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/12_vector_matrix_composition.pdf)|
|선형대수|13. 그림으로 이해하는 행렬식(determinant)|[click](https://youtu.be/6qdZygiry_E)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/13_vector_determinant_with_figures.pdf)|
|선형대수|14. 벡터에서 텐서로 - 텐서의 깊은 이해|[click](https://youtu.be/pPIFauuiwEU)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/14_from_vector_to_tensor.pdf)|
|선형대수|15. 선형 시스템(공간)간의 해석(linear empathy)|[click](https://youtu.be/JQ1k8axkbFY)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/15_linear_empathy.pdf)|
|선형대수|16. 고윳값(eigenvalue)과 고유벡터(eigenvector) 이해하기|[click](https://youtu.be/FDEIHuBanwM)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/16_eigenvector_eigenvalue.pdf)|
|선형대수|17. 차원 축소 및 확장 (dimension reduction & expansion)|[click](https://youtu.be/uAVlPBC8TGE)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/17_dimension_reduction_expansion.pdf)|
|선형대수|18. 행렬에서의 랭크 (rank in matrix)|[click](https://youtu.be/ORSP-Rd2NcU)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/18_rank_in_matrix.pdf)|
|선형대수|19. 선형변환 요약, 정보 압축 및 팽창|[click](https://youtu.be/2vgpAgsqSEc)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/19_information_compression_expansion.pdf)|
|선형대수|20. 선형대수를 마무리하며, 인사말 및 감사인사 (adjourning)|[click](https://youtu.be/3uDd-xaipoU)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/02_linear_algebra/20_adjourning.pdf)|

[맨위로 이동](#top)

<hr>

### Chapter 3. Differentiantion <a id='diff'></a>


|분야|주제|Youtube|Codes|PPT|
|---|---|---|---|---|
|미분|01. 미분 시리즈를 시작하며,<br>오리엔테이션 (Orientation)|[click](https://youtu.be/j1D1jY71Wjg)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/03_differentiation/01_orientation.pdf)|
|미분|02. 딥러닝은 어떻게 데이터로부터 지식을 배우는가?|[click](https://youtu.be/wHRAvJehL20)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/03_differentiation/02_learning_in_deeplearning.pdf)|
|미분|03. 미분의 원리 - 한번은 짚고 넘어가야 할 내용|[click](https://youtu.be/C8yzd1UOEq4)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/03_differentiation/03_principal_of_differentiation.pdf)|
|미분|04. 딥러닝에서 편미분 개념과 연산|[click](https://youtu.be/_8chLG-JFDo)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/03_differentiation/04_partial_differentiation.pdf)|
|미분|05. 경사하강 알고리즘의 개념, 해석, 학습|[click](https://youtu.be/QU99KdtT81I)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/03_differentiation/05_gradient_descent.pdf)|
|미분|06. 연쇄법칙 (chain rule) 개념과 및 연산|[click](https://youtu.be/4aWZePsJ0Ro)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/03_differentiation/06_chain_rule.pdf)|
|미분|07-01. 선형시스템에서의 편미분|[click](https://youtu.be/o3IeCWtHxG4)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/03_differentiation/07_matrix_differentiation.pdf)|
|미분|07-02. 선형시스템에서의 편미분 (실습 with toy example)|[click](https://youtu.be/_cAIzjWQ3Bg)|[codes](https://github.com/kafa46/deeplearning_math/blob/master/03_differentiation/07_matrix_differentiation_excercise_toy_example.py)|[link](https://github.com/kafa46/deeplearning_math/blob/master/03_differentiation/07_matrix_differentiation_excercise_slide.pdf)|
|미분|08. 역전파 학습 (back-propagation) 및 Computation Graph|[click](https://youtu.be/T4Im-qIl0Xc)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/03_differentiation/08_back_propagation.pdf)|
|미분|09. 활성함수 간단 소개, 기울기 소실의 근본 원인과 대책|준비중|없음|준비중|
|미분|10. 딥러닝 수학 시리즈 전체를 마무리하며, 감사인사 (adjourning)|준비중|없음|준비중|

[맨위로 이동](#top)
<hr>

### Bonus. Number Set <a id='set'></a>

|분야|주제|Youtube|Codes|PPT|
|---|---|---|---|---|
|집합|$R_{2,3}$ vs. $R^{2\times3}$ 차이가 도대체 뭐야?|[click](https://youtu.be/m7dSzu-G_Mk)|없음|[link](https://github.com/kafa46/deeplearning_math/blob/master/04_set_theory/01_intepretation_of_number_set.pdf)|

[맨위로 이동](#top)
