# ESS 배터리 수명 예측

대규모 에너지 저장 장치(BESS, Battery Energy Storage System)의 신뢰성 높은 운영과 화재·조기 열화 방지를 위해, 셀 설치 초기 100 사이클의 충·방전 데이터만으로 최종 수명(Cycle Life to 80% EOL)을 조기에 예측하는 데이터 기반 머신러닝 파이프라인을 구축합니다.  
본 프로젝트에서는 상용 LFP/흑연 18650 원통형 배터리의 전기화학적 열화 신호인 전압 차분 곡선 $\Delta Q_{100-10}(V)$, 내부 저항($IR$), 열 스트레스($T_{max}$)를 정량화한 후, **Batch 1을 학습 데이터(CV 및 Hold-out)**로 사용하고 **Batch 2를 테스트 데이터**로 평가하여 원논문(Nature Energy 2019)의 성능과 비교 분석하고 배치 간 일반화 저하(Batch Effect) 메커니즘을 규명합니다.

## 프로젝트 개요

- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12, 46개 셀)
- 평가 데이터 : Batch 2 (2018-02-20, 39개 셀)
- 태스크 : Regression (Cycle Life 예측, 타겟: $\mathbf{y = \log_{10}(\text{Cycle Life})}$) 

## 파일 구조 (sample)

```
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
├── results/
│   ├── model_performance.csv
│   └── error_analysis.csv
├── requirements.txt
└── README.md
```

- `data/` : MIT-Stanford 배터리 데이터셋 설명 및 배치 구성 안내 문서
- `notebooks/` :
  - `01_EDA.ipynb` : 수명 분포, 열화 곡선, $\Delta Q(V)$, C-rate 및 다중공선성 분석
  - `02_feature_engineering.ipynb` : $\Delta Q(V)$ 분산, 선형 피팅 슬로프, 환경 변수 등 피처 엔지니어링 파이프라인
  - `03_modeling.ipynb` : 모델 학습, 5-Fold CV 및 Hold-out 검증, Batch 2 테스트 평가, 오차 분석
- `src/` :
  - `preprocess.py` : 원시 데이터 로드 및 정제 시계열 추출 모듈
  - `features.py` : Level 1, Level 2, Level 3 피처 생성 모듈
  - `train.py` : 모델 학습, 5-Fold CV, 프로토콜 기반 Hold-out 분할 및 성능 평가 실행 모듈
- `results/` :
  - `model_performance.csv` : 전 모델 성능 지표 및 Gap 결과 테이블
  - `error_analysis.csv` : Batch 2 셀별 예측치 및 잔차 상세 분석 테이블

## 환경 설정 (sample)

```bash
git clone https://github.com/팀명/ess-battery-project
cd ess-battery-project
pip install -r requirements.txt

# 전체 모델 학습 및 평가 파이프라인 실행
python src/train.py
```

## EDA

- Cycle Life 분포
  - 분포 형태 및 장단수명 비율 요약
    - 전체 129개 유효 셀의 평균 수명은 **833.7 사이클**(중앙값 828.0)이며, 최단 392회부터 최장 1,935회까지 넓게 분포함.
    - **Batch 1 (Train)**: 평균 844.7회 (중앙값 858.5회, IQR 211.0회), 513 ~ 1,189회 범위의 대칭형 정규분포를 보임 (500회 미만 단수명 셀 0%).
    - **Batch 2 (Test)**: 평균 565.7회 (중앙값 472.0회, IQR 69.0회), 392 ~ 1,064회 범위이며 **전체 셀의 66.7%(26/39)가 400~500회 구간에 극단적으로 밀집**된 강한 우측 꼬리 분포.
    - 장수명(>1,000회) 비율: 전체 26.6% (Batch 1: 17.4%, Batch 2: 7.7%, Batch 3: 56.8%).
    - 단수명(<500회) 비율: 전체 25.8% (Batch 1: 0%, Batch 2: 66.7%).
  - 핵심 발견 : Batch 1에는 500회 미만 셀이 전혀 없는 반면 Batch 2는 66.7%가 400회대에 편중되어 있어, Batch 1으로만 학습 시 Batch 2 단수명 셀에 대한 외삽(Extrapolation) 및 배치 간 일반화 저하(Batch Effect) 발생 위험이 매우 큼.

- 열화 곡선 분석
  - 장수명 vs 단수명 셀의 열화 속도 차이
    - 초기 100 사이클 동안은 **전체 셀의 81.4%에서 오히려 방전 용량이 초기보다 미세하게 증가(+0.2% 중앙값)하거나 유지**되는 "초기 용량 불변성(Silent Degradation)"을 보임.
    - 초기 방전 용량 $Q_d$ 스칼라 값만으로는 장수명 셀과 단수명 셀을 전혀 구분할 수 없음 ($r = 0.27$).
  - Knee point 존재 여부 및 발생 시점
    - 모든 셀이 최종 수명(80% EOL, 0.88 Ah)의 **약 72% ~ 78% 시점에서 일관되게 급격한 열화 가속 변곡점(Knee Point)**을 맞이함.
    - 단수명 셀(`b2_c19`): 300 사이클 (76.5% 시점)에서 조기 발생.
    - 중간수명 셀(`b1_c0`): 857 사이클 (72.0% 시점)에서 발생.
    - 장수명 셀(`b3_c38`): 1,520 사이클 (78.6% 시점)까지 안정적으로 버틴 후 Knee Point 발생.
  - 핵심 발견 : 초기 100 사이클 동안 겉보기 방전 용량은 침묵하지만 내부에서는 이미 음극 활물질 손실(LAM)이 진행되어 전압 곡선을 왜곡시키고 있으므로, Knee Point의 조기 발생을 예측하기 위해서는 방전 용량이 아닌 전압 차분 곡선 $\Delta Q(V)$ 분석이 필수적임.

- ΔQ(V) 곡선 분석
  - Cycle 100 - Cycle 10 차이 곡선 형태
    - 사이클 100과 사이클 10의 전압별 방전 용량 곡선 차분: $\Delta Q_{100-10}(V) = Q_{100}(V) - Q_{10}(V)$
  - 장단수명 셀 간 ΔQ 형태 비교
    - 장수명 셀: 10사이클과 100사이클 간 전압 곡선 변화가 거의 없어 $\Delta Q \approx 0$에 수직으로 밀집됨.
    - 단수명 셀: 2.5V ~ 3.3V 구간에서 음(-)의 방향으로 크게 꺼지며 넓은 면적과 큰 변동성을 보임.
  - 핵심 발견 : 파생 피처 $\log_{10}(\mathrm{Var}(\Delta Q_{100-10}(V)))$ 하나만으로 수명($\log_{10}(\text{Cycle Life})$)과의 피어슨 상관계수가 **$\rho = -0.886$**에 달하여, 초기 사이클에서 배터리 수명을 결정짓는 압도적인 핵심 물리화학적 특징임을 확인.

- 충전 속도(C-rate)와 수명의 관계
  - 충전 프로토콜별 평균 수명 비교 결과
    - 1단계 급속 충전 전류($C_1$) 단독으로는 수명 설명력이 미미함 ($r = -0.08$). 6C 초고속 충전이라도 전환 SOC가 30%로 낮으면, 4.8C로 80%까지 충전하는 셀보다 가혹도가 낮음.
    - 실험 일시 중단(pause/rest)으로 인한 3,933분 이상치를 정제한 후 실제 충전 소요 시간(`chargetime`, 8.98분 ~ 13.38분)을 분석한 결과 수명과 유의미한 양의 상관성($r = +0.31$) 확인 (충전 시간이 10분 미만인 초급속 충전 셀들은 500사이클 미만으로 조기 고장).
  - 핵심 발견 : 2단계 충전 프로토콜($C_1, \text{SOC}_1, C_2$)의 복합 작용으로 결정되는 실제 충전 시간과 표면 최고 온도($T_{max}$) 등 열 스트레스가 결합될 때 수명 열화 가혹도를 정확하게 반영함.

- (추가 확인한 내용 작성)
  - 다중공선성(Multicollinearity) 분석
    - 최고 온도($T_{max}$)와 평균 온도($T_{avg}$) 간의 피어슨 상관계수가 **0.96**에 달함.
    - $\Delta Q(V)$의 요약 통계량인 `log_var_dq`, `min_dq`, `mean_dq` 상호 간 상관계수도 0.85 이상으로 매우 높음.
    - 따라서 단순 OLS 선형 회귀 적용 시 회귀계수 분산이 발산하고 수치적으로 불안정해지므로, **$L_1/L_2$ 정규화 패널티를 부여하는 Elastic Net 또는 트리 앙상블 기법 적용이 필수적**임.

## Modeling

### 피처 엔지니어링 전략

EDA에서 규명된 물리·전기화학적 열화 신호를 체계화하여 3단계 계층형 피처 세트를 구축:
- **Level 1 (Single Feature - Variance Model)** :
  - 피처 (1개): $\log_{10}(\mathrm{Var}(\Delta Q_{100-10}(V)))$
  - 근거: 초기 100사이클 $\Delta Q(V)$ 분산 하나만으로 상관계수 -0.89를 기록한 가장 직관적이고 해석력이 뛰어난 단일 물리 피처.
- **Level 2 (Discharge Feature Model)** :
  - 피처 (9개): $\Delta Q(V)$ 통계량(`log_var_dq`, `min_dq`, `mean_dq`, `skew_dq`, `kurt_dq`) + 초기 방전용량 감쇠 추세(`slope_qd`, `intercept_qd`, `qd_100_2`, `qd_2`)
  - 근거: 전압 왜곡의 고차 모멘트(왜도, 첨도) 및 100 사이클간의 미세 용량 감쇠율을 결합하여 방전 열화 프로파일 완성.
- **Level 3 (Full Feature Model)** :
  - 피처 (13개): Level 2 피처군 + 순수 충전 소요 시간(`chargetime`) + 열 스트레스(`Tmax`, `Tavg`) + 초기 내부 저항(`IR`)
  - 근거: 전기화학적 열화와 운전 스트레스(급속 충전 시간, 발열, 오믹 저항 증가)를 종합한 원논문 최적 구성.
- **Target 변수**: $\mathbf{y = \log_{10}(\text{Cycle Life})}$ (예측 후 $10^{\hat{y}}$로 역변환하여 사이클 수 복원)
  - 근거: 수명 분포가 392 ~ 1,935회로 우측 꼬리가 길게 편향되어 있으므로, 로그 변환을 통해 잔차의 정규성 확보 및 스케일 안정화.

### 모델 선택 및 근거

- 후보 모델 :
  - **ElasticNet** : $L_1$ (Lasso) + $L_2$ (Ridge) 정규화를 결합하여 높은 다중공선성을 완화하고 유의미한 피처를 자동 선택하는 원논문(Severson et al. 2019) 표준 선형 회귀 모델.
  - **Ridge** : $L_2$ 패널티를 통해 회귀 계수의 크기를 제어하고 안정적인 가중치 추정.
  - **Random Forest** : 결정 트리 앙상블을 통해 비선형 관계를 포착하고 이상치에 강건한 예측.
  - **Gradient Boosting** : 잔차 순차 학습을 통해 복합 상호작용 피처 학습.
- 최종 모델 :
  - **원논문 벤치마크 대표 모델**: **Level 3 (Full) ElasticNet** ($\alpha=0.01, \text{l1\_ratio}=0.5$)
  - **배치 간 일반화 최적 모델**: **Level 1 (Variance) RandomForest** ($\text{max\_depth}=3, \text{n\_estimators}=50$)
- 선택 이유 :
  - ElasticNet은 Batch 1 내부 훈련/검증에서 가장 정밀한 적합도(Train CV 6.29%, Valid Hold-out 9.07%)를 보여 원논문의 베이스라인 구조를 완벽히 재현함.
  - RandomForest는 Batch 1(평균 844)과 Batch 2(평균 565, 400회대 단수명 편중) 간의 극심한 분포 차이(Batch Effect) 하에서 선형 모델의 과도한 외삽 오류를 방지하고 결정 경계를 제약하여 Test MAPE 28.57%로 가장 우수한 일반화 성능을 달성함.

## 성능 결과

### Performance Reporting 기준 (Index 정의)
- **Train (Batch 1 CV)** : Batch 1 내 5-Fold Cross-Validation 평균 성능
- **Valid (Batch 1 Hold-out)** : Batch 1 내 충전 프로토콜(Policy) 기반 Hold-out 검증 성능
  - *Valid를 CV가 아닌 Hold-out으로 사용하는 이유* :
    - 배터리 데이터는 셀 단위로 독립적이며, 각 셀이 서로 다른 충전 프로토콜(C-rate)로 실험됨
    - 이 경우 단순 CV를 적용하면 동일 프로토콜 셀이 train/valid에 나뉘어 들어가 데이터 누수(leakage) 위험 잔존
    - Hold-out은 프로토콜 단위(Group) 분리를 명확히 보장하며, 배치 간 일반화를 평가하는 이번 프로젝트 구조에서 더 적합
- **Test (Batch 2)** : Batch 2 최종 평가 성능 (Batch 1 전체로 학습 후 독립 배치 평가)
- **Gap (Train-Valid)** : Train 과적합 확인 ($\text{Valid MAPE} - \text{Train MAPE}$)
- **Gap (Valid-Test)** : 배치 간 일반화 차이 확인 ($\text{Test MAPE} - \text{Valid MAPE}$)
- **Gap (Target-Test)** : 원논문 성능 대비 차이 ($\text{Test MAPE} - 9.1\%$)
  - 원논문 성능 : Regression **9.1% (MAPE)**, Classification 4.9% (1-Accuracy)
  - Performance Metric : Regression - MAPE (%)

---

### Reporting format (for Regression)

#### 1. 원논문 대표 모델 : Level 3 Full Model (ElasticNet)
| 구분 | MAPE (%) | 비고 |
|:---|:---:|:---|
| Train (Batch 1 CV) | 6.29% | Batch 1 5-Fold CV 평균 성능 |
| Valid (Batch 1 Hold-out) | 9.07% | 프로토콜 분리 Hold-out 검증 성능 |
| Test (Batch 2) | 39.59% | Batch 2 최종 평가 성능 (RMSE: 216.5 cycles) |
| Gap (Train-Valid) | +2.78% | (+) : 과적합 의심 (경미한 수준) |
| Gap (Valid-Test) | +30.53% | (+) : 배치간 일반화 저하 의심 (심각한 Batch Effect) |
| Gap (Target-Test) | +30.49% | Target : 원논문 9.1% |

#### 2. 최적 머신러닝 모델 : Level 1 Variance Model (RandomForest)
| 구분 | MAPE (%) | 비고 |
|:---|:---:|:---|
| Train (Batch 1 CV) | 6.93% | Batch 1 5-Fold CV 평균 성능 |
| Valid (Batch 1 Hold-out) | 6.41% | 프로토콜 분리 Hold-out 검증 성능 |
| Test (Batch 2) | 28.57% | Batch 2 최종 평가 성능 (RMSE: 157.9 cycles) |
| Gap (Train-Valid) | -0.52% | (+) : 과적합 의심 (과적합 없음) |
| Gap (Valid-Test) | +22.16% | (+) : 배치간 일반화 저하 의심 |
| Gap (Target-Test) | +19.47% | Target : 원논문 9.1% |

---

### 전체 모델 성능 비교 요약 (`results/model_performance.csv`)

| Feature Set | Model | Train (Batch 1 CV) | Valid (Batch 1 Hold-out) | Test (Batch 2) | Test RMSE | Gap (Train-Val) | Gap (Val-Test) | Gap (Target-Test) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Level 1 (Variance)** | ElasticNet | 9.71% | 17.16% | 36.69% | 188.1 | +7.46% | +19.52% | +27.59% |
| Level 1 (Variance) | Ridge | 9.63% | 18.64% | 35.42% | 182.6 | +9.02% | +16.78% | +26.32% |
| **Level 1 (Variance)** | **RandomForest** | **6.93%** | **6.41%** | **28.57%** | **157.9** | **-0.52%** | **+22.16%** | **+19.47%** |
| Level 1 (Variance) | GradientBoosting | 7.64% | 7.22% | 31.86% | 173.9 | -0.42% | +24.64% | +22.76% |
| Level 2 (Discharge) | ElasticNet | 6.11% | 8.26% | 39.50% | 219.2 | +2.15% | +31.24% | +30.40% |
| Level 2 (Discharge) | Ridge | 6.01% | 9.28% | 43.98% | 249.5 | +3.27% | +34.70% | +34.88% |
| Level 2 (Discharge) | RandomForest | 7.61% | 6.33% | 32.64% | 175.0 | -1.27% | +26.31% | +23.54% |
| Level 2 (Discharge) | GradientBoosting | 7.45% | 8.37% | 39.84% | 203.2 | +0.92% | +31.47% | +30.74% |
| **Level 3 (Full)** | **ElasticNet (논문)** | **6.29%** | **9.07%** | **39.59%** | **216.5** | **+2.78%** | **+30.53%** | **+30.49%** |
| Level 3 (Full) | Ridge | 6.42% | 10.39% | 43.88% | 242.1 | +3.96% | +33.49% | +34.78% |
| Level 3 (Full) | RandomForest | 7.53% | 7.35% | 32.49% | 170.1 | -0.18% | +25.14% | +23.39% |
| Level 3 (Full) | GradientBoosting | 7.03% | 7.19% | 37.34% | 191.0 | +0.16% | +30.15% | +28.24% |

## 오류 분석

- 모델이 가장 크게 틀린 셀의 공통점
  - Batch 2 테스트에서 가장 오차가 큰 상위 6개 셀:
    1. `b2_c6` : 실제 수명 393회 $\to$ 예측 707회 (ElasticNet APE **79.86%**, RF APE 65.75%) / 프로토콜: `3.6C(9%)-5C`
    2. `b2_c29` : 실제 수명 452회 $\to$ 예측 766회 (ElasticNet APE **69.40%**, RF APE 35.64%) / 프로토콜: `5.2C(58%)-4C`
    3. `b2_c18` : 실제 수명 449회 $\to$ 예측 754회 (ElasticNet APE **67.97%**, RF APE 52.27%) / 프로토콜: `5.2C(50%)-4.25C`
    4. `b2_c15` : 실제 수명 396회 $\to$ 예측 629회 (ElasticNet APE **58.80%**, RF APE 54.82%) / 프로토콜: `3.6C(9%)-5C`
    5. `b2_c21` : 실제 수명 408회 $\to$ 예측 627회 (ElasticNet APE **53.60%**, RF APE 36.01%) / 프로토콜: `6C(60%)-3C`
    6. `b2_c19` : 실제 수명 392회 $\to$ 예측 590회 (ElasticNet APE **50.56%**, RF APE 41.56%) / 프로토콜: `6C(60%)-3C`
  - **공통점 분석** :
    - 모두 최종 수명이 390~450회에 불과한 **극단 단수명 셀**에 해당함.
    - 고전류가 50~60% 이상 장시간 인가되는 초가혹 급속 충전 조건에서 작동하여, 음극 표면 금속 리튬 석출(Lithium Plating)로 인해 300 사이클 전후에서 조기 Knee Point가 발생한 셀들임.
    - 모든 모델이 이 셀들의 수명을 **200 ~ 350 사이클가량 체계적으로 과대평가(Systematic Overestimation)**함.

- 원인 가설 및 개선 방향
  - **원인 가설 1 (훈련 데이터 도메인 부재 및 선형 외삽 오류)** :
    - 학습용 Batch 1의 최소 수명은 **513회**로, 500회 미만 단수명 셀 데이터가 훈련 세트에 **단 1개도 존재하지 않았음**.
    - 이로 인해 Batch 1으로만 학습된 선형 회귀 모델(ElasticNet)은 390~450회 미학습 영역에 대해 무리한 외삽(Extrapolation)을 수행하여 600~760회로 과대평가하게 됨.
  - **원인 가설 2 (배치 간 환경 편차 / Batch Effect & Concept Drift)** :
    - 원논문(Severson et al.)의 9.1% MAPE는 Batch 1과 Batch 2를 결합한 전체 풀에서 다양한 수명 구간의 셀들을 균일하게 샘플링(Train/Test 층화 분할)하여 달성한 수치임.
    - 반면 Batch 1(2017년 5월 실험, 513~1,189회)과 Batch 2(2018년 2월 실험, 392~1,064회)는 제조 로트, 챔버 환경, 장시간 일시 중지(pause) 등 미세한 공정 차이가 존재하여, 배치 간 일반화 저하(Gap Valid-Test: +30.5%)가 극명하게 발생함.
  - **개선 방향** :
    - 1) **Train 데이터 다양화 (도메인 확장)**: 원논문과 같이 Batch 1과 Batch 2의 프로토콜 및 수명 구간을 층화 샘플링하여 400회대 단수명 셀을 Train Set에 포함.
    - 2) **비선형 트리/하이브리드 모델 활용**: 외삽 위험이 큰 선형 모델 대신 경계를 보수적으로 제한하는 Random Forest 또는 구간별 분할(Piecewise) 회귀 모델 도입.
    - 3) **Two-Stage 분류/회귀 파이프라인**: 1단계에서 500 사이클 미만 고위험 셀 여부를 이진 분류(Classification)한 후, 2단계에서 고위험군/저위험군 전용 세부 회귀 모델을 적용하여 과대평가 억제.

## ESS 도메인 해석

- 이 모델을 실제 BESS에 적용한다면 어떤 의사결정에 활용 가능한가?
  - **1) 신규 배터리 입고 품질 전수 검사 (Screening & Acceptance Test)** :
    - BESS 사이트 구축 시 수천 개의 셀 중 제조 편차나 미세 결함으로 조기 퇴화 위험이 있는 불량 셀을 초기 100회 충·방전 사이클만으로 조기 선별하여 보증 교체 요청.
  - **2) 수명 보존형 스마트 충전 스케줄링 (Degradation-Aware BMS Control)** :
    - 전력 계통 주파수 조정(FR)이나 신재생 피크컷 운전 시, 급격한 열화 가속(Knee Point 조기 도달)이 예상되는 랙(Rack)의 C-rate와 충전 SOC 한계치를 능동적으로 하향 조정하여 BESS 시스템 전체 수명 연장.
  - **3) 배터리 잔존 가치 평가 및 2차 사용(Second-life) 인증** :
    - 초기 100 사이클의 $\Delta Q(V)$ 곡선 특징량을 바탕으로 잔여 수명(RUL)을 정량화하여, ESS 운영 수수료 산정, 잔존 가치 보증, 폐배터리의 가정용 ESS 전환 가능 여부 판정에 활용.

- 어떤 한계가 있으며, 실 배포를 위해 추가로 필요한 것은 무엇인가?
  - **한계점** :
    - 1) *실제 계통 부하 프로파일과의 차이*: 본 실험 데이터셋은 고정된 2단계 CC 충전 및 4C 완전 방전(100% DOD) 환경이나, 실제 BESS는 부분 충·방전(Partial Cycles, Micro-cycles)과 불규칙한 부하 프로파일(불규칙한 C-rate)로 운전됨.
    - 2) *초기 100회 누적 소요 시간*: 하루 1사이클 운전 기준 약 3~4개월의 데이터 수집 기간이 필요하며, 완전 방전 곡선(3.5V $\to$ 2.0V) 전체를 주기적으로 얻기 어려울 수 있음.
  - **실 배포를 위한 추가 요구사항** :
    - 1) *부분 충·방전 전압 세그먼트 복원 기술*: 부분 방전(예: SOC 80% $\to$ 20%) 구간만으로 전체 $\Delta Q(V)$ 형상을 추정하는 가상 전압 곡선 재구성 알고리즘 개발.
    - 2) *계절별 온도 보상 및 전이 학습(Transfer Learning)*: 현장 ESS 챔버의 사계절 외기 온도 변동과 노후화에 대응할 수 있도록 물리 기반(Physics-informed) 보정 모델 및 도메인 적응(Domain Adaptation) 기법 적용.

## 참고문헌

- Severson, K. A., Attia, P. M., Jin, N., Perkins, N., Jiang, B., Yang, Z., ... & Braatz, R. D. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4(5), 383-391.
- Attia, P. M., Grover, A., Jin, N., Severson, K. A., Markov, T. M., Liao, Y. H., ... & Chueh, W. C. (2020). Closed-loop optimization of fast-charging protocols for batteries with machine learning. *Nature*, 578(7795), 397-402.

## 팀 구성

- 김영희 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch2)
- 박철수 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch3)
