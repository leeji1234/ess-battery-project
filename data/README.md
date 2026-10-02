# MIT-Stanford Battery Dataset 안내

본 프로젝트에서 사용하는 데이터셋은 MIT, Stanford 대학교 및 Toyota Research Institute(TRI)가 공동 연구하여 Nature Energy (2019)에 발표한 상용 원통형 LFP/흑연 18650 리튬이온 배터리 급속 충전 데이터셋입니다.

## 1. 데이터셋 개요
- **원 논문**: Severson, K.A. et al. "Data-driven prediction of battery cycle life before capacity degradation." *Nature Energy* 4, 383–391 (2019).
- **대상 배터리**: A123 Systems 상용 APR18650M1A LFP/흑연 셀 (공칭 용량 1.1 Ah, 공칭 전압 3.3 V)
- **충전 프로토콜**: 2단계 고속 충전 정책 ($C_1(\text{SOC}_1)\text{-}C_2$) 적용 (총 72가지 이상의 다양한 급속 충전 조건)
  - 1단계: $C_1$ 전류로 특정 충전 상태($\text{SOC}_1$)까지 충전
  - 2단계: $C_2$ 전류로 80% SOC까지 충전 후 1C CC-CV로 만충전
- **방전 조건**: 모든 셀 공통 4C 방전 (방전 종지 전압 2.0 V)
- **수명 종료 정의(EOL)**: 공칭 용량 1.1 Ah의 80%인 **0.88 Ah 도달 시점**

## 2. 배치(Batch) 구성
| 배치 구분 | 실험 일자 | 총 셀 수 | 유효 셀 수 | 평균 수명 (Cycles) | 수명 범위 (Min ~ Max) | 주요 역할 |
|---|---|---|---|---|---|---|
| **Batch 1** | 2017-05-12 | 46개 | 46개 | 844.7회 | 513 ~ 1,189회 | **학습 및 검증셋 (Train & Hold-out Valid)** |
| **Batch 2** | 2018-02-20 | 47개 | 39개 (8개 결측/중단) | 565.7회 | 392 ~ 1,064회 | **테스트셋 (Test Set: 배치 간 일반화 평가)** |
| **Batch 3** | 2018-04-12 | 46개 | 44개 (2개 결측/중단) | 1,059.7회 | 828 ~ 1,935회 | 추가 검증용 독립 배치 |

## 3. 전처리 데이터 구조 (`processed_data/`)
대용량(약 8GB) MATLAB `.mat` 원본 파일을 경량화하여 빠른 실험이 가능하도록 변환한 데이터입니다:
- `batch_meta.csv`: 각 셀의 식별자(`cell_key`), 소속 배치(`batch_id`), 최종 사이클 수명(`cycle_life`), 충전 정책(`policy_readable`)
- `batch_summary.pkl` & `batch_summary.csv.gz`: 셀별 전 사이클 시계열 요약 정보 (방전용량 $Q_d$, 충전용량 $Q_c$, 내부저항 $IR$, 최고온도 $T_{max}$, 평균온도 $T_{avg}$, 충전시간 `chargetime`)
- `qdlin_cycle10_100.pkl`: 사이클 10과 사이클 100에서의 1,000포인트 균일 전압 그리드 방전 곡선 $Q_{10}(V)$, $Q_{100}(V)$ 및 차분 데이터 $\Delta Q_{100-10}(V)$
