import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# 상대 경로 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_processed_data
from features import extract_features, FEATURE_SETS

def run_training_pipeline(output_dir='results'):
    """
    Batch 1을 학습 데이터(CV & Hold-out), Batch 2를 테스트 데이터로 사용하여
    모델들을 학습하고 성능을 비교 평가합니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 데이터 로드 및 피처 생성
    print("1. Loading data & extracting features...")
    meta, summary, qdlin = load_processed_data()    # data load하기
    df = extract_features(meta, summary, qdlin)     # features.py의 feature extract 함수로 dataframe을 생성

    # 2. 배치 분할 (Batch 1: Train/Valid, Batch 2: Test)
    b1 = df[df['batch_id'] == 1].copy().reset_index(drop=True)  # train / valid용 Batch 1 지정
    b2 = df[df['batch_id'] == 2].copy().reset_index(drop=True)  # Test 용 Batch 2 지정

    print(f"   • Batch 1 (Train/Valid Pool): {len(b1)} cells")
    print(f"   • Batch 2 (Test Set)        : {len(b2)} cells")

    # 3. Batch 1 내 충전 프로토콜(Policy) 기반 Hold-out 분리 (Data Leakage 방지) : 충전 프로토콜이 Valid 에서 동일할 때 모델이 train set에서 암기한 수치를 사용 - Data Leakage

    unique_policies = sorted(b1['policy_readable'].unique())                                    # Batch 1(Train/Valid)에 존재하는 충전 프로토콜을 정리한다
    np.random.seed(42)                                                                          # 동일한 검증 셋이 선택되도록 난수생성기를 고정한다
    n_val_policies = int(len(unique_policies) * 0.2)                                            # Valid 용으로 떼어놓기 위한 충전 프로토콜의 숫자를 계산
    val_policies = set(np.random.choice(unique_policies, size=n_val_policies, replace=False))   # Valid dataset을 위의 숫자 만큼으로 만든다
    
    # val_policies (검증 데이터셋 표기)이 있는 경우 boolean mask를 val_mask로 / 아니면 train_mask로
    train_mask = ~b1['policy_readable'].isin(val_policies) 
    val_mask = b1['policy_readable'].isin(val_policies)

    print(f"   • Batch 1 Protocol-based Hold-out: Train={train_mask.sum()} cells, Valid={val_mask.sum()} cells")

    # 4. 평가 모델 목록 정의 : ElasticNet / Ridge / RandomForest / GradientBoosting 의 모델 조건 setting
    models = {
        'ElasticNet (alpha=0.01, l1=0.5)': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=3000),                            # 왜 설정한 model이 오히려 낮게 나오나?
        'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=42),                                                                             # ElasticNet -> Ridge가 이미 소속됨
        'RandomForest (depth=3, n=50)': RandomForestRegressor(max_depth=3, n_estimators=50, random_state=42),                               # 가장 기초적인 모델 -> 왜 이 모델이 잘 나왔는지 의미 파악 필요성
        'GradientBoosting (depth=3, n=50)': GradientBoostingRegressor(max_depth=3, n_estimators=50, learning_rate=0.05, random_state=42)    
    }

    all_evaluations = []
    model_predictions = {}

    target_log_train = b1['log_cycle_life'].values  # Y : 종속 변수인 log10(cycle_life)
    target_act_train = b1['cycle_life'].values      # target_act_train : 원본 수명 
    target_act_test = b2['cycle_life'].values       # target_act_test : (test용) 원본 수명

    for f_name, f_cols in FEATURE_SETS.items():                                    # 1단계 -> 2단계 -> 3단계 복잡도로 순서대로 훈련
        print(f"\nEvaluating Feature Set: {f_name} ({len(f_cols)} features)...")   
        scaler = StandardScaler()                                                  # b1으로만 fit_transform 수행 / b2는 transform만 수행
        X1 = scaler.fit_transform(b1[f_cols].values)                               
        X2 = scaler.transform(b2[f_cols].values)                                   # test set은 fit은 하지 않고 transform하여 test

        for m_name, model in models.items():
            # 1) Train (Batch 1 CV): 5-Fold Cross Validation 평균 성능
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_mapes = []

            for tr_idx, val_cv_idx in kf.split(X1):
                m_cv = clone(model)                                     # 이전 fold 학습 흔적 제거 
                m_cv.fit(X1[tr_idx], target_log_train[tr_idx])          # 80% 데이터로 log scale 학습 
                p_cv = 10 ** m_cv.predict(X1[val_cv_idx])               # 20% 데이터 예측 후 지수 변환 (역로그)
                cv_mapes.append(mean_absolute_percentage_error(target_act_train[val_cv_idx], p_cv) * 100)       # 실제 수명 기준 MAPE(%) 산출

            train_cv_mape = np.mean(cv_mapes) # 5개 fold의 평균 오차

            # 2) Valid (Batch 1 Hold-out): 프로토콜 단위 Hold-out 검증 성능 -> train과 비교하여 과적합 확인 - Data Leakage 방지 여부 확인
            m_hold = clone(model)                                       # 이전 fold 학습 흔적 제거
            m_hold.fit(X1[train_mask], target_log_train[train_mask])    # 학습용 프로토콜 셀들로만 학습
            p_hold = 10 ** m_hold.predict(X1[val_mask])                 # 20% 데이터 예측 후 지수 변환 (역로그)
            valid_holdout_mape = mean_absolute_percentage_error(target_act_train[val_mask], p_hold) * 100       # 모델이 충전 프로토콜을 외워서 data leakage 방지해야

            # 3) Test (Batch 2): Batch 1 전체로 학습 후 Batch 2 테스트 성능
            m_full = clone(model)
            m_full.fit(X1, target_log_train)                            # Batch 1 전체 데이터로 최종 학습
            p_test = 10 ** m_full.predict(X2)                           # 완전히 다른 시점에 실험된 Batch 2 예측
            test_mape = mean_absolute_percentage_error(target_act_test, p_test) * 100 
            test_rmse = np.sqrt(mean_squared_error(target_act_test, p_test))                                    # 단계별로 검증된 최종 모델을 Batch 2에 적용

            # Gap : 오차 증폭(과적합 / batch effect 진단 / 원논문 대비 적합)
            gap_train_valid = valid_holdout_mape - train_cv_mape    # Valid - Train -> 과적합 방지
            gap_valid_test = test_mape - valid_holdout_mape         # Test - Valid -> Batch effect 방지
            gap_target_test = test_mape - 9.1                       # 원논문 9.1% MAPE 대비

            key = f"{f_name} | {m_name}"
            model_predictions[key] = p_test         # 피처셋과 모델명을 합친 식별자로 Model의 예측값 배열을 p_test 딕셔너리로 보관(캐싱)

            all_evaluations.append({                # 변수 * 모델 조합을 7가지 평가 지표를 유효숫자 처리 후 리스트에 추가
                'Feature_Set': f_name,
                'Model': m_name,
                'Train (Batch 1 CV)': round(train_cv_mape, 2),
                'Valid (Batch 1 Hold-out)': round(valid_holdout_mape, 2),
                'Test (Batch 2)': round(test_mape, 2),
                'Test RMSE (cycles)': round(test_rmse, 1),
                'Gap (Train-Valid)': round(gap_train_valid, 2),
                'Gap (Valid-Test)': round(gap_valid_test, 2),
                'Gap (Target-Test)': round(gap_target_test, 2)
            })

    df_results = pd.DataFrame(all_evaluations)                      # 리스트 준비
    perf_path = os.path.join(output_dir, 'model_performance.csv')

    df_results.to_csv(perf_path, index=False)                       # 12개 모델의 성능 비교표를 df에 변환, csv 파일로 자동 저장
    print(f"\n[Saved] Performance table saved to {perf_path}")

    # 5. 대표 모델(논문 기준 Level 3 ElasticNet 및 최적 RF 모델)에 대한 오차 분석
    best_lin_key = 'Level 3 (Full) | ElasticNet (alpha=0.01, l1=0.5)'       # 원논문 (3단계 - Full ElasticNet)
    best_rf_key = 'Level 1 (Variance) | RandomForest (depth=3, n=50)'       # 가장 뛰어난 최적 트리 모델 (1단계 - Variance RF)
    
    b2_errors = b2[['cell_key', 'policy_readable', 'cycle_life']].copy()    # Batch2에서 cell의 고유 식별자 / 충전 protocol / 실제 배터리 수명을 추출
    b2_errors['Pred_ElasticNet_Full'] = model_predictions[best_lin_key]     # ElasticNet의 Batch 2 수명 예측값 39개 배열 가져옴
    b2_errors['APE_ElasticNet (%)'] = np.abs(b2_errors['cycle_life'] - b2_errors['Pred_ElasticNet_Full']) / b2_errors['cycle_life'] * 100   # 절대 백분율 오차
    
    b2_errors['Pred_RandomForest'] = model_predictions[best_rf_key]         # 낮은 MAPE를 가진 RandomForest의 예측값 배열
    b2_errors['APE_RandomForest (%)'] = np.abs(b2_errors['cycle_life'] - b2_errors['Pred_RandomForest']) / b2_errors['cycle_life'] * 100    # RandomForest의 백분율 오차

    err_path = os.path.join(output_dir, 'error_analysis.csv')
    b2_errors.sort_values('APE_ElasticNet (%)', ascending=False).to_csv(err_path, index=False)      # 내림차순 정렬 : 오차가 가장 적은 cell을 가장 위로 올린다 -> 원인 파악
    print(f"[Saved] Error analysis saved to {err_path}")

    # 6. 표준 리포팅 포맷(Reporting Format for Regression) 테이블 출력
    print("\n" + "=" * 80)
    print("📊 [Reporting Format (for Regression) - Best Representative Models]")
    print("=" * 80)
    
    for rep_name, key in [('원논문 대표 모델: Full Model (ElasticNet)', best_lin_key),
                          ('최적 트리 모델: Variance Model (RandomForest)', best_rf_key)]:
        row = df_results[df_results['Feature_Set'] + ' | ' + df_results['Model'] == key].iloc[0]        # df_results 테이블에서 대표 모델의 고유 키와 일치하는 단 하나의 행을 slicing
        print(f"\n▶ {rep_name}")
        print("-" * 65)
        print(f"| {'구분':<25} | {'MAPE (%)':<10} | {'비고':<22} |")
        print("|" + "-" * 27 + "|" + "-" * 12 + "|" + "-" * 24 + "|")
        print(f"| {'Train (Batch 1 CV)':<25} | {row['Train (Batch 1 CV)']:>9.2f}% | {'':<22} |")
        print(f"| {'Valid (Batch 1 Hold-out)':<25} | {row['Valid (Batch 1 Hold-out)']:>9.2f}% | {'':<22} |")
        print(f"| {'Test (Batch 2)':<25} | {row['Test (Batch 2)']:>9.2f}% | {'':<22} |")

        # 강제 부호 표시 : 오차의 증/감이 부호로 즉시 시각화
        print(f"| {'Gap (Train-Valid)':<25} | {row['Gap (Train-Valid)']:>+9.2f}% | {'(+) : 과적합 의심':<22} |")
        print(f"| {'Gap (Valid-Test)':<25} | {row['Gap (Valid-Test)']:>+9.2f}% | {'(+) : 배치간 일반화 저하 의심':<22} |")
        print(f"| {'Gap (Target-Test)':<25} | {row['Gap (Target-Test)']:>+9.2f}% | {'Target : 원논문 9.1%':<22} |")

    return df_results, b2_errors

if __name__ == '__main__':
    run_training_pipeline()
