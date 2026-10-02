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
    meta, summary, qdlin = load_processed_data()
    df = extract_features(meta, summary, qdlin)

    # 2. 배치 분할 (Batch 1: Train/Valid, Batch 2: Test)
    b1 = df[df['batch_id'] == 1].copy().reset_index(drop=True)
    b2 = df[df['batch_id'] == 2].copy().reset_index(drop=True)

    print(f"   • Batch 1 (Train/Valid Pool): {len(b1)} cells")
    print(f"   • Batch 2 (Test Set)        : {len(b2)} cells")

    # 3. Batch 1 내 충전 프로토콜(Policy) 기반 Hold-out 분리 (Data Leakage 방지)
    #    동일 프로토콜 셀이 train/valid에 나뉘어 들어가는 누수를 막기 위해 policy 기준 split
    unique_policies = sorted(b1['policy_readable'].unique())
    np.random.seed(42)
    n_val_policies = int(len(unique_policies) * 0.2)
    val_policies = set(np.random.choice(unique_policies, size=n_val_policies, replace=False))
    
    train_mask = ~b1['policy_readable'].isin(val_policies)
    val_mask = b1['policy_readable'].isin(val_policies)

    print(f"   • Batch 1 Protocol-based Hold-out: Train={train_mask.sum()} cells, Valid={val_mask.sum()} cells")

    # 4. 평가 모델 목록 정의
    models = {
        'ElasticNet (alpha=0.01, l1=0.5)': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=3000),
        'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=42),
        'RandomForest (depth=3, n=50)': RandomForestRegressor(max_depth=3, n_estimators=50, random_state=42),
        'GradientBoosting (depth=3, n=50)': GradientBoostingRegressor(max_depth=3, n_estimators=50, learning_rate=0.05, random_state=42)
    }

    all_evaluations = []
    model_predictions = {}

    target_log_train = b1['log_cycle_life'].values
    target_act_train = b1['cycle_life'].values
    target_act_test = b2['cycle_life'].values

    for f_name, f_cols in FEATURE_SETS.items():
        print(f"\nEvaluating Feature Set: {f_name} ({len(f_cols)} features)...")
        scaler = StandardScaler()
        X1 = scaler.fit_transform(b1[f_cols].values)
        X2 = scaler.transform(b2[f_cols].values)

        for m_name, model in models.items():
            # 1) Train (Batch 1 CV): 5-Fold Cross Validation 평균 성능
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_mapes = []
            for tr_idx, val_cv_idx in kf.split(X1):
                m_cv = clone(model)
                m_cv.fit(X1[tr_idx], target_log_train[tr_idx])
                p_cv = 10 ** m_cv.predict(X1[val_cv_idx])
                cv_mapes.append(mean_absolute_percentage_error(target_act_train[val_cv_idx], p_cv) * 100)
            train_cv_mape = np.mean(cv_mapes)

            # 2) Valid (Batch 1 Hold-out): 프로토콜 단위 Hold-out 검증 성능
            m_hold = clone(model)
            m_hold.fit(X1[train_mask], target_log_train[train_mask])
            p_hold = 10 ** m_hold.predict(X1[val_mask])
            valid_holdout_mape = mean_absolute_percentage_error(target_act_train[val_mask], p_hold) * 100

            # 3) Test (Batch 2): Batch 1 전체로 학습 후 Batch 2 테스트 성능
            m_full = clone(model)
            m_full.fit(X1, target_log_train)
            p_test = 10 ** m_full.predict(X2)
            test_mape = mean_absolute_percentage_error(target_act_test, p_test) * 100
            test_rmse = np.sqrt(mean_squared_error(target_act_test, p_test))

            gap_train_valid = valid_holdout_mape - train_cv_mape
            gap_valid_test = test_mape - valid_holdout_mape
            gap_target_test = test_mape - 9.1  # 원논문 9.1% MAPE 대비

            key = f"{f_name} | {m_name}"
            model_predictions[key] = p_test

            all_evaluations.append({
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

    df_results = pd.DataFrame(all_evaluations)
    perf_path = os.path.join(output_dir, 'model_performance.csv')
    df_results.to_csv(perf_path, index=False)
    print(f"\n[Saved] Performance table saved to {perf_path}")

    # 5. 대표 모델(논문 기준 Level 3 ElasticNet 및 최적 RF 모델)에 대한 오차 분석
    best_lin_key = 'Level 3 (Full) | ElasticNet (alpha=0.01, l1=0.5)'
    best_rf_key = 'Level 1 (Variance) | RandomForest (depth=3, n=50)'
    
    b2_errors = b2[['cell_key', 'policy_readable', 'cycle_life']].copy()
    b2_errors['Pred_ElasticNet_Full'] = model_predictions[best_lin_key]
    b2_errors['APE_ElasticNet (%)'] = np.abs(b2_errors['cycle_life'] - b2_errors['Pred_ElasticNet_Full']) / b2_errors['cycle_life'] * 100
    
    b2_errors['Pred_RandomForest'] = model_predictions[best_rf_key]
    b2_errors['APE_RandomForest (%)'] = np.abs(b2_errors['cycle_life'] - b2_errors['Pred_RandomForest']) / b2_errors['cycle_life'] * 100

    err_path = os.path.join(output_dir, 'error_analysis.csv')
    b2_errors.sort_values('APE_ElasticNet (%)', ascending=False).to_csv(err_path, index=False)
    print(f"[Saved] Error analysis saved to {err_path}")

    # 6. 표준 리포팅 포맷(Reporting Format for Regression) 테이블 출력
    print("\n" + "=" * 80)
    print("📊 [Reporting Format (for Regression) - Best Representative Models]")
    print("=" * 80)
    
    for rep_name, key in [('원논문 대표 모델: Full Model (ElasticNet)', best_lin_key),
                          ('최적 트리 모델: Variance Model (RandomForest)', best_rf_key)]:
        row = df_results[df_results['Feature_Set'] + ' | ' + df_results['Model'] == key].iloc[0]
        print(f"\n▶ {rep_name}")
        print("-" * 65)
        print(f"| {'구분':<25} | {'MAPE (%)':<10} | {'비고':<22} |")
        print("|" + "-" * 27 + "|" + "-" * 12 + "|" + "-" * 24 + "|")
        print(f"| {'Train (Batch 1 CV)':<25} | {row['Train (Batch 1 CV)']:>9.2f}% | {'':<22} |")
        print(f"| {'Valid (Batch 1 Hold-out)':<25} | {row['Valid (Batch 1 Hold-out)']:>9.2f}% | {'':<22} |")
        print(f"| {'Test (Batch 2)':<25} | {row['Test (Batch 2)']:>9.2f}% | {'':<22} |")
        print(f"| {'Gap (Train-Valid)':<25} | {row['Gap (Train-Valid)']:>+9.2f}% | {'(+) : 과적합 의심':<22} |")
        print(f"| {'Gap (Valid-Test)':<25} | {row['Gap (Valid-Test)']:>+9.2f}% | {'(+) : 배치간 일반화 저하 의심':<22} |")
        print(f"| {'Gap (Target-Test)':<25} | {row['Gap (Target-Test)']:>+9.2f}% | {'Target : 원논문 9.1%':<22} |")

    return df_results, b2_errors

if __name__ == '__main__':
    run_training_pipeline()
