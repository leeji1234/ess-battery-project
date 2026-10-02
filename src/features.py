import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def parse_crate(p_str):
    """
    충전 프로토콜 문자열 파싱 (예: 6C(60%)-3C -> (6.0, 60.0, 3.0))
    """
    if not isinstance(p_str, str):
        return np.nan, np.nan, np.nan
    
    # 2단계 프로트콜 패턴 매칭 - 충전률 연결
    m = re.findall(r'([0-9.]+)C\(([0-9.]+)%\)-([0-9.]+)C', p_str)
    
    if m:
        return float(m[0][0]), float(m[0][1]), float(m[0][2])

    # 단순히 C-rate로만 적혀 있는 경우의 파싱
    m2 = re.findall(r'([0-9.]+)C', p_str)
    if len(m2) >= 2:
        return float(m2[0]), 80.0, float(m2[1])
    elif len(m2) == 1:
        return float(m2[0]), 80.0, float(m2[0])
    return np.nan, np.nan, np.nan

def extract_features(meta_df, summary_df, qdlin_dict):
    """
    초기 100 사이클 데이터를 기반으로 배터리 수명 예측 피처를 생성합니다.
    """
    # 1. 메타데이터 정제
    valid_meta = meta_df[meta_df['cycle_life'].notna()].copy()        # 배터리 cell 메타 데이터 저장
    valid_meta['cycle_life'] = valid_meta['cycle_life'].astype(float) # cycle 별 집계 데이터 - cycle / Qd / T / IR 등 저장
    valid_meta['log_cycle_life'] = np.log10(valid_meta['cycle_life']) # np.log10() : cycle-life를 log 형태로 변경하여 독립변수들과 수치 1대1 대응 맞추기

    # C-rate 파싱
    rates = valid_meta['policy_readable'].apply(parse_crate) # 위에서 정의한 파싱 규칙 대로 c rate(1/2차), SoC step을 파싱
    valid_meta['c_rate_step1'] = [r[0] for r in rates]
    valid_meta['soc_step1'] = [r[1] for r in rates]
    valid_meta['c_rate_step2'] = [r[2] for r in rates]

    # 2. 요약 시계열 이상치 필터링
    summary_clean = summary_df[summary_df['QD'].between(0.85, 1.25)].copy() # 이상치 -> Ah : <0.85 or >1.25의 이상치를 배제

    # 초기 100 사이클 정제 (이상치 필터: chargetime > 25분 제외, Tmax > 50도 제외)
    early_valid = summary_clean[
        (summary_clean['cycle'] >= 2) & (summary_clean['cycle'] <= 100) &           # 초기 cycle과 100번째 cycle에서의 비교 (1번째는 오차 배제하기 위해 사용하지 않음)
        (summary_clean['chargetime'] > 0) & (summary_clean['chargetime'] < 25) &    # chargetime 이상치 제거 - 충전 시간이 음수이거나 지나치게 길거나
        (summary_clean['Tmax'] < 50)                                                # 최대 온도 이상치 제거
    ]

    # 운영/환경 통계 : 100회에서의 평균을 계산하여 넘기기
    early_summary = early_valid.groupby('cell_key').agg({
        'chargetime': 'mean',
        'Tmax': 'mean',
        'Tavg': 'mean',
        'IR': lambda s: s[s > 0].mean() if (s > 0).any() else np.nan
    }).reset_index()

    # 방전용량 선형 추세 피처 : 선형 회귀를 위한 계수 정의
    def get_qd_trend(s):
        s = s.sort_values('cycle')
        x = s['cycle'].values
        y = s['QD'].values
        slope, intercept = np.polyfit(x, y, 1) if len(s) >= 2 else (0.0, 0.0)
        return pd.Series({
            'slope_qd': slope,
            'intercept_qd': intercept,
            'qd_100_2': y[-1] - y[0] if len(y) > 0 else 0.0,
            'qd_2': y[0] if len(y) > 0 else 1.0
        })

    qd_trends = early_valid.groupby('cell_key').apply(get_qd_trend, include_groups=False).reset_index()

    # 3. Delta Q(V) 곡선 파생 통계 피처 : Q_100(V) - Q_10(V)의 파생 통계 피처 생성 
    dq_records = []
    for k, d in qdlin_dict.items():
        cl = d['cycle_life']
        if pd.isna(cl) or d['Qdlin_10'] is None or d['Qdlin_100'] is None:
            continue
        dq = d['Qdlin_100'] - d['Qdlin_10']
        var_dq = float(np.var(dq))
        min_dq = float(np.min(dq))
        mean_dq = float(np.mean(dq))
        skew_dq = float(skew(dq))
        kurt_dq = float(kurtosis(dq))

        dq_records.append({
            'cell_key': k,
            'batch_id': d['batch_id'],
            'var_dq': var_dq,
            'log_var_dq': np.log10(var_dq) if var_dq > 0 else np.nan,
            'min_dq': min_dq,
            'mean_dq': mean_dq,
            'skew_dq': skew_dq,
            'kurt_dq': kurt_dq,
        })
    df_dq = pd.DataFrame(dq_records)

    # 4. 피처 병합
    if 'batch_id' in df_dq.columns:
        df_dq = df_dq.drop(columns=['batch_id'])
    df_features = valid_meta.merge(df_dq, on='cell_key', how='inner')
    df_features = df_features.merge(early_summary, on='cell_key', how='left')
    df_features = df_features.merge(qd_trends, on='cell_key', how='left')

    # 결측치 보정 (IR 등)
    if 'IR' in df_features.columns:
        df_features['IR'] = df_features['IR'].fillna(df_features['IR'].median())

    return df_features

# FEATURE_SETS : 3단계 복잡도를 가짐 - (1단계) 단순 dQ -> (2단계) 곡선 피쳐 개형 추가 -> (3단계) 전 지표 추가
FEATURE_SETS = {
    'Level 1 (Variance)': [
        'log_var_dq'
    ],
    'Level 2 (Discharge)': [
        'log_var_dq', 'min_dq', 'mean_dq', 'skew_dq', 'kurt_dq',
        'slope_qd', 'intercept_qd', 'qd_100_2', 'qd_2'
    ],
    'Level 3 (Full)': [
        'log_var_dq', 'min_dq', 'mean_dq', 'skew_dq', 'kurt_dq',
        'slope_qd', 'intercept_qd', 'qd_100_2', 'qd_2',
        'chargetime', 'Tmax', 'Tavg', 'IR'
    ]
}

if __name__ == '__main__':
    from preprocess import load_processed_data
    meta, summary, qdlin = load_processed_data()
    feats = extract_features(meta, summary, qdlin)
    print(f"Features extracted: shape={feats.shape}")
    print("Columns:", feats.columns.tolist())
    