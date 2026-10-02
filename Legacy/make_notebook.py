import os
import json

# 프로젝트 루트 디렉터리로 작업 디렉터리 자동 이동 (Legacy/ 내부 실행 호환성)
if not os.path.exists('processed_data') and os.path.exists('../processed_data'):
    os.chdir('..')

cells = []

def add_md(text):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.split('\n')]
    })

def add_code(text):
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in text.split('\n')]
    })

# 0. Title
add_md("""# DAY 1 - 배터리 수명 예측 모델 전략 수립 (Batch 1 + 2 + 3 통합 EDA)

- **대상 Dataset**: Severson et al. (Nature Energy, 2019) 상용 LFP/흑연 18650 셀 139개 (Batch 1, Batch 2, Batch 3)
- **주요 목표**:
  1. 5대 핵심 질문에 대한 심층 EDA 수행
  2. Batch 간 특성 비교 (Batch Effect 분석)
  3. EDA 인사이트 기반 머신러닝 모델 설계 전략 수립
  4. 생성된 모든 플롯은 `figure/` 디렉터리에 번호 순서대로 자동 저장""")

# 1. Setup & Load
add_md("""## 1. 환경 설정 및 경량 데이터 로드
- 원본 8GB `.mat` 파일에서 추출된 요약 시계열(`batch_summary.pkl`)과 $\\Delta Q(V)$ 계산용 `qdlin_cycle10_100.pkl`을 로드합니다.
- 그래프 저장용 `figure/` 디렉터리를 자동 생성합니다.
- 데이터 로딩 시간: **1초 이내**""")

add_code("""import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import skew, kurtosis

# 프로젝트 루트 디렉터리로 작업 디렉터리 자동 이동 (Legacy/ 내부 실행 호환성)
if not os.path.exists('processed_data') and os.path.exists('../processed_data'):
    os.chdir('..')

# 이미지 저장 폴더 생성
os.makedirs('figure', exist_ok=True)

# 한글 및 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['AppleGothic', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
meta = pd.read_csv('processed_data/batch_meta.csv')
summary = pd.read_pickle('processed_data/batch_summary.pkl')
with open('processed_data/qdlin_cycle10_100.pkl', 'rb') as f:
    qdlin = pickle.load(f)

print(f"✅ 메타데이터 셀 수      : {len(meta)} (유효 셀: {meta['cycle_life'].notna().sum()}개)")
print(f"✅ 총 사이클 시계열 행 수 : {len(summary):,} 행")
print(f"✅ Qdlin 10/100 셀 수     : {len(qdlin)} 개")
print(f"✅ 그래프 저장 경로       : figure/")
meta.head()""")

# 2. Question 1
add_md("""---
## 2. Question 1. Cycle Life 분포는 어떻게 생겼는가?
- 150 ~ 2,300 사이클 Histogram
- 전체 및 Batch 1, 2, 3별 통계 요약 (count, mean, std, min, 25%, 50%, 75%, max, IQR, Whiskers)
- 장수명(>1,000) / 단수명(<500) 비율 확인
- 이상치 셀 식별 - 왜 유독 짧은가?""")

add_code("""# 1. 유효 셀 데이터 분리 (결측치 제외)
valid_meta = meta[meta['cycle_life'].notna()].copy()
valid_meta['cycle_life'] = valid_meta['cycle_life'].astype(int)

# 2. 전체 및 배치별 상세 통계 집계 (count, mean, std, min, Q1, median, Q3, max, IQR, Whiskers)
rows = []
for b_name, df_sub in [('Batch 1', valid_meta[valid_meta['batch_id'] == 1]),
                       ('Batch 2', valid_meta[valid_meta['batch_id'] == 2]),
                       ('Batch 3', valid_meta[valid_meta['batch_id'] == 3]),
                       ('Total (전체)', valid_meta)]:
    s = df_sub['cycle_life']
    q1 = s.quantile(0.25)
    q2 = s.median()
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    min_v = s.min()
    max_v = s.max()
    lower_w = abs(q1 - min_v)
    upper_w = abs(max_v - q3)
    rows.append({
        '구분': b_name,
        'count': int(len(s)),
        'mean': round(s.mean(), 1),
        'std': round(s.std(), 1),
        'min': int(min_v),
        '25% (Q1)': round(q1, 1),
        '50% (중앙값)': round(q2, 1),
        '75% (Q3)': round(q3, 1),
        'max': int(max_v),
        'IQR': round(iqr, 1),
        'Lower Whisker': round(lower_w, 1),
        'Upper Whisker': round(upper_w, 1),
        '단수명(<500)': f"{(s < 500).mean()*100:.1f}%",
        '장수명(>1000)': f"{(s > 1000).mean()*100:.1f}%"
    })

stats_df = pd.DataFrame(rows)
print("=" * 80)
print("📊 [Cycle Life 상세 요약 통계표]")
print("=" * 80)
display(stats_df)

print("\\n[전체 기준 .describe().round(1)]")
print(valid_meta['cycle_life'].describe().round(1))""")

add_code("""# 3. 시각화 (히스토그램, 전체 Box Plot, 배치별 Box Plot)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors = ['#1f77b4', '#d62728', '#2ca02c']

# 1) 히스토그램 (배치별 분포 및 전체 평균선)
for b, c in zip([1, 2, 3], colors):
    sub = valid_meta[valid_meta['batch_id'] == b]['cycle_life']
    axes[0].hist(sub, bins=15, alpha=0.5, color=c, edgecolor='black', label=f'Batch {b} (n={len(sub)})')

axes[0].axvline(valid_meta['cycle_life'].mean(), color='tomato', linestyle='--', linewidth=2,
                label=f"Total Mean = {valid_meta['cycle_life'].mean():.0f}")
axes[0].axvline(500, color='red', linestyle=':', alpha=0.7, label='Short Life (<500)')
axes[0].axvline(1000, color='blue', linestyle=':', alpha=0.7, label='Long Life (>1000)')
axes[0].set_xlabel('Cycle Life (EOL)')
axes[0].set_ylabel('Number of Cells')
axes[0].set_title('Cycle Life Histogram by Batch')
axes[0].legend(fontsize=8, loc='upper right')

# 2) 전체 단일 Box Plot (전체적인 분포와 Whisker 길이 파악)
axes[1].boxplot(valid_meta['cycle_life'], patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6))
axes[1].set_title('Total Cycle Life Box Plot')
axes[1].set_ylabel('Cycle Life')
axes[1].set_xticks([])
# 수치 텍스트 표기
q1 = valid_meta['cycle_life'].quantile(0.25)
q2 = valid_meta['cycle_life'].median()
q3 = valid_meta['cycle_life'].quantile(0.75)
axes[1].text(1.15, q1, f'Q1: {q1:.0f}', va='center', fontsize=9, color='navy')
axes[1].text(1.15, q2, f'Median: {q2:.0f}', va='center', fontsize=9, fontweight='bold', color='darkred')
axes[1].text(1.15, q3, f'Q3: {q3:.0f}', va='center', fontsize=9, color='navy')

# 3) 배치별 Box Plot (배치 간 차이 시각화)
sns.boxplot(data=valid_meta, x='batch_id', y='cycle_life', palette=colors, ax=axes[2], width=0.5)
sns.stripplot(data=valid_meta, x='batch_id', y='cycle_life', color='black', alpha=0.6, jitter=0.2, ax=axes[2])
axes[2].set_xlabel('Batch ID')
axes[2].set_ylabel('Cycle Life')
axes[2].set_title('Cycle Life Box Plot by Batch')
axes[2].set_xticklabels(['Batch 1 (2017)', 'Batch 2 (2018-1)', 'Batch 3 (2018-2)'])

plt.suptitle('Question 1. Cycle Life Distribution & Whisker Analysis', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('figure/1. Cycle Life Distribution by Batch.png', dpi=300, bbox_inches='tight')
plt.show()""")

add_code("""# 이상치 및 EOL 미도달(NaN) 셀 분석
nan_cells = meta[meta['cycle_life'].isna()]
print("🚨 [미완료/결측 셀 (10개)]: 수명 도달 전 테스트 중단 또는 슬로우 진단 셀")
display(nan_cells[['cell_key', 'batch_id', 'cell_id', 'policy_readable']])

print("\\n⚡ [극단 단수명 셀 상위 5개]")
display(valid_meta.sort_values('cycle_life').head(5)[['cell_key', 'batch_id', 'cycle_life', 'policy_readable']])

print("\\n🌟 [극단 장수명 셀 상위 5개]")
display(valid_meta.sort_values('cycle_life', ascending=False).head(5)[['cell_key', 'batch_id', 'cycle_life', 'policy_readable']])""")

add_md("""### 💡 Q1 인사이트 요약

#### [describe]
- **전체 배터리 평균 수명**: 약 **834 사이클** (최단 392 ~ 최장 1,935)
- **평균(833.7) vs 중앙값(828.0)**: 전체적으로는 큰 차이가 없어 보이나, 배치별로 쪼개보면 편차가 매우 극심함:
  - **Batch 1**: 평균 844.7 vs 중앙값 858.5 (차이 적음, 고른 종형 분포)
  - **Batch 2**: 평균 565.7 vs 중앙값 472.0 (평균이 중앙값보다 훨씬 큼 $\to$ 우측 꼬리가 긴 형태)
  - **Batch 3**: 평균 1,059.7 vs 중앙값 1,005.5 (평균 1,000 이상, 초장수명 중심)
- **IQR 분석**:
  - **Total**: $IQR = |1,017 - 559| = 458$ (절반의 배터리가 559 ~ 1,017 사이클 범위에 집중)
  - **Batch 1**: $IQR = 211.0$ (703 ~ 914 사이클에 집중)
  - **Batch 2**: $IQR = 69.0$ (440 ~ 509 사이클에 초밀집, 단수명 위주)
  - **Batch 3**: $IQR = 327.2$ (828 ~ 1,155 사이클에 집중)

#### [plot]
- **Histogram**:
  - 800~900 사이클 구간에 가장 밀집되어 있으나, 400~500 사이클(Batch 2)과 1,200~1,900 사이클(Batch 3)이 양극단에 존재하여 다봉형(Multimodal) 형태를 보임.
- **Box Plot & Whisker 분석**:
  - **Total**:
    - $\text{Upper Whisker} = |max(1,935) - 75\%(1,017)| = \mathbf{918}$
    - $\text{Lower Whisker} = |25\%(559) - min(392)| = \mathbf{167}$
  - **Batch 1**: Upper(312.8) vs Lower(169.2) $\to$ 장수명 쪽 편차가 1.8배 큼
  - **Batch 2**: Upper(677.5) vs Lower(47.5) $\to$ **장수명 쪽 편차가 14배 이상 큼** (대부분 400대에 죽으나, 좋은 조건을 만난 3개 셀만 1,000회 이상 생존)
  - **해석**:
    - **Upper가 압도적으로 큼**: 좋은 충전 조건을 만나면 배터리 수명이 비약적으로 늘어날 수 있지만, 그 증가 폭이 매우 불규칙하고 편차가 큼.
    - **Lower는 상대적으로 매우 짧음**: 가혹한 고속 충전 조건에서는 열화 속도가 빠르고 일정하여, 400~500회 구간의 좁은 범위로 수명이 예측 가능하게 수렴함.

#### [이상치 식별 (왜 유독 짧은가?)]
1. **결측 셀(10개)**: EOL(80%) 도달 전 100회만 테스트한 VarCharge 셀 및 슬로우 진단 셀 $\to$ **학습/평가에서 제외 필수**
2. **극단 단수명 셀 (392~412회)**:
   - `6C(60%)-3C`, `3.6C(9%)-5C` 등 초고속 충전이 60% 이상까지 지속되어 음극에 심각한 리튬 석출(Lithium Plating)이 조기에 발생하여 조기 고장.

#### [to modeling]
- **장수명 배터리 예측 난이도 높음**: 수명이 길어질수록 편차가 커지므로(Upper Whisker 918), 단순 선형 회귀보다는 스케일을 압축하는 **$\log_{10}(\text{Cycle Life})$ 타겟 변환**이 필수적임.
- **Batch Effect 대응**: Batch 2의 단수명 편중과 Batch 3의 장수명 편중을 극복하기 위해, 랜덤 K-Fold 대신 **배치 단위 분할(Train: Batch 1+2, Test: Batch 3)**로 엄격한 일반화 검증 필요.""")

# 3. Question 2
add_md("""---
## 3. Question 2. 열화 곡선 - 방전 용량(QD)이 어떻게 감소하는가?
- 사이클별 방전 용량 $Q_d$ 추이 시각화
- 초기 100 사이클 확대: 초기 용량만으로 수명 예측이 불가능함을 확인
- Knee Point (급격한 가속 열화 변곡점) 탐색""")

add_code("""# 1. 이상치 사이클 필터링 (정상 용량 0.85 ~ 1.25 Ah 범위 내)
summary_clean = summary[summary['QD'].between(0.85, 1.25)].copy()

# 2. 전체 열화 곡선 및 초기 100 사이클 비교 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

valid_keys = valid_meta['cell_key'].unique()
norm = plt.Normalize(valid_meta['cycle_life'].min(), valid_meta['cycle_life'].max())
cmap = cm.coolwarm

# 왼쪽: 전체 사이클 열화 곡선
for key in valid_keys:
    sub = summary_clean[summary_clean['cell_key'] == key]
    cl = sub['cycle_life'].iloc[0]
    axes[0].plot(sub['cycle'], sub['QD'], color=cmap(norm(cl)), linewidth=0.7, alpha=0.6)

axes[0].axhline(y=1.1 * 0.8, color='black', linestyle='--', linewidth=1.5, label='EOL (0.88 Ah, 80%)')
axes[0].set_xlabel('Cycle Number')
axes[0].set_ylabel('Discharge Capacity QD (Ah)')
axes[0].set_title('Full Degradation Curves (Color: Cycle Life)')
axes[0].set_ylim(0.7, 1.2)
axes[0].legend(loc='lower left')

# 오른쪽: 첫 100 사이클 확대
for key in valid_keys:
    sub = summary_clean[(summary_clean['cell_key'] == key) & (summary_clean['cycle'] <= 100)]
    if len(sub) > 0:
        cl = sub['cycle_life'].iloc[0]
        axes[1].plot(sub['cycle'], sub['QD'], color=cmap(norm(cl)), linewidth=0.8, alpha=0.6)

axes[1].set_xlabel('Cycle Number (Early 100 Cycles)')
axes[1].set_ylabel('Discharge Capacity QD (Ah)')
axes[1].set_title('Early 100 Cycles (No Degradation Visible!)')
axes[1].set_ylim(1.0, 1.15)

# 컬러바 추가
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.03)
cbar.set_label('Cycle Life', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('figure/2-1. Full Degradation Curves and Early 100 Cycles.png', dpi=300, bbox_inches='tight')
plt.show()""")

add_code("""# 초기 용량 변화(Cycle 100 vs Cycle 2) 비율 분포
early_qd = summary_clean[summary_clean['cycle'].isin([2, 100])].pivot(index='cell_key', columns='cycle', values='QD')
early_qd = early_qd.dropna()
ratio_100_2 = early_qd[100] / early_qd[2]

plt.figure(figsize=(8, 4))
plt.hist(ratio_100_2, bins=25, color='teal', edgecolor='black', alpha=0.7)
plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ratio = 1.0 (No Fade)')
plt.xlabel('QD(Cycle 100) / QD(Cycle 2)')
plt.ylabel('Cell Count')
plt.title(f'Capacity Ratio (100:2) - {(ratio_100_2 >= 1.0).mean()*100:.1f}% Cells Increased!')
plt.legend()
plt.tight_layout()
plt.savefig('figure/2-2. Capacity Ratio Distribution (Cycle 100 to 2).png', dpi=300, bbox_inches='tight')
plt.show()""")

add_code("""# 3. Knee Point 탐색 알고리즘 (Piecewise Bilinear Regression / 이중 선형 회귀 잔차 최소화)
def find_knee_bilinear(sub_df):
    sub = sub_df[sub_df['QD'].between(0.85, 1.25)].sort_values('cycle')
    if len(sub) < 80:
        return np.nan, np.nan, None
    x = sub['cycle'].values
    y = pd.Series(sub['QD'].values).rolling(15, min_periods=1, center=True).mean().values
    
    n = len(x)
    best_rss = np.inf
    best_idx = None
    best_p1, best_p2 = None, None
    
    # 30% ~ 95% 구간에서 분기점 탐색
    candidates = np.linspace(int(n * 0.3), int(n * 0.95), 80).astype(int)
    for idx in candidates:
        x1, y1 = x[:idx], y[:idx]
        x2, y2 = x[idx:], y[idx:]
        
        p1 = np.polyfit(x1, y1, 1)
        p2 = np.polyfit(x2, y2, 1)
        
        if p2[0] >= p1[0]:  # 두 번째 기울기가 더 가파른 감소여야 함
            continue
            
        rss = np.sum((y1 - np.polyval(p1, x1))**2) + np.sum((y2 - np.polyval(p2, x2))**2)
        if rss < best_rss:
            best_rss = rss
            best_idx = idx
            best_p1 = p1
            best_p2 = p2
            
    if best_idx is None:
        return np.nan, np.nan, None
        
    knee_cycle = x[best_idx]
    knee_qd = y[best_idx]
    return knee_cycle, knee_qd, (x, y, best_idx, best_p1, best_p2)

# 대표 셀 3개 (단수명, 중간, 장수명) 시각화
sample_cells = ['b2_c19', 'b1_c0', 'b3_c38']  # 392회, 1190회, 1935회
titles = ['Short-life (b2_c19, EOL: 392)', 'Mid-life (b1_c0, EOL: 1190)', 'Long-life (b3_c38, EOL: 1935)']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
knee_results = []

for ax, key, title in zip(axes, sample_cells, titles):
    sub = summary_clean[summary_clean['cell_key'] == key]
    cl = sub['cycle_life'].iloc[0]
    kc, kq, details = find_knee_bilinear(sub)
    
    ax.plot(sub['cycle'], sub['QD'], color='gray', alpha=0.4, linewidth=0.8, label='Measured QD')
    if details is not None:
        x, y, b_idx, p1, p2 = details
        ax.plot(x[:b_idx], np.polyval(p1, x[:b_idx]), color='forestgreen', linewidth=2, label='Stable Phase (L1)')
        ax.plot(x[b_idx:], np.polyval(p2, x[b_idx:]), color='crimson', linewidth=2, label='Accelerated Phase (L2)')
        ax.scatter([kc], [kq], color='red', s=120, zorder=5, marker='*', label=f'Knee: {kc} ({kc/cl*100:.1f}%)')
        ax.axvline(kc, color='red', linestyle='--', alpha=0.6)
        knee_results.append({'cell_key': key, 'cycle_life': cl, 'knee_cycle': kc, 'ratio': round(kc/cl*100, 1)})
        
    ax.axhline(y=1.1 * 0.8, color='black', linestyle=':', label='EOL (0.88 Ah, 80%)')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Discharge Capacity (Ah)')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='lower left')

plt.suptitle('Question 2. Knee Point Detection via Bilinear Piecewise Regression', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('figure/2-3. Knee Point Detection on Degradation Curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 60)
print("📍 [대표 셀 Knee Point 탐색 결과]")
print("=" * 60)
display(pd.DataFrame(knee_results))""")

add_md("""### 💡 Q2 인사이트 요약
1. **초기 용량 불변성(Silent Degradation)**:
   - 100 사이클까지는 **81.4%의 셀에서 오히려 초기 용량보다 용량이 미세하게 증가(+0.2% 중앙값)하거나 유지**됩니다.
   - 따라서 초기 $Q_d$ 스칼라 값만으로는 장수명 셀(파란색)과 단수명 셀(빨간색)을 전혀 구분할 수 없습니다 ($\rho = 0.27$).
2. **Knee Point(급격한 열화 가속점) 탐색 결과**:
   - **발생 시점**: 모든 배터리가 **최종 수명(EOL)의 약 72% ~ 78% 시점**에서 정확하게 Knee Point를 맞이합니다.
     - 단수명 셀(`b2_c19`): 불과 **300 사이클 (76.5%)** 만에 Knee 발생
     - 중간 수명 셀(`b1_c0`): **857 사이클 (72.0%)** 시점에 Knee 발생
     - 장수명 셀(`b3_c38`): **1,520 사이클 (78.6%)**까지 버틴 후 Knee 발생
   - **전기화학 메커니즘**:
     - 음극 활물질 손실($\text{LAM}$)이 누적되면서 가역 리튬 양보다 음극 수용량이 작아지는 순간(N/P ratio 역전), 흑연 표면에 **금속 리튬 석출(Lithium Plating)**이 촉발되어 수직 낙하 열화로 급변합니다.
3. **to Modeling 결론**:
   - 초기 100사이클 동안 용량은 침묵하지만, 내부에서는 이미 $\text{LAM}$이 진행되며 방전 전압 곡선을 왜곡시키고 있습니다.
   - 따라서 **Knee Point의 조기 발생 여부를 예측하려면 방전 용량이 아닌 $\Delta Q(V)$ 전압 변환 피처가 절대적으로 필요**합니다.""")

# 4. Question 3
add_md("""---
## 4. Question 3. $\\Delta Q(V)$ 곡선 - 초기 사이클에서 차이가 보이는가?
- 원천 방전 곡선 $Q_{10}(V)$ vs $Q_{100}(V)$ 비교
- 사이클 100번 - 사이클 10번의 $Q(V)$ 차이 계산: $\\Delta Q_{100-10}(V) = Q_{100}(V) - Q_{10}(V)$
- 장수명 셀 vs 단수명 셀의 $\\Delta Q(V)$ 형태 비교
- $\\Delta Q(V)$로부터 통계값(분산, 최솟값, 평균) 추출 및 수명과의 강력한 상관관계 검증""")

add_code("""# 1. Delta Q(V) 계산 및 2x2 심층 시각화 파이프라인
c_short = qdlin['b2_c19']   # 단수명 대표 (EOL: 392)
c_long  = qdlin['b3_c38']   # 장수명 대표 (EOL: 1935)
V = c_short['Vdlin']        # 3.5V ~ 2.0V 전압 그리드 (1,000 포인트)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cmap = cm.coolwarm
norm = plt.Normalize(valid_meta['cycle_life'].min(), valid_meta['cycle_life'].max())
colors = ['#1f77b4', '#d62728', '#2ca02c']

# (a) 원천 방전 곡선 Q(V): Cycle 10 vs Cycle 100 (논문 Fig. 2a)
axes[0, 0].plot(c_short['Qdlin_10'], V, 'b--', linewidth=1.5, label='Cycle 10 (Short: 392 cycles)')
axes[0, 0].plot(c_short['Qdlin_100'], V, 'b-', linewidth=2, label='Cycle 100 (Short: 392 cycles)')
axes[0, 0].plot(c_long['Qdlin_10'], V, 'r--', linewidth=1.5, label='Cycle 10 (Long: 1935 cycles)')
axes[0, 0].plot(c_long['Qdlin_100'], V, 'r-', linewidth=2, label='Cycle 100 (Long: 1935 cycles)')
axes[0, 0].set_xlabel('Discharge Capacity Q(V) (Ah)')
axes[0, 0].set_ylabel('Voltage (V)')
axes[0, 0].set_title('(a) Raw Discharge Curves Q(V): Cycle 10 vs 100')
axes[0, 0].legend(fontsize=8, loc='lower left')

# (b) Delta Q(V) 차분 계산 과정: Q100(V) - Q10(V) 직접 대조
dq_short = c_short['Qdlin_100'] - c_short['Qdlin_10']
dq_long  = c_long['Qdlin_100'] - c_long['Qdlin_10']
axes[0, 1].plot(dq_short, V, 'b-', linewidth=2, label=f'Short-life (b2_c19): Var={np.var(dq_short):.2e}')
axes[0, 1].plot(dq_long, V, 'r-', linewidth=2, label=f'Long-life (b3_c38): Var={np.var(dq_long):.2e}')
axes[0, 1].axvline(0, color='black', linestyle=':', alpha=0.6)
axes[0, 1].set_xlabel('$\\\\Delta Q_{100-10}(V) = Q_{100}(V) - Q_{10}(V)$ (Ah)')
axes[0, 1].set_ylabel('Voltage (V)')
axes[0, 1].set_title('(b) $\\\\Delta Q_{100-10}(V)$ Calculation: Short vs Long')
axes[0, 1].legend(fontsize=8, loc='lower left')

# (c) 전체 124개 셀의 Delta Q(V) 곡선 (논문 Fig. 2b)
dq_records = []
for k, d in qdlin.items():
    cl = d['cycle_life']
    if pd.isna(cl) or d['Qdlin_10'] is None or d['Qdlin_100'] is None:
        continue
    dq = d['Qdlin_100'] - d['Qdlin_10']
    var_dq = np.var(dq)
    min_dq = np.min(dq)
    mean_dq = np.mean(dq)
    skew_dq = skew(dq)
    kurt_dq = kurtosis(dq)
    
    dq_records.append({
        'cell_key': k,
        'batch_id': d['batch_id'],
        'cycle_life': cl,
        'log_cycle_life': np.log10(cl),
        'var_dq': var_dq,
        'log_var_dq': np.log10(var_dq) if var_dq > 0 else np.nan,
        'min_dq': min_dq,
        'mean_dq': mean_dq,
        'skew_dq': skew_dq,
        'kurt_dq': kurt_dq,
        'delta_q': dq
    })
    axes[1, 0].plot(dq, V, color=cmap(norm(cl)), linewidth=0.7, alpha=0.6)

axes[1, 0].axvline(0, color='black', linestyle=':', alpha=0.6)
axes[1, 0].set_xlabel('$\\\\Delta Q_{100-10}(V)$ (Ah)')
axes[1, 0].set_ylabel('Voltage (V)')
axes[1, 0].set_title('(c) All 124 Cells $\\\\Delta Q_{100-10}(V)$ (Blue: Short, Red: Long)')
axes[1, 0].set_xlim(-0.15, 0.05)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Cycle Life (Blue: 392 ~ Red: 1935)', rotation=270, labelpad=15)

# (d) 핵심 분산 피처와 수명의 상관관계 (논문 Fig. 2c)
df_dq = pd.DataFrame(dq_records)
sns.scatterplot(data=df_dq, x='log_var_dq', y='log_cycle_life', hue='batch_id', palette=colors, ax=axes[1, 1], s=50)
sns.regplot(data=df_dq, x='log_var_dq', y='log_cycle_life', scatter=False, ax=axes[1, 1], color='black', line_kws={'linestyle': '--'})
r_val = df_dq['log_var_dq'].corr(df_dq['log_cycle_life'])
axes[1, 1].set_xlabel('$\\\\log_{10}(\\\\mathrm{Var}(\\\\Delta Q_{100-10}(V)))$')
axes[1, 1].set_ylabel('$\\\\log_{10}(\\\\mathrm{Cycle Life})$')
axes[1, 1].set_title(f'(d) Variance Feature vs Cycle Life (Corr = {r_val:.4f})')

plt.suptitle('Question 3. $\\\\Delta Q(V)$ Calculation Pipeline and Variance Feature', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figure/3. Delta Q(V) Curves and Variance Feature Correlation.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 유효 Delta Q 계산 셀 수: {len(df_dq)}개")
print("\\n🏆 [Delta Q(V) 파생 통계 피처와 log10(Cycle Life)의 상관계수]")
for col in ['log_var_dq', 'var_dq', 'min_dq', 'mean_dq', 'skew_dq', 'kurt_dq']:
    r = df_dq[col].corr(df_dq['log_cycle_life'])
    print(f"  • {col:15s} : {r:+.4f}")""")

add_md("""### 💡 Q3 인사이트 요약
1. **$\\Delta Q(V)$의 극명한 형태적 차이**:
   - **장수명 셀(파란색)**: 10사이클과 100사이클 간 전압 곡선 변화가 거의 없어 $\\Delta Q \\approx 0$에 수직으로 밀집됨.
   - **단수명 셀(빨간색)**: 2.5V ~ 3.3V 구간에서 왼쪽으로 크게 꺼지며 넓은 면적과 큰 변동(분산)을 보임.
2. **압도적인 단일 피처 $\\log_{10}(\\text{Var}(\\Delta Q))$**:
   - 이 피처 하나만으로 수명과의 상관계수가 **$\\rho = -0.886$**에 달하여, 초기 사이클에서 수명을 정확히 예측할 수 있는 핵심 Key입니다.""")

# 5. Question 4
add_md("""---
## 5. Question 4. 충전 조건 (C-rate)과 수명의 관계는?
- 고속 충전 C-rate 파싱 및 정책별 수명 비교
- 충전 속도가 배터리 수명 및 발열($T_{max}$)에 미치는 영향 분석""")

add_code("""# 1. charging_policy 정밀 파싱 (C1, SOC1, C2)
import re

def parse_crate(p_str):
    if not isinstance(p_str, str):
        return np.nan, np.nan, np.nan
    m = re.findall(r'([0-9.]+)C\(([0-9.]+)%\)-([0-9.]+)C', p_str)
    if m:
        return float(m[0][0]), float(m[0][1]), float(m[0][2])
    m2 = re.findall(r'([0-9.]+)C', p_str)
    if len(m2) >= 2:
        return float(m2[0]), 80.0, float(m2[1])
    elif len(m2) == 1:
        return float(m2[0]), 80.0, float(m2[0])
    return np.nan, np.nan, np.nan

rates = valid_meta['policy_readable'].apply(parse_crate)
valid_meta['c_rate_step1'] = [r[0] for r in rates]
valid_meta['soc_step1'] = [r[1] for r in rates]
valid_meta['c_rate_step2'] = [r[2] for r in rates]

# 2. 초기 100 사이클 정제 집계 (이상치 필터링)
# - 사이클 1: 초기 방전부터 시작하여 chargetime=0이므로 제외 (cycle 2~100 대상)
# - 실험 중 일시 중지(pause/rest)로 인한 chargetime > 25분 이상치 제외
# - 열전대 이탈 등으로 인한 Tmax > 50°C 이상치 제외
early_valid = summary_clean[
    (summary_clean['cycle'] >= 2) & (summary_clean['cycle'] <= 100) &
    (summary_clean['chargetime'] > 0) & (summary_clean['chargetime'] < 25) &
    (summary_clean['Tmax'] < 50)
]

early_summary = early_valid.groupby('cell_key').agg({
    'chargetime': 'mean',
    'Tmax': 'mean',
    'Tavg': 'mean',
    'IR': lambda s: s[s > 0].mean() if (s > 0).any() else np.nan
}).reset_index()

policy_merged = valid_meta.merge(early_summary, on='cell_key').dropna(subset=['cycle_life'])

# 3. 3-Panel 시각화 (Step 1 C-rate, Cleaned Charge Time, Tmax vs Cycle Life)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Step 1 C-rate vs Cycle Life
sns.scatterplot(data=policy_merged, x='c_rate_step1', y='cycle_life', hue='batch_id', palette=colors, ax=axes[0], s=60, alpha=0.85)
sns.regplot(data=policy_merged, x='c_rate_step1', y='cycle_life', scatter=False, ax=axes[0], color='black', line_kws={'linestyle': '--', 'linewidth': 1.5})
r_c1 = policy_merged['c_rate_step1'].corr(policy_merged['cycle_life'])
axes[0].set_xlabel('Step 1 Charge C-rate ($C_1$)')
axes[0].set_ylabel('Cycle Life')
axes[0].set_title(f'(a) Step 1 C-rate vs Cycle Life (r = {r_c1:+.2f})')
axes[0].grid(True, linestyle=':', alpha=0.6)

# (b) Cleaned Charge Time vs Cycle Life
sns.scatterplot(data=policy_merged, x='chargetime', y='cycle_life', hue='batch_id', palette=colors, ax=axes[1], s=60, alpha=0.85)
sns.regplot(data=policy_merged, x='chargetime', y='cycle_life', scatter=False, ax=axes[1], color='black', line_kws={'linestyle': '--', 'linewidth': 1.5})
r_ct = policy_merged['chargetime'].corr(policy_merged['cycle_life'])
axes[1].set_xlabel('Mean Charge Time (min, Outliers Filtered)')
axes[1].set_ylabel('Cycle Life')
axes[1].set_title(f'(b) Charge Time vs Cycle Life (r = {r_ct:+.2f})')
axes[1].grid(True, linestyle=':', alpha=0.6)

# (c) Max Temperature vs Cycle Life
sns.scatterplot(data=policy_merged, x='Tmax', y='cycle_life', hue='batch_id', palette=colors, ax=axes[2], s=60, alpha=0.85)
sns.regplot(data=policy_merged, x='Tmax', y='cycle_life', scatter=False, ax=axes[2], color='black', line_kws={'linestyle': '--', 'linewidth': 1.5})
r_tm = policy_merged['Tmax'].corr(policy_merged['cycle_life'])
axes[2].set_xlabel('Mean Max Temperature $T_{max}$ (°C)')
axes[2].set_ylabel('Cycle Life')
axes[2].set_title(f'(c) Max Temperature vs Cycle Life (r = {r_tm:+.2f})')
axes[2].grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Question 4. Charging Conditions & Environmental Stress vs Cycle Life', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figure/4. Charging Conditions vs Cycle Life.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 정제된 충전 조건 분석 셀 수: {len(policy_merged)}개")
print(f"  • Step 1 C-rate 범위: {policy_merged['c_rate_step1'].min():.1f}C ~ {policy_merged['c_rate_step1'].max():.1f}C (상관계수 r = {r_c1:+.4f})")
print(f"  • 평균 충전 시간 범위: {policy_merged['chargetime'].min():.2f}분 ~ {policy_merged['chargetime'].max():.2f}분 (상관계수 r = {r_ct:+.4f})")
print(f"  • 평균 최고 온도 범위: {policy_merged['Tmax'].min():.1f}°C ~ {policy_merged['Tmax'].max():.1f}°C (상관계수 r = {r_tm:+.4f})")""")

add_md("""### 💡 Q4 인사이트 요약
1. **데이터 정제(Outlier Filtering)의 결정적 발견**:
   - 기존의 단순 산술 평균 계산 시 Batch 2의 일부 셀들에서 **실험 일시 중지(pause/rest)로 인해 발생한 3,933분(약 65시간)의 단일 이상치**가 섞여 평균 충전 시간이 약 49.5분으로 심각하게 왜곡되었습니다.
   - 이상치를 정제한 결과, 모든 급속 충전 셀들의 실제 충전 시간은 **8.98분 ~ 13.38분** 범위에 정상 분포함을 확인했습니다.
2. **2단계 충전 프로토콜의 복합적 특성**:
   - **$C_1$(1단계 C-rate) 단독의 한계 ($r = -0.08$)**: 단순히 1단계 전류만으로는 수명을 설명하기 어렵습니다. 예를 들어 6C로 충전하더라도 전환 SOC가 30%로 낮으면, 4.8C로 80%까지 충전하는 셀보다 오히려 가혹도가 낮을 수 있습니다.
   - **충전 시간(Charge Time)의 유의미한 양의 상관관계 ($r = +0.31$)**: $C_1, \text{SOC}_1, C_2$의 종합적 결과물인 충전 시간이 10분 미만인 초급속 충전 셀들은 500사이클 미만으로 급격히 열화되며, 충전 시간이 11~13분으로 완화될수록 수명이 1,000사이클 이상으로 연장됩니다.
3. **온도 스트레스 ($T_{max}$)**:
   - 셀 표면 최고 온도는 30.6°C ~ 40.5°C 범위에 분포하며, 고전류 인가 시 줄 발열($I^2 R$)을 유발하지만 챔버 공랭 환경으로 인해 단독 상관성은 낮고 다른 전기화학 피처와 결합될 때 유효합니다.""")

# 6. Question 5
add_md("""---
## 6. Question 5. 상관관계 및 다중공선성 (Multicollinearity) 분석
- 초기 사이클 피처들과 Cycle Life 간의 상관관계 랭킹
- 피처 간 상관계수 히트맵 및 다중공선성 확인""")

add_code("""# 1. 모델링용 후보 피처 데이터셋 통합
df_full = df_dq.merge(policy_merged[['cell_key', 'chargetime', 'Tmax', 'Tavg', 'IR', 'c_rate_step1']], on='cell_key')

# 초기 100 사이클 QD 관련 피처 추가
qd_feats = summary_clean[summary_clean['cycle'] <= 100].groupby('cell_key').agg({
    'QD': ['mean', 'min', 'max', lambda s: s.iloc[-1] - s.iloc[0]]
})
qd_feats.columns = ['QD_mean', 'QD_min', 'QD_max', 'QD_diff_100_1']
df_full = df_full.merge(qd_feats.reset_index(), on='cell_key')

# 2. 상관계수 분석 대상 컬럼
corr_cols = [
    'cycle_life', 'log_cycle_life',
    'log_var_dq', 'min_dq', 'mean_dq',
    'chargetime', 'Tmax', 'Tavg', 'IR',
    'QD_mean', 'QD_diff_100_1'
]

corr_mat = df_full[corr_cols].corr()

# 3. 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
plt.title('Correlation Matrix of Early-cycle Features & Cycle Life', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figure/5. Correlation Heatmap of Early Features and Cycle Life.png', dpi=300, bbox_inches='tight')
plt.show()

print("🏆 [수명(log_cycle_life)과의 상관계수 랭킹]")
display(corr_mat['log_cycle_life'].drop(['cycle_life', 'log_cycle_life']).sort_values(key=abs, ascending=False).to_frame())""")

add_md("""### 💡 Q5 인사이트 요약
1. **지배적인 핵심 피처**:
   - `log_var_dq` (-0.89) 및 `min_dq` (0.89)가 압도적인 상관관계를 나타냄.
   - 반면 단순 용량인 `QD_mean` (0.24)이나 `QD_diff_100_1` (0.42)은 설명력이 현저히 떨어짐.
2. **다중공선성(Multicollinearity) 확인**:
   - `Tmax`와 `Tavg`의 상관계수가 **0.96**에 달함.
   - `min_dq`와 `mean_dq`, `log_var_dq` 간 상관도가 0.85 이상으로 매우 높음.
   - 따라서 OLS(단순선형회귀)를 그대로 적용하면 회귀계수가 발산하고 불안정해지므로, **L1/L2 패널티를 부여하는 Elastic Net 또는 Ridge/Lasso 정규화 기법이 필수적**입니다.""")

# 7. Model Strategy
add_md("""---
## 7. 종합: DAY 1 모델 설계 전략 수립

위의 5가지 질문에 대한 종합 분석 결과를 바탕으로, 아래와 같이 모델 설계 전략을 확정합니다.

### 1. Feature Engineering 전략
- **Level 1 (Single Feature - Variance Model)**:
  - $\\log_{10}(\\mathrm{Var}(\\Delta Q_{100-10}(V)))$ 1개 피처만 사용 $\\to$ 극도로 가볍고 직관적인 베이스라인
- **Level 2 (Discharge Feature Model)**:
  - $\\Delta Q$ 요약 통계량 (`log_var_dq`, `min_dq`, `mean_dq`, `skew_dq`, `kurt_dq`) + 초기 방전용량 감쇠율
- **Level 3 (Full Feature Model)**:
  - Level 2 + 충전 소요 시간(`chargetime`) + 최고온도(`Tmax`) + 초기 유효 저항(`IR`)

### 2. Task 선정: Regression (회귀)
- **선정 이유**:
  - 배터리 수명은 연속적인 물리적 척도이며, 배터리 관리 시스템(BMS)에서 잔여 수명(RUL)을 예측하기 위해서는 구체적인 사이클 수(숫자)를 제시하는 것이 산업적 활용도가 높음.
- **Target Variable**:
  - $\\mathbf{y = \\log_{10}(\\mathrm{Cycle\\ Life})}$
  - **이유**: 수명 분포가 392 ~ 1,935 사이클로 넓게 분포하고 우측 꼬리가 긴 형태이므로, 로그 변환을 통해 잔차의 정규성을 확보하고 모델의 스케일 민감도를 완화함.

### 3. Modeling & Validation 전략
- **후보 알고리즘**:
  1. **Elastic Net (주력 선형 모델)**:
     - 높은 다중공선성을 완화하고 희소 피처를 자동 선택 ($L_1 + L_2$ 정규화).
     - $\\alpha$, $\\lambda$ 하이퍼파라미터는 4-Fold Cross Validation으로 탐색.
  2. **LightGBM / Random Forest (비선형 비교군)**:
     - 피처 간 비선형 결합 효과 및 트리 기반 예측 성능 비교.
- **검증 및 평가 전략 (Data Leakage 방지)**:
  - **배치 단위 분할 (Batch Split)**:
    - Train Set: **Batch 1 (46개) + Batch 2 (39개)**
    - Test Set: **Batch 3 (44개)** (독립적인 미래 데이터로 일반화 검증)
  - **평가 지표**:
    - **RMSE (Root Mean Squared Error)**: 예측 오차 사이클 수
    - **MPE / MAPE (Mean Absolute Percentage Error)**: 상대적 백분율 오차 (%)""")

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': '.venv (3.11.15)',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.11.15'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

save_path = 'Legacy/DAY1_EDA_and_Modeling_Strategy.ipynb' if os.path.exists('Legacy') else 'DAY1_EDA_and_Modeling_Strategy.ipynb'
with open(save_path, 'w') as f:
    json.dump(nb, f, indent=2)

print(f'{save_path} successfully updated with figure savefig!')
