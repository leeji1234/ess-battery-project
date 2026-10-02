import os
import pickle
import warnings
import numpy as np
import pandas as pd
import mat73

warnings.filterwarnings('ignore')

# 프로젝트 루트 디렉터리로 작업 디렉터리 자동 이동 (Legacy/ 내부 실행 호환성)
if not os.path.exists('archive') and os.path.exists('../archive'):
    os.chdir('..')

DATA_DIR = 'archive'
BATCH_FILES = {
    1: '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    2: '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    3: '2018-04-12_batchdata_updated_struct_errorcorrect.mat'
}

all_summary_records = []
all_cell_meta = []
qdlin_dict = {}  # {cell_key: {'Vdlin': ..., 'Qdlin_10': ..., 'Qdlin_100': ..., 'cycle_life': ...}}

print("=== Starting Batch 1, 2, 3 Data Extraction ===")

for batch_id, filename in BATCH_FILES.items():
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\n[Batch {batch_id}] Loading {filename} ...")
    mat = mat73.loadmat(filepath)
    batch = mat['batch']
    
    # mat73 dict-of-lists 또는 list-of-dicts 처리
    if isinstance(batch, dict):
        keys = list(batch.keys())
        n_cells = len(batch[keys[0]])
        cell_list = [{k: batch[k][i] for k in keys} for i in range(n_cells)]
    else:
        cell_list = batch

    print(f"[Batch {batch_id}] Loaded {len(cell_list)} cells. Processing...")

    for cid, cell in enumerate(cell_list):
        cell_key = f"b{batch_id}_c{cid}"
        
        # 메타데이터
        cl_raw = cell.get('cycle_life', np.nan)
        try:
            cycle_life = float(np.squeeze(cl_raw))
        except (ValueError, TypeError):
            cycle_life = np.nan
        policy_raw = str(cell.get('policy') or 'unknown')
        policy_read = str(cell.get('policy_readable') or policy_raw)
        
        all_cell_meta.append({
            'cell_key': cell_key,
            'batch_id': batch_id,
            'cell_id': cid,
            'cycle_life': cycle_life,
            'policy': policy_raw,
            'policy_readable': policy_read
        })

        # Summary 시계열 데이터
        summary = cell.get('summary', {})
        if isinstance(summary, dict) and 'QDischarge' in summary:
            qd = np.array(summary['QDischarge'])
            qc = np.array(summary.get('QCharge', np.zeros_like(qd)))
            ir = np.array(summary.get('IR', np.zeros_like(qd)))
            tmax = np.array(summary.get('Tmax', np.zeros_like(qd)))
            tavg = np.array(summary.get('Tavg', np.zeros_like(qd)))
            tmin = np.array(summary.get('Tmin', np.zeros_like(qd)))
            ct = np.array(summary.get('chargetime', np.zeros_like(qd)))
            
            n_cycles = len(qd)
            for c in range(n_cycles):
                all_summary_records.append({
                    'cell_key': cell_key,
                    'batch_id': batch_id,
                    'cell_id': cid,
                    'cycle': c + 1,
                    'cycle_life': cycle_life,
                    'policy': policy_read,
                    'QD': qd[c],
                    'QC': qc[c],
                    'IR': ir[c],
                    'Tmax': tmax[c],
                    'Tavg': tavg[c],
                    'Tmin': tmin[c],
                    'chargetime': ct[c]
                })

        # Qdlin (Cycle 10 & 100) 추출 for ΔQ(V)
        Vdlin = np.squeeze(cell.get('Vdlin', None))
        cycles_raw = cell.get('cycles', None)
        
        qdlin_10 = None
        qdlin_100 = None
        
        if cycles_raw is not None:
            if isinstance(cycles_raw, dict) and 'Qdlin' in cycles_raw:
                qdlin_all = cycles_raw['Qdlin']
                # 0-indexed: 9번이 cycle 10, 99번이 cycle 100
                if len(qdlin_all) > 9:
                    qdlin_10 = np.array(qdlin_all[9]) if qdlin_all[9] is not None else None
                if len(qdlin_all) > 99:
                    qdlin_100 = np.array(qdlin_all[99]) if qdlin_all[99] is not None else None
            elif isinstance(cycles_raw, list):
                if len(cycles_raw) > 9 and isinstance(cycles_raw[9], dict):
                    qdlin_10 = np.array(cycles_raw[9].get('Qdlin'))
                if len(cycles_raw) > 99 and isinstance(cycles_raw[99], dict):
                    qdlin_100 = np.array(cycles_raw[99].get('Qdlin'))

        qdlin_dict[cell_key] = {
            'batch_id': batch_id,
            'cell_id': cid,
            'cycle_life': cycle_life,
            'policy': policy_read,
            'Vdlin': Vdlin,
            'Qdlin_10': qdlin_10,
            'Qdlin_100': qdlin_100
        }

# 데이터프레임 변환
df_summary = pd.DataFrame(all_summary_records)
df_meta = pd.DataFrame(all_cell_meta)

# 결과 저장
os.makedirs('processed_data', exist_ok=True)
df_summary.to_pickle('processed_data/batch_summary.pkl')
df_summary.to_csv('processed_data/batch_summary.csv.gz', index=False, compression='gzip')
df_meta.to_csv('processed_data/batch_meta.csv', index=False)

with open('processed_data/qdlin_cycle10_100.pkl', 'wb') as f:
    pickle.dump(qdlin_dict, f)

print("\n=== Preprocessing Complete! ===")
print(f"Total Cells Extracted   : {len(df_meta)}")
print(f"Total Summary Rows      : {len(df_summary)}")
print(f"Summary Saved to        : processed_data/batch_summary.pkl & batch_summary.csv.gz")
print(f"Meta Saved to           : processed_data/batch_meta.csv")
print(f"Qdlin Dictionary Saved  : processed_data/qdlin_cycle10_100.pkl")
