import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import mat73

warnings.filterwarnings('ignore')

def get_project_root():
    """
    프로젝트 루트 디렉토리(DSminiproj)의 절대 경로를 반환합니다.
    """
    curr = os.path.dirname(os.path.abspath(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, 'processed_data')) or os.path.exists(os.path.join(curr, 'requirements.txt')):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def extract_and_save_data(data_dir=None, output_dir=None):
    """
    MATLAB .mat 배치 데이터를 추출하여 가벼운 pickle 및 csv 포맷으로 저장합니다.
    """
    root = get_project_root()
    data_dir = data_dir or os.path.join(root, 'archive')
    output_dir = output_dir or os.path.join(root, 'processed_data')
    
    os.makedirs(output_dir, exist_ok=True)
    
    batch_files = {
        1: '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
        2: '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
        3: '2018-04-12_batchdata_updated_struct_errorcorrect.mat'
    }
    
    all_summary_records = []
    all_cell_meta = []
    qdlin_dict = {}

    print(f"=== Raw MATLAB Data Extraction (Root: {root}) ===")
    for batch_id, filename in batch_files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"[Warning] {filepath} does not exist. Skipping.")
            continue
            
        print(f"[Batch {batch_id}] Loading {filename} ...")
        mat = mat73.loadmat(filepath)
        batch = mat['batch']

        if isinstance(batch, dict):
            keys = list(batch.keys())
            n_cells = len(batch[keys[0]])
            cell_list = [{k: batch[k][i] for k in keys} for i in range(n_cells)]
        else:
            cell_list = batch

        for cid, cell in enumerate(cell_list):
            cell_key = f"b{batch_id}_c{cid}"
            
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

            Vdlin = np.squeeze(cell.get('Vdlin', None))
            cycles_raw = cell.get('cycles', None)
            
            qdlin_10 = None
            qdlin_100 = None
            
            if cycles_raw is not None:
                if isinstance(cycles_raw, dict) and 'Qdlin' in cycles_raw:
                    qdlin_all = cycles_raw['Qdlin']
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

    df_summary = pd.DataFrame(all_summary_records)
    df_meta = pd.DataFrame(all_cell_meta)

    df_summary.to_pickle(os.path.join(output_dir, 'batch_summary.pkl'))
    df_summary.to_csv(os.path.join(output_dir, 'batch_summary.csv.gz'), index=False, compression='gzip')
    df_meta.to_csv(os.path.join(output_dir, 'batch_meta.csv'), index=False)
    with open(os.path.join(output_dir, 'qdlin_cycle10_100.pkl'), 'wb') as f:
        pickle.dump(qdlin_dict, f)

    print(f"Data saved to {output_dir}/")
    return df_meta, df_summary, qdlin_dict

def load_processed_data(processed_dir=None):
    """
    미리 전처리된 요약 데이터 및 Qdlin을 로드합니다.
    어디서 호출하더라도 프로젝트 루트를 찾아 올바르게 로드합니다.
    """
    root = get_project_root()
    if processed_dir is None:
        processed_dir = os.path.join(root, 'processed_data')
    elif not os.path.isabs(processed_dir) and not os.path.exists(processed_dir):
        processed_dir = os.path.join(root, processed_dir)

    meta_path = os.path.join(processed_dir, 'batch_meta.csv')
    summary_path = os.path.join(processed_dir, 'batch_summary.pkl')
    qdlin_path = os.path.join(processed_dir, 'qdlin_cycle10_100.pkl')

    if not (os.path.exists(meta_path) and os.path.exists(summary_path) and os.path.exists(qdlin_path)):
        print("Processed files not found. Running extraction...")
        return extract_and_save_data(output_dir=processed_dir)

    meta = pd.read_csv(meta_path)
    summary = pd.read_pickle(summary_path)
    with open(qdlin_path, 'rb') as f:
        qdlin = pickle.load(f)

    return meta, summary, qdlin

if __name__ == '__main__':
    meta, summary, qdlin = load_processed_data()
    print(f"Loaded: Meta={len(meta)}, Summary={len(summary)}, Qdlin={len(qdlin)}")
