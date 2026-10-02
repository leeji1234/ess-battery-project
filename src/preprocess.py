# 데이터 전처리 및 로더 모듈 : 파일 불러오기
import os
import sys

legacy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Legacy'))
if legacy_dir not in sys.path:
    sys.path.insert(0, legacy_dir)

try:
    from Legacy.preprocess import extract_and_save_data, load_processed_data, get_project_root
    # load_processed_data() : 전처리된 메타데이터 / 시계열 데이터 / Q(V) 변화량 - csv, pkl 파일 불러오기
    # get_project_root() : 상위 directory를 탐색해 데이터 폴더 경로를 자동으로 매핑
    # extract_and_save_data() : 비교적 무거운 mat 파일을 정리하여 pkl 파일로 만들기 

except ImportError:
    # 직접 파일 경로로부터 import 시도
    import importlib.util
    spec = importlib.util.spec_from_file_location("legacy_preprocess", os.path.join(legacy_dir, "preprocess.py"))
    legacy_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_mod)
    extract_and_save_data = legacy_mod.extract_and_save_data
    load_processed_data = legacy_mod.load_processed_data
    get_project_root = legacy_mod.get_project_root

__all__ = ['extract_and_save_data', 'load_processed_data', 'get_project_root']

if __name__ == '__main__':
    meta, summary, qdlin = load_processed_data()
    print(f"[src/preprocess.py -> Legacy/preprocess.py] Loaded: Meta={len(meta)}, Summary={len(summary)}, Qdlin={len(qdlin)}")
