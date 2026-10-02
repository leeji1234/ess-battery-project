import os
import sys

# Legacy/preprocess.py의 핵심 함수들을 re-export하여 완벽한 역호환성을 보장합니다.
legacy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Legacy'))
if legacy_dir not in sys.path:
    sys.path.insert(0, legacy_dir)

try:
    from Legacy.preprocess import extract_and_save_data, load_processed_data, get_project_root
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
