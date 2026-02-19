#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학습된 모델 파일들을 '결과물' 폴더에 복사하는 스크립트
"""
import os
import shutil
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.resolve()

# 복사할 파일 목록
files_to_copy = {
    'models/best.pt': '결과물/yolo/best.pt',
    'runs/cnn/stage1/best.keras': '결과물/cnn/stage1/best.keras',
    'runs/cnn/stage2/best.keras': '결과물/cnn/stage2/best.keras',
    'runs/cnn/stage2/class_names.json': '결과물/cnn/stage2/class_names.json',
}

if __name__ == '__main__':
    print("=" * 60)
    print("모델 파일 복사 시작")
    print(f"프로젝트 경로: {PROJECT_ROOT}")
    print("=" * 60)

    # 폴더 생성
    for dst_path in files_to_copy.values():
        dst = PROJECT_ROOT / dst_path
        dst.parent.mkdir(parents=True, exist_ok=True)

    # 파일 복사
    copied = []
    missing = []

    for src_rel, dst_rel in files_to_copy.items():
        src = PROJECT_ROOT / src_rel
        dst = PROJECT_ROOT / dst_rel
        
        if src.exists():
            try:
                shutil.copy2(src, dst)
                size_mb = src.stat().st_size / (1024 * 1024)
                print(f"✓ {src_rel}")
                print(f"  → {dst_rel} ({size_mb:.2f} MB)")
                copied.append((src_rel, dst_rel))
            except Exception as e:
                print(f"✗ {src_rel} → 복사 실패: {e}")
                missing.append(src_rel)
        else:
            print(f"✗ {src_rel} (파일 없음)")
            missing.append(src_rel)

    print("=" * 60)
    print(f"복사 완료: {len(copied)}개 파일")
    if missing:
        print(f"누락/실패: {len(missing)}개")
        for f in missing:
            print(f"  - {f}")
    print(f"결과물 폴더: {PROJECT_ROOT / '결과물'}")
    print("=" * 60)
