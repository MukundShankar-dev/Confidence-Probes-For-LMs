#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader for indexed splits (efficient, no data duplication)

Usage:
    from scripts.load_indexed_split import load_split, get_split_info
    
    # Iterate through split
    for row in load_split("data/probe_splits_standard/qwen2.5_7b", "train"):
        features = row["h_last_256"]  # etc.
        label = row["correct"]
    
    # Load all at once
    train_data = list(load_split("path/to/split", "train"))
    
    # Get info without loading
    info = get_split_info("path/to/split")
"""

import json
from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple
import bisect


class IndexedSplitLoader:
    """Efficient loader for indexed splits"""
    
    def __init__(self, split_dir: str):
        self.split_dir = Path(split_dir)
        
        with open(self.split_dir / "split_map.json") as f:
            self.split_map = json.load(f)
        
        with open(self.split_dir / "file_index.json") as f:
            file_index = json.load(f)
            self.file_metadata = file_index["files"]
            self.total_rows = file_index["total_rows"]
        
        self._build_lookup()
    
    def _build_lookup(self):
        """Build fast lookup for binary search"""
        self.file_boundaries = [(meta["start_idx"], i) for i, meta in enumerate(self.file_metadata)]
        self.file_boundaries.sort()
    
    def _get_file_and_line(self, global_idx: int) -> Tuple[int, int]:
        """Map global index to (file_idx, line_in_file)"""
        starts = [start for start, _ in self.file_boundaries]
        file_idx = bisect.bisect_right(starts, global_idx) - 1
        if file_idx < 0:
            file_idx = 0
        meta = self.file_metadata[file_idx]
        line_in_file = global_idx - meta["start_idx"]
        return file_idx, line_in_file
    
    def load_split(self, split_name: str) -> Iterator[Dict[Any, Any]]:
        """Load and yield rows for a split"""
        split_indices = self.split_map["splits"][split_name]
        
        # Group by file for efficient reading
        indices_by_file = {}
        for global_idx in split_indices:
            file_idx, line_num = self._get_file_and_line(global_idx)
            indices_by_file.setdefault(file_idx, []).append((global_idx, line_num))
        
        # Read files
        rows_dict = {}
        for file_idx, indices_in_file in indices_by_file.items():
            filepath = self.file_metadata[file_idx]["path"]
            line_nums_needed = {line_num for _, line_num in indices_in_file}
            
            with open(filepath, encoding="utf-8") as f:
                current_line = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("overall") is True:
                            continue
                        if current_line in line_nums_needed:
                            for global_idx, ln in indices_in_file:
                                if ln == current_line:
                                    rows_dict[global_idx] = obj
                        current_line += 1
                    except Exception:
                        continue
        
        # Yield in split order
        for global_idx in split_indices:
            if global_idx in rows_dict:
                yield rows_dict[global_idx]
    
    def get_split_size(self, split_name: str) -> int:
        return len(self.split_map["splits"][split_name])


def load_split(split_dir: str, split_name: str) -> Iterator[Dict[Any, Any]]:
    """Load a split (train/val/test)"""
    loader = IndexedSplitLoader(split_dir)
    yield from loader.load_split(split_name)


def load_split_as_list(split_dir: str, split_name: str) -> List[Dict]:
    """Load entire split into memory"""
    return list(load_split(split_dir, split_name))


def get_split_info(split_dir: str) -> Dict:
    """Get split metadata"""
    loader = IndexedSplitLoader(split_dir)
    return {
        "total_rows": loader.total_rows,
        "split_sizes": {name: loader.get_split_size(name) for name in ["train", "val", "test"]},
        "num_files": len(loader.file_metadata),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python load_indexed_split.py <split_dir> <split_name>")
        sys.exit(1)
    
    split_dir, split_name = sys.argv[1], sys.argv[2]
    info = get_split_info(split_dir)
    print(f"Split info: {info}")
    print(f"\nLoading {split_name}...")
    count = 0
    for row in load_split(split_dir, split_name):
        count += 1
        if count <= 3:
            print(f"Row {count}: {row.get('question', 'N/A')[:60]}...")
    print(f"Total: {count:,} rows")
