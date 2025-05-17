# file_manager.py

import os
import sys
import pickle
import threading
from rapidfuzz import process, fuzz

INDEX_FILE = 'file_index.pkl'

class FileManager:
    def __init__(self):
        self.file_index = {}
        threading.Thread(target=self._build_index, daemon=True).start()

    def _build_index(self):
        root = os.path.abspath(os.sep)
        self.file_index = self._index_files(root)
        # persist
        try:
            with open(INDEX_FILE, 'wb') as f:
                pickle.dump(self.file_index, f)
        except:
            pass

    def _index_files(self, root_dir):
        idx = {}
        for dirpath, _, files in os.walk(root_dir):
            for fname in files:
                key = fname.lower()
                idx.setdefault(key, []).append(os.path.join(dirpath, fname))
        return idx

    def load_index(self):
        if os.path.exists(INDEX_FILE):
            try:
                with open(INDEX_FILE, 'rb') as f:
                    self.file_index = pickle.load(f)
                    return True
            except:
                pass
        return False

    def find_paths(self, name):
        name = name.lower()
        # exact
        if name in self.file_index:
            return self.file_index[name]
        # fuzzy
        choices = list(self.file_index.keys())
        match, score, _ = process.extractOne(name, choices, scorer=fuzz.ratio)
        if score > 60:
            return self.file_index.get(match, [])
        return []

    def open(self, path):
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            os.system(f"open '{path}'")
        else:
            os.system(f"xdg-open '{path}' &")
