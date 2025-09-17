from typing import List, Dict


class Retriever:
    def __init__(self, backend: str = "wikipedia", top_k: int = 5, max_passage_len: int = 384):
        self.backend = backend
        self.top_k = top_k
        self.max_len = max_passage_len
        if backend == "wikipedia":
            import wikipedia
            self.wp = wikipedia
        else:
            # TODO: add FAISS over a local corpus
            self.wp = None

    def fetch(self, query: str) -> List[str]:
        if self.backend == "wikipedia":
            try:
                titles = self.wp.search(query)[: self.top_k]
                texts = []
                for t in titles:
                    try:
                        page = self.wp.page(t, auto_suggest=False)
                        texts.append(page.content[: self.max_len])
                    except Exception:
                        continue
                return texts
            except Exception:
                return []
        return []
