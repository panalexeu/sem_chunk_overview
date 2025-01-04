import re
from typing import List

from chunking_evaluation import BaseChunker
from sentence_transformers import SentenceTransformer


class WindowSemChunker(BaseChunker):
    thresh = 0.9
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def split_text(self, text: str) -> List[str]:
        split_text = re.split(r'(?<=[.!?;:])\s+', text)

        prev = ''
        init = text[0]
        chunks = []

        for sentence in split_text:
            res = prev + ' ' + sentence
            dist = self.model.similarity(
                self.model.encode(init),
                self.model.encode(res)
            )

            if dist < self.thresh:
                chunks.append(prev)
                prev = sentence
                init = sentence
            else:
                prev = res

        if prev not in chunks:
            chunks.append(prev)

        return chunks
