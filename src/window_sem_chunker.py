import re
from typing import List
from overrides import override

import tiktoken
from chunking_evaluation import BaseChunker
from sentence_transformers import SentenceTransformer
from rich import print


class WindowSemChunker(BaseChunker):

    def __init__(
            self,
            thresh=0.9,
            model: SentenceTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    ):
        self.thresh = thresh
        self.model = model

    def _create_chunks(self, text: str) -> list[str]:
        return re.split(r'(?<=[.!?])\s+', text)

    def split_text(self, text: str) -> List[str]:
        split_text = self._create_chunks(text)

        prev = ''
        init = split_text[0]
        chunks = []

        for sentence in split_text:
            res = prev + ' ' + sentence
            dist = self.model.similarity(
                self.model.encode(init),
                self.model.encode(res)
            )

            if dist < self.thresh:
                print('formed chunk: ', prev)
                print('brk: ', sentence)
                print('dist: ', dist)
                print('=' * 50)

                chunks.append(prev)
                prev = sentence
                init = sentence
            else:
                prev = res

        if prev not in chunks:
            chunks.append(prev)

        return chunks


class TokenWindowSemChunker(WindowSemChunker):

    def __init__(
            self,
            chunk_size: int = 16,
            thresh=0.9,
            model: SentenceTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    ):
        super().__init__(thresh, model)
        self.chunk_size = chunk_size

    @override
    def _create_chunks(self, text: str) -> list[str]:
        enc = tiktoken.get_encoding('cl100k_base')
        tokens = enc.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk = tokens[i:i + self.chunk_size]
            print(chunk)
            print('-' * 50)
            chunks.append(enc.decode(chunk))

        return chunks

    def split_text(self, text: str) -> List[str]:
        return super().split_text(text)
