import re
import time
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
        chunks = re.split(r'(?<=[.!?])\s+', text)
        # clearing chunks
        clear_chunks = []
        for chunk in chunks:
            clear_chunks.append(' '.join(chunk.split()))

        return clear_chunks

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
        # split text based on \n's
        split_text = re.split(r'\n{1,2}', text)

        # if the splitted \n's chunks are bigger than chunk_size => split them on chunks of 64 tokens size
        chunks = []
        enc = tiktoken.get_encoding('cl100k_base')
        for chunk in split_text:
            tokens = enc.encode(chunk)
            if len(tokens) > self.chunk_size:
                for i in range(0, len(tokens), self.chunk_size):
                    chunk = enc.decode(tokens[i:i + self.chunk_size])
                    chunks.append(chunk)
            else:
                chunks.append(chunk)

        return chunks

    def split_text(self, text: str) -> List[str]:
        return super().split_text(text)
