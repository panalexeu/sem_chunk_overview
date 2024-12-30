import re
from typing import List

from chunking_evaluation import BaseChunker, GeneralEvaluation
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from rich import print


class WindowSemChunker(BaseChunker):
    thresh = 0.88
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def split_text(self, text: str) -> List[str]:
        split_text = re.split(r'(?<=[.!?])\s+', text)

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
                # print(f'prev: {prev}\nbreakpoint sentence: {sentence}\ndist: {dist}\nchunks count: {len(chunks) + 1}')
                # print('=' * 25)

                chunks.append(prev)
                prev = sentence
                init = sentence
            else:
                prev = res

        if prev not in chunks:
            chunks.append(prev)

        return chunks


chunker = WindowSemChunker()
evaluation = GeneralEvaluation()

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


if __name__ == '__main__':
    results = evaluation.run(chunker, ef)
    print(results)
