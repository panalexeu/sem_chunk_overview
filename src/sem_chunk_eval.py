import time
import sys

from chunking_evaluation import DatasetEvaluation, Dataset
from chromadb.utils import embedding_functions
from rich import print

from window_sem_chunker import WindowSemChunker, TokenWindowSemChunker

evaluation = DatasetEvaluation(
    datasets=[
        Dataset.STATE_OF_THE_UNION,
        Dataset.PUBMED,
        Dataset.CHATLOGS,
        Dataset.FINANCE,
        Dataset.WIKITEXTS
    ]
)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

if __name__ == '__main__':
    match sys.argv[1]:
        case 'token':
            chunker = TokenWindowSemChunker()
        case 'win':
            chunker = WindowSemChunker()
        case _:
            chunker = WindowSemChunker()

    start = time.time()
    results = evaluation.run(chunker, ef)
    end = time.time()

    print(results)
    print(f'TIME: {end - start}')
