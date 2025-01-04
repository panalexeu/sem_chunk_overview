import time

from chunking_evaluation import DatasetEvaluation, Dataset
from chromadb.utils import embedding_functions
from rich import print

from window_sem_chunker import WindowSemChunker

chunker = WindowSemChunker()
evaluation = DatasetEvaluation(
    datasets=[
        Dataset.STATE_OF_THE_UNION
    ]
)

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

if __name__ == '__main__':
    start = time.time()
    results = evaluation.run(chunker, ef)
    end = time.time()

    print(results)
    print(f'TIME: {end - start}')
