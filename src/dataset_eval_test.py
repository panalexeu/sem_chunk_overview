import time

from chunking_evaluation import DatasetEvaluation, GeneralEvaluation, Dataset
from chunking_evaluation.chunking import FixedTokenChunker
from chromadb.utils import embedding_functions
from rich import print

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chunker = FixedTokenChunker()
eval = DatasetEvaluation(
    datasets=[
        Dataset.WIKITEXTS,
        Dataset.FINANCE,
    ]
)

if __name__ == '__main__':
    start = time.time()
    results = eval.run(chunker, ef)
    end = time.time()

    print(results)
    print(f'TIME: {end - start}')
