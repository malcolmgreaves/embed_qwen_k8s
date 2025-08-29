from contextlib import contextmanager
from datetime import datetime
import json
import os
import sys
from typing import TypedDict
import torch
import daft
from daft import col, DataFrame, Series, DataType
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer


if len(sys.argv) > 1 and (sys.argv[1].lower() == "-h" or sys.argv[1].lower() == "--help"):
    print("Usage: python -m cc_embed.main_pipeline")
    print("")
    print("Configure behabvior by setting the following environment variables:")
    print("  INPUT: Path to parquet files of Common Crawl data. Supports S3 and local paths.")
    print("  OUTPUT: Path to write output chnked text + embeddings. Supports S3 and local paths.")
    print("  IS_PARQUET: Perform read_parquet on the input. Defaults to read_warc.")
    print("  LIMIT: Maximum number of records to process. Default is no limit.")
    print("  MAX_SEQ_LEN_SPACY: Maximum text length for sentence splitting. Default is 100,000 characters.")
    print(
        "  MAX_SEQ_LEN_SENTENCE_TRANSFORMER: Maximum text length for any individual embedding. Default is 8,192 characters."
    )
    print("  NUM_GPU_NODES: Number of GPU nodes in-use. Default is 1.")
    print("  NLP_MODEL_NAME: Name of spaCy model to use for sentence splitting. Default is 'en_core_web_sm'.")
    print(
        "  EMBEDDING_MODEL_NAME: Name of SentenceTransformer model to use for text embedding. Default is 'Qwen/Qwen3-Embedding-0.6B'."
    )
    print("  DEVICE: Override to use this specific device for the encoding model. Default to auto-select best device.")
    print("  CHUNKING_PARALLELISM: Number of parallel chunking processes per node. Default is 8.")
    print("  BATCH_SIZE: Number of records per embedding batch. Default is 512.")
    print(
        "  SENTENCE_TRANSFORMER_BATCH_SIZE: Number of records per embedding batch for SentenceTransformer. Default is 16."
    )
    print("  ENCODING_DIM: Dimension of the embedding vector. Default is 1024 for the default EMBEDDING_MODEL_NAME.")
    print("")
    print("**NOTE**: At minimum, you MUST set INPUT and OUTPUT environment variables!")


######## CONFIGURATION ########

if "INPUT" not in os.environ or "OUTPUT" not in os.environ:
    raise ValueError(
        "INPUT and OUTPUT environment variables must be set! "
        "Expects path(s) to parquet files of Common Crawl data. "
        "Supports S3 and local paths."
    )

INPUT: str = str(os.environ["INPUT"])

OUTPUT: str = str(os.environ["OUTPUT"])

LIMIT: int | None = None
if "LIMIT" in os.environ and len(os.environ["LIMIT"]) > 0:
    LIMIT = int(os.environ["LIMIT"])

IS_PARQUET: bool = (
    os.environ["IS_PARQUET"].lower() in ["1", "y", "true", "yes"] if "IS_PARQUET" in os.environ else False
)

# Maximum text length for sentence splitting.
MAX_SEQ_LEN_SPACY: int = int(os.environ.get("MAX_SEQ_LEN_SPACY", 100_000))

# Maximum text length for any individual embedding.
MAX_SEQ_LEN_SENTENCE_TRANSFORMER: int = int(os.environ.get("MAX_SEQ_LEN_SENTENCE_TRANSFORMER", 1024 * 8))

# GPU nodes in your cluster
NUM_GPU_NODES: int = int(os.environ.get("NUM_GPU_NODES", 1))

# spaCy model for sentence detection
NLP_MODEL_NAME: str = str(os.environ.get("NLP_MODEL_NAME", "en_core_web_sm"))

# Parallel chunking processes per node
CHUNKING_PARALLELISM: int = int(os.environ.get("CHUNKING_PARALLELISM", 8))

# Text embedding model
EMBEDDING_MODEL_NAME: str = str(os.environ.get("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"))

DEVICE: str | None = (
    os.environ["DEVICE"].strip() if "DEVICE" in os.environ and len(os.environ["DEVICE"].strip()) > 0 else None
)

# Records per embedding batch
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", 512))

# GPU batch size for embeddings
SENTENCE_TRANSFORMER_BATCH_SIZE: int = int(os.environ.get("SENTENCE_TRANSFORMER_BATCH_SIZE", 16))

# Embedding dimensions
# NOTE: Make sure this matches the embedding model you're using!
#       ==> SentenceTransformer(...).get_sentence_embedding_dimension() <==
ENCODING_DIM: int = int(os.environ.get("ENCODING_DIM", 1024))


print("Using the following configuration (**override by setting environment variables**):")
print("-" * 80)
print(f"INPUT:                            {INPUT}")
print(f"OUTPUT:                           {OUTPUT}")
print(f"LIMIT:                            {LIMIT}")
print(f"IS_PARQUET:                       {IS_PARQUET}")
print(f"MAX_SEQ_LEN_SPACY:                {MAX_SEQ_LEN_SPACY}")
print(f"MAX_SEQ_LEN_SENTENCE_TRANSFORMER: {MAX_SEQ_LEN_SENTENCE_TRANSFORMER}")
print(f"NUM_GPU_NODES:                    {NUM_GPU_NODES}")
print(f"NLP_MODEL_NAME:                   {NLP_MODEL_NAME}")
print(f"CHUNKING_PARALLELISM:             {CHUNKING_PARALLELISM}")
print(f"EMBEDDING_MODEL_NAME:             {EMBEDDING_MODEL_NAME}")
print(f"DEVICE:                           {DEVICE}")
print(f"BATCH_SIZE:                       {BATCH_SIZE}")
print(f"SENTENCE_TRANSFORMER_BATCH_SIZE:  {SENTENCE_TRANSFORMER_BATCH_SIZE}")
print(f"ENCODING_DIM:                     {ENCODING_DIM}")
print("-" * 80)

# validate configuration
if len(INPUT) == 0:
    raise ValueError("INPUT_LOC must be set!")

if len(OUTPUT) == 0:
    raise ValueError("OUTPUT_LOC must be set!")

if MAX_SEQ_LEN_SPACY <= 0:
    raise ValueError("MAX_SEQ_LEN_SPACY must be positive!")

if MAX_SEQ_LEN_SENTENCE_TRANSFORMER <= 0:
    raise ValueError("MAX_SEQ_LEN_SENTENCE_TRANSFORMER must be positive!")

if NUM_GPU_NODES <= 0:
    raise ValueError("NUM_GPU_NODES must be positive!")

if CHUNKING_PARALLELISM <= 0:
    raise ValueError("CHUNKING_PARALLELISM must be positive!")

if BATCH_SIZE <= 0:
    raise ValueError("BATCH_SIZE must be positive!")

if SENTENCE_TRANSFORMER_BATCH_SIZE <= 0:
    raise ValueError("SENTENCE_TRANSFORMER_BATCH_SIZE must be positive!")

if ENCODING_DIM <= 0:
    raise ValueError(
        "ENCODING_DIM must be positive! If you don't know your model's embedding dimenion, run the following Python code:"
        ">>> from sentence_transformers import SentenceTransformer"
        ">>> model = SentenceTransformer(EMBEDDING_MODEL_NAME)"
        ">>> print(model.get_sentence_embedding_dimension())"
    )

if LIMIT is not None and LIMIT <= 0:
    raise ValueError(f"LIMIT must be positive if set, got {LIMIT}")

######## UTILS ########


def simple_daft_struct_type_for(t: type) -> DataType:
    def inner(x: type) -> DataType:
        if issubclass(x, str):
            return DataType.string()
        if issubclass(x, int):
            return DataType.int64()
        if issubclass(x, float):
            return DataType.float64()
        if issubclass(x, bool):
            return DataType.bool()
        raise TypeError(f"Unsupported type: {x}")

    return DataType.struct({k: inner(v) for k, v in t.__annotations__.items()})


@contextmanager
def timer(name: str | None = None):
    start = datetime.now()
    yield
    end = datetime.now()
    if name is not None and len(name) > 0:
        msg = f"[{name}] "
    else:
        msg = ""
    print(f"{msg}time taken: {end - start}")


######## TEXT CHUNKING ########


class TextChunk(TypedDict):
    text: str
    chunk_id: int


@daft.udf(
    return_dtype=DataType.list(simple_daft_struct_type_for(TextChunk)),
    concurrency=NUM_GPU_NODES * (CHUNKING_PARALLELISM + 1),
    batch_size=BATCH_SIZE // CHUNKING_PARALLELISM // 2,
)
class ChunkingUDF:
    def __init__(self) -> None:
        # ensure model is already present via:
        # f"python -m spacy download {NLP_MODEL_NAME}"
        self.nlp = spacy.load(NLP_MODEL_NAME)

    def __call__(self, text_col: Series) -> list[TextChunk]:
        n_truncated_spacy = 0
        n_truncated_sentence_transformer = 0
        try:
            results = []
            for text in text_col:
                if len(text) > MAX_SEQ_LEN_SPACY:
                    n_truncated_spacy += 1
                    text = text[:MAX_SEQ_LEN_SPACY]

                doc = self.nlp(text)
                sentence_texts = []
                for i, sentence in enumerate(doc.sents):
                    if len(sentence.text) > MAX_SEQ_LEN_SENTENCE_TRANSFORMER:
                        s_text = sentence.text[:MAX_SEQ_LEN_SENTENCE_TRANSFORMER]
                        n_truncated_sentence_transformer += 1
                    else:
                        s_text = sentence.text
                    sentence_texts.append({"text": s_text, "chunk_id": i})
                results.append(sentence_texts)
            if n_truncated_spacy > 0:
                print(
                    f"Truncated {n_truncated_spacy} / {len(text_col)} sentences that were longer than {MAX_SEQ_LEN_SPACY} characters."
                )
            if n_truncated_sentence_transformer > 0:
                print(
                    f"Truncated {n_truncated_sentence_transformer} / {len(text_col)} sentences that were longer than {MAX_SEQ_LEN_SENTENCE_TRANSFORMER} characters."
                )
            return results
        except Exception as e:
            print(f"Exception in ChunkingUDF: {e}")
            raise e


######## TEXT EMBEDDING ########


@daft.udf(
    return_dtype=DataType.embedding(DataType.float32(), ENCODING_DIM),
    concurrency=NUM_GPU_NODES,
    num_gpus=1,
    batch_size=BATCH_SIZE,
)
class EncodingUDF:
    def __init__(self, device: str | None = None) -> None:
        if device is None:
            if torch.cuda.is_available():
                self.device: str = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        if ENCODING_DIM != self.model.get_sentence_embedding_dimension():
            raise ValueError(
                f"ENCODING_DIM ({ENCODING_DIM}) does not match the embedding dimension of the model ({self.model.get_sentence_embedding_dimension()})!\n"
                "If you don't know your model's embedding dimenion, run the following Python code:"
                ">>> from sentence_transformers import SentenceTransformer"
                ">>> model = SentenceTransformer(EMBEDDING_MODEL_NAME)"
                ">>> print(model.get_sentence_embedding_dimension())"
            )
        self.model = self.model.eval()
        self.model.compile()

    def __call__(self, text_col: Series) -> np.ndarray:
        try:
            with torch.inference_mode():
                embeddings_t: torch.Tensor = self.model.encode(
                    text_col.to_pylist(),
                    batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE,
                    convert_to_tensor=True,
                    torch_dtype=torch.bfloat16,
                )
                embeddings_np = embeddings_t.cpu().numpy()
            return embeddings_np
        except Exception as e:
            print(f"Exception in EncodingUDF: {e}")
            raise e


######## DATA FORMATTING ########


WarcHeaders = TypedDict(
    "WarcHeaders",
    {
        "Content-Type": str,
        "WARC-Block-Digest": str,
        "WARC-Identified-Content-Language": str,
        "WARC-Refers-To": str,
        "WARC-Target-URI": str,
    },
)


@daft.func(return_dtype=simple_daft_struct_type_for(WarcHeaders))
def json_load_warc_headers(x: str) -> WarcHeaders:
    return json.loads(x)


######## PIPELINE ########


def pipeline() -> DataFrame:
    read_func = daft.read_parquet if IS_PARQUET else daft.read_warc
    df: DataFrame = read_func(INPUT)
    # NOTE: Expected schema is:
    #   'WARC-Record-ID': str uuid,
    #   'WARC-Type': str ,
    #   'WARC-Date': Timestamp,
    #   'Content-Length': int,
    #   'WARC-Identified-Payload-Type': str | None,
    #   'warc_content': UTF-8 bytes
    #   'warc_headers': "Content-Type": str,
    #                   "WARC-Block-Digest": str,
    #                   "WARC-Refers-To": str,
    #                   "WARC-Target-URI": str,

    if LIMIT is not None and LIMIT > 0:
        df = df.limit(LIMIT)

    print(f"Ensuring that {NLP_MODEL_NAME} exists for sentence splitting.")
    dl_name = NLP_MODEL_NAME.lower().replace("-", "_")
    spacy.cli.download(dl_name)

    return (
        df.into_batches(batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE * 10)
        .with_column("text", col("warc_content").try_decode("utf-8"))
        .filter(col("text").not_null())
        .with_column("warc_headers", json_load_warc_headers(col("warc_headers")))
        .with_column("language", col("warc_headers").struct.get("WARC-Identified-Content-Language"))
        .with_column("url", col("warc_headers").struct.get("WARC-Target-URI"))
        .with_column("sentences", ChunkingUDF(col("text")))
        .explode("sentences")
        .with_column("text", col("sentences").struct.get("text"))
        .with_column("chunk_id", col("sentences").struct.get("chunk_id"))
        .exclude("sentences")
        .with_column("embedding", EncodingUDF.with_init_args(device=DEVICE)(col("text")))
        .with_column(
            "id",
            col("url").str.right(50) + "-" + col("chunk_id").cast(DataType.string()),
        )
        .select("id", "language", "url", "text", "embedding")
    )


if __name__ == "__main__":
    df = pipeline()
    print(f"Pipeline's output schema:\n{df.schema()}")
    with timer("Pipeline complete!"):
        o = df.write_parquet(OUTPUT)
    output_partitions = [r["path"] for r in o.iter_rows()]
    _o = "\n\t".join(output_partitions)
    print(f"\nPipeline output {len(output_partitions)} partition(s):\n\t{_o}")
