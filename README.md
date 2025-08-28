# text embedding common crawl with qwen

## Local Dev Setup
```bash
uv sync
```

## Running
```bash
INPUT="..." OUTPUT="..." uv run python -m cc_embed.main_pipeline
```
Where:
- `INPUT` is the path, or s3 key (prefix), to the common crawl Parquet files
- `OUTPUT` is the path, or s3 key (prefix), where the output chnked text + embeddings are written to


## Running on Kubernetes
1. Set all environment variables for configuration.
    a. You can use the defaults by executing:
```bash
source DEFAULT_ENV
```
    b. You must **still** manually set `OUTPUT`
2. Use a specific tag by setting `COMMIT` (e.g. `COMMIT=$(git rev-parse HEAD)`). Or leave it empty to use the `latest` tag.
3. Fill-in the k8s job template:
```bash
cat k8s-job-template.yaml | envsubst > k8s_job.yaml
```
4. Run the job on k8s:
```bash
NAMESPACE="..."
kubectl -n $NAMESPACE apply -f k8s_job.yaml
```
5. Monitor the job:
```bash
POD=$(kubectl -n $NAMESPACE get pods | grep qwen-embed-job | head -n 2 | tail -n 1 | cut -f1 -d' ')

# check status
kubectl -n $NAMESPACE describe pod $POD

# check logs
kubectl -n $NAMESPACE logs $POD
```

## Image Build & Run
```bash
docker build -f Dockerfie -t daft-qwen3-text-embedding:$(git rev-parse HEAD) .
```
Then enter into environment (_no need to use `uv` once inside_):
```bash
docker run --rm -it daft-qwen3-text-embedding:$(git rev-parse HEAD)
```

## Image Publish:
```bash
docker build --platform=linux/amd64 -f Dockerfie -t docker.io/malcolmgreaves/daft-qwen3-text-embedding:$(git rev-parse HEAD) .
docker push docker.io/malcolmgreaves/daft-qwen3-text-embedding:$(git rev-parse HEAD)
```
