# Deploying

One container serves both the REST API (`/inspect`, `/docs`) and the Gradio demo (`/demo`).
The fitted banks are committed under `models/` (35 MB for ten categories) so a plain
`docker build` from a git checkout is enough.

## Why not Vercel (or any serverless Python host)

Vercel tried to build this and asked for a FastAPI entrypoint. Adding it would not have helped:

| | Size |
|---|---|
| Open3D 0.19 Linux wheel | 448 MB |
| SciPy + scikit-learn + NumPy + FAISS, installed | ~210 MB |
| Vercel Python function limit | 250 MB uncompressed |

Serverless Python hosts (Vercel, Netlify, AWS Lambda without containers) cannot hold Open3D.
The same applies to the 512 MB RAM tiers of Render and Railway: importing Open3D plus one
40k-entry FAISS bank sits around 600-800 MB resident. This needs a container with 1-2 GB RAM.

## Google Cloud Run (recommended, free tier)

Free tier: 2 million requests and 360,000 GB-seconds per month, which covers a portfolio demo
with instances scaled to zero. Cold start is 10-20 s (Open3D import); warm requests ~0.5 s.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy pcinspect --source . --region asia-northeast1 --allow-unauthenticated \
  --memory 2Gi --cpu 2 --min-instances 0 --max-instances 2 --timeout 120
```

`--source .` builds the Dockerfile remotely, so Docker Desktop is not even required for this
path. The command prints the service URL; `/demo` is the UI, `/docs` the API explorer. Put the
URL in the README.

Redeploy after changes: re-run the same command.

## Local Docker

```bash
docker build -t pcinspect .
docker run --rm -p 8000:8000 pcinspect
# then http://localhost:8000/demo  and  http://localhost:8000/docs
curl -F "file=@demo/examples/bagel__hole_000.tiff" "localhost:8000/inspect?category=bagel&heatmap=false"
```

Expected image size: about 1.3 GB (Open3D dominates). `.dockerignore` keeps the dataset out
of the build context.

## Alternatives

- **Fly.io**: `fly launch` reads the Dockerfile; pick a 1 GB shared-cpu machine, scale to zero.
  Free allowance was removed for new accounts in 2024, so expect a few dollars a month.
- **Hugging Face Spaces**: Gradio and Docker Spaces need a PRO subscription since 2025.
  `scripts/build_space.py` still assembles a ready-to-upload Space if that changes.
- **Own VPS**: any 2 GB box; `docker run -d --restart unless-stopped -p 80:8000 pcinspect`.
