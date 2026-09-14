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

Measured memory of the serving process (uvicorn + API + Gradio mounted): 203 MB idle, 246 MB
after inspecting scans from three categories with heatmaps. So 512 MB container tiers do fit;
their limit is CPU (0.1 vCPU on free plans), not RAM.

## Google Cloud Run (recommended, free tier)

Deployed 2026-09-15: https://pcinspect-727953311125.asia-northeast1.run.app (project `pcinspect-508620`, region asia-northeast1).

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
path.

On a fresh project the first deploy fails with `PERMISSION_DENIED ... default service account is
missing required IAM permissions`: Cloud Build runs as the Compute Engine default service
account, which starts with no roles. Grant it once (replace the project number):

```bash
gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" --role="roles/cloudbuild.builds.builder"
```

then re-run the deploy. The command prints the service URL; `/demo` is the UI, `/docs` the API explorer. Put the
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

## Render free tier (no card, slower)

Render's free web service (512 MB RAM, 0.1 CPU, sleeps after 15 min idle, no credit card)
can run the same Dockerfile straight from the GitHub repo: New > Web Service > connect the repo,
runtime Docker, instance type Free. Render sets `$PORT`, which the Dockerfile already honours.
Expect a 1-3 minute cold start (image pull + Open3D import on a tenth of a core) and several
seconds per scan. Fine as a fallback demo; put the Cloud Run URL first in the README while it
is alive.

## Alternatives

- **Fly.io**: `fly launch` reads the Dockerfile; pick a 1 GB shared-cpu machine, scale to zero.
  Free allowance was removed for new accounts in 2024, so expect a few dollars a month.
- **Hugging Face Spaces**: Gradio and Docker Spaces need a PRO subscription since 2025.
  `scripts/build_space.py` still assembles a ready-to-upload Space if that changes.
- **Own VPS**: any 2 GB box; `docker run -d --restart unless-stopped -p 80:8000 pcinspect`.
