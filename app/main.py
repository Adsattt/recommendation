from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import recommendation_routes

app = FastAPI(
    title="Innovation Recommendation System",
    description="Sistem rekomendasi inovasi berbasis content-based filtering",
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(recommendation_routes.router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
