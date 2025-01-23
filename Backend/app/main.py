# from fastapi import FastAPI
#
# app = FastAPI()
#
#
# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
#
# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}


from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chatbot

app = FastAPI(
    title="Project 3 - Healthcare ChatBot",
    description="This is a Healthcare ChatBot API",
    version="1.0.0",
    docs_url=None  # Disable default Swagger docs endpoint
)
# app.add_middleware(BaseHTTPMiddleware, dispatch=log_middleware)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes
    allow_headers=["*"],  # Autorise tous les headers
)

app.include_router(chatbot.router)


# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Healthcare ChatBot API"}

# ReDoc will be displayed by default
@app.get("/", include_in_schema=False)
async def redoc():
    return get_redoc_html(openapi_url="/openapi.json", title="ReDoc - Auth Service")


# Swagger docs at /docs
@app.get("/docs", include_in_schema=False)
async def swagger_docs():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Swagger - Auth Service")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8999)
