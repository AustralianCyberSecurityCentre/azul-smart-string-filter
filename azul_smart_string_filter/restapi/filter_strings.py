"""Rest api for string filter."""

import asyncio
from enum import Enum

import uvicorn
from fastapi import Body, FastAPI, Query, Request, Response
from pydantic import BaseModel
from starlette_exporter import PrometheusMiddleware, handle_metrics

from azul_smart_string_filter.lib import SmartStringFilter


class FileTypes(str, Enum):
    """File types that are accepted. Currently only handle windows pe strings."""

    windows = [
        "executable/windows/dll32",
        "executable/windows/dll64",
        "executable/windows/pe",
        "executable/windows/pe32",
        "executable/windows/pe64",
        "executable/pe32",
        "executable/dll32",
    ]


class SearchResult(BaseModel):
    """A discovered instance of a particular string in a file."""

    string: str
    offset: int


gsf = None
model_loading_task = None


async def load_model_async():
    """Load the model in the background after the server starts."""
    global gsf
    print("Background model load started...")
    try:
        gsf = SmartStringFilter()
        print("SmartStringFilter model loaded successfully")
    except Exception as e:
        print("SmartStringFilter failed to load:", e)
        import traceback

        traceback.print_exc()


app = FastAPI()


@app.on_event("startup")
async def schedule_model_load():
    """Schedule the model to load AFTER the server starts."""
    global model_loading_task
    model_loading_task = asyncio.create_task(load_model_async())


app.add_middleware(
    PrometheusMiddleware,
    app_name="SmartStringFilter",
    prefix="SmartStringFilter",
    group_paths=True,
)

app.add_route("/metrics", handle_metrics)


@app.get("/")
def read_root():
    """Allow user to see server is running."""
    return "OK"


@app.post(
    "/v0/strings",
    response_model=list[SearchResult],
    response_model_exclude_unset=True,
)
async def submit_unfiltered_strings(
    request: Request,
    resp: Response,
    file_format: str = Query(...),
    strings: list[SearchResult] = Body(...),
) -> list[SearchResult]:
    """Filter strings using the model."""
    # Ensure model is ready
    if gsf is None:
        return {"error": "Model is still loading. Try again shortly."}

    if is_supported_file_format(file_format, FileTypes.windows):
        filtered_strings = []
        strings_to_be_filtered = [obj.string for obj in strings]
        predictions = gsf.find_legible_strings(strings_to_be_filtered)

        for string, is_good in zip(strings, predictions, strict=False):
            if is_good:
                filtered_strings.append(string)
        return filtered_strings


def is_supported_file_format(file_format, file_format_enum):
    """Check if the file_format is in the specified file_format_enum list."""
    return any(file_format.startswith(t) for t in file_format_enum)


def main():
    """Start server."""
    uvicorn.run(app, host="0.0.0.0", port=8851, log_level="info")  # noqa S104


if __name__ == "__main__":
    main()
