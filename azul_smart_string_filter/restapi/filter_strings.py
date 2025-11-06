"""Rest api for string filter."""

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


app = FastAPI()

app.add_middleware(
    PrometheusMiddleware,
    app_name="SmartStringFilter",
    prefix="SmartStringFilter",
    group_paths=True,
)

# metrics is only available at the root path (not api_prefix) - for prometheus
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
    # filetype strings were extracted from.
    file_format: str = Query(...),
    # strings to be filtered.
    strings: list[SearchResult] = Body(...),
) -> list[SearchResult]:
    """Endpoint that handles the POST request.

    It expects a file_format along with a list of strings for json body.
    It returns a list of FilteredStrings.
    """
    if is_supported_file_format(file_format, FileTypes.windows):
        gsf = SmartStringFilter()
        filtered_strings = []
        strings_to_be_filtered = [obj.string for obj in strings]
        predictions = gsf.find_legible_strings(strings_to_be_filtered)

        for string, is_good in zip(strings, predictions):
            if is_good:
                filtered_strings.append(string)
        return filtered_strings


def is_supported_file_format(file_format, file_format_enum):
    """Check if the file_format is in the specified file_format_enum list."""
    return any(file_format.startswith(t) for t in file_format_enum)


def main():
    """Start server."""
    uvicorn.run(app, host="0.0.0.0", port=8851, log_level="info")  # nosec B104


if __name__ == "__main__":
    main()
