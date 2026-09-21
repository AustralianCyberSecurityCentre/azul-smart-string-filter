"""Using AI models to find great strings."""

import json
import os

import numpy as np
import onnxruntime as rt
from sklearn.feature_extraction.text import TfidfVectorizer


class SmartStringFilter:
    """Use an AI model to find great strings."""

    def find_legible_strings(
        self,
        strings: list[str],
        model_type: str,
        batch_size: int = 100,
    ) -> list[bool]:
        """Return one complete result list after classifying in memory-safe batches."""

        model_type = model_type.strip().lower()

        if not model_type or not model_type.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid model type: {model_type!r}")

        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_directory = os.path.join(current_dir, "model")

        model_path = os.path.join(
            model_directory,
            f"model_{model_type}.onnx",
        )
        vectorizer_path = os.path.join(
            model_directory,
            f"vectorizer_{model_type}.json",
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model for type {model_type!r} was not found: {model_path}"
            )

        if not os.path.isfile(vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizer for type {model_type!r} was not found: "
                f"{vectorizer_path}"
            )

        with open(vectorizer_path, "r", encoding="utf-8") as file:
            vectorizer_json = json.load(file)

        vectorizer = TfidfVectorizer(
            analyzer=vectorizer_json["analyzer"],
            ngram_range=tuple(vectorizer_json["ngram_range"]),
            lowercase=vectorizer_json.get("lowercase", True),
            min_df=vectorizer_json.get("min_df", 1),
            dtype=np.float32,
        )

        vectorizer.vocabulary_ = vectorizer_json["vocabulary_"]
        vectorizer.idf_ = np.array(vectorizer_json["idf_"], dtype=np.float32)

        print(f"Loading {model_type} ONNX model")
        session = rt.InferenceSession(model_path)
        print(f"{model_type} ONNX model loaded")

        input_name = session.get_inputs()[0].name
        label_name = session.get_outputs()[0].name

        predictions: list[bool] = []

        for start in range(0, len(strings), batch_size):
            batch = strings[start : start + batch_size]

            # Convert the sparse matrix to float32 before making it dense. This
            # avoids creating a much larger intermediate float64 dense matrix.
            input_data = (
                vectorizer.transform(batch)
                .astype(np.float32)
                .toarray()
            )

            batch_predictions = session.run(
                [label_name],
                {input_name: input_data},
            )[0]

            predictions.extend(
                bool(prediction) for prediction in batch_predictions
            )

        # The WebUI receives a single list containing results for every input
        # string, in the same order as the original strings list.
        return predictions
