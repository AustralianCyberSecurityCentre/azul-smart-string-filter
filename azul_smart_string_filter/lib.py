"""using ai model to find great strings!"""

import json
import os

import numpy as np
import onnxruntime as rt
from sklearn.feature_extraction.text import TfidfVectorizer


class SmartStringFilter:
    """using ai model to find great strings."""

    def find_legible_strings(self, strings: list[str]) -> list[bool]:
        """Used to filter strings and return list of bools."""
        """True = good string, False = bad string."""

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "model", "model.onnx")
        vectorizer_path = os.path.join(current_dir, "model", "vectorizer.json")

        string_list = strings

        with open(vectorizer_path, "r") as f:
            vectorizer_json = json.load(f)

        vectorizer = TfidfVectorizer(
            analyzer=vectorizer_json["analyzer"], ngram_range=tuple(vectorizer_json["ngram_range"])
        )

        vectorizer.vocabulary_ = vectorizer_json["vocabulary_"]
        vectorizer.idf_ = np.array(vectorizer_json["idf_"])
        # Load the ONNX model.
        print("loading onnx model")
        sess = rt.InferenceSession(model_path)
        print("onnx model loaded")

        input_data = vectorizer.transform(string_list).toarray().astype(np.float32)

        # Run the model.
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        predictions = sess.run([label_name], {input_name: input_data})[0]
        predictions = [bool(x) for x in predictions]

        return predictions
