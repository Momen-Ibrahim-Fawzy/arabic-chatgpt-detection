import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from arabic_detect.data import label_counts, normalise_label, read_table, write_table
from arabic_detect.logs import best_epoch, epoch_table, read_log_history
from arabic_detect.metrics import classification_report, confusion_matrix, roc_auc
from arabic_detect.models import MODELS, resolve
from arabic_detect.translate import CachedTranslator, translate_frame


class MetricTests(unittest.TestCase):
    def test_confusion_matrix(self):
        cm = confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
        np.testing.assert_array_equal(cm, [[1, 1], [1, 2]])

    def test_report_values(self):
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 1, 1, 1, 0]
        r = classification_report(y_true, y_pred)
        self.assertAlmostEqual(r["accuracy"], 0.6)
        # class 0: P=0.5 R=0.5 ; class 1: P=2/3 R=2/3
        self.assertAlmostEqual(r["precision"], (0.5 + 2 / 3) / 2)
        self.assertAlmostEqual(r["f1_macro"], (0.5 + 2 / 3) / 2)

    def test_single_class_set_reports_recall_as_accuracy(self):
        r = classification_report([1, 1, 1, 1], [1, 0, 1, 1])
        self.assertAlmostEqual(r["accuracy"], 0.75)
        self.assertAlmostEqual(r["recall"], 0.75)

    def test_roc_auc(self):
        self.assertAlmostEqual(roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertAlmostEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertTrue(np.isnan(roc_auc([1, 1], [0.2, 0.3])))


class DataTests(unittest.TestCase):
    def test_label_spellings(self):
        self.assertEqual(normalise_label("HUMAN"), 0)
        self.assertEqual(normalise_label("chatgpt"), 1)
        self.assertEqual(normalise_label(1), 1)
        with self.assertRaises(ValueError):
            normalise_label("bot")

    def test_round_trip(self):
        df = pd.DataFrame({"arabic_answer": ["نص بشري", "نص مولد", None], "label": ["HUMAN", "CHATGPT", 0]})
        with tempfile.TemporaryDirectory() as tmp:
            for ext in ("csv", "jsonl", "xlsx"):
                path = Path(tmp) / f"split.{ext}"
                write_table(df, path)
                back = read_table(path)
                self.assertEqual(back["label"].tolist(), [0, 1, 0])
                self.assertEqual(back["arabic_answer"].iloc[2], "")
        self.assertEqual(label_counts(back), {"human": 2, "chatgpt": 1})

    def test_model_names(self):
        self.assertEqual(resolve("GigaBERT-v4"), MODELS["gigabert-v4"])
        self.assertEqual(resolve("some/other-model"), "some/other-model")


class TranslateTests(unittest.TestCase):
    def test_cache_retry_and_dedup(self):
        calls = []
        failures = {"left": 2}

        def flaky(text):
            calls.append(text)
            if failures["left"]:
                failures["left"] -= 1
                raise ConnectionError("rate limited")
            return f"AR({text})"

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            tr = CachedTranslator(flaky, cache_path=cache, sleep=lambda s: None)
            df = pd.DataFrame({"question": ["q1", "q1", "q2"], "answer": ["a", "b", "c"]})
            out = translate_frame(df, tr, progress=False)
            self.assertEqual(out["arabic_question"].tolist(), ["AR(q1)", "AR(q1)", "AR(q2)"])
            self.assertEqual(calls.count("q1"), 3)  # two failures + one success, then cached

            n_calls = len(calls)
            resumed = CachedTranslator(flaky, cache_path=cache)
            self.assertEqual(resumed("q2"), "AR(q2)")
            self.assertEqual(len(calls), n_calls)
            self.assertEqual(resumed(""), "")

    def test_gives_up_after_max_retries(self):
        def broken(text):
            raise ConnectionError

        tr = CachedTranslator(broken, max_retries=3, sleep=lambda s: None)
        with self.assertRaises(ConnectionError):
            tr("hello")


class LogTests(unittest.TestCase):
    def test_epoch_table_from_literal_dump(self):
        history = [
            {"loss": 0.39, "learning_rate": 3e-5, "epoch": 1.0},
            {"eval_loss": 0.16, "eval_f1_macro": 0.93, "eval_accuracy": 0.94, "epoch": 1.0},
            {"loss": 0.09, "learning_rate": 2.4e-5, "epoch": 2.0},
            {"eval_loss": 0.14, "eval_f1_macro": 0.975, "eval_accuracy": 0.978, "epoch": 2.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.txt"
            path.write_text(repr(history))
            table = epoch_table(read_log_history(path))
            state = Path(tmp) / "trainer_state.json"
            state.write_text(json.dumps({"log_history": history}))
            self.assertEqual(len(read_log_history(state)), 4)
        self.assertEqual(list(table["epoch"]), [1.0, 2.0])
        self.assertAlmostEqual(best_epoch(table)["eval_accuracy"], 0.978)


if __name__ == "__main__":
    unittest.main()
