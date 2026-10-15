import math
import unittest

from mock_bedrock.embeddings import handle_embedding

V2 = "amazon.titan-embed-text-v2:0"
V1 = "amazon.titan-embed-text-v1"
TEXT = "alpha bravo charlie delta echo"


def norm(v):
    return math.sqrt(sum(x * x for x in v))


class TitanV2Tests(unittest.TestCase):
    def test_default(self):
        r, n = handle_embedding(V2, {"inputText": TEXT})
        self.assertEqual(n, 1)
        self.assertEqual(r["embedding"], r["embeddingsByType"]["float"])
        self.assertEqual(list(r["embeddingsByType"]), ["float"])
        self.assertAlmostEqual(norm(r["embedding"]), 1.0, places=6)

    def test_arn_path(self):
        r, _ = handle_embedding("arn:aws:bedrock:us-east-1::foundation-model/" + V2,
                                {"inputText": TEXT})
        self.assertIn("embeddingsByType", r)

    def test_normalize_false(self):
        a, _ = handle_embedding(V2, {"inputText": TEXT})
        b, _ = handle_embedding(V2, {"inputText": TEXT, "normalize": False})
        nb = norm(b["embedding"])
        self.assertGreater(abs(nb - 1.0), 0.1)
        for x, y in zip(a["embedding"], b["embedding"]):
            self.assertAlmostEqual(x, y / nb, places=9)

    def test_binary_only(self):
        r, _ = handle_embedding(V2, {"inputText": TEXT, "embeddingTypes": ["binary"]})
        self.assertNotIn("embedding", r)
        b = r["embeddingsByType"]["binary"]
        self.assertEqual(list(r["embeddingsByType"]), ["binary"])
        self.assertEqual(len(b), 1024)
        self.assertTrue(all(type(x) is int and x in (0, 1) for x in b))

    def test_both_types(self):
        r, _ = handle_embedding(V2, {"inputText": TEXT, "embeddingTypes": ["float", "binary"]})
        f, b = r["embeddingsByType"]["float"], r["embeddingsByType"]["binary"]
        self.assertEqual(r["embedding"], f)
        self.assertEqual(b, [1 if x > 0 else 0 for x in f])

    def test_dimensions_with_binary(self):
        r, _ = handle_embedding(V2, {"inputText": TEXT, "dimensions": 256,
                                     "embeddingTypes": ["binary"]})
        self.assertEqual(len(r["embeddingsByType"]["binary"]), 256)

    def test_invalid(self):
        for bad in ([], ["int8"], "float", ["float", 1]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                handle_embedding(V2, {"inputText": TEXT, "embeddingTypes": bad})
        with self.assertRaises(ValueError):
            handle_embedding(V2, {"inputText": TEXT, "normalize": "yes"})

    def test_v1_unchanged(self):
        r, _ = handle_embedding(V1, {"inputText": TEXT})
        self.assertEqual(set(r), {"embedding", "inputTextTokenCount"})


if __name__ == "__main__":
    unittest.main()
