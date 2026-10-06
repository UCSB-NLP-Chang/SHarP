import unittest
from sharp.screen import one_sided_p, saliency, pruning_ladder


class SaliencyTests(unittest.TestCase):
    def test_constant_difference_convention(self):
        self.assertEqual(one_sided_p([0, 0], "less"), 1)
        self.assertEqual(one_sided_p([-1, -1], "less"), 0)
        self.assertEqual(one_sided_p([-1, -1], "greater"), 1)
        self.assertEqual(one_sided_p([1, 1], "greater"), 0)

    def test_median_field_and_test_directions(self):
        data = [{"task_id": str(t), "component": c, "score": score, "tokens": cost}
                for t in range(5) for c, score, cost in [("a", 0, 100), ("b", 1, 100), ("c", 1, 1000)]]
        rows = saliency(data, ["a", "b", "c"])["rows"]
        self.assertEqual(rows[0]["p_pass"], 0)
        self.assertEqual(rows[1]["delta_tokens_mean"], 0)
        self.assertEqual(rows[2]["p_eff"], 0)
        efficiency = pruning_ladder(rows, "efficiency", step=1)
        self.assertEqual(efficiency["protected"], ["a"])
        self.assertEqual(efficiency["retention_order"], ["a", "c", "b"])
        self.assertEqual(pruning_ladder(rows, "performance")["protected"], ["c"])

    def test_missing_duplicate_and_nonfinite_rejected(self):
        rows = [{"task_id": str(t), "component": c, "score": 1, "tokens": 100}
                for t in range(2) for c in ["a", "b"]]
        for invalid in [rows[:-1], rows + rows[:1], [dict(rows[0], tokens=float("nan"))] + rows[1:]]:
            with self.assertRaises(ValueError):
                saliency(invalid, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
