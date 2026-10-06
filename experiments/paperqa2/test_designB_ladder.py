import unittest
from components import component_names
from validation_ladder import (BOTTOM_UP_ORDER, PRIMARY_CONFIGS, balanced_config_order,
    disabled_components, index_label, kept_components)

class DesignBLadderTests(unittest.TestCase):
    def test_order_is_permutation(self):
        self.assertEqual(sorted(BOTTOM_UP_ORDER), sorted(component_names()))
    def test_primary_rungs(self):
        self.assertEqual(PRIMARY_CONFIGS, ("A3", "A6", "A9", "A12", "A15", "A18"))
    def test_a3_is_pass_gated_set(self):
        self.assertEqual(set(kept_components("A3")), {"gen_answer_tool", "paper_search_tool", "evidence_summarization"})
    def test_rungs_cumulative_and_full(self):
        for k in (3, 6, 9, 12, 15, 18):
            self.assertEqual(kept_components(f"A{k}"), BOTTOM_UP_ORDER[:k])
        self.assertEqual(disabled_components("A18"), ())
    def test_index_substrates(self):
        self.assertEqual(index_label("A3"), "low-parsing")
        self.assertEqual(index_label("A6"), "low-parsing")
        self.assertEqual(index_label("A9"), "chunk5000-nomm-doc")
        self.assertEqual(index_label("A12"), "chunk7000-nomm-doc")
        self.assertEqual(index_label("A15"), "chunk7000-nomm-doc")
        self.assertEqual(index_label("A18"), "full-parsing")
    def test_cyclic_latin(self):
        for pos in range(38):
            self.assertEqual(sorted(balanced_config_order(pos, PRIMARY_CONFIGS)), sorted(PRIMARY_CONFIGS))

if __name__ == "__main__":
    unittest.main()
