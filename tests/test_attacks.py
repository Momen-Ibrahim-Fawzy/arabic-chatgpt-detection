import unittest

from arabic_detect.attacks import (ATTACKS, BASE_ATTACKS, HOMOGLYPHS, KEYBOARD_NEIGHBOURS, apply_attack,
                                   homoglyph_substitution, hybrid, keyboard_errors, phonetic_substitution,
                                   random_char_replacement, word_segmentation_errors)

TEXT = "ذهب الطالب إلى المكتبة في الصباح وقرأ كتابا عن تاريخ الحضارة العربية ثم كتب ملخصا قصيرا"


class AttackTests(unittest.TestCase):
    def test_rate_zero_is_identity(self):
        for name, fn in ATTACKS.items():
            with self.subTest(attack=name):
                self.assertEqual(fn(TEXT, rate=0.0, seed=1), TEXT)

    def test_full_rate_changes_text(self):
        for name, fn in ATTACKS.items():
            with self.subTest(attack=name):
                self.assertNotEqual(fn(TEXT, rate=1.0, seed=1), TEXT)

    def test_seeded_attacks_are_reproducible(self):
        for name, fn in ATTACKS.items():
            with self.subTest(attack=name):
                self.assertEqual(fn(TEXT, rate=0.3, seed=7), fn(TEXT, rate=0.3, seed=7))

    def test_character_attacks_keep_length(self):
        for fn in (random_char_replacement, homoglyph_substitution, phonetic_substitution, keyboard_errors):
            with self.subTest(attack=fn.__name__):
                self.assertEqual(len(fn(TEXT, rate=0.5, seed=3)), len(TEXT))

    def test_homoglyphs_leave_arabic_block_letters_only_where_untouched(self):
        attacked = homoglyph_substitution(TEXT, rate=1.0, seed=0)
        for src, dst in HOMOGLYPHS.items():
            if src in TEXT:
                self.assertNotIn(src, attacked)
                self.assertIn(dst, attacked)

    def test_keyboard_neighbours_are_adjacent_keys(self):
        self.assertEqual(sorted(KEYBOARD_NEIGHBOURS["ض"]), ["ص"])
        self.assertEqual(sorted(KEYBOARD_NEIGHBOURS["س"]), sorted(["ش", "ي"]))
        attacked = keyboard_errors("س", rate=1.0, seed=0)
        self.assertIn(attacked, {"ش", "ي"})

    def test_word_segmentation_keeps_letters(self):
        attacked = word_segmentation_errors(TEXT, rate=1.0, seed=2)
        self.assertNotEqual(attacked.split(), TEXT.split())
        self.assertEqual(attacked.replace(" ", ""), TEXT.replace(" ", ""))

    def test_non_arabic_text_is_untouched(self):
        text = "Hello world 2023"
        for name, fn in ATTACKS.items():
            with self.subTest(attack=name):
                self.assertEqual(fn(text, rate=1.0, seed=0), text)

    def test_hybrid_differs_from_each_single_attack(self):
        mixed = hybrid(TEXT, rate=0.5, seed=4)
        for name, fn in BASE_ATTACKS.items():
            with self.subTest(attack=name):
                self.assertNotEqual(mixed, fn(TEXT, rate=0.5, seed=4))

    def test_apply_attack_per_row_seeds(self):
        texts = [TEXT, TEXT]
        out = apply_attack("random_char", texts, rate=0.5, seed=10)
        self.assertEqual(out[1], random_char_replacement(TEXT, rate=0.5, seed=11))
        with self.assertRaises(KeyError):
            apply_attack("unknown", texts)


if __name__ == "__main__":
    unittest.main()
