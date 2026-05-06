import unittest

import run_all_pipeline


class DomainConflictTests(unittest.TestCase):
    def test_domains_conflict_when_same_hostname(self):
        self.assertTrue(
            run_all_pipeline._domains_conflict(
                "https://shared.ngrok.app",
                "shared.ngrok.app",
            )
        )

    def test_domains_do_not_conflict_when_one_is_missing(self):
        self.assertFalse(run_all_pipeline._domains_conflict("shared.ngrok.app", None))

    def test_domains_do_not_conflict_when_different(self):
        self.assertFalse(
            run_all_pipeline._domains_conflict(
                "llm.ngrok.app",
                "cli.ngrok.app",
            )
        )


if __name__ == "__main__":
    unittest.main()
