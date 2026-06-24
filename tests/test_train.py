from src.train import selection_key


def test_selection_key_prefers_higher_val_pr_auc():
    rf = {"pr_auc": 0.35, "precision_at_top_20": 0.07}
    lr = {"pr_auc": 0.20, "precision_at_top_20": 0.17}
    assert selection_key(rf) > selection_key(lr)


def test_selection_key_breaks_ties_on_val_p_at_20():
    a = {"pr_auc": 0.40, "precision_at_top_20": 0.30}
    b = {"pr_auc": 0.40, "precision_at_top_20": 0.25}
    assert selection_key(a) > selection_key(b)
