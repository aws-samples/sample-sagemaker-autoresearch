from sm_research import slicing


def test_hash_is_stable_and_order_independent():
    a = {"tables": ["t1", "t2"], "columns": ["b", "a"], "filter": "x>1", "seed": 42}
    b = {"seed": 42, "filter": "x>1", "columns": ["a", "b"], "tables": ["t2", "t1"]}
    assert slicing.slice_hash(a) == slicing.slice_hash(b)
    assert len(slicing.slice_hash(a)) == 16


def test_hash_changes_with_content():
    a = {"tables": ["t1"], "columns": ["a"], "filter": None, "seed": 1}
    b = {"tables": ["t1"], "columns": ["a"], "filter": None, "seed": 2}
    assert slicing.slice_hash(a) != slicing.slice_hash(b)


def test_s3_prefix_format():
    spec = {"tables": ["t1"], "columns": ["a"], "filter": None, "seed": 1}
    h = slicing.slice_hash(spec)
    assert slicing.s3_prefix("bk", "run9", spec) == f"s3://bk/sm-research/run9/slices/{h}/"
