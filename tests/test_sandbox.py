"""Sandbox sessions: notebook state, feedback to the model, guards, limits and recovery."""

import os

import pytest

from datamate.agent.sandbox import ExecContext, Sandbox, SandboxConfig


@pytest.fixture
def ctx(mini_split, mini_ref):
    return ExecContext(
        mini_ref.pickle, mini_ref.data_dir, str(mini_split / "calendar.csv"), "2025-08-31", "doanh thu tháng 7/2025?"
    )


def test_session_keeps_state_and_echoes_the_last_expression(ctx):
    with Sandbox(30).open(ctx) as s:
        r = s.run("p = dm.month(7, 2025)\nrev = S.revenue(p)\nrev")
        assert (r.stdout.strip(), r.error, r.answer) == ("135000", None, None)
        r = s.run("print(AS_OF, QUESTION)\nS.rank('product', 'qty', p, k=1)['product'].tolist()")
        assert "2025-08-31 doanh thu tháng 7/2025?" in r.stdout and "['Sting dâu 330ml']" in r.stdout
        r = s.run("answer('number', rev)\nprint('never printed')")  # `rev` survived two steps
        assert r.answer == {"type": "number", "value": 135000, "reason": None} and "never" not in r.stdout
        r = s.run("x = 1")  # an assignment shows nothing, and the old answer is not reported again
        assert (r.stdout, r.answer) == ("", None)
        assert s.runs == 4


def test_errors_come_back_with_a_useful_hint(ctx):
    with Sandbox(30).open(ctx) as s:
        r = s.run("T['chi_tiet']['Thanh tien'].sum()")
        assert "KeyError" in r.error and "T['chi_tiet']['Thành tiền']" in r.error and "dòng 1" in r.error
        assert "các bảng: ['chi_tiet', 'don_hang', 'khach_hang']" in s.run("T['chitiet']").error
        s.run("tong = 5")
        r = s.run("print(tonng)")
        assert "NameError" in r.error and "biến bạn đã tạo: tong" in r.error
        assert "SyntaxError" in s.run("x = (1 +").error
        r = s.run("def f():\n    return 1 / 0\nf()")
        assert "ZeroDivisionError" in r.error and "dòng 2: return 1 / 0" in r.error
        r = s.run("try:\n    answer('number', 7)\nexcept Exception:\n    pass")  # answer() cannot be swallowed
        assert r.answer["value"] == 7


def test_one_shot_run_matches_the_old_interface(ctx):
    r = Sandbox(30).run("print(sorted(T)); answer('boolean', S is not None)", ctx)
    assert "['chi_tiet', 'don_hang', 'khach_hang']" in r.stdout and r.answer["value"] is True


def test_guard_limits_what_model_code_can_touch(ctx, mini_split, tmp_path):
    outside = tmp_path / "outside.txt"
    cfg = SandboxConfig(timeout_s=30, deny_read=(str(tmp_path / "ws"),))
    with Sandbox(config=cfg).open(ctx, allow_read=[str(mini_split / "rulebook.md")]) as s:
        # the shop's own files, the calendar and the rulebook are readable
        r = s.run(
            "print(pd.read_csv(DATA_DIR + '/khach_hang.csv').shape, len(CAL))\n"
            "print(open(DATA_DIR + '/../../rulebook.md', encoding='utf-8').read()[:10])"
        )
        assert r.error is None and "(2, 2) 1" in r.stdout and "# Rulebook" in r.stdout
        # the reference answers, the questions, the split folder and the workspace cache are not
        for code in (
            "open(DATA_DIR + '/../../answers.jsonl').read()",
            "pd.read_json(DATA_DIR + '/../../questions.jsonl', lines=True)",
            "import os\nos.listdir(DATA_DIR + '/../..')",
            "import glob\nglob.glob(DATA_DIR + '/../*')",
            f"open({str(tmp_path / 'ws' / 'biz_m01.pkl')!r}, 'rb').read()",
        ):
            r = s.run(code)
            assert r.error and "PermissionError" in r.error and "không được đọc" in r.error, code
        # writing: only temporary files in the session's own folder
        r = s.run(f"open({str(outside)!r}, 'w').write('x')")
        assert "không được ghi" in r.error and not outside.exists()
        r = s.run(f"import os\nos.remove({str(mini_split / 'rulebook.md')!r})")
        assert "PermissionError" in r.error and (mini_split / "rulebook.md").exists()
        r = s.run("pd.DataFrame({'a': [1]}).to_csv('tam.csv', index=False)\nprint(open('tam.csv').read().split())")
        assert r.error is None and "['a', '1']" in r.stdout
        # no other process, no network
        assert "không được chạy tiến trình khác" in s.run("import subprocess\nsubprocess.run(['echo', 'x'])").error
        assert "không được chạy tiến trình khác" in s.run("import os\nos.system('echo x')").error
        assert "không có mạng" in s.run("import socket\nsocket.create_connection(('example.com', 80), 1)").error
        # the session survives all of that
        assert s.run("S.revenue(dm.month(7, 2025))").stdout.strip() == "135000"


def test_guard_can_be_switched_off(ctx):
    with Sandbox(config=SandboxConfig(guard=False)).open(ctx) as s:
        assert s.run("len(open(DATA_DIR + '/../../answers.jsonl').read()) > 0").stdout.strip() == "True"


def test_timeout_and_crash_restart_the_session(ctx):
    with Sandbox(config=SandboxConfig(timeout_s=1.5)).open(ctx) as s:
        s.run("kept = 1")
        r = s.run("while True:\n    pass")
        assert r.timed_out and "TIMEOUT" in r.error and "khởi động lại" in r.error
        r = s.run("print('kept' in dir())\nS.revenue(dm.month(7, 2025))")
        assert r.restarted and "False" in r.stdout and "135000" in r.stdout
        r = s.run("import os\nos._exit(3)")
        assert "dừng đột ngột" in r.error and not r.timed_out
        assert s.run("1 + 1").stdout.strip() == "2"


@pytest.mark.skipif(os.name != "posix", reason="resource limits are POSIX only")
def test_memory_limit_becomes_an_error_not_a_dead_machine(ctx):
    with Sandbox(config=SandboxConfig(mem_mb=2048)).open(ctx) as s:
        r = s.run("x = np.ones((40000, 40000))")
        assert "MemoryError" in r.error and "quá nhiều bộ nhớ" in r.error
        assert s.run("2 * 3").stdout.strip() == "6"


def test_failed_workspace_still_gives_a_usable_session(mini_split):
    ctx = ExecContext(None, str(mini_split / "data" / "biz_m01"), None, "2025-08-31", "q")
    with Sandbox(30).open(ctx) as s:
        r = s.run("print(S, T, CAL)\npd.read_csv(DATA_DIR + '/don_hang.csv').shape")
        assert r.error is None and "None {} None" in r.stdout and "(6, 5)" in r.stdout
        assert "`S` là None" in s.run("S.revenue()").error
