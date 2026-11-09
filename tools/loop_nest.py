"""제출 파일의 Halide 루프 구조를 본다 (분석용 — 채점과 무관, import 제한 없음).

    uv run tools/loop_nest.py 02_nhwc            # submission02_nhwc.py 의 루프 구조 (print_loop_nest)
    uv run tools/loop_nest.py 05_nhwc            # Part B 는 모델 전체 (층이 많아 길다 — | less)
    uv run tools/loop_nest.py 02_nhwc --stmt     # + 낮춘 코드 (lowered stmt) 를 tools/out/02_nhwc.stmt.html 로

제출 파일에 출발점 코드처럼 _algorithm() · _schedule(s) (Part A) 또는 _build(params) · _schedule(net) (Part B) 가
있어야 한다 (이름을 바꿨다면 이 파일도 맞춰 고친다).
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import halide as hl  # noqa: E402


def load(key):
    path = ROOT / f"submission{key}.py"
    spec = importlib.util.spec_from_file_location(f"submission{key}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_stderr(fn):
    """Halide 는 C++ 쪽 stderr (fd 2) 로 출력한다 — 파이썬의 sys.stderr 를 바꿔서는 잡히지 않는다."""
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            fn()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        tmp.seek(0)
        return tmp.read().decode(errors="replace")


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    key = argv[0]
    module = load(key)
    if hasattr(module, "_build"):                     # Part B (05 · 06): 모델 전체 파이프라인
        import verify
        num, layout = key.split("_")
        net = module._build(verify.load_params(verify.PROBLEMS[num]["model"], layout))
        module._schedule(net)
        s = {"out": net.out, "inputs": [net.img]}
    else:                                             # Part A (01 – 04)
        s = module._algorithm()
        module._schedule(s)
    print(capture_stderr(s["out"].print_loop_nest))
    if "--stmt" in argv:
        out = ROOT / "tools" / "out"
        out.mkdir(exist_ok=True)
        path = out / f"{key}.stmt.html"
        s["out"].compile_to_lowered_stmt(str(path), s["inputs"], hl.StmtOutputFormat.HTML,
                                         hl.get_jit_target_from_environment())
        print(f"낮춘 코드: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
