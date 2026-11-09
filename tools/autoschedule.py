"""Halide 자동 스케줄러로 제출 파일의 알고리즘에 스케줄을 붙여 보고, 걸린 시간을 잰다 (분석용 — 채점과 무관).

    uv run tools/autoschedule.py 02_nhwc                     # Mullapudi2016 · Adams2019 · Li2018 모두
    uv run tools/autoschedule.py 02_nhwc Adams2019           # 하나만

자동 스케줄러가 만든 스케줄 (C++ 문법) 을 출력한다. 제출 파일에서는 apply_autoscheduler 를 쓸 수 없으니
(채점 규칙) 마음에 드는 스케줄을 파이썬 문법으로 옮겨 _schedule 에 적고, 거기서부터 손보자.
  C++  conv.compute_at(y, co).vectorize(c, 8);     ->  파이썬  conv.compute_at(y, co).vectorize(c, 8)
  자동 스케줄러가 새로 만든 Var (예: ci, xi) 는 파이썬에서 hl.Var("ci") 로 만들어 쓴다.
채점과 같게 스레드 1 개 (parallelism 1) 로 스케줄을 고르고 잰다. 작은 크기 검사까지 통과하는지는 verify.py 로 확인한다.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ["HL_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import halide as hl  # noqa: E402
import verify  # noqa: E402

PLUGINS = {"Mullapudi2016": "autoschedule_mullapudi2016", "Adams2019": "autoschedule_adams2019",
           "Li2018": "autoschedule_li2018"}


def load(key):
    spec = importlib.util.spec_from_file_location(f"submission{key}", ROOT / f"submission{key}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plugin(name):
    """pip 의 halide 패키지 안에 자동 스케줄러 플러그인 (.so / .dylib / .dll) 이 들어 있다 — 전체 경로로 불러온다."""
    package = os.path.dirname(hl.__file__)
    stem = PLUGINS[name]
    for root, _, files in os.walk(package):
        for f in files:
            if f.startswith(f"lib{stem}.") or f.startswith(f"{stem}."):
                hl.load_plugin(os.path.join(root, f))
                return
    raise FileNotFoundError(f"{stem} 플러그인을 찾지 못했습니다")


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    key = argv[0]
    names = argv[1:] or list(PLUGINS)
    num, layout = key.split("_")
    if verify.PROBLEMS[num]["part"] != "A":
        print("자동 스케줄러 도구는 Part A (01 – 04) 만 지원합니다. Part B 는 Part A 에서 고른 합성곱 스케줄을 층마다 거세요.")
        return 1
    size = verify.PROBLEMS[num]["sizes"][0]
    args = verify.make_args(num, layout, size, seed=0)
    ref = verify.reference_a(num, layout, args)
    flops = verify.flops_a(num, size)
    target = hl.get_jit_target_from_environment()
    for name in names:
        plugin(name)
        module = load(key)
        s = module._algorithm()                   # 스케줄은 붙이지 않는다 — 자동 스케줄러가 정한다
        for param, arr in zip(s["inputs"], args):
            param.set_estimates([(0, d) for d in reversed(arr.shape)])
        s["out"].set_estimates([(0, d) for d in reversed(ref.shape)])
        pipeline = hl.Pipeline(s["out"])
        started = time.perf_counter()
        result = pipeline.apply_autoscheduler(target, hl.AutoschedulerParams(name, {"parallelism": "1"}))
        took = time.perf_counter() - started
        print(f"\n===== {name} — 스케줄 고르는 데 {took:.1f} 초 =====")
        print(result.schedule_source.strip())
        run = pipeline.compile_to_callable(s["inputs"], target)
        y = np.empty(ref.shape, np.float32)
        buffers = [hl.Buffer(a) for a in args] + [hl.Buffer(y)]
        run(*buffers)
        times = []
        for _ in range(5):
            t = time.perf_counter()
            run(*buffers)
            times.append(time.perf_counter() - t)
        ms = sorted(times)[2] * 1e3
        print(f"--> {name}: {ms:.2f} ms · {flops / ms / 1e6:.1f} GFLOP/s · 상대오차 {verify.relerr(y, ref):.1e}"
              f" (이 컴퓨터, 실제 크기, 5 번 중 가운데 값)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
