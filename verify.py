"""제11주 과제 검증기 — Halide 스케줄 (알고리즘과 스케줄의 분리), 단일 스레드, 12 파일, 실행 시간(ms).

    uv run verify.py submission01_nchw.py
    uv run verify.py submission05_nhwc.py
    uv run verify.py 03                    03_nchw, 03_nhwc 두 파일
    uv run verify.py                       12 파일 모두
    uv run verify.py --no-time             규칙 · 정확도 · 실행 검사만 (리더보드에 내지 않는다)
    uv run verify.py submission06_nhwc.py --verbose --json /tmp/result.json

파일별 표준 출력은 success 또는 fail 한 줄이다. success 이면 실행 시간(ms)과 측정값을, fail 이면 이유를
표준 오류로 더 적는다. 규칙 · 정확도 · 실행 검사를 통과하면 목표 시간보다 느려도 success 다.
Classroom50 / GitHub Actions 에서는 리더보드 결과 접수까지 끝나야 success 다. 실패는 fail, ms=null 로 제출한다.

문제 (PROBLEMS) — 6 개 x 레이아웃 2 개 (NCHW, NHWC). 7주차와 같은 문제 · 같은 크기 · 같은 검증이다.
  Part A 연산 (퓨전)  01 ResNet-18 stem  02 ResNet-18 BasicBlock  03 MobileNetV2 inverted residual
                      04 전처리 + MobileNetV2 stem.  호출마다 입력 · 가중치 · BN 이 바뀐다.
  Part B 모델 전체    05 ResNet-18  06 MobileNetV2.  CIFAR-10 이미지 50 장 (uint8) -> logits (50, 10).
                      prepare_<lay>(params) 는 한 번 부르고 재지 않는다 (Halide 파이프라인을 만들고 JIT 컴파일하는 곳).
                      infer_<lay>(state, images) 를 잰다.
기준 기계는 Raspberry Pi 5 (Cortex-A76) 코어 1 개다. 참고 시간은 TIME_MS 에 있다
(BASE_MS = 출발점 코드 (스케줄 없음), GOAL_MS = 조교 정답 x 1.25).

실행 시간은 준비 호출 3 회(측정 제외) 뒤 서로 다른 입력으로 9 회 재서, 가장 빠른 2 회와 가장 느린 2 회를
뺀 5 회의 평균이다. 리더보드에는 최솟값이 아니라 이 평균을 보낸다.

정확도: 입력은 채점마다 새로 뽑고(VERIFY_SEED 로 재현) float64 기준값은 이 파일이 직접 계산한다.
  Part A  실제 크기 + 작은 크기 2 개, 전체 출력, 상대오차 2e-4
  Part A  시간 측정 호출도 매번 전체 출력을 검사한다.
  Part B  모델 압축 (양자화 · 프루닝 · 재학습 · calibration) 을 허용하므로 logits 를 기준값과 비교하지 않고
          CIFAR-10 test 앞 TEST_TOTAL 장의 정답률로 채점한다. 이미지마다 채점마다 무작위 화소 값 8 개를 ±1 바꾸고
          (미리 계산해 둔 답을 쓰지 못하게), 시간을 재는 9 번의 묶음도 이 test 이미지이며 정답률에 들어간다.
          정답 장수가 MIN_CORRECT 보다 적으면 fail. 리더보드 점수 = 시간 점수 x 정답률 배율
          (1 - ACCURACY_PENALTY x max(0, float 모델 정답 - ACCURACY_SLACK - 정답), 0 ~ 1).
          그 밖에 (N, 10) float32 · 유한값 · 같은 이미지는 묶음과 상관없이 같은 결과 (상대오차 1e-3) · 입력 · params 불변.
          가중치는 model/<모델>.npz 가 있으면 그 파일, 없으면 weights/<모델>.npz. 지식증류는 금지 (코드 리뷰).
  weights/ 와 samples/ 의 파일은 SHA-256 으로 원본인지 확인한다.

제출 구현은 Halide 파이프라인 (Func.realize · Pipeline.realize · Callable 호출) 안에서 시간의 50% 이상을
보내야 한다. Halide 의 스레드는 1 개로 고정하고 (HL_NUM_THREADS=1) 코어 하나에 묶는다 — parallel() 은 써도
빨라지지 않는다. 제출 코드는 별도 프로세스에서 실행하고 GitHub OIDC 및 API 토큰을 전달하지 않는다.
파일 하나의 채점 프로세스 한도는 540 초다 (Classroom50 테스트 한도 600 초보다 먼저 끝나 fail 을 리더보드에 남긴다).

Classroom50 에서 실행하면 결과를 내부망 리더보드 HTTP API 에 자동 제출한다.
서버는 등록된 채점 Runner 의 실제 접속 IP 에서 오는 제출만 허용한다."""
import os

os.environ["HL_NUM_THREADS"] = "1"                       # Halide 스레드 풀 1 개 (halide import 전에)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import halide as hl  # noqa: E402

HERE = Path(__file__).resolve().parent
COMPILE_LIMIT = 300.0       # 첫 호출(JIT 컴파일 포함) 한도 (초)
PREPARE_LIMIT = 60.0        # Part B prepare 한도 (초) — 파이프라인 만들기와 JIT 컴파일 포함
WORKER_LIMIT = 540.0        # 파일 하나의 채점 프로세스 한도 (초) — Classroom50 테스트 한도(최대 600 초)보다 짧게


# ------------------------------------------------- 과제 상수

PERIOD = "week11"
FIXTURE_SCHEMA = 1
EPS = 1e-5                  # BN eps
TOL = 2e-4                  # Part A 상대오차 허용치
TOL_E2E = 1e-3              # Part B logits 상대오차 허용치 (층이 많아 float32 오차가 쌓인다)
WARMUP = 3                  # 측정 전 준비 호출 횟수 (측정에서 제외)
REPEAT = 9                  # 시간 측정 횟수 (호출마다 다른 입력)
TRIM = 2                    # 가장 빠른 측정값과 가장 느린 측정값을 각각 이만큼 버리고 나머지 평균을 쓴다
N_E2E = 50                  # Part B 묶음 크기
PROBE_IMAGES = 8            # (쓰지 않음 — 10 주차 이전과 같은 FIXTURE 구성을 위해 남김)
TEST_TOTAL = 2000           # Part B test 정답률을 재는 장수 (CIFAR-10 test 의 앞 2,000 장, 10 주차와 같다)
TEST_NOISE = 8              # 채점마다 test 이미지마다 무작위 화소 값 8 개를 ±1 바꾼다 — 미리 계산해 둔 답을 쓰지 못하게
ACCURACY_PENALTY = 0.01     # 정답률 배율 = 1 - 0.01 x max(0, float 정답 - SLACK - 정답): 한 장마다 1% 씩
ACCURACY_SLACK = 3          # 잡음 때문에 float 모델도 흔들리는 장수 — 이만큼은 덜 맞혀도 깎지 않는다
MIN_CORRECT = {"05": 1600, "06": 1700}     # 이보다 적게 맞히면 fail (ResNet-18 80% · MobileNetV2 85%, 10 주차와 같다)
CONSIST_TOL = 1e-3          # 같은 이미지를 다른 묶음에 넣어도 같은 결과여야 한다 (상대오차)
LAYOUTS = ("nchw", "nhwc")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
MBV2_CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

# 번호 -> 문제. sizes 는 Part A 의 (실제 크기, 작은 크기 1, 작은 크기 2). 7주차와 같다.
PROBLEMS = {
    "01": dict(part="A", fn="stem", name="ResNet-18 stem: conv7x7/2 + BN + ReLU + maxpool3x3/2",
               sizes=[(1, 224, 224, 64), (1, 17, 23, 8), (2, 32, 32, 16)]),                    # (N, H, W, C_out)
    "02": dict(part="A", fn="basic_block", name="ResNet-18 BasicBlock",
               sizes=[(1, 56, 56, 64), (1, 9, 11, 8), (2, 8, 8, 16)]),                         # (N, H, W, C)
    "03": dict(part="A", fn="inverted_residual", name="MobileNetV2 inverted residual (t=6)",
               sizes=[(4, 56, 56, 24, 6), (1, 7, 9, 8, 6), (2, 8, 8, 16, 4)]),                 # (N, H, W, C, t)
    "04": dict(part="A", fn="preprocess_stem", name="전처리 + MobileNetV2 stem: uint8 -> 정규화 -> conv3x3/2 + BN + ReLU6",
               sizes=[(4, 224, 224, 32), (1, 17, 19, 8), (2, 8, 8, 16)]),                      # (N, H, W, C_out)
    "05": dict(part="B", model="resnet18", name="ResNet-18 end-to-end (CIFAR-10, 50 장)", float=1726),
    "06": dict(part="B", model="mobilenetv2", name="MobileNetV2 end-to-end (CIFAR-10, 50 장)", float=1846),
}

# 참고 시간 (ms) — Raspberry Pi 5, 코어 1 개에서 위 측정 방식(5 회 평균)으로 잰 값.
#   BASE_MS  출발점 코드 (단계마다 compute_root, 벡터 · 타일 없음) 의 시간,  GOAL_MS  조교 정답 시간 x 1.25 (0.1 ms 단위 올림)
# 다시 잴 때:  uv run verify.py --answers ../answers-w11 --baseline   (starter/ 에 출발점 사본) -> 맨 끝에 제안 TIME_MS
TIME_MS = {             # 파일: (BASE_MS, GOAL_MS)   2026-11-10 보정 (sota01, 코어 1 개)
    "01_nchw": (135.6, 12.5), "01_nhwc": (134.6, 16.3),
    "02_nchw": (324.3, 25.7), "02_nhwc": (354.2, 21.2),
    "03_nchw": (134.2, 18.8), "03_nhwc": (108.5, 19.2),
    "04_nchw": (74.3, 6.5), "04_nhwc": (76.4, 8.5),
    "05_nchw": (3233.3, 311.7), "05_nhwc": (2691.5, 239.0),
    "06_nchw": (7358.4, 1077.2), "06_nhwc": (5533.2, 772.2),
}

# 제공 데이터의 SHA-256 — 가중치나 샘플을 바꾸면 기준값도 같이 바뀌므로 원본인지 먼저 확인한다
DATA_SHA256 = {
    "weights/resnet18.npz": "b5c7d203bd8068577a6eaa431abf251c967156019cd45a33502731f9a78b7b5f",
    "weights/mobilenetv2.npz": "c52c1f22335380398848d69ebcf16d4d3fc90d7a633403e2ad38a18f2f7a4671",
    "samples/cifar10_50.npz": "dbd00afb4724aa26301a6126aabee310cf62e4471ef1e26f913b52c69c5e063f",
    "samples/cifar10_test.npz": "003e0f78170cb2914fcc621d3fa65455857ae28119144c1f9310b62cad89a5af",
    "samples/cifar10_calib.npz": "f157686bcd8de9165345c7ab2e660708fb3d42c71740deb315650b2c70403644",
}

# 리더보드가 같은 채점 기준의 결과만 받도록 하는 버전 (11 주차는 내려받는 기준값 파일이 없다)
HALIDE_VERSION = "22.1.0"     # pyproject.toml · uv.lock 과 같은 판

FIXTURE_ID = hashlib.sha256(json.dumps({
    "schema": FIXTURE_SCHEMA, "period": PERIOD, "problems": PROBLEMS, "tol": [TOL, TOL_E2E],
    "timing": [WARMUP, REPEAT, TRIM], "probe_images": PROBE_IMAGES,
    "accuracy": [TEST_TOTAL, TEST_NOISE, ACCURACY_PENALTY, ACCURACY_SLACK, MIN_CORRECT, CONSIST_TOL], "data": DATA_SHA256, "halide": HALIDE_VERSION,
}, sort_keys=True).encode()).hexdigest()[:20]
LEADERBOARD_URL = "http://192.168.1.22:7777"


def in_ci():
    return os.environ.get("GITHUB_ACTIONS") == "true" or bool(
        os.environ.get("CLASSROOM") and os.environ.get("ASSIGNMENT") and os.environ.get("GITHUB_REPOSITORY"))


def run_seed():
    """채점마다 새 난수 씨앗 (입력을 미리 알 수 없게). VERIFY_SEED 를 주면 그 값으로 재현한다."""
    value = os.environ.get("VERIFY_SEED", "").strip()
    return int(value) if value else secrets.randbits(31)


# ================================================= 레이아웃 도우미

def to_layout(a, layout):
    """NCHW (또는 OIHW) 4 차원 배열을 레이아웃에 맞게. 1·2 차원은 그대로."""
    if layout == "nhwc" and a.ndim == 4:
        return np.ascontiguousarray(a.transpose(0, 2, 3, 1))
    return a


def from_layout(a, layout):
    """레이아웃의 4 차원 배열을 NCHW (OIHW) 로."""
    if layout == "nhwc" and a.ndim == 4:
        return a.transpose(0, 3, 1, 2)
    return a


# ================================================= float64 기준값 (NCHW, 채점기만 BLAS 를 쓴다)

class Ref:
    """기준값 연산. x 는 NCHW float64, 가중치는 OIHW. flops 를 센다."""
    flops = 0

    @classmethod
    def conv(cls, x, w, stride, pad):
        x = np.asarray(x, np.float64)
        w = np.asarray(w, np.float64)
        n, c, h, wd = x.shape
        co, ci, kh, kw = w.shape
        oh, ow = (h + 2 * pad - kh) // stride + 1, (wd + 2 * pad - kw) // stride + 1
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        y = np.zeros((n, co, oh, ow))
        for a in range(kh):
            for d in range(kw):
                patch = xp[:, :, a:a + oh * stride:stride, d:d + ow * stride:stride]
                y += np.tensordot(w[:, :, a, d], patch, axes=([1], [1])).transpose(1, 0, 2, 3)
        cls.flops += 2 * n * co * oh * ow * ci * kh * kw
        return y

    @classmethod
    def dwconv(cls, x, w, stride, pad):
        x = np.asarray(x, np.float64)
        w = np.asarray(w, np.float64)
        n, c, h, wd = x.shape
        _, _, kh, kw = w.shape
        oh, ow = (h + 2 * pad - kh) // stride + 1, (wd + 2 * pad - kw) // stride + 1
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        y = np.zeros((n, c, oh, ow))
        for a in range(kh):
            for d in range(kw):
                y += xp[:, :, a:a + oh * stride:stride, d:d + ow * stride:stride] * w[:, 0, a, d][None, :, None, None]
        cls.flops += 2 * n * c * oh * ow * kh * kw
        return y

    @staticmethod
    def bn(x, g, b, m, v):
        s = np.asarray(g, np.float64) / np.sqrt(np.asarray(v, np.float64) + EPS)
        return x * s[None, :, None, None] + (np.asarray(b, np.float64) - np.asarray(m, np.float64) * s)[None, :, None, None]

    @staticmethod
    def relu(x):
        return np.maximum(x, 0.0)

    @staticmethod
    def relu6(x):
        return np.clip(x, 0.0, 6.0)

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def maxpool(x, k=3, stride=2, pad=1):
        n, c, h, wd = x.shape
        oh, ow = (h + 2 * pad - k) // stride + 1, (wd + 2 * pad - k) // stride + 1
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), constant_values=-np.inf)
        y = np.full((n, c, oh, ow), -np.inf)
        for a in range(k):
            for d in range(k):
                y = np.maximum(y, xp[:, :, a:a + oh * stride:stride, d:d + ow * stride:stride])
        return y

    @staticmethod
    def gap(x):
        return x.mean(axis=(2, 3))

    @classmethod
    def linear(cls, x, w, b):
        cls.flops += 2 * x.shape[0] * w.shape[0] * w.shape[1]
        return np.asarray(x, np.float64) @ np.asarray(w, np.float64).T + np.asarray(b, np.float64)

    @staticmethod
    def preprocess(img, mean, std):
        x = np.asarray(img, np.float64) / 255.0
        return (x - np.asarray(mean, np.float64)[None, :, None, None]) / np.asarray(std, np.float64)[None, :, None, None]



# ================================================= Part A — 연산 정의 (기준값이 쓰는 식)

def stem(o, x, w, bn):
    return o.maxpool(o.relu(o.bn(o.conv(x, w, 2, 3), *bn)))


def basic_block(o, x, w1, bn1, w2, bn2):
    y = o.relu(o.bn(o.conv(x, w1, 1, 1), *bn1))
    y = o.bn(o.conv(y, w2, 1, 1), *bn2)
    return o.relu(o.add(y, x))


def inverted_residual(o, x, w_exp, bn_exp, w_dw, bn_dw, w_proj, bn_proj):
    y = o.relu6(o.bn(o.conv(x, w_exp, 1, 0), *bn_exp))
    y = o.relu6(o.bn(o.dwconv(y, w_dw, 1, 1), *bn_dw))
    y = o.bn(o.conv(y, w_proj, 1, 0), *bn_proj)
    return o.add(y, x)


def preprocess_stem(o, img, mean, std, w, bn):
    return o.relu6(o.bn(o.conv(o.preprocess(img, mean, std), w, 2, 1), *bn))


OPS_A = {"stem": stem, "basic_block": basic_block, "inverted_residual": inverted_residual,
         "preprocess_stem": preprocess_stem}


def _rng(kind, seed):
    return np.random.default_rng([int(seed), kind])


def _bn(rng, c):
    """[gamma, beta, running_mean, running_var] — 학습된 BN 과 비슷한 범위."""
    return np.stack([rng.uniform(0.5, 1.5, c), rng.uniform(-0.5, 0.5, c),
                     rng.uniform(-0.5, 0.5, c), rng.uniform(0.5, 2.0, c)]).astype(np.float32)


def _w(rng, shape):
    """He 초기화 크기의 [-a, a) 균등 분포 (OIHW, fan_in = C_in x KH x KW)."""
    fan_in = int(np.prod(shape[1:]))
    return (rng.uniform(-1, 1, shape) * math.sqrt(3.0 / fan_in)).astype(np.float32)


def make_args(num, layout, size, seed=0):
    """Part A 입력 (레이아웃 순서). seed 마다 다르다."""
    r = _rng(int(num), seed)
    L = lambda a: to_layout(a, layout)                   # noqa: E731
    if num == "01":
        n, h, w, co = size
        x = r.uniform(-1, 1, (n, 3, h, w)).astype(np.float32)
        return L(x), L(_w(r, (co, 3, 7, 7))), _bn(r, co)
    if num == "02":
        n, h, w, c = size
        x = r.uniform(-1, 1, (n, c, h, w)).astype(np.float32)
        return L(x), L(_w(r, (c, c, 3, 3))), _bn(r, c), L(_w(r, (c, c, 3, 3))), _bn(r, c)
    if num == "03":
        n, h, w, c, t = size
        hid = c * t
        x = r.uniform(-1, 1, (n, c, h, w)).astype(np.float32)
        return (L(x), L(_w(r, (hid, c, 1, 1))), _bn(r, hid), L(_w(r, (hid, 1, 3, 3))), _bn(r, hid),
                L(_w(r, (c, hid, 1, 1))), _bn(r, c))
    if num == "04":
        n, h, w, co = size
        img = r.integers(0, 256, (n, 3, h, w), dtype=np.uint8)
        if size == PROBLEMS["04"]["sizes"][0]:              # 실제 크기 (검증 · 시간) 는 ImageNet 값
            mean, std = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
        else:                                               # 작은 크기는 다른 mean/std (상수를 박아 넣지 못하게)
            mean = r.uniform(0.3, 0.6, 3).astype(np.float32)
            std = r.uniform(0.15, 0.35, 3).astype(np.float32)
        return L(img), mean, std, L(_w(r, (co, 3, 3, 3))), _bn(r, co)
    raise KeyError(num)


def reference_a(num, layout, args):
    """float64 기준값 (레이아웃 순서)."""
    nchw = [from_layout(a, layout) if isinstance(a, np.ndarray) and a.ndim == 4 else a for a in args]
    y = OPS_A[PROBLEMS[num]["fn"]](Ref, *nchw)
    return to_layout(y, layout) if layout == "nhwc" else y


def flops_a(num, size):
    if num == "01":
        n, h, w, co = size
        oh, ow = (h - 1) // 2 + 1, (w - 1) // 2 + 1
        return 2 * n * oh * ow * co * 147
    if num == "02":
        n, h, w, c = size
        return 2 * 2 * n * h * w * c * c * 9
    if num == "03":
        n, h, w, c, t = size
        return n * h * w * (2 * c * c * t * 2 + 2 * c * t * 9)
    if num == "04":
        n, h, w, co = size
        oh, ow = (h - 1) // 2 + 1, (w - 1) // 2 + 1
        return 2 * n * oh * ow * co * 27
    return 0


# ================================================= Part B — 모델 (기준값이 쓰는 식)

def _bnp(p, pre):
    return p[pre + ".weight"], p[pre + ".bias"], p[pre + ".running_mean"], p[pre + ".running_var"]


def resnet18(o, img, p):
    x = o.preprocess(img, p["preprocess.mean"], p["preprocess.std"])
    x = o.maxpool(o.relu(o.bn(o.conv(x, p["conv1.weight"], 2, 3), *_bnp(p, "bn1"))))
    for li, s0 in zip((1, 2, 3, 4), (1, 2, 2, 2)):
        for bi in (0, 1):
            pre = f"layer{li}.{bi}."
            s = s0 if bi == 0 else 1
            y = o.relu(o.bn(o.conv(x, p[pre + "conv1.weight"], s, 1), *_bnp(p, pre + "bn1")))
            y = o.bn(o.conv(y, p[pre + "conv2.weight"], 1, 1), *_bnp(p, pre + "bn2"))
            idt = x
            if pre + "downsample.0.weight" in p:
                idt = o.bn(o.conv(x, p[pre + "downsample.0.weight"], s, 0), *_bnp(p, pre + "downsample.1"))
            x = o.relu(o.add(y, idt))
    return o.linear(o.gap(x), p["fc.weight"], p["fc.bias"])


def mobilenetv2(o, img, p):
    x = o.preprocess(img, p["preprocess.mean"], p["preprocess.std"])
    x = o.relu6(o.bn(o.conv(x, p["stem.conv.weight"], 1, 1), *_bnp(p, "stem.bn")))
    i, cin = 0, 32
    for t, c, n, s0 in MBV2_CFG:
        for j in range(n):
            s = s0 if j == 0 else 1
            pre = f"blocks.{i}."
            y = x
            if t != 1:
                y = o.relu6(o.bn(o.conv(y, p[pre + "expand.conv.weight"], 1, 0), *_bnp(p, pre + "expand.bn")))
            y = o.relu6(o.bn(o.dwconv(y, p[pre + "dw.conv.weight"], s, 1), *_bnp(p, pre + "dw.bn")))
            y = o.bn(o.conv(y, p[pre + "project.conv.weight"], 1, 0), *_bnp(p, pre + "project.bn"))
            x = o.add(y, x) if (s == 1 and cin == c) else y
            i, cin = i + 1, c
    x = o.relu6(o.bn(o.conv(x, p["head.conv.weight"], 1, 0), *_bnp(p, "head.bn")))
    return o.linear(o.gap(x), p["fc.weight"], p["fc.bias"])


MODELS = {"resnet18": (resnet18, CIFAR_MEAN, CIFAR_STD), "mobilenetv2": (mobilenetv2, CIFAR_MEAN, CIFAR_STD)}


def submitted_model(model):
    """학생이 올린 압축 · 재학습 모델 model/<model>.npz (없으면 None)."""
    path = HERE / "model" / f"{model}.npz"
    return path if path.is_file() else None


def load_params(model, layout, submitted=False):
    """weights/<model>.npz -> {이름: 배열}. 4 차원 배열 (합성곱 가중치) 은 레이아웃 순서 (OIHW / OHWI) 로 바꾸고,
    전처리 상수 preprocess.mean / preprocess.std (3,) 를 더한다 (파일에 있으면 그 값).
    submitted 이면 model/<model>.npz 가 있을 때 그 파일 — 배열 이름 · 모양 · dtype 은 학생 마음 (prepare 가 읽는다).
    4 차원 배열은 OIHW 로 저장해 두면 레이아웃에 맞게 바뀌어 들어간다. 정수 dtype 은 그대로 둔다."""
    path = (submitted and submitted_model(model)) or HERE / "weights" / f"{model}.npz"
    with np.load(path, allow_pickle=False) as f:
        p = {}
        for k in f.files:
            a = f[k]
            a = a.astype(np.float32) if a.dtype.kind == "f" else a
            p[k] = to_layout(a, layout) if a.ndim == 4 else a
    _, mean, std = MODELS[model]
    p.setdefault("preprocess.mean", np.array(mean, np.float32))
    p.setdefault("preprocess.std", np.array(std, np.float32))
    return p


def load_calib(layout):
    """samples/cifar10_calib.npz -> images uint8 (1000, ...) — CIFAR-10 train 에서 클래스마다 100 장 (calibration 용)."""
    with np.load(HERE / "samples" / "cifar10_calib.npz") as f:
        return to_layout(f["images"], layout)


def test_batches(layout, seed):
    """CIFAR-10 test 앞 TEST_TOTAL 장 (채점마다 이미지마다 화소 값 TEST_NOISE 개를 ±1) 을 N_E2E 장씩 -> ([묶음, ...], labels)."""
    with np.load(HERE / "samples" / "cifar10_test.npz") as f:
        images, labels = f["images"][:TEST_TOTAL], f["labels"][:TEST_TOTAL].astype(np.int64)
    rng = _rng(91, seed)
    flat = images.reshape(len(images), -1).astype(np.int16)
    rows = np.repeat(np.arange(len(images)), TEST_NOISE)
    cols = np.concatenate([rng.choice(flat.shape[1], TEST_NOISE, replace=False) for _ in range(len(images))])
    flat[rows, cols] += rng.choice(np.array([-1, 1], np.int16), rows.size)
    flat[rows, cols] = np.where((flat[rows, cols] < 0) | (flat[rows, cols] > 255), 2 * images.reshape(len(images), -1)
                                [rows, cols].astype(np.int16) - flat[rows, cols], flat[rows, cols])     # 0 · 255 는 반대쪽으로
    images = to_layout(flat.reshape(images.shape).astype(np.uint8), layout)
    return [np.ascontiguousarray(images[i:i + N_E2E]) for i in range(0, TEST_TOTAL, N_E2E)], labels


def load_samples(layout):
    """samples/cifar10_50.npz -> (images uint8, labels). images 는 레이아웃 순서."""
    with np.load(HERE / "samples" / "cifar10_50.npz") as f:
        return to_layout(f["images"], layout), f["labels"].astype(np.int64)


def random_images(layout, n, seed):
    return to_layout(_rng(77, seed).integers(0, 256, (n, 3, 32, 32), dtype=np.uint8), layout)


def reference_b(model, layout, images, params):
    """float64 기준 logits (N, 10)."""
    nchw = {k: (from_layout(v, layout) if v.ndim == 4 else v) for k, v in params.items()}
    return MODELS[model][0](Ref, from_layout(images, layout), nchw)


_FLOPS_B = {}


def flops_b(model):
    """묶음 50 장의 FLOP (합성곱 · 완전연결층의 곱셈-덧셈 x 2)."""
    if model not in _FLOPS_B:
        Ref.flops = 0
        MODELS[model][0](Ref, np.zeros((1, 3, 32, 32), np.uint8), load_params(model, "nchw"))   # 제공 모델 (dense)
        _FLOPS_B[model] = Ref.flops * N_E2E
    return _FLOPS_B[model]


# ================================================= 규칙 (정적 분석)

ALLOWED_IMPORTS = {"numpy", "halide", "math"}
FORBIDDEN = [                       # 점으로 이은 이름의 앞부분이 이것이면 위반
    "numpy.dot", "numpy.matmul", "numpy.einsum", "numpy.tensordot", "numpy.inner",
    "numpy.outer", "numpy.vdot", "numpy.kron", "numpy.linalg", "numpy.fft", "numpy.convolve",
    "numpy.correlate", "numpy.lib", "numpy.ctypeslib",
    "halide.Internal", "halide.load_plugin", "halide.Target", "halide.Feature", "halide.ExternFuncArgument",
    "halide.ThreadPoolKeepAwake", "halide.Generator", "halide.generator", "halide.imageio", "halide.halide_",
]
FORBIDDEN_ATTRS = {"dot", "matmul", "tensordot", "einsum",          # x.dot(y) 같은 메서드
                   "define_extern", "apply_autoscheduler", "with_feature", "set_custom_trace",
                   "set_custom_print", "set_custom_allocator", "set_custom_do_par_for", "set_custom_do_task",
                   "add_custom_lowering_pass", "jit_handlers", "set_jit_externs", "add_autoscheduler",
                   "compile_to", "compile_to_assembly", "compile_to_bitcode", "compile_to_c", "compile_to_file",
                   "compile_to_header", "compile_to_llvm_assembly", "compile_to_lowered_stmt",
                   "compile_to_conceptual_stmt", "compile_to_module", "compile_to_multitarget_object_files",
                   "compile_to_multitarget_static_library", "compile_to_object", "compile_to_static_library",
                   "gpu", "gpu_blocks", "gpu_threads", "gpu_tile", "gpu_single_thread", "hexagon", "offload"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "getattr", "setattr", "globals", "vars", "open"}

RULES_TEXT = """\
필수 — Halide 로 구현한다: 실행 시간의 50% 이상을 Halide 파이프라인 안 (Func.realize · Pipeline.realize ·
       compile_to_callable 로 만든 Callable 호출) 에서 보내야 한다 (채점기가 실제로 돌려서 잰다)
필수 — 단일 스레드: Halide 스레드는 채점기가 1 개로 고정한다 (HL_NUM_THREADS=1). parallel() 을 써도 되지만
       빨라지지 않는다. 실행 중에 스레드가 늘거나 여러 코어를 동시에 쓰면 fail
금지 — 하나라도 쓰면 그 파일은 fail
    numpy 행렬곱·선형대수·FFT: dot · matmul · @ · einsum · tensordot · inner · outer · vdot · kron ·
                              linalg · fft · convolve · correlate · x.dot(...)
    Halide 의 바깥 코드 · 컴파일 대상 바꾸기: define_extern · Internal · load_plugin · Target · Feature ·
        with_feature · compile_to_* (compile_to_callable 만 허용) · set_custom_* · jit_handlers ·
        apply_autoscheduler (자동 스케줄러는 분석 스크립트에서 돌리고, 나온 스케줄을 코드로 옮겨 적는다) · gpu*
    numpy · halide · math 밖의 import (같은 폴더의 다른 파일, numba, scipy, torch 포함)
    eval · exec · compile · __import__ · getattr · setattr · globals · vars · open
    (Part A 는 호출마다 x · 가중치 · BN 을 바꾸므로 BN 접기 · 가중치 다시 늘어놓기도 시간 안에 들어간다.
     Part B 는 prepare 에서 BN 을 접고 파이프라인을 만들어 컴파일해 두지만, 이미지는 호출마다 바뀐다)
    Part B 지식증류: 제공 모델 (또는 다른 모델) 의 출력 · soft label · 중간 특징을 학습 목표로 쓰는 것
                    (채점기로는 가려내지 못하므로 model/ 의 학습 코드를 조교가 읽고 확인한다)
허용 (Part B)
    모델 압축 — 양자화 (PTQ · QAT, 비트 수 자유, 정수 연산도 Halide 로) · 프루닝 (비구조 · 구조) · 재학습
    (정답 레이블로 fine-tuning) · calibration (prepare 의 calib = CIFAR-10 train 1,000 장). 밖에서 학습한 모델은
    model/<모델>.npz 와 학습 코드 (model/ 폴더) 를 함께 올린다. 정답률은 test 2,000 장 (채점마다 화소 몇 개를 ±1) 으로 잰다
허용
    halide 의 Func · Var · RDom · Expr · ImageParam · Param · Buffer · Pipeline · BoundaryConditions ·
    모든 스케줄 명령 (split · tile · reorder · vectorize · unroll · compute_at · store_at · compute_root ·
    compute_with · fuse · rfactor · bound · align_bounds · prefetch · memoize ...), compile_jit · compile_to_callable ·
    get_jit_target_from_environment · get_host_target, numpy 의 배열 생성·변형·원소별 연산"""


def _dotted(node, alias):
    """ast 노드 -> 'numpy.linalg.norm' 같은 이름 (별칭을 풀어서). 모르면 None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(alias.get(node.id, node.id))
        return ".".join(reversed(parts))
    return None


def check_rules(path):
    """제출 파일에서 규칙 위반을 찾아 [(줄 번호, 설명), ...] 로 돌려준다."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    alias, bad = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORTS and a.name != "verify":
                    bad.append((node.lineno, f"import {a.name}"))
                alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else root
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                bad.append((node.lineno, f"from {node.module} import ..."))
            for a in node.names:
                if a.name == "*":
                    bad.append((node.lineno, f"from {node.module} import * (이름을 하나씩 가져오세요)"))
                alias[a.asname or a.name] = f"{node.module}.{a.name}"
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            bad.append((node.lineno, "@ (행렬곱)"))
        elif isinstance(node, (ast.Attribute, ast.Name)):
            name = _dotted(node, alias)
            if name and any(name == f or name.startswith(f + ".") for f in FORBIDDEN):
                bad.append((node.lineno, name))
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                bad.append((node.lineno, f".{node.attr}(...)"))
            if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
                bad.append((node.lineno, f".{node.attr} (이중 밑줄 속성)"))
        elif isinstance(node, ast.keyword) and node.arg is None:
            bad.append((getattr(node.value, "lineno", 0), "**kwargs 로 인자 넘기기 (옵션 숨기기 방지)"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in FORBIDDEN_CALLS:
            bad.append((node.lineno, f"{node.func.id}(...)"))
    if not any(v == "halide" or v.startswith("halide.") for v in alias.values()):
        bad.append((0, "halide 를 import 하지 않았습니다 (Halide 로 구현하세요)"))
    return sorted(set(bad))


# ================================================= 정확도 · 시간 측정 입력

def relerr(a, ref):
    a = np.asarray(a, np.float64)
    return float(np.max(np.abs(a - ref)) / max(float(np.max(np.abs(ref))), 1e-12))


def _check_out(y, ref, tol):
    """(통과?, 설명)."""
    if not isinstance(y, np.ndarray) or y.shape != ref.shape:
        return False, f"모양이 다릅니다: {getattr(y, 'shape', type(y))} != {ref.shape}"
    if y.dtype != np.float32:
        return False, f"dtype 이 float32 가 아닙니다: {y.dtype}"
    if not np.all(np.isfinite(y)):
        return False, "NaN 또는 inf 가 있습니다"
    err = relerr(y, ref)
    return err <= tol, f"상대오차 {err:.1e}" + ("" if err <= tol else f" > {tol:g}")


def _same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def validate_a(fn, num, layout, seed, timing):
    """Part A: 실제 크기 + 작은 크기 2 개. [(이름, 통과?, 설명), ...]. 첫 호출 시간은 timing["first_s"]."""
    out = []
    for i, size in enumerate(PROBLEMS[num]["sizes"]):
        label = ("실제 크기 " if i == 0 else "작은 크기 ") + str(size)
        args = make_args(num, layout, size, seed=seed * 100 + i)
        keep = [a.copy() for a in args]
        try:
            started = time.perf_counter()
            y = fn(*args)
            if i == 0:
                timing["first_s"] = time.perf_counter() - started
        except Exception as e:                      # noqa: BLE001 — 어떤 오류든 실패로 보인다
            out.append((label, False, f"{type(e).__name__}: {e}"))
            continue
        ok, msg = _check_out(y, reference_a(num, layout, keep), TOL)
        if ok and not _same(args, keep):
            ok, msg = False, "입력 배열을 바꿨습니다"
        out.append((label, ok, msg))
    return out


def _check_logits(y, n):
    """(통과?, 설명) — (n, 10) float32 유한값."""
    if not isinstance(y, np.ndarray) or y.shape != (n, 10):
        return False, f"모양이 다릅니다: {getattr(y, 'shape', type(y))} != {(n, 10)}"
    if y.dtype != np.float32:
        return False, f"dtype 이 float32 가 아닙니다: {y.dtype}"
    if not np.all(np.isfinite(y)):
        return False, "NaN 또는 inf 가 있습니다"
    return True, "(N, 10) float32"


def validate_b(prepare, infer, model, layout, seed, timing):
    """Part B: prepare 한 번 + 샘플 50 장 + 무작위 이미지 3 장 · 1 장 (모양 · 같은 이미지는 같은 결과).
    logits 값은 기준과 비교하지 않는다 (압축 허용) — 정확도는 grade_file 의 test 정답률. (결과, state, 원본 params)."""
    params = load_params(model, layout, submitted=True)
    keep = {k: v.copy() for k, v in params.items()}
    calib = load_calib(layout)
    calib0 = calib.copy()
    started = time.perf_counter()
    try:
        state = prepare(params, calib)
    except Exception as e:                          # noqa: BLE001
        return [("prepare", False, f"{type(e).__name__}: {e}")], None, keep
    timing["prepare_s"] = time.perf_counter() - started
    out = []
    if timing["prepare_s"] > PREPARE_LIMIT:
        out.append(("prepare", False, f"{timing['prepare_s']:.0f} 초 > 한도 {PREPARE_LIMIT:.0f} 초"))
    samples, labels = load_samples(layout)
    three = random_images(layout, 3, seed * 100 + 1)
    cases = [("샘플 50 장", samples), ("무작위 3 장", three), ("무작위 1 장 (3 장의 둘째와 같은 이미지)", three[1:2].copy())]
    outputs = []
    for i, (label, img) in enumerate(cases):
        img0 = img.copy()
        try:
            started = time.perf_counter()
            y = infer(state, img)
            if i == 0:
                timing["first_s"] = time.perf_counter() - started
        except Exception as e:                      # noqa: BLE001
            out.append((label, False, f"{type(e).__name__}: {e}"))
            outputs.append(None)
            continue
        ok, msg = _check_logits(y, img0.shape[0])
        if ok and i == 0:
            msg += f" · 정답률 {np.mean(np.argmax(y, 1) == labels):.0%} (참고)"
        if ok and i == 2 and outputs[1] is not None:
            err = relerr(y, np.asarray(outputs[1][1:2], np.float64))
            if err > CONSIST_TOL:
                ok, msg = False, f"같은 이미지인데 3 장 묶음 때와 결과가 다릅니다 (상대오차 {err:.1e})"
        if ok and not np.array_equal(img, img0):
            ok, msg = False, "입력 이미지를 바꿨습니다"
        out.append((label, ok, msg))
        outputs.append(np.array(y) if ok else None)
    if not all(np.array_equal(params[k], keep[k]) for k in keep):
        out.append(("params", False, "params 배열을 바꿨습니다 (복사해서 쓰세요)"))
    if not np.array_equal(calib, calib0):
        out.append(("calib", False, "calib 이미지를 바꿨습니다"))
    return out, state, keep


def accuracy_factor(num, correct):
    """정답률 배율 = 1 - ACCURACY_PENALTY x max(0, float 정답 - ACCURACY_SLACK - 정답), 0 ~ 1."""
    short = max(0, PROBLEMS[num]["float"] - ACCURACY_SLACK - correct)
    return min(1.0, max(0.0, 1.0 - ACCURACY_PENALTY * short))


def check_data(model):
    """Part B 가 쓰는 제공 파일이 원본인지. 문제가 있으면 설명, 없으면 None."""
    for name in (f"weights/{model}.npz", "samples/cifar10_50.npz", "samples/cifar10_test.npz",
                 "samples/cifar10_calib.npz"):
        path = HERE / name
        if not path.is_file():
            return f"{name} 이 없습니다"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != DATA_SHA256[name]:
            return f"{name} 이 원본과 다릅니다 (git checkout -- {name} 로 되돌리세요)"
    return None


def benchmark_inputs(key, seed, state=None):
    """(index, 인자) 를 차례로 낸다. index 0 은 준비 호출 WARMUP 번 (같은 입력, 재지 않는다),
    그다음은 측정 REPEAT 번 (index 1..REPEAT, 서로 다른 입력).

    호출마다 **다른 입력** 을 주므로, 앞 호출의 결과를 저장해 두었다가 다시 쓰는 제출은 빨라지지 않는다.
    Part A 는 x · 가중치 · BN 이 모두 바뀌고, Part B 는 state 는 그대로 두고 test 이미지 50 장 (채점마다 잡음) 이 바뀐다.
    """
    num, layout = key.split("_")
    if PROBLEMS[num]["part"] == "A":
        size = PROBLEMS[num]["sizes"][0]
        make = lambda index: make_args(num, layout, size, seed=seed * 100 + 10 + index)    # noqa: E731
    else:
        batches = test_batches(layout, seed)[0]
        make = lambda index: (state, batches[index])                                       # noqa: E731
    args = make(0)
    for _ in range(WARMUP):
        yield 0, args
    for index in range(1, REPEAT + 1):
        yield index, make(index)


def measure(fn, inputs):
    """benchmark_inputs() 의 입력으로 fn 을 돌려 측정 호출의 시간(초) 목록. 대표값은 trimmed_mean()."""
    samples = []
    for index, args in inputs:
        started = time.perf_counter()
        fn(*args)
        if index:
            samples.append(time.perf_counter() - started)
    return samples


def trimmed_mean(samples):
    ordered = sorted(samples)
    kept = ordered[TRIM:len(ordered) - TRIM] if len(ordered) > 2 * TRIM else ordered
    return math.fsum(kept) / len(kept)


def probe_check(key, y, args, originals, params, rng):
    """시간 측정 호출 하나의 결과를 검사한다. 문제가 있으면 설명, 없으면 None.
    Part A 는 전체 출력을 float64 기준값과 비교한다. Part B 는 모양 · 유한값만 보고, 예측은 test 정답률에 들어간다."""
    num, layout = key.split("_")
    if PROBLEMS[num]["part"] == "A":
        ok, msg = _check_out(y, reference_a(num, layout, originals), TOL)
        if ok and not _same(args, originals):
            ok, msg = False, "입력 배열을 바꿨습니다"
        return None if ok else msg
    images = originals[1]
    ok, msg = _check_logits(y, images.shape[0])
    if ok and not np.array_equal(args[1], images):
        ok, msg = False, "입력 이미지를 바꿨습니다"
    return None if ok else msg


def problem_flops(key):
    """호출 한 번의 명목 연산량 (FLOP) — 합성곱 · 완전연결층의 곱셈-덧셈 x 2, 정의대로 (패딩 탭 포함) 센다.
    Part A 는 실제 크기 한 번, Part B 는 50 장 묶음. 리더보드는 이것을 시간으로 나눈 GFLOP/s 로 순위를 매긴다."""
    num, _ = key.split("_")
    pr = PROBLEMS[num]
    return flops_a(num, pr["sizes"][0]) if pr["part"] == "A" else flops_b(pr["model"])


CORRECT_RATIO = 0.2          # 리더보드 점수에서 통과(정확도)만으로 받는 배점의 비율, 나머지는 시간
FIXED_POINTS = {"05": 40.0, "06": 52.0}     # 문제별 배점 (NCHW + NHWC 합, 7주차와 같다). 나머지는 남은 점수를 연산량에 비례해


def problem_points(total=100.0):
    """(조교용) 리더보드 배점 {파일: 점수}, 합 total. 05 · 06 은 FIXED_POINTS, 01–04 는 남은 점수를 연산량에 비례해
    나누고, 한 문제의 배점은 NCHW 판과 NHWC 판이 반씩 갖는다. 소수 둘째 자리 (반올림 차이는 큰 파일부터 0.01 씩)."""
    flops = {num: problem_flops(f"{num}_nchw") for num in PROBLEMS if num not in FIXED_POINTS}
    rest = total - sum(FIXED_POINTS.values())
    per_problem = {**FIXED_POINTS, **{num: rest * value / sum(flops.values()) for num, value in flops.items()}}
    points = {key: round(per_problem[key[:2]] / 2, 2) for key in TIME_MS}
    cents = round((total - sum(points.values())) * 100)
    order = sorted(points, key=lambda key: (-problem_flops(key), key))
    for i in range(abs(cents)):
        key = order[i % len(order)]
        points[key] = round(points[key] + (0.01 if cents > 0 else -0.01), 2)
    return points


def leaderboard_points(key, ms, correct=None):
    """리더보드에서 이 파일이 받는 점수 (leaderboard-for-aidl 의 store.file_points 와 같은 식).
        t >= BASE_MS          배점 x 0.2                       (통과)
        GOAL_MS <= t < BASE   배점 x (0.2 + 0.8 x log(BASE / t) / log(BASE / GOAL))
        t < GOAL_MS           배점 x GOAL / t                  (목표보다 빠르면 상한 없이 빠른 만큼 = GFLOP/s 에 비례)
    Part B (05 · 06) 는 여기에 정답률 배율 accuracy_factor(correct) 을 곱한다.
    """
    base, goal = TIME_MS[key]
    points = problem_points()[key]
    factor = accuracy_factor(key[:2], correct) if PROBLEMS[key[:2]]["part"] == "B" and correct is not None else 1.0
    if ms < goal:
        return points * goal / ms * factor
    fraction = 0.0 if ms >= base else math.log(base / ms) / math.log(base / goal)
    return points * (CORRECT_RATIO + (1 - CORRECT_RATIO) * fraction) * factor


def gflops(key, ms):
    return problem_flops(key) / (ms * 1e-3) / 1e9


# ================================================= 코어 고정 · 실행 검사

_ALL_CPUS = None


def pin():
    """Linux: 코어 하나에 묶는다 (원래 쓸 수 있던 코어 목록은 기억해 둔다)."""
    global _ALL_CPUS
    if hasattr(os, "sched_setaffinity"):
        try:
            if _ALL_CPUS is None:
                _ALL_CPUS = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {sorted(_ALL_CPUS)[0]})
        except OSError:
            pass


def unpin():
    """코어 제한을 풀어 원래 쓸 수 있던 코어를 모두 쓰게 한다 (여러 코어 사용 검사용)."""
    if hasattr(os, "sched_setaffinity") and _ALL_CPUS:
        try:
            os.sched_setaffinity(0, _ALL_CPUS)
        except OSError:
            pass


CPU_RATIO_LIMIT = 1.3       # CPU 시간 / 경과 시간 이 이보다 크면 여러 코어를 동시에 쓴 것
HALIDE_SHARE_MIN = 0.5      # 실행 시간 중 Halide 파이프라인 안에서 보낸 비율의 하한


def os_threads():
    """이 프로세스의 OS 스레드 수 (Linux 는 /proc, macOS 는 ps). 알 수 없으면 None."""
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        pass
    try:
        out = subprocess.run(["ps", "-M", "-p", str(os.getpid())], capture_output=True,
                             text=True, timeout=10).stdout
        return max(0, len(out.strip().splitlines()) - 1)
    except Exception:                               # noqa: BLE001
        return None


def snapshot():
    return {"threads": os_threads()}


class HalideMeter:
    """Halide 파이프라인을 실행하는 세 입구 (Func.realize · Pipeline.realize · Callable 호출) 를 감싸
    호출 수와 그 안에서 보낸 시간을 잰다. 제출 모듈을 불러오기 전에 켠다 (모듈이 메서드를 미리 꺼내 둬도 잡히게)."""
    ENTRIES = ((hl.Func, "realize"), (hl.Pipeline, "realize"), (hl.Callable, "__call__"))

    def __init__(self):
        self.calls, self.seconds, self._orig = 0, 0.0, {}

    def install(self):
        for cls, name in self.ENTRIES:
            original = getattr(cls, name)
            self._orig[(cls, name)] = original

            def timed(*args, _original=original, **kwargs):
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.calls += 1
                    self.seconds += time.perf_counter() - started
            setattr(cls, name, timed)
        return self


METER = HalideMeter()


def check_runtime(before):
    """한 번 이상 돌린 뒤의 실행 규칙 위반 목록 [(0, 설명), ...]."""
    bad = []
    after = snapshot()
    if before["threads"] is not None and after["threads"] is not None \
            and after["threads"] > before["threads"] and sys.platform.startswith("linux"):
        bad.append((0, f"실행 중에 스레드가 생겼습니다 ({before['threads']} -> {after['threads']} 개)"))
    if os.environ.get("HL_NUM_THREADS") != "1":
        bad.append((0, "HL_NUM_THREADS 가 1 이 아닙니다"))
    if METER.calls == 0:
        bad.append((0, "Halide 파이프라인이 한 번도 실행되지 않았습니다 (realize 또는 Callable 로 계산하세요)"))
    return bad


def profile_call(mod, fn_name, args):
    """코어 제한을 풀고 돌려 (Halide 안 비율, CPU/경과 비율, 경과 초) 를 잰다. 실패하면 None."""
    unpin()
    try:                                            # 짧으면 비율이 흔들리므로 누적 0.3 초까지 (최대 10 번)
        inside0 = METER.seconds
        c0, t0 = time.process_time(), time.perf_counter()
        for _ in range(10):
            getattr(mod, fn_name)(*args)
            if time.perf_counter() - t0 >= 0.3:
                break
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        inside = METER.seconds - inside0
    except Exception:                               # noqa: BLE001
        return None
    finally:
        pin()
    return min(1.0, inside / wall), cpu / wall, wall


def check_profile(prof):
    """profile_call 결과의 위반 목록."""
    if prof is None:
        return [(0, "Halide 사용 비율 및 CPU 실행 검사를 완료하지 못했습니다")]
    share, ratio, _ = prof
    bad = []
    if ratio > CPU_RATIO_LIMIT:
        bad.append((0, f"여러 코어를 동시에 썼습니다 (CPU 시간 / 경과 시간 = {ratio:.2f})"))
    if share < HALIDE_SHARE_MIN:
        bad.append((0, f"실행 시간의 {share:.0%} 만 Halide 파이프라인 안에서 보냈습니다 (최소 {HALIDE_SHARE_MIN:.0%})"))
    return bad


def naive_starter(num, layout):
    """(조교용 --baseline) 출발점 코드 (스케줄 없음) 의 함수들. starter/ 에 템플릿의 출발점 파일을 둔다.
    Part A 는 (연산 함수,), Part B 는 (prepare, infer)."""
    path = HERE / "starter" / f"submission{num}_{layout}.py"
    spec = importlib.util.spec_from_file_location(f"starter{num}_{layout}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(getattr(module, name) for name in entry_names(f"{num}_{layout}"))


# ================================================= 문제 하나 채점 · 리더보드 제출

def machine():
    """채점 기계 이름. Docker 안에서는 /proc/device-tree 가 보이지 않으므로 /proc/cpuinfo 의 Model 줄을 본다."""
    model = f"{platform.system()} {platform.machine()}"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("Model") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return model


# ------------------------------------------------- GitHub 정보 · 리더보드 제출

def pushed_at():
    """이 실행을 일으킨 푸시의 시각 (GitHub push 이벤트의 repository.pushed_at, UTC ISO). 없으면 빈 문자열.
    리더보드는 이 시각으로 Classroom50 제출 기간(available_from ~ due) 안의 제출인지 판정한다."""
    try:
        if os.environ.get("GITHUB_EVENT_NAME") != "push":
            return ""
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        value = event.get("repository", {}).get("pushed_at")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))
        return value if isinstance(value, str) else ""
    except (KeyError, OSError, ValueError, AttributeError):
        return ""


def github_metadata():
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    return {
        "repository": repository,
        "repository_id": os.environ.get("GITHUB_REPOSITORY_ID", ""),
        "repository_owner_id": os.environ.get("GITHUB_REPOSITORY_OWNER_ID", ""),
        "repository_name": repository.partition("/")[2],
        "nickname": repository.partition("/")[2].split("-")[-1],
        "actor": os.environ.get("GITHUB_ACTOR", ""),
        "actor_id": os.environ.get("GITHUB_ACTOR_ID", ""),
        "classroom_owner": os.environ.get("OWNER") or os.environ.get("USERNAME"),
        "sha": os.environ.get("GITHUB_SHA", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF", ""),
        "pushed_at": pushed_at(),
    }


class _NoSubmissionRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        raise urllib.error.HTTPError(request.full_url, code, "제출 주소의 리다이렉트는 허용하지 않습니다", headers, response)


def submit_result(result):
    base = os.environ.get("LEADERBOARD_URL", LEADERBOARD_URL).rstrip("/")
    # Internal submissions must bypass environment HTTP proxies and redirects.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoSubmissionRedirect())
    payload = {
        "period": os.environ.get("LEADERBOARD_PERIOD", PERIOD),
        "fixture_id": FIXTURE_ID,
        "result": result,
        "github": github_metadata(),
    }
    body = json.dumps(payload, allow_nan=False, ensure_ascii=False).encode()
    for attempt in range(3):
        request = urllib.request.Request(base + "/api/submissions", data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=20) as response:
                receipt = json.load(response)
            if not receipt.get("accepted"):
                raise RuntimeError("리더보드가 제출을 승인하지 않았습니다")
            return receipt
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"리더보드 제출 실패 (HTTP {error.code})") from None
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise RuntimeError("리더보드 서버에 연결하지 못했습니다") from None
        time.sleep(0.5 * (attempt + 1))


# ------------------------------------------------- 파일별 검증 · 별도 프로세스 · CLI

def verifier_sha256():
    return hashlib.sha256((HERE / "verify.py").read_bytes()).hexdigest()


def failure(key, reason, **extra):
    return {"problem": key, "status": "fail", "ms": None,
            "reason": str(reason)[:1000], **extra}


def entry_names(key):
    num, layout = key.split("_")
    pr = PROBLEMS[num]
    if pr["part"] == "A":
        return [f"{pr['fn']}_{layout}"]
    return [f"prepare_{layout}", f"infer_{layout}"]


def log(message):
    """진행 상황 (표준 오류). 채점 프로세스의 표준 오류는 --verbose 일 때만 보인다."""
    print(message, file=sys.stderr, flush=True)


def grade_file(key, path=None, timed=True, baseline=False):
    num, layout = key.split("_")
    pr = PROBLEMS[num]
    path = Path(path or HERE / f"submission{key}.py").resolve()
    try:
        violations = check_rules(path)
    except (SyntaxError, OSError, UnicodeDecodeError) as error:
        return failure(key, f"{type(error).__name__}: {error}")
    if violations:
        return failure(key, "; ".join(f"{path.name}:{line} {what}" if line else what for line, what in violations),
                       violations=violations)
    if pr["part"] == "B":
        problem = check_data(pr["model"])
        if problem:
            return failure(key, problem)
    seed = run_seed()
    log(f"[{key}] {pr['name']} · {path.name} · seed {seed}")
    pin()
    # Halide JIT (LLVM) 와 스레드 풀을 먼저 한 번 띄운다: 컴파일러 초기화는 학생 코드의 병렬 실행이 아니다
    _warm_x = hl.Var("x")
    _warm = hl.Func("verify_warmup")
    _warm[_warm_x] = hl.cast(hl.Float(32), _warm_x)
    _warm.parallel(_warm_x)
    _warm.realize([8])
    before = snapshot()
    METER.install()
    METER.calls, METER.seconds = 0, 0.0
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    missing = [name for name in entry_names(key) if not callable(getattr(module, name, None))]
    if missing:
        return failure(key, f"함수가 없습니다: {', '.join(missing)}", seed=seed)
    fns = [getattr(module, name) for name in entry_names(key)]
    timing = {"first_s": 0.0}
    if pr["part"] == "A":
        checks, state, params = validate_a(fns[0], num, layout, seed, timing), None, None
    else:
        checks, state, params = validate_b(fns[0], fns[1], pr["model"], layout, seed, timing)
    for label, ok, message in checks:
        log(f"  [{'PASS' if ok else 'FAIL'}] 정확도 {label}: {message}")
    if not all(ok for _, ok, _ in checks) or len(checks) != 3:
        return failure(key, "; ".join(f"{label}: {message}" for label, ok, message in checks if not ok),
                       checks=checks, seed=seed)
    if timing["first_s"] > COMPILE_LIMIT:
        return failure(key, f"첫 호출이 {timing['first_s']:.0f} 초로 한도({COMPILE_LIMIT:.0f} 초)를 넘었습니다",
                       checks=checks, seed=seed)
    violations = check_runtime(before)
    if violations:
        return failure(key, "; ".join(what for _, what in violations), checks=checks, violations=violations, seed=seed)
    run = fns[-1]
    samples = []
    predicted = {}                                  # Part B: test 묶음 번호 -> 예측 클래스
    if timed:
        rng = np.random.default_rng([seed, 99])
        for index, args in benchmark_inputs(key, seed, state):
            if index == 0:                          # 준비 호출 — 재지도, 검사하지도 않는다
                run(*args)
                continue
            originals = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args)
            started = time.perf_counter()
            y = run(*args)
            elapsed = time.perf_counter() - started
            problem = probe_check(key, y, args, originals, params, rng)
            if problem:
                return failure(key, f"시간 측정 입력 {index}: {problem}", checks=checks, seed=seed)
            if pr["part"] == "B":
                predicted[index] = np.argmax(y, 1)
            samples.append(elapsed)
            log(f"  측정 {index}/{REPEAT}: {elapsed * 1e3:.2f} ms")
            del y, args, originals
    accuracy = None
    if pr["part"] == "B":
        batches, labels = test_batches(layout, seed)
        try:
            for index, batch in enumerate(batches):
                if index in predicted:
                    continue
                y = run(state, batch.copy())
                ok, msg = _check_logits(y, batch.shape[0])
                if not ok:
                    return failure(key, f"test 묶음 {index}: {msg}", checks=checks, seed=seed)
                predicted[index] = np.argmax(y, 1)
        except Exception as e:                      # noqa: BLE001
            return failure(key, f"test 묶음: {type(e).__name__}: {e}", checks=checks, seed=seed)
        pred = np.concatenate([predicted[i] for i in range(len(batches))])
        correct = int(np.count_nonzero(pred == labels))
        accuracy = {"test_correct": correct, "test_total": TEST_TOTAL,
                    "accuracy_factor": accuracy_factor(num, correct), "submitted_model": bool(submitted_model(pr["model"]))}
        log(f"  test 정답 {correct:,} / {TEST_TOTAL:,} 장 (float 모델 {pr['float']:,} 장)")
        if correct < MIN_CORRECT[num]:
            return failure(key, f"test 정답 {correct:,} / {TEST_TOTAL:,} 장 — 최소 {MIN_CORRECT[num]:,} 장 "
                                f"({MIN_CORRECT[num] / TEST_TOTAL:.0%}) 보다 적습니다", checks=checks, seed=seed, **accuracy)
    profile_args = next(benchmark_inputs(key, seed, state))[1]
    profile = profile_call(module, entry_names(key)[-1], profile_args)
    violations = check_runtime(before) + check_profile(profile)
    if violations:
        return failure(key, "; ".join(what for _, what in violations), checks=checks, violations=violations, seed=seed)
    result = {"problem": key, "status": "success", "ms": None,
              "checks": checks, "first_s": timing["first_s"], "halide_share": profile[0], "cpu_ratio": profile[1],
              "reason": "", "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "verifier_sha256": verifier_sha256(), "seed": seed,
              "machine": {"system": platform.system(), "arch": platform.machine(), "model": machine()}}
    if "prepare_s" in timing:
        result["prepare_s"] = timing["prepare_s"]
    if accuracy:
        result.update(accuracy)
    result["halide_version"] = getattr(hl, "__version__", HALIDE_VERSION)
    if timed:
        result["ms"] = trimmed_mean(samples) * 1000                 # 리더보드에 보내는 값
        result["samples_ms"] = [value * 1000 for value in samples]  # 측정 순서대로 REPEAT 개
        result["warmup"], result["repeat"], result["trim"] = WARMUP, REPEAT, TRIM
        result["base_ms"], result["goal_ms"] = TIME_MS[key]
        result["gflops"] = gflops(key, result["ms"])
    if baseline:
        if pr["part"] == "A":
            naive, naive_state = naive_starter(num, layout)[0], None
        else:
            prepare_b, naive = naive_starter(num, layout)
            naive_state = prepare_b(load_params(pr["model"], layout), load_calib(layout))
        result["naive_ms"] = trimmed_mean(measure(naive, benchmark_inputs(key, seed, naive_state))) * 1000
    return result


def run_worker(key, path, timed=True, baseline=False, verbose=False):
    with tempfile.TemporaryDirectory(prefix="aidl-verify-") as directory:
        report = Path(directory) / "result.json"
        env = os.environ.copy()
        for name in ("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_URL", "GITHUB_TOKEN", "GH_TOKEN"):
            env.pop(name, None)
        command = [sys.executable, str(HERE / "verify.py"), "--_worker", key,
                   "--_result", str(report), "--_submission", str(path.resolve())]
        if not timed:
            command.append("--no-time")
        if baseline:
            command.append("--baseline")
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=None if verbose else subprocess.PIPE,
                                   env=env, cwd=HERE, start_new_session=os.name == "posix")
        try:
            process.communicate(timeout=WORKER_LIMIT)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.communicate()
            return failure(key, f"채점 실행 시간 한도({WORKER_LIMIT:.0f} 초)를 넘었습니다")
        if not report.is_file():
            return failure(key, "채점 프로세스가 정상적으로 종료되지 않았습니다 (메모리 부족이나 강제 종료)")
        try:
            result = json.loads(report.read_text())
            if result.get("problem") != key or result.get("status") not in ("success", "fail"):
                raise ValueError("invalid result")
            return result
        except (OSError, ValueError):
            return failure(key, "채점 결과를 읽지 못했습니다")


def time_summary(result):
    """success 아래에 적는 실행 시간 두 줄 (표준 오류). 표준 출력은 success/fail 한 줄을 유지한다."""
    model = str(result.get("machine", {}).get("model", ""))
    note = "" if "Raspberry Pi 5" in model else " · 기준 기계(Raspberry Pi 5)가 아니므로 참고용"
    samples = " ".join(f"{value:.1f}" for value in result.get("samples_ms", []))
    base, goal = TIME_MS[result["problem"]]
    return (f"  {result['ms']:.1f} ms  (준비 {WARMUP}회 뒤 {REPEAT}회 측정, 가장 빠른 {TRIM}회와 가장 느린 {TRIM}회를 뺀 "
            f"{REPEAT - 2 * TRIM}회 평균{note})\n"
            f"  출발점 코드 {base:g} ms · 목표 {goal:g} ms · {result.get('gflops', 0):.1f} GFLOP/s · "
            f"리더보드 점수 {leaderboard_points(result['problem'], result['ms'], result.get('test_correct')):.2f} "
            f"(배점 {problem_points()[result['problem']]:g})\n"
            + (f"  test 정답 {result['test_correct']:,} / {result['test_total']:,} 장 (float 모델 "
               f"{PROBLEMS[result['problem'][:2]]['float']:,} 장) -> 정답률 배율 {result['accuracy_factor']:.2f}"
               + (" · model/ 의 모델" if result.get("submitted_model") else "") + "\n"
               if result.get("test_correct") is not None else "") +
            f"  측정값(ms, 측정 순서): {samples}")


def _keys(values):
    values = values or [f"{n}_{layout}" for n in PROBLEMS for layout in LAYOUTS]
    result = []
    for value in values:
        if value in PROBLEMS:
            result.extend(f"{value}_{layout}" for layout in LAYOUTS)
            continue
        name = Path(value).name
        match = re.fullmatch(r"(?:submission|answer)?(0[1-6]_(?:nchw|nhwc))(?:\.py)?", name)
        if not match:
            raise ValueError(f"제출 파일 이름이 유효하지 않습니다: {value} (예: submission01_nchw.py, 05_nhwc, 03)")
        result.append(match.group(1))
    return list(dict.fromkeys(result))


def suggest_time_ms(results):
    """(조교용) 정답의 결과로 TIME_MS 를 제안한다. BASE_MS = 출발점 코드, GOAL_MS = 정답 x 1.25 (0.1 ms 올림)."""
    lines = ["제안 TIME_MS (BASE_MS = 출발점 코드, GOAL_MS = 정답 x 1.25, 0.1 ms 단위 올림):"]
    for r in results:
        if r["status"] == "success" and r.get("ms") is not None and r.get("naive_ms") is not None:
            base = math.ceil(r["naive_ms"] * 10) / 10
            goal = math.ceil(r["ms"] * 1.25 * 10) / 10
            lines.append(f'    "{r["problem"]}": ({base:g}, {goal:g}),'
                         f'      # 기준 {r["naive_ms"]:.2f} ms, 정답 {r["ms"]:.2f} ms ({r["naive_ms"] / r["ms"]:.1f}x)')
    return "\n".join(lines)


def period_spec():
    """(조교용) 리더보드에 등록할 11 주차 기간 설정. 서버에서 manage add 로 등록한다:
        uv run python -c "import verify, json; print(json.dumps(verify.period_spec(), ensure_ascii=False, indent=2))"
    """
    return {
        "id": PERIOD, "title": "11주차", "subtitle": "Halide 스케줄 · 연산 퓨전 + 모델 전체 · NCHW / NHWC",
        "kind": "weekly", "order": 11, "fixture_id": FIXTURE_ID, "runtime_checks": True, "metric": "ms",
        "classroom_assignment": "week-11",      # Classroom50 과제 이름 — 이 과제의 제출 기간 안의 푸시만 순위에 반영
        # 점수로 순위: 파일 점수는 leaderboard_points() (목표보다 빠르면 상한 없이 GOAL / t 배), 배점은 problem_points().
        # 점수가 같으면 파일별 GFLOP/s 의 조화평균이 큰 쪽이 앞선다
        "ranking": "points", "correct_ratio": CORRECT_RATIO, "beyond_goal": "speedup",
        # 실행 확인 (리더보드 integrity.py): 이 파일들이 템플릿과 같은 커밋의 결과만 받는다
        "integrity": {"protected": ["verify.py", "pyproject.toml", "uv.lock", ".python-version"]},
        # 05 · 06 의 test 정답률: Runner 가 결과의 test_correct 에 적어 보낸다. 파일 점수 x 정답률 배율
        # (1 - penalty x max(0, float_correct - slack - test_correct)), min_correct 미만은 0 점 (채점기에서도 fail)
        "accuracy": {"field": "test_correct", "total": TEST_TOTAL, "source": "runner", "penalty": ACCURACY_PENALTY,
                     "slack": ACCURACY_SLACK, "points": True},
        "problems": [{"key": key, "filename": f"submission{key}.py", "title": PROBLEMS[key[:2]]["name"],
                      "check_count": 3, "base_ms": base, "goal_ms": goal, "flops": problem_flops(key),
                      "points": problem_points()[key],
                      **({"model": PROBLEMS[key[:2]]["model"], "float_correct": PROBLEMS[key[:2]]["float"],
                          "min_correct": MIN_CORRECT[key[:2]]} if PROBLEMS[key[:2]]["part"] == "B" else {})}
                     for key, (base, goal) in TIME_MS.items()],
    }


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--no-time", action="store_true")
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--json")
    parser.add_argument("--answers", type=Path)
    options = parser.parse_args(argv)
    keys = _keys(options.files)
    results = []
    for key in keys:
        path = options.answers / f"answer{key}.py" if options.answers else HERE / f"submission{key}.py"
        matching = next((Path(value) for value in options.files if Path(value).name == f"submission{key}.py"), None)
        if matching is not None:
            path = matching
        if len(keys) > 1:
            print(f"[{key}] {PROBLEMS[key[:2]]['name']}", file=sys.stderr, flush=True)
        try:
            result = run_worker(key, path, not options.no_time, options.baseline, options.verbose)
        except Exception as error:
            result = failure(key, f"{type(error).__name__}: {error}")
        if (options.submit or in_ci()) and not options.no_submit and not options.no_time and not options.answers:
            try:
                result["receipt"] = submit_result(result)
            except Exception as error:
                result["submission_error"] = str(error)
        if options.verbose:
            print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        success = result["status"] == "success" and not result.get("submission_error")
        print("success" if success else "fail", flush=True)
        if success and result.get("ms") is not None:
            print(time_summary(result), file=sys.stderr, flush=True)
        if result.get("receipt", {}).get("counted") is False:
            print(f"  리더보드: {result['receipt'].get('note', '제출했지만 점수에 반영하지 않음')}"
                  " — Classroom50 제출 기간 밖의 푸시입니다", file=sys.stderr, flush=True)
        elif not success:
            print(f"  이유: {result.get('submission_error') or result.get('reason')}", file=sys.stderr, flush=True)
        results.append(result)
        report_dir = os.environ.get("VERIFY_REPORT_DIR")
        if report_dir:
            directory = Path(report_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{key}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if options.answers and options.baseline:
        print(suggest_time_ms(results), file=sys.stderr, flush=True)
    if options.json:
        total_ms = sum(r["ms"] for r in results) if all(
            r["status"] == "success" and r.get("ms") is not None for r in results) else None
        Path(options.json).write_text(json.dumps({"github": github_metadata(), "results": results,
                                                  "total_ms": total_ms, "unit": "ms"},
                                                 ensure_ascii=False, indent=2, allow_nan=False))
    return all(r["status"] == "success" and not r.get("submission_error") for r in results)


def entrypoint(argv):
    if "--_worker" in argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--_worker", required=True)
        parser.add_argument("--_result", type=Path, required=True)
        parser.add_argument("--_submission", type=Path, required=True)
        parser.add_argument("--no-time", action="store_true")
        parser.add_argument("--baseline", action="store_true")
        options = parser.parse_args(argv)
        try:
            result = grade_file(options._worker, options._submission, not options.no_time, options.baseline)
        except Exception as error:
            result = failure(options._worker, f"{type(error).__name__}: {error}")
        options._result.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return result["status"] == "success"
    try:
        return main(argv)
    except (Exception, SystemExit) as error:
        if isinstance(error, SystemExit) and error.code in (0, None):
            raise
        print(f"  이유: {error}", file=sys.stderr)
        print("fail", flush=True)
        return False


if __name__ == "__main__":
    if "--rules" in sys.argv:
        print(RULES_TEXT)
        raise SystemExit(0)
    raise SystemExit(0 if entrypoint(sys.argv[1:]) else 1)
