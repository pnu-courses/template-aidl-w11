# 제11주 과제 — Halide 스케줄: 알고리즘과 스케줄의 분리 (단일 스레드)

강의에서 **무엇을 계산할지 (알고리즘)** 와 **어떻게 계산할지 (스케줄)** 를 따로 쓰는 Halide 를 배웠습니다. 이번 주에는
**7주차 문제 전체** — Part A 의 네 연산 (커널 퓨전) 과 Part B 의 **모델 두 개 (ResNet-18 · MobileNetV2 전체)** — 를 Halide 로
다시 풉니다. 알고리즘은 출발점 코드에 이미 맞게 들어 있습니다.
여러분이 고칠 것은 **스케줄** 입니다 — 루프를 자르고 (split · tile), 순서를 바꾸고 (reorder), 벡터로 묶고
(vectorize · unroll), 생산자를 소비자의 어느 루프 안에서 계산할지 (compute_at · store_at) 정합니다.
7주차에 Numba 로 손수 고쳐 쓴 것과 같은 최적화를 **계산식은 건드리지 않고** 스케줄 몇 줄로 하게 됩니다.
Part B 에서는 같은 생각을 **모델 단위로** 넓힙니다 — 모델 전체가 Halide 파이프라인 하나이고, 스케줄은 층 목록을 돌며
층마다 정합니다. Part A 에서 찾은 합성곱 스케줄을 함수로 만들어 수십 개 층에 걸되, 층 모양 (가로 32 → 1 칸, 채널 3 → 1280)
에 맞게 고르는 것이 일입니다.

모든 문제를 **NCHW · NHWC 두 레이아웃** 으로 풀므로 제출할 파일은 6 문제 x 2 = **12 개** 입니다. 출발점 코드는 받은 그대로
제출해도 `success` 이고, 스케줄을 바꿀수록 빨라집니다. 규칙과 정확도 검증을 통과하면 실행 시간(ms)으로 점수를 매깁니다.

## 구현할 연산

7주차와 **같은 문제 · 같은 크기 · 같은 검증** 입니다 — 7주차의 자기 Numba 기록과 바로 비교할 수 있습니다.

**Part A · 연산 (퓨전)** — 호출마다 입력 · 가중치 · BN 이 바뀝니다.

| 번호 | 연산 | 입력 → 출력 (NHWC 로 적음) | 스케줄의 쟁점 |
|---|---|---|---|
| 01 | ResNet-18 stem: conv7x7/2 + BN + ReLU + maxpool3x3/2 | (1,224,224,3) → (1,56,56,64) | conv 를 pool 의 어느 루프에서 (compute_at), 겹치는 줄 재사용 (store_at) |
| 02 | ResNet-18 BasicBlock: conv-BN-ReLU-conv-BN-(+x)-ReLU | (1,56,56,64) → (1,56,56,64) | 계산에 막힌 문제 — 누적값 타일을 레지스터에 |
| 03 | MobileNetV2 inverted residual: 1x1 확장(t=6) · 3x3 depthwise · 1x1 축소 (+x), ReLU6 | (4,56,56,24) → (4,56,56,24) | 6 배로 커진 중간 결과를 줄 단위로만 (sliding window) |
| 04 | 전처리 + MobileNetV2 stem: uint8 → (x/255 − mean)/std → conv3x3/2 + BN + ReLU6 | (4,224,224,3) uint8 → (4,112,112,32) | 출력 쓰기 · 정규화 패딩 |

**Part B · 모델 전체** — CIFAR-10 test 이미지 50 장 (uint8) → logits (50, 10). 함수가 두 개입니다.

| 번호 | 모델 | 가중치 | 50 장 연산량 | 스케줄의 쟁점 |
|---|---|---|---|---|
| 05 | ResNet-18 (7주차 05 와 같은 구조) | `weights/resnet18.npz` (44.8 MB) | 3.70 GFLOP | 층마다 가로 16 → 1 칸 — 작은 층에서 무엇을 벡터로 · 50 장 묶음으로 가중치 재사용 |
| 06 | MobileNetV2 (CIFAR-10 용, ReLU6) | `weights/mobilenetv2.npz` (9.2 MB) | 8.80 GFLOP | 층 종류 셋 (1x1 · depthwise · stem) 마다 스케줄 · 블록 안 퓨전 |

- `prepare_<lay>(params)` — 한 번만 부르고 **시간을 재지 않습니다** (한도 60 초, JIT 컴파일 포함). 출발점은 여기서 BN 을
  가중치에 접고 (`_fold`), 층마다 Func 를 쌓아 모델 전체를 파이프라인 하나로 만들고 (`_build`), 스케줄을 걸어 (`_schedule`)
  컴파일합니다. 가중치는 `hl.Buffer` 로 파이프라인에 박혀 상수가 됩니다.
- `infer_<lay>(state, images)` — **이 시간을 잽니다** (전처리 포함). 호출마다 다른 이미지 50 장 (검증에는 3 장 · 1 장도).
- 샘플 50 장은 `samples/cifar10_50.npz`, 모델 구조는 `hint.py` (PyTorch, 참고용), 가중치 이름은 `weights/<모델>.json`.

| 파일 | 입력 `x` | 가중치 | 출력 |
|---|---|---|---|
| `submission0N_nchw.py` | (N, C, H, W) | (C_out, C_in, KH, KW) (OIHW, depthwise 는 (C, 1, 3, 3)) | (N, C, H, W) |
| `submission0N_nhwc.py` | (N, H, W, C) | (C_out, KH, KW, C_in) (OHWI, depthwise 는 (C, 3, 3, 1)) | (N, H, W, C) |

BN 은 `(4, C)` 배열 `[gamma, beta, running_mean, running_var]` 이고 eps 는 1e-5 입니다. 모두 `float32` 이고 합성곱은 편향이
없습니다. 코드는 각 파일의 `START EDIT IT HERE` 와 `END EDIT IT HERE` 사이에만 작성하세요. 파일 머리말에 크기, 계산식,
검증 항목, 생각할 점이 있습니다.

## 출발점 코드의 구조

Part A (01–04):

```python
def L(c, i, j, n): ...          # 텐서 좌표 -> 이 레이아웃의 Halide 좌표 순서 (NCHW 는 (i, j, c, n), NHWC 는 (c, i, j, n))
def _algorithm():               # 무엇을 계산하나 — Func · Var · RDom. 고치지 않아도 맞다
    ...
    return dict(inputs=[...], out=y, conv=conv, ...)   # 스케줄에서 쓸 Func 들
def _schedule(s):               # 어떻게 계산하나 — 여기를 바꾼다
    s["conv"].compute_root()    # 출발점: 단계마다 compute_root, 벡터 · 타일 없음
def _compile():                 # 한 번만 JIT 컴파일 (compile_to_callable)
def basic_block_nhwc(x, ...):   # 채점기가 부르는 함수 — 출력 배열을 만들고 파이프라인을 실행
```

- **좌표 순서** — Halide 는 **첫 좌표가 가장 안쪽 (메모리에서 연속)** 축입니다. numpy `(N, C, H, W)` 배열은 Halide 에서
  `(W, H, C, N)` 입니다. 그래서 두 레이아웃 파일의 알고리즘은 같은 글자인데 (`L(c, i, j, n)` 이 순서를 맞춘다), **좋은
  스케줄은 다릅니다** — 벡터는 연속인 축에 걸어야 합니다 (NCHW 는 가로 `i`, NHWC 는 채널 `c`).
- **가중치 Func (`wf` …)** 는 가중치를 `(co, ci, kx, ky)` 순서로 읽는 Func 입니다. 출발점에서는 inline (복사 없음) 이고,
  `compute_root()` 하면 호출마다 한 번 다시 늘어놓아 `co` 가 연속인 배열이 됩니다 (`reorder_storage` 로 다른 순서도).
- **패딩 Func (`xp` · `a_pad` · `x_pad` …)** 는 바깥 좌표를 0 으로 읽습니다. inline 이면 읽을 때마다 select 를 하므로
  `compute_root()` 로 한 번 만들어 두는 것이 보통 빠릅니다.
- **알고리즘을 바꿔도 됩니다** (예: BN 을 가중치에 접는 Func 를 따로 두기). 값만 기준과 같으면 됩니다.

Part B (05–06):

```python
class Net:                      # 층을 하나씩 Func 로 쌓는다: input · conv · dwconv · maxpool · gap_fc
    layers = [...]              # 스케줄에서 쓰는 층 목록 — dict(kind, out, acc, pad, r, C, H, W, CI, K, stride)
def _build(params):             # 모델 구조 (알고리즘) — hint.py 와 같은 순서로 net.conv(...) 를 쌓는다
def _schedule(net):             # 여기를 바꾼다 — 출발점은 모든 층 out · acc 를 compute_root
    for layer in net.layers: ...
def prepare_nhwc(params):       # _build -> _schedule -> compile_to_callable
def infer_nhwc(state, images):  # 출력 배열을 만들고 실행
```

- 층의 가로 · 세로 · 채널은 파이썬 정수라 (이미지 크기 3 x 32 x 32 를 Halide 에 알려 두었다) **층마다 다른 스케줄** 을 고를 수
  있고, 상수 크기라 unroll 도 됩니다. 이미지 수 N 만 실행 때 정해집니다 (50 · 3 · 1).
- 층 하나는 `acc` (합성곱 누적, 갱신 정의가 있어 inline 할 수 없다) → `out` (편향 · 잔차 · 활성화) 입니다. 층 사이를 이을지
  (`compute_at` 으로 블록 안 퓨전) 는 여러분이 정합니다.
- 가로가 1 · 2 칸인 층 (ResNet layer3 · 4) 에서는 3x3 합성곱의 탭 대부분이 패딩 (0) 입니다. 알고리즘을 바꿔
  (`RDom.where`) 그 탭을 건너뛰어도 됩니다 — 연산량은 패딩 탭까지 센 값이라 그만큼 GFLOP/s 가 오릅니다.

## 구현 규칙

- **Halide 로 계산해야 합니다.** 실행 시간의 50% 이상을 Halide 파이프라인 안 (`Func.realize` · `Pipeline.realize` ·
  `compile_to_callable` 로 만든 Callable 호출) 에서 보내야 합니다. numpy 로 계산하고 Halide 는 흉내만 내면 `fail` 입니다.
- **단일 스레드입니다.** 채점기가 Halide 스레드를 1 개로 고정합니다 (`HL_NUM_THREADS=1`) — `parallel()` 을 써도 되지만
  빨라지지 않습니다. 실행 중에 스레드가 늘거나 여러 코어를 쓰면 `fail` 입니다.
- **Halide 의 바깥 코드와 컴파일 대상은 건드리지 않습니다.** `define_extern` · `Internal` · `load_plugin` · `Target` ·
  `with_feature` · `compile_to_*` (`compile_to_callable` 만 허용) · `set_custom_*` · `jit_handlers` · `gpu*` 은 쓸 수 없습니다.
- **자동 스케줄러는 제출 파일 밖에서.** `apply_autoscheduler` 는 제출 파일에서 쓸 수 없습니다. `tools/autoschedule.py` 로
  돌려 보고, 나온 스케줄을 `_schedule` 에 옮겨 적은 뒤 손보세요 (아래 "도구").
- **연산은 직접 구현해야 합니다.** NumPy 의 행렬곱 · 선형대수 · FFT (`dot`, `@`, `matmul`, `einsum`, `tensordot`, `linalg`,
  `fft`, `convolve` …) 는 쓸 수 없습니다.
- **`import` 는 `numpy`, `halide`, `math` 만 허용합니다** (Numba 도 이번 주에는 안 됩니다). 같은 폴더의 다른 파일도 불러올 수
  없습니다. `eval · exec · getattr · setattr · open` 같은 우회도 금지입니다.
- **입력 배열 (과 Part B 의 `params`) 을 직접 수정하면 안 됩니다.** `weights/` 와 `samples/` 의 파일도 바꾸면 안 됩니다.
- **채점 파일은 고치지 않습니다.** `verify.py` · `pyproject.toml` · `uv.lock` · `.python-version` 과 `.github/workflows/` 는 받은
  그대로 두세요 (`uv add` 도 하지 마세요 — 분석에 필요한 것은 `uv run --with <패키지>`). 리더보드는 결과를 받을 때마다 GitHub 에서
  그 실행과 커밋을 확인합니다 (진행 중인 그 커밋의 채점 실행인지, 그 실행을 돌리는 Raspberry Pi 가 보냈는지, 위 파일들이
  템플릿과 같은지, 저장소 맨 위에 `.venv` · `uv.toml` · `halide/` 같은 것이 없는지). 어긋나면 결과를 받지 않아 `fail` 입니다.

전체 규칙은 `uv run verify.py --rules` 로 볼 수 있습니다.

## 검증과 성능 측정

제출한 12 개 파일은 각각 자동으로 검증합니다. 기준 기계는 **Raspberry Pi 5 (ARM Cortex-A76, NEON)** 이며 코어 1 개만
사용합니다.

- **정확도** — 입력은 검증할 때마다 새로 뽑고 float64 기준값과 비교합니다. Part A 는 실제 크기 + **작은 크기 2 개**
  (다른 H · W · C, 예: 가로 9 칸 · 채널 8 개) 에서 상대오차 2e-4 이하. Part B 는 샘플 50 장 (예측 클래스도 같아야 함) +
  무작위 이미지 3 장 · 1 장에서 logits 상대오차 1e-3 이하. 시간을 재는 호출의 출력도 매번 검사합니다 (Part B 는 무작위 8 장).
- **실행 시간** — 준비 실행 3 회 (첫 호출의 JIT 컴파일 포함, 한도 300 초) 뒤 서로 다른 입력으로 9 회를 재서, 가장 빠른 2 회와
  가장 느린 2 회를 뺀 **5 회의 평균** 을 씁니다. Part B 는 `infer` 만 잽니다 (`prepare` 는 60 초 한도).

> **작은 크기 검사 주의.** 파이프라인은 한 번만 컴파일하고 크기만 바꿔 실행합니다. 출력 Func 를 `split(i, io, ii, 16)` 처럼
> 자르면 기본값 (`ShiftInwards`) 은 출력이 16 칸보다 작을 때 실행 오류가 납니다. split 크기를 작은 크기 (가로 8 칸 · 채널 8 개
> 등) 이하로 두거나, `tail=hl.TailStrategy.GuardWithIf` 를 주세요 (대신 조금 느려집니다). `bound()` 로 크기를 못 박아도
> 작은 크기에서 틀립니다. Part B 의 이미지 수도 50 · 3 · 1 로 바뀝니다 — 이미지 축 (`n`) 을 묶으려면 같은 주의가 필요합니다.

아래는 Raspberry Pi 5 에서 같은 방법으로 잰 참고 시간입니다. **출발점** 은 받은 그대로의 코드 (단계마다 `compute_root`,
벡터 · 타일 없음), **목표** 는 조교 정답 시간의 1.25 배입니다. 배점 (합 100 점) 은 7주차와 같이 05 가 40 점, 06 이 52 점
(NCHW · NHWC 반씩), 01–04 는 남은 8 점을 연산량에 비례해 나눕니다. 연산량은 합성곱 · 완전연결층의 곱셈-덧셈 x 2 를 정의대로
(패딩 탭 포함) 센 값으로, Part A 는 호출 한 번, Part B 는 50 장 묶음입니다.

| 파일 | 문제 | 연산량 (MFLOP) | 배점 | 출발점 (`BASE_MS`, ms) | 목표 (`GOAL_MS`, ms) | 목표의 GFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `submission01_nchw.py` | ResNet-18 stem (NCHW) | 236.0 | 0.95 | 135.6 | 12.5 | 18.9 |
| `submission01_nhwc.py` | ResNet-18 stem (NHWC) | 236.0 | 0.95 | 134.6 | 16.3 | 14.5 |
| `submission02_nchw.py` | ResNet-18 BasicBlock (NCHW) | 462.4 | 1.87 | 324.3 | 25.7 | 18.0 |
| `submission02_nhwc.py` | ResNet-18 BasicBlock (NHWC) | 462.4 | 1.87 | 354.2 | 21.2 | 21.8 |
| `submission03_nchw.py` | MobileNetV2 inverted residual (NCHW) | 205.9 | 0.83 | 134.2 | 18.8 | 11.0 |
| `submission03_nhwc.py` | MobileNetV2 inverted residual (NHWC) | 205.9 | 0.83 | 108.5 | 19.2 | 10.7 |
| `submission04_nchw.py` | 전처리 + MobileNetV2 stem (NCHW) | 86.7 | 0.35 | 74.3 | 6.5 | 13.3 |
| `submission04_nhwc.py` | 전처리 + MobileNetV2 stem (NHWC) | 86.7 | 0.35 | 76.4 | 8.5 | 10.2 |
| `submission05_nchw.py` | ResNet-18 모델 전체 (NCHW) | 3,701.7 | 20.00 | 3,233.3 | 311.7 | 11.9 |
| `submission05_nhwc.py` | ResNet-18 모델 전체 (NHWC) | 3,701.7 | 20.00 | 2,691.5 | 239.0 | 15.5 |
| `submission06_nchw.py` | MobileNetV2 모델 전체 (NCHW) | 8,797.6 | 26.00 | 7,358.4 | 1,077.2 | 8.2 |
| `submission06_nhwc.py` | MobileNetV2 모델 전체 (NHWC) | 8,797.6 | 26.00 | 5,533.2 | 772.2 | 11.4 |

조교 정답은 파일마다 스무 줄 안팎의 스케줄입니다 (05 는 1x1 · 2x2 층의 패딩 탭을 건너뛰는 알고리즘 변경 포함).
실행 시간은 7주차 조교의 Numba 정답의 0.65 ~ 2.2 배입니다 — 손으로 고친 Numba 를 넘어서는 것이 이번 주의 도전입니다. 단일 코어의 이론 최고 성능은
38.4 GFLOP/s (2.4 GHz x NEON FMA 2 개 x 4 칸 x 2) 입니다.

## 실행 방법

```bash
uv sync                                   # 처음 한 번 (numpy, halide 22.1.0)
uv run verify.py                          # 12 파일 모두
uv run verify.py submission02_nhwc.py     # 한 파일
uv run verify.py 03                       # 03_nchw, 03_nhwc
uv run verify.py --no-time                # 규칙 · 정확도 · 실행 검사만
uv run verify.py submission01_nchw.py --verbose --json /tmp/result.json
```

파일마다 `success` 또는 `fail` 을 출력합니다. 시간을 잰 경우 통과한 파일의 실행 시간 · GFLOP/s · 리더보드 점수를 함께
보여 주고, 실패한 파일은 이유를 보여 줍니다. 집 컴퓨터 (x86 · Apple Silicon) 의 시간은 참고용입니다 — 벡터 폭이 다르면
좋은 타일 크기도 다릅니다.

## 도구 (분석용 — 채점과 무관, import 제한 없음)

```bash
uv run tools/loop_nest.py 02_nhwc           # 지금 스케줄이 만든 루프 구조 (print_loop_nest) — 강의 슬라이드의 그 출력
uv run tools/loop_nest.py 02_nhwc --stmt    # + 낮춘 코드 (HTML) — 벡터 명령 · 할당 크기 확인
uv run tools/autoschedule.py 02_nhwc        # 자동 스케줄러 3 종 (Mullapudi2016 · Adams2019 · Li2018) 의 스케줄과 시간
```

`autoschedule.py` 는 스케줄을 C++ 문법으로 출력합니다. 끝의 `;` 를 빼고, 새로 나온 Var (`ci`, `xi` …) 를 `hl.Var("ci")` 로
만들어 파이썬으로 옮기면 됩니다. Adams2019 는 학습한 비용 모델로 고르므로 기계마다 결과가 다릅니다 — Pi 에서 돌려 보거나,
옮겨 적은 뒤 Pi 기준으로 다시 재세요. 조교가 Pi 5 (스레드 1 개) 에서 02_nhwc 로 돌려 보니 Mullapudi2016 은 461 ms,
Adams2019 는 176 ms (스케줄을 고르는 데만 3 분 반) 로, 손으로 고친 정답 (17 ms) 보다 10 배 넘게 느렸습니다 — 출발점으로만 쓰세요.

## 어디서부터

- **루프 구조부터 보세요.** `tools/loop_nest.py` 로 출발점의 루프를 보고, 스케줄을 한 줄 바꿀 때마다 어떻게 변하는지
  확인합니다. 무엇을 바꾸든 값은 그대로라 (`verify.py --no-time` 이 확인) 마음껏 시도해도 됩니다.
- **누적값을 레지스터에.** 합성곱 출력을 (출력 채널 몇 개 x 가로 몇 칸) 타일로 자르고, 합성곱 Func 를 그 타일 안에서
  (`compute_at`) 계산하고, 갱신 정의의 루프 순서를 `update().reorder(...)` 로 바꿔 축소 축 (`r`) 을 타일 바깥에 둡니다.
  강의의 행렬곱 스케줄과 같은 모양입니다. **vectorize 하고 남은 타일 축은 모두 `unroll`** 해야 타일 크기가 상수가 되어
  LLVM 이 누적값을 레지스터에 둡니다 — 남은 2 칸짜리 루프 하나 때문에 2.5 배 느려지기도 합니다.
- **생산자를 어디서 계산할까.** `compute_root` (전부 먼저) · `compute_at` (필요한 만큼 그때그때) · `store_at` +
  `compute_at` (겹치는 줄은 버퍼에 두고 다시 쓰기, sliding window) 를 바꿔 가며 재 보세요. 강의의 블러 실험처럼
  재계산 · 지역성 · 병렬성 사이의 줄다리기입니다.
- **레이아웃마다 다르게.** 같은 스케줄을 두 파일에 그대로 쓰면 한쪽은 느립니다. 연속인 축이 어디인지 보고 벡터 축을 정하세요.
- **4 번의 함정:** 정규화를 가중치에 접으면 테두리 패딩이 달라집니다 (패딩은 정규화한 공간의 0).

## 제출 방법

작성을 마친 뒤 변경 사항을 커밋하고 push 하세요.

```bash
git add . && git commit -m "solve" && git push
```

push 하면 Classroom50 의 Raspberry Pi 5 Runner 가 12 개 파일을 각각 검증하고, 결과를
[리더보드](http://164.125.249.13:25086/?period=week11) 의 **11주차** 에 자동으로 올립니다. 별도의 토큰이나 설정은
필요하지 않습니다. 브라우저가 주소를 HTTPS 로 바꾸면 주소의 프로토콜을 `http://` 로 변경해 접속하세요.

- 리더보드에는 파일별 최신 실행 시간과 검증 결과가 기록됩니다. GitHub 계정과 커밋 정보도 함께 기록됩니다.
- 리더보드는 **점수** 로 순위를 매깁니다. 배점은 위 표와 같고 (합 100 점), 파일마다 실행 시간 `t` 로 점수를 받습니다.

  | 실행 시간 | 점수 |
  |---|---|
  | `t >= BASE_MS` (통과만) | 배점 x 0.2 |
  | `GOAL_MS <= t < BASE_MS` | 배점 x (0.2 + 0.8 x log(BASE_MS / t) / log(BASE_MS / GOAL_MS)) — 목표 시간에서 배점 만점 |
  | `t < GOAL_MS` | 배점 x GOAL_MS / t — **상한이 없습니다.** 목표보다 2 배 빠르면 배점의 2 배 |

  실행 (푸시) 마다 12 개 파일의 점수를 더해 가장 높은 실행으로 순위를 매기고, 점수가 같으면 파일별 GFLOP/s 의 조화평균이
  큰 쪽이 앞섭니다. 실패한 파일은 0 점입니다.
- 목표 시간보다 느리더라도 검증을 통과하면 `success` 로 처리합니다.
- Classroom50 의 제출 기간 (마감 전) 에 **푸시한** 제출만 순위에 반영합니다. 채점 대기 때문에 결과가 마감 뒤에 도착해도,
  마감 전에 푸시했다면 반영됩니다.
- 실행 상태는 저장소의 **Actions** 탭에서 확인할 수 있습니다. 막히면 저장소의 **Issue** 로 물어보세요.
