"""제1문제 (NHWC) — ResNet-18 stem: conv7x7/2 + BN + ReLU + maxpool3x3/2   [Halide 스케줄]

    stem_nhwc(x, w, bn)
        x  (N, H, W, C)           = (1, 224, 224, 3)     float32 (정규화된 이미지)
        w  OHWI                   = (64, 7, 7, 3)        편향 없음
        bn (4, C_out)              = (4, 64)              [gamma, beta, running_mean, running_var], eps 1e-5
        ->  y                      = (1, 56, 56, 64)      float32

    계산 (PyTorch 와 같은 결과 — 7주차 01 과 같은 문제)
        a = conv(x, w, stride 2, padding 3)                       # 112 x 112 x 64
        a = (a - mean) / sqrt(var + 1e-5) * gamma + beta          # BN (추론)
        a = max(a, 0)                                             # ReLU
        y = maxpool(a, kernel 3, stride 2, padding 1)             # 패딩은 -inf -> 56 x 56 x 64
        H_conv = (H + 6 - 7) // 2 + 1,  H_out = (H_conv + 2 - 3) // 2 + 1   (W 도 같은 식)

생각할 점 — conv 를 pool 의 어느 루프 안에서 계산할까 (compute_at: 강의의 블러와 같은 생산자-소비자 문제).
pool 의 출력 한 줄은 conv 출력 세 줄을 읽고, 이웃한 줄끼리 한 줄씩 겹친다 (store_at 으로 겹치는 줄을 다시 쓰기).
C_in = 3 이라 입력 채널 축에 벡터를 걸면 낭비다 — 출력 채널 64 나 가로 축에 걸어 보자.

Halide 로 쓴다 (11주차 강의: 알고리즘과 스케줄의 분리)
    알고리즘 (_algorithm) 은 무엇을 계산하는지만 적는다. 출발점 코드의 알고리즘은 그대로 맞으니 고칠 필요가 없다.
    스케줄 (_schedule) 이 어떤 순서로 · 어디에 저장하며 계산할지를 정한다. 출발점은 단계마다 compute_root —
    단계마다 전체 중간 결과를 만들고 다음 단계가 읽는다 (7주차의 '융합하지 않은 연산 사슬') — 이고, 벡터도 타일도 없다.
    여기에 split · tile · reorder · vectorize · unroll · compute_at · store_at 을 걸어 빠르게 만든다.
    알고리즘을 바꿔도 된다 (예: BN 을 가중치에 접는 단계를 따로 두기, 가중치를 다시 늘어놓는 Func) — 값만 같으면 된다.

    좌표 순서 주의 — Halide 는 첫 좌표가 가장 안쪽 (메모리에서 연속) 축이다. 이 파일의 L(c, i, j, n) 이
    텐서 좌표를 NHWC 의 Halide 좌표 순서로 바꿔 준다. 그래서 두 레이아웃 파일의 알고리즘은 같은 글자인데,
    메모리 순서가 달라 좋은 스케줄은 서로 다르다 (vectorize 할 축이 어디가 연속인지에 달렸다).

    파이프라인은 처음 한 번만 JIT 컴파일해 (compile_to_callable) 모듈 전역에 두고, 호출마다 새 입력으로 실행한다.
    입력 크기가 바뀌어도 (작은 크기 검사) 다시 컴파일하지 않는다. 그래서 출력 Func 를 split · tile 할 때는
    tail=hl.TailStrategy.GuardWithIf 를 주자 — 기본값 (ShiftInwards) 은 출력이 split 크기보다 작으면 (작은 크기의
    가로 9 칸 같은) 실행 중 오류가 난다. bound() 로 크기를 못 박아도 작은 크기 검사에서 틀린다.
    가중치를 다시 늘어놓는 Func (wf ...) 와 패딩 Func (xp ...) 는 출발점에서 inline 이다 — compute_root 하면 한 번만
    만들어 두고 읽는다 (Func 의 저장 순서는 인자 순서: 첫 인자가 연속 축. reorder_storage 로 바꿀 수 있다).

검증 (uv run verify.py submission01_nhwc.py)   ->  success 또는 fail
    규칙    단일 스레드 · Halide 로 계산 — 아래 금지 목록. 어기면 fail
            실행 검사: 실제로 돌려서 스레드가 늘거나 여러 코어를 쓰거나, Halide 파이프라인 안에서 보낸 시간이
            50% 미만이면 fail. 채점기가 HL_NUM_THREADS=1 로 돌리므로 parallel() 은 빨라지지 않는다
    정확도  실제 크기 + 작은 크기 2 개 (다른 H, W, C)에서 float64 기준값과 상대오차 2e-4 이하,
            출력 모양 · float32 · 입력 배열 불변. 시간을 재는 호출의 출력도 매번 검사한다
    시간    Raspberry Pi 5 · 코어 1 개. 준비 3 번 뒤 9 번 재서 (호출마다 다른 입력 · 가중치 · BN)
            가장 빠른 2 번과 가장 느린 2 번을 뺀 5 번의 평균 (ms) -> 리더보드
            참고: 출발점 코드 134.6 ms · 목표 16.3 ms (조교 정답 x 1.25). 7주차 같은 문제의 Numba 기록과 비교해 보자

금지
    numpy 행렬곱·선형대수·FFT: dot · matmul · @ · einsum · tensordot · inner · outer · linalg · fft ·
                              convolve · correlate · x.dot(...)
    Halide 의 바깥 코드 · 대상 바꾸기: define_extern · Internal · load_plugin · Target · compile_to_* (callable 만 허용) ·
                              apply_autoscheduler (자동 스케줄러는 분석 스크립트에서 돌리고 결과 스케줄을 옮겨 적는다)
    numpy · halide · math 밖의 import (numba · scipy · torch · 같은 폴더의 다른 파일 포함)
허용
    Halide 의 모든 알고리즘 · 스케줄 명령, numpy 의 배열 생성·변형·원소별 연산

확인: uv run submission01_nhwc.py      (= uv run verify.py submission01_nhwc.py)
전체 규칙: uv run verify.py --rules
"""
import numpy as np
import halide as hl

#################### START EDIT IT HERE  ####################

def L(c, i, j, n):
    """텐서 좌표 (채널 c, 가로 i, 세로 j, 이미지 n) -> Halide 좌표. numpy (N, H, W, C) = Halide (C, W, H, N)."""
    return (c, i, j, n)


def LW(co, ci, ky, kx):
    """가중치 좌표 -> Halide 좌표. numpy OHWI (C_out, KH, KW, C_in) = Halide (C_in, KW, KH, C_out)."""
    return (ci, kx, ky, co)


CH, HD, WD = 0, 2, 1        # Halide 차원 번호: 채널 · 세로 · 가로 (x.dim(CH).extent() = C)

def _algorithm():
    x = hl.ImageParam(hl.Float(32), 4, "x")
    w = hl.ImageParam(hl.Float(32), 4, "w")
    bn = hl.ImageParam(hl.Float(32), 2, "bn")              # numpy (4, C) = Halide (C, 4)
    c, i, j, n = hl.Var("c"), hl.Var("i"), hl.Var("j"), hl.Var("n")

    xp = hl.BoundaryConditions.constant_exterior(x, 0.0)  # 패딩 3 은 0
    co, ci, kx, ky = hl.Var("co"), hl.Var("ci"), hl.Var("kx"), hl.Var("ky")
    wf = hl.Func("wf")                                     # 가중치를 (co, ci, kx, ky) 순서로 — 저장하면 co 가 연속
    wf[co, ci, kx, ky] = w[LW(co, ci, ky, kx)]
    r = hl.RDom([(0, 7), (0, 7), (0, x.dim(CH).extent())], "r")    # r.x = kx, r.y = ky, r.z = ci
    conv = hl.Func("conv")
    conv[L(c, i, j, n)] = hl.f32(0)
    conv[L(c, i, j, n)] += wf[c, r.z, r.x, r.y] * xp[L(r.z, 2 * i + r.x - 3, 2 * j + r.y - 3, n)]

    inv = bn[c, 0] / hl.sqrt(bn[c, 3] + hl.f32(1e-5))
    act = hl.Func("act")
    act[L(c, i, j, n)] = hl.max((conv[L(c, i, j, n)] - bn[c, 2]) * inv + bn[c, 1], 0.0)

    # maxpool 3x3/2, 패딩 -inf. 창은 언제나 안쪽 칸을 하나 이상 품으므로, 바깥 좌표를 가장자리로 당겨 읽어도
    # (clamp) 최댓값이 같다 — 당겨 온 칸은 이미 같은 창 안에 있다. 그래서 act 는 [0, W_conv) x [0, H_conv) 만 계산한다.
    hc = (x.dim(HD).extent() - 1) // 2 + 1
    wc = (x.dim(WD).extent() - 1) // 2 + 1
    p = hl.RDom([(0, 3), (0, 3)], "p")
    pool = hl.Func("pool")
    pool[L(c, i, j, n)] = hl.maximum(act[L(c, hl.clamp(2 * i + p.x - 1, 0, wc - 1),
                                           hl.clamp(2 * j + p.y - 1, 0, hc - 1), n)])
    return dict(inputs=[x, w, bn], out=pool, conv=conv, act=act, xp=xp, wf=wf, vars=(c, i, j, n), r=r, p=p)

def _schedule(s):
    """출발점: 단계마다 compute_root (전체 중간 결과를 만들고 다음 단계가 읽는다). 벡터 · 타일 없음."""
    s["conv"].compute_root()
    s["act"].compute_root()

def _compile():
    """알고리즘 + 스케줄 -> 한 번만 JIT 컴파일한 Callable. 호출: _RUN(입력 Buffer ..., 출력 Buffer)."""
    s = _algorithm()
    _schedule(s)
    return s["out"].compile_to_callable(s["inputs"], hl.get_jit_target_from_environment())


_RUN = None

def stem_nhwc(x, w, bn):
    """conv7x7/2 -> BN -> ReLU -> maxpool3x3/2. 반환은 float32 연속 배열 (NHWC)."""
    global _RUN
    if _RUN is None:
        _RUN = _compile()
    n, h, wd, _ = x.shape
    co = w.shape[0]
    hc, wc = (h - 1) // 2 + 1, (wd - 1) // 2 + 1
    y = np.empty((n, (hc - 1) // 2 + 1, (wc - 1) // 2 + 1, co), np.float32)
    _RUN(hl.Buffer(x), hl.Buffer(w), hl.Buffer(bn), hl.Buffer(y))
    return y

#################### END EDIT IT HERE  ####################

def main():
    import verify
    raise SystemExit(0 if verify.main(["01_nhwc"]) else 1)


if __name__ == "__main__":
    main()
