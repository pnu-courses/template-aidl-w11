"""제3문제 (NHWC) — MobileNetV2 inverted residual (t = 6)   [Halide 스케줄]

    inverted_residual_nhwc(x, w_exp, bn_exp, w_dw, bn_dw, w_proj, bn_proj)
        x       (N, H, W, C)       = (4, 56, 56, 24)    float32 (4 장 묶음)
        w_exp   OHWI               = (144, 1, 1, 24)    1x1 확장 (C x t = 144)
        w_dw    (C, 3, 3, 1)       = (144, 3, 3, 1)     3x3 depthwise (채널마다 따로), stride 1 · padding 1
        w_proj  OHWI               = (24, 1, 1, 144)    1x1 축소
        bn_exp (4, 144) · bn_dw (4, 144) · bn_proj (4, 24)  [gamma, beta, running_mean, running_var], eps 1e-5
        ->  y                      = (4, 56, 56, 24)    float32

    계산 (PyTorch 와 같은 결과 — 7주차 03 과 같은 문제)
        a = relu6(bn_exp(conv1x1(x, w_exp)))                      # 56 x 56 x 144 (6 배로 커진다)
        a = relu6(bn_dw(depthwise3x3(a, w_dw)))
        y = bn_proj(conv1x1(a, w_proj)) + x                       # 잔차 (활성화 없음)
        relu6(v) = min(max(v, 0), 6)

생각할 점 — 확장된 56 x 56 x 144 중간 결과 (한 장 1.8 MB) 를 통째로 만들지 말고, 축소 (proj) 의 행 몇 줄을
계산할 때 필요한 확장 행만 만들기 (compute_at). depthwise 는 이웃 행 3 줄을 읽으므로 확장 행은 이웃 타일과
겹친다 — 겹치는 줄을 다시 계산할지 (compute_at), 버퍼에 두고 다시 쓸지 (store_at) 를 재 보고 고른다.

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

검증 (uv run verify.py submission03_nhwc.py)   ->  success 또는 fail
    규칙    단일 스레드 · Halide 로 계산 — 아래 금지 목록. 어기면 fail
            실행 검사: 실제로 돌려서 스레드가 늘거나 여러 코어를 쓰거나, Halide 파이프라인 안에서 보낸 시간이
            50% 미만이면 fail. 채점기가 HL_NUM_THREADS=1 로 돌리므로 parallel() 은 빨라지지 않는다
    정확도  실제 크기 + 작은 크기 2 개 (다른 H, W, C)에서 float64 기준값과 상대오차 2e-4 이하,
            출력 모양 · float32 · 입력 배열 불변. 시간을 재는 호출의 출력도 매번 검사한다
    시간    Raspberry Pi 5 · 코어 1 개. 준비 3 번 뒤 9 번 재서 (호출마다 다른 입력 · 가중치 · BN)
            가장 빠른 2 번과 가장 느린 2 번을 뺀 5 번의 평균 (ms) -> 리더보드
            참고: 출발점 코드 108.5 ms · 목표 19.2 ms (조교 정답 x 1.25). 7주차 같은 문제의 Numba 기록과 비교해 보자

금지
    numpy 행렬곱·선형대수·FFT: dot · matmul · @ · einsum · tensordot · inner · outer · linalg · fft ·
                              convolve · correlate · x.dot(...)
    Halide 의 바깥 코드 · 대상 바꾸기: define_extern · Internal · load_plugin · Target · compile_to_* (callable 만 허용) ·
                              apply_autoscheduler (자동 스케줄러는 분석 스크립트에서 돌리고 결과 스케줄을 옮겨 적는다)
    numpy · halide · math 밖의 import (numba · scipy · torch · 같은 폴더의 다른 파일 포함)
허용
    Halide 의 모든 알고리즘 · 스케줄 명령, numpy 의 배열 생성·변형·원소별 연산

확인: uv run submission03_nhwc.py      (= uv run verify.py submission03_nhwc.py)
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

def _zero_pad(f, wd, ht, name):
    """f 를 가로 [0, wd) x 세로 [0, ht) 밖에서 0 으로 읽는 Func (합성곱의 패딩)."""
    c, i, j, n = hl.Var("c"), hl.Var("i"), hl.Var("j"), hl.Var("n")
    g = hl.Func(name)
    inside = (i >= 0) & (i < wd) & (j >= 0) & (j < ht)
    g[L(c, i, j, n)] = hl.select(inside, f[L(c, hl.clamp(i, 0, wd - 1), hl.clamp(j, 0, ht - 1), n)], 0.0)
    return g


def _bn(bn, c, v):
    return (v - bn[c, 2]) * (bn[c, 0] / hl.sqrt(bn[c, 3] + hl.f32(1e-5))) + bn[c, 1]


def _algorithm():
    x = hl.ImageParam(hl.Float(32), 4, "x")
    w_exp = hl.ImageParam(hl.Float(32), 4, "w_exp")
    bn_exp = hl.ImageParam(hl.Float(32), 2, "bn_exp")
    w_dw = hl.ImageParam(hl.Float(32), 4, "w_dw")
    bn_dw = hl.ImageParam(hl.Float(32), 2, "bn_dw")
    w_proj = hl.ImageParam(hl.Float(32), 4, "w_proj")
    bn_proj = hl.ImageParam(hl.Float(32), 2, "bn_proj")
    c, i, j, n = hl.Var("c"), hl.Var("i"), hl.Var("j"), hl.Var("n")
    C, H, W = x.dim(CH).extent(), x.dim(HD).extent(), x.dim(WD).extent()
    HID = w_exp.dim(3).extent()                               # C x t (가중치의 C_out 차원은 두 레이아웃 모두 3 번)

    co, ci, kx, ky = hl.Var("co"), hl.Var("ci"), hl.Var("kx"), hl.Var("ky")
    wexp, wdw, wproj = hl.Func("wexp"), hl.Func("wdw"), hl.Func("wproj")    # 가중치를 co (또는 c) 가 첫 인자인 순서로
    wexp[co, ci] = w_exp[LW(co, ci, 0, 0)]
    wdw[co, kx, ky] = w_dw[LW(co, 0, ky, kx)]
    wproj[co, ci] = w_proj[LW(co, ci, 0, 0)]

    re = hl.RDom([(0, C)], "re")
    exp = hl.Func("exp")
    exp[L(c, i, j, n)] = hl.f32(0)
    exp[L(c, i, j, n)] += wexp[c, re.x] * x[L(re.x, i, j, n)]
    e = hl.Func("e")
    e[L(c, i, j, n)] = hl.clamp(_bn(bn_exp, c, exp[L(c, i, j, n)]), 0.0, 6.0)

    ep = _zero_pad(e, W, H, "e_pad")
    k = hl.RDom([(0, 3), (0, 3)], "k")
    dw = hl.Func("dw")
    dw[L(c, i, j, n)] = hl.f32(0)
    dw[L(c, i, j, n)] += wdw[c, k.x, k.y] * ep[L(c, i + k.x - 1, j + k.y - 1, n)]
    d = hl.Func("d")
    d[L(c, i, j, n)] = hl.clamp(_bn(bn_dw, c, dw[L(c, i, j, n)]), 0.0, 6.0)

    rp = hl.RDom([(0, HID)], "rp")
    proj = hl.Func("proj")
    proj[L(c, i, j, n)] = hl.f32(0)
    proj[L(c, i, j, n)] += wproj[c, rp.x] * d[L(rp.x, i, j, n)]
    y = hl.Func("y")
    y[L(c, i, j, n)] = _bn(bn_proj, c, proj[L(c, i, j, n)]) + x[L(c, i, j, n)]
    return dict(inputs=[x, w_exp, bn_exp, w_dw, bn_dw, w_proj, bn_proj], out=y,
                exp=exp, e=e, e_pad=ep, dw=dw, d=d, proj=proj, wexp=wexp, wdw=wdw, wproj=wproj,
                vars=(c, i, j, n), re=re, k=k, rp=rp)

def _schedule(s):
    """출발점: 단계마다 compute_root. 벡터 · 타일 없음."""
    for name in ("exp", "e", "dw", "d", "proj"):
        s[name].compute_root()

def _compile():
    """알고리즘 + 스케줄 -> 한 번만 JIT 컴파일한 Callable. 호출: _RUN(입력 Buffer ..., 출력 Buffer)."""
    s = _algorithm()
    _schedule(s)
    return s["out"].compile_to_callable(s["inputs"], hl.get_jit_target_from_environment())


_RUN = None

def inverted_residual_nhwc(x, w_exp, bn_exp, w_dw, bn_dw, w_proj, bn_proj):
    """1x1 확장 -> 3x3 depthwise -> 1x1 축소 (+x). 반환은 float32 연속 배열 (NHWC)."""
    global _RUN
    if _RUN is None:
        _RUN = _compile()
    y = np.empty_like(x)
    _RUN(hl.Buffer(x), hl.Buffer(w_exp), hl.Buffer(bn_exp), hl.Buffer(w_dw), hl.Buffer(bn_dw),
         hl.Buffer(w_proj), hl.Buffer(bn_proj), hl.Buffer(y))
    return y

#################### END EDIT IT HERE  ####################

def main():
    import verify
    raise SystemExit(0 if verify.main(["03_nhwc"]) else 1)


if __name__ == "__main__":
    main()
