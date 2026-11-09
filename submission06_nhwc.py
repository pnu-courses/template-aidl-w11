"""제6문제 (NHWC) — MobileNetV2 모델 전체 (CIFAR-10, 50 장)   [Part B · 모델 단위 Halide 스케줄]

    prepare_nhwc(params, calib) -> state     한 번만 부른다. 시간을 재지 않는다 (한도 60 초 — JIT 컴파일 포함)
        calib:  CIFAR-10 train 1,000 장 uint8 (레이아웃 순서) — 양자화 calibration 등에 써도 된다
        params: dict {이름: float32 배열} — weights/mobilenetv2.npz 의 PyTorch 이름 그대로 (weights/mobilenetv2.json 참고).
                model/mobilenetv2.npz 를 올리면 그 파일 (아래 "모델 압축")
                합성곱 가중치는 (C_out, KH, KW, C_in) OHWI (depthwise 는 (C, 3, 3, 1)), BN 은 <이름>.weight · .bias ·
                .running_mean · .running_var (eps 1e-5), fc.weight (10, 1280) · fc.bias (10,), preprocess.mean · std (3,)
    infer_nhwc(state, images) -> logits       이 시간을 잰다 (전처리 포함)
        images (N, 32, 32, 3) uint8 = 50 장 (검증에는 3 장 · 1 장도 들어온다)  ->  logits (N, 10) float32

    구조 (hint.py 의 MobileNetV2 · 7주차 06 과 같다)
        x = (images / 255 - mean) / std
        stem conv3x3/1 (3->32) + BN + ReLU6                          32x32
        inverted residual 17 개 (t, c, n, s) = (1,16,1,1) (6,24,2,1) (6,32,3,2) (6,64,4,2) (6,96,3,1) (6,160,3,2) (6,320,1,1)
        head conv1x1 (320->1280) + BN + ReLU6 -> 평균 풀링 -> fc (1280 -> 10)
    50 장 묶음 = 8.80 GFLOP, 가중치 9.2 MB.

모델 단위의 알고리즘과 스케줄 분리 — _build 가 층마다 Func 를 쌓아 모델 전체를 Halide 파이프라인 하나로 만들고
(알고리즘), _schedule 이 층 목록 net.layers 를 돌며 층마다 어떻게 계산할지 정한다 (스케줄). 층 종류가 셋 (1x1 ·
depthwise 3x3 · stem 3x3) 이라 종류마다 스케줄 함수를 두고, 3 번 문제처럼 블록 안에서 확장 결과를 줄 단위로만
만들지 (compute_at) 정해 보자. 확장 층은 출력이 입력의 6 배라 메모리가 병목이다.

검증 (uv run verify.py submission06_nhwc.py)   ->  success 또는 fail
    규칙    Part A 와 같다 (단일 스레드 · Halide 안 50% 이상 · 금지 목록). 실행 검사는 infer 를 돌려서 한다
    정확도  CIFAR-10 test 앞 2,000 장 (채점마다 이미지마다 화소 8 개를 ±1) 의 정답 장수 — logits 값은 기준과 비교하지 않는다.
            최소 1,700 장보다 적으면 fail. 리더보드 점수 = 시간 점수 x 정답률 배율
            (1 - 0.01 x max(0, float 모델 1,846 장 - 3 - 정답 장수)). 그 밖에 (N, 10) float32 · 유한값,
            같은 이미지는 묶음과 상관없이 같은 결과, 입력 · params · calib 불변. weights/ · samples/ 는 바꾸면 fail
    시간    prepare 한 번 (재지 않음, 60 초 한도) 뒤 infer 를 test 50 장 묶음으로 준비 3 번, 측정 9 번
            (호출마다 다른 이미지, 이 예측도 정답률에 들어간다), 가장 빠른 2 번과 가장 느린 2 번을 뺀 5 번의 평균 (ms)
            참고: 출발점 코드 5533.2 ms · 목표 772.2 ms (조교 정답 x 1.25). 7주차 같은 문제의 Numba 기록과 비교해 보자

금지 · 허용 — Part A 와 같다 (uv run verify.py --rules). prepare 안의 numpy 계산 (BN 접기 · 가중치 다시 늘어놓기 ·
양자화 · 프루닝) 은 시간을 재지 않으므로 마음껏 해도 된다. 단 infer 는 Halide 파이프라인 안에서 시간의 50% 이상을 보내야 한다.

모델 압축 (허용) — 양자화 (PTQ · QAT, 비트 수 자유) · 프루닝 (비구조 · 구조) · 재학습 (정답 레이블로 fine-tuning) ·
calibration. 밖에서 학습한 가중치는 model/<모델>.npz (+ 학습 코드를 model/ 에) 로 올린다. 4 차원 배열은 OIHW 로 저장하면
레이아웃에 맞게 바뀌어 들어온다. 금지 — 지식증류 (제공 모델이나 다른 모델의 출력 · soft label · 중간 특징을 학습 목표로
쓰는 것). 채점기는 가려내지 못하므로 조교가 model/ 의 학습 코드를 읽고 확인한다 — 위반이면 그 주 점수 0.

확인: uv run submission06_nhwc.py      (= uv run verify.py submission06_nhwc.py)
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

EPS = 1e-5


def _fold(params, wname, bnname):
    """(prepare, 시간 안 잼) BN 을 합성곱에 접는다 -> (가중치 OIHW float32, 편향 (C_out,) float32)."""
    w = np.asarray(params[wname], np.float64)
    w = w.transpose(0, 3, 1, 2)                                                # 이 레이아웃 -> OIHW
    g, b, m, v = (np.asarray(params[bnname + s], np.float64)
                  for s in (".weight", ".bias", ".running_mean", ".running_var"))
    s = g / np.sqrt(v + EPS)
    return (w * s[:, None, None, None]).astype(np.float32), (b - m * s).astype(np.float32)


class Net:
    """층을 하나씩 Func 로 쌓는다. 모양 (C, H, W) 는 prepare 때 정해지는 파이썬 정수 — 이미지 수 N 만 실행 때 정해진다.

    net.layers 는 스케줄에서 쓰는 층 목록이다. 층마다 dict:
        kind   "conv" · "dw" · "pool" · "gap" · "fc"
        out    그 층의 출력 Func (BN · 활성화 · 잔차까지 들어간 값)
        acc    합성곱 누적 Func (갱신 정의가 있어 inline 할 수 없다 — 꼭 어디서 계산할지 정한다)
        pad    0 패딩 Func (패딩이 없으면 None) — inline 이면 읽을 때마다 select
        r      축소 RDom (r.x = kx, r.y = ky, r.z = ci)
        C, H, W  출력 모양,  CI · K · stride  입력 채널 · 커널 크기 · stride
    """

    def __init__(self):
        self.c, self.i, self.j, self.n = hl.Var("c"), hl.Var("i"), hl.Var("j"), hl.Var("n")
        self.layers = []
        self.keep = []                                           # Buffer 가 감싼 numpy 배열 (파이프라인과 함께 살아 있게)

    def buf(self, a):
        a = np.ascontiguousarray(a, np.float32)
        self.keep.append(a)
        return hl.Buffer(a)

    def input(self, mean, std):
        """uint8 이미지 -> (x / 255 - mean) / std."""
        c, i, j, n = self.c, self.i, self.j, self.n
        self.img = hl.ImageParam(hl.UInt(8), 4, "img")
        for d, e in ((CH, 3), (HD, 32), (WD, 32)):              # 이미지 크기는 언제나 3 x 32 x 32 — Halide 에 알려 주면
            self.img.dim(d).set_bounds(0, e)                     # 층마다 가로 · 세로가 상수가 된다 (unroll 할 수 있다)
        mb, sb = self.buf(mean), self.buf(std)
        x = hl.Func("input")
        x[L(c, i, j, n)] = (hl.f32(self.img[L(c, i, j, n)]) / 255.0 - mb[c]) / sb[c]
        self.layers.append(dict(kind="input", out=x, C=3, H=32, W=32))
        return x, (3, 32, 32)

    def _pad(self, f, shape, p, name):
        if p == 0:
            return None
        c, i, j, n = self.c, self.i, self.j, self.n
        _, H, W = shape
        g = hl.Func(name + "_pad")
        inside = (i >= 0) & (i < W) & (j >= 0) & (j < H)
        g[L(c, i, j, n)] = hl.select(inside, f[L(c, hl.clamp(i, 0, W - 1), hl.clamp(j, 0, H - 1), n)], 0.0)
        return g

    def _epilogue(self, name, acc, act, res):
        c, i, j, n = self.c, self.i, self.j, self.n
        v = acc[L(c, i, j, n)]
        if res is not None:
            v = v + res[L(c, i, j, n)]
        if act == "relu":
            v = hl.max(v, 0.0)
        elif act == "relu6":
            v = hl.clamp(v, 0.0, 6.0)
        out = hl.Func(name)
        out[L(c, i, j, n)] = v
        return out

    def conv(self, x, shape, w, b, stride, pad, act=None, res=None, name="conv"):
        """합성곱 (+ 접은 BN 의 편향) (+ 잔차) (+ 활성화). w 는 OIHW."""
        c, i, j, n = self.c, self.i, self.j, self.n
        CI, H, W = shape
        CO, _, K, _ = w.shape
        Ho, Wo = (H + 2 * pad - K) // stride + 1, (W + 2 * pad - K) // stride + 1
        wb = self.buf(w.transpose(2, 3, 1, 0))                   # numpy (KH, KW, CI, CO) = Halide (co, ci, kx, ky)
        bb = self.buf(b)
        xp = self._pad(x, shape, pad, name)
        src = xp if xp is not None else x
        r = hl.RDom([(0, K), (0, K), (0, CI)], name + "_r")       # r.x = kx, r.y = ky, r.z = ci
        acc = hl.Func(name + "_acc")
        acc[L(c, i, j, n)] = bb[c]
        acc[L(c, i, j, n)] += wb[c, r.z, r.x, r.y] * src[L(r.z, stride * i + r.x - pad, stride * j + r.y - pad, n)]
        out = self._epilogue(name, acc, act, res)
        self.layers.append(dict(kind="conv", name=name, out=out, acc=acc, pad=xp, r=r,
                                C=CO, H=Ho, W=Wo, CI=CI, K=K, stride=stride))
        return out, (CO, Ho, Wo)

    def dwconv(self, x, shape, w, b, stride, act="relu6", name="dw"):
        """3x3 depthwise (+ 편향) (+ 활성화). w 는 (C, 1, 3, 3)."""
        c, i, j, n = self.c, self.i, self.j, self.n
        C, H, W = shape
        K = w.shape[2]
        Ho, Wo = (H + 2 - K) // stride + 1, (W + 2 - K) // stride + 1
        wb = self.buf(w[:, 0].transpose(1, 2, 0))                 # numpy (KH, KW, C) = Halide (c, kx, ky)
        bb = self.buf(b)
        xp = self._pad(x, shape, 1, name)
        r = hl.RDom([(0, K), (0, K)], name + "_r")
        acc = hl.Func(name + "_acc")
        acc[L(c, i, j, n)] = bb[c]
        acc[L(c, i, j, n)] += wb[c, r.x, r.y] * xp[L(c, stride * i + r.x - 1, stride * j + r.y - 1, n)]
        out = self._epilogue(name, acc, act, None)
        self.layers.append(dict(kind="dw", name=name, out=out, acc=acc, pad=xp, r=r,
                                C=C, H=Ho, W=Wo, CI=C, K=K, stride=stride))
        return out, (C, Ho, Wo)

    def maxpool(self, x, shape, name="maxpool"):
        """3x3/2, 패딩 1 (-inf). 창은 언제나 안쪽 칸을 품으므로 바깥 좌표를 가장자리로 당겨 읽어도 최댓값이 같다."""
        c, i, j, n = self.c, self.i, self.j, self.n
        C, H, W = shape
        Ho, Wo = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        p = hl.RDom([(0, 3), (0, 3)], name + "_r")
        out = hl.Func(name)
        out[L(c, i, j, n)] = hl.maximum(x[L(c, hl.clamp(2 * i + p.x - 1, 0, W - 1), hl.clamp(2 * j + p.y - 1, 0, H - 1), n)])
        self.layers.append(dict(kind="pool", name=name, out=out, r=p, C=C, H=Ho, W=Wo))
        return out, (C, Ho, Wo)

    def gap_fc(self, x, shape, fw, fb):
        """평균 풀링 -> fc. 결과 logits[o, n] = numpy (N, 10)."""
        c, i, j, n = self.c, self.i, self.j, self.n
        C, H, W = shape
        g = hl.Func("gap")
        r = hl.RDom([(0, W), (0, H)], "gap_r")
        g[c, n] = hl.f32(0)
        g[c, n] += x[L(c, r.x, r.y, n)]
        self.layers.append(dict(kind="gap", name="gap", out=g, r=r, C=C, H=1, W=1))
        wb, bb = self.buf(fw), self.buf(fb)                      # fw numpy (10, C) = Halide (c, o)
        o = hl.Var("o")
        k = hl.RDom([(0, C)], "fc_r")
        logits = hl.Func("logits")
        logits[o, n] = bb[o]
        logits[o, n] += wb[k.x, o] * g[k.x, n] * (1.0 / (H * W))
        self.layers.append(dict(kind="fc", name="fc", out=logits, r=k, C=fw.shape[0], H=1, W=1, CI=C))
        self.out = logits
        return logits

MBV2_CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


def _build(params):
    """MobileNetV2 (CIFAR-10, hint.py 와 같다). 반환: Net (net.out = logits Func, net.layers = 층 목록)."""
    net = Net()
    x, shape = net.input(params["preprocess.mean"], params["preprocess.std"])
    w, b = _fold(params, "stem.conv.weight", "stem.bn")
    x, shape = net.conv(x, shape, w, b, 1, 1, act="relu6", name="stem")
    k, cin = 0, 32
    for t, c, n, s0 in MBV2_CFG:
        for rep in range(n):
            s = s0 if rep == 0 else 1
            pre = f"blocks.{k}."
            y, yshape = x, shape
            if t != 1:
                w, b = _fold(params, pre + "expand.conv.weight", pre + "expand.bn")
                y, yshape = net.conv(y, yshape, w, b, 1, 0, act="relu6", name=f"b{k}_expand")
            w, b = _fold(params, pre + "dw.conv.weight", pre + "dw.bn")
            y, yshape = net.dwconv(y, yshape, w, b, s, name=f"b{k}_dw")
            w, b = _fold(params, pre + "project.conv.weight", pre + "project.bn")
            res = x if (s == 1 and cin == c) else None
            x, shape = net.conv(y, yshape, w, b, 1, 0, res=res, name=f"b{k}_project")
            k, cin = k + 1, c
    w, b = _fold(params, "head.conv.weight", "head.bn")
    x, shape = net.conv(x, shape, w, b, 1, 0, act="relu6", name="head")
    net.gap_fc(x, shape, params["fc.weight"], params["fc.bias"])
    return net

def _schedule(net):
    """출발점: 모든 층을 compute_root (층마다 전체 결과를 만들고 다음 층이 읽는다). 벡터 · 타일 없음.
    패딩 Func 는 inline (읽을 때마다 select). 층의 모양 (layer["C"], ["H"], ["W"]) 을 보고 층마다 다르게 정해도 된다."""
    for layer in net.layers:
        layer["out"].compute_root()
        if "acc" in layer:
            layer["acc"].compute_root()

def prepare_nhwc(params, calib=None):
    """(시간 안 잼, 한도 60 초) BN 을 접어 모델 파이프라인을 만들고 JIT 컴파일한다. 반환이 그대로 infer 에 들어간다.
    calib (CIFAR-10 train 1,000 장 uint8) 은 출발점에서는 쓰지 않는다 — 양자화 calibration 에 쓸 수 있다."""
    net = _build(params)
    _schedule(net)
    run = net.out.compile_to_callable([net.img], hl.get_jit_target_from_environment())
    return dict(run=run, net=net)


def infer_nhwc(state, images):
    """uint8 이미지 묶음 -> logits (N, 10) float32. 이 시간을 잰다."""
    y = np.empty((images.shape[0], 10), np.float32)
    state["run"](hl.Buffer(images), hl.Buffer(y))
    return y

#################### END EDIT IT HERE  ####################

def main():
    import verify
    raise SystemExit(0 if verify.main(["06_nhwc"]) else 1)


if __name__ == "__main__":
    main()
