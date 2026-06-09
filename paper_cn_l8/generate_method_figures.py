from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.pagesizes import landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#1f5aa6"
LIGHT = "#eef5ff"
GRAY = "#555555"
PALE = "#f7f9fc"
ORANGE = "#f7efe2"

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
FONT_CN = "STSong-Light"
FONT_PATH = "/System/Library/Fonts/Supplemental/Songti.ttc"


def pil_font(size):
    return ImageFont.truetype(FONT_PATH, size=size)


class PDF:
    def __init__(self, path, w, h):
        self.c = canvas.Canvas(str(path), pagesize=(w, h))
        self.w = w
        self.h = h

    def color(self, value):
        if value == "white":
            return white
        if value == "black":
            return black
        return HexColor(value) if isinstance(value, str) else value

    def save(self):
        self.c.showPage()
        self.c.save()

    def rect(self, x, y, w, h, fill=LIGHT, stroke=BLUE, radius=7, lw=1.2):
        c = self.c
        c.setLineWidth(lw)
        c.setStrokeColor(self.color(stroke))
        c.setFillColor(self.color(fill))
        c.roundRect(x, self.h - y - h, w, h, radius, stroke=1, fill=1)

    def circle(self, x, y, r, fill=white, stroke=BLUE, lw=1.3):
        c = self.c
        c.setLineWidth(lw)
        c.setStrokeColor(self.color(stroke))
        c.setFillColor(self.color(fill))
        c.circle(x, self.h - y, r, stroke=1, fill=1)

    def line(self, x1, y1, x2, y2, color="#222222", lw=0.7, arrow=False):
        c = self.c
        c.setStrokeColor(self.color(color))
        c.setFillColor(self.color(color))
        c.setLineWidth(lw)
        c.line(x1, self.h - y1, x2, self.h - y2)
        if arrow:
            draw_arrow_head_pdf(c, x1, self.h - y1, x2, self.h - y2, color)

    def text(self, x, y, s, size=9, color="#000000", align="center", font=FONT_CN):
        c = self.c
        c.setFillColor(self.color(color))
        c.setFont(font, size)
        lines = str(s).split("\n")
        line_h = size * 1.22
        start_y = y - (len(lines) - 1) * line_h / 2
        for i, line in enumerate(lines):
            yy = self.h - (start_y + i * line_h)
            if align == "center":
                c.drawCentredString(x, yy, line)
            elif align == "right":
                c.drawRightString(x, yy, line)
            else:
                c.drawString(x, yy, line)


class PNG:
    def __init__(self, path, w, h, scale=4):
        self.scale = scale
        self.w = int(w * scale)
        self.h = int(h * scale)
        self.im = Image.new("RGB", (self.w, self.h), "white")
        self.d = ImageDraw.Draw(self.im)
        self.path = path

    def save(self):
        self.im.save(self.path, dpi=(600, 600))

    def S(self, v):
        return int(round(v * self.scale))

    def rect(self, x, y, w, h, fill=LIGHT, stroke=BLUE, radius=7, lw=1.2):
        self.d.rounded_rectangle(
            [self.S(x), self.S(y), self.S(x + w), self.S(y + h)],
            radius=self.S(radius),
            fill=fill,
            outline=stroke,
            width=max(1, self.S(lw)),
        )

    def circle(self, x, y, r, fill="white", stroke=BLUE, lw=1.3):
        self.d.ellipse(
            [self.S(x - r), self.S(y - r), self.S(x + r), self.S(y + r)],
            fill=fill,
            outline=stroke,
            width=max(1, self.S(lw)),
        )

    def line(self, x1, y1, x2, y2, color="#222222", lw=0.7, arrow=False):
        self.d.line([self.S(x1), self.S(y1), self.S(x2), self.S(y2)], fill=color, width=max(1, self.S(lw)))
        if arrow:
            draw_arrow_head_png(self.d, self.S(x1), self.S(y1), self.S(x2), self.S(y2), color)

    def text(self, x, y, s, size=9, color="#000000", align="center", font=None):
        font_obj = pil_font(int(size * self.scale))
        lines = str(s).split("\n")
        line_h = int(size * self.scale * 1.22)
        total_h = line_h * len(lines)
        y0 = self.S(y) - total_h // 2
        for i, line in enumerate(lines):
            bbox = self.d.textbbox((0, 0), line, font=font_obj)
            tw = bbox[2] - bbox[0]
            if align == "center":
                xx = self.S(x) - tw // 2
            elif align == "right":
                xx = self.S(x) - tw
            else:
                xx = self.S(x)
            self.d.text((xx, y0 + i * line_h), line, font=font_obj, fill=color)


def draw_arrow_head_pdf(c, x1, y1, x2, y2, color):
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 5.5
    pts = []
    for a in (math.pi * 0.82, -math.pi * 0.82):
        pts.append((x2 + size * math.cos(ang + a), y2 + size * math.sin(ang + a)))
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(pts[0][0], pts[0][1])
    p.lineTo(pts[1][0], pts[1][1])
    p.close()
    c.setFillColor(HexColor(color))
    c.drawPath(p, stroke=0, fill=1)


def draw_arrow_head_png(d, x1, y1, x2, y2, color):
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 18
    pts = [(x2, y2)]
    for a in (math.pi * 0.82, -math.pi * 0.82):
        pts.append((x2 + size * math.cos(ang + a), y2 + size * math.sin(ang + a)))
    d.polygon(pts, fill=color)


def outputs(name, w, h):
    return [PDF(OUT / f"{name}.pdf", w, h), PNG(OUT / f"{name}.png", w, h)]


def save_all(draw_fn, name, w, h):
    for obj in outputs(name, w, h):
        draw_fn(obj)
        obj.save()


def box(obj, cx, cy, w, h, text, fill=LIGHT, size=8.5):
    obj.rect(cx - w / 2, cy - h / 2, w, h, fill=fill)
    obj.text(cx, cy, text, size=size)


def arrow(obj, x1, y1, x2, y2, color="#222222", lw=0.8):
    obj.line(x1, y1, x2, y2, color=color, lw=lw, arrow=True)


def draw_method_flow(obj):
    w, h = obj.w, obj.h
    xs = [42, 112, 188, 264, 188, 112, 42]
    ys = [42, 42, 42, 42, 92, 92, 92]
    labels = [
        "原始数据\n候选字段",
        "六类相关性\n宽松初筛",
        "L1门控DNN\n推断压缩",
        "活跃字段\n风险源定位",
        "DNN再训练\n能力验证",
        "风险解释\n字段处置",
        "相关性权重\n统一边界",
    ]
    fills = [PALE, LIGHT, LIGHT, LIGHT, PALE, PALE, ORANGE]
    for x, y, lab, fill in zip(xs, ys, labels, fills):
        box(obj, x, y, 55, 25, lab, fill=fill, size=8.5)
    arrow(obj, 70, 42, 84, 42)
    arrow(obj, 140, 42, 158, 42)
    arrow(obj, 216, 42, 236, 42)
    arrow(obj, 264, 56, 206, 84)
    arrow(obj, 160, 92, 140, 92)
    arrow(obj, 84, 92, 70, 92)
    arrow(obj, 112, 55, 57, 82, color="#777777", lw=0.65)
    arrow(obj, 70, 92, 236, 56, color="#777777", lw=0.65)
    obj.text(w / 2, 118, "统计依赖发现 → 推断能力验证 → 稀疏风险定位 → 相关性权重解释", size=8.2, color=GRAY)


def draw_corr_screening(obj):
    box(obj, 38, 68, 55, 30, "候选字段 Xi\n敏感目标 Y", fill=PALE, size=8.5)
    metric_pos = [(105, 25, "Pearson\n线性"), (105, 48, "Spearman\n秩相关"), (105, 71, "Kendall\n一致性"),
                  (105, 94, "NMI\n信息共享"), (105, 117, "DC/HSIC\n非线性")]
    for x, y, lab in metric_pos:
        box(obj, x, y, 52, 20, lab, fill=LIGHT, size=7.4)
        arrow(obj, 66, 68, x - 27, y, lw=0.55)
    box(obj, 180, 68, 58, 34, "六维向量 Ri\n归一化处理", fill=PALE, size=8.2)
    for x, y, _ in metric_pos:
        arrow(obj, x + 27, y, 151, 68, lw=0.55)
    box(obj, 252, 50, 56, 24, "阈值判断\nr_i^(j) ≥ μ_j", fill=LIGHT, size=7.8)
    box(obj, 252, 91, 56, 24, "任一通过\n即保留", fill=ORANGE, size=8.2)
    box(obj, 322, 68, 55, 30, "候选集合 C\n进入门控模型", fill=PALE, size=8.1)
    arrow(obj, 209, 68, 224, 56)
    arrow(obj, 252, 63, 252, 79)
    arrow(obj, 280, 91, 294, 76)
    obj.text(180, 146, "多指标互补覆盖：线性、单调、信息论与核空间依赖", size=8, color=GRAY)


def draw_dense(obj, left, right, color="#222222"):
    for x1, y1 in left:
        for x2, y2 in right:
            obj.line(x1, y1, x2, y2, color=color, lw=0.42)


def draw_l1_gate_structure(obj):
    ys = [48, 78, 108, 138]
    x_cols = [32, 80, 135, 195, 255, 315]
    labels = ["x1", "x2", "x3", "xm"]
    for y, lab in zip(ys, labels):
        obj.text(13, y, lab, size=10, align="left")
        arrow(obj, 31, y, x_cols[0] - 16, y, lw=0.7)
        obj.circle(x_cols[0], y, 13, fill="white")
    obj.text(19, 123, "⋮", size=16)
    obj.text(x_cols[0], 166, "输入层\nL_input", size=8.5)

    gate_pts = []
    for y, lab in zip(ys, ["g1", "g2", "g3", "gm"]):
        obj.circle(x_cols[1], y, 11, fill=ORANGE)
        obj.text(x_cols[1], y, lab, size=8.5)
        gate_pts.append((x_cols[1] + 11, y))
        arrow(obj, x_cols[0] + 13, y, x_cols[1] - 12, y, lw=0.65)
    obj.text(x_cols[1], 166, "输入门控\nL_gate", size=8.5)
    obj.text(x_cols[1], 151, "L1", size=8, color=GRAY)

    layers = []
    for xc, count, title in [(x_cols[2], 5, "隐藏层\n64"), (x_cols[3], 5, "隐藏层\n32"), (x_cols[4], 4, "隐藏层\n16")]:
        yy = [38 + i * (108 / (count - 1)) for i in range(count)]
        pts = []
        for y in yy:
            obj.circle(xc, y, 12, fill="white")
            pts.append((xc - 12, y, xc + 12, y))
        obj.text(xc, 166, title, size=8.5)
        if count == 5:
            obj.text(xc, 128, "⋮", size=14)
        layers.append(pts)

    out_y = [58, 93, 128]
    out_pts = []
    for y in out_y:
        obj.circle(x_cols[5], y, 13, fill="white")
        out_pts.append((x_cols[5] - 13, y, x_cols[5] + 13, y))
    obj.text(x_cols[5], 166, "输出层\nL_output", size=8.5)
    obj.text(337, 93, "ŷ", size=12, align="left")
    arrow(obj, x_cols[5] + 13, 93, 334, 93, lw=0.7)

    draw_dense(obj, [(x_cols[1] + 11, y) for y in ys], [(p[0], p[1]) for p in layers[0]], color="#222222")
    for a, b in zip(layers[:-1], layers[1:]):
        draw_dense(obj, [(p[2], p[1]) for p in a], [(p[0], p[1]) for p in b], color="#222222")
    draw_dense(obj, [(p[2], p[1]) for p in layers[-1]], [(p[0], p[1]) for p in out_pts], color="#222222")
    obj.text(176, 17, "输入门控加权：x̃ = g ⊙ X", size=9.2, color=GRAY)


def draw_corr_gate_structure(obj):
    box(obj, 36, 42, 50, 24, "字段 Xi\n目标 Y", fill=PALE, size=8.2)
    metric_labels = ["NMI", "Spearman", "Pearson", "Kendall", "DC", "HSIC"]
    for i, lab in enumerate(metric_labels):
        y = 18 + i * 17
        box(obj, 100, y, 43, 14, lab, fill=LIGHT, size=6.8)
        arrow(obj, 61, 42, 78, y, lw=0.45)
    box(obj, 165, 60, 55, 30, "相关性向量\nRi∈R6", fill=PALE, size=8.1)
    for i in range(6):
        y = 18 + i * 17
        arrow(obj, 122, y, 137, 60, lw=0.45)
    box(obj, 235, 42, 60, 26, "加权融合\nz=W^T Ri+b", fill=ORANGE, size=8.0)
    box(obj, 235, 90, 60, 24, "学习权重 W\n解释指标贡献", fill=PALE, size=7.8)
    box(obj, 310, 42, 48, 24, "Sigmoid\nσ(z)", fill=LIGHT, size=8.0)
    box(obj, 374, 42, 50, 24, "门控值 gi\n0~1", fill=PALE, size=8.0)
    box(obj, 374, 91, 58, 25, "0.5自然边界\ngi>0.5保留", fill=ORANGE, size=7.5)
    box(obj, 444, 42, 54, 24, "加权输入\ngi·Xi", fill=LIGHT, size=8.0)
    box(obj, 510, 42, 47, 24, "DNN\n推断 Y", fill=PALE, size=8.0)
    arrow(obj, 192, 60, 205, 48)
    arrow(obj, 235, 77, 235, 55, color="#777777", lw=0.6)
    arrow(obj, 265, 42, 286, 42)
    arrow(obj, 334, 42, 349, 42)
    arrow(obj, 374, 54, 374, 78, color="#777777", lw=0.6)
    arrow(obj, 399, 42, 417, 42)
    arrow(obj, 471, 42, 486, 42)
    obj.text(275, 120, "新字段快速评估：Rnew → σ(W^T Rnew+b)", size=8.4, color=GRAY)


if __name__ == "__main__":
    save_all(draw_method_flow, "method_flow", 310, 130)
    save_all(draw_corr_screening, "corr_screening_flow", 350, 160)
    save_all(draw_l1_gate_structure, "l1_gate_structure", 355, 180)
    save_all(draw_corr_gate_structure, "corr_gate_structure", 545, 130)
    print("generated method figures in", OUT)
