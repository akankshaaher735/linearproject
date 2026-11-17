"""
House Price Predictor - Flask app serving a scikit-learn LinearRegression model.

Run locally :  python app.py
Run on Render: gunicorn app:app
"""
import os
import pickle
import warnings

import numpy as np
from flask import Flask, Response, jsonify, request

# The model was trained on a DataFrame, but we predict from a plain array.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Feature order must match the order used during training.
FEATURES = list(getattr(model, "feature_names_in_", [
    "Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built",
    "Lot_Size", "Garage_Size", "Neighborhood_Quality",
]))

# Allowed input ranges (used for validation and to configure the sliders).
LIMITS = {
    "Square_Footage":       (300, 10000),
    "Num_Bedrooms":         (1, 10),
    "Num_Bathrooms":        (1, 8),
    "Year_Built":           (1800, 2030),
    "Lot_Size":             (0.1, 20),
    "Garage_Size":          (0, 6),
    "Neighborhood_Quality": (1, 10),
}

# A "typical home" used as the reference point for the impact bars.
BASELINE = {
    "Square_Footage": 2000, "Num_Bedrooms": 3, "Num_Bathrooms": 2,
    "Year_Built": 2000, "Lot_Size": 2.0, "Garage_Size": 1,
    "Neighborhood_Quality": 5,
}

app = Flask(__name__)


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    values = []
    for name in FEATURES:
        raw = data.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return jsonify(error=f"'{name}' must be a number."), 400
        lo, hi = LIMITS.get(name, (-np.inf, np.inf))
        if not (lo <= value <= hi):
            return jsonify(error=f"'{name}' must be between {lo} and {hi}."), 400
        values.append(value)

    x = np.array([values], dtype=float)
    price = float(model.predict(x)[0])

    # Price impact of each feature relative to the typical home above.
    contributions = {
        name: float(coef * (val - BASELINE.get(name, val)))
        for name, coef, val in zip(FEATURES, np.ravel(model.coef_), values)
    }
    return jsonify(
        price=round(price, 2),
        intercept=float(np.ravel(model.intercept_)[0]),
        contributions=contributions,
    )


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b1026">
<title>Aurora Estate · House Price Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>
  :root{
    --text:#eef1ff; --muted:rgba(226,230,255,.68);
    --glass:rgba(255,255,255,.08); --glass-strong:rgba(255,255,255,.14);
    --border:rgba(255,255,255,.18);
    --c1:#7c5cff; --c2:#22d3ee; --c3:#f472b6; --c4:#34d399;
    --radius:26px;
  }
  *{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
  html{scroll-behavior:smooth}
  body{
    font-family:'Outfit',system-ui,-apple-system,'Segoe UI',sans-serif;
    color:var(--text);min-height:100vh;min-height:100dvh;overflow-x:hidden;
    background:linear-gradient(-45deg,#0a0f2c,#2b1760,#0b4b6f,#4a1a6b,#07334d,#1d1b5e);
    background-size:500% 500%;
    animation:bgShift 32s ease-in-out infinite;
  }
  @keyframes bgShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}

  /* ---------- Orbs & particles ---------- */
  .scene{position:fixed;inset:0;overflow:hidden;pointer-events:none;z-index:0}
  .orb{
    position:absolute;border-radius:50%;filter:blur(2px);opacity:.75;
    animation:float var(--d,18s) ease-in-out infinite;animation-delay:var(--delay,0s);
    will-change:transform;
  }
  .orb::after{content:"";position:absolute;inset:-35%;border-radius:50%;background:inherit;filter:blur(50px);opacity:.55;z-index:-1}
  .o1{width:min(34vw,360px);aspect-ratio:1;left:-6%;top:6%;--d:22s;background:radial-gradient(circle at 30% 30%,#a78bfa,#7c3aed 60%,transparent 72%)}
  .o2{width:min(26vw,280px);aspect-ratio:1;right:4%;top:12%;--d:17s;--delay:-6s;background:radial-gradient(circle at 30% 30%,#67e8f9,#0891b2 60%,transparent 72%)}
  .o3{width:min(30vw,320px);aspect-ratio:1;left:38%;bottom:-8%;--d:26s;--delay:-11s;background:radial-gradient(circle at 30% 30%,#f9a8d4,#db2777 60%,transparent 72%)}
  .o4{width:min(16vw,170px);aspect-ratio:1;right:22%;bottom:22%;--d:14s;--delay:-3s;background:radial-gradient(circle at 30% 30%,#6ee7b7,#059669 60%,transparent 72%)}
  .o5{width:min(12vw,120px);aspect-ratio:1;left:14%;bottom:20%;--d:19s;--delay:-8s;background:radial-gradient(circle at 30% 30%,#fde68a,#f59e0b 60%,transparent 72%)}
  @keyframes float{
    0%,100%{transform:translate3d(0,0,0) scale(1)}
    25%{transform:translate3d(40px,-50px,0) scale(1.08)}
    50%{transform:translate3d(-30px,-90px,0) scale(.94)}
    75%{transform:translate3d(-60px,-20px,0) scale(1.05)}
  }
  #particles{position:absolute;inset:0;width:100%;height:100%}

  /* ---------- Layout ---------- */
  .wrap{
    position:relative;z-index:1;max-width:1180px;margin:0 auto;
    padding:clamp(18px,4vw,48px) clamp(14px,3vw,32px) 48px;
    padding-bottom:max(48px,env(safe-area-inset-bottom));
  }
  header{text-align:center;margin-bottom:clamp(22px,4vw,44px);animation:rise .9s cubic-bezier(.2,.8,.2,1) both}
  .badge{
    display:inline-flex;align-items:center;gap:8px;padding:7px 16px;border-radius:999px;
    background:var(--glass);border:1px solid var(--border);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
    font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:18px
  }
  .badge i{width:8px;height:8px;border-radius:50%;background:var(--c4);box-shadow:0 0 12px var(--c4);animation:pulse 2s infinite}
  @keyframes pulse{50%{transform:scale(1.7);opacity:.5}}
  h1{
    font-family:'Space Grotesk','Outfit',sans-serif;font-weight:700;
    font-size:clamp(2rem,6vw,4rem);line-height:1.05;letter-spacing:-.02em;
    background:linear-gradient(100deg,#fff 10%,#c4b5fd 35%,#67e8f9 60%,#f9a8d4 90%);
    background-size:250% auto;-webkit-background-clip:text;background-clip:text;color:transparent;
    animation:shine 8s linear infinite;
  }
  @keyframes shine{to{background-position:250% center}}
  .sub{margin:14px auto 0;max-width:560px;color:var(--muted);font-weight:300;font-size:clamp(.95rem,2.2vw,1.12rem);line-height:1.6}

  .grid{display:grid;grid-template-columns:1.25fr .9fr;gap:clamp(16px,2.5vw,28px);align-items:start}

  /* ---------- Glass cards ---------- */
  .card{
    position:relative;border-radius:var(--radius);padding:clamp(18px,3vw,32px);
    background:linear-gradient(145deg,rgba(255,255,255,.14),rgba(255,255,255,.05));
    border:1px solid var(--border);
    backdrop-filter:blur(22px) saturate(160%);-webkit-backdrop-filter:blur(22px) saturate(160%);
    box-shadow:0 20px 60px rgba(4,6,30,.45),inset 0 1px 0 rgba(255,255,255,.28);
    animation:rise .9s cubic-bezier(.2,.8,.2,1) both;overflow:hidden;
  }
  .card::before{
    content:"";position:absolute;inset:0;border-radius:inherit;padding:1px;
    background:linear-gradient(135deg,rgba(255,255,255,.5),transparent 35%,transparent 65%,rgba(124,92,255,.5));
    -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;
    pointer-events:none;opacity:.7
  }
  .card:nth-child(2){animation-delay:.15s}
  @keyframes rise{from{opacity:0;transform:translateY(28px) scale(.98)}to{opacity:1;transform:none}}
  .card h2{font-family:'Space Grotesk',sans-serif;font-size:1.15rem;font-weight:500;margin-bottom:22px;display:flex;align-items:center;gap:10px}
  .card h2 span{display:grid;place-items:center;width:34px;height:34px;border-radius:12px;background:linear-gradient(135deg,var(--c1),var(--c2));box-shadow:0 6px 18px rgba(124,92,255,.5)}

  /* ---------- Form ---------- */
  .fields{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .field{
    padding:14px 16px 16px;border-radius:18px;background:var(--glass);border:1px solid rgba(255,255,255,.1);
    transition:transform .35s ease,border-color .35s ease,background .35s ease,box-shadow .35s ease;
  }
  .field:hover,.field:focus-within{transform:translateY(-3px);border-color:rgba(167,139,250,.6);background:var(--glass-strong);box-shadow:0 10px 30px rgba(124,92,255,.25)}
  .field.wide{grid-column:1/-1}
  .top{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;gap:8px}
  .top label{font-size:.88rem;color:var(--muted);display:flex;align-items:center;gap:8px}
  .top label em{font-style:normal;font-size:1.1rem}
  .val{
    font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1rem;min-width:64px;text-align:right;
    padding:4px 10px;border-radius:10px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.12)
  }
  input[type=range]{
    -webkit-appearance:none;appearance:none;width:100%;height:8px;border-radius:99px;outline:none;cursor:pointer;
    background:linear-gradient(90deg,var(--c1),var(--c2)) 0/var(--pct,50%) 100% no-repeat,rgba(255,255,255,.14);
  }
  input[type=range]::-webkit-slider-thumb{
    -webkit-appearance:none;width:24px;height:24px;border-radius:50%;border:3px solid #fff;
    background:linear-gradient(135deg,var(--c3),var(--c1));box-shadow:0 0 0 4px rgba(124,92,255,.25),0 0 18px rgba(244,114,182,.8);
    transition:transform .2s ease
  }
  input[type=range]::-moz-range-thumb{width:20px;height:20px;border-radius:50%;border:3px solid #fff;background:linear-gradient(135deg,var(--c3),var(--c1));box-shadow:0 0 18px rgba(244,114,182,.8)}
  input[type=range]:active::-webkit-slider-thumb{transform:scale(1.25)}
  input[type=range]:focus-visible::-webkit-slider-thumb{box-shadow:0 0 0 5px rgba(34,211,238,.55)}

  .btn{
    margin-top:22px;width:100%;border:0;cursor:pointer;color:#fff;font:600 1.05rem 'Outfit',sans-serif;letter-spacing:.04em;
    padding:17px 22px;border-radius:18px;position:relative;overflow:hidden;
    background:linear-gradient(110deg,#7c5cff,#22d3ee,#f472b6,#7c5cff);background-size:300% 100%;
    animation:btnflow 6s linear infinite;box-shadow:0 12px 34px rgba(124,92,255,.5);
    transition:transform .25s ease,box-shadow .25s ease
  }
  .btn:hover{transform:translateY(-2px);box-shadow:0 18px 44px rgba(34,211,238,.45)}
  .btn:active{transform:scale(.98)}
  .btn::after{content:"";position:absolute;top:0;left:-60%;width:40%;height:100%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.5),transparent);transform:skewX(-20deg);animation:sweep 3.6s ease-in-out infinite}
  @keyframes sweep{0%{left:-60%}55%,100%{left:130%}}
  @keyframes btnflow{to{background-position:300% 0}}
  .btn[disabled]{opacity:.7;cursor:progress}

  /* ---------- Result ---------- */
  .result{position:sticky;top:20px;text-align:center}
  .ring{
    position:relative;width:min(100%,300px);aspect-ratio:1;margin:4px auto 22px;border-radius:50%;display:grid;place-items:center;
    background:conic-gradient(from 0deg,var(--c1),var(--c2),var(--c4),var(--c3),var(--c1));
    animation:spin 14s linear infinite;box-shadow:0 0 60px rgba(124,92,255,.5),0 0 120px rgba(34,211,238,.25)
  }
  @keyframes spin{to{transform:rotate(360deg)}}
  .ring-inner{
    width:calc(100% - 16px);height:calc(100% - 16px);border-radius:50%;display:grid;place-content:center;gap:4px;padding:14px;
    background:radial-gradient(circle at 30% 25%,rgba(40,34,110,.96),rgba(10,14,40,.97));
    animation:spin 14s linear infinite reverse
  }
  .ring-inner small{font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted)}
  .price{
    font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(1.7rem,6vw,2.6rem);letter-spacing:-.02em;
    background:linear-gradient(90deg,#fff,#a5f3fc);-webkit-background-clip:text;background-clip:text;color:transparent;
    word-break:break-word
  }
  .price.pop{animation:pop .55s cubic-bezier(.3,1.6,.5,1)}
  @keyframes pop{0%{transform:scale(.85)}100%{transform:scale(1)}}
  .note{font-size:.82rem;color:var(--muted);font-weight:300;line-height:1.5}
  .err{color:#fda4af;font-size:.9rem;margin-top:10px;min-height:1.2em}

  .bars{margin-top:22px;text-align:left;display:grid;gap:11px}
  .bars h3{font-family:'Space Grotesk',sans-serif;font-size:.85rem;font-weight:500;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:2px;text-align:center}
  .bar-row{display:grid;grid-template-columns:112px 1fr 76px;align-items:center;gap:10px;font-size:.78rem;color:var(--muted)}
  .track{height:8px;border-radius:99px;background:rgba(255,255,255,.1);overflow:hidden}
  .fill{height:100%;width:0;border-radius:inherit;background:linear-gradient(90deg,var(--c2),var(--c1));transition:width .9s cubic-bezier(.2,.8,.2,1);box-shadow:0 0 12px rgba(34,211,238,.6)}
  .fill.neg{background:linear-gradient(90deg,var(--c3),#fb7185);box-shadow:0 0 12px rgba(244,114,182,.6)}
  .bar-row b{font-weight:500;color:var(--text);text-align:right;font-variant-numeric:tabular-nums}

  footer{margin-top:34px;text-align:center;color:var(--muted);font-size:.8rem;font-weight:300}

  /* ---------- Responsive ---------- */
  @media (max-width:980px){            /* tablet */
    .grid{grid-template-columns:1fr}
    .result{position:relative;top:0}
    .ring{width:min(70%,280px)}
  }
  @media (max-width:600px){            /* mobile */
    .fields{grid-template-columns:1fr}
    .bar-row{grid-template-columns:96px 1fr 68px;font-size:.74rem}
    .card{border-radius:22px}
    .orb{opacity:.55}
    .o1,.o3{width:60vw}.o2{width:44vw}.o4,.o5{width:24vw}
  }
  @media (prefers-reduced-motion:reduce){
    *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}
  }
</style>
</head>
<body>
  <div class="scene" aria-hidden="true">
    <div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div>
    <div class="orb o4"></div><div class="orb o5"></div>
    <canvas id="particles"></canvas>
  </div>

  <div class="wrap">
    <header>
      <div class="badge"><i></i> Linear Regression Model · Live</div>
      <h1>Aurora Estate<br>Price Predictor</h1>
      <p class="sub">Shape your dream home with the sliders and watch the estimated market value update instantly.</p>
    </header>

    <main class="grid">
      <section class="card" aria-label="Property details">
        <h2><span>🏡</span> Property details</h2>
        <form id="form" autocomplete="off">
          <div class="fields" id="fields"></div>
          <button class="btn" id="btn" type="submit">Predict Price ✨</button>
          <p class="err" id="err" role="alert"></p>
        </form>
      </section>

      <aside class="card result" aria-live="polite">
        <h2><span>💎</span> Estimated value</h2>
        <div class="ring">
          <div class="ring-inner">
            <small>Predicted price</small>
            <div class="price" id="price">$0</div>
          </div>
        </div>
        <p class="note">Estimate from a linear regression model. Actual market prices vary.</p>
        <div class="bars" id="bars">
          <h3>Impact vs. a typical home</h3>
        </div>
      </aside>
    </main>
    <footer>Built with Flask · scikit-learn · Deployed on Render</footer>
  </div>

<script>
const FIELDS = [
  {k:"Square_Footage",      label:"Square footage",       icon:"📐", min:300,  max:10000, step:50,  val:2000, fmt:v=>Number(v).toLocaleString()+" ft²", wide:true},
  {k:"Num_Bedrooms",        label:"Bedrooms",             icon:"🛏️", min:1,    max:10,    step:1,   val:3,    fmt:v=>v},
  {k:"Num_Bathrooms",       label:"Bathrooms",            icon:"🛁", min:1,    max:8,     step:1,   val:2,    fmt:v=>v},
  {k:"Year_Built",          label:"Year built",           icon:"🏗️", min:1800, max:2030,  step:1,   val:2000, fmt:v=>v},
  {k:"Lot_Size",            label:"Lot size",             icon:"🌳", min:0.1,  max:20,    step:0.1, val:2.0,  fmt:v=>Number(v).toFixed(1)},
  {k:"Garage_Size",         label:"Garage size (cars)",   icon:"🚗", min:0,    max:6,     step:1,   val:1,    fmt:v=>v},
  {k:"Neighborhood_Quality",label:"Neighborhood quality", icon:"⭐", min:1,    max:10,    step:1,   val:5,    fmt:v=>v+" / 10"}
];
const LABELS = Object.fromEntries(FIELDS.map(f=>[f.k,f.label]));
const $ = id => document.getElementById(id);
const money = n => new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",maximumFractionDigits:0}).format(n);

/* Build sliders */
const box = $("fields");
FIELDS.forEach(f=>{
  const d = document.createElement("div");
  d.className = "field" + (f.wide ? " wide" : "");
  d.innerHTML =
    `<div class="top"><label for="${f.k}"><em>${f.icon}</em>${f.label}</label><span class="val" id="v_${f.k}"></span></div>
     <input type="range" id="${f.k}" name="${f.k}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.val}">`;
  box.appendChild(d);
  const input = d.querySelector("input"), out = d.querySelector(".val");
  const paint = () => {
    out.textContent = f.fmt(input.value);
    input.style.setProperty("--pct", ((input.value - f.min) / (f.max - f.min) * 100) + "%");
  };
  input.addEventListener("input", () => { paint(); live(); });
  paint();
});

/* Predict */
let shown = 0, timer = null, ctrl = null;
function payload(){ const o = {}; FIELDS.forEach(f => o[f.k] = $(f.k).value); return o; }

function countTo(target){
  const start = shown, t0 = performance.now(), dur = 900;
  const el = $("price");
  el.classList.remove("pop"); void el.offsetWidth; el.classList.add("pop");
  (function tick(now){
    const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
    shown = start + (target - start) * e;
    el.textContent = money(shown);
    if (p < 1) requestAnimationFrame(tick);
  })(t0);
}

function drawBars(c){
  const rows = Object.entries(c);
  const max = Math.max(...rows.map(([,v]) => Math.abs(v)), 1);
  const wrap = $("bars");
  wrap.querySelectorAll(".bar-row").forEach(n => n.remove());
  rows.sort((a,b) => Math.abs(b[1]) - Math.abs(a[1])).forEach(([k,v]) => {
    const r = document.createElement("div");
    r.className = "bar-row";
    r.innerHTML = `<span>${LABELS[k] || k}</span><div class="track"><div class="fill${v < 0 ? " neg" : ""}"></div></div><b>${(v >= 0 ? "+" : "−") + money(Math.abs(v))}</b>`;
    wrap.appendChild(r);
    requestAnimationFrame(() => requestAnimationFrame(() => r.querySelector(".fill").style.width = (Math.abs(v)/max*100) + "%"));
  });
}

async function predict(){
  $("err").textContent = "";
  if (ctrl) ctrl.abort();
  ctrl = new AbortController();
  try{
    const res = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload()), signal:ctrl.signal});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    countTo(Math.max(0, data.price));
    drawBars(data.contributions);
  }catch(e){
    if (e.name !== "AbortError") $("err").textContent = e.message;
  }
}
function live(){ clearTimeout(timer); timer = setTimeout(predict, 220); }

$("form").addEventListener("submit", async e => {
  e.preventDefault();
  const b = $("btn"); b.disabled = true; b.textContent = "Calculating…";
  await predict();
  b.disabled = false; b.textContent = "Predict Price ✨";
});
predict();

/* Subtle particles */
(function(){
  const cv = $("particles"), ctx = cv.getContext("2d");
  let W, H, dpr, parts = [], raf;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  function size(){
    dpr = Math.min(devicePixelRatio || 1, 2);
    W = cv.width = innerWidth * dpr; H = cv.height = innerHeight * dpr;
    const n = Math.round(Math.min(90, Math.max(28, innerWidth / 16)));
    parts = Array.from({length:n}, () => ({
      x:Math.random()*W, y:Math.random()*H,
      r:(Math.random()*1.8+.6)*dpr,
      vx:(Math.random()-.5)*.18*dpr, vy:-(Math.random()*.35+.08)*dpr,
      a:Math.random()*.5+.2, tw:Math.random()*Math.PI*2,
      hue:[190,260,320,160][Math.floor(Math.random()*4)]
    }));
  }
  function frame(){
    ctx.clearRect(0,0,W,H);
    for (const p of parts){
      p.x += p.vx; p.y += p.vy; p.tw += .02;
      if (p.y < -10){ p.y = H + 10; p.x = Math.random()*W; }
      if (p.x < -10) p.x = W + 10; if (p.x > W + 10) p.x = -10;
      const a = p.a * (.6 + .4*Math.sin(p.tw));
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.2832);
      ctx.fillStyle = `hsla(${p.hue},95%,80%,${a})`;
      ctx.shadowColor = `hsla(${p.hue},95%,70%,.9)`; ctx.shadowBlur = 10*dpr;
      ctx.fill();
    }
    raf = requestAnimationFrame(frame);
  }
  size(); addEventListener("resize", size);
  document.addEventListener("visibilitychange", () => { cancelAnimationFrame(raf); if (!document.hidden && !reduce) frame(); });
  if (!reduce) frame();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
