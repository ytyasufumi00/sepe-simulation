import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# --- フォント設定 ---
# packages.txt で導入した日本語フォントを使用
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# --- メイン処理 ---
st.set_page_config(page_title="SePE Simulation (EC-4A10c)", layout="wide")

# --- サイドバー ---
st.sidebar.header("患者・治療パラメータ設定")
st.sidebar.subheader("患者情報")

use_height_formula = st.sidebar.checkbox("身長を入力して計算（小川の式）", value=True)
if use_height_formula:
    height = st.sidebar.number_input("身長 (cm)", value=170.0, step=0.1)
else:
    height = None
    st.sidebar.caption("身長入力なし：簡易式 (70mL/kg)")

weight = st.sidebar.number_input("体重 (kg)", value=65.0, step=0.1)
hct = st.sidebar.number_input("血中ヘマトクリット値 (%)", value=30.0, step=0.1)
alb_initial = st.sidebar.number_input("血清アルブミン値 (g/dL)", value=3.5, step=0.1)

st.sidebar.subheader("治療目標")
target_removal = st.sidebar.slider("病因物質の除去目標 (%)", 30, 95, 60, step=5)
qp = st.sidebar.number_input("血漿流量 QP (mL/min)", value=30.0, step=5.0)

st.sidebar.subheader("膜特性 (Evacure EC-4A10c)")
st.sidebar.markdown("<small>※in vivoでの目詰まりや安全域を考慮して調整</small>", unsafe_allow_html=True)
sc_pathogen = st.sidebar.slider("病因物質SC", 0.0, 1.0, 0.90, 0.01)
sc_albumin = st.sidebar.slider("アルブミンSC", 0.0, 1.0, 0.65, 0.01)

# --- 計算ロジック ---
if use_height_formula and height is not None:
    # 小川の式 (m換算)
    h_m = height / 100.0
    bv_L = 0.16874 * h_m + 0.05986 * weight - 0.0305
    bv_calc = bv_L * 1000 # L -> mL
    bv_method = "小川の式 (日本人成人)"
else:
    bv_calc = weight * 70
    bv_method = "簡易式 (70mL/kg)"

epv = bv_calc * (1 - hct / 100)

if sc_pathogen > 0:
    # 目標除去率Rに対する必要処理量V: V = -ln(1-R) * EPV / SC
    required_pv = -np.log(1 - target_removal/100.0) * epv / sc_pathogen
else:
    required_pv = 0

treatment_time_min = required_pv / qp if qp > 0 else 0
vol_per_set = 50 + 140
num_sets = required_pv / vol_per_set
num_sets_ceil = np.ceil(num_sets)
actual_replacement_vol = num_sets_ceil * vol_per_set
supplied_albumin_g = num_sets_ceil * 10

total_alb_body_g = (epv / 100) * alb_initial
alb_remaining_ratio = np.exp(-required_pv * sc_albumin / epv)
predicted_alb_loss_g = total_alb_body_g * (1 - alb_remaining_ratio)

# --- 表示エリア ---
st.title("選択的血漿交換 (SePE) シミュレーション")

# メトリクス表示
col1, col2, col3, col4 = st.columns(4)
col1.metric("予測循環血漿量 (EPV)", f"{int(epv)} mL", f"BV: {int(bv_calc)} mL")
col2.metric("治療時間", f"{int(treatment_time_min)} 分", f"QP: {qp} mL/min")
col3.metric(f"必要処理量 ({target_removal}%除去)", f"{int(required_pv)} mL", f"倍率: {required_pv/epv:.2f} × EPV")
col4.metric("予測喪失アルブミン", f"{predicted_alb_loss_g:.1f} g", f"補充: {int(supplied_albumin_g)} g")

st.divider()

# 画像と設定概要
c_img, c_info = st.columns([1, 1])
with c_img:
    img_files = ["circuit.png", "circuit.jpg", "circuit.jpeg"]
    found_img = None
    for f in img_files:
        if os.path.exists(f):
            found_img = f
            break
    
    if found_img:
        try:
            img = Image.open(found_img)
            st.image(img, caption="SePE 回路構成図", use_container_width=True)
        except:
            st.error("画像読み込みエラー")
    else:
        st.info("※回路図画像 (circuit.png/jpg) が未アップロードです")

with c_info:
    st.info(f"""
    **💉 補充液構成 (1セットあたり):**
    * **20% アルブミン:** 50mL (Alb 10g)
    * **フィジオ140:** 140mL
    * **合計:** 190mL (Alb濃度 約5.3%)
    
    **📊 補充計画:**
    * **必要セット数:** {int(num_sets_ceil)} セット
    * **総補充液量:** {int(actual_replacement_vol)} mL
    * **総補充アルブミン:** {int(supplied_albumin_g)} g
    """)

st.divider()

# --- グラフ描画 ---
st.subheader(f"治療経過シミュレーション (目標: 病因物質 {target_removal}% 除去)")

v_process = np.linspace(0, required_pv * 1.2, 100)
pathogen_remaining = np.exp(-v_process * sc_pathogen / epv) * 100
alb_loss_curve = total_alb_body_g * (1 - np.exp(-v_process * sc_albumin / epv))

fig, ax1 = plt.subplots(figsize=(10, 6))

color_1 = 'tab:red'
ax1.set_xlabel('血漿処理量 (mL)', fontsize=12)
ax1.set_ylabel('【赤】病因物質 残存率 (%)', color=color_1, fontweight='bold', fontsize=12)
line1 = ax1.plot(v_process, pathogen_remaining, color=color_1, linewidth=3, label='病因物質 残存率 (%)')
ax1.tick_params(axis='y', labelcolor=color_1)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_ylim(0, 105)

# 目標点プロット
ax1.scatter([required_pv], [100 - target_removal], color='red', s=100, zorder=5)

# 吹き出し表示
ax1.annotate(f'目標達成\n{int(required_pv)}mL処理\n(残存{100-target_removal}%)',
             xy=(required_pv, 100 - target_removal), 
             xytext=(0, 60), textcoords='offset points',
             ha='center', va='bottom',
             color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', linewidth=1.5),
             bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5', alpha=0.9))

ax2 = ax1.twinx()
color_2 = 'tab:blue'
ax2.set_ylabel('【青】累積アルブミン喪失量 (g)', color=color_2, fontweight='bold', fontsize=12)
line2 = ax2.plot(v_process, alb_loss_curve, color=color_2, linestyle='--', linewidth=2.5, label='予測アルブミン喪失量 (g)')
ax2.tick_params(axis='y', labelcolor=color_2)
ax2.set_ylim(0, max(alb_loss_curve)*1.2)

# 凡例配置（グラフ下）
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11, frameon=False)
plt.tight_layout()
st.pyplot(fig)

# --- 詳細な用語解説・根拠 ---
st.divider()
st.header("用語解説・計算根拠")

# タブではなくExpanderで詳細記述（省略なし）
with st.expander("1. 用語解説 (QP, SC, 阻止率)", expanded=True):
    st.markdown(r"""
    * **QP (Plasma Flow Rate):**
        * 血漿流量（mL/min）。血漿分離器（EC-4A10c）へ供給される血漿のスピードです。
        * QPが高いほど時間あたりの処理量は増えますが、膜面圧（TMP）の上昇や目詰まり（Fouling）のリスクも高まります。通常30～40mL/min程度で設定されます。
    
    * **ふるい係数 (Sieving Coefficient, SC):**
        * 膜における物質の「通りやすさ」を示す指標です（0.0～1.0）。
        * $SC = \frac{C_{Filtrate}（濾液中の濃度）}{C_{Plasma}（血漿中の濃度）}$
        * 1.0に近いほど素通りし、0に近いほど阻止されます。SePEでは「病因物質は1.0に近く、アルブミンは0.6～0.7程度」の膜を使用します。
        
    * **阻止率 (Rejection Coefficient, RC):**
        * 膜が物質を「どれだけ通さないか」を示す指標です。
        * $RC = 1 - SC$
        * 例: SCが0.7の場合、阻止率は0.3（30%は膜で阻止されて体内に戻る）となります。
    """)

with st.expander("2. Evacure EC-4A10c におけるSC設定の根拠と考え方", expanded=True):
    st.markdown("""
    **カタログ値と臨床値の乖離（Safety Margin）について**
    
    * **In vitro（カタログ値）:** 牛血を用いた実験データであり、理想的な条件下での数値です（例: アルブミンSC=0.7付近、IgG SC≒1.0）。
    * **In vivo（臨床）:** 実際の治療では、時間経過とともに膜内面にタンパク質が付着し「二次膜（Secondary membrane）」が形成されます。これにより、実効孔径が狭くなり、SCはカタログ値よりも低下する傾向があります。
    
    **安全域を考慮したシミュレーション設定:**
    本システムでは、安全性を最優先した計画を立てるため、以下の考え方を推奨しています。
    
    1.  **病因物質のSC（除去効果） → 低めに見積もる**
        * *理由:* 「思ったより抜けなかった」という治療不全を防ぐため。
        * SCを低く設定して計算することで、目標達成に必要な処理量（V）を**多め**に算出し、確実な除去を目指します。
        
    2.  **アルブミンのSC（喪失量） → 高めに見積もる**
        * *理由:* 「予想以上にアルブミンが抜けて低アルブミン血症になった」という事故を防ぐため。
        * SCを高く設定して計算することで、アルブミン喪失量を**最大リスク**で予測し、十分な補充液を用意できるようにします。
    """)

with st.expander("3. 予測循環血漿量・必要処理量の計算方法と根拠", expanded=True):
    st.markdown(r"""
    **A. 予測循環血漿量 (Estimated Plasma Volume, EPV)**
    
    本システムでは、日本人の体格に適合した**小川の式 (Ogawa's Formula)** を採用しています。
    
    1.  **循環血液量 (BV) の算出:**
        $$ BV (L) = 0.16874 \times Height (m) + 0.05986 \times Weight (kg) - 0.0305 $$
        *(※身長入力がない場合は、簡易的に $70mL/kg$ として計算します)*
        
    2.  **循環血漿量 (EPV) の算出:**
        ヘマトクリット (Hct) 分を除いた容積が血漿量となります。
        $$ EPV (mL) = BV (mL) \times (1 - \frac{Hct (\%)}{100}) $$

    ---

    **B. 必要な血漿処理量 (Required PV) の計算**
    
    SePEは「ワンコンパートメントモデル（単一プールモデル）」に従って物質が除去されると仮定して計算します。
    
    * **基本式:**
        処理量 $V$、ふるい係数 $SC$、循環血漿量 $EPV$ とすると、治療後の濃度 $C$ は以下の指数関数で減衰します。
        $$ \frac{C}{C_0} = e^{-\frac{V \times SC}{EPV}} $$
        
    * **処理量の逆算:**
        目標とする除去率を $R$（例: 60%除去なら $R=0.6$）とすると、残存率は $1-R$ となります。
        $$ 1 - R = e^{-\frac{V \times SC}{EPV}} $$
        
        これを $V$ について解くと、本システムで使用している計算式になります：
        $$ V = \frac{- \ln(1 - R) \times EPV}{SC} $$
    """)
