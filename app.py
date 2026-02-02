import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# --- フォント設定 ---
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# --- メイン処理 ---
st.set_page_config(page_title="SePE Simulation (EC-4A10c)", layout="wide")

# --- サイドバー ---
st.sidebar.header("患者・治療パラメータ設定")

# 1. 患者情報
st.sidebar.subheader("患者情報")

# 身長: デフォルトを0.0にし、未入力(0)の場合は簡易式を使うロジックへ
height = st.sidebar.number_input("身長 (cm) ※任意", value=0.0, step=0.1, help="入力なし(0.0)の場合は簡易式(70mL/kg)が適用されます。")
weight = st.sidebar.number_input("体重 (kg)", value=65.0, step=0.1)
hct = st.sidebar.number_input("血中ヘマトクリット値 (%)", value=30.0, step=0.1)
alb_initial = st.sidebar.number_input("血清アルブミン値 (g/dL)", value=3.5, step=0.1)

# 2. 治療目標
st.sidebar.subheader("治療目標")
target_removal = st.sidebar.slider("病因物質の除去目標 (%)", 30, 95, 60, step=5)
qp = st.sidebar.number_input("血漿流量 QP (mL/min)", value=30.0, step=5.0)

# 3. 膜特性
st.sidebar.subheader("膜特性 (Evacure EC-4A10c)")
st.sidebar.markdown("<small>※in vivoでの目詰まりや安全域を考慮して調整</small>", unsafe_allow_html=True)
sc_pathogen = st.sidebar.slider("病因物質SC", 0.0, 1.0, 0.90, 0.01)
sc_albumin = st.sidebar.slider("アルブミンSC", 0.0, 1.0, 0.65, 0.01)

# --- 計算ロジック ---

# A. 循環血液量 (BV) の計算分岐
if height > 0:
    # 小川の式 (m換算)
    h_m = height / 100.0
    bv_L = 0.16874 * h_m + 0.05986 * weight - 0.0305
    bv_calc = bv_L * 1000 # L -> mL
    bv_method = "小川の式 (日本人成人)"
else:
    # 簡易式
    bv_calc = weight * 70
    bv_method = "簡易式 (70mL/kg)"

# B. 循環血漿量 (EPV)
epv = bv_calc * (1 - hct / 100)

# C. 必要処理量 (Required PV)
if sc_pathogen > 0:
    required_pv = -np.log(1 - target_removal/100.0) * epv / sc_pathogen
else:
    required_pv = 0

# D. 治療時間・補充液
treatment_time_min = required_pv / qp if qp > 0 else 0
vol_per_set = 50 + 140 # 190mL (20%Alb 50mL + Physio 140mL)
num_sets = required_pv / vol_per_set
num_sets_ceil = np.ceil(num_sets)
actual_replacement_vol = num_sets_ceil * vol_per_set
supplied_albumin_g = num_sets_ceil * 10 # 1セットあたり10g

# E. アルブミン予測喪失量 (Washoutモデル)
total_alb_body_g = (epv / 100) * alb_initial
alb_remaining_ratio = np.exp(-required_pv * sc_albumin / epv)
predicted_alb_loss_g = total_alb_body_g * (1 - alb_remaining_ratio)

# F. 分析用データ
# 補充液のAlb濃度
repl_alb_conc = (10 / 190) * 100 # g/dL (約5.26%)
# 排液(濾液)の平均Alb濃度 (初期値ベースの概算)
filtrate_alb_conc = alb_initial * sc_albumin # g/dL

# --- 表示エリア ---
st.title("選択的血漿交換 (SePE) シミュレーション")

# メトリクス表示
col1, col2, col3, col4 = st.columns(4)

# 循環血漿量の表示に、使用した式と入力状況を明記
bv_label_color = "off" if height == 0 else "normal"
col1.metric("予測循環血漿量 (EPV)", f"{int(epv)} mL", f"{bv_method}")
if height == 0:
    col1.caption("※身長未入力のため簡易式を使用中")

col2.metric("治療時間", f"{int(treatment_time_min)} 分", f"QP: {qp} mL/min")
col3.metric(f"必要処理量 ({target_removal}%除去)", f"{int(required_pv)} mL", f"{required_pv/epv:.2f} × EPV")

# アルブミンバランスの強調表示
diff_alb = supplied_albumin_g - predicted_alb_loss_g
col4.metric("アルブミン収支予測", f"{int(diff_alb):+d} g", f"補充:{int(supplied_albumin_g)}g / 喪失:{int(predicted_alb_loss_g)}g")

st.divider()

# --- 画像と分析 ---
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
        st.info("※回路図画像がありません")

with c_info:
    # ユーザー指摘事項への対策（原因分析と対策）
    st.warning(f"⚠️ **分析: アルブミン補充過多の可能性**")
    st.markdown(f"""
    **現状の乖離:**
    * **予測喪失量:** 約 {int(predicted_alb_loss_g)} g
    * **必要補充量:** {int(supplied_albumin_g)} g
    * **収支バランス:** <span style="color:red">**+{int(diff_alb)} g のプラスバランス**</span>
    
    **原因:**
    SePEではアルブミンが部分的に体内に残るため、排液中のアルブミン濃度は低くなります。
    対して、現在の補充液レシピは濃度が高いため、等量置換すると過剰になります。
    * **排液中のAlb濃度:** 約 {filtrate_alb_conc:.1f} g/dL (体内 {alb_initial} × SC {sc_albumin})
    * **補充液のAlb濃度:** 約 {repl_alb_conc:.1f} g/dL (20%製剤 50mL + 生食等 140mL)
    
    **対策 (Clinical Action):**
    実際の治療では、以下の調整が検討されます。
    1.  **アルブミン製剤の間引き:** 全てのボトルに入れず、2回に1回にする等で総量を調整する。
    2.  **低濃度レシピへの変更:** SePE専用の低濃度アルブミン溶液を使用する。
    """, unsafe_allow_html=True)

st.divider()

# --- グラフ描画 ---
st.subheader(f"治療経過シミュレーション")

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

# 凡例配置
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11, frameon=False)
plt.tight_layout()
st.pyplot(fig)

# --- 詳細な用語解説・根拠 ---
st.divider()
st.header("用語解説・計算根拠")

with st.expander("1. 用語解説 (QP, SC, RC)", expanded=True):
    st.markdown(r"""
    * **QP (Plasma Flow Rate):** 血漿流量（mL/min）。
    * **ふるい係数 (SC):** 膜の透過性（0=阻止、1=通過）。SePEでは病因物質SC≒1.0、Alb SC≒0.6-0.7の膜を使用します。
    * **阻止率 (RC):** 膜による阻止性能 ($RC = 1 - SC$)。
    """)

with st.expander("2. Evacure EC-4A10c のSC設定と安全域", expanded=True):
    st.markdown("""
    **カタログ値と安全域:**
    カタログ値（In vitro牛血）に対し、臨床（In vivo）では二次膜形成によりSCが低下します。
    * **病因物質:** 除去不足を防ぐため、SCを**低め**に見積もり、必要処理量を確保します。
    * **アルブミン:** 喪失過多を防ぐため、SCを**高め**に見積もり、補充計画を立てます。
    """)

with st.expander("3. 循環血漿量・必要処理量の計算根拠", expanded=True):
    st.markdown(r"""
    **A. 予測循環血漿量 (EPV)**
    * **身長入力あり:** 小川の式 (Ogawa's Formula) を使用。
      $$ BV(L) = 0.16874 \times Height(m) + 0.05986 \times Weight(kg) - 0.0305 $$
    * **身長入力なし:** 簡易式 ($70mL/kg$) を使用。
    * 血漿量算出: $EPV = BV \times (1 - Hct/100)$

    **B. 必要な血漿処理量 (Required PV)**
    ワンコンパートメントモデル（Washout）に基づく計算：
    $$ V = \frac{- \ln(1 - R) \times EPV}{SC_{pathogen}} $$
    ($R$: 除去目標率 0.0~1.0, $V$: 処理量)
    """)
