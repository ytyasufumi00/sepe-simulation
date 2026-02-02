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
height = st.sidebar.number_input("身長 (cm) ※任意", value=0.0, step=0.1, help="入力なし(0.0)の場合は簡易式(70mL/kg)が適用されます。")
weight = st.sidebar.number_input("体重 (kg)", value=65.0, step=0.1)
hct = st.sidebar.number_input("血中ヘマトクリット値 (%)", value=30.0, step=0.1)
alb_initial = st.sidebar.number_input("血清アルブミン値 (g/dL)", value=4.0, step=0.1)

# 2. 治療目標
st.sidebar.subheader("治療目標")
target_removal = st.sidebar.slider("病因物質の除去目標 (%)", 30, 95, 60, step=5)
qp = st.sidebar.number_input("血漿流量 QP (mL/min)", value=30.0, step=5.0)

# 3. アルブミンバランス調整
st.sidebar.subheader("アルブミン収支設定")
target_balance_ratio = st.sidebar.slider("収支目標 (対喪失量 %)", -10, 20, 0, step=1, help="基準予測喪失量に対して、何％上乗せして補充するか設定します。")

# 4. 膜特性
st.sidebar.subheader("膜特性 (Evacure EC-4A10c)")
st.sidebar.markdown("<small>※in vivoでの目詰まりや安全域を考慮して調整</small>", unsafe_allow_html=True)
sc_pathogen = st.sidebar.slider("病因物質SC", 0.0, 1.0, 0.90, 0.01)
sc_albumin = st.sidebar.slider("アルブミンSC", 0.0, 1.0, 0.50, 0.01)

# --- 計算ロジック ---

# A. 循環血液量 (BV)
if height > 0:
    h_m = height / 100.0
    bv_L = 0.16874 * h_m + 0.05986 * weight - 0.0305
    bv_calc = bv_L * 1000
    bv_method = "小川の式 (日本人成人)"
else:
    bv_calc = weight * 70
    bv_method = "簡易式 (70mL/kg)"

epv = bv_calc * (1 - hct / 100)

# B. 必要処理量
if sc_pathogen > 0:
    required_pv = -np.log(1 - target_removal/100.0) * epv / sc_pathogen
else:
    required_pv = 0

# C. 治療時間
treatment_time_min = required_pv / qp if qp > 0 else 0

# --- 💡 グラム数優先・液量調整ロジック ---

# 1. 目標の設定
# 基準喪失量 (自然減衰モデル)
total_alb_body_g = (epv / 100) * alb_initial
alb_remaining_ratio_base = np.exp(-required_pv * sc_albumin / epv)
base_loss_g = total_alb_body_g * (1 - alb_remaining_ratio_base)

# 目標補充量 (g)
target_supply_g = base_loss_g * (1 + target_balance_ratio / 100.0)

# 2. 探索用部品の定義 (看護師が調整しやすい量限定)
# フィジオ量は 50mL 刻み (300mL ~ 500mL)
physio_options = [500, 450, 400, 350, 300]

# セットの種類 (Alb 1本 or 2本)
bottle_options = [1, 2]

best_plan = None
best_score = float('inf')

# 3. 探索実行
# 戦略: 
# Step 1: 目標グラム数に最も近い「総ボトル数 (10g単位)」を決める
# Step 2: そのボトル数を実現するセット数と内訳を決める
# Step 3: 液量が目標(required_pv)に近づくようフィジオ量を調整する

# 目標ボトル数 (四捨五入)
target_bottles = max(1, round(target_supply_g / 10))
# 探索範囲: 目標ボトル数 ±1本
bottle_search_range = range(max(1, target_bottles - 1), target_bottles + 2)

found_plans = []

for total_bottles in bottle_search_range:
    current_supply_g = total_bottles * 10
    
    # このボトル数を実現するための「セット数」を考える
    # セット数は 1セットあたり1本～2本なので、 total_bottles ～ ceil(total_bottles/2) の範囲
    min_sets = int(np.ceil(total_bottles / 2))
    max_sets = total_bottles
    
    for n_sets in range(min_sets, max_sets + 1):
        # 2本入りセットの数 (鶴亀算)
        # x + y = n_sets
        # 1x + 2y = total_bottles
        # -> y = total_bottles - n_sets
        n_double = total_bottles - n_sets
        n_single = n_sets - n_double
        
        if n_double < 0 or n_single < 0:
            continue
            
        # 液量の最適化
        # 各セットのフィジオ量を調整して、Total Volume を Required PV に近づける
        # 使えるフィジオ量: physio_options (500, 450, 400, 350, 300)
        
        # 全組み合わせは重いので、代表的な組み合わせを探索
        for p_vol_single in physio_options:
            for p_vol_double in physio_options:
                
                vol_single = p_vol_single + 50 # Alb 50mL
                vol_double = p_vol_double + 100 # Alb 100mL (2本)
                
                total_vol = (vol_single * n_single) + (vol_double * n_double)
                
                # 液量チェック
                # 許容範囲: 必要量の 90% ～ 120% (少し多めは許容、少なすぎはNG)
                if total_vol < required_pv * 0.90:
                    continue
                
                # スコア計算 (低いほど良い)
                # 1. グラム数誤差 (最重要) -> Step 1でループしてるので自然に考慮されるが念のため
                score_g = abs(current_supply_g - target_supply_g) * 100
                
                # 2. 液量誤差
                score_vol = abs(total_vol - required_pv) / 10
                
                # 3. 複雑さペナルティ (種類の混在や、変な液量は避ける)
                score_complex = 0
                if n_single > 0 and n_double > 0: score_complex += 20 # 混在
                if p_vol_single != 500: score_complex += 10 # 全量以外は手間
                if p_vol_double != 500: score_complex += 10
                
                final_score = score_g + score_vol + score_complex
                
                found_plans.append({
                    "n_single": n_single, "p_single": p_vol_single,
                    "n_double": n_double, "p_double": p_vol_double,
                    "total_g": current_supply_g,
                    "total_vol": total_vol,
                    "score": final_score
                })

# ベストプランの選択
if found_plans:
    found_plans.sort(key=lambda x: x["score"])
    best_plan = found_plans[0]
else:
    # 万が一見つからない場合のフォールバック (標準的構成)
    sets = int(np.ceil(required_pv / 550))
    best_plan = {
        "n_single": sets, "p_single": 500,
        "n_double": 0, "p_double": 500,
        "total_g": sets*10, "total_vol": sets*550,
        "score": 999
    }

# データ展開
n_a = best_plan["n_single"] # 1本タイプ
p_a = best_plan["p_single"]
n_b = best_plan["n_double"] # 2本タイプ
p_b = best_plan["p_double"]

actual_replacement_vol = best_plan["total_vol"]
supplied_albumin_g = best_plan["total_g"]

# --- シミュレーション (実経過計算) ---
steps = 100
dt_vol = required_pv / steps
current_alb_mass = (epv / 100) * alb_initial
current_pathogen = 100.0 

log_v = [0]
log_alb_loss_cum = [0]
log_pathogen = [100.0]

cum_loss = 0
avg_repl_conc_g_dl = supplied_albumin_g / actual_replacement_vol if actual_replacement_vol > 0 else 0

for _ in range(steps):
    current_alb_conc = current_alb_mass / epv * 100 # g/dL
    step_loss = (current_alb_conc * sc_albumin / 100) * dt_vol
    step_gain = (avg_repl_conc_g_dl / 100) * dt_vol 
    
    current_alb_mass = current_alb_mass - step_loss + step_gain
    cum_loss += step_loss
    
    current_pathogen *= np.exp(-dt_vol * sc_pathogen / epv)
    
    log_v.append(log_v[-1] + dt_vol)
    log_alb_loss_cum.append(cum_loss)
    log_pathogen.append(current_pathogen)

predicted_total_loss_real = cum_loss
final_diff_g = supplied_albumin_g - predicted_total_loss_real
final_balance_percent = (supplied_albumin_g / predicted_total_loss_real - 1) * 100 if predicted_total_loss_real > 0 else 0

# --- 警告判定 ---
alert_msg = None
alert_type = "none"
if final_diff_g < -20:
    alert_type = "error"
    alert_msg = f"⚠️ 警告: アルブミンが大幅に不足します ({int(final_diff_g)}g)。スライダー設定を上げてください。"
elif final_diff_g > 30:
    alert_type = "warning"
    alert_msg = f"⚠️ 警告: アルブミンが過剰です (+{int(final_diff_g)}g)。スライダー設定を下げてください。"

# --- 表示エリア ---
st.title("選択的血漿交換 (SePE) シミュレーション")

if alert_msg:
    if alert_type == "error":
        st.error(alert_msg)
    else:
        st.warning(alert_msg)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("予測循環血漿量 (EPV)", f"{int(epv)} mL", f"{bv_method}")
col2.metric("治療時間", f"{int(treatment_time_min)} 分", f"QP: {qp} mL/min")
col3.metric(f"必要処理量 ({target_removal}%除去)", f"{int(required_pv)} mL", f"{required_pv/epv:.2f} × EPV")
col4.metric("予想Alb喪失量", f"{predicted_total_loss_real:.1f} g", f"基準(0%): {base_loss_g:.1f}g")

balance_color = "normal"
if final_diff_g < -20 or final_diff_g > 30:
    balance_color = "off"

col5.metric(f"アルブミン収支", f"{int(final_diff_g):+d} g", f"目標:{target_supply_g:.1f}g → 採用:{int(supplied_albumin_g)}g", delta_color=balance_color)

st.divider()

# --- 画像と処方提案 ---
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
        st.info("※回路図画像 (circuit.png) がありません")

with c_info:
    st.subheader("📋 補充液作成プラン")
    
    st.success(f"**目標アルブミン量 {target_supply_g:.1f}g に最も近いプラン（{supplied_albumin_g}g）を提案します**")
    
    # 1本タイプ (Type A)
    if n_a > 0:
        vol_a = p_a + 50
        st.markdown(f"""
        #### 🅰️ 基本セット: {vol_a}mL × **{n_a}回**
        * **細胞外液組成:** 500mLバッグのうち **{p_a}mL** を使用
        * **20%アルブミン:** **1本** (10g/50mL) を添加
        """)

    # 2本タイプ (Type B)
    if n_b > 0:
        vol_b = p_b + 100
        st.markdown(f"""
        #### 🅱️ 濃厚セット: {vol_b}mL × **{n_b}回**
        * **細胞外液組成:** 500mLバッグのうち **{p_b}mL** を使用
        * **20%アルブミン:** **2本** (20g/100mL) を添加
        """)
        
    st.markdown("---")
    st.markdown(f"""
    **合計準備:**
    * **細胞外液組成(500mL):** {n_a + n_b} 袋
    * **20%アルブミン:** {n_a*1 + n_b*2} 本
    * **総液量:** {actual_replacement_vol} mL (必要量比 {actual_replacement_vol/required_pv*100:.0f}%)
    """)

st.divider()

# --- グラフ描画 ---
st.subheader(f"治療経過シミュレーション")

fig, ax1 = plt.subplots(figsize=(10, 6))

color_1 = 'tab:red'
ax1.set_xlabel('血漿処理量 (mL)', fontsize=12)
ax1.set_ylabel('【赤】病因物質 残存率 (%)', color=color_1, fontweight='bold', fontsize=12)
line1 = ax1.plot(log_v, log_pathogen, color=color_1, linewidth=3, label='病因物質 残存率 (%)')
ax1.tick_params(axis='y', labelcolor=color_1)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_ylim(0, 105)

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
line2 = ax2.plot(log_v, log_alb_loss_cum, color=color_2, linestyle='--', linewidth=2.5, label='予測アルブミン喪失量 (g)')
ax2.tick_params(axis='y', labelcolor=color_2)
max_y2 = max(max(log_alb_loss_cum), supplied_albumin_g) * 1.2
ax2.set_ylim(0, max_y2)

ax2.axhline(y=supplied_albumin_g, color='green', linestyle=':', alpha=0.7, label=f'総補充量 ({int(supplied_albumin_g)}g)')

# 警告ライン
if final_diff_g > 30:
    ax2.text(0, predicted_total_loss_real + 30, '過剰警告 (+30g)', color='orange', fontsize=9, ha='left')
if final_diff_g < -20:
    ax2.text(0, predicted_total_loss_real - 20, '不足警告 (-20g)', color='red', fontsize=9, ha='left')

lines = line1 + line2 + [ax2.get_lines()[-1]]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11, frameon=False)
plt.tight_layout()
st.pyplot(fig)

# --- 解説 ---
st.divider()
st.header("用語解説・計算根拠")

with st.expander("1. 用語解説 (QP, SC, RC)", expanded=True):
    st.markdown(r"""
    * **QP (Plasma Flow Rate):** 血漿流量（mL/min）。
    * **ふるい係数 (SC):** 膜の透過性（0=阻止、1=通過）。SePEでは病因物質SC≒1.0、Alb SC≒0.6-0.7の膜を使用します。
    * **阻止率 (RC):** 膜による阻止性能 ($RC = 1 - SC$)。
    """)

with st.expander("2. 補液最適化ロジック", expanded=True):
    st.markdown("""
    **アルブミン本数優先:**
    1.  まず、目標とする総アルブミン量（g）に最も近くなる「ボトル本数（1本10g単位）」を決定します。
        * *例: 目標45.7g → 5本(50g)を採用*
    2.  決定した本数を使って、必要液量に最も近づく「細胞外液の量」を50mL刻み（300～500mL）で調整します。
        * *例: 5セットで3000mL必要 → 1セットあたり600mL (フィジオ550+Alb50) は作れないので、フィジオ500+Alb50(550mL) x 5回 + 不足分調整...といった計算を行います。*
    """)
