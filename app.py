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

# --- 💡 多彩なプランからの最適化ロジック ---

# 1. 目標の設定
total_alb_body_g = (epv / 100) * alb_initial
alb_remaining_ratio_base = np.exp(-required_pv * sc_albumin / epv)
base_loss_g = total_alb_body_g * (1 - alb_remaining_ratio_base)

# 目標補充量 (g)
target_supply_g = base_loss_g * (1 + target_balance_ratio / 100.0)

# 2. レシピパターンの定義 (多彩なバリエーション)
# alb_btl: 20%アルブミン(50mL)の本数 (1本=10g)
# p_vol: 細胞外液の使用量
# vol: 総液量 (p_vol + 50*本数)
recipe_patterns = [
    # --- 通常セット (Alb 1本 = 10g) ---
    {"name": "Std-500", "p_vol": 500, "alb_btl": 1, "vol": 550, "alb_g": 10},
    {"name": "Std-450", "p_vol": 450, "alb_btl": 1, "vol": 500, "alb_g": 10},
    {"name": "Std-400", "p_vol": 400, "alb_btl": 1, "vol": 450, "alb_g": 10},
    {"name": "Std-350", "p_vol": 350, "alb_btl": 1, "vol": 400, "alb_g": 10},
    
    # --- 濃厚セット (Alb 2本 = 20g) ---
    {"name": "Dbl-450", "p_vol": 450, "alb_btl": 2, "vol": 550, "alb_g": 20},
    {"name": "Dbl-400", "p_vol": 400, "alb_btl": 2, "vol": 500, "alb_g": 20},
    {"name": "Dbl-350", "p_vol": 350, "alb_btl": 2, "vol": 450, "alb_g": 20},
    
    # --- 希釈のみ (Alb なし) ---
    {"name": "Plain-500", "p_vol": 500, "alb_btl": 0, "vol": 500, "alb_g": 0},
    {"name": "Plain-400", "p_vol": 400, "alb_btl": 0, "vol": 400, "alb_g": 0},
]

# 3. 最適な組み合わせ探索
# 戦略: 
#  - 最大2種類のレシピを組み合わせる (現場の混乱防止)
#  - 総当たりで「Alb誤差」と「液量誤差」が最小になるものを探す

best_plan = None
# 必要セット数の概算 (平均500mLとして)
approx_sets = int(required_pv / 500)
# 探索範囲: 少なめ～多めまで幅広く
search_range = range(max(1, approx_sets - 2), approx_sets + 4)

found_plans = []

for n_total_sets in search_range:
    # 2種類のレシピ (rec_a, rec_b) を選ぶループ
    # rec_a と rec_b が同じ場合も含む(=1種類のみ使用)
    for i in range(len(recipe_patterns)):
        for j in range(i, len(recipe_patterns)):
            rec_a = recipe_patterns[i]
            rec_b = recipe_patterns[j]
            
            # 内訳を決めるループ (aがk個, bが残り)
            for k in range(n_total_sets + 1):
                count_a = k
                count_b = n_total_sets - k
                
                # 合計計算
                total_vol = (rec_a["vol"] * count_a) + (rec_b["vol"] * count_b)
                total_alb = (rec_a["alb_g"] * count_a) + (rec_b["alb_g"] * count_b)
                
                # スコア計算 (ペナルティ方式: 0に近いほど良い)
                
                # 1. アルブミン誤差 (最重要: 重み大)
                # 目標との差(g)の2乗ペナルティ
                diff_g = abs(total_alb - target_supply_g)
                score_g = (diff_g ** 2) * 50
                
                # 2. 液量誤差 (重要: 重み中)
                # 許容範囲(±10%)を超えるとペナルティ激増
                diff_vol = abs(total_vol - required_pv)
                if 0.95 * required_pv <= total_vol <= 1.15 * required_pv:
                     score_vol = diff_vol / 10
                else:
                     score_vol = diff_vol * 10 # 範囲外は採用したくない
                
                # 3. 複雑さペナルティ (なるべく1種類、なるべく500mL全量使用が良い)
                score_complex = 0
                if count_a > 0 and count_b > 0: score_complex += 50 # 2種類混在は少しペナルティ
                if rec_a["p_vol"] != 500: score_complex += 5 # 分取作業の手間
                if count_b > 0 and rec_b["p_vol"] != 500: score_complex += 5
                
                total_score = score_g + score_vol + score_complex
                
                found_plans.append({
                    "rec_a": rec_a, "count_a": count_a,
                    "rec_b": rec_b, "count_b": count_b,
                    "total_g": total_alb, "total_vol": total_vol,
                    "score": total_score
                })

# ベストプラン選出
if found_plans:
    found_plans.sort(key=lambda x: x["score"])
    best_plan = found_plans[0]
else:
    # フォールバック
    def_rec = recipe_patterns[0]
    n = int(required_pv / 550) + 1
    best_plan = {"rec_a": def_rec, "count_a": n, "rec_b": def_rec, "count_b": 0, "total_g": n*10, "total_vol": n*550, "score": 999}

# データ展開
rec_a = best_plan["rec_a"]
count_a = best_plan["count_a"]
rec_b = best_plan["rec_b"]
count_b = best_plan["count_b"]
actual_replacement_vol = best_plan["total_vol"]
supplied_albumin_g = best_plan["total_g"]

# --- シミュレーション (実経過) ---
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
    st.subheader("📋 補充液作成プラン (最適化済み)")
    
    # 詳細プラン表示関数
    def display_recipe(rec, count, label):
        vol_total = rec['vol']
        physio_use = rec['p_vol']
        alb_bottles = rec['alb_btl']
        
        # アルブミン本数の表記
        if alb_bottles == 0:
            alb_text = "なし"
        else:
            alb_text = f"**{alb_bottles}本** ({alb_bottles*10}g)"
            
        st.markdown(f"""
        #### {label}: {vol_total}mL × **{count}回**
        * **細胞外液組成(フィジオ140等):** 500mLのうち **{physio_use}mL** を使用
        * **20%アルブミン:** {alb_text}
        """)

    # プランA
    if count_a > 0:
        display_recipe(rec_a, count_a, "🅰️ パターンA")
        
    # プランB
    if count_b > 0:
        display_recipe(rec_b, count_b, "🅱️ パターンB")
        
    st.markdown("---")
    st.markdown(f"""
    **合計準備数:**
    * **細胞外液組成(500mL):** {count_a + count_b} 袋
    * **20%アルブミン:** {count_a*rec_a['alb_btl'] + count_b*rec_b['alb_btl']} 本
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

with st.expander("2. 補液最適化ロジック (Advanced)", expanded=True):
    st.markdown("""
    **多彩なレシピ選択:**
    以下のパターンを自動で組み合わせ、**「目標アルブミン量」と「目標液量」の誤差が最も少ないプラン**を提案します。
    * **通常セット:** 細胞外液(350~500mL) + Alb 10g
    * **濃厚セット:** 細胞外液(350~450mL) + Alb 20g
    * **希釈セット:** 細胞外液(400~500mL) + Alb なし
    """)
