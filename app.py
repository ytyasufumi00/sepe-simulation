import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# --- フォント設定 ---
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# --- メイン処理 ---
st.set_page_config(page_title="SePE Simulation - 信州上田医療センター 腎臓内科ver.", layout="wide")

# --- サイドバー ---
st.sidebar.header("患者・治療パラメータ設定")

# 1. 患者情報
st.sidebar.subheader("患者情報")
height = st.sidebar.number_input("身長 (cm) ※任意", value=0.0, step=0.1, help="入力なし(0.0)の場合は簡易式(70mL/kg)が適用されます。")
weight = st.sidebar.number_input("体重 (kg)", value=50.0, step=0.1)
hct = st.sidebar.number_input("血中ヘマトクリット値 (%)", value=30.0, step=0.1)
alb_initial = st.sidebar.number_input("血清アルブミン値 (g/dL)", value=3.5, step=0.1)

# 2. 治療目標
st.sidebar.subheader("治療目標")
# デフォルト値を50%に変更
target_removal = st.sidebar.slider("病因物質の除去目標 (%)", 30, 95, 50, step=5)
qp = st.sidebar.number_input("血漿流量 QP (mL/min)", value=30.0, step=5.0)

# 3. アルブミンバランス調整
st.sidebar.subheader("アルブミン収支設定")
target_balance_ratio = st.sidebar.slider("収支目標 (対喪失量 %)", -10, 20, 5, step=1, help="予測喪失量に対して、何％上乗せして補充するか設定します。")

# 4. 膜特性
st.sidebar.subheader("膜特性 (Evacure EC-4A10c)")
st.sidebar.info("💡 **設定のポイント:**\n初期値はカタログ値のアルブミンSC=0.6と設定していますが、実際の治療(in vivo)では、タンパク付着(ファウリング)によりSCはカタログ値SC=0.6より低下：予測値よりアルブミンを喪失しない可能性があります。病因物質SCの初期値は、エバキュアーEC-4AのIgGに対するカタログ値SC=0.4としています")
sc_pathogen = st.sidebar.slider("病因物質SC", 0.0, 1.0, 0.40, 0.01)
sc_albumin = st.sidebar.slider("アルブミンSC", 0.0, 1.0, 0.60, 0.01)

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

# --- 💡 喪失量計算 (線形モデル) ---

# 1. アルブミン喪失量の計算
# 排液中濃度(g/dL) = 血清Alb × SC
filtrate_alb_conc = alb_initial * sc_albumin

# 基準予測喪失量(g) = 処理量(dL) × 排液中濃度(g/dL)
base_loss_g = (required_pv / 100.0) * filtrate_alb_conc

# 目標補充量 (g)
target_supply_g = base_loss_g * (1 + target_balance_ratio / 100.0)

# 2. レシピパターンの定義
recipe_patterns = [
    # 通常セット (Alb 1本 = 10g)
    {"name": "Std-500", "p_vol": 500, "alb_btl": 1, "vol": 550, "alb_g": 10},
    {"name": "Std-450", "p_vol": 450, "alb_btl": 1, "vol": 500, "alb_g": 10},
    {"name": "Std-400", "p_vol": 400, "alb_btl": 1, "vol": 450, "alb_g": 10},
    {"name": "Std-350", "p_vol": 350, "alb_btl": 1, "vol": 400, "alb_g": 10},
    # 濃厚セット (Alb 2本 = 20g)
    {"name": "Dbl-450", "p_vol": 450, "alb_btl": 2, "vol": 550, "alb_g": 20},
    {"name": "Dbl-400", "p_vol": 400, "alb_btl": 2, "vol": 500, "alb_g": 20},
    {"name": "Dbl-350", "p_vol": 350, "alb_btl": 2, "vol": 450, "alb_g": 20},
    # 希釈のみ (Alb なし)
    {"name": "Plain-500", "p_vol": 500, "alb_btl": 0, "vol": 500, "alb_g": 0},
    {"name": "Plain-400", "p_vol": 400, "alb_btl": 0, "vol": 400, "alb_g": 0},
]

# 3. 最適な組み合わせ探索
best_plan = None
approx_sets = int(required_pv / 500)
search_range = range(max(1, approx_sets - 2), approx_sets + 4)
found_plans = []

for n_total_sets in search_range:
    for i in range(len(recipe_patterns)):
        for j in range(i, len(recipe_patterns)):
            rec_a = recipe_patterns[i]
            rec_b = recipe_patterns[j]
            
            for k in range(n_total_sets + 1):
                count_a = k
                count_b = n_total_sets - k
                
                total_vol = (rec_a["vol"] * count_a) + (rec_b["vol"] * count_b)
                total_alb = (rec_a["alb_g"] * count_a) + (rec_b["alb_g"] * count_b)
                
                # スコア計算
                diff_g = abs(total_alb - target_supply_g)
                score_g = (diff_g ** 2) * 50
                
                diff_vol = abs(total_vol - required_pv)
                if 0.85 * required_pv <= total_vol <= 1.25 * required_pv:
                     score_vol = diff_vol / 10
                else:
                     score_vol = diff_vol * 10 
                
                score_complex = 0
                if count_a > 0 and count_b > 0: score_complex += 50
                if rec_a["p_vol"] != 500: score_complex += 5
                if count_b > 0 and rec_b["p_vol"] != 500: score_complex += 5
                
                total_score = score_g + score_vol + score_complex
                
                found_plans.append({
                    "rec_a": rec_a, "count_a": count_a,
                    "rec_b": rec_b, "count_b": count_b,
                    "total_g": total_alb, "total_vol": total_vol,
                    "score": total_score
                })

if found_plans:
    found_plans.sort(key=lambda x: x["score"])
    best_plan = found_plans[0]
else:
    def_rec = recipe_patterns[0]
    n = int(required_pv / 550) + 1
    best_plan = {"rec_a": def_rec, "count_a": n, "rec_b": def_rec, "count_b": 0, "total_g": n*10, "total_vol": n*550, "score": 999}

rec_a = best_plan["rec_a"]
count_a = best_plan["count_a"]
rec_b = best_plan["rec_b"]
count_b = best_plan["count_b"]
actual_replacement_vol = best_plan["total_vol"]
supplied_albumin_g = best_plan["total_g"]

# --- 指標計算 ---
repl_alb_conc = supplied_albumin_g / actual_replacement_vol * 100 if actual_replacement_vol > 0 else 0
final_diff_g = supplied_albumin_g - base_loss_g
avg_loss_conc = base_loss_g / required_pv * 100 if required_pv > 0 else 0

# --- シミュレーション (グラフ用) ---
steps = 100
dt_vol = required_pv / steps
log_v = np.linspace(0, required_pv * 1.2, steps)
log_pathogen = 100 * np.exp(-log_v * sc_pathogen / epv)
# 喪失量線形増加
log_alb_loss_cum = (log_v / 100.0) * filtrate_alb_conc

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
st.markdown("#### 信州上田医療センター 腎臓内科ver.")

if alert_msg:
    if alert_type == "error":
        st.error(alert_msg)
    else:
        st.warning(alert_msg)

col1, col2, col3 = st.columns(3)
col1.metric("予測循環血漿量 (EPV)", f"{int(epv)} mL", f"{bv_method}")
col2.metric("治療時間", f"{int(treatment_time_min)} 分", f"QP: {qp} mL/min")
col3.metric(f"必要処理量 ({target_removal}%除去)", f"{int(required_pv)} mL", f"{required_pv/epv:.2f} × EPV")

col4, col5, col6 = st.columns(3)
col4.metric("予想Alb喪失量", f"{base_loss_g:.1f} g", f"廃液中濃度: {filtrate_alb_conc:.2f}g/dL")
col5.metric("排液中アルブミン濃度", f"{filtrate_alb_conc:.2f} g/dL", f"患者Alb {alb_initial} × SC {sc_albumin}")
col6.metric("補充液アルブミン濃度 (平均)", f"{repl_alb_conc:.2f} g/dL", f"総Alb {supplied_albumin_g}g / 総液量 {actual_replacement_vol}mL")

st.markdown("---")
c_bal, c_plan = st.columns([1, 2])

with c_bal:
    st.subheader("アルブミン収支")
    balance_color = "normal"
    if final_diff_g < -20 or final_diff_g > 30:
        balance_color = "off"
    st.metric(f"収支結果", f"{int(final_diff_g):+d} g", f"目標:{target_supply_g:.1f}g → 採用:{int(supplied_albumin_g)}g", delta_color=balance_color)
    
    st.markdown(f"""
    * **補充:** {supplied_albumin_g} g
    * **喪失:** {base_loss_g:.1f} g
    * **設定目標:** {target_balance_ratio:+}%
    """)

with c_plan:
    st.subheader("📋 最適化補充液プラン")
    
    def display_plan(rec, count, label):
        vol = rec['vol']
        p_vol = rec['p_vol']
        btl = rec['alb_btl']
        
        alb_text = f"**{btl}本** ({btl*10}g)" if btl > 0 else "なし"
        
        st.markdown(f"""
        #### {label}: {vol}mL × **{count}回**
        * **細胞外液:** 500mLバッグのうち **{p_vol}mL** を使用
        * **20%アルブミン 50ml:** {alb_text} 添加
        """)

    if count_a > 0:
        icon = "🅰️" if count_b == 0 else "🅰️"
        display_plan(rec_a, count_a, f"{icon} パターンA")
        
    if count_b > 0:
        display_plan(rec_b, count_b, "🅱️ パターンB")
        
    st.markdown("---")
    st.markdown(f"""
    ### 合計準備数
    * **細胞外液 (500mL):** **{count_a+count_b}** 袋
    * **20%アルブミン 50ml:** **{count_a*rec_a['alb_btl'] + count_b*rec_b['alb_btl']}** 本
    * **総液量:** **{actual_replacement_vol}** mL
    """)

st.divider()

# --- 画像 (最初から表示) ---
if os.path.exists("circuit.png") or os.path.exists("circuit.jpg"):
    st.subheader("SePE 回路構成図")
    img_path = "circuit.png" if os.path.exists("circuit.png") else "circuit.jpg"
    st.image(img_path)

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
    ax2.text(0, base_loss_g + 30, '過剰警告 (+30g)', color='orange', fontsize=9, ha='left')
if final_diff_g < -20:
    ax2.text(0, base_loss_g - 20, '不足警告 (-20g)', color='red', fontsize=9, ha='left')

lines = line1 + line2 + [ax2.get_lines()[-1]]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11, frameon=False)
plt.tight_layout()
st.pyplot(fig)

# --- 解説 (完全版) ---
st.divider()
st.header("用語解説・計算根拠")

with st.expander("1. 用語解説 (QP, SC, RC)", expanded=True):
    st.markdown(r"""
    * **QP (Plasma Flow Rate):** * 血漿分離器（EC-4A10c）へ供給される血漿流量（mL/min）です。
    * **ふるい係数 (SC, Sieving Coefficient):** * 膜における物質の「通りやすさ」を示す指標です（0.0～1.0）。
        * $SC = \frac{C_{Filtrate}}{C_{Plasma}}$
        * 1.0に近いほど素通りし、0に近いほど阻止されます。SePEでは「病因物質は1.0に近く、アルブミンは0.6～0.7程度」の膜を使用します。
    * **阻止率 (RC, Rejection Coefficient):** * 膜が物質を「どれだけ通さないか」を示す指標です。$RC = 1 - SC$
    * **排液中アルブミン濃度:**
        * 膜を通過して廃棄される液体中のアルブミン濃度です。本システムでは $\text{患者Alb} \times SC$ で計算します。
    """)

with st.expander("2. Evacure EC-4A10c におけるSC設定の根拠と調整", expanded=True):
    st.markdown("""
    **カタログ値と臨床値の乖離（Safety Margin）**
    In vivo（実際の治療）では、タンパク質の付着や目詰まり（**ファウリング**）により、二次膜が形成され、実効SCはカタログ値よりも低下する傾向があります。
    
    **推奨される調整:**
    * **病因物質SC:** 除去不全を防ぐため、**低め**に見積もって必要処理量を計算します。
    * **アルブミンSC:** 喪失過多を防ぐため、**高め**（0.6程度）に見積もって補充計画を立てます。
    """)

with st.expander("3. 循環血漿量・必要処理量の計算根拠", expanded=True):
    st.markdown(r"""
    **A. 予測循環血漿量 (EPV)**
    * **小川の式:** $BV(L) = 0.16874 \times Height(m) + 0.05986 \times Weight(kg) - 0.0305$
    * **血漿量:** $EPV = BV \times (1 - Hct/100)$

    **B. 必要な血漿処理量 (Required PV)**
    * 病因物質は補充されないため、指数関数的に減少（Washout）します。
      $$ V = \frac{- \ln(1 - R) \times EPV}{SC_{pathogen}} $$

    **C. アルブミン喪失量の予測**
    * アルブミンは補充液により濃度が維持される前提のため、処理量に比例して喪失します（線形モデル）。
      $$ \text{Loss} (g) = \text{排液中濃度} (g/dL) \times \text{処理量} (dL) $$
    """)





