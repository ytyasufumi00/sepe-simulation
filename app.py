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

# D. アルブミン喪失予測
total_alb_body_g = (epv / 100) * alb_initial
alb_remaining_ratio = np.exp(-required_pv * sc_albumin / epv)
predicted_alb_loss_g = total_alb_body_g * (1 - alb_remaining_ratio)

# --- 💡 高度なレシピ設計ロジック ---

# 1. 目標の設定
# 喪失する液体の平均濃度 = 喪失Alb総量 / 処理量PV
if required_pv > 0:
    avg_loss_conc = predicted_alb_loss_g / required_pv * 100 # %
else:
    avg_loss_conc = 0

# 目標補充濃度 (喪失濃度 + 15% の安全マージン)
target_conc = avg_loss_conc * 1.15
target_alb_g = predicted_alb_loss_g * 1.15

# 2. 使用可能な「セットの型」を定義
# (名前, フィジオ量mL, Alb本数, 総容量mL, Alb量g, 濃度%)
# フィジオは500mLバッグから抜き取る前提 (残液: 500 - Physio量)
recipe_patterns = [
    # 濃度低め (Alb 1本)
    {"name": "Light",   "p_vol": 500, "alb_btl": 1, "vol": 550, "alb_g": 10, "conc": 1.81},
    {"name": "Std-1",   "p_vol": 450, "alb_btl": 1, "vol": 500, "alb_g": 10, "conc": 2.00},
    {"name": "Std-2",   "p_vol": 400, "alb_btl": 1, "vol": 450, "alb_g": 10, "conc": 2.22},
    {"name": "Conc-1",  "p_vol": 350, "alb_btl": 1, "vol": 400, "alb_g": 10, "conc": 2.50},
    # 濃度高め (Alb 2本 = 100mL)
    {"name": "Double-1", "p_vol": 450, "alb_btl": 2, "vol": 550, "alb_g": 20, "conc": 3.63},
    {"name": "Double-2", "p_vol": 400, "alb_btl": 2, "vol": 500, "alb_g": 20, "conc": 4.00},
    {"name": "Double-3", "p_vol": 300, "alb_btl": 2, "vol": 400, "alb_g": 20, "conc": 5.00},
]

# 3. 最適な組み合わせの探索
best_plan = None
min_error = float('inf')

# 必要なセット数の概算 (平均500mLとして)
approx_sets = int(np.ceil(required_pv / 500))
# 探索範囲: 概算セット数 ±1
search_sets_range = range(max(1, approx_sets), approx_sets + 2)

found_plans = []

# パターンAとパターンBを組み合わせる総当たり探索
for n_total_sets in search_sets_range:
    for i in range(len(recipe_patterns)):
        for j in range(i, len(recipe_patterns)): # 同じか、それ以降のパターン (重複組み合わせ)
            rec_a = recipe_patterns[i]
            rec_b = recipe_patterns[j]
            
            # Aを k 個、 Bを (n_total_sets - k) 個 使う
            for k in range(n_total_sets + 1):
                count_a = k
                count_b = n_total_sets - k
                
                total_vol = (rec_a["vol"] * count_a) + (rec_b["vol"] * count_b)
                total_alb = (rec_a["alb_g"] * count_a) + (rec_b["alb_g"] * count_b)
                
                # 制約1: 容量が足りているか？ (95%以上)
                if total_vol < required_pv * 0.95:
                    continue
                    
                # 制約2: アルブミンバランス (喪失量 + 0% ～ +30% の範囲)
                # ユーザー希望は+15%前後だが、組み合わせによってはピッタリいかないので幅を持たせる
                if predicted_alb_loss_g > 0:
                    balance_pct = (total_alb / predicted_alb_loss_g - 1) * 100
                else:
                    balance_pct = 0
                
                if 0 <= balance_pct <= 30:
                    # 評価スコア: +15%からの乖離 + 容量の無駄のなさ
                    score = abs(balance_pct - 15) + abs(total_vol - required_pv)/100
                    
                    found_plans.append({
                        "rec_a": rec_a,
                        "count_a": count_a,
                        "rec_b": rec_b,
                        "count_b": count_b,
                        "total_vol": total_vol,
                        "total_alb": total_alb,
                        "balance": balance_pct,
                        "score": score
                    })

# スコア順にソートしてベストを選択
if found_plans:
    found_plans.sort(key=lambda x: x["score"])
    best_plan = found_plans[0]
else:
    # 条件に合うものが見つからない場合、最もマシなもの（標準セットのみ）をデフォルトにする安全策
    def_rec = recipe_patterns[1] # Std-1
    n = int(np.ceil(required_pv / def_rec["vol"]))
    best_plan = {
        "rec_a": def_rec, "count_a": n,
        "rec_b": def_rec, "count_b": 0,
        "total_vol": def_rec["vol"]*n, "total_alb": def_rec["alb_g"]*n,
        "balance": (def_rec["alb_g"]*n / predicted_alb_loss_g - 1)*100 if predicted_alb_loss_g else 0,
        "score": 999
    }

# 結果の展開
rec_a = best_plan["rec_a"]
count_a = best_plan["count_a"]
rec_b = best_plan["rec_b"]
count_b = best_plan["count_b"]
actual_replacement_vol = best_plan["total_vol"]
supplied_albumin_g = best_plan["total_alb"]
balance_percent = best_plan["balance"]


# --- 表示エリア ---
st.title("選択的血漿交換 (SePE) シミュレーション")

col1, col2, col3, col4 = st.columns(4)
col1.metric("予測循環血漿量 (EPV)", f"{int(epv)} mL", f"{bv_method}")
col2.metric("治療時間", f"{int(treatment_time_min)} 分", f"QP: {qp} mL/min")
col3.metric(f"必要処理量 ({target_removal}%除去)", f"{int(required_pv)} mL", f"{required_pv/epv:.2f} × EPV")
col4.metric("アルブミン収支", f"{int(supplied_albumin_g - predicted_alb_loss_g):+d} g", f"補充:{int(supplied_albumin_g)}g (喪失+{balance_percent:.1f}%)")

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
    st.subheader("📋 補充液作成プラン (自動最適化)")
    
    # ロジックの説明
    st.info(f"""
    **計算根拠:**
    * **予測喪失濃度:** 約 {avg_loss_conc:.2f}% ({predicted_alb_loss_g:.1f}g / {int(required_pv)}mL)
    * **目標補充濃度:** {target_conc:.2f}% (喪失+15%設定)
    * これに適合するよう、以下の組み合わせを提案します。
    """)
    
    # パターンAの表示
    if count_a > 0:
        st.markdown(f"""
        #### 🅰️ パターンA: {rec_a['name']} ({rec_a['vol']}mL) × **{count_a}セット**
        * **フィジオ140:** 500mLから **{rec_a['p_vol']}mL** を分取
        * **20%アルブミン:** **{rec_a['alb_btl']}本** ({rec_a['alb_btl']*50}mL) 添加
        """)
        
    # パターンBの表示 (あれば)
    if count_b > 0:
        st.markdown(f"""
        #### 🅱️ パターンB: {rec_b['name']} ({rec_b['vol']}mL) × **{count_b}セット**
        * **フィジオ140:** 500mLから **{rec_b['p_vol']}mL** を分取
        * **20%アルブミン:** **{rec_b['alb_btl']}本** ({rec_b['alb_btl']*50}mL) 添加
        """)
        
    st.markdown("---")
    st.markdown(f"""
    **合計準備数:**
    * **フィジオ140 (500mL):** {count_a + count_b} 袋
    * **20%アルブミン (50mL):** {count_a*rec_a['alb_btl'] + count_b*rec_b['alb_btl']} 本
    * **総液量:** {actual_replacement_vol} mL (対処理量 {actual_replacement_vol/required_pv*100:.0f}%)
    """)

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

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11, frameon=False)
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

with st.expander("2. Evacure EC-4A10c のSC設定と安全域", expanded=True):
    st.markdown("""
    **カタログ値と安全域:**
    カタログ値（In vitro牛血）に対し、臨床（In vivo）では二次膜形成によりSCが低下します。
    * **病因物質:** 除去不足を防ぐため、SCを**低め**に見積もり、必要処理量を確保します。
    * **アルブミン:** 喪失過多を防ぐため、SCを**高め**に見積もり、補充計画を立てます。
    """)

with st.expander("3. 補充液レシピの自動設計ロジック (新)", expanded=True):
    st.markdown("""
    **濃度逆算アプローチ:**
    1.  **予測喪失濃度**を算出 ($= \text{予測喪失量} / \text{必要処理量}$)
    2.  これに対し、**+15%の安全マージン**を乗せた目標濃度を設定します。
    3.  **組み合わせ最適化:** * フィジオ+Alb1本 (1.8%~2.5%)
        * フィジオ+Alb2本 (3.6%~5.0%)
        これらのプリセットから、目標濃度と総液量に最も合致する組み合わせ（例: Aセット4回 + Bセット2回）を自動算出します。
    """)
