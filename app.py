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
alb_initial = st.sidebar.number_input("血清アルブミン値 (g/dL)", value=4.0, step=0.1) # デフォルトを4.0に変更

# 2. 治療目標
st.sidebar.subheader("治療目標")
target_removal = st.sidebar.slider("病因物質の除去目標 (%)", 30, 95, 60, step=5)
qp = st.sidebar.number_input("血漿流量 QP (mL/min)", value=30.0, step=5.0)

# 3. 膜特性
st.sidebar.subheader("膜特性 (Evacure EC-4A10c)")
st.sidebar.markdown("<small>※in vivoでの目詰まりや安全域を考慮して調整</small>", unsafe_allow_html=True)
sc_pathogen = st.sidebar.slider("病因物質SC", 0.0, 1.0, 0.90, 0.01)
sc_albumin = st.sidebar.slider("アルブミンSC", 0.0, 1.0, 0.50, 0.01) # デフォルトを0.5に変更

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

# --- 💡 安全性重視のレシピ設計ロジック ---

# 1. 目標濃度の設定（ここを修正）
# 以前の「平均喪失」ではなく、「現在のAlb値を維持するために必要な濃度」を基準にする
# 基準喪失濃度(g/dL) = 患者Alb * SC
est_loss_conc_g_dl = alb_initial * sc_albumin
est_loss_conc_percent = est_loss_conc_g_dl # g/dL = % (近似)

# 安全目標: 喪失濃度 + 5%
target_repl_conc_percent = est_loss_conc_percent * 1.05

# 2. 使用可能な「セットの型」を定義
recipe_patterns = [
    # 濃度低め (Alb 1本)
    {"name": "Light",   "p_vol": 500, "alb_btl": 1, "vol": 550, "alb_g": 10, "conc": 1.81},
    {"name": "Std-1",   "p_vol": 450, "alb_btl": 1, "vol": 500, "alb_g": 10, "conc": 2.00},
    {"name": "Std-2",   "p_vol": 400, "alb_btl": 1, "vol": 450, "alb_g": 10, "conc": 2.22},
    {"name": "Conc-1",  "p_vol": 350, "alb_btl": 1, "vol": 400, "alb_g": 10, "conc": 2.50},
    # 濃度高め (Alb 2本 = 20g)
    {"name": "Double-1", "p_vol": 450, "alb_btl": 2, "vol": 550, "alb_g": 20, "conc": 3.63},
    {"name": "Double-2", "p_vol": 400, "alb_btl": 2, "vol": 500, "alb_g": 20, "conc": 4.00},
    {"name": "Double-3", "p_vol": 300, "alb_btl": 2, "vol": 400, "alb_g": 20, "conc": 5.00},
]

# 3. 最適な組み合わせの探索
best_plan = None
min_diff = float('inf')

# 概算セット数
approx_sets = int(np.ceil(required_pv / 500))
search_sets_range = range(max(1, approx_sets), approx_sets + 2)
found_plans = []

for n_total_sets in search_sets_range:
    for i in range(len(recipe_patterns)):
        for j in range(i, len(recipe_patterns)):
            rec_a = recipe_patterns[i]
            rec_b = recipe_patterns[j]
            
            for k in range(n_total_sets + 1):
                count_a = k
                count_b = n_total_sets - k
                
                total_vol = (rec_a["vol"] * count_a) + (rec_b["vol"] * count_b)
                total_alb = (rec_a["alb_g"] * count_a) + (rec_b["alb_g"] * count_b)
                
                # 液量チェック (95%以上)
                if total_vol < required_pv * 0.95:
                    continue
                
                # 補充液の平均濃度
                avg_repl_conc = (total_alb / total_vol) * 100
                
                # 判定基準: 「目標濃度 (Loss+5%)」にどれだけ近いか
                # 許容範囲: 目標 ±10% (厳しすぎると解なしになるため)
                # 目標: target_repl_conc_percent
                
                diff_from_target = abs(avg_repl_conc - target_repl_conc_percent)
                
                # スコアリング (目標濃度との乖離 + 容量の無駄)
                score = diff_from_target * 2 + abs(total_vol - required_pv)/200
                
                found_plans.append({
                    "rec_a": rec_a, "count_a": count_a,
                    "rec_b": rec_b, "count_b": count_b,
                    "total_vol": total_vol, "total_alb": total_alb,
                    "repl_conc": avg_repl_conc,
                    "score": score
                })

if found_plans:
    found_plans.sort(key=lambda x: x["score"])
    best_plan = found_plans[0]
else:
    # 万が一見つからない場合
    best_plan = {"rec_a": recipe_patterns[1], "count_a": approx_sets, "rec_b": recipe_patterns[1], "count_b": 0, "total_vol": 500*approx_sets, "total_alb": 10*approx_sets, "repl_conc": 2.0, "score": 999}

# 決定したレシピ
rec_a = best_plan["rec_a"]
count_a = best_plan["count_a"]
rec_b = best_plan["rec_b"]
count_b = best_plan["count_b"]
actual_replacement_vol = best_plan["total_vol"]
supplied_albumin_g = best_plan["total_alb"]

# --- シミュレーション (ステップ計算) ---
# 実際にこの補充液を使った場合の推移を正確に計算
steps = 100
dt_vol = required_pv / steps
current_alb_mass = (epv / 100) * alb_initial
current_pathogen = 100.0 # %

log_v = [0]
log_alb_loss_cum = [0]
log_pathogen = [100.0]

cum_loss = 0
avg_repl_conc_g_dl = supplied_albumin_g / actual_replacement_vol # g/dL

for _ in range(steps):
    # 1ステップの排液量 = 補充量 = dt_vol
    
    # 1. 排液中のAlb喪失 (現在の濃度 * SC * 液量)
    current_alb_conc = current_alb_mass / epv * 100 # g/dL
    step_loss = (current_alb_conc * sc_albumin / 100) * dt_vol
    
    # 2. 補充によるAlb付加
    step_gain = (avg_repl_conc_g_dl / 100) * dt_vol
    
    # 3. マスバランス更新
    current_alb_mass = current_alb_mass - step_loss + step_gain
    cum_loss += step_loss
    
    # 病因物質 (補充なし, 単純Washout)
    current_pathogen *= np.exp(-dt_vol * sc_pathogen / epv)
    
    log_v.append(log_v[-1] + dt_vol)
    log_alb_loss_cum.append(cum_loss)
    log_pathogen.append(current_pathogen)

predicted_total_loss_real = cum_loss
diff_alb = supplied_albumin_g - predicted_total_loss_real
balance_percent = (supplied_albumin_g / predicted_total_loss_real - 1) * 100

# --- 表示エリア ---
st.title("選択的血漿交換 (SePE) シミュレーション")

col1, col2, col3, col4, col5 = st.columns(5) # カラム数を増やしてレイアウト調整

col1.metric("予測循環血漿量 (EPV)", f"{int(epv)} mL", f"{bv_method}")
col2.metric("治療時間", f"{int(treatment_time_min)} 分", f"QP: {qp} mL/min")
col3.metric(f"必要処理量 ({target_removal}%除去)", f"{int(required_pv)} mL", f"{required_pv/epv:.2f} × EPV")

# ご要望の配置: 左に予想喪失量、右に収支
col4.metric("予想Alb喪失量", f"{predicted_total_loss_real:.1f} g", f"平均濃度: {predicted_total_loss_real/required_pv*100:.2f}%")
col5.metric("アルブミン収支 (目標+5%)", f"{int(diff_alb):+d} g", f"補充:{int(supplied_albumin_g)}g (+{balance_percent:.1f}%)")

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
    st.subheader("📋 補充液作成プラン (安全設計モデル)")
    
    # 設計根拠の表示
    st.success(f"""
    **🛡️ 安全設計 (低Alb血症の防止):**
    * **患者Alb値:** {alb_initial} g/dL (SC {sc_albumin})
    * **基準喪失濃度:** **{est_loss_conc_percent:.2f}%** (初期値を維持するために必要な濃度)
    * **目標補充濃度:** **{target_repl_conc_percent:.2f}%** (基準 + 5% の安全マージン)
    """)
    
    if count_a > 0:
        st.markdown(f"""
        #### 🅰️ セットA: {rec_a['name']} ({rec_a['vol']}mL) × **{count_a}回**
        * **フィジオ140:** 500mLから **{rec_a['p_vol']}mL** を分取
        * **20%アルブミン:** **{rec_a['alb_btl']}本** ({rec_a['alb_btl']*50}mL) 添加
        * <small>濃度: {rec_a['conc']:.2f}%</small>
        """)
        
    if count_b > 0:
        st.markdown(f"""
        #### 🅱️ セットB: {rec_b['name']} ({rec_b['vol']}mL) × **{count_b}回**
        * **フィジオ140:** 500mLから **{rec_b['p_vol']}mL** を分取
        * **20%アルブミン:** **{rec_b['alb_btl']}本** ({rec_b['alb_btl']*50}mL) 添加
        * <small>濃度: {rec_b['conc']:.2f}%</small>
        """)
        
    st.markdown("---")
    st.markdown(f"""
    **合計:**
    * **フィジオ:** {count_a + count_b} 袋 / **Alb:** {count_a*rec_a['alb_btl'] + count_b*rec_b['alb_btl']} 本
    * **補充液平均濃度:** **{best_plan['repl_conc']:.2f}%**
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
# グラフの上限を、喪失量または補充量の大きい方に合わせる
max_y2 = max(max(log_alb_loss_cum), supplied_albumin_g) * 1.2
ax2.set_ylim(0, max_y2)

# 補充量のラインを引く（参考）
ax2.axhline(y=supplied_albumin_g, color='green', linestyle=':', alpha=0.7, label=f'総補充量 ({int(supplied_albumin_g)}g)')

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

with st.expander("2. 安全設計ロジック (危険回避)", expanded=True):
    st.markdown("""
    **以前の計算との違い:**
    * 旧ロジック: 「アルブミン濃度が下がる」ことを前提とした平均計算 → **過小評価のリスク**
    * **新ロジック:** 「初期アルブミン濃度を維持する」ことを前提とした安全計算
    
    **計算式:**
    $$ \text{目標補充濃度} = (\text{患者Alb値} \times SC) \times 1.05 $$
    これにより、治療初期の高濃度排液にも負けない十分な補充を行い、低アルブミン血症を確実に防ぎます。
    """)
