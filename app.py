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
st.sidebar.info("💡 **設定のポイント:**\n実際の治療(in vivo)では、タンパク付着(ファウリング)によりSCはカタログ値より低下します。安全のため、アルブミン喪失見積もりには高めの値(0.6程度)を使用することを推奨します。")
sc_pathogen = st.sidebar.slider("病因物質SC", 0.0, 1.0, 0.90, 0.01)
sc_albumin = st.sidebar.slider("アルブミンSC", 0.0, 1.0, 0.60, 0.01) # デフォルト0.6に変更

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

# 2. 探索用部品
physio_options = [500, 450, 400, 350, 300]
recipe_patterns = [
    # 通常セット (Alb 1本)
    {"name": "Std", "btl": 1, "alb_g": 10},
    # 濃厚セット (Alb 2本)
    {"name": "Dbl", "btl": 2, "alb_g": 20},
    # 希釈セット (Alb なし)
    {"name": "Plain", "btl": 0, "alb_g": 0},
]

best_plan = None
# 目標ボトル数
target_bottles = max(0, round(target_supply_g / 10))
# 探索範囲
bottle_search_range = range(max(0, target_bottles - 2), target_bottles + 3)

found_plans = []

for total_bottles in bottle_search_range:
    current_supply_g = total_bottles * 10
    
    # セット数の概算 (液量ベース)
    min_sets = max(1, int(required_pv / 550))
    max_sets = int(required_pv / 300) + 1
    
    for n_sets in range(min_sets, max_sets + 1):
        # ボトル配分 (2本入, 1本入, 0本入)
        # x*2 + y*1 + z*0 = total_bottles
        # x + y + z = n_sets
        
        # 簡易化: 最大2種類の混合で探索
        # パターン1: 2本入(x) と 1本入(y)
        # 2x + y = total_bottles
        # x + y = n_sets -> y = n_sets - x
        # 2x + (n_sets - x) = total_bottles -> x + n_sets = total_bottles -> x = total_bottles - n_sets
        
        x = total_bottles - n_sets # 2本入の数
        y = n_sets - x             # 1本入の数
        z = 0                      # 0本入
        
        # 負の数になったらこの組み合わせは成立しない -> 他の組み合わせ(0本入を使う等)を試す
        valid_combos = []
        
        # Combo A: 2本と1本の混合
        if x >= 0 and y >= 0:
            valid_combos.append({"dbl": x, "std": y, "pln": 0})
            
        # Combo B: 1本と0本の混合 (目標gが少ない場合)
        # 1*y + 0*z = total_bottles -> y = total_bottles
        # y + z = n_sets -> z = n_sets - total_bottles
        y2 = total_bottles
        z2 = n_sets - total_bottles
        if y2 >= 0 and z2 > 0: # z2>0でないとCombo Aと同じになる
            valid_combos.append({"dbl": 0, "std": y2, "pln": z2})

        # Combo C: 2本と0本の混合 (極端な場合)
        # 2*x + 0*z = total_bottles -> x = total_bottles / 2
        if total_bottles % 2 == 0:
            x3 = total_bottles // 2
            z3 = n_sets - x3
            if x3 > 0 and z3 > 0:
                valid_combos.append({"dbl": x3, "std": 0, "pln": z3})

        for combo in valid_combos:
            n_dbl = combo["dbl"]
            n_std = combo["std"]
            n_pln = combo["pln"]
            
            # 液量の最適化
            # 各セットのフィジオ量を physio_options から選ぶ
            # 全探索は重いので、液不足なら多い方、液過剰なら少ない方へ寄せる
            
            # 平均必要液量
            avg_vol_needed = required_pv / n_sets
            
            # Alb液量分を引いた、必要なフィジオ量
            # Dbl: +100mL, Std: +50mL, Pln: +0mL
            alb_vol_total = n_dbl*100 + n_std*50
            physio_needed_total = required_pv - alb_vol_total
            avg_physio_needed = physio_needed_total / n_sets
            
            # physio_optionsの中で最も近いものを選ぶ
            closest_p = min(physio_options, key=lambda x: abs(x - avg_physio_needed))
            
            # 総液量
            total_vol = (closest_p * n_pln) + ((closest_p+50) * n_std) + ((closest_p+100) * n_dbl)
            
            # スコア計算
            diff_g = abs(current_supply_g - target_supply_g)
            diff_vol = abs(total_vol - required_pv)
            
            # 液量許容範囲 (90% - 120%)
            if not (required_pv * 0.9 <= total_vol <= required_pv * 1.2):
                score_vol = diff_vol * 100 # ペナルティ大
            else:
                score_vol = diff_vol / 10
            
            # 複雑性ペナルティ (種類が多いとダメ)
            types = 0
            if n_dbl > 0: types += 1
            if n_std > 0: types += 1
            if n_pln > 0: types += 1
            score_complex = (types - 1) * 20
            
            total_score = (diff_g ** 2) * 10 + score_vol + score_complex
            
            found_plans.append({
                "n_dbl": n_dbl, "p_dbl": closest_p,
                "n_std": n_std, "p_std": closest_p,
                "n_pln": n_pln, "p_pln": closest_p,
                "total_g": current_supply_g,
                "total_vol": total_vol,
                "score": total_score
            })

if found_plans:
    found_plans.sort(key=lambda x: x["score"])
    best_plan = found_plans[0]
else:
    # フォールバック
    sets = int(required_pv / 550) + 1
    best_plan = {"n_dbl": 0, "p_dbl": 500, "n_std": sets, "p_std": 500, "n_pln": 0, "p_pln": 500, "total_g": sets*10, "total_vol": sets*550, "score": 999}

# データ展開
n_dbl = best_plan["n_dbl"]
p_dbl = best_plan["p_dbl"]
n_std = best_plan["n_std"]
p_std = best_plan["p_std"]
n_pln = best_plan["n_pln"]
p_pln = best_plan["p_pln"]

actual_replacement_vol = best_plan["total_vol"]
supplied_albumin_g = best_plan["total_g"]

# --- 追加指標の計算 ---
# 1. 排液中のAlb濃度 (推定)
# 患者Alb * SC で近似 (治療開始時の最大濃度)
filtrate_alb_conc = alb_initial * sc_albumin

# 2. 補充液のAlb濃度
repl_alb_conc = supplied_albumin_g / actual_replacement_vol * 100 if actual_replacement_vol > 0 else 0


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

# 3行表示に変更 (情報量が増えたため)
col1, col2, col3 = st.columns(3)
col1.metric("予測循環血漿量 (EPV)", f"{int(epv)} mL", f"{bv_method}")
col2.metric("治療時間", f"{int(treatment_time_min)} 分", f"QP: {qp} mL/min")
col3.metric(f"必要処理量 ({target_removal}%除去)", f"{int(required_pv)} mL", f"{required_pv/epv:.2f} × EPV")

col4, col5, col6 = st.columns(3)
col4.metric("予想Alb喪失量", f"{predicted_total_loss_real:.1f} g", f"基準(0%): {base_loss_g:.1f}g")
# 新しい指標の表示
col5.metric("排液中アルブミン濃度 (推定)", f"{filtrate_alb_conc:.2f} g/dL", f"患者Alb {alb_initial} × SC {sc_albumin}")
col6.metric("補充液アルブミン濃度 (平均)", f"{repl_alb_conc:.2f} g/dL", f"総Alb {supplied_albumin_g}g / 総液量 {actual_replacement_vol}mL")

# 収支は目立つように単独行またはディバイダ後
st.markdown("---")
c_bal, c_plan = st.columns([1, 2])

with c_bal:
    st.subheader("アルブミン収支")
    balance_color = "normal"
    if final_diff_g < -20 or final_diff_g > 30:
        balance_color = "off"
    st.metric(f"収支結果", f"{int(final_diff_g):+d} g", f"目標:{target_supply_g:.1f}g → 採用:{int(supplied_albumin_g)}g", delta_color=balance_color)
    
    st.info(f"""
    **収支設定:** {target_balance_ratio:+}%
    **詳細:**
    * 補充量: {supplied_albumin_g} g
    * 喪失量: {predicted_total_loss_real:.1f} g
    """)

with c_plan:
    st.subheader("📋 最適化補充液プラン")
    
    # 2本タイプ
    if n_dbl > 0:
        vol = p_dbl + 100
        st.markdown(f"""
        #### 🟧 濃厚セット: {vol}mL × **{n_dbl}回**
        * **細胞外液:** 500mLバッグのうち **{p_dbl}mL** を使用
        * **20%アルブミン:** **2本** (20g) 添加
        """)
        
    # 1本タイプ
    if n_std > 0:
        vol = p_std + 50
        st.markdown(f"""
        #### 🟦 通常セット: {vol}mL × **{n_std}回**
        * **細胞外液:** 500mLバッグのうち **{p_std}mL** を使用
        * **20%アルブミン:** **1本** (10g) 添加
        """)
        
    # 0本タイプ
    if n_pln > 0:
        st.markdown(f"""
        #### ⬜ 希釈セット: {p_pln}mL × **{n_pln}回**
        * **細胞外液:** 500mLバッグのうち **{p_pln}mL** を使用
        * **20%アルブミン:** **なし**
        """)
        
    st.caption(f"合計: 細胞外液 {n_dbl+n_std+n_pln}袋 / Alb {n_dbl*2+n_std}本 / 総液量 {actual_replacement_vol}mL")

st.divider()

# --- 画像 ---
# 回路図 (必要なら)
if os.path.exists("circuit.png") or os.path.exists("circuit.jpg"):
    with st.expander("回路構成図を見る"):
        img_path = "circuit.png" if os.path.exists("circuit.png") else "circuit.jpg"
        st.image(img_path, caption="SePE 回路構成図")

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
        * 膜を通過して廃棄される液体中のアルブミン濃度です。本システムでは $C_{Plasma} \times SC$ で推定しています。
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
    * **小川の式 (Ogawa's Formula):** 日本人成人の体格に適合した循環血液量(BV)推定式です。
      $$ BV(L) = 0.16874 \times Height(m) + 0.05986 \times Weight(kg) - 0.0305 $$
    * **血漿量:** $EPV = BV \times (1 - Hct/100)$

    **B. 必要な血漿処理量 (Required PV)**
    * ワンコンパートメントモデル（対数減衰モデル）に基づき算出します。
      $$ V = \frac{- \ln(1 - R) \times EPV}{SC_{pathogen}} $$
      ($R$: 除去目標率, $V$: 処理量)
    """)
