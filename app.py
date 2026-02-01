import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ページ設定
st.set_page_config(page_title="SePE Simulation (EC-4A10c)", layout="wide")

# --- サイドバー：パラメータ入力 ---
st.sidebar.header("患者・治療パラメータ設定")

# 患者情報
st.sidebar.subheader("患者情報")
height = st.sidebar.number_input("身長 (cm)", value=170.0, step=0.1)
weight = st.sidebar.number_input("体重 (kg)", value=65.0, step=0.1)
sex = st.sidebar.radio("性別", ["男性", "女性"])
hct = st.sidebar.number_input("血中ヘマトクリット値 (%)", value=30.0, step=0.1)
alb_initial = st.sidebar.number_input("血清アルブミン値 (g/dL)", value=3.5, step=0.1)

# 治療目標
st.sidebar.subheader("治療目標")
target_removal = st.sidebar.slider("病因物質の除去目標 (%)", 30, 95, 60, step=5)
qp = st.sidebar.number_input("血漿流量 QP (mL/min)", value=30.0, step=5.0)

# 膜特性 (Evacure EC-4A10c)
st.sidebar.subheader("膜特性設定 (Evacure EC-4A10c)")
st.sidebar.markdown("""
<small>※カタログ値(in vitro牛血)に対し、in vivoでの目詰まり(Secondary membrane)や安全域を考慮して調整してください。</small>
""", unsafe_allow_html=True)

# 安全域の説明: in vitroよりin vivoでは低くなる傾向があるため、デフォルトを調整
sc_pathogen = st.sidebar.slider(
    "病因物質のふるい係数 (SC)", 
    0.0, 1.0, 0.90, 0.01, 
    help="カタログ値はIgGなどでほぼ1.0ですが、安全域（除去効率を過大評価しない）のため0.90を推奨初期値としています。"
)

sc_albumin = st.sidebar.slider(
    "アルブミンのふるい係数 (SC)", 
    0.0, 1.0, 0.65, 0.01, 
    help="EC-4A10cのカタログ値周辺ですが、アルブミン喪失のリスク管理のため、やや高めの見積もりも考慮して操作してください。"
)

# --- 計算ロジック ---

def calculate_blood_volume(h, w, s):
    # Nadlerの式を使用
    h_m = h / 100
    if s == "男性":
        bv = 0.3669 * (h_m ** 3) + 0.03219 * w + 0.6041
    else:
        bv = 0.3561 * (h_m ** 3) + 0.03308 * w + 0.1833
    return bv * 1000 # L to mL

def calculate_epv(bv, hct):
    return bv * (1 - hct / 100)

def calculate_required_pv(target_removal_percent, epv, sc):
    # C/Co = exp(-V * SC / EPV)
    # Target Removal = 1 - C/Co
    # 1 - Target = exp(-V * SC / EPV)
    # ln(1 - Target) = -V * SC / EPV
    # V = - ln(1 - Target) * EPV / SC
    target_ratio = target_removal_percent / 100.0
    if sc == 0:
        return 0
    v = -np.log(1 - target_ratio) * epv / sc
    return v

# 計算実行
bv = calculate_blood_volume(height, weight, sex)
epv = calculate_epv(bv, hct)
required_pv = calculate_required_pv(target_removal, epv, sc_pathogen)

# 治療時間
treatment_time_min = required_pv / qp if qp > 0 else 0
treatment_time_hr = treatment_time_min / 60

# 補充液計算 (20%アルブミン50ml + フィジオ140 = 190ml/セット と仮定)
# 構成: 20% Alb 50ml (10g Alb) + Physio 140ml = 190ml Total
# 濃度: 10g / 190ml ≈ 5.26%
vol_per_set = 50 + 140
num_sets = required_pv / vol_per_set
num_sets_ceil = np.ceil(num_sets) # 切り上げ（本数計算のため）

required_alb_vials = num_sets_ceil
required_physio_bags = num_sets_ceil
actual_replacement_vol = num_sets_ceil * vol_per_set
supplied_albumin_g = required_alb_vials * 10 # 1バイアル10g

# --- メイン画面表示 ---

st.title("選択的血漿交換 (SePE) シミュレーション")
st.markdown("**対象膜**: Kuraray Evacure EC-4A10c (想定)")

# 結果表示エリア（カード風レイアウト）
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("予測循環血漿量 (EPV)", f"{int(epv)} mL", help="Nadlerの式より算出")
    st.metric("必要血漿処理量", f"{int(required_pv)} mL", f"{required_pv/epv:.2f} x EPV")

with col2:
    st.metric("治療時間", f"{int(treatment_time_min)} 分", f"{treatment_time_hr:.1f} 時間")
    st.metric("血漿流量 (QP)", f"{qp} mL/min")

with col3:
    st.metric("必要補充液セット数", f"{int(num_sets_ceil)} セット", help="1セット=20%Alb 50ml + Physio 140ml")
    st.metric("総補充液量", f"{int(actual_replacement_vol)} mL", help="端数切り上げによる実補充量")

st.info(f"""
**補充液構成の詳細:**
* **20%アルブミン (50mL製剤):** {int(required_alb_vials)} 本
* **フィジオ140:** {int(required_physio_bags)} 本 (または同等の電解質液)
* **補充液中アルブミン濃度:** 約 5.3%
* **総補充アルブミン量:** {int(supplied_albumin_g)} g
""")

st.divider()

# --- グラフ描画 ---
st.subheader("治療経過シミュレーション")

# データ生成
v_process = np.linspace(0, required_pv * 1.2, 100) # 目標の1.2倍まで描画
# 病因物質残存率 C/C0 = exp(-V * SC / EPV)
pathogen_remaining = np.exp(-v_process * sc_pathogen / epv) * 100
# アルブミン残存率 (体内プールのみの減衰、補充なしの場合)
albumin_remaining_kinetics = np.exp(-v_process * sc_albumin / epv) * 100
# アルブミン総喪失量 (g) = 初期総量 * (1 - 残存率) 
# ※単純なワンコンパートメントモデルでの推定
total_alb_body = epv * alb_initial / 100 # g
alb_loss_g = total_alb_body * (1 - albumin_remaining_kinetics / 100)

fig, ax1 = plt.subplots(figsize=(10, 6))

# 病因物質除去曲線
ax1.plot(v_process, pathogen_remaining, 'r-', linewidth=2, label='病因物質 残存率 (%)')
ax1.set_xlabel('血漿処理量 (mL)')
ax1.set_ylabel('残存率 (%)', color='black')
ax1.grid(True, which='both', linestyle='--', alpha=0.6)
ax1.set_ylim(0, 105)

# 目標点プロット
ax1.scatter([required_pv], [100 - target_removal], color='red', zorder=5)
ax1.text(required_pv, 100 - target_removal + 2, f' 目標: {target_removal}%除去\n ({int(required_pv)}mL)', color='red', fontweight='bold')

# アルブミン喪失曲線 (第2軸)
ax2 = ax1.twinx()
ax2.plot(v_process, alb_loss_g, 'b--', linewidth=2, label='累積アルブミン喪失量 (g)')
ax2.set_ylabel('累積アルブミン喪失量 (g)', color='blue')
ax2.set_ylim(0, max(alb_loss_g)*1.2)

# 凡例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

st.pyplot(fig)

st.warning("""
**グラフの解釈に関する注意:** 上記の「アルブミン喪失量」は、排液中に捨てられるアルブミンの推定総量です。
実際の治療では補充液（アルブミン製剤）が投与されるため、血清アルブミン濃度の推移はこのグラフ通りにはなりません。
本シミュレーションでは **「約 {0} g のアルブミンが体外へ捨てられるため、それに見合う補充（今回の計算では {1} g）が必要である」** と解釈してください。
""".format(int(alb_loss_g[np.searchsorted(v_process, required_pv)]), int(supplied_albumin_g)))

st.divider()

# --- 用語解説と根拠 ---
st.header("用語解説・計算根拠")

with st.expander("用語の説明 (クリックして展開)"):
    st.markdown("""
    * **SePE (Selective Plasma Exchange):** 選択的血漿交換療法。特定の分子量以下の物質（アルブミンなど）はなるべく残し、それより大きい病因物質（免疫複合体など）を除去する方法です。
    * **ふるい係数 (Sieving Coefficient, SC):** 膜をどれだけ物質が通過しやすいかを示す指標です。0は完全に阻止、1は完全に通過を意味します。
        * $SC = \dfrac{C_{Filtrate}}{C_{Plasma}}$
    * **阻止率 (Rejection Coefficient, RC):** 膜がどれだけ物質を阻止するかを示す指標です。
        * $RC = 1 - SC$
    * **QP (Plasma Flow Rate):** 血漿分離器へ流れる血漿の流量です。治療効率や膜への負荷（膜面圧）に関与します。
    * **EC-4A10c:** エバキュアー（クラレ社製）のSePE用血漿成分分離器。アルブミンのSCが0.6-0.7程度、IgGなどのグロブリン系のSCが1.0近くになるよう設計されています。
    """)

with st.expander("計算式とロジック (クリックして展開)"):
    st.markdown(r"""
    ### 1. 予測循環血漿量 (EPV)
    Nadlerの式を用いて総血液量(TBV)を算出し、ヘマトクリット(Hct)で血漿量を求めます。
    
    $$ EPV = TBV \times (1 - \frac{Hct}{100}) $$

    ### 2. 必要な血漿処理量 (Required PV)
    ワンコンパートメントモデルに基づき計算します。
    病因物質の除去目標を $R$ ($0 \le R < 1$)、目標残存率を $1-R$ とすると：
    
    $$ \frac{C}{C_0} = e^{-\frac{V \times SC}{EPV}} $$
    
    これより必要な処理量 $V$ は以下の通り逆算されます。
    
    $$ V = \frac{- \ln(1 - R) \times EPV}{SC_{pathogen}} $$

    ### 3. 安全域 (Safety Margin) について
    カタログ値（in vitro, 牛血）は理想的な条件下でのデータです。実際の臨床（in vivo, 人血）では以下の理由により、性能が変動します。
    * **Secondary Membrane (二次膜)の形成:** タンパク質が膜表面に付着し、実効孔径が小さくなることで、アルブミンや病因物質の透過率(SC)が時間経過とともに低下します。
    * **安全域の設定:**
        * **病因物質:** SCを低めに見積もることで、「実際には除去しきれていない」リスクを防ぐため、処理量を多めに計算するようにしています。
        * **アルブミン:** SCを高めに見積もる（またはスライダーで確認する）ことで、アルブミン喪失を過小評価せず、十分な補充量を確保できるようにしています。
    """)

st.caption("Disclaimer: 本システムは学習・シミュレーション用であり、臨床判断を代替するものではありません。実治療においては、患者の状態や各施設のプロトコルに従ってください。")