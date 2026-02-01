import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request  # 標準ライブラリでダウンロード機能を使用

# --- フォント設定 (完全自動ダウンロード版) ---
def setup_japanese_font():
    # フォントファイル名
    font_filename = "NotoSansJP-Regular.ttf"
    
    # ファイルがなければダウンロードする
    if not os.path.exists(font_filename):
        # GitHubのRawデータへの直接リンク
        url = "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
        try:
            # 標準ライブラリでダウンロード (ライブラリ追加不要)
            urllib.request.urlretrieve(url, font_filename)
        except Exception as e:
            # 万が一失敗した場合はエラーを表示せず英語フォントで進める
            pass

    # フォントをmatplotlibに登録して適用
    if os.path.exists(font_filename):
        fm.fontManager.addfont(font_filename)
        plt.rc('font', family='Noto Sans JP')
    else:
        plt.rc('font', family='sans-serif')

# アプリ起動時に実行
setup_japanese_font()

# --- ここからメイン処理 ---

# ページ設定
st.set_page_config(page_title="SePE Simulation (EC-4A10c)", layout="wide")

# --- サイドバー：パラメータ入力 ---
st.sidebar.header("患者・治療パラメータ設定")

# 患者情報
st.sidebar.subheader("患者情報")

# 身長入力の任意化
use_height_formula = st.sidebar.checkbox("身長を入力して計算（小川の式）", value=True)

if use_height_formula:
    height = st.sidebar.number_input("身長 (cm)", value=170.0, step=0.1)
else:
    height = None
    st.sidebar.caption("身長入力なし：簡易式 (70mL/kg) を使用します")

weight = st.sidebar.number_input("体重 (kg)", value=65.0, step=0.1)
hct = st.sidebar.number_input("血中ヘマトクリット値 (%)", value=30.0, step=0.1)
alb_initial = st.sidebar.number_input("血清アルブミン値 (g/dL)", value=3.5, step=0.1)

# 治療目標
st.sidebar.subheader("治療目標")
target_removal = st.sidebar.slider("病因物質の除去目標 (%)", 30, 95, 60, step=5)
qp = st.sidebar.number_input("血漿流量 QP (mL/min)", value=30.0, step=5.0)

# 膜特性
st.sidebar.subheader("膜特性設定 (Evacure EC-4A10c)")
st.sidebar.markdown("<small>※in vivoでの目詰まりや安全域を考慮して調整</small>", unsafe_allow_html=True)
sc_pathogen = st.sidebar.slider("病因物質のふるい係数 (SC)", 0.0, 1.0, 0.90, 0.01)
sc_albumin = st.sidebar.slider("アルブミンのふるい係数 (SC)", 0.0, 1.0, 0.65, 0.01)

# --- 計算ロジック ---

def calculate_epv(bv, hct):
    return bv * (1 - hct / 100)

def calculate_required_pv(target_removal_percent, epv, sc):
    target_ratio = target_removal_percent / 100.0
    if sc == 0: return 0
    v = -np.log(1 - target_ratio) * epv / sc
    return v

# 1. 循環血液量(BV)の計算
if use_height_formula and height is not None:
    # 小川の式 (Ogawa's Formula): BV(mL)換算
    # 文献値: BV(L) = 0.16874*H(m) + 0.05986*W(kg) - 0.0305
    bv_calc = (0.16874 * height + 0.05986 * weight - 0.0305) * 1000
    bv_method = "小川の式"
else:
    # 簡易式
    bv_calc = weight * 70
    bv_method = "簡易式 (70mL/kg)"

# 2. 循環血漿量(EPV)
epv = calculate_epv(bv_calc, hct)

# 3. 必要処理量
required_pv = calculate_required_pv(target_removal, epv, sc_pathogen)

# 4. 治療時間
treatment_time_min = required_pv / qp if qp > 0 else 0

# 5. 補充液計算 (20%Alb 50ml + Physio 140ml = 190ml/Set)
vol_per_set = 50 + 140
num_sets = required_pv / vol_per_set
num_sets_ceil = np.ceil(num_sets)
actual_replacement_vol = num_sets_ceil * vol_per_set
supplied_albumin_g = num_sets_ceil * 10

# 6. アルブミン予測喪失量
total_alb_body_g = (epv / 100) * alb_initial
alb_remaining_ratio = np.exp(-required_pv * sc_albumin / epv)
predicted_alb_loss_g = total_alb_body_g * (1 - alb_remaining_ratio)

# --- メイン画面表示 ---

st.title("選択的血漿交換 (SePE) シミュレーション")

# --- 結果表示エリア ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("予測循環血漿量 (EPV)", f"{int(epv)} mL", help=f"算出根拠: {bv_method}")
    st.metric("必要血漿処理量", f"{int(required_pv)} mL", f"{required_pv/epv:.2f} x EPV")

with col2:
    st.metric("治療時間", f"{int(treatment_time_min)} 分", f"{treatment_time_min/60:.1f} 時間")
    st.metric("血漿流量 (QP)", f"{qp} mL/min")

with col3:
    st.metric("必要補充液セット数", f"{int(num_sets_ceil)} セット", "20%Alb(50mL) + Physio(140mL)")
    st.metric("総補充液量", f"{int(actual_replacement_vol)} mL")

with col4:
    st.metric("予測喪失アルブミン", f"{predicted_alb_loss_g:.1f} g", help="排液中に失われる推定アルブミン総量")
    st.metric("補充アルブミン量", f"{int(supplied_albumin_g)} g", f"差引: {int(supplied_albumin_g - predicted_alb_loss_g)}g")

st.divider()

# --- 回路図と設定の表示 ---
st.subheader("治療回路・設定概要")
c_img, c_info = st.columns([1, 1])

with c_img:
    # 画像表示 (circuit.png があれば表示)
    if os.path.exists("circuit.png"):
        st.image("circuit.png", caption="SePE 回路構成図", use_container_width=True)
    elif os.path.exists("circuit.jpg"):
        st.image("circuit.jpg", caption="SePE 回路構成図", use_container_width=True)
    else:
        st.info("※回路図画像 (circuit.png) がアップロードされていません")

with c_info:
    st.markdown("### 💉 治療設定サマリー")
    st.info(f"""
    **1. 流量設定**
    * **血漿流量 (QP):** {qp} mL/min
    
    **2. 補充液組成 (1セットあたり)**
    * **ベース:** フィジオ140 (140mL)
    * **製剤:** 20% アルブミン製剤 (50mL/10g)
    * **合計:** 190 mL (アルブミン濃度 約5.3%)
    
    **3. 準備量**
    * **必要セット数:** {int(num_sets_ceil)} セット
    * **総予定補充量:** {int(actual_replacement_vol)} mL
    """)

st.divider()

# --- グラフ描画 ---
st.subheader("治療経過シミュレーション")

# データ生成
v_process = np.linspace(0, required_pv * 1.2, 100)
pathogen_remaining = np.exp(-v_process * sc_pathogen / epv) * 100
alb_loss_curve = total_alb_body_g * (1 - np.exp(-v_process * sc_albumin / epv))

fig, ax1 = plt.subplots(figsize=(10, 5))

# --- 軸1: 病因物質 (左軸・赤) ---
color_1 = 'tab:red'
ax1.set_xlabel('血漿処理量 (mL)', fontsize=12)
ax1.set_ylabel('【赤】病因物質 残存率 (%)', color=color_1, fontsize=12, fontweight='bold')
line1 = ax1.plot(v_process, pathogen_remaining, color=color_1, linewidth=3, label='病因物質 残存率 (%)')
ax1.tick_params(axis='y', labelcolor=color_1)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.set_ylim(0, 105)

# 目標点のプロットとテキスト
ax1.scatter([required_pv], [100 - target_removal], color='red', s=100, zorder=5)
ax1.text(required_pv, 100 - target_removal + 10, 
         f' 目標達成点\n {int(required_pv)}mL処理\n 残存{100-target_removal}%', 
         color='red', fontweight='bold', ha='center',
         bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))

# --- 軸2: アルブミン喪失 (右軸・青) ---
ax2 = ax1.twinx()
color_2 = 'tab:blue'
ax2.set_ylabel('【青】累積アルブミン喪失量 (g)', color=color_2, fontsize=12, fontweight='bold')
line2 = ax2.plot(v_process, alb_loss_curve, color=color_2, linestyle='--', linewidth=2.5, label='予測アルブミン喪失量 (g)')
ax2.tick_params(axis='y', labelcolor=color_2)
ax2.set_ylim(0, max(alb_loss_curve)*1.3)

# 凡例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=10)

st.pyplot(fig)

st.markdown("""
> **グラフの見方:**
> * **<span style='color:red'>赤線（左軸）</span>**: 治療が進むにつれて病因物質が減っていく様子（残存率）を示します。
> * **<span style='color:blue'>青点線（右軸）</span>**: 治療が進むにつれて体外へ捨てられるアルブミンの総量（g）が増えていく様子を示します。
> * **横軸**: 血漿処理量（mL）です。右に行くほど治療が進んでいることを意味します。
""", unsafe_allow_html=True)

# --- 用語解説 ---
st.divider()
st.header("用語解説・計算根拠")

with st.expander("用語の説明 (クリックして展開)"):
    st.markdown("""
    * **SePE (Selective Plasma Exchange):** 選択的血漿交換療法。
    * **ふるい係数 (SC):** 膜をどれだけ物質が通過しやすいかを示す指標です。
    * **小川の式:** 日本人の体格に基づいた循環血液量(BV)の推定式です。
    """)

with st.expander("計算式とロジック (クリックして展開)"):
    st.markdown(r"""
    ### 1. 予測循環血漿量 (EPV)
    身長入力がある場合は**小川の式**、ない場合は簡易式($70mL/kg$)を用いてBVを算出します。
    $$ EPV = BV \times (1 - \frac{Hct}{100}) $$

    **(参考) 小川の式:** $$ BV(L) = 0.16874 \times Height(m) + 0.05986 \times Weight(kg) - 0.0305 $$

    ### 2. 必要な血漿処理量 (Required PV)
    $$ V = \frac{- \ln(1 - R) \times EPV}{SC_{pathogen}} $$

    ### 3. アルブミン予測喪失量
    $$ Loss_{Alb} = Total_{Alb} \times (1 - e^{-\frac{V \times SC_{alb}}{EPV}}) $$
    """)
