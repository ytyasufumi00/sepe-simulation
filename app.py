import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request

# --- フォント設定 (修正・完全版) ---
def setup_japanese_font():
    # キャッシュ対策でファイル名を変更
    font_filename = "NotoSansJP-Regular_v2.ttf"
    
    # ファイルがなければ正しいURLから取得
    if not os.path.exists(font_filename):
        # 修正: raw.githubusercontent.com を使用
        url = "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
        try:
            with urllib.request.urlopen(url) as response, open(font_filename, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            st.error(f"フォントダウンロードエラー: {e}")

    # フォント設定
    if os.path.exists(font_filename):
        try:
            fm.fontManager.addfont(font_filename)
            plt.rc('font', family='Noto Sans JP')
        except Exception as e:
            st.warning(f"フォント適用エラー: {e}")
    else:
        # 万が一失敗した場合は英語フォント
        plt.rc('font', family='sans-serif')

setup_japanese_font()

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

# --- 計算 ---
if use_height_formula and height is not None:
    # 小川の式 (mL換算)
    bv_calc = (0.16874 * height + 0.05986 * weight - 0.0305) * 1000
    bv_method = "小川の式"
else:
    bv_calc = weight * 70
    bv_method = "簡易式 (70mL/kg)"

epv = bv_calc * (1 - hct / 100)

if sc_pathogen > 0:
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

# --- 表示 ---
st.title("選択的血漿交換 (SePE) シミュレーション")

col1, col2, col3, col4 = st.columns(4)
col1.metric("予測循環血漿量 (EPV)", f"{int(epv)} mL", bv_method)
col2.metric("治療時間", f"{int(treatment_time_min)} 分", f"QP: {qp} mL/min")
col3.metric("必要補充液セット数", f"{int(num_sets_ceil)} セット", f"総量: {int(actual_replacement_vol)} mL")
col4.metric("予測喪失アルブミン", f"{predicted_alb_loss_g:.1f} g", f"補充: {int(supplied_albumin_g)} g")

st.divider()

c_img, c_info = st.columns([1, 1])
with c_img:
    if os.path.exists("circuit.png"):
        st.image("circuit.png", caption="SePE 回路構成図", use_container_width=True)
    elif os.path.exists("circuit.jpg"):
        st.image("circuit.jpg", caption="SePE 回路構成図", use_container_width=True)
    else:
        st.info("※回路図画像がアップロードされていません")

with c_info:
    st.info(f"""
    **補充液構成 (1セット):**
    * 20%アルブミン (50mL)
    * フィジオ140 (140mL)
    * 合計 190mL (Alb濃度 約5.3%)
    """)

st.divider()

# --- グラフ ---
v_process = np.linspace(0, required_pv * 1.2, 100)
pathogen_remaining = np.exp(-v_process * sc_pathogen / epv) * 100
alb_loss_curve = total_alb_body_g * (1 - np.exp(-v_process * sc_albumin / epv))

fig, ax1 = plt.subplots(figsize=(10, 5))
color_1 = 'tab:red'
ax1.set_xlabel('血漿処理量 (mL)', fontsize=12)
ax1.set_ylabel('病因物質 残存率 (%)', color=color_1, fontweight='bold', fontsize=12)
line1 = ax1.plot(v_process, pathogen_remaining, color=color_1, linewidth=3, label='病因物質残存率')
ax1.tick_params(axis='y', labelcolor=color_1)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_ylim(0, 105)

ax1.scatter([required_pv], [100 - target_removal], color='red', s=100, zorder=5)
ax1.text(required_pv, 100 - target_removal + 10, f'目標達成\n{int(required_pv)}mL', color='red', ha='center', fontweight='bold',
         bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))

ax2 = ax1.twinx()
color_2 = 'tab:blue'
ax2.set_ylabel('累積アルブミン喪失量 (g)', color=color_2, fontweight='bold', fontsize=12)
line2 = ax2.plot(v_process, alb_loss_curve, color=color_2, linestyle='--', linewidth=2.5, label='Alb喪失量')
ax2.tick_params(axis='y', labelcolor=color_2)
ax2.set_ylim(0, max(alb_loss_curve)*1.3)

# 凡例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=10)

st.pyplot(fig)

# --- 用語解説 ---
st.divider()
st.header("用語解説・計算根拠")
with st.expander("用語の説明・計算式 (クリックして展開)"):
    st.markdown(r"""
    * **SePE (Selective Plasma Exchange):** 選択的血漿交換療法。
    * **ふるい係数 (SC):** 膜透過性。
    * **小川の式:** $BV(L) = 0.16874 \times H(m) + 0.05986 \times W(kg) - 0.0305$
    """)
