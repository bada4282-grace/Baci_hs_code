import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 페이지 설정
st.set_page_config(page_title="Korea Trade Dashboard", layout="wide")
st.title("🇰🇷 대한민국 전기기기(HS 85) 수출 분석 대시보드")

# 2. 데이터 로드 및 전처리 (캐싱 적용)
@st.cache_data
def load_and_preprocess():
    # 데이터 읽기
    df = pd.read_csv('./baci_korea_85_only.csv')
    countries = pd.read_csv('./country_codes_V202501.csv', encoding='utf-8')
    
    # 튀르키예 이름 수정
    countries['country_name'] = countries['country_name'].str.replace('TÃ¼rkiye', 'Türkiye')
    
    # 컬럼명 변경 및 병합
    df.rename(columns={'j': 'country_code'}, inplace=True)
    baci_final = pd.merge(df, countries, on='country_code', how='left')
    
    # 연도 랜덤 할당 및 시드 고정 (재현성)
    np.random.seed(42)
    year_list = [2021, 2022, 2023]
    baci_final["t"] = np.random.choice(year_list, size=len(baci_final))
    
    return baci_final

baci_final = load_and_preprocess()

# 3. 사이드바 필터
st.sidebar.header("📊 분석 옵션")
selected_years = st.sidebar.multiselect("연도 선택", options=[2021, 2022, 2023], default=[2021, 2022, 2023])
top_n = st.sidebar.slider("상위 국가 수", 5, 20, 5)

# 데이터 필터링
filtered_data = baci_final[baci_final['t'].isin(selected_years)]

# 4. 시각화 스타일 설정 (폰트 깨짐 방지)
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")

# 5. 대시보드 레이아웃 (Tabs)
tab1, tab2 = st.tabs(["📈 연도별/국가별 추이", "📍 시장 분포 분석"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("연도별 수출 총액")
        yearly_v = filtered_data.groupby('t')['v'].sum().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=yearly_v, x='t', y='v', palette='viridis', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader(f"상위 {top_n}개국 수출 트렌드")
        top_list = baci_final.groupby('country_name')['v'].sum().sort_values(ascending=False).head(top_n).index.tolist()
        trend_data = filtered_data[filtered_data['country_name'].isin(top_list)].groupby(['country_name', 't'])['v'].sum().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=trend_data, x='country_name', y='v', hue='t', palette='coolwarm', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

with tab2:
    st.subheader("수출액 vs 물량 분포 (Scatter Plot)")
    dist_data = filtered_data.groupby('country_name').agg({'v':'sum', 'q':'sum'}).reset_index()
    dist_data['unit_price'] = dist_data['v'] / dist_data['q'].replace(0, np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=dist_data, x='v', y='q', size='unit_price', hue='unit_price', 
                    sizes=(50, 1000), palette='magma', alpha=0.6, ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    st.pyplot(fig)

# 데이터 표 출력
st.markdown("### 📄 상세 데이터 내역")
st.dataframe(filtered_data.head(100))

