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

# 공통 레이블 정의 (CTO의 깔끔한 변수 관리)
LABEL_YEAR = "Year"
LABEL_COUNTRY = "Importing Country"
LABEL_VALUE = "Export Value (1,000 USD)"
LABEL_QTY = "Quantity"
LABEL_UNIT_PRICE = "Unit Price"

with tab1:
    col1, col2 = st.columns(2)
    
    # [왼쪽 차트] 연도별 수출 총액 (연도는 짧으므로 기본 유지)
    with col1:
        st.subheader("연도별 수출 총액 변화")
        yearly_v = filtered_data.groupby('t')['v'].sum().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=yearly_v, x='t', y='v', palette='viridis', ax=ax)
        ax.set_xlabel(LABEL_YEAR, fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel(LABEL_VALUE, fontsize=12, fontweight='bold')
        st.pyplot(fig)
        
    # [오른쪽 차트] 상위 국가별 트렌드 (국가명이 길어지는 구간)
    with col2:
        st.subheader(f"상위 {top_n}개국 수출 트렌드")
        top_list = baci_final.groupby('country_name')['v'].sum().sort_values(ascending=False).head(top_n).index.tolist()
        trend_data = filtered_data[filtered_data['country_name'].isin(top_list)].groupby(['country_name', 't'])['v'].sum().reset_index()
        
        # 💡 [CTO의 팁] 너무 긴 국가 이름은 15자까지만 보여주고 뒤는 '..'로 생략
        trend_data['country_name_short'] = trend_data['country_name'].apply(lambda x: x[:15] + '..' if len(x) > 15 else x)
        
        fig, ax = plt.subplots()
        # 생략된 이름을 x축에 사용
        sns.barplot(data=trend_data, x='country_name_short', y='v', hue='t', palette='coolwarm', ax=ax)
        
        # 🎯 [핵심 수정] rotation과 함께 ha='right'를 써야 막대 아래 딱 붙습니다.
        plt.xticks(rotation=45, ha='right', fontsize=10) 
        
        # 🎯 [핵심 수정] labelpad를 조절하여 축 이름이 너무 내려가지 않게 고정합니다.
        ax.set_xlabel(LABEL_COUNTRY, fontsize=12, fontweight='bold', labelpad=0) 
        ax.set_ylabel(LABEL_VALUE, fontsize=12, fontweight='bold')
        
        # 범례 위치를 그래프 안쪽 적절한 곳으로 이동 (밖으로 나가서 잘리지 않게)
        ax.legend(title=LABEL_YEAR, title_fontsize=10, fontsize=9, loc='upper right')
        
        plt.tight_layout() # 전체적인 여백 자동 최적화
        st.pyplot(fig)

with tab2:
    st.subheader("수출액 vs 물량 분포 (Scatter Plot)")
    st.caption("💡 원의 크기와 색상은 단가(Unit Price)를 나타냅니다. 우상단으로 갈수록 규모가 큰 시장입니다.")

    # 데이터 집계
    dist_data = filtered_data.groupby('country_name').agg({'v':'sum', 'q':'sum'}).reset_index()
    # 단가 계산 (ZeroDivisionError 방지)
    dist_data['unit_price'] = dist_data['v'] / dist_data['q'].replace(0, np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 스캐터 플롯 그리기
    sns.scatterplot(data=dist_data, x='v', y='q', size='unit_price', hue='unit_price', 
                    sizes=(50, 1000), palette='magma', alpha=0.7, ax=ax)
    
    # 로그 스케일 적용
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 🏷️ 레이블 변경 적용 (로그 스케일 표시 추가)
    ax.set_xlabel(f"{LABEL_VALUE} [Log Scale]", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"{LABEL_QTY} [Log Scale]", fontsize=12, fontweight='bold')
    
    # 범례 제목 변경 (Seaborn이 자동으로 생성한 범례 가져오기)
    if ax.get_legend() is not None:
        ax.get_legend().set_title(LABEL_UNIT_PRICE)
        plt.setp(ax.get_legend().get_texts(), fontsize='9') # 범례 텍스트 크기 조절
        plt.setp(ax.get_legend().get_title(), fontsize='10') # 범례 제목 크기 조절

    ax.tick_params(axis='both', which='major', labelsize=10)
    st.pyplot(fig)

# 6. 데이터 표 출력
st.markdown("---")
st.subheader("📄 필터링된 상세 데이터")
st.dataframe(filtered_data.head(100), use_container_width=True)