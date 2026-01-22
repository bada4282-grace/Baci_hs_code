import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('./baci_korea_85_only.csv')
countries = pd.read_csv('./country_codes_V202501.csv', encoding='cp949')
products = pd.read_csv('./data/product_codes_HS22_V202501.csv')
print(df.head())
print(df.info())
print(df.columns)

# 만약 튀르키예 이름이 이미 'TÃ¼rkiye'로 깨져 있다면 강제로 수정해주는 로직 (CTO의 디테일)
countries['country_name'] = countries['country_name'].str.replace('TÃ¼rkiye', 'Türkiye')

# 폰트 설정 (가장 중요)
# 'DejaVu Sans'는 전 세계 대부분의 특수 문자를 깨짐 없이 지원하는 표준 폰트입니다.
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

""" 파일 "baci_korea_85_only.csv" 생성 """
# baci_korea_85 = df[df["k"]==852352].copy()
# print(len(baci_korea_85))
# print(baci_korea_85.head())
# baci_korea_85.to_csv("./baci_korea_85_only.csv", index = False)
""" 열 이름 뜻 """
# t: year
# i: exporter
# j: importer
# k: product (HS code 등 품목 코드)
# v: value (수출 금액)
# q: quantity (수출 물량)

# 컬럼명 변경 (j -> country_code)
df.rename(columns={'j': 'country_code'}, inplace=True)
print(df.head())

# (country_code)를 기준으로 병합
""" how='left'의 의미: 우리나라의 수출 실적(df)은 단 한 줄도 버리지 말고 다 챙기되, countries 파일에 이름 정보가 있는 것들만 '옆으로 붙이라' """
baci_final = pd.merge(df , countries, on = 'country_code', how = "left")
print(baci_final.head())

# 2023년 하나밖에 없는 데이터, 연습을 위해 2023년을 랜덤하게 2021,2022,2023으로 바꾸기
year = [2021,2022,2023]
baci_final["t"] = np.random.choice(year, size=len(baci_final))  # size= : 몇개뽑을지 
print(baci_final.head())

# 연도별/국가별 수출량, 증가량, 증감율, scatter 활용해서 어디에 많이 분포되어있는지 등등 여러 차트 만들기
# 💠1. 연도별 수출금액(v) 총합 구하기
# 't'(연도) 컬럼으로 묶고 'v'(수출액)의 합계를 계산합니다.
yearly_export = baci_final.groupby('t')['v'].sum().reset_index()

# 시각화 스타일 설정 (세련된 테마 적용)
sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")
plt.figure(figsize=(12, 7)) # 시원한 느낌을 위해 가로를 더 넓게 설정

# 막대 그래프 생성
# alpha=0.9: 약간의 투명도를 주어 부드러운 느낌
# edgecolor='black', linewidth=1: 막대 테두리를 또렷하게 마감
ax = sns.barplot(data=yearly_export, x='t', y='v', palette='viridis', 
                 alpha=0.9, edgecolor='black', linewidth=1)

# 차트 제목 및 축 라벨 설정 (폰트 및 크기 강화)
plt.title('Annual Total Export Value (2021-2023)', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=16, fontweight='bold', labelpad=15)
plt.ylabel('Total Value (1,000 USD)', fontsize=16, fontweight='bold', labelpad=15)

# Y축 천 단위 콤마 표시 (가독성 향상)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# 수치 표시 (막대 위에 금액 표시) - 위치 및 폰트 조정
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                fontsize=14, fontweight='bold', color='black', 
                xytext=(0, 10), # 막대 상단에서 10포인트 위로 띄움
                textcoords='offset points')

plt.tight_layout() # 여백 자동 조정
plt.savefig('yearly_export_total_refined.png', dpi=300) # 고해상도 저장
plt.show()

# 💠2. 연도별/국가별 데이터 집계
# 국가별, 연도별 수출액 합계
country_year_v = baci_final.groupby(['country_name', 't'])['v'].sum().unstack(fill_value=0)

# 증가량 및 증감율 계산 (2022년 대비 2023년)
country_year_v['growth_amt'] = country_year_v[2023] - country_year_v[2022]
country_year_v['growth_rate'] = (country_year_v['growth_amt'] / country_year_v[2022].replace(0, np.nan)) * 100

# ---------------------------------------------------------
# 차트 1: 상위 5개국 연도별 수출 트렌드 (Grouped Bar)
# ---------------------------------------------------------

# 1. 상위 5개 국가가 어디인지 먼저 계산 (차트는 안 그림)
top_5_list = baci_final.groupby('country_name')['v'].sum().sort_values(ascending=False).head(5).index.tolist()

# 2. 상위 5개국 데이터만 필터링해서 연도별 합계 계산
top_5_trend = baci_final[baci_final['country_name'].isin(top_5_list)].groupby(['country_name', 't'])['v'].sum().reset_index()

# 3. 차트 그리기
plt.figure(figsize=(12, 7))

# 튀르키예 등 특수문자가 포함된 경우를 위해 색상 팔레트와 폰트 재확인
ax = sns.barplot(data=top_5_trend, x='country_name', y='v', hue='t', palette='coolwarm', edgecolor='black')

# 제목 및 라벨
plt.title('Export Trends for Top 5 Countries (HS 85)', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Country Name', fontsize=15, labelpad=10)
plt.ylabel('Export Value (1,000 USD)', fontsize=15, labelpad=10)

# 범례(Legend) 이쁘게 정리
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('top_5_trends_fixed.png', dpi=300)


# ---------------------------------------------------------
# 차트 2: 수출액 vs 물량 분포 (Scatter Plot)
# 어디에 많이 분포되어 있는지 시각화
# ---------------------------------------------------------
country_dist = baci_final.groupby('country_name').agg({'v':'sum', 'q':'sum'}).reset_index()
country_dist['unit_price'] = country_dist['v'] / country_dist['q'].replace(0, np.nan)

plt.figure(figsize=(12, 8))
# 원의 크기를 단가(unit_price)로 설정하여 부가가치가 높은 시장을 식별
sns.scatterplot(data=country_dist, x='v', y='q', size='unit_price', hue='unit_price', 
                sizes=(50, 1000), palette='magma', alpha=0.6)
plt.xscale('log') # 값의 편차가 커서 로그 스케일 적용
plt.yscale('log')
plt.title('Market Distribution: Value vs. Quantity (Log Scale)', fontweight='bold')
plt.xlabel('Total Value (1,000 USD)')
plt.ylabel('Total Quantity (Metric Tons)')


# 주요 국가 라벨링
for i in range(5):
    row = country_dist.sort_values('v', ascending=False).iloc[i]
    plt.text(row['v'], row['q'], row['country_name'], fontsize=12)
plt.tight_layout()
plt.savefig('scatter_distribution.png')
plt.show()
