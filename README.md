# 뉴욕 에어비앤비 숙소 가격 예측 
- 뉴욕 에어비앤비 숙소에 대해 가격을 예측하는 회귀분석 프로젝트입니다.
- 에어비앤비 호스팅 경험 시, 스마트 가격 서비스로 주변 가격을 분석하여 자동으로 가격이 설정되는 경험을 통해 이를 구현하고자 본 프로젝트를 진행하였습니다. 
- 본 프로젝트에서는 캐글에서 제공된 데이터를 활용하여 진행되었습니다. 
- 본 프로젝트의 목적은 회귀 진행에 대한 다양한 모델링 활용과 데이터 전처리, 그리고 Polinomial을 활용한 다항회귀 이용을 목적으로 하였습니다. 

## Getting Started
### Requirements
- Python 3.6+
### Installation
The quick way:
```
pip install pandas
pip install matplotlib
pip install seaborn
pip install sklearn
```
### Dataset
- New York City Airbnb Open Data
  - https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

## 분석 진행순서
### 1. 데이터 EDA 
- 각 feature의 상관관계 

  <img src="https://user-images.githubusercontent.com/72847093/104142502-3bcd2c80-53ff-11eb-83ce-64f8be7ea130.png"></img>
  
- 지역별 숙소의 개수

  <img src="https://user-images.githubusercontent.com/72847093/104142681-ec3b3080-53ff-11eb-8db1-cf42268f0ac7.png"></img>
- 지역별 숙소의 분포와 예약 가능일 수 

  <img src="https://user-images.githubusercontent.com/72847093/104142716-0aa12c00-5400-11eb-9a21-febecf3e333f.png"></img>
- 숙소 유형의 개수 

  <img src="https://user-images.githubusercontent.com/72847093/104142732-21478300-5400-11eb-9d09-4cf9ebbf0079.png"></img>
- 지역별 숙소 유형에 따른 분포

  <img src="https://user-images.githubusercontent.com/72847093/104142740-2c9aae80-5400-11eb-82ce-9f61f77968a8.png"></img>
- 주요 도시에 대한 워드클라우드 시각화

  <img src="https://user-images.githubusercontent.com/72847093/104142760-420fd880-5400-11eb-92b7-378bb401c628.png"></img>
  
### 2. 데이터가공 및 전처리
#### 데이터 전처리 
```python
# drop column 타겟에 영향을 미치지 않는 독립변수 제거 

airbnb.drop(['host_id', 'latitude', 'longitude', 'neighbourhood', 'number_of_reviews', 'reviews_per_month'], axis = 1, inplace=True)
airbnb.head()

# 범주형 데이터는 인코딩 

def Encode(airbnb):
    for column in airbnb.columns[airbnb.columns.isin(['neighbourhood_group','room_type'])]:
        airbnb[column] = airbnb[column].factorize()[0] #factorize 범주형 
    return airbnb
airbnb_en = Encode(airbnb.copy())
```
#### 데이터 타겟에 대한 로그화 
```python
y_target.hist()
# 타겟의 분포가 심하게 편중되어 있음 
# 로그화 처리하여 분포를 변경하기 
log_price = np.log1p(airbnb_en['price'])
sns.distplot(log_price)
```
  <img src="https://user-images.githubusercontent.com/72847093/104143211-f0684d80-5401-11eb-922e-e0d60a30e996.png"></img>
#### 이상치 제거 
```python
# 회귀 계수 확인 

coef = pd.Series(lr_reg.coef_, index=X_features.columns)
coef_sort = coef.sort_values(ascending=False)
sns.barplot(x= coef_sort.values, y= coef_sort.index)

# room_type 에 대해 영향을 많이 받음을 알 수 있다 
```
  <img src="https://user-images.githubusercontent.com/72847093/104143295-34f3e900-5402-11eb-8e13-aced5c68d0a4.png"></img>
```python
# 이상치 제거 
# - room_type 과 가격간의 관계에 대해 plot 시각화 

plt.scatter(x= airbnb_en['room_type'], y=airbnb_en['price'])
plt.show()
```
  <img src="https://user-images.githubusercontent.com/72847093/104143352-6d93c280-5402-11eb-9ba4-ebeba85ade87.png"></img>

### 3. 모델별 학습, 예측, 평가 
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_features = airbnb_en.drop(columns='price', axis=1)
y_target = airbnb_en['price']

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=13)

# 모델 학습 객체 생성 
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)
lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

models = [lr_reg, ridge_reg, lasso_reg]
get_evals(models)

# 약간의 성능 향상은 있지만 많이 낮음 
```
### 4. 모델 검증
```python
# 다른 모델 사용
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


# polynomial 사용하기 
from sklearn.preprocessing import PolynomialFeatures

poly_ftr = PolynomialFeatures(degree=3).fit_transform(X_features)
X_train, X_test, y_train, y_test = train_test_split(poly_ftr, y_target, test_size=0.2, random_state=13)

# 모델 학습 객체 생성 
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)
lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)
lgbm_reg = LGBMRegressor()
lgbm_reg.fit(X_train, y_train)

models = [lr_reg, ridge_reg, lasso_reg, rf_reg, lgbm_reg]
get_evals(models)
```
### 5. polinomial 활용한 다항회귀 이용 
```python
# polynomial 사용하기 
from sklearn.preprocessing import PolynomialFeatures

poly_ftr = PolynomialFeatures(degree=3).fit_transform(X_features)
X_train, X_test, y_train, y_test = train_test_split(poly_ftr, y_target, test_size=0.2, random_state=13)

# 모델 학습 객체 생성 
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)
lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

models = [lr_reg, ridge_reg, lasso_reg]
get_evals(models)
```

### 6. 프로젝트를 마치며 
- 제한된 데이터에서의 성능 향상에 대한 고민을 많이 하였음 
- 숙소 평점, 편의시설 등 가격에 영향을 주는 데이터를 구할 수 있는지에 대한 추가 고민이 필요함 

  
