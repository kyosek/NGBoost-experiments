train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

tr.head()
tr.isnull().sum()
tr.dtypes

tr = tr.dropna(subset=['GarageCond','BsmtFinType2','MasVnrArea'])
te = te.dropna(subset=['GarageCond','BsmtFinType2','MasVnrArea'])

columns = ['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','BedroomAbvGr','KitchenAbvGr','Fireplaces','GarageYrBlt','GarageCars','GarageArea','YrSold']

tr,val = train_test_split(tr,test_size=.2,random_state=42)

