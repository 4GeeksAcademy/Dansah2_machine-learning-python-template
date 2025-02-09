import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

#read the data
raw_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
raw_data.head()

#drop some variables
raw_data.drop(['id', 'name', 'host_name', 'last_review', 'reviews_per_month'], axis=1, inplace=True)
raw_data.head()

# keep the data where the price is higher than 0
raw_data = raw_data[raw_data['price'] > 0]

# Keep the data where the minimum_nights is smaller than or equal to 15 days
raw_data = raw_data[raw_data['minimum_nights'] <= 11]

# keep the data where the calculated_host_listings_count is smaller than 4
raw_data = raw_data[raw_data['calculated_host_listings_count'] <4]

# scale the data (except for the target variable)
variables = ['number_of_reviews', 'minimum_nights', 'calculated_host_listings_count', 'availability_365', 'neighbourhood_group', 'room_type']
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(raw_data[variables])
df_scaled = pd.DataFrame(scaled_features, index=raw_data.index, columns = variables)
df_scaled['price'] = raw_data['price']

# break up the data into training and testing samples and select the best variables
X = df_scaled.drop('price', axis = 1)
y = df_scaled['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

select_model = SelectKBest(chi2, k=4)
select_model.fit(X_train, y_train)
ix = select_model.get_support()

X_train_sel = pd.DataFrame(select_model.transform(X_train), columns=X_train.columns.values[ix])
X_test_sel = pd.DataFrame(select_model.transform(X_test), columns=X_test.columns.values[ix])

# save the data in the appropriate directory
X_train_sel['price'] = list(y_train)
X_test_sel['price'] = list(y_test)
X_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index = False)

