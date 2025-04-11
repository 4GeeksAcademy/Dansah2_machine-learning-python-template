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

# break up the data into training and testing samples and select the best variables
X = raw_data[['number_of_reviews', 'minimum_nights', 'calculated_host_listings_count', 'availability_365', 'neighbourhood_group', 'room_type']]
y = raw_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#select the best 4 features
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

