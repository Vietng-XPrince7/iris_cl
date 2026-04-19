from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dữ liệu Iris
iris = load_iris()
X = iris.data
y = iris.target

# 2. Chia tập dữ liệu (80% train, 20% test)
# random_state=0 giúp kết quả giống hệt nhau mỗi lần chạy
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 3. Chuẩn hóa dữ liệu (Scaling)
# Giúp các đặc trưng có cùng thang đo, giúp model học tốt hơn
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. Khởi tạo và huấn luyện mô hình Random Forest
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 5. Dự đoán trên tập test
y_pred = model.predict(x_test)

# 6. In kết quả so sánh giữa dự đoán (Prediction) và thực tế (Label)
for pred, label in zip(y_pred, y_test):
    print("Prediction: {}. Label: {}".format(pred, label))

# 7. Tính toán độ chính xác tổng thể
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))