from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_breast_cancer()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

log_reg = LogisticRegression(random_state=42,max_iter=1000)

log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"准确率 (Accuracy): {accuracy:.2%}")

print("\n--- 详细报告 ---")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 随便挑测试集的前5个病人
sample_data = x_test[:5]
sample_true_labels = y_test[:5]

# 预测概率
probs = log_reg.predict_proba(sample_data)

print("\n--- 概率预测演示 ---")
for i in range(5):
    # probs[i][0] 是第一类(恶性)的概率，probs[i][1] 是第二类(良性)的概率
    malignant_prob = probs[i][0]
    benign_prob = probs[i][1]
    true_label = "良性" if sample_true_labels[i] == 1 else "恶性"
    
    print(f"病人{i+1}: 恶性概率 {malignant_prob:.4f}, 良性概率 {benign_prob:.4f} -> 真实结果: {true_label}")