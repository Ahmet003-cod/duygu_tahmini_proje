import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns


df=pd.read_excel("C:/Users/Huzur Bilgisayar/OneDrive/Masaüstü/python uygulamalaı/TurkishTweets.xlsx")
df["Etiket"].unique()
df.info()
df.describe()
df.isnull().sum()
df.dropna(subset=["Tweet", "Etiket"],inplace=True)
X=df["Tweet"]
y=df["Etiket"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

tfd=TfidfVectorizer()
X_train_tfd=tfd.fit_transform(X_train)
X_test_tfd=tfd.transform(X_test)

"""from sklearn.linear_model import LogisticRegression

model=LogisticRegression(max_iter=1000)
model.fit(X_train_tfd,y_train)
y_pred=model.predict(X_test_tfd)
cm=confusion_matrix(y_test,y_pred)
print("LogisticRegression accuary score=",accuracy_score(y_test,y_pred))#LogisticRegression accuary score= 0.9675
print(f"LogisticRegression confusion_matrix={cm}")
etiketler = model.classes_  # y_test.unique() yerine bu daha güvenli olur

# Görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=etiketler, yticklabels=etiketler)
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.title('Logistic Regression Confusion Matrix')
plt.tight_layout()
plt.show()


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=200,random_state=42)
rfc.fit(X_train_tfd,y_train)
y_pred=rfc.predict(X_test_tfd)
cm=confusion_matrix(y_test,y_pred)
print(" RandomForestClassifier accuary score=",accuracy_score(y_test,y_pred))#RandomForestClassifier accuary score= 0.9625
print(f" RandomForestClassifierconfusion_matrix={cm}")
# Görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=etiketler, yticklabels=etiketler)
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.title(' RandomForestClassifier Confusion Matrix')
plt.tight_layout()
plt.show()


from sklearn.naive_bayes import MultinomialNB

mnb=MultinomialNB()
mnb.fit(X_train_tfd,y_train)
y_pred=mnb.predict(X_test_tfd)
cm=confusion_matrix(y_test,y_pred)
print("MultinomialNB accuary score=",accuracy_score(y_test,y_pred))#MultinomialNB accuary score= 0.94625
print(f"MultinomialNBconfusion_matrix=\n{cm}")
# Görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=etiketler, yticklabels=etiketler)
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.title('LMultinomialNBgistic Regression Confusion Matrix')
plt.tight_layout()
plt.show()"""



from sklearn.svm import LinearSVC

lvc=LinearSVC(random_state=42)
lvc.fit(X_train_tfd,y_train)
y_pred=lvc.predict(X_test_tfd)
cm=confusion_matrix(y_test,y_pred)
print("LinearSVC accuary score=",accuracy_score(y_test,y_pred))#LinearSVC accuary score= 0.9775
print(f"LinearSVC confusion_matrix=\n{cm}")
# Görselleştir
"""plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=df["Etiket"].unique(), yticklabels=df["Etiket"].unique())
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.title('LinearSVC Confusion Matrix')
plt.tight_layout()
plt.show()"""

import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd


etiketler = lvc.classes_  # modelin öğrendiği sınıflar

# 3. Tahmin fonksiyonu
def tahmin_et():
    girilen_cumle = entry.get()
    if not girilen_cumle.strip():
        messagebox.showwarning("Uyarı", "Lütfen bir cümle girin.")
        return
    
    cumle_tfd = tfd.transform([girilen_cumle])
    tahmin = lvc.predict(cumle_tfd)[0]
    
    sonuc_label.config(text=f"Tahmin Edilen Duygu: {tahmin}")

# 4. Arayüz tasarımı
pencere = tk.Tk()
pencere.title("Tweet Duygu Tahmin Aracı")
pencere.geometry("600x300")

etiket = tk.Label(pencere, text="Bir cümle girin:", font=("Arial", 12))
etiket.pack(pady=10)

entry = tk.Entry(pencere, width=60, font=("Arial", 12))
entry.pack(pady=5)

buton = tk.Button(pencere, text="Tahmin Et", command=tahmin_et, font=("Arial", 12), bg="#4CAF50", fg="white")
buton.pack(pady=10)

sonuc_label = tk.Label(pencere, text="", font=("Arial", 14), fg="blue")
sonuc_label.pack(pady=10)

# 🔽 5 duygu etiketi göster
duygu_yazisi = "Duygular: " + " | ".join(etiketler)
etiket_label = tk.Label(pencere, text=duygu_yazisi, font=("Arial", 11), fg="gray")
etiket_label.pack(pady=10)

pencere.mainloop()