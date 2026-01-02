# =========================
# T I T A N I C  (Kısaltılmış tam akış)
# =========================

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, CategoricalDtype
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

pd.set_option('display.max_columns', None)        # Tüm sütunları göster
pd.set_option('display.max_colwidth', None)       # Hücrelerdeki tüm metni göster

# ---------- 1) VERİYİ YÜKLE
train_raw = pd.read_csv("train.csv")              # HAM
test_raw  = pd.read_csv("test.csv")               # HAM
train = train_raw.copy()                          # İşlenecek kopya
test  = test_raw.copy()                           # İşlenecek kopya

# ---------- 2) HIZLI EDA (çıktılar öncekiyle aynı)
print("Train veri seti ilk 5 satır:"); print(train.head())
print("\nTest veri seti ilk 5 satır:"); print(test.head())
print(train.shape); print(train.columns); print(train.dtypes); train.info()
print(train.isnull().sum())
print("\nEksik veri yüzdeleri:"); print((train.isnull().sum() / len(train)) * 100)
print(train.describe())
print(train['Survived'].value_counts()); print(train['Survived'].value_counts(normalize=True) * 100)
print("\nCinsiyete göre hayatta kalma oranı :"); print(train.groupby('Sex')['Survived'].mean() * 100)
print("\nEmbarked a göre hayatta kalma oranı:"); print(train.groupby('Embarked')['Survived'].mean() * 100)
print("\nBilet sınıfına göre hayatta kalma oranı :"); print(train.groupby('Pclass')['Survived'].mean() * 100)

aynilar = train[train['Ticket'].duplicated(keep=False)]                      # aynı Ticket'lar
sonuc = aynilar.groupby('Ticket')['Name'].apply(list).reset_index()          # aynı biletteki isim listesi
sonuc['KisiSayisi'] = sonuc['Name'].apply(len)                                # kişi sayısı
print(sonuc)

kategorik_sutunlar = train.select_dtypes(include=['object']).columns         # kategorik kolonlar
for sutun in kategorik_sutunlar:                                             # her birinin dağılımı
    print(f"\n--- {sutun} sütunu ---")
    print(train[sutun].value_counts())
    print(train[sutun].value_counts(normalize=True) * 100)

sayisal_sutunlar = train.select_dtypes(include=['int64', 'float64'])         # sayısal kolonlar
print(sayisal_sutunlar.describe())

sayisal = train.select_dtypes(include=['int64', 'float64'])                  # korelasyon için sayısallar
survived_corr = sayisal.corr()['Survived'].drop('Survived').sort_values(ascending=False)
print("Survived ile korelasyonlar:"); print(survived_corr)

print("\nCabin sütunundaki eksik değer sayısı:"); print(train['Cabin'].isnull().sum())
train['Cabin'] = train['Cabin'].fillna('Unknown')                            # Cabin eksikleri doldur
train['CabinDeck'] = train['Cabin'].apply(lambda x: x[0])                    # ilk harf (deck)
print(pd.crosstab(train['CabinDeck'], train['Survived']))                    # çapraz tablo
print(train.groupby('CabinDeck')['Survived'].mean() * 100)                   # deck göre oran

print(pd.crosstab(train['Embarked'], train['Survived']))                     # Embarked - Survived
tekrar_eden_isimler = train[train['Name'].duplicated(keep=False)]            # aynı isimler
print(tekrar_eden_isimler)
biletsiz = train[train['Ticket'].isna()]                                     # NaN ticket
ucretsiz = train[train['Fare'] == 0]                                         # ücreti 0 olanlar
print("Biletsiz Yolcular:"); print(biletsiz)
print("\nÜcretsiz Yolcular:"); print(ucretsiz)


# TEST.CS
test = pd.read_csv("test.csv")                                               # test yeniden yükle (EDA için)
print(test.shape); print(test.columns); print(test.dtypes); test.info()
print(test.isnull().sum()); print(test.describe())

kategorik_sutunlar = test.select_dtypes(include=['object']).columns          # testteki kategorikler
for sutun in kategorik_sutunlar:
    print(f"\n--- {sutun} sütunu ---")
    print(test[sutun].value_counts())
    print((test[sutun].value_counts(normalize=True) * 100).round(2))

print(train.isnull().sum())                                                  # eksikler (train)

print("ÖNCE (TRAIN) -> Age, Cabin, Embarked");                               # seçili kolonlardaki boşluk
print(train[['Age','Cabin','Embarked']].isnull().sum())
print("\nÖNCE (TEST) -> Age, Fare, Cabin ");                                 # testte seçili kolonlar
print(test[['Age', 'Fare','Cabin']].isnull().sum())

# ---------- 3) EKSİKLERİ DOLDUR
train['Age'] = train['Age'].fillna(train['Age'].median())                    # Age medyan
train['Cabin']=train['Cabin'].fillna('Unknown')                              # Cabin Unknown
train['Embarked']=train['Embarked'].fillna('S')                              # Embarked mod S

test['Age']=test['Age'].fillna(test['Age'].median())                         # test Age
test['Fare']=test['Fare'].fillna(test['Fare'].median())                      # test Fare
test['Cabin']=test['Cabin'].fillna('Unknown')                                # test Cabin

print("\nSONRA (TRAIN) -> Age, Cabin, Embarked");                            # doldurma sonrası kontrol
print(train[['Age','Cabin','Embarked']].isnull().sum())
print("\nSONRA (TEST) -> Age, Fare, Cabin");                                 # doldurma sonrası test
print(test[['Age','Fare','Cabin']].isnull().sum())

# ---------- 4) TÜREV ÖZELLİKLER + YAKINBIN
for df in (train, test):                                                     # train & test birlikte
    df['CabinDeck'] = df['Cabin'].fillna('Unknown').astype(str).str[0].str.upper()   # deck
    df.loc[~df['CabinDeck'].isin(list('ABCDEFGTU')), 'CabinDeck'] = 'U'              # beklenmeyen → U
    df['YakinSayisi'] = df['SibSp'] + df['Parch']                                    # akraba sayısı
    df['FamilySize']  = df['YakinSayisi'] + 1                                        # aile boyutu
    df['IsAlone']     = (df['YakinSayisi'] == 0).astype(int)                         # yalnız mı
    df['YakinBin']    = pd.cut(df['YakinSayisi'], bins=[-1,0,2,5,100],
                               labels=['0','1-2','3-5','6+'])                        # binlenmiş yakın

print("\nYakinSayisi -> Hayatta kalma oranı (%):");                              # oranlar
print((train.groupby('YakinSayisi')['Survived'].mean() * 100).round(1))
print("\nYakinBin -> Hayatta kalma oranı (%):")
print((train.groupby('YakinBin', observed=True)['Survived'].mean() * 100).round(1))

# ---------- 5) KATEGORİK KODLAMA
def safe_map(df, col, mapping, unknown_value=-1, to_upper=False, strip=True):
    s = df[col]                                                                 # kaynak seri
    if is_numeric_dtype(s):                                                     # sayısal ise sadece doldur
        df[col] = s.fillna(unknown_value).astype(int); return
    s_norm = s.astype(str)                                                      # stringe çevir
    if strip:   s_norm = s_norm.str.strip()                                     # Başındaki ve sonundaki boşlukları sil
    if to_upper:
        # Büyük harfe çevir ve mapping sözlüğündeki anahtarları da büyük harfe dönüştür
        s_norm = s_norm.str.upper(); map_use = {k.upper(): v for k, v in mapping.items()}
    else:
        # Küçük harfe çevir ve mapping sözlüğündeki anahtarları da küçük harfe dönüştür
        s_norm = s_norm.str.lower(); map_use = {k.lower(): v for k, v in mapping.items()}
    mapped = s_norm.map(map_use)               # Normalize edilmiş değerleri mapping sözlüğüne göre eşleştir
    if mapped.isna().any():                    # eşleşmeyenleri bildir
        print(f"[UYARI] {col} sütununda eşleşmeyen değerler:", sorted(set(s_norm[mapped.isna()].unique())))
    df[col] = mapped.fillna(unknown_value).astype(int)                          # -1 ile doldur

#KATEGORİK VERİLERİ SAYISALLAŞTIRMA
for df in (train, test):                        # Sex kodlama
    safe_map(df, 'Sex', {'male':0, 'female':1}, unknown_value=-1, to_upper=False)
for df in (train, test):                       # Embarked kodlama
    safe_map(df, 'Embarked', {'S':0,'C':1,'Q':2}, unknown_value=-1, to_upper=True)

for df in (train, test):                       # CabinDeck kodlama
    if 'CabinDeck' not in df.columns:
        df['CabinDeck'] = df['Cabin'].fillna('Unknown').astype(str).str[0].str.upper()
        df.loc[~df['CabinDeck'].isin(list('ABCDEFGTU')), 'CabinDeck'] = 'U'
deck_map = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7,'U':8}
for df in (train, test):
    safe_map(df, 'CabinDeck', deck_map, unknown_value=8, to_upper=True)

print("\nKodlama sonrası benzersiz değerler (train):")                         # kontrol yazdır
print("Sex:", sorted(train['Sex'].unique()))
print("Embarked:", sorted(train['Embarked'].unique()))     # Embarked sütununda hangi farklı değerler var göster
print("CabinDeck:", sorted(train['CabinDeck'].unique()))

# 6) GEREKSİZ KOLONLARI AT
for df in (train, test):
    for c in ['Name','Ticket','Cabin']:
        if c in df.columns: df.drop(columns=[c], inplace=True) # Kullanılmayacak sütunları (Name, Ticket, Cabin) veri setinden sil

print("Sütun kontrolü (train):", train.columns.tolist())
print("Sütun kontrolü (test):",  test.columns.tolist())

# ---------- 7) NOISE DÜZELTME + NEGATİF KONTROL
for df in (train, test):
    #'SEX,EMBARKED,..' SÜTUNU VAR MI KONTROLET , BİLİNMEYEN DEĞER VAR MI BAK -EĞER VARSA BİLİNMEYENİ MODLA DEĞİŞTİR
    if 'Sex' in df.columns and (df['Sex'] == -1).any(): df.loc[df['Sex'] == -1, 'Sex'] = train['Sex'].mode()[0]
    if 'Embarked' in df.columns and (df['Embarked'] == -1).any(): df.loc[df['Embarked'] == -1, 'Embarked'] = train['Embarked'].mode()[0]
    if 'CabinDeck' in df.columns: df.loc[~df['CabinDeck'].isin(range(0,9)), 'CabinDeck'] = 8
    if 'Age' in df.columns:  df.loc[df['Age']  < 0, 'Age']  = pd.NA              # negatifleri NaN yap
    if 'Fare' in df.columns: df.loc[df['Fare'] < 0, 'Fare'] = pd.NA

if 'Age' in train.columns:  train['Age']  = train['Age'].fillna(train['Age'].median());  test['Age']  = test['Age'].fillna(train['Age'].median())
if 'Fare' in train.columns: train['Fare'] = train['Fare'].fillna(train['Fare'].median()); test['Fare'] = test['Fare'].fillna(train['Fare'].median())

# ---------- 8) OUTLIER (IQR) + WINSORIZE
def iqr_bounds(s, k=1.5):     # IQR alt/üst
    q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1; return (q1 - k*iqr, q3 + k*iqr)
num_cols = [c for c in ['Age','Fare','SibSp','Parch','FamilySize','YakinSayisi']
            if c in train.columns]   #sadece trainde var olan sütunları seç
bounds = {c: iqr_bounds(train[c].dropna()) for c in num_cols}  # Her sayısal sütunun IQR tabanlı aykırı sınırlarını hesaplayıp 'bounds'a yazar
print("\nIQR sınırları (train):")
for c,(lo,hi) in bounds.items():   # bounds'taki her sütun için (alt, üst) sınırları al
    print(f"{c}: alt={lo:.2f}, üst={hi:.2f} | outlier sayısı={((train[c]<lo)|(train[c]>hi)).sum()}")   # sütun adı, sınırlar ve aykırı (sınır dışı) gözlem sayısını yazdır

for g, sub in train.groupby('Pclass'):          # train Fare'ı sınıfa göre kırp
    lo, hi = iqr_bounds(sub['Fare'].dropna()); train.loc[sub.index, 'Fare'] = sub['Fare'].clip(lo, hi)
for g, sub in test.groupby('Pclass'):           # test Fare'ı train sınırlarına göre kırp
    lo, hi = iqr_bounds(train.loc[train['Pclass']==g, 'Fare'].dropna()); test.loc[sub.index, 'Fare'] = sub['Fare'].clip(lo, hi)

def outlier_summary(df, col):                                                  # kısa özet
    s=df[col].dropna(); Q1=s.quantile(0.25); Q2=s.quantile(0.5); Q3=s.quantile(0.75); IQR=Q3-Q1
    low=Q1-1.5*IQR; high=Q3+1.5*IQR; n_low=(s<low).sum(); n_high=(s>high).sum()  #ALT SINIR ; ÜST SINIR; ALTT SINIRIN ALTINDA KALAN DEĞER SAYISI;ÜST SINIRIN ALTINDA KALAN DEĞER SAYISI
    print(f"\n[{col}]"); print(f"Q1={Q1:.2f} | Medyan(Q2)={Q2:.2f} | Q3={Q3:.2f} | IQR={IQR:.2f}")
    print(f"Alt sınır={low:.2f} | Üst sınır={high:.2f}"); print(f"Outlier sayısı: alt={n_low} | üst={n_high} | toplam={n_low+n_high}")

print("\n TRAIN IQR OUTLIER ÖZETİ "); [outlier_summary(train, c) for c in num_cols]
print("\n TEST IQR OUTLIER ÖZETİ ");  [outlier_summary(test, c)  for c in num_cols if c in test.columns]

#OUTLIER GRAFİĞİ
for c in num_cols:
    s = train[c].dropna()
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    plt.figure(figsize=(6, 4))  plt.scatter(range(len(s)), s, alpha=0.7)
    plt.axhline(low, color='red', linestyle='--', label='Alt sınır')   plt.axhline(high, color='green', linestyle='--', label='Üst sınır')
    plt.title(f"{c} - Outlier Scatter (Train)")   plt.ylabel(c)    #grafik başlığı- y eksenine sütun adı yaz
    plt.legend() #alt-üst sınır etiketlerinin çizgilerini göster
    plt.show()  #grafiği çizdir

# ---------- 9) YakinBin ONE-HOT
if 'YakinBin' not in train.columns or 'YakinBin' not in test.columns:     # EĞER YAKINBİN SÜTUNU YOKSA OLUŞTUR
    for df in (train, test):
        if 'YakinSayisi' not in df.columns: df['YakinSayisi'] = df['SibSp'] + df['Parch']
        df['YakinBin'] = pd.cut(df['YakinSayisi'], bins=[-1,0,2,5,100],
                    labels=['0','1-2','3-5','6+']) #yakın sayısını aralıklara böl,kategorik etiketle
cats = ['0','1-2','3-5','6+']; cat_type = CategoricalDtype(categories=cats, ordered=True)  #kategori sırasını tanımla
train['YakinBin'] = train['YakinBin'].astype(cat_type); #trainde kategorik tipe çevir
test['YakinBin'] = test['YakinBin'].astype(cat_type) #tstte kategorik tipe çevir
d_train = pd.get_dummies(train['YakinBin'], prefix='YakinBin', drop_first=True) # OHE(pd.get_dummies =>her kategori için ayrı üstun oluşturur
#drop_first=True=> ilk kategori('0')siliniyor,böylece dummpy trap(çokludoğrusalbağımlılık) engelleniyor
d_test  = pd.get_dummies(test['YakinBin'],  prefix='YakinBin', drop_first=True)
d_test  = d_test.reindex(columns=d_train.columns, fill_value=0)     # Test sütunlarını train ile hizala, eksikleri 0 doldur
train = pd.concat([train.drop(columns=['YakinBin']), d_train], axis=1)   # Train’den eski YakinBin’i sil, OHE sütunlarını ekle
test  = pd.concat([test.drop(columns=['YakinBin']),  d_test],  axis=1)   #Testte de aynısını yap
print("One-hot sonrası sütunlar (train):", [c for c in train.columns if c.startswith('YakinBin_')])
print("One-hot sonrası sütunlar (test):",  [c for c in test.columns  if c.startswith('YakinBin_')])

# ---------- 10) EKSİK SÜTUNLARI TAMAMLAMA
target='Survived'; id_col='PassengerId'                                        # hedef değişken(y)/IDsütunu
drop_cols=[target,id_col,'Name','Ticket','Cabin','YakinBin']                   # modele girmeyecek sütunlar
X = train[[c for c in train.columns if c not in drop_cols]].copy()         # Özellik matrisi (X) → drop_cols hariç tüm sütunlar
y = train[target].astype(int).copy()                                       # Hedef değişken (y) → int tipine dönüştür
num_mask = X.dtypes.apply(lambda t: t.kind in 'biufc')
#x.dtypes->x veri setindeki her sütunun tipini döndürür ,t.kind->tipin türünü verir
#yani hangi stunlar sayısal hangileri değil mask olarak bulduk
X.loc[:, num_mask] = X.loc[:, num_mask].fillna(X.loc[:, num_mask].median())     #sadece sayısal sütunları seç boşsa:NaN ise→medyanla doldur
X.loc[:, ~num_mask] = X.loc[:, ~num_mask].fillna(0)                             #sayısal sütun değilse NaN→0

# ---------- 11) TRAIN/VAL BÖLME + 5-KAT CV
#RANDOM FOREST VE LOGISTIC REGRESSION KIYASLAMASI YAPTIK
#VERİYİ TRAIN VE VALIDATIN SETLERİNE AYIRIYORUZ
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)  #%20VAL/%80TRAIN
print("Train/Val şekil:", X_tr.shape, X_val.shape)
print("Sınıf oranı (train):", (y_tr.value_counts(normalize=True)*100).round(2).to_dict())
print("Sınıf oranı (val):",   (y_val.value_counts(normalize=True)*100).round(2).to_dict())

logreg = Pipeline([('sc', StandardScaler(with_mean=False)), #pıpelıne=StandartScaler ve logistic regression u birleştirir
                   #with_mean=False çünkü One-Hot Encoding sonrası sparse matrislerde mean almak sorun çıkarır.
                   ('clf', LogisticRegression(solver='liblinear', max_iter=500, random_state=42))])
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)  #n_jobs=-1 → tüm işlemci çekirdeklerini kullan, daha hızlı çalış

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)                 # 5-kat CV
scoring = {'acc':'accuracy','f1':'f1','roc':'roc_auc'}      # metrikler(ACC,F1,ROC) HER TURDA HESAPLANIR.5 TUR YAPILIR
#cv_lr=logıstıcregression sonuçları    cv_rf=random forest sonuçları
cv_lr = cross_validate(logreg, X, y, cv=cv, scoring=scoring, n_jobs=-1)   #YUKARDAKİ HESAPLARIN SONUÇLARI CV_LR SÖZLÜĞÜNDE TUTULUR
cv_rf = cross_validate(rf,     X, y, cv=cv, scoring=scoring, n_jobs=-1)   #YUKARDAKİ HESAPLARIN SONUÇLARI CV_RF SÖZLÜĞÜNDE TUTULUR
def summarize(cv_res, name):  #cv_res İÇİNDE: TEST_ACC,TEST_F1,TEST_ROC VAR
    return {'Model':name,      #FONKSİYON Bu 3 metriğin ortalamasını (mean) ve standart sapmasını (std) hesaplıyor
            'Accuracy (mean±std)': f"{cv_res['test_acc'].mean():.3f} ± {cv_res['test_acc'].std():.3f}",
            'F1 (mean±std)':       f"{cv_res['test_f1'].mean():.3f} ± {cv_res['test_f1'].std():.3f}",
            'ROC AUC (mean±std)':  f"{cv_res['test_roc'].mean():.3f} ± {cv_res['test_roc'].std():.3f}"}
summary = pd.DataFrame([summarize(cv_lr,"Logistic Regression"), summarize(cv_rf,"Random Forest")])
print("\n=== 5-Kat CV Sonuçları ==="); print(summary.to_string(index=False))

# ---------- 12) BASİT MODEL + SUBMISSION
rf_base = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1) # temel RF
              #400 ağaçlı bir Random Forest sınıflandırıcı oluşturuluyor
X_test = test.reindex(columns=X.columns, fill_value=0).copy()   #Test setinin sütunları train’deki X.columns ile aynı sıraya getiriliyor.
              #Train’de olup test’te olmayan sütunlar varsa → 0 ile dolduruluyor.
X_test.loc[:, X_test.dtypes.apply(lambda t: t.kind in 'biufc')] = \   #biufc=sayısal tiplerin kısa kodları
    X_test.loc[:, X_test.dtypes.apply(lambda t: t.kind in 'biufc')].fillna(
        X_test.loc[:, X_test.dtypes.apply(lambda t: t.kind in 'biufc')].median()) # Test setindeki sayısal sütunlardaki NaN değerler, o sütunun medyanı ile dolduruluyor
X_test.loc[:, ~X_test.dtypes.apply(lambda t: t.kind in 'biufc')] = \
    X_test.loc[:, ~X_test.dtypes.apply(lambda t: t.kind in 'biufc')].fillna(0)   # diğer NaN
rf_base.fit(X, y)                                                                # tüm train ile fit
pred_test = rf_base.predict(X_test).astype(int)                                  # sınıf tahmini
pd.DataFrame({'PassengerId': test[id_col], 'Survived': pred_test}).to_csv('submission.csv', index=False)
print("[OK] submission.csv kaydedildi.")



# G R A F İ K L E R  (KISA)
import os, matplotlib.pyplot as plt, pandas as pd
figdir = "figures"; os.makedirs(figdir, exist_ok=True)   # çıktı klasörü

# Küçük yardımcılar
def bar(idx, vals, ttl, x, y, fn, rot=0):
    plt.figure(); plt.bar(range(len(vals)), vals)
    plt.title(ttl); plt.xlabel(x); plt.ylabel(y)
    plt.xticks(range(len(idx)), idx, rotation=rot)
    plt.tight_layout(); plt.savefig(os.path.join(figdir, fn), dpi=140); plt.close()

def hist(s, bins, ttl, x, y, fn):
    plt.figure(); plt.hist(s.dropna(), bins=bins)
    plt.title(ttl); plt.xlabel(x); plt.ylabel(y)
    plt.tight_layout(); plt.savefig(os.path.join(figdir, fn), dpi=140); plt.close()

def norm_deck(s):  # Cabin -> A..G,T,U (bilinmeyen=U)
    s = s.fillna("Unknown").astype(str).str.strip().str.upper().str[0]
    return s.where(s.isin(list("ABCDEFGTU")), "U")

# 1) Eksik veriler (ÖNCE/SONRA)
for tag, df in [("before", train_raw), ("after", train)]:
    na = df.isnull().sum().sort_values(ascending=False); na = na[na>0]
    if len(na):
        bar(na.index.tolist(), na.values.tolist(),
            f"Eksik Hücre Sayıları ({'Önce' if tag=='before' else 'Sonra'} - TRAIN)",
            "Sütun", "Eksik Sayısı",
            f"{'01' if tag=='before' else '02'}_train_na_counts_{tag}.png", rot=45)

# 2) Sayısal histogramlar (ÖNCE/SONRA)
for col in [c for c in ['Age','Fare','SibSp','Parch'] if c in train.columns]:
    if col in train_raw.columns:
        hist(train_raw[col], 30, f"{col} Dağılımı (Önce - TRAIN)", col, "Frekans", f"10_hist_{col}_before.png")
    hist(train[col], 30, f"{col} Dağılımı (Sonra - TRAIN)", col, "Frekans", f"11_hist_{col}_after.png")

# 3) Survived kırılımları (yalnız TRAIN)
bar(['Hayır (0)','Evet (1)'], train['Survived'].value_counts().sort_index().values,
    "Hayatta Kalma (Genel - TRAIN)", "Sınıf", "Kişi Sayısı", "20_survived_overall.png")
if 'Sex' in train_raw.columns:
    r = (train_raw.groupby('Sex')['Survived'].mean()*100).sort_values(ascending=False)
    bar(r.index.tolist(), r.values.tolist(), "Cinsiyete Göre Hayatta Kalma Oranı (Önce - TRAIN)",
        "Cinsiyet", "Oran (%)", "21_survival_by_sex_before.png")
if 'Embarked' in train_raw.columns:
    r = (train_raw.groupby('Embarked')['Survived'].mean()*100).sort_values(ascending=False)
    bar(r.index.tolist(), r.values.tolist(), "Embarked'a Göre Hayatta Kalma Oranı (Önce - TRAIN)",
        "Liman", "Oran (%)", "22_survival_by_embarked_before.png")
if 'Pclass' in train_raw.columns:
    r = (train_raw.groupby('Pclass')['Survived'].mean()*100).sort_index()
    bar([str(i) for i in r.index], r.values, "Pclass'a Göre Hayatta Kalma Oranı (TRAIN)",
        "Pclass", "Oran (%)", "23_survival_by_pclass.png")

# CabinDeck (ÖNCE/SONRA)
if {'Cabin','Survived'}.issubset(train_raw.columns):
    tmp = train_raw.copy(); tmp['CabinDeck'] = norm_deck(tmp['Cabin'])
    r = (tmp.groupby('CabinDeck')['Survived'].mean()*100).sort_index()
    bar(r.index.tolist(), r.values.tolist(), "CabinDeck'e Göre Hayatta Kalma Oranı (Önce - TRAIN)",
        "CabinDeck", "Oran (%)", "24_survival_by_cabindeck_before.png")
if 'CabinDeck' in train.columns:
    r = (train.groupby('CabinDeck')['Survived'].mean()*100).sort_index()
    inv = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'T',8:'U'}
    bar([inv.get(i,str(i)) for i in r.index], r.values,
        "CabinDeck'e Göre Hayatta Kalma Oranı (Sonra - TRAIN)",
        "CabinDeck", "Oran (%)", "25_survival_by_cabindeck_after.png")
# Yakın sayısı
if all(c in train_raw.columns for c in ['SibSp','Parch','Survived']):
    ys = train_raw['SibSp'] + train_raw['Parch']
    r = pd.DataFrame({'YakinSayisi': ys, 'Survived': train_raw['Survived']}).groupby('YakinSayisi')['Survived'].mean()*100
    r = r.sort_index()
    bar([str(i) for i in r.index], r.values, "YakinSayisi'na Göre Hayatta Kalma Oranı (Önce - TRAIN)",
        "YakinSayisi", "Oran (%)", "26_survival_by_yakinsayisi_before.png")
if 'YakinSayisi' in train.columns:
    cats = ['0','1-2','3-5','6+']
    yb = pd.cut(train['YakinSayisi'], bins=[-1,0,2,5,100], labels=cats)
    r = (train.groupby(yb)['Survived'].mean()*100).reindex(cats)
    bar(cats, r.values, "YakinBin'e Göre Hayatta Kalma Oranı (TRAIN)",
        "YakinBin", "Oran (%)", "27_survival_by_yakinbin.png")
# 4) Embarked dağılım (ÖNCE/SONRA)
if 'Embarked' in train_raw.columns:
    cnt = train_raw['Embarked'].fillna('NaN').value_counts()
    bar(cnt.index.tolist(), cnt.values.tolist(), "Embarked Dağılımı (Önce - TRAIN)",
        "Liman", "Kişi Sayısı", "30_embarked_counts_before.png")
if 'Embarked' in train.columns:
    inv = {0:'S',1:'C',2:'Q',-1:'UNK'}
    cnt = train['Embarked'].map(inv).fillna('UNK').value_counts()
    bar(cnt.index.tolist(), cnt.values.tolist(), "Embarked Dağılımı (Sonra - TRAIN)",
        "Liman", "Kişi Sayısı", "31_embarked_counts_after.png")
# 5) Korelasyon (Survived ile)
num = train.select_dtypes(include=['int64','float64'])
c = num.corr(numeric_only=True)['Survived'].drop('Survived').abs().sort_values(ascending=False).head(12)
bar(c.index.tolist(), num.corr(numeric_only=True)['Survived'][c.index].values,
    "Survived ile Korelasyon (En Yüksek 12)", "Değişken", "Korelasyon",
    "40_corr_with_survived.png", rot=45)

print(f"[OK] {figdir}/ altına grafikler kaydedildi.")

figdir="figures"; os.makedirs(figdir, exist_ok=True)                             # çıktı klasörü
def save_bar(ix, vals, title, x, y, fname, rot=0, **kw):                         # çubuk grafik kaydet
    plt.figure(); plt.bar(range(len(vals)), vals, **kw); plt.title(title)
    plt.xlabel(x); plt.ylabel(y); plt.xticks(range(len(ix)), ix, rotation=rot)
    plt.tight_layout(); plt.savefig(os.path.join(figdir, fname), dpi=140); plt.close()
def save_hist(s, bins, title, x, y, fname, **kw):                                # histogram kaydet
    plt.figure(); plt.hist(s.dropna(), bins=bins, **kw); plt.title(title)
    plt.xlabel(x); plt.ylabel(y); plt.tight_layout()
    plt.savefig(os.path.join(figdir, fname), dpi=140); plt.close()

EMB = {'S':'#2E7D32','C':'#C62828','Q':'#1565C0','UNK':'#9E9E9E','NaN':'#9E9E9E'} # renk paleti
PCLS = ['#2E7D32','#C62828','#1565C0']                                           # pclass renkleri
def ttl(base, ds): return f"{base} ({ds})"                                       # başlık eki

def plots(df, raw, ds):                                                           # TRAIN/TEST küçük panel
    for tag, d in [('before', raw), ('after', df)]:
        na = d.isnull().sum(); na = na[na>0].sort_values(ascending=False)
        if len(na): save_bar(na.index, na.values, ttl("Eksik Hücre Sayıları", ds),
                              "Sütun","Eksik",f"{ds.lower()}_na_{tag}.png", rot=45)
    for col in [c for c in ['Age','Fare'] if c in df.columns]:
        if col in raw.columns: save_hist(raw[col],30,ttl(f"{col} Dağılımı",ds),col,"Frekans",f"{ds.lower()}_{col}_before.png",edgecolor="#222",alpha=.9)
        save_hist(df[col],30,ttl(f"{col} Dağılımı",ds),col,"Frekans",f"{ds.lower()}_{col}_after.png",edgecolor="#222",alpha=.9)
    if 'Embarked' in raw.columns:
        cnt = raw['Embarked'].fillna('NaN').value_counts()
        save_bar(cnt.index.tolist(), cnt.values.tolist(), ttl("Embarked Dağılımı", ds),
                 "Liman","Kişi",f"{ds.lower()}_emb_before.png",
                 color=[EMB.get(k,'#9E9E9E') for k in cnt.index], edgecolor="#222", alpha=.9)
    if 'Embarked' in df.columns:
        inv = {0:'S',1:'C',2:'Q',-1:'UNK'}; cnt = df['Embarked'].map(inv).fillna('UNK').value_counts()
        save_bar(cnt.index.tolist(), cnt.values.tolist(), ttl("Embarked Dağılımı", ds),
                 "Liman","Kişi",f"{ds.lower()}_emb_after.png",
                 color=[EMB.get(k,'#9E9E9E') for k in cnt.index], edgecolor="#222", alpha=.9)
    if 'Pclass' in df.columns:
        if 'Survived' in df.columns:
            rate=(raw.groupby('Pclass')['Survived'].mean()*100).sort_index()
            save_bar([str(i) for i in rate.index], rate.values, ttl("Pclass'a Göre Hayatta Kalma", ds),
                     "Pclass","Oran (%)",f"{ds.lower()}_pclass_rate.png", color=PCLS[:len(rate)], edgecolor="#222", alpha=.9)
        else:
            pc=df['Pclass'].value_counts().sort_index()
            save_bar([str(i) for i in pc.index], pc.values, ttl("Pclass Dağılımı", ds),
                     "Pclass","Kişi",f"{ds.lower()}_pclass_counts.png", color=PCLS[:len(pc)], edgecolor="#222", alpha=.9)
    if 'Survived' in df.columns:
        vc=df['Survived'].value_counts().sort_index()
        save_bar(['Hayır (0)','Evet (1)'], vc.values, ttl("Hayatta Kalma", ds),
                 "Sınıf","Kişi",f"{ds.lower()}_survived_overall.png",
                 color=['#C62828','#2E7D32'], edgecolor="#222", alpha=.9)

plots(train, train_raw, "TRAIN"); plots(test, test_raw, "TEST")                  # iki set için çiz
save_hist(test_raw['Fare'],30,"Fare Dağılımı (TEST-RAW)","Fare","Frekans","test_fare_raw_0_520.png",edgecolor="#222",alpha=.9,range=(0,520))

# Max Fare kontrol — ham vs işlenmiş
def show_max(df, name):                                                          # en büyük ücreti yazdır
    m = np.nanmax(pd.to_numeric(df["Fare"], errors="coerce").values)
    ridx = pd.to_numeric(df["Fare"], errors="coerce").idxmax()
    row = df.loc[ridx, ["PassengerId","Pclass","Fare"]]
    print(f"{name} CSV  max Fare: {float(m):.6f}  | PassengerId={int(row['PassengerId'])}  Pclass={int(row['Pclass'])}")
    return m
m_tr = show_max(train_raw[['PassengerId','Pclass','Fare']], "TRAIN")
m_te = show_max(test_raw[['PassengerId','Pclass','Fare']],  "TEST")
print("CSV max'lar eşit mi? ->", "EVET" if abs(m_tr - m_te) < 1e-9 else "HAYIR")
print(f"TRAIN PROC max Fare: {float(pd.to_numeric(train['Fare'], errors='coerce').max()):.6f}")
print(f"TEST  PROC max Fare: {float(pd.to_numeric(test['Fare'],  errors='coerce').max()):.6f}")
print(f"[OK] {figdir}/ altına TRAIN & TEST grafikler kaydedildi.")


# ---------- 14) HİPERPARAMETRE OPTİMİZASYONU ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)                 # Stratified 5-katlı CV

# LogReg: küçük arama uzayı
logreg_grid = Pipeline([('sc', StandardScaler(with_mean=False)),                # sayısal verileri ölçekle
                        ('clf', LogisticRegression(max_iter=1000, solver='liblinear',
                                                   class_weight='balanced', random_state=42))])  # lojistik regresyon
gs_lr = RandomizedSearchCV(logreg_grid, param_distributions={'clf__C':[0.1,0.5,1.0,3.0,10.0]},
                           n_iter=5, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42, refit=True) # hiperparametre arama
          #scoring='roc_auc': Kıyas metriği ROC AUC  - cv=cv: Az önce tanımlanan 5-kat stratified CV ile değerlendirme
          #refit=True: CV sonunda en iyi modeli yeniden eğit (tüm veriyle).
gs_lr.fit(X, y); best_lr = gs_lr.best_estimator_
print(f"[LogReg] best C={gs_lr.best_params_['clf__C']} | mean ROC AUC (CV)={gs_lr.best_score_:.4f}")
# Aramayı çalıştır; en iyi Lojistik Regresyon modelini best_lr olarak al.
# En iyi C ve onun CV ortalama AUC skorunu yazdır.

# RF: daha geniş arama
rs_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), # Random Forest modeli
    param_distributions={                                                      # denenecek hiperparametreler
        'n_estimators':[200,300,400,600,800],
        'max_depth':[None,4,6,8,10,12],
        'min_samples_split':[2,4,6,10],
        'min_samples_leaf':[1,2,3,5],
        'max_features':['sqrt','log2',0.5,0.7,None],
        'bootstrap':[True,False],
        'class_weight':[None,'balanced']
    },
    n_iter=40, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42, refit=True)
rs_rf.fit(X, y); best_rf = rs_rf.best_estimator_                                # en iyi RF modeli
print(f"[RF] best params={rs_rf.best_params_} | mean ROC AUC (CV)={rs_rf.best_score_:.4f}")

# Val setinde iki modeli karşılaştır
def eval_on_val(model, name):                                                   # doğrulama setinde ölçüm fonksiyonu
    model.fit(X_tr, y_tr); y_pred=model.predict(X_val)                          # modeli eğit ve tahmin yap
    y_proba = model.predict_proba(X_val)[:,1] if hasattr(model,"predict_proba") else None
    acc=accuracy_score(y_val,y_pred); f1=f1_score(y_val,y_pred)                 # doğruluk ve F1
    auc=roc_auc_score(y_val,y_proba) if y_proba is not None else np.nan         # Accuracy, F1 ve mümkünse ROC AUC (olasılık çıkarabiliyorsa) hesaplar
    print(f"{name}: Acc={acc:.3f} | F1={f1:.3f} | ROC AUC={auc:.3f}")
    return auc, f1, acc, y_proba

auc_lr, f1_lr, acc_lr, proba_lr = eval_on_val(best_lr, "LogReg(best)")          # LogReg doğrulama performansı
auc_rf, f1_rf, acc_rf, proba_rf = eval_on_val(best_rf, "RF(best)")              # RF doğrulama performansı
winner     = best_rf if (auc_rf >= auc_lr) else best_lr                         # AUC'ye göre daha iyi modeli seç
winner_p   = proba_rf if (auc_rf >= auc_lr) else proba_lr
winner_nm  = "RandomForest" if (auc_rf >= auc_lr) else "LogReg"
print(f"→ Seçilen model: {winner_nm}")         #ROC AUC’ye göre kazanan modeli seçiyorum (eşitlikte RF’yi tercih etmişim)
                                               #Seçilen modelin olasılık çıktısını (winner_p) da yanına alıyorum

# Eşik optimizasyonu (F1 maks.)
best_t = 0.5                                                                    # başlangıç eşiği
if winner_p is not None:                                                        # olasılık çıktıysa
    thr = np.linspace(0.1, 0.9, 81)                                             # 0.1-0.9 arası eşikler
    f1s = [f1_score(y_val, (winner_p >= t).astype(int)) for t in thr]           # her eşik için F1
    best_t = float(thr[int(np.argmax(f1s))])                                    # en yüksek F1 veren eşiği bul
    print(f"En iyi eşik (F1): t={best_t:.2f} | F1={np.max(f1s):.3f}")

# Tüm eğitimle eğit ve TEST'e uygula
winner.fit(X, y)                                                                 # seçilen modeli tüm train ile eğit
X_test = test.reindex(columns=X.columns, fill_value=0).copy()                    # test setini train ile hizala
num_mask_test = X_test.dtypes.apply(lambda t: t.kind in 'biufc')                 # sayısal sütun maskesi
X_test.loc[:, num_mask_test]  = X_test.loc[:, num_mask_test].fillna(X_test.loc[:, num_mask_test].median()) # sayısal NaN doldur
X_test.loc[:, ~num_mask_test] = X_test.loc[:, ~num_mask_test].fillna(0)          # kategorik NaN doldur
if hasattr(winner,"predict_proba"):                                              # olasılık tabanlı tahmin
    p_test = winner.predict_proba(X_test)[:,1]; pred_test = (p_test >= best_t).astype(int)  # en iyi eşikle karar
else:
    pred_test = winner.predict(X_test).astype(int)                               # direkt tahmin
pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_test}).to_csv('submission_opt.csv', index=False) # çıktıyı kaydet
print("[OK] submission_opt.csv yazıldı.")                                        # dosya oluşturuldu


# ---------- 15) OPTİMİZASYON ÖNCE–SONRA KIYAS + GRAFİK ----------
def eval_model(model, name, setup):                                             # metrik tablo satırı
    model.fit(X_tr,y_tr); y_hat=model.predict(X_val)
    y_pro = model.predict_proba(X_val)[:,1] if hasattr(model,"predict_proba") else None
    acc=accuracy_score(y_val,y_hat); f1=f1_score(y_val,y_hat)
    auc=roc_auc_score(y_val,y_pro) if y_pro is not None else np.nan
    return dict(Model=name, Setup=setup, Accuracy=acc, F1=f1, ROC_AUC=auc)

logreg_base = Pipeline([('sc', StandardScaler(with_mean=False)),
                        ('clf', LogisticRegression(solver='liblinear', max_iter=500, random_state=42))])
rf_base = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)

res = [eval_model(logreg_base,"LogReg","Baseline"),
       eval_model(rf_base,"RandomForest","Baseline"),
       eval_model(best_lr,"LogReg","Tuned"),
       eval_model(best_rf,"RandomForest","Tuned")]
df_res = pd.DataFrame(res).round(3)
print("\n=== ÖNCE–SONRA (VAL SETİ) ==="); print(df_res.to_string(index=False))

def deltas_for(model_name):                                                     # delta hesapla
    a=df_res[(df_res.Model==model_name)&(df_res.Setup=='Baseline')]
    b=df_res[(df_res.Model==model_name)&(df_res.Setup=='Tuned')]
    if len(a)==1 and len(b)==1:
        return pd.DataFrame({'Model':[model_name],
                             'ΔAccuracy':[float(b.Accuracy.values[0]-a.Accuracy.values[0])],
                             'ΔF1':[float(b.F1.values[0]-a.F1.values[0])],
                             'ΔROC_AUC':[float(b.ROC_AUC.values[0]-a.ROC_AUC.values[0])]}).round(3)
delta_lr, delta_rf = deltas_for("LogReg"), deltas_for("RandomForest")
print("\n=== Delta (Tuned - Baseline) ===")
if delta_lr is not None: print(delta_lr.to_string(index=False))
if delta_rf is not None: print(delta_rf.to_string(index=False))

winner_nm = globals().get('winner_nm', 'RandomForest')                           # grafikte kazananı kullan
sub = df_res[df_res.Model==winner_nm].set_index('Setup')[['Accuracy','F1','ROC_AUC']]
if len(sub)==2:
    ax = sub.plot(kind='bar', rot=0); ax.set_ylim(0,1); ax.set_title(f"{winner_nm}: Baseline vs Tuned (Val)")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, f"{winner_nm.lower()}_baseline_vs_tuned.png"), dpi=140); plt.close()
    print(f"[OK] Grafik kaydedildi: figures/{winner_nm.lower()}_baseline_vs_tuned.png")

# Ek özet (ikinci tablo formatı)
thr_label = f"t={best_t:.2f}" if winner_p is not None else "t=0.50"
baseline_pick = "LogReg" if winner_nm=="LogReg" else "RandomForest"
acc_b = float(df_res[(df_res.Model==baseline_pick)&(df_res.Setup=='Baseline')].Accuracy)
f1_b  = float(df_res[(df_res.Model==baseline_pick)&(df_res.Setup=='Baseline')].F1)
auc_b = float(df_res[(df_res.Model==baseline_pick)&(df_res.Setup=='Baseline')].ROC_AUC)
acc_t = float(df_res[(df_res.Model==winner_nm)&(df_res.Setup=='Tuned')].Accuracy)
f1_t  = float(df_res[(df_res.Model==winner_nm)&(df_res.Setup=='Tuned')].F1)
auc_t = float(df_res[(df_res.Model==winner_nm)&(df_res.Setup=='Tuned')].ROC_AUC)
summary2 = pd.DataFrame({'Aşama':['Önce (Baseline)', f'Sonra (Tuned, {thr_label})', 'Δ (Sonra-Önce)'],
                         'Accuracy':[acc_b, acc_t, acc_t-acc_b],
                         'F1':[f1_b, f1_t, f1_t-f1_b],
                         'ROC_AUC':[auc_b, auc_t, auc_t-auc_b]}).round(3)
print("\n=== Optimizasyon Etkisi (Validation) ==="); print(summary2.to_string(index=False))

# ---------- 16) SUBMISSION KARŞILAŞTIRMA (eski vs yeni) ----------
old = pd.read_csv("submission.csv").sort_values("PassengerId").reset_index(drop=True)   # önceki dosya
new = pd.read_csv("submission_opt.csv").sort_values("PassengerId").reset_index(drop=True) # yeni dosya
assert old["PassengerId"].equals(new["PassengerId"]), "PassengerId listeleri uyuşmuyor!"
changed_mask = old["Survived"] != new["Survived"]                                        # değişen satırlar
n_changed = int(changed_mask.sum()); total = int(len(old))                               # adet/oran
print(f"Farklı tahmin sayısı: {n_changed}/{total} ({100*n_changed/total:.2f}%)")
diff = pd.DataFrame({"PassengerId": old.loc[changed_mask, "PassengerId"],
                     "Survived_old": old.loc[changed_mask, "Survived"],
                     "Survived_new": new.loc[changed_mask, "Survived"]})
print(diff.head(10).to_string(index=False))                                              # ilk 10 fark
diff.to_csv("submission_diff.csv", index=False)                                          # değişenleri kaydet
print("[OK] Değişen satırlar kaydedildi: submission_diff.csv")
print("\nDağılım (eski vs yeni):")
print("old:\n", old["Survived"].value_counts().to_string())
print("new:\n", new["Survived"].value_counts().to_string())


