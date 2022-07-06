
##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
df = sns.load_dataset("titanic")
df.head()

df.info()
df.shape
df.describe().T

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################
df["sex"].value_counts()

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################
df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.
#########################################
df["pclass"].nunique()
len(df["pclass"].unique())

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################
df[["pclass", "parch"]].nunique()
#########################################

# Görev 6: embarked değişkeninin tipini kontrol ediniz.
# Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

df["embarked"].dtypes # object
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype # category


#########################################
# Görev 7: embarked değeri C olanların tüm bilgilerini gösteriniz.
#########################################

df[df["embarked"] == "C"].head(20)
#########################################

# Görev 8: embarked değeri S olmayanların tüm bilgilerini gösteriniz.
#########################################

df[df["embarked"] != "S"]["embarked"].head()
# kontrol amaçlı
df[df["embarked"] != "S"]["embarked"].unique()
df[df["embarked"] != "S"]["embarked"].value_counts()

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
df.columns
df["age"] < 30 #bool
df["sex"] == "female" #bool

# df[(df["age"] < 30) & (df["sex"] == "female")]["age"].max()
# df[(df["age"] < 30) & (df["sex"] == "female")]["sex"].value_counts()
# yaş bilgileri indexleriyle birlikte gelir.
df[(df["age"] < 30) & (df["sex"] == "female")]["age"].head()

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################
df[(df["fare"] > 500) | (df["age"] > 70)].head()

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################
df.isnull().sum()

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################
df.drop(columns="who", axis=1, inplace=True)

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################
type(df["deck"].mode()) # pandas.core.series.Series

deck_mode = df["deck"].mode()[0] #string değer döndürmek için [0] alındı

# type (df["deck"].mode()[0]) str
df["deck"].fillna(deck_mode, inplace=True)


#########################################
# Görev 14: age değişkenindeki boş değerleri age değişkenin medyanı ile doldurun.
df["age"].fillna(df["age"].median(), inplace=True)

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında
# sum, count, mean değerlerini bulunuz.

df.groupby(["pclass", "sex"]).agg({"survived": ["count", "sum", "mean"]})
df.pivot_table(values="survived", index=["pclass","sex"],aggfunc=["sum","count","mean"])

#Farklı bir gözlemleme şekli ile ;
df.pivot_table(values="survived", index="sex", columns="pclass", aggfunc=["count", "sum", "mean"])

#########################################

# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz.
# (apply ve lambda yapılarını kullanınız)

def age_30_flag(age):
    if age < 30:
        return 1
    else:
        return 0


# Aşağıdaki kullanım hata alır çünkü seriye değil dataframe'e uyguluyoruz,satır sütun söylemeliyiz ya da seriye uygulamalıyız
#,df.loc[:, df.columns.str.contains("age")].apply(age_30_flag).head()

# Aşağıdaki kullanım hata alır, çünkü dataframe'e uyguluyoruz, matris formunda olduğu için x["age"] gibi bir kullanım bekler:
#df.loc[:, df.columns.str.contains("age")].apply(lambda x: 1 if x < 30 else 0, axis=1).head()

# Aşağıdaki kullanım hata alır,  çünkü applymap dataframe'e uygulanıyor, ancak df["age"] ile seri döner
#df["age"].applymap(lambda x: 1 if x < 30 else 0)  -> bu hata alır

#  Aşağıdaki kullanım hata alır,  çünkü values üzerinde satır satır gezezeceğimizi ifade ederken sadece değişkeni ifade etmiyoruz; x["age"] olabilirdi
# df[["age"]].apply(lambda x: 1 if x < 30 else 0, axis=1).head()
# df[["age"]].apply(lambda x: 1 if x < 30 else 0)

# 1.Yöntem
df["age"].apply(age_30_flag).head()

# 2.Yöntem : Seriye uygulayacağımız için ayrıca axis = 1 gibi ifade kullanmaya gerek yok, hata alır!
df["age"].apply(lambda x: 1 if x < 30 else 0).head()


# 3.Yöntem : Dataframe'e applymap uygulayabiliriz
df.loc[:, df.columns.str.contains("age")].applymap(age_30_flag).head()

# 4.Yöntem : Dataframe'e apply ve lambda uygulayabiliriz
df.loc[:, df.columns.str.contains("age")].apply(lambda x: 1 if x["age"] < 30 else 0, axis=1).head()

# 5.Yöntem :
df.apply(lambda x: age_30_flag(x["age"]), axis=1).head()


#age_Flag ekleyelim:
df["age_flag"] = df.apply(lambda x: age_30_flag(x["age"]), axis=1)


#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################
df = sns.load_dataset("tips")
df.head()
df.shape

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını,
# min, max ve ortalamasını bulunuz.
df["time"].value_counts()
df.groupby("time").agg({"total_bill": ["min", "max", "mean", "sum"]})

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min","max","mean"]})

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını,
# min, max ve ortalamasını bulunuz.
df.head()
df["time"] == "Lunch"  #bool
df["sex"] == "Female"  #bool

df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").\
                    agg({"total_bill": ["sum","mean","max","min"], \
                         "tip":  ["sum","mean","max","min"]})

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?

df.head()
df[(df["size"] < 3) & (df["total_bill"] > 10)]["total_bill"].mean()
df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun.
# Her bir müşterinin ödediği totalbill ve tip in toplamını versin.

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()
#########################################

# Görev 23: Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulun.
# Bulduğunuz ortalamaların altında olanlara 0, üstünde ve eşit olanlara 1 verildiği yeni
# bir total_bill_flag değişkeni oluşturun.
df.columns
f_avg = df.loc[df["sex"] == "Female", "total_bill"].mean() # 18.06
m_avg = df.loc[df["sex"] == "Male", "total_bill"].mean() # 20.74

df["total_bill"].head()

def total_bill_flag (cinsiyet, total_bill):
    if cinsiyet == "Female":
        if total_bill < f_avg:
            return 0
        else:
            return 1
    else:
        if total_bill < m_avg:
            return 0
        else:
            return 1



df["total_bill_flag"] = df.apply(lambda x: total_bill_flag(x["sex"], x["total_bill"]), axis=1)
df.head(50)

# Dikkat !! Female olanlar için kadınlar için bulunan ortalama dikkate alıncak, male için ise erkekler için bulunan ortalama.
# parametre olarak cinsiyet ve total_bill alan bir fonksiyon yazarak başlayın. (If-else koşulları içerecek)

#########################################
# Görev 24: total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde
# olanların sayısını gözlemleyin.
#########################################
df.groupby(["sex", "total_bill_flag"]).agg({"total_bill_flag": "count"})

#########################################
# Görev 25: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve
# ilk 30 kişiyi yeni bir dataframe'e atayınız.

temp_df = df["total_bill_tip_sum"].sort_values(ascending=False).head(30)
temp_df.head()
 
temp_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
temp_df





