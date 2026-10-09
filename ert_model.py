# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:38:41 2025

@author: tunce
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#pip install pyet
import pyet
import itertools

import sys
from importlib.metadata import version, PackageNotFoundError

print("Python", sys.version.split()[0])

for pkg in ["numpy", "pandas", "matplotlib", "scikit-learn", "shap", "pyet", "xgboost", "statsmodels"]:
    try:
        print(f"{pkg}=={version(pkg)}")
    except PackageNotFoundError:
        print(f"# {pkg} kurulu değil")

class ExpandingTimeSeriesSplitTVT:
    def __init__(self, n_splits):
        if n_splits < 3:
            raise ValueError("En az 3 blok olmalı (train, val, test için)")
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits - 2

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Veriyi eşit bloklara böl
        base_fold_size = n_samples // self.n_splits
        remainder = n_samples % self.n_splits

        fold_sizes = [base_fold_size] * self.n_splits
        fold_sizes[-1] += remainder  # artanı son bloğa ekle

        # Blok sınırlarını oluştur
        current = 0
        blocks = []
        for fold_size in fold_sizes:
            start = current
            stop = current + fold_size
            blocks.append(indices[start:stop])
            current = stop

        # Expanding window split
        for i in range(self.n_splits - 2):
            train_idx = np.concatenate(blocks[:i+1])
            val_idx   = blocks[i+1]
            test_idx  = blocks[i+2]

            yield train_idx, val_idx, test_idx

def importMeteorologicalData(file_name):
    """
    Verilen dosyayı okuyarak, meteorolojik verileri bir pandas DataFrame'e döndürür.
    İlk 13 satır meta bilgi, 15. satır birimler içerir ve her ikisi de atlanır. 
    14. satır kolon isimleri olarak kullanılır.
    
    Parameters:
        file_name (str): Data klasöründeki dosyanın adı.
    
    Returns:
        pd.DataFrame: Meteorolojik verileri içeren DataFrame.
    """
    # Dosya yolunu oluştur
    file_path = os.path.join(os.getcwd(), 'data', file_name)
    
    try:
        # İlk olarak sadece kolon isimlerini almak için 14. satırı oku
        with open(file_path, 'r') as file:
            lines = file.readlines()
            column_names = lines[13].split()  # 14. satırdaki kolon isimlerini al
        
        # Asıl veriyi 15. satırdan sonra oku ve kolon isimlerini uygula
        df = pd.read_csv(
            file_path,
            delim_whitespace=True,  # Boşluklarla ayrılmış
            skiprows=15,            # İlk 15 satırı atla
            header=None,            # Kolon isimlerini elle belirleyeceğiz
            names=column_names      # Kolon isimlerini ata
        )
        
            
        date = [datetime(df["year"][i], df["mo"][i], df["da"][i]) for i in range(df.shape[0])]
        df["date"] = date
        df['date'] = df['date'].astype(str)
        df['date'] = df['date'].apply(lambda x: '2' + x[1:])
        df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df = df.set_index("date")
        
        df["tmean"] = (df.tmax + df.tmin)/2
        
        def e(t):
            return 0.6108 * np.exp(17.27*t/(t + 243.5))

        df['rh'] = 100 * df['tdew'].apply(e) / df['tmean'].apply(e)

        return df
    except Exception as e:
        print(f"Veriyi işlerken bir hata oluştu: {e}")
        return None

# Örnek kullanım
file_name = "Tokat"
df = importMeteorologicalData(file_name)

if df is not None:
    print(df.head())  # İlk birkaç satırı göster

def meteorological2et(data, method = 'Penman', lat = 40.32, elevation = 623, gwl = 120):
    # Convert Glabalstrahlung J/cm2 to MJ/m2 by dividing to 100
    meteo = pd.DataFrame({"time":data.index, "tmean":data.tmean, "tmax":data.tmax, "tmin":data.tmin, "rh":data.rh,
                          "wind":data["w-vl"], "rs":data.rad * 0.04184})
    time, tmean, tmax, tmin, rh, wind, rs = [meteo[col] for col in meteo.columns]

    lat_radians = lat * np.pi / 180  # Latitude of the meteorological station, converting from degrees to radians

    # Estimate evapotranspiration with four different methods and create a dataframe
    pet_df = pyet.calculate_all(tmean, wind, rs, elevation, lat_radians, tmax=tmax,
                                    tmin=tmin, rh=rh
                                    )

    fig, axs = plt.subplots(figsize=(13,4), ncols=2)
    pet_df.plot(ax=axs[0])
    pet_df.cumsum().plot(ax=axs[1], legend=False)

    axs[0].set_ylabel("PET [mm/day]", fontsize=12)
    axs[1].set_ylabel("Cumulative PET [mm]", fontsize=12)
    axs[0].legend(ncol=6, loc=[0,1.])
    for i in (0,1):
        axs[i].set_xlabel("Date", fontsize=12)
    
    # İstediğiniz kolonları seçin
    selected_columns = ['da', 'mo', 'year', 'prcp']
    
    # Yeni bir alt dataframe oluşturun
    subset_df = data[selected_columns]
    subset_df['PE'] = pet_df[f"{method}"]
    subset_df['SOIL_VMC'] = -9999
    subset_df['GWL'] = gwl
    # Kolon isimlerini değiştirme
    subset_df = subset_df.rename(columns={'da': 'DAY', 'mo': 'MONTH', 'year': 'YEAR', 'prcp': 'RAIN',})
    
    return subset_df
 
#ABS Generator
def absGenerator(start= '2001-01-01', end='2050-12-31', freq ='D', intercept = 0, slope = 0, fluctuation = 0, n = 10*365):
    
    dates = pd.date_range(start=start, end=end, freq=freq)
    # Sine dalgasını kullanarak mevsimsel değişimi simüle etme (yıl ortasında yüksek, yıl başında ve sonunda düşük)
    # 2*np.pi/365 yılı bir döngüde simüle eder ve faz kaymasıyla yılı ortasında yüksek çekim oluşturur
    seasonal_variation = np.sin(2 * np.pi * (dates.dayofyear) / 365 - np.pi / 2)
    
    # Günlük artış oranı (trend), örneğin her gün için 0.1 birim artış
    #daily_trend = slope * np.arange(len(dates))  # Gün sayısına göre artış
    daily_trend = np.zeros(len(dates))  # Tüm elemanları başlangıçta 0 olarak tanımla
    daily_trend[n:] = slope * np.arange(len(dates) - n)  # n adım ötelenmiş trendi hesapla


    pumping_rate = intercept + fluctuation * seasonal_variation + daily_trend
    
    # Veri çerçevesi oluşturma
    df = pd.DataFrame({'Date': dates, 'PumpingRate': pumping_rate})
    
    # Grafiği çizme
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['PumpingRate'], color= '#1f3c88')
    plt.title(f'Yeraltı Suyu Çekimi (eğim: {slope:.1e})')
    plt.xlabel('Tarih')
    plt.ylabel('Çekim Miktarı (m³/gün)')
    plt.grid(True)
    
    # Başlangıç ve bitiş tarihlerini belirleyin
    start_date = '2020-01-01'
    end_date = '2025-12-31'
    
    # 'Date' sütununu datetime formatına çevirin (eğer değilse)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Belirtilen tarih aralığında veri süzme
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Grafiği çizme
    plt.figure(figsize=(10, 5))
    plt.plot(df_filtered['Date'], df_filtered['PumpingRate'], color='#1f3c88')
    plt.title(f'Yeraltı Suyu Çekimi (eğim: {slope:.1e})')
    plt.xlabel('Tarih')
    plt.ylabel('Çekim Miktarı (m³/gün)')
    plt.grid(True)
    plt.show()
    
    output_dir = os.path.join(os.getcwd(), 'graphs')
    # Grafik kaydetme yolu
    output_path = os.path.join(output_dir, f"YAS_ABS_{slope:.1e}.png")
    # Grafiği kaydet
    #plt.savefig(output_path, dpi=900, bbox_inches="tight")  # Yüksek çözünürlük için dpi ayarı yapılır
    plt.show()
    
    return df

##########
# MODELING
##########

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap

# Sabit parametreler
intercept = 0.2  # Sabit intercept değeri
slope_values_ = np.linspace(0, 0.00005, 5)  # Slope değerlerini 0.1'den 1.0'a kadar 10 adımda artırıyoruz.
# 50 yıl için en cok egim 0.0001

# Sonuçları tutacak sözlük
slope_results_ea = []
slope_model_ea = []

general_train_r2, general_test_r2 = [], []
general_train_rmse, general_test_rmse = [], []
general_train_nrmse, general_test_nrmse = [], []
general_train_kge, general_test_kge = [], []


def kge(y_true, y_pred):
    
    r = np.corrcoef(y_true, y_pred)[0, 1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gamma = (np.std(y_pred) / np.mean(y_pred)) / (np.std(y_true) / np.mean(y_true))
    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

#slope = 0.00005

errors = []
metrics = []   
hoResults = []

all_shap_records = []   # Tüm slope × split kombinasyonlarının SHAP kayıtlarını tutacak

# Slope döngüsü
for slope in slope_values_:

    abs_df = absGenerator(start= df.index[0], end=df.index[-1], intercept = 0.2, slope = slope, fluctuation= 0.2)
    abs_df = abs_df.set_index("Date")
    
    aquimod_df = meteorological2et(df)
    aquimod_df['ABS'] = abs_df['PumpingRate']
    
    # Dosya yolunu belirleyin
    file_name = 'Observations.txt'
    file_path = os.path.join(os.getcwd(), 'lumped_model', file_name)
    
    # Veriyi tab ile ayrılmış formatta, ancak öncelikle başlıkları ekleyerek dosyaya yazalım
    with open(file_path, 'w', newline='') as f:
        # İlk satıra "NUMBER OF OBSERVATIONS"
        f.write("NUMBER OF OBSERVATIONS\n")
        
        # İkinci satıra veri sayısını yazalım
        f.write(f"{len(aquimod_df)}\n")
        
        # Dataframe'i tab ile ayrılmış formatta yazalım
        aquimod_df.to_csv(f, sep='\t', index=False)
    
    print(f"Dataframe başarıyla {file_path} dosyasına yazıldı.")
    
    # .bat dosyasının yolu
    bat_file_path = os.path.join(os.getcwd(), "RunAquimod.bat")
    os.system(bat_file_path)
    
    # Dosya yolunu belirleyin
    fao_name = 'FAO_TimeSeries1.out'
    weibull_name = 'Weibull_TimeSeries1.out'
    q2k2s1_name = 'Q2K2S1_TimeSeries1.out'
    fao_path = os.path.join(os.getcwd(), 'lumped_model', 'Output', fao_name)
    weibull_path = os.path.join(os.getcwd(), 'lumped_model', 'Output', weibull_name)
    q2k2s1_path = os.path.join(os.getcwd(), 'lumped_model', 'Output', q2k2s1_name)
    
    fao_df = pd.read_csv(
                fao_path,
                delim_whitespace=True,  # Boşluklarla ayrılmış
            )
    
    weibull_df = pd.read_csv(
                weibull_path,
                delim_whitespace=True,  # Boşluklarla ayrılmış
            )
    
    q2k2s1_df = pd.read_csv(
                q2k2s1_path,
                delim_whitespace=True,  # Boşluklarla ayrılmış
            )
    
    ###
    
    plt.figure(figsize=(12, 6))  # Geniş format
    
    plt.plot(
        aquimod_df.index,
        q2k2s1_df["GWL(m)"],
        linestyle='-',
        linewidth=1.8,
        color='navy',
        label='Groundwater Level'
    )
    
    # Eksensel etiketler okunaklı ve profesyonel
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Groundwater Level (m)", fontsize=16)
    
    # Tick (eksen değerleri) fontları büyütüldü
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Grid hafif gri, daha düzgün bir çizgi tipi
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    
    # Legend sade ve okunaklı
    plt.legend(fontsize=14, loc='best', frameon=False)
    
    # Plot başlığı kullanılmıyor — akademik görsellerde genelde açıklama figür altında olur
    plt.tight_layout()
    
    # Kayıt klasörü ve dosya yolu
    output_dir = os.path.join(os.getcwd(), 'graphs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"YASS_{slope:.1e}.png")
    
    # Yüksek çözünürlük ve kenarsız çıktı
    plt.savefig(output_path, dpi=900, bbox_inches="tight", pad_inches=0.05),
    plt.show()
    plt.close()
    
    ###
    
    results = pd.concat([fao_df, weibull_df, q2k2s1_df], axis=1)
    results = results.loc[:, ~results.columns.duplicated()]
    
    #Data for XAI
    results.index = aquimod_df.index
    results['RAIN'] = aquimod_df['RAIN']
    results['PE'] = aquimod_df['PE']
    #results['Ro'] = results['q_ro(mm/d)'] + (results['Q_2(m3/d)'] / 3) # to mm/day
    results['Ro'] = (results['q_ro(mm/d)'] * 3) + results['Q_2(m3/d)'] # to m3/day
    
    results_with_ABS = results
    results_with_ABS['ABS'] = abs_df['PumpingRate']
    slope_model_ea.append(results_with_ABS)
    
    results_daily_df = pd.DataFrame()
    results_daily_df['P'] = aquimod_df['RAIN']
    results_daily_df['T'] = df['tmean']
    results_daily_df['GWL'] = results['GWL(m)']
    results_daily_df['Ro'] = results['Ro']
    results_daily_df['ETp'] = results['PE']
    #results_daily_df['ETa'] = results['E_a(mm/d)']
    
    # Aylık toplam alınacak değişkenler
    toplam_alanacaklar = ["P", #"Ro", 
                          "ETp", 
                          #"ETa"
                          ]
    # Aylık ortalama alınacak değişkenler
    ortalama_alanacaklar = ["T", "GWL", "Ro"]
    # Yeni DataFrame oluşturma
    results_df = results_daily_df.resample("M").agg({
        **{col: "sum" for col in toplam_alanacaklar},
        **{col: "mean" for col in ortalama_alanacaklar}
    })
    
    #Lag parametreler
    results_df['PLag-1'] = results_df['P'].shift(1)
    results_df['TLag-1'] = results_df['T'].shift(1)
    #results_df['P'] = results_df['P'].shift(1)
    #results_df['T'] = results_df['T'].shift(1)
    results_df['GWLlag-1'] = results_df['GWL'].shift(1)
    
    #XAI model
    results_XGB_df = pd.DataFrame()
    """
    params = {
        "n_estimators": 300,
        "max_depth": 16,
        #"min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }
    """

    
    """
    Best TimeSeries CV R2: 0.9128124600607597
    Best params: {'n_estimators': 300, 'max_depth': 16, 'min_samples_leaf': 1}
    """
       
    # Tarih aralıkları
    start = "2011-01-01"
    end = "2100-12-31"
    overall_data = results_df.loc[start:end]
    
    """
    split_ratio = 0.80
    split_index = int(len(overall_data) * split_ratio)
    
    # Zamansal sırayı bozmadan ayırma
    trainVal_data = overall_data.iloc[:split_index]
    test_data     = overall_data.iloc[split_index:]
    
    from sklearn.model_selection import TimeSeriesSplit
    n_splits = 5

    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    train_datas = []
    val_datas = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(trainVal_data)):
        
        train_fold = trainVal_data.iloc[train_idx]
        val_fold   = trainVal_data.iloc[val_idx]
        
        train_datas.append(train_fold)
        val_datas.append(val_fold)
        
        print(f"\nFold {fold+1}")
        print("Train:", train_fold.index.min(), "-", train_fold.index.max())
        print("Val  :", val_fold.index.min(), "-", val_fold.index.max())
    
    #split = 0
    all_test_preds = []
    """
    
    splitter = ExpandingTimeSeriesSplitTVT(n_splits=5)
    
    train_datas = []
    validation_datas = []
    test_datas = []
    
    # Splitleri üret
    for train_idx, val_idx, test_idx in splitter.split(overall_data):
    
        train_datas.append(overall_data.iloc[train_idx])
        validation_datas.append(overall_data.iloc[val_idx])
        test_datas.append(overall_data.iloc[test_idx])
    
    print(f"Toplam split sayısı: {len(train_datas)}")
    
    #split = 0
    n_splits = len(train_datas)
    
    # Parametre grid
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [5, 10, 16, None],
        "min_samples_split": [2, 5],
        #"min_samples_leaf": [1, 2, 4],
        #"max_features": ["sqrt", "log2", None],
        #"bootstrap": [False],   # genelde False kullanılır
        "random_state": [42]
    }
    
    # Grid kombinasyonları
    param_combinations = list(itertools.product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["min_samples_split"]
    ))
    
    
    for split in range(n_splits):
        
        print(f"\n--- Split {split} ---")
        
        """
        train_data = train_datas[split]
        val_data = val_datas[split]
        
        # Eğitim ve test setlerinin ayrıştırılması
        y_train = train_data["GWL"].values
        X_train = train_data.drop(columns=["GWL"])
        
        y_test = test_data["GWL"].values
        X_test = test_data.drop(columns=["GWL"])
    
        # Model oluşturma ve eğitim
        model = ExtraTreesRegressor(**params)
        model.fit(X_train, y_train)
        
        # For Total Error, Bias and Variance
        preds = model.predict(X_test)
        all_test_preds.append(preds)
        """
        
        train_data = train_datas[split]
        val_data = validation_datas[split]
        test_data = test_datas[split]
    
        # Eğitim ve test setlerinin ayrıştırılması
        y_train = train_data["GWL"].values
        X_train = train_data.drop(columns=["GWL"])
        
        y_val = val_data["GWL"].values
        X_val = val_data.drop(columns=["GWL"])
        
        y_test = test_data["GWL"].values
        X_test = test_data.drop(columns=["GWL"])
        
        best_score = np.inf
        best_params = None
        
        # 🔹 Grid Search (manual, validation bazlı)
        for n_estimators, max_depth, min_samples_split in param_combinations:
        
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )
        
            model.fit(X_train, y_train)
        
            y_pred_val = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        
            if rmse < best_score:
                best_score = rmse
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split
                }
        
        print("Best params:", best_params)
        print("Validation RMSE:", best_score)
        
        # 🔹 Final model (train + val ile yeniden eğit)
        X_train = np.vstack([X_train, X_val])
        y_train = np.concatenate([y_train, y_val])
        
        model = ExtraTreesRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        """
        # Model oluşturma ve eğitim
        model = ExtraTreesRegressor(**params)
        model.fit(X_train, y_train)
        """
        
        # Tahminler
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Doğruluk metrikleri
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
    
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_rmse = float(train_rmse)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_rmse = float(test_rmse)
    
        general_train_r2.append(train_r2)
        general_test_r2.append(test_r2)
        general_train_rmse.append(train_rmse)
        general_test_rmse.append(test_rmse)
        
        # --- NRMSE (standart sapmaya göre normalize) ---
        nrmse_train = train_rmse / np.std(y_train)
        nrmse_train = float(nrmse_train)
        nrmse_test = test_rmse / np.std(y_test)
        nrmse_test = float(nrmse_test)
        #If this value is larger than 1, you'd obtain a better model by simply
        #generating a random time series of the same mean and standard deviation as Y.
        
        general_train_nrmse.append(nrmse_train)
        general_test_nrmse.append(nrmse_test)
        
        # --- KGE ---
        kge_train = kge(y_train, y_train_pred)
        kge_train = float(kge_train)
        kge_test = kge(y_test, y_test_pred)
        kge_test = float(kge_test)
        
        general_train_kge.append(kge_train)
        general_test_kge.append(kge_test)
        # correlation, average rate, rate of CV. If this value 1 is a perfect model. Like NSE
        
        hoResults.append({
            "slope": slope,
            "split": split,
            "best_params": best_params,
            "val_rmse": best_score,
            "test_rmse": test_rmse
        })
        
        metrics.append({
            "slope": slope,
            "fold": split,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_nrmse": nrmse_train,
            "test_nrmse": nrmse_test,
            "train_kge": kge_train,
            "test_kge": kge_test
        })

        
        ############
        # SHAP
        ############
      
        # Calculate Shapley values
        #explainer = shap.Explainer(model)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        # SHAP interaction değerlerini hesapla
        shap_interaction_values = explainer.shap_interaction_values(X_test)
        
        ###################################################
        # --- Combined dependence plot için kayıt oluştur ---
        record_df = X_test.copy()
        for col in X_test.columns:
            record_df[f"shap_{col}"] = shap_values.values[:, X_test.columns.get_loc(col)]
        record_df["slope"] = slope
        record_df["split"] = split
        all_shap_records.append(record_df)
        ####################################################
        
        
        # Lokal SHAP değerlerini dataframe'e ekle
        shap_df = pd.DataFrame(shap_values.values, columns=[f"shap_{col}" for col in X_test.columns], index=X_test.index)
    
        def monthly_aggregation(results, start, end, mean_cols, sum_cols):
            # Belirtilen tarih aralığında filtreleme
            subset = results.loc[start:end]
            
            # Aylık bazda yeniden örnekleme yaparak istenen istatistikleri hesapla
            monthly_results = subset.resample("M").agg({
                **{col: "mean" for col in mean_cols},  # Ortalama alınacak kolonlar
                **{col: "sum" for col in sum_cols}     # Toplam alınacak kolonlar
            })
            
            return monthly_results
        
        # 
        mean_columns = ["GWL(m)"]  # Aylık ortalaması alınacak kolonlar
        column_list = results.columns.tolist()
        filtered_list = [item for item in column_list if item != "GWL(m)"]
        monthly_results = monthly_aggregation(results, start, end, mean_columns, filtered_list)
    
        # Orijinal veri çerçevesi ile birleştir
        monthly_results = pd.concat([monthly_results.loc[shap_df.index], shap_df], axis=1)
        monthly_results = pd.concat([monthly_results, X_test], axis=1)
        
        abs_subset = aquimod_df.loc[start:end]
        monthly_results["ABSvalue"] = abs_subset["ABS"].resample("M").mean()
    
        slope_results_ea.append(monthly_results)
        
        """
        # Global Shapley özet grafiği (summary plot)
        shap.summary_plot(shap_values.values, X_test, feature_names=X_test.columns.tolist(), show=False) 
        # Grafik kaydetme yolu
        output_path = os.path.join(output_dir, f"Shap_Summary_{slope:.1e}_{split}.png")
        # Grafiği kaydet
        plt.savefig(output_path, dpi=900, bbox_inches="tight")  # Yüksek çözünürlük için dpi ayarı yapılı
        plt.show()
        plt.close()
        """
        
        #################
        
        shap_vals = shap_values.values
        feature_names = X_test.columns.tolist()
        
        # Mean |SHAP|
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        # Sıralama
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_shap = shap_vals[:, sorted_idx]
        sorted_mean_abs = mean_abs_shap[sorted_idx]
        
        # Kümülatif yüzde
        cumulative = np.cumsum(sorted_mean_abs)
        cumulative_percent = 100 * cumulative / cumulative[-1]
        
        # ---- FIGURE ----
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6),
            gridspec_kw={"width_ratios": [1.2, 1]}
        )
        
        # =========================
        # (a) SHAP SUMMARY
        # =========================
        plt.sca(ax1)
        shap.summary_plot(
            sorted_shap,
            X_test[sorted_features],
            feature_names=sorted_features,
            show=False,
            color_bar=True
        )
        ax1.set_title("(a)", loc="left")
        
        # =========================
        # (b) BAR PLOT
        # =========================
        y_pos = np.arange(len(sorted_features))
        
        bars = ax2.barh(y_pos, sorted_mean_abs)
        
        # Y ekseni kaldır (etiketler solda zaten var)
        ax2.set_yticks([])
        ax2.set_ylabel("")
        
        ax2.invert_yaxis()
        ax2.set_xlabel("Mean |SHAP value|")
        ax2.set_title("(b)", loc="left")
        
        # Maks değer
        max_val = max(sorted_mean_abs)
        
        # Yüzde etiketleri (taşmayı önleyen akıllı yerleşim)
        for i, (val, perc) in enumerate(zip(sorted_mean_abs, cumulative_percent)):
            
            # Eğer barın sonuna sığarsa dışarı yaz
            if val < 0.85 * max_val:
                ax2.text(
                    val + max_val * 0.01,
                    i,
                    f"{perc:.0f}%",
                    va="center",
                    ha="left",
                    fontsize=9
                )
            else:
                # büyük barlarda içeri yaz
                ax2.text(
                    val - max_val * 0.02,
                    i,
                    f"{perc:.0f}%",
                    va="center",
                    ha="right",
                    fontsize=9,
                    color="white"
                )
        
        # X limitini biraz genişlet (etiketler için)
        ax2.set_xlim(0, max_val * 1.1)
        
        plt.tight_layout()
        
        # Kaydet
        output_path = os.path.join(output_dir, f"Shap_Combined_{slope:.1e}_{split}.png")
        plt.savefig(output_path, dpi=900, bbox_inches="tight")
        
        plt.show()
        plt.close()
        
        #################
        

        # SHAP interaction summary plot çizdir
        shap.summary_plot(shap_interaction_values, X_test,)# feature_names=X_test.columns.tolist())
        plt.close()

        
        
        """
        shap.plots.partial_dependence(
            "Ro", model.predict, X_test, ice=True,
            model_expected_value=True, feature_expected_value=True
        )
        # FEATURES: ETa , GWLlag-1 , P , PLag-1 , Ro
        """
        
        """
        # Tüm değişkenler için ICE plot oluştur
        for i in X_test.columns.tolist():
            shap.plots.partial_dependence(
                i, model.predict, X_test, ice=False,
                #model_expected_value=True, feature_expected_value=True,
                show=False,
            )
            #plt.title(f"ICE Plot: {i}")
            
            output_path = os.path.join(output_dir, f"ICE_{i}_{slope:.1e}.png")
            plt.savefig(output_path, dpi=900, bbox_inches="tight")  # Yüksek çözünürlük için dpi ayarı yapılı
            plt.show()
            plt.close()
        """
        
        
        # Tüm değişkenler için SHAP dependence plot oluştur
        for feature in X_test.columns.tolist():
            plt.figure() #plt.figure(figsize=(6, 4))
            shap.dependence_plot(feature,
                                 shap_values.values,
                                 X_test,
                                 #interaction_index="FLD1",
                                 show=False,
                                 )
            #plt.title(f"SHAP Dependence Plot: {feature}")
            
            output_path = os.path.join(output_dir, f"SHAP_dependence_{feature}_{slope:.1e}_{split}.png")
            plt.savefig(output_path, dpi=900, bbox_inches="tight")  # Yüksek çözünürlük için dpi ayarı yapılı
            plt.show()
            plt.close()
            
        
        # Renklendirme yapılacak öznitelik (örnek: "MedInc")
        color_feature = y_test
        # Her öznitelik için grafik çizimi
        for i, feature in enumerate(X_test.columns):
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                X_test.iloc[:, i],  # X ekseni: Öznitelik değerleri
                shap_values[:, i].values,  # Y ekseni: Shapley değerleri
                c=y_test,  # Renk: Belirtilen özniteliğin Shapley değerleri
                cmap="coolwarm",  # Renk haritası
                #alpha=0.6,
                edgecolor=None
            )
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
            plt.title(f"Shapley Value vs {feature} (Colored by GWL)", fontsize=14)
            plt.xlabel(f"{feature}", fontsize=12)
            plt.ylabel("Shapley Value", fontsize=12)
            plt.grid(True, linestyle='--', #alpha=0.7
                     )
            # Renk skalası (colorbar) ekleme
            cbar = plt.colorbar(scatter)
            cbar.set_label("GWL", fontsize=12)
            plt.tight_layout()
            plt.tight_layout()
            # Grafik kaydetme yolu
            output_path = os.path.join(output_dir, f"Shap_{feature}_{slope:.1e}_{split}.png")
            # Grafiği kaydet
            plt.savefig(output_path, dpi=900, bbox_inches="tight")  # Yüksek çözünürlük için dpi ayarı yapılı
            plt.show()
            plt.close()
            
        # SHAP değerlerini numpy array olarak al
        shap_array = shap_values.values  # (n_samples, n_features) boyutunda bir array
        # Her özelliğin ortalama SHAP değerini hesapla
        mean_shap_values = np.mean(np.abs(shap_array), axis=0)  # Mutlak değer üzerinden ortalama
        # Listeye çevir
        mean_shap_list = mean_shap_values.tolist()
        print(slope)
        print("['P', 'Ro', 'ETp', 'T', 'PLag-1', 'TLag-1', 'GWLlag-1']")
        print(mean_shap_list)
        
    
    """
    # Bias and Variance
    
    # (n_models, n_test_samples)
    all_test_preds = np.array(all_test_preds)
    # Ortalama tahmin
    mean_pred = np.mean(all_test_preds, axis=0)
    # Bias^2
    bias_sq = np.mean((mean_pred - y_test)**2)
    # Variance
    variance = np.mean(np.var(all_test_preds, axis=0))
    # Total error
    total_error = bias_sq + variance
    
    print("Bias^2     :", bias_sq)
    print("Variance   :", variance)
    print("Total Error:", total_error)
    
    error = {
        "slope": slope,
        "bias2": bias_sq,
        "variance": variance,
        "totalError": total_error
        }
    errors.append(error)
    """



# metrics list → DataFrame
df_metrics = pd.DataFrame(metrics)

# CSV olarak kaydet
df_metrics.to_csv("metrics_resultsXTR.csv", index=False)

##########

df_hoResults = pd.DataFrame(hoResults)


df_hoResults.to_csv("ho_results_XTR.csv", index=False)


######################
"""
acc = pd.DataFrame()
acc["R2"] = general_test_r2 + general_train_r2
acc["KGE"] = general_test_kge + general_train_kge
acc["NRMSE"] = general_test_nrmse + general_train_nrmse
acc["RMSE"] = general_test_rmse + general_train_rmse
"""

##########################################
# COMBINED SHAP DEPENDENCE PLOTS (15 model tek grafikte)
##########################################

from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "Times New Roman",

    "font.size": 42,

    "axes.titlesize": 30,
    "axes.titleweight": "bold",

    "axes.labelsize": 23,
    "axes.labelweight": "bold",

    "xtick.labelsize": 22,
    "ytick.labelsize": 22,

    "legend.fontsize": 25
})

all_shap_df = pd.concat(all_shap_records, axis=0, ignore_index=True)
all_shap_df.to_csv("ERT_scenario_split_shap_table.csv", index=False)

feature_cols = [c for c in all_shap_df.columns 
                 if not c.startswith("shap_") and c not in ("slope", "split")]

# Split -> sembol eşlemesi
markers = ['o', '^', 's', 'D', 'P', 'X']  # gerekirse split sayısı arttıkça genişletilebilir
splits_unique = sorted(all_shap_df['split'].unique())
split_marker_map = {sp: markers[i] for i, sp in enumerate(splits_unique)}

# Slope -> renk eşlemesi
slopes_unique = sorted(all_shap_df['slope'].unique())
cmap = plt.cm.viridis
slope_color_map = {
    s: cmap(i / max(len(slopes_unique) - 1, 1)) for i, s in enumerate(slopes_unique)
}

for feature in feature_cols:
    fig, ax = plt.subplots(figsize=(8, 6))

    for s in slopes_unique:
        for sp in splits_unique:
            subset = all_shap_df[(all_shap_df['slope'] == s) & (all_shap_df['split'] == sp)]
            if subset.empty:
                continue
            ax.scatter(
                subset[feature],
                subset[f"shap_{feature}"],
                color=slope_color_map[s],
                marker=split_marker_map[sp],
                alpha=0.6,
                edgecolor='k',
                linewidth=0.3,
                s=35
            )

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(feature, fontsize=26)
    ax.set_ylabel(f"SHAP value of {feature}", fontsize=26)
    #ax.set_title(f"Combined SHAP Dependence Graph: {feature}", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"Shap_Combined_Dependence_{feature}_ERT.png")
    plt.savefig(output_path, dpi=900, bbox_inches="tight")
    plt.show()
    plt.close()


#########
# LEJAND FIGURE
################

fig_leg, ax_leg = plt.subplots(figsize=(8, 2.8))
ax_leg.axis('off')

# Split legend
split_handles = [
    Line2D([0], [0],
           marker=split_marker_map[sp],
           color='gray',
           linestyle='',
           markersize=9,
           label=f"Split {sp+1}")
    for sp in splits_unique
]

# Slope legend
slope_handles = [
    Line2D([0], [0],
           marker='o',
           color=slope_color_map[s],
           linestyle='',
           markersize=9,
           label=f"{s:.1e}")
    for s in slopes_unique
]

# Sol tarafta Split
leg1 = ax_leg.legend(
    handles=split_handles,
    title="Split (marker)",
    loc='center left',
    bbox_to_anchor=(0.00, 0.5),
    frameon=False,
    fontsize=18,
    title_fontsize=20
)
ax_leg.add_artist(leg1)

# Sağ tarafta Slope
leg2 = ax_leg.legend(
    handles=slope_handles,
    title="Slope (color)",
    loc='center left',
    bbox_to_anchor=(0.45, 0.5),
    frameon=False,
    fontsize=18,
    title_fontsize=20
)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "Combined_SHAP_Legend_ERT.png"),
    dpi=900,
    bbox_inches='tight'
)

plt.show()
plt.close()


"""
all_shap_df = pd.concat(all_shap_records, axis=0, ignore_index=True)

feature_cols = [c for c in all_shap_df.columns 
                 if not c.startswith("shap_") and c not in ("slope", "split")]

# Split -> sembol eşlemesi
markers = ['o', '^', 's', 'D', 'P', 'X']  # gerekirse split sayısı arttıkça genişletilebilir
splits_unique = sorted(all_shap_df['split'].unique())
split_marker_map = {sp: markers[i] for i, sp in enumerate(splits_unique)}

# Slope -> renk eşlemesi
slopes_unique = sorted(all_shap_df['slope'].unique())
cmap = plt.cm.viridis
slope_color_map = {
    s: cmap(i / max(len(slopes_unique) - 1, 1)) for i, s in enumerate(slopes_unique)
}

for feature in feature_cols:
    fig, ax = plt.subplots(figsize=(8, 6))

    for s in slopes_unique:
        for sp in splits_unique:
            subset = all_shap_df[(all_shap_df['slope'] == s) & (all_shap_df['split'] == sp)]
            if subset.empty:
                continue
            ax.scatter(
                subset[feature],
                subset[f"shap_{feature}"],
                color=slope_color_map[s],
                marker=split_marker_map[sp],
                alpha=0.6,
                edgecolor='k',
                linewidth=0.3,
                s=35
            )

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel(f"SHAP value ({feature})", fontsize=12)
    ax.set_title(f"Combined SHAP Dependence Graph: {feature}", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)

    # İki ayrı lejant: şekil (split) ve renk (slope)
    split_legend_elements = [
        Line2D([0], [0], marker=split_marker_map[sp], color='gray', linestyle='',
               markersize=8, label=f"Split {sp + 1}")
        for sp in splits_unique
    ]
    slope_legend_elements = [
        Line2D([0], [0], marker='o', color=slope_color_map[s], linestyle='',
               markersize=8, label=f"{s:.1e}")
        for s in slopes_unique
    ]

    leg1 = ax.legend(handles=split_legend_elements, title="Split (marker)",
                      loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=slope_legend_elements, title="Slope (color)",
              loc='upper left', bbox_to_anchor=(1.02, 0.55))

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"Shap_Combined_Dependence_{feature}_ERT.png")
    plt.savefig(output_path, dpi=900, bbox_inches="tight")
    plt.show()
    plt.close()
"""


"""
# Hollow figure

for feature in feature_cols:
    fig, ax = plt.subplots(figsize=(8, 6))

    for s in slopes_unique:
        for sp in splits_unique:
            subset = all_shap_df[(all_shap_df['slope'] == s) & (all_shap_df['split'] == sp)]
            if subset.empty:
                continue
            ax.scatter(
                subset[feature],
                subset[f"shap_{feature}"],
                facecolor='none',                 # içi boş
                edgecolor=slope_color_map[s],      # renk artık stroke'ta
                marker=split_marker_map[sp],
                linewidth=1.1,                     # stroke kalınlığı - görünürlük için önemli
                alpha=0.75,                        # üst üste binmede yine de yardımcı olur
                s=45
            )

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel(f"SHAP value ({feature})", fontsize=12)
    ax.set_title(f"Combined SHAP Dependence Graph: {feature}", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)

    # Lejantlar da hollow style'a uygun olsun
    split_legend_elements = [
        Line2D([0], [0], marker=split_marker_map[sp], color='gray',
               markerfacecolor='none', markeredgecolor='gray',
               linestyle='', markersize=9, markeredgewidth=1.3,
               label=f"Split {sp + 1}")
        for sp in splits_unique
    ]
    slope_legend_elements = [
        Line2D([0], [0], marker='o', color=slope_color_map[s],
               markerfacecolor='none', markeredgecolor=slope_color_map[s],
               linestyle='', markersize=9, markeredgewidth=1.3,
               label=f"{s:.1e}")
        for s in slopes_unique
    ]

    leg1 = ax.legend(handles=split_legend_elements, title="Split (şekil)",
                      loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=slope_legend_elements, title="Slope (renk)",
              loc='upper left', bbox_to_anchor=(1.02, 0.55))

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"Shap_Combined_Dependence_{feature}_hlw_ERT.png")
    plt.savefig(output_path, dpi=900, bbox_inches="tight")
    plt.show()
    plt.close()
"""


'''
###########################################
# Runoff - Precipitation issue
###########################################

# ============================================================
# 1. P - Ro ilişkisi
# ============================================================

corr_pr = results_df[["P", "Ro"]].corr(method="pearson").iloc[0, 1]
spearman_pr = results_df[["P", "Ro"]].corr(method="spearman").iloc[0, 1]

print("Pearson(P, Ro)   =", corr_pr)
print("Spearman(P, Ro)  =", spearman_pr)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

tmp = results_df[["P", "Ro"]].dropna()

lr = LinearRegression()
lr.fit(tmp[["P"]], tmp["Ro"])

ro_pred = lr.predict(tmp[["P"]])

print("Ro = a + bP")
print("a =", lr.intercept_)
print("b =", lr.coef_[0])
print("R2 =", r2_score(tmp["Ro"], ro_pred))


import numpy as np
import pandas as pd

# ============================================================
# BLOK A) COLLINEARITY TANISI: P ile Ro ne kadar orantili?
#          (results_df hazir oldugu her slope/split donguesunde calistirin)
# ============================================================
def collinearity_report(results_df, feature_cols=("P", "PLag-1", "Ro",
                                                    "ETp", "T", "TLag-1",
                                                    "GWLlag-1"),
                         target_col="GWL"):
    """
    1) Ham (marjinal) Pearson korelasyonlari: P-GWL pozitif olmali.
    2) P ile Ro arasindaki korelasyon: yuksekse (|r|>0.6-0.7) collinearity
       ciddi demektir ve SHAP/coefficient isaretlerinin guvenilirligini
       dusurur -- bu tam da makalenizde soylediginiz seyin nicel kaniti.
    3) VIF (Variance Inflation Factor): VIF(P) > 5-10 ise, P'nin katsayi/
       SHAP isaretinin Ro'nun varligina bagli olarak degisebilecegini
       istatistiksel olarak destekler.
    """
    df = results_df[list(feature_cols) + [target_col]].dropna()

    print("\n--- Marjinal (ham) Pearson korelasyonlari ile", target_col, "---")
    print(df.corr()[target_col].sort_values(ascending=False))

    print("\n--- P ile diger degiskenler arasindaki korelasyon ---")
    print(df.corr()["P"].sort_values(ascending=False))

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X = df[list(feature_cols)].values
        vif = pd.Series(
            [variance_inflation_factor(X, i) for i in range(X.shape[1])],
            index=feature_cols, name="VIF"
        )
        print("\n--- Variance Inflation Factor (VIF) ---")
        print(vif.sort_values(ascending=False))
        print("(VIF > 5-10 -> ciddi collinearity; ozellikle P ve Ro icin bekleniyor)")
    except ImportError:
        print("statsmodels yuklu degil: pip install statsmodels --break-system-packages")

    return df.corr()


# ============================================================
# BLOK B) SUPPRESSION-EFFECT TESTI (basit OLS ile isaret kirilmasi)
#          Bu, SHAP'e hic ihtiyac duymadan hakemi ikna edebilecek
#          en dogrudan ve klasik istatistiksel kanittir.
# ============================================================
def suppression_effect_test(results_df, target_col="GWL"):
    """
    Adim 1: GWL ~ P (tek basina)          -> katsayi beklenen isaretiyle (pozitif) cikar
    Adim 2: GWL ~ P + Ro (birlikte)       -> Ro, P'nin varyansinin cogunu
                                              acikladigi icin P'nin katsayisi
                                              KUCULEBILIR, hatta isaret
                                              DEGISTIREBILIR (suppression effect,
                                              bkz. Conger 1974; Friedman & Wall 2005)
    Bu davranisin gozlenmesi, makalenizdeki aciklamayi (Ro'nun P'nin
    hidrolojik bilgisini buyuk olcude "yuttugu") dogrudan destekler.
    """
    import statsmodels.api as sm

    df = results_df[["P", "Ro", target_col]].dropna()

    X1 = sm.add_constant(df[["P"]])
    m1 = sm.OLS(df[target_col], X1).fit()
    print("\n--- Model 1: GWL ~ P ---")
    print(m1.params, "\nR2 =", m1.rsquared)

    X2 = sm.add_constant(df[["P", "Ro"]])
    m2 = sm.OLS(df[target_col], X2).fit()
    print("\n--- Model 2: GWL ~ P + Ro ---")
    print(m2.params, "\nR2 =", m2.rsquared)

    print("\nP katsayisi degisimi: {:+.5f} -> {:+.5f}".format(
        m1.params["P"], m2.params["P"]))
    print("Isaret degisti mi?", np.sign(m1.params["P"]) != np.sign(m2.params["P"]))

    return m1, m2


# ============================================================
# BLOK C) SHAP INTERACTION DEKOMPOZISYONU
#          shap_interaction_values ZATEN hesaplaniyor (satir ~619) ama
#          su an sadece summary plot icin kullaniliyor. Bu matristen
#          P'nin "ana etkisi" (main effect) ile "Ro ile etkilesimi"
#          (interaction effect) ayristirilabilir.
# ============================================================
def shap_interaction_decomposition(shap_interaction_values, X_test,
                                     feature_a="P", feature_b="Ro"):
    """
    shap_interaction_values boyutu: (n_samples, n_features, n_features)
    Kosegen (diagonal) terimler -> her degiskenin SAF ana etkisi
    Kosegen disi terimler        -> ikili etkilesim (paylasilan) etkisi

    Beklenti: P'nin ana etkisi (diagonal) pozitif veya ~0 iken,
    P-Ro etkilesim terimi guclu NEGATIF olabilir; toplam SHAP
    (main + tum interactionlar) o zaman negatif gorunur. Bu durum
    "negatif iliski" gibi gorunenin aslinda P-Ro paylasiminin bir
    urunu oldugunu, P'nin tek basina GWL'yi dusurmedigini gosterir.
    """
    cols = X_test.columns.tolist()
    ia = cols.index(feature_a)
    ib = cols.index(feature_b)

    main_effect_a = shap_interaction_values[:, ia, ia]
    interaction_ab = shap_interaction_values[:, ia, ib] + shap_interaction_values[:, ib, ia]
    total_shap_a = shap_interaction_values[:, ia, :].sum(axis=1)  # main + tum etkilesimler

    report = pd.DataFrame({
        f"{feature_a}_main_effect": main_effect_a,
        f"{feature_a}_x_{feature_b}_interaction": interaction_ab,
        f"{feature_a}_total_shap": total_shap_a,
    }, index=X_test.index)

    print(f"\n--- {feature_a} icin SHAP ayristirmasi ---")
    print(report.mean())
    print("\nOrtalama ana etki isareti :", np.sign(report[f"{feature_a}_main_effect"].mean()))
    print("Ortalama etkilesim isareti:", np.sign(report[f"{feature_a}_x_{feature_b}_interaction"].mean()))
    print("Ortalama toplam SHAP isareti:", np.sign(report[f"{feature_a}_total_shap"].mean()))

    return report


# ============================================================
# BLOK D) EN GUCLU KANIT: Ro'yu bilesenlerine ayirip yeniden egitmek
#          results dataframe'inde q_ro(mm/d) (yuzeysel) ve Q_2(m3/d)
#          (2. depo = baz akim) zaten AYRI duruyor -- toplamadan once
#          yakalayip ayri feature olarak modele verin.
# ============================================================
def build_decomposed_feature_set(results, results_daily_df):
    """
    Orijinal script'te satir 341-342'de:
        results['Ro'] = (results['q_ro(mm/d)'] * 3) + results['Q_2(m3/d)']
    seklinde tek degiskende toplaniyor. Bunun yerine SurfaceRunoff ve
    Baseflow'u ayri ayri aylik seriye ekleyin:
    """
    results_daily_df = results_daily_df.copy()
    results_daily_df["SurfaceRunoff"] = results["q_ro(mm/d)"] * 3   # m3/day
    results_daily_df["Baseflow"] = results["Q_2(m3/d)"]             # m3/day
    # NOT: results_daily_df['Ro'] satirini KALDIRMAYIN, karsilastirma icin tutun

    # Aylik agregasyon (mevcut mantikla ayni, ortalama alarak)
    results_monthly = results_daily_df.resample("M").agg({
        "P": "sum", "ETp": "sum", "T": "mean",
        "GWL": "mean", "Ro": "mean",
        "SurfaceRunoff": "mean", "Baseflow": "mean",
    })
    results_monthly["PLag-1"] = results_monthly["P"].shift(1)
    results_monthly["TLag-1"] = results_monthly["T"].shift(1)
    results_monthly["GWLlag-1"] = results_monthly["GWL"].shift(1)

    print("\n--- Baseflow / SurfaceRunoff ile P, GWL korelasyonlari ---")
    print(results_monthly[["P", "SurfaceRunoff", "Baseflow", "Ro", "GWL"]].corr()["GWL"])

    return results_monthly


# ============================================================
# BLOK E) TAM KOSUM: Ro'yu SurfaceRunoff+Baseflow olarak ayirip
#          AYNI ExtraTreesRegressor mimarisiyle yeniden egitim ve
#          SHAP karsilastirmasi (Blok D'nin calisir hali)
# ============================================================
def compare_feature_sets(overall_data_with_Ro, model_params, target_col="GWL",
                          test_ratio=0.2, random_state=42):
    """
    overall_data_with_Ro: 'Ro', 'SurfaceRunoff', 'Baseflow' kolonlarinin
        HEPSININ bulundugu overall_data (asagidaki entegrasyon notuna bakin,
        bu kolonlari once results_daily_df/results_df'e eklemeniz gerekiyor).
    model_params: orijinal grid-search'unuzden cikan best_params sozlugu
        (n_estimators, max_depth, min_samples_split).

    Iki paralel model egitir:
      (1) ORIJINAL   : ..., P, PLag-1, Ro, ...            (Ro birlesik)
      (2) DEKOMPOZE   : ..., P, PLag-1, SurfaceRunoff, Baseflow, ...  (Ro yok)
    Ayni kronolojik train/test bolmesini kullanir, ayni model mimarisiyle
    (ExtraTreesRegressor) egitir ve P degiskeninin SHAP dependence egilimini
    iki senaryoda karsilastirir.
    """
    from sklearn.ensemble import ExtraTreesRegressor
    import shap

    data = overall_data_with_Ro.dropna().copy()
    split_idx = int(len(data) * (1 - test_ratio))

    def _fit_and_get_shap(df_subset, drop_cols):
        d = df_subset.drop(columns=drop_cols, errors="ignore")
        train = d.iloc[:split_idx]
        test = d.iloc[split_idx:]
        y_train = train[target_col].values
        X_train = train.drop(columns=[target_col])
        y_test = test[target_col].values
        X_test = test.drop(columns=[target_col])

        m = ExtraTreesRegressor(**model_params, random_state=random_state, n_jobs=-1)
        m.fit(X_train, y_train)

        explainer = shap.TreeExplainer(m)
        sv = explainer(X_test)
        return m, X_test, sv

    print("\n=== (1) ORIJINAL FEATURE SETI: Ro birlesik ===")
    model_orig, X_test_orig, shap_orig = _fit_and_get_shap(
        data, drop_cols=["SurfaceRunoff", "Baseflow"])
    p_idx_orig = X_test_orig.columns.get_loc("P")
    print("P icin ortalama SHAP:", shap_orig.values[:, p_idx_orig].mean())
    print("corr(P, SHAP_P):", np.corrcoef(X_test_orig["P"], shap_orig.values[:, p_idx_orig])[0, 1])

    print("\n=== (2) DEKOMPOZE FEATURE SETI: SurfaceRunoff + Baseflow, Ro yok ===")
    model_decomp, X_test_decomp, shap_decomp = _fit_and_get_shap(
        data, drop_cols=["Ro"])
    p_idx_decomp = X_test_decomp.columns.get_loc("P")
    bf_idx_decomp = X_test_decomp.columns.get_loc("Baseflow")
    print("P icin ortalama SHAP:", shap_decomp.values[:, p_idx_decomp].mean())
    print("corr(P, SHAP_P):", np.corrcoef(X_test_decomp["P"], shap_decomp.values[:, p_idx_decomp])[0, 1])
    print("Baseflow icin corr(Baseflow, SHAP_Baseflow):",
          np.corrcoef(X_test_decomp["Baseflow"], shap_decomp.values[:, bf_idx_decomp])[0, 1])

    # Yan yana dependence plot (P): orijinal vs dekompoze
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(X_test_orig["P"], shap_orig.values[:, p_idx_orig], s=12, alpha=0.6)
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_title("P SHAP (Ro birlesik iken)")
    axes[0].set_xlabel("P"); axes[0].set_ylabel("SHAP value")

    axes[1].scatter(X_test_decomp["P"], shap_decomp.values[:, p_idx_decomp], s=12, alpha=0.6, color="darkorange")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_title("P SHAP (Ro -> SurfaceRunoff+Baseflow)")
    axes[1].set_xlabel("P"); axes[1].set_ylabel("SHAP value")
    plt.tight_layout()
    plt.savefig("graphs/P_SHAP_before_after_decomposition.png", dpi=300)
    plt.show()

    return {
        "model_orig": model_orig, "shap_orig": shap_orig, "X_test_orig": X_test_orig,
        "model_decomp": model_decomp, "shap_decomp": shap_decomp, "X_test_decomp": X_test_decomp,
    }


print("\n" + "=" * 70)
print("P - Ro - GWL SUPPRESSION EFFECT HIPOTEZ TESTLERI")
print("=" * 70)
 
output_dir = os.path.join(os.getcwd(), 'graphs')
os.makedirs(output_dir, exist_ok=True)
 
# --- BLOK A) Collinearity raporu -----------------------------------------
print("\n>>> BLOK A: Collinearity raporu (P vs Ro vs GWL) <<<")
corr_report = collinearity_report(results_df)
corr_report.to_csv(os.path.join(output_dir, "P_Ro_collinearity_corr_matrix.csv"))
 
# --- BLOK B) Suppression-effect testi (OLS ile isaret kirilmasi) ---------
print("\n>>> BLOK B: Suppression-effect testi (GWL ~ P  vs  GWL ~ P + Ro) <<<")
m1, m2 = suppression_effect_test(results_df)
with open(os.path.join(output_dir, "P_Ro_suppression_OLS_summary.txt"), "w") as f:
    f.write("MODEL 1: GWL ~ P\n")
    f.write(str(m1.summary()))
    f.write("\n\nMODEL 2: GWL ~ P + Ro\n")
    f.write(str(m2.summary()))
 
# --- BLOK C) SHAP interaction dekompozisyonu (son split icin) ------------
print("\n>>> BLOK C: SHAP interaction dekompozisyonu (P main effect vs P x Ro) <<<")
p_ro_shap_report = shap_interaction_decomposition(
    shap_interaction_values, X_test, feature_a="P", feature_b="Ro"
)
p_ro_shap_report.to_csv(os.path.join(output_dir, "P_Ro_shap_interaction_decomposition.csv"))
 
# --- BLOK D) Ro'yu SurfaceRunoff + Baseflow olarak ayristirma ------------
print("\n>>> BLOK D: Ro -> SurfaceRunoff + Baseflow ayristirmasi <<<")
results_monthly_decomp = build_decomposed_feature_set(results, results_daily_df)
 
# Ana script'teki ile ayni tarih araligini uygula
overall_data_decomp = results_monthly_decomp.loc[start:end]
 
# --- BLOK E) Ro-birlesik vs Ro-ayristirilmis model karsilastirmasi -------
#     (en guclu / en gorsel kanit -- P'nin SHAP dependence egiliminin
#      Ro ayristirildiktan sonra nasil degistigini gosterir)
print("\n>>> BLOK E: Ro birlesik vs Ro ayristirilmis model karsilastirmasi <<<")
comparison_results = compare_feature_sets(
    overall_data_with_Ro=overall_data_decomp,
    model_params=best_params,
)
 
print("\n" + "=" * 70)
print("TUM TESTLER TAMAMLANDI -- ciktilar 'graphs/' klasorune kaydedildi:")
print(" - P_Ro_collinearity_corr_matrix.csv")
print(" - P_Ro_suppression_OLS_summary.txt")
print(" - P_Ro_shap_interaction_decomposition.csv")
print(" - P_SHAP_before_after_decomposition.png")
print("=" * 70)
'''
