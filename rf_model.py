# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:42:49 2026

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
"""
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "Times New Roman",

    "font.size": 19,

    "axes.titlesize": 24,
    "axes.titleweight": "bold",

    "axes.labelsize": 20,
    "axes.labelweight": "bold",

    "xtick.labelsize": 19,
    "ytick.labelsize": 19,

    "legend.fontsize": 22
})
"""

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
        """
        df = pd.read_csv(
            file_path,
            sep=r"\s+",        # Bir veya daha fazla boşluk
            skiprows=15,
            header=None,
            names=column_names,
            engine="python"    # Önemli! Regex için önerilir
        )
        """
            
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap

# Sabit parametreler
intercept = 0.2  # Sabit intercept değeri
slope_values_ = np.linspace(0, 0.00005, 5)  # Slope değerlerini 0.1'den 1.0'a kadar 10 adımda artırıyoruz.
# 50 yıl için en cok egim 0.0001

# Sonuçları tutacak sözlük
slope_results_ea = []

general_train_r2, general_test_r2 = [], []
general_train_rmse, general_test_rmse = [], []
general_train_nrmse, general_test_nrmse = [], []
general_train_kge, general_test_kge = [], []
metrics = []    

def kge(y_true, y_pred):
    
    r = np.corrcoef(y_true, y_pred)[0, 1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gamma = (np.std(y_pred) / np.mean(y_pred)) / (np.std(y_true) / np.mean(y_true))
    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

#slope = 0.0000

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
    
    # YASS Zaman Serisi
    plt.figure(figsize=(10, 6))
    plt.plot(aquimod_df.index, q2k2s1_df["GWL(m)"], #marker='o', 
             linestyle='-', color='navy', label='YASS')
    plt.title(f"YASS Zaman Serisi Grafiği (eğim: {slope:.1e})", fontsize=14)
    plt.xlabel("Tarih", fontsize=12)
    plt.ylabel("Yeraltı Suyu Seviyesi (m)", fontsize=12)
    plt.grid(True, linestyle='--', #alpha=0.6
             )
    plt.legend()
    plt.tight_layout()
    
    output_dir = os.path.join(os.getcwd(), 'graphs')
    # Grafik kaydetme yolu
    output_path = os.path.join(output_dir, f"YASS_{slope:.1e}.png")
    # Grafiği kaydet
    #plt.savefig(output_path, dpi=900, bbox_inches="tight")  # Yüksek çözünürlük için dpi ayarı yapılır
    plt.show()
    
    results = pd.concat([fao_df, weibull_df, q2k2s1_df], axis=1)
    results = results.loc[:, ~results.columns.duplicated()]
    
    #Data for XAI
    results.index = aquimod_df.index
    results['RAIN'] = aquimod_df['RAIN']
    results['PE'] = aquimod_df['PE']
    #results['Ro'] = results['q_ro(mm/d)'] + (results['Q_2(m3/d)'] / 3) # to mm/day
    results['Ro'] = (results['q_ro(mm/d)'] * 3) + results['Q_2(m3/d)'] # to m3/day
    
    results_daily_df = pd.DataFrame()
    results_daily_df['P'] = aquimod_df['RAIN']
    results_daily_df['T'] = df['tmean']
    results_daily_df['GWL'] = results['GWL(m)']
    results_daily_df['Ro'] = results['Ro']
    results_daily_df['ETp'] = results['PE']
    #results_daily_df['ETa'] = results['E_a(mm/d)']
    """
    # Başlangıç ve bitiş tarihlerini belirleyin
    start_date = '2020-01-01'
    end_date = '2025-12-31'
    
    # 'Date' sütununu datetime formatına çevirin (eğer değilse)
    # Date sütunu zaten index ise doğrudan filtreleme yapabiliriz
    df_filtered = results_daily_df.loc[start_date:end_date]
    
    # Grafiği çizme
    plt.figure(figsize=(10, 5))
    plt.plot(df_filtered.index, df_filtered['ETa'], color='#1f3c88')
    plt.title(f'Ea (eğim: {slope:.1e})')
    plt.xlabel('Tarih')
    plt.ylabel('Çekim Miktarı (m³/gün)')
    plt.grid(True)
    plt.show()
    
    
    #
    model_df = pd.DataFrame()
    model_df["rech"] = results["q_rech(mm/d)"]
    model_df["baseflow"] = results["Q_2(m3/d)"]
    """
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
    results_df['GWLlag-1'] = results_df['GWL'].shift(1)
    """
    # Yağış-Akış-Baz Akım Grafiği
    plt.figure(figsize=(10, 6))
    # İlk Y ekseni için yüzey akışı ve baz akımı grafiği
    plt.plot(results_df.index, results_df["Ro"], linestyle='-', color='#17becf', label='Akış')
    plt.plot(results_df.index, results['Q_2(m3/d)'], linestyle='-', color='#005f99', label='Baz Akımı')
    plt.title(f"Yağış-Akış Grafiği (Eğim: {slope:.1e})", fontsize=14)
    plt.xlabel("Tarih", fontsize=12)
    plt.ylabel("Debi (m³/gün)", fontsize=12)
    #plt.grid(True, linestyle='--')
    plt.legend(loc='upper left', framealpha=0.8)
    # İkinci Y ekseni (tersine çevrili) için yağış verisi
    ax2 = plt.gca().twinx()  # İkinci bir Y ekseni oluşturur
    ax2.invert_yaxis()  # Ekseni tersine çevirir
    ax2.bar(results_df.index, results_df["P"], color='k', alpha=0.9, label='Yağış', width=1.0)
    ax2.set_ylabel("Yağış (mm)", fontsize=12)
    ax2.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    output_dir = os.path.join(os.getcwd(), 'graphs')
    # Grafik kaydetme yolu
    output_path = os.path.join(output_dir, f"Yagis-Akis_Grafiği_{slope:.1e}.png")
    # Grafiği kaydet
    #plt.savefig(output_path, dpi=900, bbox_inches="tight")  # Yüksek çözünürlük için dpi ayarı yapılır
    plt.show()
    """
    #XAI model
    results_XGB_df = pd.DataFrame()
    
    # Define hyperparameters
    '''
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }

    params = {
        "n_estimators": 300,
        "max_depth": 10,
        #"min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }
    '''
    
    """
    Best TimeSeries CV R2: 0.8948125722119233
    Best params: {'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 1}
    """
       
    # Tarih aralıkları
    start = "2011-01-01"
    end = "2100-12-31"
    overall_data = results_df.loc[start:end]

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
        
        train_data = train_datas[split]
        val_data = validation_datas[split]
        test_data = test_datas[split]
        
        # --- TARİH ARALIKLARINI YAZDIR ---
        print(f"\n===== SPLIT {split+1} =====")
        print(f"Train  : {train_data.index.min()}  -->  {train_data.index.max()}  | Boyut: {len(train_data)}")
        print(f"Val    : {val_data.index.min()}    -->  {val_data.index.max()}    | Boyut: {len(val_data)}")
        print(f"Test   : {test_data.index.min()}   -->  {test_data.index.max()}   | Boyut: {len(test_data)}")

    
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
        
            model = RandomForestRegressor(
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
        
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        """
        # Model oluşturma ve eğitim
        model = RandomForestRegressor(**params)
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
    
        ######################
    
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
        
        abs_subset = aquimod_df.loc[start:end]
        monthly_results["ABSvalue"] = abs_subset["ABS"].resample("M").mean()
    
        slope_results_ea.append(monthly_results)

        #############################################
        
        ############################
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
        """
        shap_vals = shap_values.values
        feature_names = X_test.columns.tolist()
        
        # Mean absolute SHAP hesapla
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        # Feature importance sıralaması (summary plot ile aynı mantık)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_shap = shap_vals[:, sorted_idx]
        sorted_mean_abs = mean_abs_shap[sorted_idx]
        
        # Kümülatif yüzde hesapla
        cumulative = np.cumsum(sorted_mean_abs)
        cumulative_percent = 100 * cumulative / cumulative[-1]
        
        # ---- FIGURE ----
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]}
        )
        
        # =========================
        # (a) SHAP SUMMARY PLOT
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
        # (b) MEAN |SHAP| BAR PLOT
        # =========================
        y_pos = np.arange(len(sorted_features))
        
        bars = ax2.barh(y_pos, sorted_mean_abs)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_features)
        ax2.invert_yaxis()  # en önemli en üstte
        ax2.set_xlabel("Mean |SHAP value|")
        ax2.set_title("(b)", loc="left")
        
        # Yüzde etiketlerini yaz
        for i, (val, perc) in enumerate(zip(sorted_mean_abs, cumulative_percent)):
            ax2.text(
                val + max(sorted_mean_abs) * 0.01,
                i,
                f"{perc:.0f}%",
                va="center",
                fontsize=9
            )
        
        plt.tight_layout()
        
        # Kaydet
        output_path = os.path.join(output_dir, f"Shap_Combined_{slope:.1e}_{split}.png")
        plt.savefig(output_path, dpi=900, bbox_inches="tight")
        
        plt.show()
        plt.close()
        """
        ####################

        '''
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
            # Grafik kaydetme yolu
            output_path = os.path.join(output_dir, f"Shap_{feature}_{slope:.1e}_{split}.png")
            # Grafiği kaydet
            plt.savefig(output_path, dpi=900, bbox_inches="tight")  # Yüksek çözünürlük için dpi ayarı yapılı
            plt.show()
            plt.close()
        '''


# metrics list → DataFrame
df_metrics = pd.DataFrame(metrics)

# CSV olarak kaydet
df_metrics.to_csv("metrics_results_RF.csv", index=False)


##########

df_hoResults = pd.DataFrame(hoResults)


df_hoResults.to_csv("ho_results_RF.csv", index=False)
    
#slope_results_ea[-1].to_csv("output_rf.csv")

acc = pd.DataFrame()
acc["R2"] = general_test_r2 + general_train_r2
acc["KGE"] = general_test_kge + general_train_kge
acc["NRMSE"] = general_test_nrmse + general_train_nrmse
acc["RMSE"] = general_test_rmse + general_train_rmse

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
all_shap_df.to_csv("RF_scenario_split_shap_table.csv", index=False)

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
    output_path = os.path.join(output_dir, f"Shap_Combined_Dependence_{feature}_RF.png")
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
    os.path.join(output_dir, "Combined_SHAP_Legend_RF.png"),
    dpi=900,
    bbox_inches='tight'
)

plt.show()
plt.close()

