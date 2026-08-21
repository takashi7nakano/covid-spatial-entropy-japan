"""prefecture_coords.py — 47都道府県の正準順序と県庁所在地の座標 (lat, lon)。

make_figS1.py と robustness_variants.py が共有する唯一の定義元。
（もとは robustness_variants.py に直書きされていたものを 2026-08-10 に切り出し）
"""

PREFS = ['Hokkaido','Aomori','Iwate','Miyagi','Akita','Yamagata','Fukushima',
         'Ibaraki','Tochigi','Gunma','Saitama','Chiba','Tokyo','Kanagawa',
         'Niigata','Toyama','Ishikawa','Fukui','Yamanashi','Nagano',
         'Gifu','Shizuoka','Aichi','Mie','Shiga','Kyoto','Osaka','Hyogo',
         'Nara','Wakayama','Tottori','Shimane','Okayama','Hiroshima',
         'Yamaguchi','Tokushima','Kagawa','Ehime','Kochi','Fukuoka',
         'Saga','Nagasaki','Kumamoto','Oita','Miyazaki','Kagoshima','Okinawa']
idx = {name: i for i, name in enumerate(PREFS)}
N = 47

COORDS = {
    'Hokkaido':(43.06,141.35),'Aomori':(40.82,140.74),'Iwate':(39.70,141.15),
    'Miyagi':(38.27,140.87),'Akita':(39.72,140.10),'Yamagata':(38.24,140.36),
    'Fukushima':(37.75,140.47),'Ibaraki':(36.34,140.45),'Tochigi':(36.57,139.88),
    'Gunma':(36.39,139.06),'Saitama':(35.86,139.65),'Chiba':(35.61,140.12),
    'Tokyo':(35.69,139.69),'Kanagawa':(35.45,139.64),'Niigata':(37.90,139.02),
    'Toyama':(36.70,137.21),'Ishikawa':(36.59,136.63),'Fukui':(36.07,136.22),
    'Yamanashi':(35.66,138.57),'Nagano':(36.65,138.18),'Gifu':(35.39,136.72),
    'Shizuoka':(34.98,138.38),'Aichi':(35.18,136.91),'Mie':(34.73,136.51),
    'Shiga':(35.00,135.87),'Kyoto':(35.02,135.76),'Osaka':(34.69,135.52),
    'Hyogo':(34.69,135.18),'Nara':(34.69,135.83),'Wakayama':(34.23,135.17),
    'Tottori':(35.50,134.24),'Shimane':(35.47,133.05),'Okayama':(34.66,133.93),
    'Hiroshima':(34.40,132.46),'Yamaguchi':(34.19,131.47),'Tokushima':(34.07,134.56),
    'Kagawa':(34.34,134.04),'Ehime':(33.84,132.77),'Kochi':(33.56,133.53),
    'Fukuoka':(33.61,130.42),'Saga':(33.25,130.30),'Nagasaki':(32.74,129.87),
    'Kumamoto':(32.79,130.74),'Oita':(33.24,131.61),'Miyazaki':(31.91,131.42),
    'Kagoshima':(31.56,130.56),'Okinawa':(26.21,127.68),
}
