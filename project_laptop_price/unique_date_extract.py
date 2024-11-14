# unique_company = ['Apple' 'HP' 'Acer' 'Asus' 'Dell' 'Lenovo' 'Chuwi' 'MSI' 'Microsoft'
#  'Toshiba' 'Huawei' 'Xiaomi' 'Vero' 'Razer' 'Mediacom' 'Samsung' 'Google'
#  'Fujitsu' 'LG']

# unique_typeName = ['Ultrabook' 'Notebook' 'Gaming' '2 in 1 Convertible' 'Workstation']

# unique_ScreenResolution = ['2560x1600', '1440x900', '1920x1080', '2880x1800', '1366x768', '2304x1440',
#  '3200x1800', '2256x1504', '3840x2160', '2160x1440', '2560x1440', '1600x900'
#  '2736x1824', '2400x1600', '1920x1200']

# unique_Ram = [2, 4, 8, 12, 16, 32, 64]

# unique_OpSys = ['macOS' 'No OS' 'Windows 10' 'Linux' 'Chrome OS' 'Windows 7']

# unique_Memory = ['128GB SSD' '128GB Flash Storage' '256GB SSD' '512GB SSD' '500GB HDD'
#  '256GB Flash Storage' '1TB HDD' '32GB Flash Storage'
#  '128GB SSD +  1TB HDD' '256GB SSD +  256GB SSD' '64GB Flash Storage'
#  '256GB SSD +  1TB HDD' '256GB SSD +  2TB HDD' '32GB SSD' '2TB HDD'
#  '64GB SSD' '1.0TB Hybrid' '512GB SSD +  1TB HDD' '1TB SSD'
#  '256GB SSD +  500GB HDD' '128GB SSD +  2TB HDD' '512GB SSD +  512GB SSD'
#  '16GB SSD' '16GB Flash Storage' '512GB SSD +  256GB SSD'
#  '512GB SSD +  2TB HDD' '64GB Flash Storage +  1TB HDD' '180GB SSD'
#  '1TB HDD +  1TB HDD' '32GB HDD' '1TB SSD +  1TB HDD'
#  '512GB Flash Storage' '128GB HDD' '240GB SSD' '8GB SSD' '508GB Hybrid'
#  '1.0TB HDD' '512GB SSD +  1.0TB Hybrid' '256GB SSD +  1.0TB Hybrid']

# unique_Cpu = ['Intel Core i5 2.3GHz' 'Intel Core i5 1.8GHz'
#  'Intel Core i5 7200U 2.5GHz' 'Intel Core i7 2.7GHz'
#  'Intel Core i5 3.1GHz' 'AMD A9-Series 9420 3GHz' 'Intel Core i7 2.2GHz'
#  'Intel Core i7 8550U 1.8GHz' 'Intel Core i5 8250U 1.6GHz'
#  'Intel Core i3 6006U 2GHz' 'Intel Core i7 2.8GHz'
#  'Intel Core M m3 1.2GHz' 'Intel Core i7 7500U 2.7GHz'
#  'Intel Core i7 2.9GHz' 'Intel Core i3 7100U 2.4GHz'
#  'Intel Atom x5-Z8350 1.44GHz' 'Intel Core i5 7300HQ 2.5GHz'
#  'AMD E-Series E2-9000e 1.5GHz' 'Intel Core i5 1.6GHz'
#  'Intel Core i7 8650U 1.9GHz' 'Intel Atom x5-Z8300 1.44GHz'
#  'AMD E-Series E2-6110 1.5GHz' 'AMD A6-Series 9220 2.5GHz'
#  'Intel Celeron Dual Core N3350 1.1GHz' 'Intel Core i3 7130U 2.7GHz'
#  'Intel Core i7 7700HQ 2.8GHz' 'Intel Core i5 2.0GHz'
#  'AMD Ryzen 1700 3GHz' 'Intel Pentium Quad Core N4200 1.1GHz'
#  'Intel Celeron Dual Core N3060 1.6GHz' 'Intel Core i5 1.3GHz'
#  'AMD FX 9830P 3GHz' 'Intel Core i7 7560U 2.4GHz'
#  'AMD E-Series 6110 1.5GHz' 'Intel Core i5 6200U 2.3GHz'
#  'Intel Core M 6Y75 1.2GHz' 'Intel Core i5 7500U 2.7GHz'
#  'Intel Core i3 6006U 2.2GHz' 'AMD A6-Series 9220 2.9GHz'
#  'Intel Core i7 6920HQ 2.9GHz' 'Intel Core i5 7Y54 1.2GHz'
#  'Intel Core i7 7820HK 2.9GHz' 'Intel Xeon E3-1505M V6 3GHz'
#  'Intel Core i7 6500U 2.5GHz' 'AMD E-Series 9000e 1.5GHz'
#  'AMD A10-Series A10-9620P 2.5GHz' 'AMD A6-Series A6-9220 2.5GHz'
#  'Intel Core i5 2.9GHz' 'Intel Core i7 6600U 2.6GHz'
#  'Intel Core i3 6006U 2.0GHz' 'Intel Celeron Dual Core 3205U 1.5GHz'
#  'Intel Core i7 7820HQ 2.9GHz' 'AMD A10-Series 9600P 2.4GHz'
#  'Intel Core i7 7600U 2.8GHz' 'AMD A8-Series 7410 2.2GHz'
#  'Intel Celeron Dual Core 3855U 1.6GHz'
#  'Intel Pentium Quad Core N3710 1.6GHz' 'AMD A12-Series 9720P 2.7GHz'
#  'Intel Core i5 7300U 2.6GHz' 'AMD A12-Series 9720P 3.6GHz'
#  'Intel Celeron Quad Core N3450 1.1GHz'
#  'Intel Celeron Dual Core N3060 1.60GHz' 'Intel Core i5 6440HQ 2.6GHz'
#  'Intel Core i7 6820HQ 2.7GHz' 'AMD Ryzen 1600 3.2GHz'
#  'Intel Core i7 7Y75 1.3GHz' 'Intel Core i5 7440HQ 2.8GHz'
#  'Intel Core i7 7660U 2.5GHz' 'Intel Core i7 7700HQ 2.7GHz'
#  'Intel Core M m3-7Y30 2.2GHz' 'Intel Core i5 7Y57 1.2GHz'
#  'Intel Core i7 6700HQ 2.6GHz' 'Intel Core i3 6100U 2.3GHz'
#  'AMD A10-Series 9620P 2.5GHz' 'AMD E-Series 7110 1.8GHz'
#  'Intel Celeron Dual Core N3350 2.0GHz' 'AMD A9-Series A9-9420 3GHz'
#  'Intel Core i7 6820HK 2.7GHz' 'Intel Core M 7Y30 1.0GHz'
#  'Intel Xeon E3-1535M v6 3.1GHz' 'Intel Celeron Quad Core N3160 1.6GHz'
#  'Intel Core i5 6300U 2.4GHz' 'Intel Core i3 6100U 2.1GHz'
#  'AMD E-Series E2-9000 2.2GHz' 'Intel Celeron Dual Core N3050 1.6GHz'
#  'Intel Core M M3-6Y30 0.9GHz' 'AMD A9-Series 9420 2.9GHz'
#  'Intel Core i5 6300HQ 2.3GHz' 'AMD A6-Series 7310 2GHz'
#  'Intel Atom Z8350 1.92GHz' 'Intel Xeon E3-1535M v5 2.9GHz'
#  'Intel Core i5 6260U 1.8GHz' 'Intel Pentium Dual Core N4200 1.1GHz'
#  'Intel Celeron Quad Core N3710 1.6GHz' 'Intel Core M 1.2GHz'
#  'AMD A12-Series 9700P 2.5GHz' 'Intel Core i7 7500U 2.5GHz'
#  'Intel Pentium Dual Core 4405U 2.1GHz' 'AMD A4-Series 7210 2.2GHz'
#  'Intel Core i7 6560U 2.2GHz' 'Intel Core M m7-6Y75 1.2GHz'
#  'AMD FX 8800P 2.1GHz' 'Intel Core M M7-6Y75 1.2GHz'
#  'Intel Core i5 7200U 2.50GHz' 'Intel Core i5 7200U 2.70GHz'
#  'Intel Atom X5-Z8350 1.44GHz' 'Intel Core i5 7200U 2.7GHz'
#  'Intel Core M 1.1GHz' 'Intel Atom x5-Z8550 1.44GHz'
#  'Intel Pentium Dual Core 4405Y 1.5GHz'
#  'Intel Pentium Quad Core N3700 1.6GHz' 'Intel Core M 6Y54 1.1GHz'
#  'Intel Core i7 6500U 2.50GHz' 'Intel Celeron Dual Core N3350 2GHz'
#  'Samsung Cortex A72&A53 2.0GHz' 'AMD E-Series 9000 2.2GHz'
#  'Intel Core M 6Y30 0.9GHz' 'AMD A9-Series 9410 2.9GHz']


# unique_Gpu = ['Intel Iris Plus Graphics 640' 'Intel HD Graphics 6000'
#  'Intel HD Graphics 620' 'AMD Radeon Pro 455'
#  'Intel Iris Plus Graphics 650' 'AMD Radeon R5' 'Intel Iris Pro Graphics'
#  'Nvidia GeForce MX150' 'Intel UHD Graphics 620' 'Intel HD Graphics 520'
#  'AMD Radeon Pro 555' 'AMD Radeon R5 M430' 'Intel HD Graphics 615'
#  'AMD Radeon Pro 560' 'Nvidia GeForce 940MX' 'Intel HD Graphics 400'
#  'Nvidia GeForce GTX 1050' 'AMD Radeon R2' 'AMD Radeon 530'
#  'Nvidia GeForce 930MX' 'Intel HD Graphics' 'Intel HD Graphics 500'
#  'Nvidia GeForce 930MX ' 'Nvidia GeForce GTX 1060' 'Nvidia GeForce 150MX'
#  'Intel Iris Graphics 540' 'AMD Radeon RX 580' 'Nvidia GeForce 920MX'
#  'AMD Radeon R4 Graphics' 'AMD Radeon 520' 'Nvidia GeForce GTX 1070'
#  'Nvidia GeForce GTX 1050 Ti' 'Nvidia GeForce MX130' 'AMD R4 Graphics'
#  'Nvidia GeForce GTX 940MX' 'AMD Radeon RX 560' 'Nvidia GeForce 920M'
#  'AMD Radeon R7 M445' 'AMD Radeon RX 550' 'Nvidia GeForce GTX 1050M'
#  'Intel HD Graphics 515' 'AMD Radeon R5 M420' 'Intel HD Graphics 505'
#  'Nvidia GTX 980 SLI' 'AMD R17M-M1-70' 'Nvidia GeForce GTX 1080'
#  'Nvidia Quadro M1200' 'Nvidia GeForce 920MX ' 'Nvidia GeForce GTX 950M'
#  'AMD FirePro W4190M ' 'Nvidia GeForce GTX 980M' 'Intel Iris Graphics 550'
#  'Nvidia GeForce 930M' 'Intel HD Graphics 630' 'AMD Radeon R5 430'
#  'Nvidia GeForce GTX 940M' 'Intel HD Graphics 510' 'Intel HD Graphics 405'
#  'AMD Radeon RX 540' 'Nvidia GeForce GT 940MX' 'AMD FirePro W5130M'
#  'Nvidia Quadro M2200M' 'AMD Radeon R4' 'Nvidia Quadro M620'
#  'AMD Radeon R7 M460' 'Intel HD Graphics 530' 'Nvidia GeForce GTX 965M'
#  'Nvidia GeForce GTX1080' 'Nvidia GeForce GTX1050 Ti'
#  'Nvidia GeForce GTX 960M' 'AMD Radeon R2 Graphics' 'Nvidia Quadro M620M'
#  'Nvidia GeForce GTX 970M' 'Nvidia GeForce GTX 960<U+039C>'
#  'Intel Graphics 620' 'Nvidia GeForce GTX 960' 'AMD Radeon R5 520'
#  'AMD Radeon R7 M440' 'AMD Radeon R7' 'Nvidia Quadro M520M'
#  'Nvidia Quadro M2200' 'Nvidia Quadro M2000M' 'Intel HD Graphics 540'
#  'Nvidia Quadro M1000M' 'AMD Radeon 540' 'Nvidia GeForce GTX 1070M'
#  'Nvidia GeForce GTX1060' 'Intel HD Graphics 5300' 'AMD Radeon R5 M420X'
#  'AMD Radeon R7 Graphics' 'Nvidia GeForce 920' 'Nvidia GeForce 940M'
#  'Nvidia GeForce GTX 930MX' 'AMD Radeon R7 M465' 'AMD Radeon R3'
#  'Nvidia GeForce GTX 1050Ti' 'AMD Radeon R7 M365X' 'AMD Radeon R9 M385'
#  'Intel HD Graphics 620 ' 'Nvidia Quadro 3000M' 'Nvidia GeForce GTX 980 '
#  'AMD Radeon R5 M330' 'AMD FirePro W4190M' 'AMD FirePro W6150M'
#  'AMD Radeon R5 M315' 'Nvidia Quadro M500M' 'AMD Radeon R7 M360'
#  'Nvidia Quadro M3000M' 'Nvidia GeForce 960M' 'ARM Mali T860 MP4']

unique_company = ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI', 'Microsoft',
                  'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer', 'Mediacom', 'Samsung', 'Google',
                  'Fujitsu', 'LG']

unique_typeName = ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation']

unique_ScreenResolution = ['2560x1600', '1440x900', '1920x1080', '2880x1800', '1366x768', 
                          '2304x1440', '3200x1800', '2256x1504', '3840x2160', '2160x1440', 
                          '2560x1440', '1600x900', '2736x1824', '2400x1600', '1920x1200']

unique_Ram = [2, 4, 8, 12, 16, 32, 64]

unique_OpSys = ['macOS', 'No OS', 'Windows 10', 'Linux', 'Chrome OS', 'Windows 7']

unique_Memory = ['128GB SSD', '128GB Flash Storage', '256GB SSD', '512GB SSD', '500GB HDD',
                 '256GB Flash Storage', '1TB HDD', '32GB Flash Storage', '128GB SSD + 1TB HDD', 
                 '256GB SSD + 256GB SSD', '64GB Flash Storage', '256GB SSD + 1TB HDD', 
                 '256GB SSD + 2TB HDD', '32GB SSD', '2TB HDD', '64GB SSD', '1.0TB Hybrid', 
                 '512GB SSD + 1TB HDD', '1TB SSD', '256GB SSD + 500GB HDD', 
                 '128GB SSD + 2TB HDD', '512GB SSD + 512GB SSD', '16GB SSD', 
                 '16GB Flash Storage', '512GB SSD + 256GB SSD', 
                 '512GB SSD + 2TB HDD', '64GB Flash Storage + 1TB HDD', '180GB SSD', 
                 '1TB HDD + 1TB HDD', '32GB HDD', '1TB SSD + 1TB HDD', 
                 '512GB Flash Storage', '128GB HDD', '240GB SSD', '8GB SSD', 
                 '508GB Hybrid', '1.0TB HDD', '512GB SSD + 1.0TB Hybrid', 
                 '256GB SSD + 1.0TB Hybrid']

unique_Cpu = ['Intel Core i5 2.3GHz', 'Intel Core i5 1.8GHz', 'Intel Core i5 7200U 2.5GHz', 
               'Intel Core i7 2.7GHz', 'Intel Core i5 3.1GHz', 'AMD A9-Series 9420 3GHz', 
               'Intel Core i7 2.2GHz', 'Intel Core i7 8550U 1.8GHz', 'Intel Core i5 8250U 1.6GHz', 
               'Intel Core i3 6006U 2GHz', 'Intel Core i7 2.8GHz', 'Intel Core M m3 1.2GHz', 
               'Intel Core i7 7500U 2.7GHz', 'Intel Core i7 2.9GHz', 'Intel Core i3 7100U 2.4GHz', 
               'Intel Atom x5-Z8350 1.44GHz', 'Intel Core i5 7300HQ 2.5GHz', 'AMD E-Series E2-9000e 1.5GHz', 
               'Intel Core i5 1.6GHz', 'Intel Core i7 8650U 1.9GHz', 'Intel Atom x5-Z8300 1.44GHz', 
               'AMD E-Series E2-6110 1.5GHz', 'AMD A6-Series 9220 2.5GHz', 
               'Intel Celeron Dual Core N3350 1.1GHz', 'Intel Core i3 7130U 2.7GHz', 
               'Intel Core i7 7700HQ 2.8GHz', 'Intel Core i5 2.0GHz', 'AMD Ryzen 1700 3GHz', 
               'Intel Pentium Quad Core N4200 1.1GHz', 'Intel Celeron Dual Core N3060 1.6GHz', 
               'Intel Core i5 1.3GHz', 'AMD FX 9830P 3GHz', 'Intel Core i7 7560U 2.4GHz', 
               'AMD E-Series 6110 1.5GHz', 'Intel Core i5 6200U 2.3GHz', 'Intel Core M 6Y75 1.2GHz', 
               'Intel Core i5 7500U 2.7GHz', 'Intel Core i3 6006U 2.2GHz', 
               'AMD A6-Series 9220 2.9GHz', 'Intel Core i7 6920HQ 2.9GHz', 'Intel Core i5 7Y54 1.2GHz', 
               'Intel Core i7 7820HK 2.9GHz', 'Intel Xeon E3-1505M V6 3GHz', 
               'Intel Core i7 6500U 2.5GHz', 'AMD E-Series 9000e 1.5GHz', 
               'AMD A10-Series A10-9620P 2.5GHz', 'AMD A6-Series A6-9220 2.5GHz', 
               'Intel Core i5 2.9GHz', 'Intel Core i7 6600U 2.6GHz', 'Intel Core i3 6006U 2.0GHz', 
               'Intel Celeron Dual Core 3205U 1.5GHz', 'Intel Core i7 7820HQ 2.9GHz', 
               'AMD A10-Series 9600P 2.4GHz', 'Intel Core i7 7600U 2.8GHz', 
               'AMD A8-Series 7410 2.2GHz', 'Intel Celeron Dual Core 3855U 1.6GHz', 
               'Intel Pentium Quad Core N3710 1.6GHz', 'AMD A12-Series 9720P 2.7GHz', 
               'Intel Core i5 7300U 2.6GHz', 'AMD A12-Series 9720P 3.6GHz', 
               'Intel Celeron Quad Core N3450 1.1GHz', 'Intel Celeron Dual Core N3060 1.60GHz', 
               'Intel Core i5 6440HQ 2.6GHz', 'Intel Core i7 6820HQ 2.7GHz', 
               'AMD Ryzen 1600 3.2GHz', 'Intel Core i7 7Y75 1.3GHz', 'Intel Core i5 7440HQ 2.8GHz', 
               'Intel Core i7 7660U 2.5GHz', 'Intel Core i7 7700HQ 2.7GHz', 
               'Intel Core M m3-7Y30 2.2GHz', 'Intel Core i5 7Y57 1.2GHz', 
               'Intel Core i7 6700HQ 2.6GHz', 'Intel Core i3 6100U 2.3GHz', 
               'AMD A10-Series 9620P 2.5GHz', 'AMD E-Series 7110 1.8GHz', 
               'Intel Celeron Dual Core N3350 2.0GHz', 'AMD A9-Series A9-9420 3GHz', 
               'Intel Core i7 6820HK 2.7GHz', 'Intel Core M 7Y30 1.0GHz', 
               'Intel Xeon E3-1535M v6 3.1GHz', 'Intel Celeron Quad Core N3160 1.6GHz', 
               'Intel Core i5 6300U 2.4GHz', 'Intel Core i3 6100U 2.1GHz', 
               'AMD E-Series E2-9000 2.2GHz', 'Intel Celeron Dual Core N3050 1.6GHz', 
               'Intel Core M M3-6Y30 0.9GHz', 'AMD A9-Series 9420 2.9GHz', 
               'Intel Core i5 6300HQ 2.3GHz', 'AMD A6-Series 7310 2GHz', 
               'Intel Atom Z8350 1.92GHz', 'Intel Xeon E3-1535M v5 2.9GHz', 
               'Intel Core i5 6260U 1.8GHz', 'Intel Pentium Dual Core N4200 1.1GHz', 
               'AMD A10-Series 9620P 2.8GHz', 'AMD Ryzen 5 3550H 2.1GHz', 
               'Intel Core i7 7500U 2.7GHz', 'Intel Celeron Dual Core N3350 1.0GHz', 
               'Intel Core i7 8850H 2.6GHz', 'Intel Core M m5-6Y54 1.1GHz', 
               'Intel Core i3 8100H 3.3GHz', 'AMD A9-Series 9410 2.9GHz', 
               'Intel Core i5 6300U 2.0GHz', 'Intel Core M 7Y54 1.3GHz', 
               'AMD A8-Series A8-9600P 2.4GHz', 'Intel Core i5 8250U 1.6GHz', 
               'Intel Pentium Gold 4415U 2.3GHz', 'Intel Core i5 7200U 2.5GHz', 
               'AMD Ryzen 7 2700U 2.2GHz', 'Intel Core i5 7300HQ 2.5GHz', 
               'AMD A10-Series 9700P 2.5GHz', 'Intel Core i7 7820HK 2.9GHz', 
               'Intel Core i5 8200Y 1.3GHz', 'Intel Core i7 4790K 4.0GHz', 
               'Intel Core i3 8100U 2.2GHz']

unique_Gpu = ['Intel HD Graphics 620', 'NVIDIA GeForce GTX 1050', 'AMD Radeon R7 M445', 
               'Intel UHD Graphics 620', 'NVIDIA GeForce GTX 1070', 'AMD Radeon 530', 
               'Intel Iris Plus Graphics 640', 'NVIDIA GeForce GTX 1060', 'Intel HD Graphics 600', 
               'NVIDIA GeForce GTX 1650', 'AMD Radeon R5 M330', 'Intel HD Graphics 615', 
               'AMD Radeon RX 550', 'NVIDIA GeForce MX150', 'NVIDIA GeForce RTX 2060', 
               'NVIDIA GeForce GTX 980', 'AMD Radeon R9 M370X', 'NVIDIA GeForce RTX 2070', 
               'NVIDIA GeForce GTX 960', 'AMD Radeon RX 560', 'NVIDIA GeForce GTX 1050 Ti', 
               'NVIDIA GeForce RTX 2080', 'NVIDIA GeForce GTX 850M', 'AMD Radeon R9 M290X', 
               'Intel Iris Plus Graphics 655', 'Intel UHD Graphics 620', 'NVIDIA GeForce GTX 980M', 
               'Intel HD Graphics 630', 'AMD Radeon R7 370', 'Intel Iris Plus Graphics 655', 
               'Intel HD Graphics 630', 'AMD Radeon R9 M295X', 'AMD Radeon R7 340', 
               'NVIDIA GeForce GTX 970', 'NVIDIA GeForce RTX 3080', 'Intel Iris Xe Graphics', 
               'NVIDIA GeForce MX250', 'NVIDIA GeForce GTX 780', 'NVIDIA GeForce GTX 750', 
               'AMD Radeon R7 240', 'AMD Radeon R9 380', 'NVIDIA GeForce GTX 1650 Ti', 
               'NVIDIA GeForce MX330', 'NVIDIA GeForce GTX 770', 'Intel Iris Graphics 540', 
               'NVIDIA GeForce GTX 750 Ti', 'NVIDIA GeForce GTX 880M', 'NVIDIA GeForce MX450', 
               'NVIDIA GeForce RTX 3060', 'AMD Radeon R5 240', 'NVIDIA GeForce GTX 1070 Ti', 
               'NVIDIA GeForce RTX 3050', 'AMD Radeon RX 580', 'NVIDIA GeForce RTX 3070', 
               'NVIDIA GeForce GTX 780M', 'AMD Radeon 530', 'NVIDIA GeForce RTX 4060', 
               'NVIDIA GeForce GTX 750 Ti', 'NVIDIA GeForce GTX 670', 'Intel HD Graphics 520', 
               'NVIDIA GeForce GTX 880', 'NVIDIA GeForce RTX 4050', 'AMD Radeon RX 570', 
               'NVIDIA GeForce GTX 680', 'AMD Radeon R9 290', 'AMD Radeon RX 590', 
               'NVIDIA GeForce GTX 650', 'NVIDIA GeForce RTX 4030', 'AMD Radeon R7 360', 
               'NVIDIA GeForce GTX 1650 Super', 'AMD Radeon RX 5700', 'NVIDIA GeForce GTX 660', 
               'Intel Iris Graphics 5000', 'NVIDIA GeForce GTX 770M', 'AMD Radeon R9 280X', 
               'NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 3080 Ti', 'AMD Radeon RX 5600', 
               'NVIDIA GeForce GTX 970M', 'AMD Radeon RX 5900', 'NVIDIA GeForce RTX 2060 Super', 
               'AMD Radeon RX 6800', 'NVIDIA GeForce GTX 680M', 'AMD Radeon RX 5500', 
               'NVIDIA GeForce RTX 3080 Super', 'AMD Radeon RX 5700 XT', 'NVIDIA GeForce RTX 3090 Ti', 
               'AMD Radeon 7000', 'NVIDIA GeForce GTX 880M', 'NVIDIA GeForce RTX 3050 Ti']
