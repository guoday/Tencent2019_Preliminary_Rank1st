### 1. 题目介绍

请直接查看guide.pdf了解赛题，该项目是初赛第一名的模型。

### 2.模型介绍

![avatar](https://ucb07b1b4b5c5ccad50aaaf2db0e.previews.dropboxusercontent.com/p/thumb/AA7smXx7RQ5XKGvt3Z2fknZmBQTySL4NHk8srgV2ydKPMr5S8gyEwvWm2o0nObMXFtJH-kBQW43ZSES0GG_pavV1lOfe7Z1r0ld49deXqZWxrszpu8R8D4fiiDV2cL6zQFOHXws85wbFBB2vcvyvUSz_b2wNQV6uOllezc67-H-F8Ss4fAYxNW9FQVxm8NryxOnXhU5JM-p3-ei132QX-kWEhutgeoZq3lf-Ku_6F_1UmOMQByxakGSU0dab6dWYOt6GyE69o0zEwrZQBw4vSc08WD0a1GFD_jKkvlFuuxulxoqH2SykoSy0GgvLesceCTHRj2PdG5wqlk3cybl4JQ85yavNUajewFLvsvojHZcnU7XY0GuEiIpi5AbHLll7uMNyX9axKbo2S3k2pM6fzFvj/p.png?fv_content=true&size_mode=5)

![avatar](https://uc5c9c4f8e4563750af2bc1864bf.previews.dropboxusercontent.com/p/thumb/AA5bs_amewpUHvJY4mqxcRMMTRM2C1YgA6Vm4mtTQ7btMmgN3sQjsGkMXAHRzTbzfSDAE0luiju6Lta4MHFpLIw3RKbTG3d7S461GT0i3kBfTap1nUsONUQagUndLypwda3uXu__2pkIELHN2lJ8jPPkK1ahMk4q_6ziq7tpFRH7EJYJpF1DLZE79THYN2341KoXsO2-7_Ds7Fdvxm65JyiDtHGGzSZKJxU5SHBNnmwNoOpnMgwmO-IxVGipq9IaqWzbpacnz3L4j2Jo5arcy-4Lv940_bmzcSSZy-BPYOKenSxjgPVSajkFV0SmUEqv0wueATNOsWw8VBPupt5OmlZxfgfrjlf7lEenJ0NoCC7M3ymmpTa-92eUQvMy4CoVQAM1Xqx57pq_icvanYLbVAUM/p.png?size=2048x1536&size_mode=3)

![avatar](https://ucf37476630a07bc398073492533.previews.dropboxusercontent.com/p/thumb/AA6BTyAulsN80hRzxlA3YsvAyBynB5R1yceja9Ue_dlgdACWQnPp7LwebnCBjxHyec4ATjr4bSOImrF7kNZiIuitGT9iJArknQUb8PuDIbiObcBMPB9cNlq-3oL13CnODiOnOdgsonmsZe3SnRYFCJcsvLzOQ76UxiDqlDBY5AAnP1NZFrXZ6LrWhCgb5oTBrd93HY_Je53UgBwZheaNm31h16s1C37iu4vdorEqeaQ1_PRLYmrgwexuWWCepHdDw3SNieCUwBh1jbKIzSIFWN4Ro4OvMv2zajlM6kMUI66W-31u11v2E6QmxKlF8C4R5FfucNqgKi5LSAyuZNsPBjuOuc4tNAEJ-WgjnEqSLWQiyQHX-NxQZn0XCsjCXiEHhwb8J1s8jmIrR3wQeqscRUYU/p.png?fv_content=true&size_mode=5)

### 3. 配置环境

- scikit-learn
- tqdm
- pandas
- numpy
- scipy
- tensorFlow=1.12.0 (其他版本≥1.4且不等于1.5或1.6)
- Linux Ubuntu 16.04, 128G内存(64G应该足够)，一张显卡 

### 4.数据下载
```shell
mkdir data 
cd data
#Download data from https://pan.baidu.com/s/1ASQMms_u70psRgW_KEyT2Q 
#Password: burw
unzip algo.qq.com_641013010_testa.zip imps_log.zip user.zip
cd ..
```

### 5.数据预处理

```shell
python src/preprocess.py
```

### 6.提取特征

```shell
python src/extract_feature.py
```

### 7.转换数据格式

```shell
python src/convert_format.py
```

1）缺失值NA用0填充

2）将Word2Vec和DeepWalk得到的embedding拼接起来，并且掩盖到5%的广告

3）将需要用key-values的稠密特征正则化到[0,1]之间

### 8.训练模型

```shell
mkdir submission
python train.py
```

