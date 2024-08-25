# 🍵 webStreamlit
一个简单的智慧茶园管理系统示例，基于Streamlit库构建web界面，实现以下功能：

1. 获取气象数据，对茶叶生长环境所需温度、湿度、风速、降水等因素进行预警
2. 模拟茶园数据进行展示，即数据的可视化分析
3. 叠加视频监控点，可以进行人、背景、鸟等21类物体检测
4. 将示例部署至Streamlit Cloud，点击链接一键访问👉[示例](https://fre-air-webstreamlit-webstreamlit-e8tmh8.streamlit.app/)

<img src=".\picture\web.png"> 


### 版本需求
- Python 3.9


### 安装步骤
1. **下载zip文件或克隆此存储库**
   ```bash
   git clone https://github.com/fre-air/webStreamlit.git
   cd webStreamlit
   ```

2. **安装项目依赖项**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行streamlit示例**

   ```bash
   streamlit run webStreamlit.py
   ```

### 文件目录
```
webStreamlit
├── .idea   #创建项目时自动生成的配置目录，可忽略
├── __pycache__  #python编译文件和源文件，可忽略
├── icon   #项目图标及城市列表数据
├── models  #物体检测模型
├── picture  #茶园图片
├── pyecharts-assets-master  #pyecharts图表渲染时的静态资源文件
│  ├── /assets/
│  │  ├── jquery.min.js  #静态资源文件
│  │  └── ...
├── video  #茶园视频
├── .gitattributes   #使用git LFS上传大文件时，配置文件
├── README.md   #项目说明
├── requirements.txt  #项目环境依赖包
└── webStreamlit.py  #项目主文件
```

### 项目部署
本项目依托Streamlit Cloud 进行部署，总共有如下三个步骤：

1. 通过git将项目文件上传至Github
2. 添加requirements.txt文件
3. 通过Streamlit Cloud部署应用


### 参考资料
1. 气象数据来源于"和风天气开发服务网站"，选用逐小时天气预报，使用方式详见[官方文档](https://dev.qweather.com/docs/api/)
2. 数据可视化通过Pyecharts库绘制实现，使用方式详见[pyecharts教程](https://www.heywhale.com/mw/project/5eb7958f366f4d002d783d4a)与[pyecharts文档](https://05x-docs.pyecharts.org/#/zh-cn/charts_base)
3. 视频物体检测功能使用streamlit-webrtc组件实现，详情见[streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)
