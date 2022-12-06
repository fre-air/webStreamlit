import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from typing import List, NamedTuple
import numpy as np
import pandas as pd
import requests
import re
import av
import cv2
import random
import logging
import time
import queue
from PIL import Image
from pathlib import Path
import urllib.request
from pyecharts.charts import *
from pyecharts import options as opts

#Pyecharts图表生成需要一些静态资源文件，通过下面代码更改为kesci提供的资源，提高加载速度～
from pyecharts.globals import CurrentConfig
CurrentConfig.ONLINE_HOST = "https://cdn.kesci.com/lib/pyecharts_assets/"

image1 = Image.open('./icon/tea-garden.png')#读取图标为np.array类型
#1、配置页面的全局信息
st.set_page_config(
    page_title="茶园管理系统",#页面标题
    page_icon=image1,   #页面图标
    layout="wide",      #页面布局
    initial_sidebar_state="auto", #页面侧边栏
)

#从Github下载object-detection-app训练好的物体检测模型model
HERE = Path(__file__).parent
ROOT = HERE.parent
logger = logging.getLogger(__name__)
MODEL_URL = "https://github.com/robmarkcole/object-detection-app/blob/master/model/MobileNetSSD_deploy.caffemodel"
MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL ="https://github.com/robmarkcole/object-detection-app/blob/master/model/MobileNetSSD_deploy.prototxt.txt"
PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"
#分类
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

def download_file(url, download_to: Path, expected_size=None):
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return
    download_to.parent.mkdir(parents=True, exist_ok=True)
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

#用于存储单一实例对象的函数修饰器；物体检测时，标记颜色
@st.experimental_singleton
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))

COLORS = generate_label_colors()
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

class Detection(NamedTuple):
    name: str
    prob: float

# Session-specific caching
cache_key = "object_detection_dnn"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
    st.session_state[cache_key] = net

def _annotate_image(image, detections):
    (h, w) = image.shape[:2]
    result: List[Detection] = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            name = CLASSES[idx]
            result.append(Detection(name=name, prob=float(confidence)))
            # display the prediction
            label = f"{name}: {round(confidence * 100, 2)}%"
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image,
                label,
                (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLORS[idx],
                2,
            )
    return image, result
result_queue: queue.Queue = (
    queue.Queue()
)  # TODO: A general-purpose shared state object may be more useful.

def callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()
    annotated_image, result = _annotate_image(image, detections)
    # NOTE: This `recv` method is called in another thread,
    # so it must be thread-safe.
    result_queue.put(result)  # TODO:
    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

#读取video视频
@st.experimental_singleton
def get_video_byte():
    video_file=open('./video/福建北苑御茶园-简介.mp4','rb')
    video_bytes1 = video_file.read()
    video_file.close()
    video_file = open('./video/福建诏安公田茶园-航拍.mp4', 'rb')
    video_bytes2 = video_file.read()
    video_file.close()
    return video_bytes1, video_bytes2

#获取城市ID，作为获取城市天气的参数
@st.cache(ttl=3600)
def get_city_mapping():
    datas=pd.read_csv(r"./icon/China-City-List-latest.csv",sep=',',header=1,usecols=[0,2,7,9],encoding='utf-8')
    data=dict(zip(datas['Adm2_Name_ZH'],datas['Location_ID']))
    fuzhou=0
    for i in data.keys():
        if i!='福州市':
            fuzhou += 1
        else:
            break
    return data,fuzhou

#在缓存中保留条目的最大秒数,获取城市天气
@st.cache(ttl=3600)
def get_city_weather(ID):
    url=f"https://devapi.qweather.com/v7/weather/24h?location={ID}&key=05892c272b4f4a67b5ab30b7b95cdb9d"
    datas=requests.get(url).json()
    data_updateTime = datas['updateTime'] #api更新时间
    forecastHours=[]
    for i in range(len(datas['hourly'])):
        tmp={}
        tmp['fxTime']=datas['hourly'][i]['fxTime'] #预报时间
        tmp['temp'] = datas['hourly'][i]['temp'] #温度，默认单位：摄氏度
        tmp['humidity'] = datas['hourly'][i]['humidity'] #相对湿度，百分比数值
        tmp['windDir'] = datas['hourly'][i]['windDir'] #风向
        tmp['windSpeed'] = datas['hourly'][i]['windSpeed'] #风速，公里/小时
        tmp['precip'] = datas['hourly'][i]['precip'] #当前小时累计降水量，默认单位：毫米
        tmp['pressure'] = datas['hourly'][i]['pressure'] #大气压强，默认单位：百帕
        forecastHours.append(tmp)
    df_forecastHours=pd.DataFrame(forecastHours,columns=['fxTime','temp','humidity','windDir',
                                  'windSpeed','precip','pressure'])
    return data_updateTime,df_forecastHours

# 初始化变量
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True
else:
    st.session_state.first_visit = False
# 初始化全局配置
if st.session_state.first_visit:
    st.session_state.city_mapping, st.session_state.random_city_index = get_city_mapping()

#2、获取和风天气的气象数据，进行监测预警
st.sidebar.header('气象数据监测')
col7,col8=st.sidebar.columns(2)
city=col8.selectbox('选择',st.session_state.city_mapping.keys(),
                          index=st.session_state.random_city_index,label_visibility="collapsed")
col7.markdown(time.strftime('%Y-%m-%d %H:%M:%S'))
data_updateTime,df_forecastHours=get_city_weather(st.session_state.city_mapping[city])
index=['温度','相对湿度','风速','降水'] #指标
standard=['18℃～20℃','40％～60％','0m/s～4m/s','2mm～4mm'] #标准值
limit=[18,20,40,60,0,4,2,4]
present=[df_forecastHours.iloc[0][1],df_forecastHours.iloc[0][2],
         round((float(df_forecastHours.iloc[0][4])/3.6),1),df_forecastHours.iloc[0][5]]
monitor=[] #预警
for i in range(4):
    tem={}
    tem['指标']=index[i]
    tem['标准值']=standard[i]
    tem['当前值']=present[i]
    if float(present[i])<limit[2*i]:
        judge='偏低'
    elif float(present[i])>limit[2*i+1]:
        judge='偏高'
    else:
        judge='正常'
    tem['预警']=judge
    monitor.append(tem)
df_monitor=pd.DataFrame(monitor,columns=['指标','标准值','当前值','预警']) #创建DataFrame
st.sidebar.dataframe(monitor) #显示数据

#3、视频监控+物体检测功能
st.sidebar.header("视频监控点")
confidence_threshold = st.sidebar.slider(
    "置信度阈值", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)
with st.sidebar.container():
    webrtc_ctx=webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
if st.sidebar.checkbox("显示检测结果", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.sidebar.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                result = None
            labels_placeholder.table(result)

#4、智能化茶园管理系统
st.header('智能化茶园管理系统:tea:')
st.markdown('<br>', unsafe_allow_html=True)#html语句，换行

#5、茶园数据
col1,col2,col3,col4=st.columns(4)
col1.metric("茶园种植面积:mountain:","35(万亩)",delta=None)
col2.metric("茶产值:moneybag:","180(亿元)","0.02%")
col3.metric("茶产量:chart_with_upwards_trend:","8.2(万吨)","0.35%")
col4.metric("茶农👨‍🌾","36(万人)",delta=None)

#6、气象数据可视化
with st.container():
    # 获取x轴
    data=df_forecastHours['fxTime']
    x_data=[]
    for i in range(len(data)):
        x_data.append(''.join(re.findall(r'T(.*)\+',data[i])))
    bar = (
        Bar()
        .add_xaxis(x_data)
        .add_yaxis(
            "相对湿度",
            list(map(eval,df_forecastHours['humidity'].values.tolist())),
            yaxis_index=0,
            color="#d14a61",

        )
        .add_yaxis(
            "降水量",
            list(map(eval,df_forecastHours['precip'].values.tolist())),
            yaxis_index=1,
            color="#5793f3",

        )
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="降水量",
                type_="value",
                min_=0,
                max_=5,
                position="right",
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#d14a61")
                ),
                axislabel_opts=opts.LabelOpts(formatter="{value} mm"),
            )
        )
        .extend_axis(
            yaxis=opts.AxisOpts(
                type_="value",
                name="温度",
                min_=-5,
                max_=20,
                position="left",
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#675bba")
                ),
                axislabel_opts=opts.LabelOpts(formatter="{value} °C"),
                splitline_opts=opts.SplitLineOpts(
                    is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=1)
                ),
            )
        )
        .set_global_opts(
            yaxis_opts=opts.AxisOpts(
                name="相对湿度",
                min_=0,
                max_=100,
                position="right",
                offset=80,
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#5793f3")
                ),
                axislabel_opts=opts.LabelOpts(formatter="{value} ％"),
            ),
            title_opts=opts.TitleOpts(title=None),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(is_show=True,textstyle_opts=opts.TextStyleOpts(color='#fafafa'),item_gap=20,
                                        pos_left='10%',pos_right='90%'),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color='#fafafa')),
            datazoom_opts=opts.DataZoomOpts(
                type_="inside",
                range_start=0,
                range_end=100),
        )
    )
    line = (
        Line()
        .add_xaxis(x_data)
        .add_yaxis(
            "温度",
            list(map(eval,df_forecastHours['temp'].values.tolist())),
            yaxis_index=2,
            color="#675bba",
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True))
    )
    bar.overlap(line)
    grid = Grid()
    grid.add(bar, opts.GridOpts(pos_left="5%", pos_right="20%"), is_control_axis_index=True)
    components.html(grid.render_embed(), width=1400, height=520)

col5,col6=st.columns(2)
with col5.container():
    x = df_forecastHours['windDir'].values.tolist()
    y = list(map(eval, df_forecastHours['windSpeed'].values.tolist()))
    y_data = []
    for i in range(len(y)):
        y_data.append(round(float(y[i]) / 3.6, 1))#km/h与m/s单位换算
    #配色方案从Echarts投过来，径向渐变
    radial_item_color_js = """new echarts.graphic.RadialGradient(0, 0, 1, [{
                                            offset: 0,
                                            color: 'rgb(251, 118, 123)'
                                        }, {
                                            offset: 1,
                                            color: 'rgb(204, 46, 72)'
                                        }])"""
    c = (
        Polar(init_opts=opts.InitOpts(width='400px',height='400px'))
        .add_schema(angleaxis_opts=opts.AngleAxisOpts(data=x, type_="category",
                                                      axislabel_opts=opts.LabelOpts(color='#fafafa'),
                                                      axistick_opts=opts.AxisTickOpts(is_show=True)),
                    radiusaxis_opts=opts.RadiusAxisOpts(axislabel_opts=opts.LabelOpts(color='#fafafa'))
                    )
        .add("风速", y_data, type_="bar", stack="stack0")
        # .set_series_opts(label_opts=opts.LabelOpts(is_show=True,color='#ffffff',position='inside'))
        # .set_series_opts(itemstyle_opts=opts.ItemStyleOpts(color=JsCode(radial_item_color_js)))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="风玫瑰图",
                                      title_textstyle_opts=opts.TextStyleOpts(color='#fafafa')),
            legend_opts=opts.LegendOpts(is_show=True, pos_left='70%',
                                        pos_right='30%',
                                        textstyle_opts=opts.TextStyleOpts(color='#fafafa')),
        )
    )
    components.html(c.render_embed(),height=390,width=400)

with col6.container():
    x_tea=["漳平水仙","茉莉花茶","坦洋工夫","白芽奇兰","金骏眉","武夷岩茶"]
    y_num=random.sample([i for i in range(30000,150000)],6)
    data_pair=[]
    for k, v, c in zip(x_tea, y_num, ["#5470c6","#91cc75","#fac858","#ee6666","#73c0de","#fc8452"]):
        data_pair.append(
            opts.BarItem(
                name=k,
                value=v,
                itemstyle_opts=opts.ItemStyleOpts(color=c)
            ))
    c_tea = (
        Bar(init_opts=opts.InitOpts(width='600px',height='400px'))
        .add_xaxis(x_tea)
        .add_yaxis('',data_pair)
        .reversal_axis()
        .set_series_opts(label_opts=opts.LabelOpts(position="right"))
        .set_global_opts(title_opts=opts.TitleOpts(title="种植品种分析",pos_left='center',pos_top='top',
                                                   title_textstyle_opts=opts.TextStyleOpts(color='#fafafa')),
                         yaxis_opts=opts.AxisOpts(axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#ffffff'),is_show=False),
                                                  name="面积/亩",
                                                  axistick_opts=opts.AxisTickOpts(is_show=False)),
                         xaxis_opts=opts.AxisOpts(axisline_opts=opts.AxisLineOpts(is_show=False),
                                                  axistick_opts=opts.AxisTickOpts(is_show=False),
                                                  axislabel_opts=opts.LabelOpts(is_show=False)),
                         toolbox_opts=opts.ToolboxOpts(is_show=True,
                                                       orient='horizontal',
                                                       feature={"saveAsImage": {},"dataZoom":{"yAxisIndex": "none"},"restore":{},
                                                                "magicType":{"show": True, "type":["line","bar"]},"dataView": {}},
                                                       pos_left='center',
                                                       pos_top='5%'
                                                       ))
    )
    components.html(c_tea.render_embed(), height=350, width=800)
st.markdown('<br>', unsafe_allow_html=True)#html语句，换行
#7、茶园文字介绍
st.markdown("### 大田美人景区茶园")
st.markdown('<br>', unsafe_allow_html=True)#html语句，换行
st.markdown(
    """ 
    &emsp;&emsp;大仙峰茶美人景区位于福建省大田县屏山乡，占地面积约3000亩，是国内首家以“高山茶”为主题，融文化体验、环境教育、文创展示、休闲度假等功能为一体的原生态景区。
                          
    &emsp;&emsp;大田的茶贸历史悠久，可追溯到南宋隆兴二年(1164年)，大田种植高山茶始于大仙峰崇圣岩寺僧人种茶，《康熙字典》中就有对大田茶叶生产的注解。经过数代茶人不懈的传承耕耘，大田全县现有高山茶园近10万亩，产值7.19亿元，涉茶人员6.3万余人。
      
    &emsp;&emsp;大田不仅有“中国高山硒谷”之称，还是全国唯一的“中国高山茶之乡”，特别是，大田海拔千米以上的山峰有175座，是闽江、九龙江、晋江三大水系支流的发源地，被列
    为福建省级重点生态功能区。县域内茶园广布，由于大田高山云雾多，漫射光较多，茶树芽头肥壮、叶质柔软、茸毛甚多，氨基酸、咖啡碱、芳香油等物质积累较多，以此
    加工成的高山茶形美味浓、滋味甘醇，并且比较耐泡。
    
    &emsp;&emsp;以大田尤为知名的“江山美人茶”为例，其特别之处在于，茶青必须是小绿叶蝉叮咬吸食过，小绿叶蝉在茶青上分泌水解酶，合成萜烯醇，挥发出蜜糖香气，这是美人茶醇
    厚果香蜜味的来源，也是其有别于其它茶最显著的地方。   
    """
    )
#8、茶园无人机影像
st.image("./picture/福建大田县大仙峰茶美人景区茶园.jpg",caption="大仙峰茶美人景区茶园")
st.markdown('<br>', unsafe_allow_html=True)#html语句，换行
#9、茶园视频介绍
st.markdown("### 茶园视频介绍")
st.markdown('<br>', unsafe_allow_html=True)#html语句，换行
col11,col12=st.columns(2)
video11,video12=get_video_byte()
col11.video(video11, format='video/mp4',start_time=1)
col12.video(video12, format='video/mp4',start_time=9)
#源代码
with st.expander("View Code"):
    with open('webStreamlit.py', 'r', encoding='utf-8') as f:
        code = f.read()
    st.code(code, language="python")
