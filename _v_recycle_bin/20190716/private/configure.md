#### python3虚拟环境virtualenv:
1. 安装   pip3 install virtualenv
2. 创建虚拟环境   virtualenv --no-site-packages --python=python3.6 venv(虚拟环境名字)
3. 激活虚拟环境   source venv/bin/activate
4. 安装package    pip install -r requirements.txt
5. 退出当前虚拟环境    deactivate
6. 参考文章    [五分钟了解Python Virtualenv(csdn)](https://blog.csdn.net/ysbj123/article/details/79727396)
```
#requirements.txt里的内容
numpy
gradio
tqdm
pandas
matplotlib
scipy
seaborn
sklearn
pillow
pydot
opencv-python
jupyter
notebook
imutils
#tensorflow-gpu==2.0.0b1
#tensorflow-gpu
jupyter-tensorboard
```

#### python3虚拟环境升级版Virtualenvwrapper:(推荐)
1. 安装   pip3 install --user virtualenvwrapper
2. 执行        echo "source virtualenvwrapper.sh" >> ~/.bashrc
3. 添加到.bashrc里      VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
4. 执行        source ~/.bashrc
5. 创建虚拟环境     mkvirtualenv --no-site-packages --python=python3.6 venv(虚拟环境名字)
6. 列出虚拟环境列表    workon
7. 切换环境    workon [venv]
8. 删除环境    rmvirtualenv venv
9. 参考文章    [最全的Python虚拟环境使用方法(知乎)](https://zhuanlan.zhihu.com/p/60647332)

#### edgeTPU在虚拟环境里的配置
sym-link in the edgetpu library to your Python virtual environment:
1. cd /home/mao/.virtualenvs/tf1/lib/python3.6/site-packages
2. ln -s /usr/local/lib/python3.6/dist-packages/edgetpu edgetpu
3. 参考文章    [Getting started with Google Coral’s TPU USB Accelerator(pyimagesearch)](https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator/)
