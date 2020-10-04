#### sourcegraph使用手册
1. 搜索仓库   repo:^github\.com/docker/go$
2. 搜索仓库里的函数     repo:^github\.com/tensorflow/models$ create_training_graph
3. [逛Github不可缺少的插件(知乎)](https://zhuanlan.zhihu.com/p/44153011)

#### DVC使用手册
data version control:
```
1.      git init
2.      dvc init
3.      git commit -m "initialize DVC"
以上三步为git和dvc的初始化
```

```
4.      dvc add data/data.xml
5.      git add data/.gitignore data/data.xml.dvc
6.      git commit -m "add source data to DVC"
以上三步为跟踪数据文件以及提交数据文件
```

#### git上传大文件报错:
1. git一些报错解决方法    [每一项都亲测，保证不踩坑(知乎)](https://zhuanlan.zhihu.com/p/53961303)