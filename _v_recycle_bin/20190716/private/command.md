#### 实用Linux命令:
1. nautilus /path/to/that/folder 从终端打开文件夹
2. sudo nvidia-docker run -it --name tf -p 8888:8888 -p 6006:6006 -p 9999:9999 -v /home/mao/data:/home/mao/data 413b9533f92a
3. sudo docker run -it --name tf -v /home/qq1044467857:/home/qq1044467857 2c9b7cf6a404
4. nohup jupyter lab --ip 0.0.0.0 --no-browser --port 8888 > jupyter_log 2> error_log &
5. gsutil -m rsync -d -r data gs://mybucket/data
6. sudo docker start tf
7. sudo docker exec -it tf /bin/bash
https://raw.githubusercontent.com/breakwa11/gfw_whitelist/master/whiteiplist.pac
```
git config --global user.email "mao@example.com"
git config --global user.name "mao"
git remote add origin https://github.com/user/repo.git    添加远程仓库url
git remote -v    显示远程仓库url
git push --set-upstream origin master
```
8. nohup python3 downloadOI.py --mode train --classes "Person" > download_log 2> error_log &