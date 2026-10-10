~~~bash
ssh -vvv -i ~/.ssh/PrivateKey_AC7FWPK root@198.18.3.179

# scp -i ~/.ssh/PrivateKey_AC7FWPK v801.HDY   root@198.18.3.179:/storage/self/primary/FishView


ssh -vvv -i "D:\projects\LeiSong\PrivateKey_AC7FWPK.fixed"  root@198.18.3.179
scp -O -i "D:\projects\LeiSong\PrivateKey_AC7FWPK.fixed" "E:\V80LiveSonar-20261009.apk" root@198.18.3.179:/data/local/tmp/
~~~
~~~bash
mkdir -p /data/local/tmp
mv "/data/海底鹰动态声纳-v3.6.3.260717.apk" /data/local/tmp/
#chmod 644 "/data/local/tmp/海底鹰动态声纳-v3.6.3.260717.apk"
pm install -r "/data/local/tmp/海底鹰动态声纳-v3.6.3.260717.apk"

getprop ro.board.platform
#rk3399
~~~