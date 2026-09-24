1. 清理日誌
   ~~~bash
   sudo du -xhd1 /path/to/check 2>/dev/null | sort -h

   sudo head -n 5 /var/log/syslog
   sudo tail -n 5 /var/log/syslog
   sudo truncate -s 0 /var/log/syslog
   ~~~
2. ip
   ~~~bash
   #temp
   sudo ip addr add 192.168.19.101/24 dev eth0
   sudo ip link set eth0 up
192.168.19.113
   #always
   nmcli connection show
   sudo nmcli connection modify "有线连接 1"     ipv4.method manual     ipv4.addresses 192.168.53.101/24
   sudo nmcli connection down "有线连接 1"
   sudo nmcli connection up "有线连接 1"
   ~~~