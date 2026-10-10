~~~bash
#查看
Get-NetIPAddress -InterfaceAlias "以太网 2" -AddressFamily IPv4

#添加
New-NetIPAddress `
  -InterfaceAlias "以太网 2" `
  -IPAddress 192.168.53.101 `
  -PrefixLength 24 `
  -SkipAsSource $true

#删除
Remove-NetIPAddress `
  -InterfaceAlias "以太网 2" `
  -IPAddress 192.168.53.101 `
  -Confirm:$false
~~~