# [重置Windows的WSL子系统密码](https://github.com/ruheyun/yc-blog/issues/11)

在 Windows PowerShell（不是 Ubuntu 终端）运行：

```
wsl -u root
```

进入root后，再输入下面命令（这里ruhe是wsl的用户名）：

```
passwd ruhe
```

然后可以输入新密码，之后输入退出命令：

```
exit
```