该目录下有这些文件：
chat.py - 对话引擎，APPID是硬编码进去的
config.toml - 对话引擎的配置文件，和dify有关的已经删了，主要是配置硅基流动的api-key
pirate.py - 算法，全自动执行。

怎么保持会话？
使用 `tmux` 保持连接，这样尽管不在shell操作也可以连接上。注意不要断网，断网自动关连接的，只能手动重连。

一次攻击流程如下：
- 执行 `python3 ./pirate.py`，所有环境已经配置好了。
- 观察执行，**如果出现拒绝连接或者连接已重置，需要执行 `~/B-RAG/deploy_scripts/run_secure.sh restart` 重启服务，这一般是因为显存爆了**，也可以考虑当出现这个问题时开一个进程去自动重启。
- 也可以观察执行的效果如何，最下面的 main 函数里面可改 `max_iteration=` 后面填迭代步数。
   `attacker.execute_attack(max_iterations=20) # 这里填迭代多少步，一般t步需要1.5t~2t分钟跑完。`
- 初始化有一个 `keyword` 填写领域相关的词汇。
- 结果保存在当前目录下的 `chunk_i.txt`，i 是 chunk 的编号。`stolen_knowledge.txt` 不用管，就是把这些 chunk 的内容拼到一起了。
- 使用 scp 命令下载到本地：`scp funhpc@111.6.167.21:~/RAG/pirate/chunk* ./`

然后把数据打包发给我，我整合去重即可。


