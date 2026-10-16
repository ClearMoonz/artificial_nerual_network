# artificial_nerual_network

人工神经网络课程实验：批量反向传播训练与优化算法可视化。

| 子项目 | 内容 | 运行入口 |
| --- | --- | --- |
| [batch_backpropagation](batch_backpropagation/) | 三层前馈网络、批量 BP、单极性/双极性 sigmoid、XOR 验证 | `python exercise45.py` |
| [optimization_visualizer](optimization_visualizer/) | 最速下降、牛顿法、Fletcher–Reeves 共轭梯度，二维/三维轨迹 | `python gradient_descent_visualizer/main.py` |

## 安装与运行

建议为两个子项目分别创建 Python 虚拟环境，以分别管理 PyQt5 和 PyQt6。
进入对应子项目后，按其 README 安装依赖并启动。

仓库保留源码、测试、技术说明、样例数据及已有实验结果；不包含本机安装的依赖、缓存和 IDE 配置。
原有本地目录保留，当前目录为整理后的上传副本。

## 来源与许可

`batch_backpropagation` 基于 [seanwu1105/neural-network-sandbox](https://github.com/seanwu1105/neural-network-sandbox) 扩展，保留原 MIT [LICENSE](batch_backpropagation/LICENSE) 和 [上游说明](batch_backpropagation/UPSTREAM_README.md)。
仓库中其他自编内容暂未单独指定开源许可证。
