# Batch Backpropagation · 批量反向传播实验

原“4.5”子项目，对应三层前馈神经网络的整批 BP 训练实验。
支持单极性和双极性 sigmoid、中文 Qt/QML 界面、XOR 示例、误差曲线以及 CSV/JSON/NPZ 结果导出。

## 运行

```powershell
python -m pip install -r requirements.txt
python exercise45.py
```

Windows 也可双击 `启动习题4.5.cmd`。命令行实验与数值验证：

```powershell
python run_experiment.py --activation unipolar
python run_experiment.py --activation bipolar
python -m unittest test_batch_bp -v
```

`results/` 保留已有实验结果和界面预览，`docs/` 保存原实验资料。
完整算法约定及本机开发记录见 [技术路线与使用说明](技术路线与使用说明.md)，其中的旧目录和解释器路径为原环境记录。

基于 [Neural Network Sandbox](https://github.com/seanwu1105/neural-network-sandbox) 扩展。
原项目说明见 [UPSTREAM_README.md](UPSTREAM_README.md)，MIT 许可见 [LICENSE](LICENSE)。
