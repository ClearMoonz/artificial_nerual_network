# Optimization Visualizer · 优化算法可视化

原“梯度下降”子项目，包含最速下降、牛顿法与 Fletcher–Reeves 共轭梯度，以及同步二维/三维轨迹演示。

## 标准环境安装与运行

```powershell
python -m pip install -r gradient_descent_visualizer/requirements.txt
python gradient_descent_visualizer/main.py
python gradient_descent_visualizer/main.py --3d
```

安装依赖后，Windows 可双击 `启动程序.cmd` 或 `启动三维演示.cmd`。
运行时代码支持标准安装的 PyQt6，也兼容原来的可选本地依赖目录；仓库不携带这些依赖包。
`software/` 中的旧版预览及原软件启动脚本是原环境辅助资料，运行主面板不需要它们。

下文保留原项目的详细操作说明与开发环境记录，其中本机解释器和本地依赖路径属于原环境记录；新环境请以上述安装步骤为准。

---

# 第9章9.2节：三种优化算法可视化实验

已完成统一中文实验面板。双击本目录 **启动程序.cmd**，或在PyCharm选择“梯度下降可视化”运行配置。software/launch.cmd也已改为打开完整面板，旧窗口需关闭后重新启动。

## 运行环境

- 本机解释器：D:/program/anaconda/python.exe，Python 3.13.9。
- 项目内Qt依赖：software/anaconda_runtime，PyQt6 6.9.1＋Qt6 6.9.2。NumPy和Matplotlib使用Anaconda现有环境。启动代码显式配置Qt插件路径，支持中文目录。
- software/runtime为先前Python 3.12预览的依赖。迁移到其他电脑时，安装gradient_descent_visualizer/requirements.txt，并调整启动脚本及runtime.py的依赖路径。

## 操作方式

1. 选择实验A、B或C。默认实验B，三个算法使用相同函数、起点、阈值和最大步数。
2. 输入起点，或点击左上等高线图选点，再点击“运行 / 继续”。选点会重置原结果。
3. “单步”执行一次更新；“暂停”保留状态，“运行 / 继续”继续。“重置 · 默认起点”恢复当前教材实验的默认起点：A为(0.5,0.5)，B为(0.8,-0.25)，C为(1.5,0)。算法、学习率、阈值等其他参数保留。
4. 动画默认每步500ms。修改动画速度不改变计算结果；靠近终点时移动可能小于一个像素，请查看步数与梯度范数。
5. “计算至结束”分批快速计算，可暂停。修改计算参数会停止并重置旧结果。
6. “导出结果”选择目录，程序新建带时间的子目录，输出iterations.csv、experiment.json、plots.png、interface.png。暂停中的部分结果也可导出，未收敛状态会保留。

图表工具栏支持缩放与移动；处于工具栏移动/缩放模式时，点击不会修改起点。图域对所有算法一致且固定，发散轨迹可能越出图域；计算区、曲线及CSV继续保留这些点。

## 三维模式与录屏

在图表上方的“视图”选择 **三维曲面 · 同步轨迹**。左上仍为二维等高线，右上变为三维目标函数曲面；三维高度为F(x1,x2)，不是增加一个优化参数。切换视图保留迭代结果和播放状态。

也可以双击项目根目录“启动三维演示.cmd”直接进入三维模式。

- 左键拖动三维图旋转，右键拖动缩放，中键拖动平移；点击二维图或输入参数框选择起点。
- 三维轨迹和当前点从同一份迭代记录更新，使用原始函数值作为高度；运行、暂停、单步均与二维图同步。
- “重置视角”恢复默认角度，播放不会重置已旋转的视角。
- “放大三维”打开独立大窗口，适合录屏。该窗口有单步、运行、暂停、重置实验、重置视角和导出按钮，直接控制同一个实验。“重置 · 默认起点”停止播放、把所有算法恢复到当前教材实验的默认起点、清空轨迹并归零步数，其他参数保留；“重置视角”只改变观察角度。
- 实验C默认勾选“聚焦谷底”，隐藏F>12的曲面高处，防止谷底与鞍点被压缩；图中明确标注裁切范围。取消勾选可查看完整曲面。此选项不裁切实验数据或改变计算。
- 超出三维显示范围的轨迹段隐藏，并注明当前点超出图域；结果表和导出的CSV/JSON保留真实坐标与函数值。
- 三维模式导出额外生成surface3d.png和view.json，分别保存独立三维图及视角、图域、谷底聚焦设置。

推荐录屏顺序：实验B→三维模式→放大三维→重置实验→单步两次→运行至结束，再切到实验C的鞍点预设。可在暂停时旋转视角，解释三维曲面和二维等高线的对应关系。

## 教材实验

| 实验 | 函数与设置 | 观察重点 |
|---|---|---|
| A | F=x1²+25x2²，起点(0.5,0.5) | 学习率0.01、0.035、0.039、0.04、0.041；稳定条件0<alpha<0.04 |
| B | F=0.5 x^T[[2,1],[1,2]]x，起点(0.8,-0.25) | 最速下降精确线搜索、牛顿法、FR共轭梯度同条件对比 |
| C | F=(x2-x1)^4+8x1x2-x1+x2+3 | 四个教材起点、局部二次近似、极小点与鞍点 |

实验B默认梯度阈值1e-6：最速下降19步、牛顿1步、共轭梯度2步。实验C四个教材起点分别到达右下极小点、左上极小点、鞍点、左上极小点。

实验C不启用三算法对比或共轭梯度，本版FR精确线搜索仅覆盖正定二次函数。该实验可选最速下降固定步长或回溯线搜索；回溯明确标为扩展，不将二次函数步长公式应用于非二次函数。

## 算法约定

- 最速下降：p=-g；固定步长或二次函数精确步长alpha=-(g^T p)/(p^T H p)。
- 牛顿法：解H p=-g，alpha=1。允许函数值增加、趋向鞍点，不隐式加入阻尼或伪逆。
- 共轭梯度：Fletcher–Reeves，beta=(g^T g)/(g_previous^T g_previous)，p=-g+beta*p_previous；使用二次函数精确线搜索。
- 内部使用双精度，显示舍入不参与下一步。梯度范数达到阈值后，用Hessian特征值判断驻点类型；梯度小不证明全局最优。
- 最大步数、越界、奇异Hessian、非有限数、线搜索退化、停滞有独立状态。初始点零梯度直接结束。
- 对数梯度曲线仅在显示时将零值绘为1e-16；表格和导出保留真实零值。函数值曲线保持线性坐标，可显示负数。
- 调用次数包括算法和终止判断，不包含可视化二次模型的额外计算；计算用时排除绘图、等待及导出。这是本机单次测量，不作为严谨性能基准。

计算区展示最后一次更新的旧点、梯度、方向、alpha、beta、新点及Hessian特征值，JSON包含全部更新。精确线搜索的最速下降展示相邻方向正交，共轭梯度展示p_previous^T H p。

## 验证与结果

在本目录使用Anaconda执行：

```powershell
& 'D:/program/anaconda/python.exe' 'gradient_descent_visualizer/tests/test_optimizers.py'
& 'D:/program/anaconda/python.exe' 'gradient_descent_visualizer/tests/test_gui.py'
& 'D:/program/anaconda/python.exe' 'gradient_descent_visualizer/tests/test_surface_gui.py'
& 'D:/program/anaconda/python.exe' 'gradient_descent_visualizer/run_experiments.py'
```

10项数值测试覆盖导数差分核对、教材关键更新、正交与共轭关系、四个非凸起点、学习率稳定性、异常及导出。界面测试实际触发按钮、Qt定时器和图中点击，验证播放/暂停/继续、单步、重置、参数变更、导出、鞍点与发散，并与原软件逐步结果对比。

- results/教材验证/验证汇总.md：11组教材实验，每组附CSV与JSON。
- results/完整界面预览：默认三算法对比的完整结果与截图。
- results/界面验收：三算法、牛顿鞍点、学习率发散的界面与导出文件。
- results/三维验收：三维轨迹、放大窗口及鞍点示例；三维界面测试验证真实高度、实际鼠标旋转、相机角度保留、视图切换不修改记录、同步播放/暂停和导出。

## 来源与结构

依据本目录参考书.pdf第9章9.2节，教材131—141页。书内MATLAB提示作为参考，不当作用户要求。

配套软件Neural Network Design Demonstrations，nndesigndemos 1.3.0，来源：https://pypi.org/project/nndesigndemos/ 。原软件和许可证保留在software/deps，software/launch_original.cmd打开原演示。

正式版本沿用PyQt6＋Matplotlib技术，在gradient_descent_visualizer建立独立中文面板和与界面分离的教材算法，并与原软件交叉验证，不将原软件演示称为自行原创。

核心文件：objectives.py（函数）、optimizers.py（算法、记录、导出）、main_window.py（界面和播放）、surface_view.py（三维曲面）、runtime.py（环境）、main.py（入口）、run_experiments.py（教材实验）。当前不包含自定义函数、神经网络训练及其他优化器。
