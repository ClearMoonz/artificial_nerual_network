import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12

ApplicationWindow {
    id: window
    visible: true
    width: 1100; height: 850
    minimumWidth: 900; minimumHeight: 650
    title: "习题4.5 · 三层前馈神经网络 · 批训练 BP"
    font.family: "Microsoft YaHei"
    color: "#f7f9fc"
    Rectangle { anchors.fill: parent; color: "#f7f9fc"; z: -1 }
    function loadXor() {
        nInput.value = 2; nOutput.value = 1
        dataset.text = activation.currentIndex === 0 ? "0,0,0\n0,1,1\n1,0,1\n1,1,0" : "-1,-1,-1\n-1,1,1\n1,-1,1\n1,1,-1"
    }
    ScrollView {
        id: viewport
        anchors.fill: parent
        contentWidth: availableWidth
        ColumnLayout {
            objectName: "bodyLayout"
            width: viewport.availableWidth
            height: Math.max(viewport.availableHeight, implicitHeight)
            spacing: 12
            Label { text: "三层网络：输入层 → 隐含层 → 输出层"; font.pixelSize: 23; Layout.leftMargin: 16; Layout.topMargin: 12 }
            Label { text: "权值与偏置初始化为 [-1,1] 随机数；固定学习率；整批更新；不使用动量。"; Layout.leftMargin: 16 }
            GridLayout {
                columns: 6; Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16; enabled: !batchBP.busy
                Label { text: "输入节点" }
                SpinBox { id: nInput; from: 1; to: 1000; value: 2; editable: true; Layout.fillWidth: true }
                Label { text: "隐含节点" }
                SpinBox { id: nHidden; from: 1; to: 1000; value: 4; editable: true; Layout.fillWidth: true }
                Label { text: "输出节点" }
                SpinBox { id: nOutput; from: 1; to: 1000; value: 1; editable: true; Layout.fillWidth: true }
                Label { text: "学习率 η" }
                TextField { id: lr; text: "0.8"; Layout.preferredWidth: 145; Layout.fillWidth: true }
                Label { text: "激活函数" }
                ComboBox { id: activation; model: ["单极性 sigmoid", "双极性 sigmoid"]; Layout.preferredWidth: 180; Layout.fillWidth: true }
                Label { text: "随机种子" }
                TextField { id: seed; text: "42"; Layout.preferredWidth: 145; Layout.fillWidth: true }
                Label { text: "RMSE 阈值" }
                TextField { id: threshold; text: "0.05"; Layout.preferredWidth: 145; Layout.fillWidth: true }
                Label { text: "最大训练轮数" }
                TextField { id: epochs; text: "20000"; Layout.preferredWidth: 180; Layout.fillWidth: true }
                Button { text: "载入 XOR 示例"; Layout.columnSpan: 2; Layout.fillWidth: true; onClicked: window.loadXor() }
            }
            RowLayout {
                Layout.fillWidth: true; Layout.fillHeight: true
                Layout.minimumHeight: 320; Layout.preferredHeight: 320
                Layout.leftMargin: 16; Layout.rightMargin: 16; spacing: 16
                ColumnLayout {
                    objectName: "dataPanel"
                    Layout.fillHeight: true; Layout.fillWidth: false
                    Layout.preferredWidth: Math.max(300, viewport.availableWidth * 0.32)
                    Layout.minimumWidth: 300; Layout.maximumWidth: Math.max(300, Math.min(560, viewport.availableWidth * 0.32))
                    Label { text: "训练数据：每行输入在前、目标在后" }
                    Label { text: "逗号或空格分隔，无表头；可粘贴自定义数据。"; font.pixelSize: 12 }
                    ScrollView {
                        Layout.fillWidth: true; Layout.fillHeight: true; Layout.minimumHeight: 160
                        TextArea { id: dataset; text: "0,0,0\n0,1,1\n1,0,1\n1,1,0"; enabled: !batchBP.busy; selectByMouse: true; font.family: "Consolas" }
                    }
                    Label { text: "目标范围：单极性 [0,1]；双极性 [-1,1]。"; font.pixelSize: 12 }
                    RowLayout {
                        Button {
                            objectName: "startButton"
                            text: "开始训练"; enabled: !batchBP.busy
                            onClicked: {
                                plot.points = []; plot.requestPaint()
                                batchBP.start(nInput.value, nHidden.value, nOutput.value, lr.text,
                                              activation.currentIndex === 0 ? "unipolar" : "bipolar",
                                              threshold.text, epochs.text, seed.text, dataset.text)
                            }
                        }
                        Button { text: "停止"; enabled: batchBP.busy; onClicked: batchBP.stop() }
                    }
                }
                Rectangle {
                    objectName: "chartPanel"
                    Layout.fillWidth: true; Layout.fillHeight: true; Layout.minimumWidth: 350
                    color: "white"; border.color: "#dce3eb"; radius: 6
                    Canvas {
                        id: plot; anchors.fill: parent; anchors.margins: 8
                        property var points: []
                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.clearRect(0, 0, width, height)
                            var left = 55, top = 36, pw = width-75, ph = height-80
                            var maxEpoch = points.length ? Math.max(100, points[points.length-1][0]) : 100
                            var maxLoss = 0.1
                            for (var i=0; i<points.length; i++) maxLoss = Math.max(maxLoss, points[i][1]*1.1)
                            ctx.font = "14px sans-serif"; ctx.fillStyle = "#344054"
                            ctx.fillText("训练误差（RMSE）", left, 20)
                            ctx.strokeStyle = "#e5e9ef"; ctx.lineWidth = 1
                            for (var j=0; j<=4; j++) {
                                var yy = top+ph*j/4
                                ctx.beginPath(); ctx.moveTo(left,yy); ctx.lineTo(left+pw,yy); ctx.stroke()
                                ctx.fillText((maxLoss*(1-j/4)).toFixed(2), 3, yy+4)
                            }
                            ctx.fillText("0", left, top+ph+22)
                            ctx.fillText(String(maxEpoch), left+pw-40, top+ph+22)
                            ctx.fillText("训练轮数", left+pw/2-28, height-4)
                            ctx.strokeStyle = "#2375bf"; ctx.lineWidth = 2; ctx.beginPath()
                            for (var k=0; k<points.length; k++) {
                                var xx = left+pw*points[k][0]/maxEpoch
                                var y = top+ph*(1-points[k][1]/maxLoss)
                                if (k===0) ctx.moveTo(xx,y); else ctx.lineTo(xx,y)
                            }
                            ctx.stroke()
                        }
                        onWidthChanged: requestPaint()
                        onHeightChanged: requestPaint()
                    }
                }
            }
            Label { text: "训练结果"; font.pixelSize: 18; Layout.leftMargin: 16 }
            ScrollView {
                objectName: "resultView"
                Layout.fillWidth: true; Layout.fillHeight: true
                Layout.minimumHeight: 150; Layout.preferredHeight: 150
                Layout.leftMargin: 16; Layout.rightMargin: 16
                TextArea { text: batchBP.message; readOnly: true; selectByMouse: true; wrapMode: TextArea.Wrap }
            }
            Label { text: "停止约定：RMSE ≤ 阈值，或达到最大轮数。图中 E_RME 定义未提供，暂按 RMSE 实现。"; Layout.fillWidth: true; wrapMode: Text.Wrap; Layout.leftMargin: 16; Layout.rightMargin: 16; Layout.bottomMargin: 16; font.pixelSize: 12 }
        }
    }
    Connections {
        target: batchBP
        function onPoint(epoch, rmse) {
            plot.points.push([epoch, rmse])
            plot.requestPaint()
        }
    }
}
